"""Geocode a NISAR RSLC into a NISAR-format GSLC: ISCE3's GSLC workflow, ported.

`geocode_rslc` is `nisar.workflows.gslc.run` followed by
`GslcWriter.populate_metadata`, reimplemented on this package's numpy
kernels (`nisar_tools._geocode_kernels`, `nisar_tools.radar`). The
output reads with `nisar_tools.GSLC` like any granule from ASF.

What is reproduced, checked against an ASF GSLC and the RSLC it was made from:

- **Pixels** -- 99.8% bit-identical (frequency A; 99.87% B), the rest one
  truncation step apart; NaN pattern and subswath `mask` identical. DEM
  heights with ISCE3's float32 spline, zero-Doppler `geo2rdr` from mid-swath,
  8-tap sinc with Doppler baseband/re-ramp in ISCE3's float32 operation order,
  flattening by the *uncorrected* range, 10-bit mantissa truncation.
- **Geometry** -- `geo2rdr` slant range to 6.4 nm; every `radarGrid` cube
  layer equal to float32 rounding.
- **Timing corrections** -- ionosphere from the NISAR TEC file identical, solid
  Earth tides to 0.2 mm, against the GSLC's stored LUTs.
- **Format** -- every dataset, dtype, filter and attribute of the L2 GSLC spec
  1.5.0 as ISCE3 writes it (`tests/gslc_format_diff.py` checks it).

Departures, each deliberate:

- The mean in each image's statistics is float64 (ISCE3's float32 running mean
  over ~10^9 samples is mostly rounding error).
- `geo2rdr` seeds every pixel at mid-swath, as `geocodeSlc` does, but the
  cube seeds per node rather than from the previous node (<1e-8 s apart).
- Without a frame database the output box is the swath footprint snapped
  outward to `snap` metres. ASF's frame boxes are larger; `like=` a GUNW or
  GSLC of the same frame reuses its box exactly (`product_bounds`).
- `softwareVersion`, `processingType` ("Custom"), `processingCenter` and
  `runConfigurationContents` describe this run, not an SDS one.
- The RSLC's last line and last range sample are read. ISCE3's radar blocks
  end one short (an exclusive end on an inclusive box), so along the far-range
  edge a ribbon about one pixel wide is NaN in ASF's products and finite here.
  Its `mask` is 0 in both.
"""

import datetime as _dt
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np

from . import _gslc_metadata, gslc_product as gp, radar
from ._geocode_kernels import interpolate_slc, radar_indices, truncate_mantissa
from .corrections import compute_timing_corrections
from .dem import DEMInterpolator
from .qa import BandOffset
from .rslc import RSLC, geogrid_from_bounds, metadata_geogrid

SOFTWARE_VERSION = "nisar_tools 0.1.0 (port of the ISCE3 0.25.16 GSLC workflow)"
CUBE_HEIGHTS = tuple(float(h) for h in np.arange(-1000, 9001, 500))
_INTERP_MARGIN = {"nearest": 0, "bilinear": 0, "biquintic": 5}


def default_gslc_name(rslc_granule_id):
    """The GSLC granule name for an RSLC, marked as Science On-Demand (`OD`).

    NISAR's naming convention: `L1`->`L2`, `RSLC`->`GSLC`, processing
    type `OD` rather than `PR`/`UR` so the file cannot be mistaken for an
    SDS product; every other field (track, frame, mode, times, CRID, counter)
    stays that of the source RSLC.
    """
    parts = rslc_granule_id.split("_")
    if len(parts) < 4 or parts[0] != "NISAR" or parts[3] != "RSLC":
        raise ValueError(f"not a NISAR RSLC granule id: {rslc_granule_id!r}")
    parts[1], parts[2], parts[3] = "L2", "OD", "GSLC"
    return "_".join(parts)


# ---------------------------------------------------------------------------
# Orbit
# ---------------------------------------------------------------------------
def load_orbit(rslc, orbit_file=None):
    """`nisar.products.readers.orbit.load_orbit`: embedded, or external cropped.

    An external XML is referenced to the radar grid's epoch and cropped to the
    RSLC orbit's span plus `max(120 s / spacing, 7)` state vectors.
    """
    internal = rslc.orbit
    if orbit_file is None:
        return internal
    external = radar.Orbit.from_xml(orbit_file, reference_epoch=rslc.ref_epoch)
    npad = max(int(120.0 / external.spacing), 7)
    return radar.crop_external_orbit(external, internal, npad=npad)


# ---------------------------------------------------------------------------
# Metadata LUT geocoding (GeocodeCov::geocodeInterp)
# ---------------------------------------------------------------------------
def geocode_lut(data, lut_grid, geogrid, dem, orbit, method="biquintic",
                threshold=1e-8, maxiter=25):
    """Geocode a radar-coordinate LUT onto a map grid, `geocodeInterp`-style.

    `lut_grid` is the `nisar_tools.radar.RadarGrid` the LUT is sampled
    on. Each node: DEM height, zero-Doppler `geo2rdr`, reject outside the grid
    (or within the interpolator's margin of its edge), interpolate. Returns an
    array of `data`'s dtype on the grid, NaN where not geocoded.
    """
    from pyproj import Transformer

    data = np.asarray(data)
    X, Y = np.meshgrid(geogrid.x_coordinates, geogrid.y_coordinates)
    if geogrid.epsg == 4326:
        lon, lat = X, Y
    else:
        lon, lat = Transformer.from_crs(f"EPSG:{geogrid.epsg}", "EPSG:4326",
                                        always_xy=True).transform(X, Y)
    lon, lat = np.deg2rad(np.ravel(lon)), np.deg2rad(np.ravel(lat))
    h = dem.heights(lon, lat) if dem is not None else np.zeros(lon.size)
    t, r, ok = radar.geo2rdr(np.stack([lon, lat, h], axis=1), orbit, None,
                             lut_grid.wavelength, lut_grid.look_side,
                             t0=lut_grid.sensing_mid, threshold=threshold, maxiter=maxiter)
    ok &= lut_grid.contains(np.where(ok, t, np.nan), np.where(ok, r, np.nan))
    row = (t - lut_grid.sensing_start) / lut_grid.azimuth_time_interval
    col = (r - lut_grid.starting_range) / lut_grid.range_pixel_spacing
    ok &= (row >= 0) & (col >= 0) & (row <= lut_grid.length - 1) & (col <= lut_grid.width - 1)

    fill = np.nan + 1j * np.nan if np.iscomplexobj(data) else np.nan
    out = np.full(lon.size, fill, dtype=data.dtype)
    if ok.any():
        margin = _INTERP_MARGIN[method]
        az0 = max(int(np.floor(row[ok].min())) - margin, 0)
        rg0 = max(int(np.floor(col[ok].min())) - margin, 0)
        az1 = min(int(np.ceil(row[ok].max())) + margin, lut_grid.length - 1)
        rg1 = min(int(np.ceil(col[ok].max())) + margin, lut_grid.width - 1)
        block = data[az0:az1 + 1, rg0:rg1 + 1]
        ry, rx = row - az0, col - rg0
        keep = ok & (rx >= margin) & (ry >= margin) \
            & (rx <= block.shape[1] - margin - 1) & (ry <= block.shape[0] - margin - 1)
        if keep.any():
            fn = radar._INTERPOLATORS[method]
            out[keep] = fn(block, rx[keep], ry[keep])
    return out.reshape(geogrid.length, geogrid.width)


def _lut_method(lines, samples, requested=None):
    """`geocode_metadata_group`'s default interpolator choice."""
    if requested is not None:
        return requested
    if lines == 1 or samples == 1:
        return "nearest"
    if lines < 5 or samples < 5:
        return "bilinear"
    return "biquintic"


# ---------------------------------------------------------------------------
# Secondary registration
# ---------------------------------------------------------------------------
class RegistrationField:
    """Extra radar-position shifts for `geocode_rslc`'s `registration`.

    A regular map grid (`x`, `y` in the output projection, `y` may
    descend) carrying, per node, the fractional `lines` and `samples` to
    add to an output pixel's radar position before it is interpolated.
    Evaluated bilinearly; beyond the grid, the nearest edge node. The field
    must be finite everywhere -- fill gaps before building one.

    **What it is for.** Two GSLCs geocoded independently are registered only
    as well as their geolocation agrees, and ASF's per-date ionospheric
    geolocation correction (from the global TEC map) can disagree with the
    radar by metres -- about a resolution cell, which halves the pair's
    coherence. The GUNW resamples its secondary to offsets ("rubbersheeting":
    a dense field, or, in the solid-earth PR products, one quadratic per frame
    fitted to the dense offsets); this is the same thing for a GSLC: geocode
    the secondary at its geometric position plus the measured offsets, while
    the phase is still flattened by the geometric range.

    **The carrier.** NISAR keeps the ~950 Hz Doppler carrier in its SLCs, so
    sampling a pixel `dt` further along track changes its phase by
    `2 pi f_dc dt`, ~1.7 cm of LOS per metre of shift at L-band. `carrier`
    says where that phase is taken:

    - ``"geometric"`` (default): at the position *before* this field's shift.
      The pair keeps the coherence the registration buys, but its phase is the
      unregistered pair's: ground motion projects on the look at the native
      Doppler centroid -- the `radarGrid` cube's `losUnitVector` -- wherever
      the field follows it or not, and an ionospheric along-track shift that
      the field follows leaves no carrier term. A relative timing error
      between the two RSLCs that the offsets hold stays in the phase, as in an
      unregistered pair: correct it with `geocode_rslc(azimuth_shift=)`.
    - ``"shifted"``: at the sampled position, as ISCE3's resampler does (the
      GUNW). Right for a shift that moves the image but not the ground, such
      as the reference date's own TEC and tide shifts carried by the
      secondary. Where such a field follows ground motion, the phase projects
      that motion on the zero-Doppler look instead, and where it follows the
      ionosphere's along-track shift it keeps `-2 pi f_dc dt` (Joughin's
      "carrier error"), which scales with frequency, so no split-spectrum
      screen removes it.

    Fields add (`measured + reference_shift`), each part keeping its own
    `carrier`. `carrier_phase` converts a pair already geocoded with one
    convention to the other without geocoding again.
    """

    CARRIERS = ("geometric", "shifted")

    def __init__(self, x, y, lines, samples, description=None, carrier="geometric"):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        lines = np.asarray(lines, dtype=np.float64)
        samples = np.asarray(samples, dtype=np.float64)
        shape = (y.size, x.size)
        if lines.shape != shape or samples.shape != shape:
            raise ValueError(f"lines/samples must be (len(y), len(x)) = {shape}")
        if x.size < 2 or y.size < 2:
            raise ValueError("a registration field needs at least 2 nodes along x and y")
        if not (np.isfinite(lines).all() and np.isfinite(samples).all()):
            raise ValueError("the registration field has NaN; fill its gaps first")
        dx, dy = np.diff(x), np.diff(y)
        if not (np.allclose(dx, dx[0]) and np.allclose(dy, dy[0])):
            raise ValueError("a registration field needs a regular grid")
        if carrier not in self.CARRIERS:
            raise ValueError(f"carrier must be one of {self.CARRIERS}, not {carrier!r}")
        self.x, self.y = x, y
        self.lines, self.samples = lines, samples
        self.description = description or "registration field"
        self.carrier = carrier

    @classmethod
    def from_offsets(cls, x, y, along_track, slant_range, along_track_spacing,
                     range_spacing, description=None, carrier="geometric"):
        """A field from offsets in metres, e.g. a GUNW's `pixelOffsets`.

        `along_track / along_track_spacing` lines and `slant_range / range_spacing`
        samples, with `along_track_spacing` the RSLC's ground-track metres per line.
        """
        return cls(x, y, np.asarray(along_track, dtype=np.float64) / along_track_spacing,
                   np.asarray(slant_range, dtype=np.float64) / range_spacing,
                   description=description, carrier=carrier)

    def carrier_lines(self, x, y):
        """The lines of this field whose carrier stays at the unshifted position:
        all of them for ``carrier="geometric"``, none for ``"shifted"``."""
        lines, _ = self(x, y)
        return lines if self.carrier == "geometric" else np.zeros_like(lines)

    def carrier_phase(self, x, y, doppler_hz, line_rate):
        """Interferogram phase (rad) that ``"geometric"`` adds to ``"shifted"`` at map `x`, `y`.

        `2 pi f_dc lines / line_rate`, with `doppler_hz` the band's Doppler
        centroid there and `line_rate` the RSLC's lines per second
        (`1 / zeroDopplerTimeSpacing`). With this field on the secondary, add
        it to the phase of `reference * conj(secondary)` to turn a pair geocoded
        with ``"shifted"`` into the ``"geometric"`` one, or subtract it for the
        reverse. It is the same fraction of every band's phase, so a
        split-spectrum screen does not see it and the whole of it goes to the
        non-dispersive phase.
        """
        lines, _ = self(x, y)
        return 2.0 * np.pi * np.asarray(doppler_hz, dtype=np.float64) * lines / float(line_rate)

    def __add__(self, other):
        return RegistrationSum(self, other)

    def __radd__(self, other):
        return RegistrationSum(other, self)

    def __call__(self, x, y):
        from scipy.ndimage import map_coordinates

        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        col = (x - self.x[0]) / (self.x[1] - self.x[0])
        row = (y - self.y[0]) / (self.y[1] - self.y[0])
        coords = np.stack([row.ravel(), col.ravel()])
        sample = lambda f: map_coordinates(f, coords, order=1, mode="nearest").reshape(x.shape)  # noqa: E731
        return sample(self.lines), sample(self.samples)

    def __repr__(self):
        return (f"RegistrationField({self.description!r}, {self.y.size} x {self.x.size} nodes, "
                f"lines {np.min(self.lines):+.3f}..{np.max(self.lines):+.3f}, "
                f"samples {np.min(self.samples):+.3f}..{np.max(self.samples):+.3f}, "
                f"carrier {self.carrier})")


class RegistrationSum:
    """Registration fields added together (`field_a + field_b`); each part keeps its own `carrier`."""

    def __init__(self, *parts):
        flat = []
        for part in parts:
            flat.extend(part.parts if isinstance(part, RegistrationSum) else [part])
        if not flat:
            raise ValueError("a registration sum needs at least one field")
        self.parts = tuple(flat)
        self.description = " + ".join(_registration_label(p) for p in self.parts)

    def __call__(self, x, y):
        lines = samples = 0.0
        for part in self.parts:
            dl, ds = part(x, y)
            lines, samples = lines + np.asarray(dl, float), samples + np.asarray(ds, float)
        return lines, samples

    def carrier_lines(self, x, y):
        """The sum of the parts' `carrier_lines`; a plain callable part contributes none."""
        total = np.zeros(np.shape(x))
        for part in self.parts:
            lines = _carrier_lines(part, x, y)
            if lines is not None:
                total = total + lines
        return total

    def __add__(self, other):
        return RegistrationSum(self, other)

    def __radd__(self, other):
        return RegistrationSum(other, self)

    def __repr__(self):
        return f"RegistrationSum({self.description})"


def _carrier_lines(registration, x, y):
    """Lines whose carrier stays at the unshifted position, or None for a plain callable (whose
    shifts keep ISCE3's carrier at the sampled position)."""
    fn = getattr(registration, "carrier_lines", None)
    return None if fn is None else np.asarray(fn(x, y), dtype=np.float64)


def _registration_label(registration):
    """How a registration is recorded in `runConfigurationContents`."""
    if isinstance(registration, RegistrationSum):
        return registration.description
    label = getattr(registration, "description", type(registration).__name__)
    carrier = getattr(registration, "carrier", None)
    return label if carrier is None else f"{label} (carrier {carrier})"


def _timing_hooks(azimuth_shift):
    """`(shift(t), transition(t) or None)` from a callable or a `nisar_tools.l0b.TimingReport`."""
    if azimuth_shift is None:
        return None, None
    if hasattr(azimuth_shift, "azimuth_shift") and hasattr(azimuth_shift, "transition"):
        return azimuth_shift.azimuth_shift, azimuth_shift.transition
    return azimuth_shift, None


def _measured_band_offset(range_shift):
    """The `qa.BandOffset` a `range_shift` value is, given as itself or as its `shift`; else None."""
    if isinstance(range_shift, BandOffset):
        return range_shift
    owner = getattr(range_shift, "__self__", None)
    if isinstance(owner, BandOffset) and getattr(range_shift, "__func__", None) is BandOffset.shift:
        return owner
    return None


def _range_shift_hook(range_shift, corrections, rslc, freq, clamp=False):
    """The `range_shift(t, r)` that geocodes band `freq`.

    A `qa.BandOffset` was measured on the RSLC, so besides band B's own offset
    it holds B's dispersive delay against A: B, the higher band, is delayed
    less, 0.0217 m per slant TECU. The timing corrections remove each band's
    own delay at its own centre frequency, so wherever they apply the TEC part
    (everywhere with `clamp`, as `radar_indices`), that part is taken out of
    the shift here; left in, B would lose it twice. Any other callable is
    returned as given: an offset of the band's own, such as k x 3.1228 m +
    0.98 m.
    """
    offset = _measured_band_offset(range_shift)
    if offset is None:
        return range_shift
    ionosphere = getattr(corrections, "ionosphere_range", None)
    if ionosphere is None:
        return offset.shift
    table = corrections.slant_range
    # the TEC delay goes as 1/f^2: tau_X - tau_A = tau_X (1 - f_X^2 / f_A^2)
    ratio = 1.0 - (rslc.center_frequency(freq) / rslc.center_frequency("A")) ** 2

    def shift(t, r):
        t, r = np.asarray(t, float), np.asarray(r, float)
        inside = np.isfinite(t) & np.isfinite(r)
        if not clamp:
            inside &= table.contains(np.where(inside, t, table.y_start), np.where(inside, r, table.x_start))
        delay = ionosphere.eval(np.where(inside, t, ionosphere.y_start), np.where(inside, r, ionosphere.x_start))
        return np.asarray(offset.shift(t, r), float) - np.where(inside, ratio * delay, 0.0)

    return shift


def pixel_radar_indices(x, y, epsg, dem, grid, orbit, doppler, valid, corrections=None,
                        registration=None, range_shift=None, azimuth_shift=None,
                        chip_rows=False, clamp_corrections=False):
    """Steps 1-5 of the geocoding for output pixels at map `x`, `y` (1-D).

    The DEM height, zero-Doppler `geo2rdr`, the timing `corrections`, the
    RSLC's own `azimuth_shift(t)` (s) and `range_shift(t, r)` (m) at the
    corrected time and range, and last the `registration` shifts (see
    `RegistrationField`). Returns `(lon, lat, h, az_index, rg_index,
    flatten_range, mask)`, lon/lat in radians. `flatten_range` is the geometric
    range in every case, and the subswath `mask` is that of the position
    *before* the shifts -- a shift of a line or two does not change it except
    at a subswath edge -- except that a `TimingReport` given as `azimuth_shift`
    sets it to 0 where a timing defect's transition mixes good and bad echoes.
    `chip_rows` and `clamp_corrections` are `radar_indices`'s fixes of ISCE3's
    mask and correction-domain tests.
    """
    lon, lat = _to_lonlat(epsg, x, y)
    lon, lat = np.deg2rad(lon), np.deg2rad(lat)
    h = dem.heights(lon, lat)
    az_corr = corrections.azimuth if corrections is not None else None
    rg_corr = corrections.slant_range if corrections is not None else None
    az, rg, rng, mask = radar_indices(lon, lat, h, grid, orbit, doppler, valid, az_corr, rg_corr,
                                      chip_rows=chip_rows, clamp_corrections=clamp_corrections)
    shift, transition = _timing_hooks(azimuth_shift)
    if shift is not None or range_shift is not None:
        t = grid.sensing_start + az / grid.prf
        r = grid.starting_range + rg * grid.range_pixel_spacing
        if shift is not None:
            az = az + np.asarray(shift(t), float) * grid.prf
        if range_shift is not None:
            rg = rg + np.asarray(range_shift(t, r), float) / grid.range_pixel_spacing
        if transition is not None:
            mask = np.where(transition(t) & (mask != 255), np.uint8(0), mask)
    if registration is not None:
        d_lines, d_samples = registration(x, y)
        az = az + d_lines
        rg = rg + d_samples
    return lon, lat, h, az, rg, rng, mask


# ---------------------------------------------------------------------------
# The workflow
# ---------------------------------------------------------------------------
def product_bounds(path):
    """`((x_min, y_max, x_max, y_min), epsg)` of a NISAR L2 product's map box.

    Any geocoded product works (GSLC, GUNW, GCOV, GOFF): ASF grids every L2
    product of a track/frame on the same box, so this is how to put a GSLC
    made here on the grid ASF uses for that frame.
    """
    import h5py

    with h5py.File(path, "r") as f:
        found = []

        def visit(name, obj):
            if isinstance(obj, h5py.Group) and "xCoordinates" in obj and "yCoordinates" in obj \
                    and "projection" in obj and "/grids/" in f"/{name}/":
                found.append(name)
                return True
            return None

        f["science"].visititems(visit)
        if not found:
            raise ValueError(f"{path}: no geocoded grid with x/yCoordinates and projection")
        g = f["science"][found[0]]
        x = g["xCoordinates"][()]
        y = g["yCoordinates"][()]
        epsg = int(np.asarray(g["projection"][()]))
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
    return (float(x[0] - dx / 2), float(y[0] - dy / 2),
            float(x[-1] + dx / 2), float(y[-1] + dy / 2)), epsg


def geocode_rslc(rslc, output, dem, *, frequencies=None, polarizations=None,
                 epsg=None, bounds=None, like=None, posting=None, snap=240.0,
                 orbit_file=None, tec_file=None, solid_earth_tides=True,
                 flatten=True, registration=None, range_shift=None, azimuth_shift=None,
                 isce3_compatible="pixels", block_size=1024, workers=None, granule_id=None,
                 dem_description=None, overwrite=False, log=print):
    """Geocode an RSLC into a NISAR L2 GSLC HDF5 file. Returns the output path.

    Parameters
    ----------
    rslc : str, Path or RSLC
    output : str or Path
        Output `.h5`. A directory writes `default_gslc_name` into it.
    dem : str or Path
        Heights above the WGS84 ellipsoid. Use the NISAR DEM v1.2
        (`nisar_tools.dem.stage_nisar_dem`) to flatten against the same
        surface as ASF's products -- a different DEM changes the phase by
        ~0.4 rad per centimetre of height difference.
    frequencies, polarizations
        Subsets to process (default: everything in the RSLC). `polarizations`
        is a list applied to every frequency or a `{freq: [pols]}` dict.
    epsg : int, optional
        Output projection (default: the UTM zone of the swath centre).
    bounds : (x_min, y_max, x_max, y_min), optional
        Explicit output box in `epsg` units -- the SDS's
        `top_left`/`bottom_right`. Default: swath footprint snapped to
        `snap`.
    like : str or Path, optional
        A NISAR L2 product (GSLC, GUNW, ...) of the same track and frame whose
        box and projection to use instead (`product_bounds`) -- the way to
        land on ASF's frame grid. Excludes `bounds` and `epsg`.
    orbit_file : optional
        NISAR orbit-ephemeris XML (MOE/POE) instead of the embedded orbit.
    tec_file : optional
        NISAR IMAGEN TEC JSON (`NISAR_ANC_TEC_*`) for the ionospheric
        geolocation correction; see `nisar_tools.download.download_tec`.
    solid_earth_tides : bool
        Apply the solid Earth tide correction (needs `pysolid`).
    registration : callable or dict, optional
        `registration(x, y) -> (lines, samples)`, or `{frequency: registration}`:
        extra fractional radar shifts, in that band's own lines and samples, for
        output pixels at map `x`, `y` (output projection). The bands' range
        samples differ (band B's are about 8 times A's), so one callable is taken
        only when a single band is geocoded;
        added after the timing corrections and before interpolation; the
        flattening range is not changed. Usually a `RegistrationField`
        built from dense offsets, to make this GSLC the coregistered
        secondary of a reference GSLC -- see that class for why ASF's GSLCs
        need it, and for its `carrier` convention (a plain callable keeps
        ISCE3's, the carrier at the shifted position). Recorded in
        `runConfigurationContents`.
    range_shift : dict, optional
        `{frequency: f(t, r)}`, metres to add to that band's slant range at
        zero-Doppler time `t` and range `r`: corrects the band's own range
        error. The flattening range is not changed. For band B pass
        `nisar_tools.qa.band_offset(rslc)` or its `shift`: B's offset against
        A (whole periods of the 48 MHz clock, constant per data window, plus
        B's calibration), which otherwise leaks deformation into
        split-spectrum screens. Measured on the RSLC, it also holds B's
        dispersive delay against A, which a `tec_file` removes on its own
        (each band's at its own frequency): that part is then taken out of the
        offset wherever the TEC correction applies, or B would lose it twice
        and land 0.0217 m per slant TECU off A. Any other callable is applied
        as given.
    azimuth_shift : callable or `nisar_tools.l0b.TimingReport`, optional
        `f(t)`, seconds to add to zero-Doppler time `t` in every band: corrects
        pulses tagged with the wrong transmit time. A `TimingReport` shifts by
        a pulse where a slip fully decides the pixel and sets the mask to 0
        where a slip's edge mixes good and bad echoes; unless
        `isce3_compatible=True`, it also restores the start of a datatake that
        ISCE3 left out of `validSamplesSubSwath` (`TimingReport.repair_mask`).
        Do not combine with a `registration` measured on this RSLC, which
        already holds the shift.
    isce3_compatible : {"pixels", True, False}
        Which of ISCE3 v0.25.16's defects to reproduce.
        - `"pixels"` (default): pixels identical to ISCE3's, so this GSLC
          pairs with ASF's; masks and metadata fixed. The mask tests every row
          of the interpolation chip (ISCE3: its corners), lines that drop out
          of `validSamplesSubSwath` are repaired, band B's
          `inputDataExceptionMask` is on B's own grid, the cube's
          ground-track velocity uses a proper `rdr2geo` tolerance, and the
          real and imaginary statistics of complex LUTs are real numbers.
        - `True`: ISCE3's masks and metadata too.
        - `False`: also fixes the pixels: a centred DEM spline window (ISCE3's
          is one node off, ~0.4 rad of flattening phase per cm of height it
          gets wrong), the Doppler table read at zero-Doppler coordinates,
          timing corrections clamped at their table's edge (ISCE3 skips them
          there, leaving pixels near the swath edge uncorrected or NaN), and
          the total TEC alone where the TEC file flags a topside sample over
          the frame (ISCE3 v0.25.16 subtracts it unchecked).
          Geocode both dates of a pair this way, never with an ASF GSLC.
    block_size : int
        Output block edge in pixels; rounded to a multiple of the 512 chunk.
    workers : int, optional
        Threads (default: CPU count).
    """
    start_wall = time.perf_counter()
    own = not isinstance(rslc, RSLC)
    if own:
        rslc = RSLC(rslc)
    workers = workers or os.cpu_count() or 1
    block_size = max(gp.CHUNK, (int(block_size) // gp.CHUNK) * gp.CHUNK)

    output = Path(output)
    if output.is_dir():
        output = output / (default_gslc_name(rslc.granule_id) + ".h5")
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} exists; pass overwrite=True")
    partial = output.with_name(output.name + ".partial")

    freq_pols = _select(rslc, frequencies, polarizations)
    first = sorted(freq_pols)[0]
    if registration is not None and not isinstance(registration, dict):
        if len(freq_pols) > 1:
            raise ValueError(
                f"registration shifts are in one band's lines and samples, and bands {sorted(freq_pols)} are "
                "geocoded: pass registration={band: field} with one field per band, or geocode one band at a time "
                "(frequencies=)")
        registration = {first: registration}
    orbit = load_orbit(rslc, orbit_file)
    if like is not None:
        if bounds is not None or epsg is not None:
            raise ValueError("pass either like= or bounds=/epsg=, not both")
        bounds, epsg = product_bounds(like)
    epsg = int(epsg or rslc.utm_epsg(first))

    if isce3_compatible not in (True, False, "pixels"):
        raise ValueError(f"isce3_compatible must be True, False or 'pixels', not {isce3_compatible!r}")
    fix_metadata, fix_pixels = isce3_compatible is not True, isce3_compatible is False
    grids = _product_grids(rslc, freq_pols, epsg, bounds, posting, snap)
    # ISCE3's metadata postings (RunConfig.prep_geocode_metadata): 500 m cubes and 1 km LUTs, or on a geographic
    # grid 0.005 and 0.01 degrees, where a missing elif gives ISCE3's cubes 0.01 too
    geographic = epsg == 4326
    cube_spacing = (0.005 if fix_metadata else 0.01) if geographic else 500.0
    cube_grid = metadata_geogrid(grids[first], cube_spacing, cube=True)
    lut_grid = metadata_geogrid(grids[first], 0.01 if geographic else 1000.0)
    dem_interp = DEMInterpolator(dem, centered=fix_pixels)
    for f_, g in grids.items():
        log(f"frequency {f_}: {g.length} x {g.width} px at {g.spacing_x:g} x {-g.spacing_y:g} m, "
            f"EPSG:{epsg}, x {g.start_x:.1f}..{g.end_x:.1f}, y {g.end_y:.1f}..{g.start_y:.1f}")

    processing_datetime = _dt.datetime.now(_dt.timezone.utc)
    granule_id = granule_id or output.stem
    f = gp.create_product_file(partial)
    try:
        src = rslc._file
        stats = {}
        for freq, pols in freq_pols.items():
            gp.create_image_datasets(f, freq, grids[freq], pols)
            _create_exception_mask(f, freq, grids[freq] if fix_metadata else grids[first])
        corrections = {}
        has_exception = _has_input_data_exception(src)
        for freq, pols in freq_pols.items():
            grid = rslc.radar_grid(freq)
            corrections[freq] = compute_timing_corrections(
                grid, orbit, rslc.center_frequency(freq), dem=dem_interp,
                tec_file=tec_file, solid_earth_tides=solid_earth_tides, check_topside=fix_pixels)
            _write_timing_corrections(f, freq, corrections[freq])
            # ISCE3 samples every band's exception mask at the first band's pixels
            bands = ([freq] if fix_metadata else list(freq_pols) if freq == first else []) if has_exception else []
            stats.update(_geocode_frequency(
                f, rslc, freq, pols, grids[freq], grid, orbit, dem_interp,
                corrections[freq], flatten, block_size, workers, log, bands,
                registration=(registration or {}).get(freq),
                range_shift=_range_shift_hook((range_shift or {}).get(freq), corrections[freq], rslc, freq,
                                              clamp=fix_pixels),
                azimuth_shift=azimuth_shift, fix_metadata=fix_metadata, fix_pixels=fix_pixels))
        for (freq, pol), st in stats.items():
            st.write(f[f"{gp.PRODUCT}/grids/frequency{freq}/{pol}"])

        log("radar grid cubes ...")
        _write_cubes(f, rslc, first, cube_grid, orbit, workers,
                     rdr2geo_threshold=1e-5 if fix_metadata else None)
        log("metadata ...")
        run_text = _gslc_metadata.describe_run(
            software=SOFTWARE_VERSION, input_file=rslc.path, output_file=str(output),
            dem_file=dem, orbit_file=orbit_file, tec_file=tec_file,
            solid_earth_tides=solid_earth_tides, flatten=flatten,
            registration=None if not registration else {
                k: _registration_label(v) for k, v in registration.items() if k in freq_pols},
            range_shift=None if not range_shift else {
                k: _describe(v) + ("; its dispersive delay against band A taken out where the TEC correction applies"
                                   if tec_file is not None and _measured_band_offset(v) is not None else "")
                for k, v in range_shift.items() if k in freq_pols},
            azimuth_shift=None if azimuth_shift is None else _describe(azimuth_shift),
            isce3_compatible=isce3_compatible,
            frequencies=dict(freq_pols), epsg=epsg, bounds=bounds,
            like=str(like) if like is not None else None, snap=snap,
            geo2rdr_threshold=1e-8, geo2rdr_maxiter=25, block_size=block_size,
            output_data_type="complex64_zero_mantissa", mantissa_bits=10)
        _gslc_metadata.populate_metadata(
            f, rslc, freq_pols, grids, lut_grid, orbit, dem_interp,
            granule_id=granule_id, processing_datetime=processing_datetime,
            orbit_file=orbit_file, tec_file=tec_file, flatten=flatten,
            solid_earth_tides=solid_earth_tides, dem_path=dem,
            dem_description=dem_description or _dem_description(dem, rslc),
            bounds=bounds, snap=snap, epsg=epsg, software_version=SOFTWARE_VERSION,
            run_config_text=run_text, fix_metadata=fix_metadata)
        gp.decorate_from_spec(f)
    except BaseException:
        f.close()
        partial.unlink(missing_ok=True)
        raise
    f.close()
    os.replace(partial, output)
    if own:
        rslc.close()
    log(f"wrote {output} in {time.perf_counter() - start_wall:.0f} s")
    return output


def _select(rslc, frequencies, polarizations):
    freqs = list(frequencies) if frequencies else list(rslc.frequencies)
    out = {}
    for freq in freqs:
        if freq not in rslc.frequencies:
            raise ValueError(f"frequency {freq} not in {rslc.granule_id}")
        available = sorted(rslc.polarizations(freq))
        if polarizations is None:
            want = available
        elif isinstance(polarizations, dict):
            want = sorted(polarizations.get(freq) or available)
        else:
            want = sorted(polarizations)
        missing = [p for p in want if p not in available]
        if missing:
            raise ValueError(f"polarizations {missing} not in frequency {freq}")
        out[freq] = want
    return out


def _product_grids(rslc, freq_pols, epsg, bounds, posting, snap):
    """One box for every frequency, as the SDS passes one top_left/bottom_right."""
    first = sorted(freq_pols)[0]
    if bounds is None:
        g = rslc.default_geogrid(first, epsg=epsg, snap=snap,
                                 posting=posting.get(first) if isinstance(posting, dict) else posting)
        bounds = (g.start_x, g.start_y, g.end_x, g.end_y)
    x_min, y_max, x_max, y_min = bounds
    grids = {}
    for freq in freq_pols:
        p = posting.get(freq) if isinstance(posting, dict) else posting
        sx, sy = p if p is not None else rslc.default_posting(freq)
        grids[freq] = geogrid_from_bounds(x_min, y_max, x_max, y_min, sx, sy, epsg)
    return grids


def _has_input_data_exception(src):
    path = gp.IDENT + "/hasInputDataException"
    return path in src and int(np.asarray(src[path][()])) != 0


def _create_exception_mask(f, freq, grid):
    """`inputDataExceptionMask`: uint8 on `grid` (ISCE3: the *first* frequency's, for every band)."""
    root = f"{gp.PRODUCT}/grids/frequency{freq}"
    group = f[root]
    shape = (grid.length, grid.width)
    ds = group.create_dataset("inputDataExceptionMask", shape=shape, dtype=np.uint8,
                              chunks=gp._chunks(shape), compression="gzip",
                              compression_opts=9, shuffle=False)
    # The frequency's *own* scales even when they are the wrong length for this
    # shape (frequency B): HDF5 does not check, and ISCE3's writer does not either.
    ds.dims[0].attach_scale(group["yCoordinates"])
    ds.dims[1].attach_scale(group["xCoordinates"])
    ds.attrs["grid_mapping"] = np.bytes_("projection")
    return ds


def _write_timing_corrections(f, freq, corr):
    """`AzSrgCorrections.write_corrections_hdf5`."""
    if not corr.arrays:
        return
    group = f.require_group(f"{gp.PRODUCT}/metadata/processingInformation/timingCorrections/frequency{freq}")
    epoch = np.bytes_(radar.epoch_units(corr.grid.ref_epoch))
    items = [
        ("slantRange", corr.rg_vec, {"units": np.bytes_("meters")}),
        ("slantRangeSpacing", np.float64(corr.grid.range_pixel_spacing), {"units": np.bytes_("meters")}),
        ("zeroDopplerTime", corr.az_vec, {"units": epoch}),
        ("zeroDopplerTimeSpacing", np.float64(1.0 / corr.grid.prf), {"units": np.bytes_("seconds")}),
    ]
    units = {"azimuthIonosphere": epoch, "slantRangeIonosphere": np.bytes_("meters"),
             "slantRangeSolidEarthTides": np.bytes_("meters")}
    for name in ("azimuthIonosphere", "slantRangeIonosphere", "slantRangeSolidEarthTides"):
        if name in corr.arrays:
            items.append((name, np.asarray(corr.arrays[name], np.float64),
                          {"units": units[name], "_FillValue": np.nan}))
    for name, value, attrs in items:
        ds = group.create_dataset(name, data=value)
        for k, v in attrs.items():
            ds.attrs[k] = v


# ---------------------------------------------------------------------------
# The block loop
# ---------------------------------------------------------------------------
_TRANSFORMERS = threading.local()


def _describe(hook):
    """A short label of a shift hook for `runConfigurationContents`."""
    owner = getattr(hook, "__self__", hook)
    if hasattr(owner, "granule"):
        return f"{type(owner).__name__} of {owner.granule}"
    return getattr(hook, "__name__", type(hook).__name__)


def _to_lonlat(epsg, x, y):
    from pyproj import Transformer

    cache = getattr(_TRANSFORMERS, "cache", None)
    if cache is None:
        cache = _TRANSFORMERS.cache = {}
    tr = cache.get(epsg)
    if tr is None:
        tr = cache[epsg] = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    return tr.transform(x, y)


def _block_may_touch_swath(geogrid, row0, row1, col0, col1, grid, orbit, margin_px=64):
    """Cheap cull: do the block's edges, at -500 m and 9000 m, reach the grid?

    The map->radar mapping over one block is nearly affine, so the radar-space
    bounding box of its perimeter at the extreme heights contains the block.
    """
    n = 5
    rows = np.concatenate([np.full(n, row0), np.full(n, row1), np.linspace(row0, row1, n), np.linspace(row0, row1, n)])
    cols = np.concatenate([np.linspace(col0, col1, n), np.linspace(col0, col1, n), np.full(n, col0), np.full(n, col1)])
    x = geogrid.start_x + (cols + 0.5) * geogrid.spacing_x
    y = geogrid.start_y + (rows + 0.5) * geogrid.spacing_y
    lon, lat = _to_lonlat(geogrid.epsg, x, y)
    ts, rs = [], []
    for h in (-500.0, 9000.0):
        llh = np.stack([np.deg2rad(lon), np.deg2rad(lat), np.full(x.size, h)], axis=1)
        t, r, ok = radar.geo2rdr(llh, orbit, None, grid.wavelength, grid.look_side,
                                 t0=grid.sensing_mid, threshold=1e-6, maxiter=25)
        ts.append(t[ok]); rs.append(r[ok])
    t, r = np.concatenate(ts), np.concatenate(rs)
    if t.size == 0:
        return True
    line = (t - grid.sensing_start) * grid.prf
    sample = (r - grid.starting_range) / grid.range_pixel_spacing
    return not (line.max() < -margin_px or line.min() > grid.length + margin_px
                or sample.max() < -margin_px or sample.min() > grid.width + margin_px)


def _valid_samples(rslc, freq, fix_metadata, azimuth_shift=None):
    """The band's `validSamplesSubSwath`, repaired unless ISCE3's masks are reproduced: lines that
    drop out for one line, and a datatake start a `TimingReport` finds masked."""
    from .qa import repair_drops

    valid = rslc.valid_samples(freq)
    if fix_metadata:
        valid = repair_drops(valid)
        if hasattr(azimuth_shift, "repair_mask"):
            valid = azimuth_shift.repair_mask(valid)
    return valid


def _geocode_frequency(f, rslc, freq, pols, geogrid, grid, orbit, dem, corr,
                       flatten, block_size, workers, log, exception_bands,
                       registration=None, range_shift=None, azimuth_shift=None,
                       fix_metadata=False, fix_pixels=False):
    root = f"{gp.PRODUCT}/grids/frequency{freq}"
    writers = {pol: gp.ChunkWriter(f[f"{root}/{pol}"]) for pol in pols}
    mask_writer = gp.ChunkWriter(f[f"{root}/mask"])
    stats = {(freq, pol): gp.ComplexStats() for pol in pols}
    readers = {pol: rslc.reader(freq, pol) for pol in pols}
    doppler = rslc.doppler_centroid(freq, zero_doppler=fix_pixels)
    valid = _valid_samples(rslc, freq, fix_metadata, azimuth_shift)

    blocks = []
    for row0 in range(0, geogrid.length, block_size):
        for col0 in range(0, geogrid.width, block_size):
            blocks.append((row0, min(row0 + block_size, geogrid.length),
                           col0, min(col0 + block_size, geogrid.width)))

    lock = threading.Lock()
    progress = {"done": 0, "skipped": 0, "pixels": 0, "t0": time.perf_counter()}

    def run(block):
        row0, row1, col0, col1 = block
        if not _block_may_touch_swath(geogrid, row0, row1 - 1, col0, col1 - 1, grid, orbit):
            with lock:
                progress["skipped"] += 1
            return
        xs = geogrid.x_coordinates[col0:col1]
        ys = geogrid.y_coordinates[row0:row1]
        X, Y = np.meshgrid(xs, ys)
        lon, lat, h, az, rg, rng, mask = pixel_radar_indices(
            X.ravel(), Y.ravel(), geogrid.epsg, dem, grid, orbit, doppler, valid,
            corr, registration, range_shift=range_shift, azimuth_shift=azimuth_shift,
            chip_rows=fix_metadata, clamp_corrections=fix_pixels)
        shape = (row1 - row0, col1 - col0)
        ok = np.isfinite(az)
        if ok.any():
            l0 = max(int(np.floor(np.nanmin(az))) - 5, 0)
            l1 = min(int(np.ceil(np.nanmax(az))) + 5, grid.length - 1)
            s0 = max(int(np.floor(np.nanmin(rg))) - 5, 0)
            s1 = min(int(np.ceil(np.nanmax(rg))) + 5, grid.width - 1)
            windows = [np.asarray(readers[pol][l0:l1 + 1, s0:s1 + 1]) for pol in pols]
            carrier = None if registration is None else _carrier_lines(registration, X.ravel(), Y.ravel())
            if carrier is not None and not np.any(carrier):
                carrier = None                       # all "shifted": ISCE3's arithmetic exactly
            interpolated = interpolate_slc(windows, l0, s0, az, rg, rng, grid, doppler,
                                           flatten=flatten, carrier_lines=carrier)
            for pol, values in zip(pols, interpolated):
                values = values.reshape(shape)
                truncate_mantissa(values)
                for r_ in range(0, shape[0], gp.CHUNK):
                    for c_ in range(0, shape[1], gp.CHUNK):
                        chunk = values[r_:r_ + gp.CHUNK, c_:c_ + gp.CHUNK]
                        stats[(freq, pol)].update(chunk)
                        writers[pol].write(row0 + r_, col0 + c_, chunk)
        mask = mask.reshape(shape)
        for r_ in range(0, shape[0], gp.CHUNK):
            for c_ in range(0, shape[1], gp.CHUNK):
                mask_writer.write(row0 + r_, col0 + c_, mask[r_:r_ + gp.CHUNK, c_:c_ + gp.CHUNK])
        if exception_bands:
            _exception_mask_block(f, rslc, exception_bands, lon, lat, h, grid, orbit,
                                  row0, col0, shape)
        with lock:
            progress["done"] += 1
            progress["pixels"] += int(ok.sum())

    t0 = time.perf_counter()
    log(f"frequency {freq}: {len(blocks)} blocks of {block_size} px, {workers} threads, pols {pols}")
    with ThreadPoolExecutor(workers, thread_name_prefix=f"gslc{freq}") as pool:
        futures = [pool.submit(run, b) for b in blocks]
        last = time.perf_counter()
        for i, fut in enumerate(as_completed(futures), 1):
            fut.result()
            now = time.perf_counter()
            if now - last > 30 or i == len(futures):
                last = now
                log(f"  {freq}: {i}/{len(futures)} blocks ({progress['skipped']} outside swath), "
                    f"{progress['pixels'] / 1e6:.0f} M px, {now - t0:.0f} s")
    return stats


def _exception_mask_block(f, rslc, bands, lon, lat, h, grid, orbit, row0, col0, shape):
    """Nearest-neighbour `inputDataExceptionMask` of each of `bands` at these pixels.

    GeocodeCov with a Byte raster: zero-Doppler geo2rdr without timing
    corrections, nearest sample, 0 where not geocoded.
    """
    t, r, ok = radar.geo2rdr(np.stack([lon, lat, h], axis=1), orbit, None,
                             grid.wavelength, grid.look_side, t0=grid.sensing_mid)
    for freq in bands:
        fgrid = rslc.radar_grid(freq)
        row = (t - fgrid.sensing_start) * fgrid.prf
        col = (r - fgrid.starting_range) / fgrid.range_pixel_spacing
        good = ok & fgrid.contains(t, r) & (row >= 0) & (col >= 0) \
            & (row <= fgrid.length - 1) & (col <= fgrid.width - 1)
        out = np.zeros(t.size, np.uint8)
        if good.any():
            ds = rslc.swath(freq)["inputDataExceptionMask"]
            rr = radar._round_half_away(row[good])
            cc = radar._round_half_away(col[good])
            l0, l1, s0, s1 = rr.min(), rr.max(), cc.min(), cc.max()
            with gp._H5_WRITE_LOCK:
                window = ds[l0:l1 + 1, s0:s1 + 1]
            out[good] = window[rr - l0, cc - s0]
        out = out.reshape(shape)
        writer = gp.ChunkWriter(f[f"{gp.PRODUCT}/grids/frequency{freq}/inputDataExceptionMask"])
        for r_ in range(0, shape[0], gp.CHUNK):
            for c_ in range(0, shape[1], gp.CHUNK):
                writer.write(row0 + r_, col0 + c_, out[r_:r_ + gp.CHUNK, c_:c_ + gp.CHUNK])


def _write_cubes(f, rslc, freq, cube_grid, orbit, workers, rdr2geo_threshold=None):
    """`add_radar_grid_cubes_to_hdf5` with the RSLC's (first) frequency grid."""
    from pyproj import Transformer

    root = f"{gp.PRODUCT}/metadata/radarGrid"
    grid = rslc.radar_grid(freq)
    datasets = gp.create_cube_datasets(f, root, cube_grid, CUBE_HEIGHTS, grid.ref_epoch)
    doppler = rslc.doppler_centroid(freq)
    X, Y = np.meshgrid(cube_grid.x_coordinates, cube_grid.y_coordinates)
    lon, lat = Transformer.from_crs(f"EPSG:{cube_grid.epsg}", "EPSG:4326",
                                    always_xy=True).transform(X, Y)
    lon, lat = np.deg2rad(lon), np.deg2rad(lat)

    def layer(k):
        return k, radar.radar_grid_cube_layer(lon, lat, CUBE_HEIGHTS[k], grid, orbit, doppler,
                                              rdr2geo_threshold=rdr2geo_threshold)

    with ThreadPoolExecutor(workers) as pool:
        for k, out in pool.map(layer, range(len(CUBE_HEIGHTS))):
            with gp._H5_WRITE_LOCK:
                for name, arr in out.items():
                    datasets[name][k] = arr.astype(radar.CUBE_LAYERS[name])


def _dem_description(dem, rslc):
    """The RSLC's own DEM description when `dem` was staged from NISAR DEM v1.2.

    `nisar_tools.dem.stage_nisar_dem` tags its output; an RSLC processed
    by the SDS records the same DEM's description, so reusing it states the
    truth in the mission's words. Otherwise None, which is written as
    "(NOT SPECIFIED)".
    """
    import rasterio

    try:
        with rasterio.open(str(dem)) as src_dem:
            tags = src_dem.tags()
    except Exception:  # noqa: BLE001
        tags = {}
    tagged = tags.get("NISAR_DEM_VERSION")
    if tagged is None and "NISAR mission" in tags.get("DESCRIPTION", ""):
        tagged = tags.get("VERSION")          # the mission's own COG tags
    path = f"{gp.RSLC_PRODUCT}/metadata/processingInformation/inputs/demSource"
    if tagged and path in rslc._file:
        text = rslc._file[path][()]
        text = text.decode() if isinstance(text, bytes) else str(text)
        if f"version {tagged}" in text:
            return text
    return None
