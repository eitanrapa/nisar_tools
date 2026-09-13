"""The :class:`RSLC` class: one NISAR L1 RSLC granule, and the way to a GSLC.

An RSLC is the range-Doppler SLC every other NISAR product is made from; a GSLC
is that image geocoded onto a map grid. ASF normally ships both, but not always
-- for some acquisitions only the RSLC exists. :meth:`RSLC.to_gslc` makes the
missing GSLC, as an HDF5 file in the NISAR L2 GSLC layout that :class:`GSLC`
(and so the rest of this package) reads like any ASF granule.

The geocoding is a port of the ISCE3 v0.25.16 GSLC workflow that produces ASF's
products -- see :mod:`nisar_tools.gslc_geocode` for the algorithm and
:mod:`nisar_tools.gslc_product` for the file layout. This module holds the
reader and the grid definitions.
"""

from dataclasses import dataclass
from functools import cached_property

import h5py
import numpy as np

from . import radar
from .gslc import DirectChunkReader, _decode

RSLC_PATH = "science/LSAR/RSLC"
IDENT_PATH = "science/LSAR/identification"

#: GSLC postings (x, y) in metres by processed range bandwidth (MHz), from the
#: NISAR data user guide's GSLC pixel-spacing table.
POSTING_BY_BANDWIDTH = {5: (40.0, 5.0), 20: (10.0, 5.0), 40: (5.0, 5.0), 77: (2.5, 5.0)}


@dataclass(frozen=True)
class GeoGrid:
    """A north-up map grid: ``start`` is the top-left *corner*, not a centre.

    Same convention as ``isce3::product::GeoGridParameters``: pixel ``(i, j)``
    is centred at ``(start_x + (j + 0.5) * spacing_x, start_y + (i + 0.5) *
    spacing_y)`` with ``spacing_y < 0``.
    """

    start_x: float
    start_y: float
    spacing_x: float
    spacing_y: float
    width: int
    length: int
    epsg: int

    def __post_init__(self):
        if self.spacing_y >= 0:
            raise ValueError("GeoGrid spacing_y must be negative (north-up)")
        if self.width <= 0 or self.length <= 0:
            raise ValueError("GeoGrid must be at least one pixel")

    @property
    def end_x(self):
        return self.start_x + self.spacing_x * self.width

    @property
    def end_y(self):
        return self.start_y + self.spacing_y * self.length

    @property
    def x_coordinates(self):
        """Pixel-centre x (``set_get_geo_info``'s ``linspace``)."""
        x0 = self.start_x + 0.5 * self.spacing_x
        return np.linspace(x0, x0 + (self.width - 1) * self.spacing_x, self.width)

    @property
    def y_coordinates(self):
        y0 = self.start_y + 0.5 * self.spacing_y
        return np.linspace(y0, y0 + (self.length - 1) * self.spacing_y, self.length)


def _grid_size(stop, start, size):
    """``geogrid._grid_size``: ``round(|stop - start| / size)``."""
    return int(np.round(abs((stop - start) / size)))


def snap_geogrid(grid, x_snap, y_snap):
    """``geogrid.create``'s snapping: start floor/ceil, end ceil/floor."""
    if x_snap % grid.spacing_x != 0 or y_snap % abs(grid.spacing_y) != 0:
        raise ValueError("snap values must be exact multiples of the spacing")
    start_x = np.floor(grid.start_x / x_snap) * x_snap
    start_y = np.ceil(grid.start_y / y_snap) * y_snap
    end_x = np.ceil(grid.end_x / x_snap) * x_snap
    end_y = np.floor(grid.end_y / y_snap) * y_snap
    return GeoGrid(float(start_x), float(start_y), grid.spacing_x, grid.spacing_y,
                   _grid_size(end_x, start_x, grid.spacing_x),
                   _grid_size(end_y, start_y, grid.spacing_y), grid.epsg)


def geogrid_from_bounds(x_min, y_max, x_max, y_min, spacing_x, spacing_y, epsg):
    """A grid from explicit corner bounds (the PGE's ``top_left``/``bottom_right``)."""
    spacing_y = -abs(spacing_y)
    return GeoGrid(float(x_min), float(y_max), float(spacing_x), spacing_y,
                   _grid_size(x_max, x_min, spacing_x),
                   _grid_size(y_min, y_max, spacing_y), int(epsg))


def metadata_geogrid(product_grid, spacing, cube=False):
    """The ``radarGrid`` (``cube=True``, 500 m) or LUT (1 km) grid of a product.

    ``RunConfig.prep_geocode_metadata`` + ``geogrid.create`` with the first
    product grid as reference: same extent at the coarser posting, cubes padded
    by 5 cells (+1), then snapped to the posting. Reproduces ASF's reference
    GSLC exactly (674 x 690 at 500 m, 332 x 340 at 1 km).
    """
    margin = 5
    width = _grid_size(product_grid.end_x, product_grid.start_x, spacing)
    length = _grid_size(product_grid.end_y, product_grid.start_y, -spacing)
    start_x, start_y = product_grid.start_x, product_grid.start_y
    if cube:
        start_x -= margin * spacing
        start_y += margin * spacing
        width += 2 * margin + 1
        length += 2 * margin + 1
    grid = GeoGrid(start_x, start_y, float(spacing), -float(spacing), width,
                   length, product_grid.epsg)
    return snap_geogrid(grid, spacing, spacing)


class RSLC:
    """A NISAR L1 RSLC granule: metadata eagerly, imagery on demand."""

    def __init__(self, path):
        if isinstance(path, h5py.File):
            # An already-open handle -- e.g. over an fsspec file for remote reads.
            self._file = path
            self.path = str(path.filename)
        else:
            self.path = str(path)
            self._file = h5py.File(self.path, "r")
        f = self._file
        if RSLC_PATH not in f:
            raise ValueError(f"{self.path} is not a NISAR RSLC (no {RSLC_PATH})")
        self._root = f[RSLC_PATH]
        ident = f[IDENT_PATH]
        self.look_side = _decode(ident["lookDirection"][()]).lower()
        self.pass_direction = _decode(ident["orbitPassDirection"][()])
        self.granule_id = _decode(ident["granuleId"][()]) if "granuleId" in ident else None
        self.frequencies = [
            _decode(v) for v in ident["listOfFrequencies"][()]
        ] if "listOfFrequencies" in ident else sorted(
            k[-1] for k in self._root["swaths"] if k.startswith("frequency"))
        swaths = self._root["swaths"]
        time_ds = swaths["zeroDopplerTime"]
        self.ref_epoch = radar.parse_epoch(time_ds.attrs["units"])
        self.zero_doppler_time = time_ds[()]
        self.time_spacing = float(swaths["zeroDopplerTimeSpacing"][()])

    # -- structure ------------------------------------------------------------
    def swath(self, frequency):
        return self._root[f"swaths/frequency{frequency}"]

    def polarizations(self, frequency):
        return [_decode(p) for p in self.swath(frequency)["listOfPolarizations"][()]]

    @cached_property
    def orbit(self):
        """The embedded orbit, referenced to the radar grid's epoch."""
        orbit = radar.Orbit.from_h5(self._root["metadata/orbit"])
        if orbit.reference_epoch != self.ref_epoch:
            orbit = orbit.with_reference_epoch(self.ref_epoch)
        return orbit

    def center_frequency(self, frequency):
        return float(self.swath(frequency)["processedCenterFrequency"][()])

    def radar_grid(self, frequency):
        """``RadarGridParameters(product, frequency)``.

        The range spacing is ``slantRange[1] - slantRange[0]``, **not** the
        ``slantRangeSpacing`` dataset (they differ in the eleventh digit), and
        the PRF is ``1 / zeroDopplerTimeSpacing`` -- both as ISCE3 builds it.
        """
        sw = self.swath(frequency)
        sr = sw["slantRange"]
        r0, r1 = sr[0], sr[1]
        return radar.RadarGrid(
            sensing_start=float(self.zero_doppler_time[0]),
            wavelength=radar.SPEED_OF_LIGHT / self.center_frequency(frequency),
            prf=1.0 / self.time_spacing,
            starting_range=float(r0),
            range_pixel_spacing=float(r1 - r0),
            look_side=self.look_side,
            length=int(self.zero_doppler_time.size),
            width=int(sr.shape[0]),
            ref_epoch=self.ref_epoch,
        )

    def doppler_centroid(self, frequency):
        """The native Doppler centroid LUT (Hz), bilinear, bounds-checked."""
        group = self._root[f"metadata/processingInformation/parameters/frequency{frequency}"]
        parent = self._root["metadata/processingInformation/parameters"]
        t = group["zeroDopplerTime"][()] if "zeroDopplerTime" in group else parent["zeroDopplerTime"][()]
        r = group["slantRange"][()] if "slantRange" in group else parent["slantRange"][()]
        return radar.LUT2d.from_coords(r, t, group["dopplerCentroid"][()])

    def valid_samples(self, frequency):
        """Per-subswath ``(lines, 2)`` arrays of ``[first, end)`` valid samples.

        Loaded exactly as ISCE3's ``Swath`` does: ``numberOfSubSwaths`` arrays,
        stopping at the first missing one. Half-open, despite the product's
        "first and last valid sample" description.
        """
        sw = self.swath(frequency)
        n = int(sw["numberOfSubSwaths"][()]) if "numberOfSubSwaths" in sw else 1
        arrays = []
        for i in range(1, n + 1):
            name = f"validSamplesSubSwath{i}"
            if name not in sw:
                break
            arrays.append(sw[name][()].astype(np.int64))
        return arrays

    def image(self, frequency, polarization):
        """The raw HDF5 dataset of one image layer."""
        return self.swath(frequency)[polarization]

    def reader(self, frequency, polarization):
        """A :class:`DirectChunkReader` for one layer, or the dataset itself."""
        dset = self.image(frequency, polarization)
        try:
            return DirectChunkReader(dset)
        except (ValueError, AttributeError, ImportError):
            return dset

    def data(self, frequency="A", polarization="HH"):
        """A lazy ``(time, slant_range)`` DataArray of one image layer."""
        import dask.array as da
        import xarray as xr

        reader = self.reader(frequency, polarization)
        chunks = getattr(self.image(frequency, polarization), "chunks", None) or (2048, 2048)
        arr = da.from_array(reader, chunks=(chunks[0] * 4, chunks[1] * 4),
                            lock=not isinstance(reader, DirectChunkReader))
        grid = self.radar_grid(frequency)
        return xr.DataArray(
            arr, dims=("time", "slant_range"),
            coords={"time": self.zero_doppler_time,
                    "slant_range": grid.slant_range(np.arange(grid.width))},
            name=polarization,
            attrs={"epoch": str(self.ref_epoch), "frequency": frequency},
        )

    # -- geometry -------------------------------------------------------------
    def footprint(self, frequency="A", heights=(-500.0, 9000.0), points_per_edge=11):
        """Lon/lat (degrees) perimeter of the swath at each constant height.

        ``getGeoPerimeter`` for each height with a zero-Doppler grid: returns an
        ``(n_heights, 4 * (points_per_edge - 1), 2)`` array of (lon, lat).
        """
        grid = self.radar_grid(frequency)
        return np.stack([
            np.degrees(radar.geo_perimeter(grid, self.orbit, h, points_per_edge)[:, :2])
            for h in heights])

    def utm_epsg(self, frequency="A"):
        """UTM zone of the swath centre (the NISAR L2 convention within 60 deg)."""
        from .geo import utm_epsg

        grid = self.radar_grid(frequency)
        llh, _ = radar.rdr2geo([grid.sensing_mid], [grid.mid_range], self.orbit,
                               height=0.0, wavelength=grid.wavelength,
                               look_side=grid.look_side, threshold=1e-5)
        lon, lat = np.degrees(llh[0, 0]), np.degrees(llh[0, 1])
        if lat > 60.0:
            return 3413
        if lat < -60.0:
            return 3031
        return utm_epsg(lon, lat)

    def default_geogrid(self, frequency, epsg=None, posting=None, snap=None):
        """ISCE3's default product grid for one frequency, optionally snapped.

        ``bbox2GeoGrid`` over the swath footprint at -500 m and 9000 m with the
        bandwidth's posting. ISCE3 does not snap it; ASF's products are made
        with an explicit frame box instead, whose corners (in every product we
        have checked) are whole multiples of 240 m. ``snap`` rounds the corners
        outward to such a multiple so the pixel centres land on the same lattice
        as ASF's grids.
        """
        from pyproj import Transformer

        epsg = int(epsg or self.utm_epsg(frequency))
        if posting is None:
            posting = self.default_posting(frequency)
        spacing_x, spacing_y = posting
        lonlat = self.footprint(frequency).reshape(-1, 2)
        tr = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
        x, y = tr.transform(lonlat[:, 0], lonlat[:, 1])
        x_min, x_max, y_min, y_max = np.min(x), np.max(x), np.min(y), np.max(y)
        grid = GeoGrid(float(x_min), float(y_max), float(spacing_x), -float(spacing_y),
                       int(np.ceil((x_max - x_min) / spacing_x)),
                       int(np.ceil((y_max - y_min) / spacing_y)), epsg)
        if snap:
            grid = snap_geogrid(grid, snap, snap)
        return grid

    def default_posting(self, frequency):
        bandwidth = float(self.swath(frequency)["processedRangeBandwidth"][()])
        mhz = int(round(bandwidth * 1e-6))
        if mhz not in POSTING_BY_BANDWIDTH:
            raise ValueError(f"no standard GSLC posting for a {mhz} MHz band; pass posting=")
        return POSTING_BY_BANDWIDTH[mhz]

    # -- GSLC -------------------------------------------------------------------
    def to_gslc(self, output, dem, **kwargs):
        """Geocode this RSLC into a NISAR-format GSLC file. Returns the path.

        See :func:`nisar_tools.gslc_geocode.geocode_rslc` for the parameters.
        """
        from .gslc_geocode import geocode_rslc

        return geocode_rslc(self, output, dem, **kwargs)

    # -- lifetime ---------------------------------------------------------------
    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __repr__(self):
        return (f"<RSLC {self.granule_id or self.path} {self.pass_direction} "
                f"look={self.look_side} freqs={self.frequencies}>")
