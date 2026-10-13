"""Dense pixel offsets between two RSLCs: NISAR's ROFF, ported from ISCE3 to numpy.

ISCE3 v0.25.16 makes a ROFF in four steps (``nisar.workflows.insar``):

1. ``rdr2geo``: every reference pixel onto the DEM (zero-Doppler; threshold 1e-7 m, 25 + 10
   iterations);
2. ``geo2rdr``: those targets into the secondary (zero-Doppler; 1e-8 s, 25 iterations) --
   per-pixel line and sample offsets, secondary index minus reference index;
3. ``resample_slc`` (coarse): the secondary resampled onto the reference grid by
   ``ResampSlc`` -- a 9x9 chip deramped by the secondary's native Doppler,
   ``Sinc2dInterpolator(8, 8192)``, the Doppler phase put back, 0 where invalid, no
   flattening;
4. ``offsets_product``: PyCuAmpcor per layer (:mod:`nisar_tools.ampcor`) on the reference and
   the coregistered secondary; then metres -- along-track by the ground-track velocity times
   the reference's azimuth time spacing, slant range by its range spacing.

So the offsets are what the geometry leaves: orbit and timing errors, deformation, the
ionosphere. Windows start at ``start + k*skip`` (top-left corners, ``start = margin +
min(half_search)``) and every layer is labelled at ``start + min(window)//2``
(``get_offset_radar_grid``) -- a layer with larger windows than the smallest is labelled off
its own centre by half the difference: **32 px in both axes for GOFF's 128 x 128 layer2**.
:func:`dense_offsets` keeps ISCE3's labels and records each layer's true centre offset.

Departures (none moves an offset beyond rounding; see the module tests and the validation
against a real ROFF):

- steps 1-2 are solved on a node lattice every ``geometry_step`` pixels and interpolated
  bicubically, not at every pixel;
- the resample chip is centred on ``trunc`` of the position instead of ``round`` and the
  Doppler is evaluated at the input position instead of at the output index on the
  secondary's grid (this is :func:`~nisar_tools._geocode_kernels.interpolate_slc`, shared with
  the GSLC geocoder) -- the same interpolant, different float32 rounding;
- the correlator runs in numpy, not cuFFT.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from . import radar
from ._geocode_kernels import interpolate_slc
from .ampcor import AmpcorParams, ampcor_windows

#: The layers of NISAR's GOFF/ROFF (``offsets_product`` in the PGE runconfig, v0.25.16).
GOFF_LAYERS = {
    "layer1": AmpcorParams(window_range=64, window_azimuth=64,
                           half_search_range=16, half_search_azimuth=16),
    "layer2": AmpcorParams(window_range=128, window_azimuth=128,
                           half_search_range=16, half_search_azimuth=16),
}

#: ``rdr2geo``/``geo2rdr`` settings of the InSAR PGE runconfig.
RDR2GEO = dict(threshold=1e-7, maxiter=25, extraiter=10)
GEO2RDR = dict(threshold=1e-8, maxiter=25)

ROFF_PATH = "science/LSAR/ROFF/swaths/frequency{freq}/pixelOffsets"
OFFSET_VARIABLES = ("alongTrackOffset", "slantRangeOffset", "alongTrackOffsetVariance",
                    "slantRangeOffsetVariance", "crossOffsetVariance", "snr",
                    "correlationSurfacePeak")


def _rslc(x):
    from .rslc import RSLC
    return x if isinstance(x, RSLC) else RSLC(x)


def _dem(x):
    """A DEM path, or anything with ``heights(lon_rad, lat_rad)`` (a DEMInterpolator)."""
    from .dem import DEMInterpolator
    return x if hasattr(x, "heights") else DEMInterpolator(x)


@dataclass(frozen=True)
class OffsetLattice:
    """The window lattice shared by every layer (``get_start_pixels`` + ``get_offsets_shape``)."""

    az_start: int
    rg_start: int
    skip_azimuth: int
    skip_range: int
    length: int
    width: int
    label_azimuth: int          # label = start + this (min window // 2)
    label_range: int

    @classmethod
    def for_grid(cls, lines, samples, layers, margin=250, skip_azimuth=15, skip_range=15,
                 start_pixel_azimuth=None, start_pixel_range=None,
                 gross_offset_azimuth=0, gross_offset_range=0):
        layers = list(layers.values()) if isinstance(layers, dict) else list(layers)
        margin = max(margin, abs(gross_offset_range), abs(gross_offset_azimuth))
        hs_az = min(L.half_search_azimuth for L in layers)
        hs_rg = min(L.half_search_range for L in layers)
        win_az = min(L.window_azimuth for L in layers)
        win_rg = min(L.window_range for L in layers)
        az0 = margin + hs_az if start_pixel_azimuth is None else int(start_pixel_azimuth)
        rg0 = margin + hs_rg if start_pixel_range is None else int(start_pixel_range)
        return cls(az0, rg0, int(skip_azimuth), int(skip_range),
                   (lines - (2 * margin + 2 * hs_az + win_az)) // skip_azimuth,
                   (samples - (2 * margin + 2 * hs_rg + win_rg)) // skip_range,
                   win_az // 2, win_rg // 2)

    def window_lines(self, rows):
        """Top-left reference line of the windows in lattice ``rows``."""
        return self.az_start + np.asarray(rows) * self.skip_azimuth

    def window_samples(self, cols):
        return self.rg_start + np.asarray(cols) * self.skip_range

    def label_lines(self, rows):
        """ISCE3's label: the line the offsets are assigned to."""
        return self.window_lines(rows) + self.label_azimuth

    def label_samples(self, cols):
        return self.window_samples(cols) + self.label_range


def geometry_offsets(reference, secondary, dem, lines, samples, frequency="A"):
    """Steps 1-2 at reference pixels ``lines`` x ``samples`` (1-D, any spacing).

    Returns ``(d_line, d_sample)``, each ``(len(lines), len(samples))``: the secondary's
    fractional line and sample of the target each reference pixel images, minus the
    reference pixel's own line and sample (``isce3::geometry::Geo2Rdr``'s ``azimuth.off`` /
    ``range.off``). NaN where either solve fails. Both orbits are the products' own.
    """
    reference, secondary, dem = _rslc(reference), _rslc(secondary), _dem(dem)
    rg = reference.radar_grid(frequency)
    sg = secondary.radar_grid(frequency)
    L, S = np.meshgrid(np.asarray(lines, float), np.asarray(samples, float), indexing="ij")
    t = rg.sensing_start + L.ravel() / rg.prf
    r = rg.starting_range + S.ravel() * rg.range_pixel_spacing
    llh, ok1 = radar.rdr2geo(t, r, reference.orbit, doppler_hz=0.0, wavelength=rg.wavelength,
                             look_side=rg.look_side, dem=dem.heights, **RDR2GEO)
    # ``Topo`` keeps every rdr2geo answer, converged or not: the DEM spline is float32, so
    # heights jitter at ~1e-5 m and a 1e-7 m threshold is often unreachable (22% of a hilly
    # tile) although the target sits on the range sphere to 1e-10 m.
    del ok1
    t2, r2, ok2 = radar.geo2rdr(llh, secondary.orbit, None, sg.wavelength, sg.look_side,
                                t0=t, **GEO2RDR)
    ok = np.isfinite(llh).all(axis=1) & ok2
    d_line = np.where(ok, (t2 - sg.sensing_start) * sg.prf - L.ravel(), np.nan)
    d_sample = np.where(ok, (r2 - sg.starting_range) / sg.range_pixel_spacing - S.ravel(), np.nan)
    return d_line.reshape(L.shape), d_sample.reshape(L.shape)


def ground_track_velocity(rslc, dem, t, r, frequency="A"):
    """``getGeolocationGrid``'s ground-track velocity (m/s) at radar ``(t, r)`` (1-D).

    Target on the DEM at zero Doppler; platform at the target's native-Doppler time;
    ``cos(alpha) |target| |v| / |platform|`` with ``alpha`` the geocentric angle between them
    (``writeVectorDerivedCubes``' cosine law). This is what ISCE3 multiplies along-track pixel
    offsets by (times the reference's azimuth time spacing).
    """
    rslc, dem = _rslc(rslc), _dem(dem)
    g = rslc.radar_grid(frequency)
    t = np.asarray(t, float).ravel()
    r = np.broadcast_to(np.asarray(r, float), t.shape).ravel()
    llh, _ = radar.rdr2geo(t, r, rslc.orbit, doppler_hz=0.0, wavelength=g.wavelength,
                           look_side=g.look_side, threshold=1e-8, maxiter=25, extraiter=10,
                           dem=dem.heights)
    dop = rslc.doppler_centroid(frequency)
    dop = radar.LUT2d(dop.x_start, dop.y_start, dop.dx, dop.dy, dop.data, dop.method,
                      bounds_error=False)
    tn, _, okn = radar.geo2rdr(llh, rslc.orbit, dop, g.wavelength, g.look_side, t0=t,
                               threshold=1e-8, maxiter=50, delta_range=10.0)
    sat, vel = rslc.orbit.interpolate(np.where(okn, tn, t))
    target = radar.lonlat_to_xyz(llh[:, 0], llh[:, 1], llh[:, 2])
    slant = np.linalg.norm(target - sat, axis=1)
    rt = np.linalg.norm(target, axis=1)
    rp = np.linalg.norm(sat, axis=1)
    cos_a = (rt * rt + rp * rp - slant * slant) / (2.0 * rt * rp)
    v = cos_a * rt * np.linalg.norm(vel, axis=1) / rp
    return np.where(np.isfinite(llh).all(axis=1) & okn, v, np.nan)


def _bicubic(nodes_l, nodes_s, values, lines, samples):
    from scipy.interpolate import RectBivariateSpline

    if not np.isfinite(values).all():
        raise ValueError("geometry offsets failed to converge at a lattice node inside the block")
    return RectBivariateSpline(nodes_l, nodes_s, values, kx=3, ky=3)(lines, samples)


def _node_axis(i0, i1, step):
    """Lattice nodes every ``step`` covering ``[i0, i1)`` with one node to spare each side."""
    a = (int(np.floor(i0 / step)) - 1) * step
    b = (int(np.ceil((i1 - 1) / step)) + 1) * step
    return np.arange(a, b + step, step, dtype=float)


def coarse_resample(secondary, reader, lines, samples, d_line, d_sample, frequency="A"):
    """Step 3 for the reference block ``lines`` x ``samples`` (contiguous 1-D index ranges).

    ``d_line``/``d_sample`` are the per-pixel offsets of that block. Returns the coregistered
    secondary block, complex64, 0 where the secondary cannot be interpolated (``ResampSlc``'s
    invalid value).
    """
    sg = secondary.radar_grid(frequency)
    dop = secondary.doppler_centroid(frequency)
    az = np.asarray(lines, float)[:, None] + d_line
    rg = np.asarray(samples, float)[None, :] + d_sample
    out = np.zeros(az.shape, np.complex64)
    fin = np.isfinite(az) & np.isfinite(rg)
    if not fin.any():
        return out
    l0 = max(int(np.floor(az[fin].min())) - 5, 0)
    l1 = min(int(np.ceil(az[fin].max())) + 6, sg.length)
    s0 = max(int(np.floor(rg[fin].min())) - 5, 0)
    s1 = min(int(np.ceil(rg[fin].max())) + 6, sg.width)
    if l1 - l0 < 9 or s1 - s0 < 9:
        return out
    block = np.asarray(reader[l0:l1, s0:s1], np.complex64)
    vals = interpolate_slc(block, l0, s0, az.ravel(), rg.ravel(), None, sg, dop, flatten=False)
    vals = vals.reshape(az.shape)
    good = np.isfinite(vals)
    out[good] = vals[good]
    return out


def _tile_offsets(ref, sec, ref_reader, sec_reader, dem, lattice, layers, rows, cols,
                  frequency, geometry_step, batch):
    """Everything for one tile of the lattice; returns a dict of ``(ni, nj)`` arrays."""
    ni, nj = len(rows), len(cols)
    lay = list(layers.values())
    W_a = max(L.window_azimuth for L in lay)
    W_r = max(L.window_range for L in lay)
    H_a = max(L.half_search_azimuth for L in lay)
    H_r = max(L.half_search_range for L in lay)
    wl = lattice.window_lines(rows)
    ws = lattice.window_samples(cols)
    L0, L1 = int(wl[0]) - H_a, int(wl[-1]) + W_a + H_a
    S0, S1 = int(ws[0]) - H_r, int(ws[-1]) + W_r + H_r
    lines = np.arange(L0, L1)
    samples = np.arange(S0, S1)

    # steps 1-2 on the node lattice, 3 on the block
    nl, ns = _node_axis(L0, L1, geometry_step), _node_axis(S0, S1, geometry_step)
    dl_n, ds_n = geometry_offsets(ref, sec, dem, nl, ns, frequency)
    d_line = _bicubic(nl, ns, dl_n, lines, samples)
    d_sample = _bicubic(nl, ns, ds_n, lines, samples)
    sec_block = coarse_resample(sec, sec_reader, lines, samples, d_line, d_sample, frequency)
    ref_block = np.asarray(ref_reader[L0:L1, S0:S1], np.complex64)
    if ref_block.shape != sec_block.shape:
        raise ValueError("reference block left the reference grid: check the lattice")

    out = {}
    ii, jj = np.meshgrid(np.arange(ni), np.arange(nj), indexing="ij")
    ii, jj = ii.ravel(), jj.ravel()
    for name, p in layers.items():
        wa, wr = p.window_azimuth, p.window_range
        ha, hr = p.half_search_azimuth, p.half_search_range
        res = {k: np.empty(ni * nj, np.float32) for k in ("down", "across", "snr", "peak")}
        res["cov"] = np.empty((ni * nj, 3), np.float32)
        for b0 in range(0, ni * nj, batch):
            b = slice(b0, min(b0 + batch, ni * nj))
            r0 = (wl[ii[b]] - L0)[:, None, None]
            c0 = (ws[jj[b]] - S0)[:, None, None]
            ra = np.arange(wa)[None, :, None]
            ca = np.arange(wr)[None, None, :]
            rw = ref_block[r0 + ra, c0 + ca]
            sa = np.arange(wa + 2 * ha)[None, :, None]
            sc = np.arange(wr + 2 * hr)[None, None, :]
            sw = sec_block[r0 - ha + sa, c0 - hr + sc]
            o = ampcor_windows(rw, sw, p)
            for k in res:
                res[k][b] = o[k]
        out[name] = {k: v.reshape((ni, nj) + v.shape[1:]) for k, v in res.items()}

    # the labels: geometry offsets there, and the metres conversion
    lab_l = lattice.label_lines(rows).astype(float)
    lab_s = lattice.label_samples(cols).astype(float)
    out["geometry_line"] = _bicubic(nl, ns, dl_n, lab_l, lab_s).astype(np.float32)
    out["geometry_sample"] = _bicubic(nl, ns, ds_n, lab_l, lab_s).astype(np.float32)
    g = ref.radar_grid(frequency)
    T, R = np.meshgrid(g.sensing_start + lab_l / g.prf,
                       g.starting_range + lab_s * g.range_pixel_spacing, indexing="ij")
    out["groundTrackVelocity"] = ground_track_velocity(ref, dem, T.ravel(), R.ravel(),
                                                       frequency).reshape(T.shape)
    return out


def dense_offsets(reference, secondary, dem, *, frequency="A", polarization="HH",
                  layers=None, margin=250, skip_azimuth=15, skip_range=15, rows=None,
                  cols=None, tile=(32, 128), geometry_step=8, workers=4, batch=256,
                  progress=None):
    """A ROFF's pixel-offset layers from two RSLCs (steps 1-4 of the module docstring).

    ``reference``/``secondary`` are RSLC paths or :class:`~nisar_tools.rslc.RSLC`; ``dem`` a
    raster path or :class:`~nisar_tools.dem.DEMInterpolator` (NISAR's DEM, as the PGE).
    ``layers`` defaults to :data:`GOFF_LAYERS`; the lattice to the PGE's (margin 250, skip 15,
    start ``margin + min(half_search)``). ``rows``/``cols`` select lattice indices (``range``
    or slice; default all) -- a subset gives exactly the same values as the full run.

    Tiles of ``tile`` = (rows, cols) windows run on ``workers`` threads; ``batch`` windows go
    through the correlator at once. ``progress(done, total)`` is called after each tile.

    Returns an :class:`xarray.Dataset` on ``(row, col)`` lattice indices with ISCE3's
    ``zeroDopplerTime``/``slantRange`` labels, the ROFF variables on ``(layer, row, col)``
    (metres; variances in m**2), the raw pixel offsets ``down``/``across`` (reference pixels),
    the ``groundTrackVelocity`` used, and the geometry offsets at the labels
    (``geometryLineOffset``/``geometrySampleOffset``, secondary pixels) -- add those to the
    pixel offsets for the total secondary-minus-reference offset.
    """
    import xarray as xr

    ref, sec, dem = _rslc(reference), _rslc(secondary), _dem(dem)
    layers = dict(GOFF_LAYERS if layers is None else layers)
    g = ref.radar_grid(frequency)
    lat = OffsetLattice.for_grid(g.length, g.width, layers, margin, skip_azimuth, skip_range)
    rows = np.arange(lat.length) if rows is None else np.asarray(
        range(*rows.indices(lat.length)) if isinstance(rows, slice) else rows)
    cols = np.arange(lat.width) if cols is None else np.asarray(
        range(*cols.indices(lat.width)) if isinstance(cols, slice) else cols)
    if rows.size == 0 or cols.size == 0:
        raise ValueError("dense_offsets: empty selection")
    if (np.diff(rows) != 1).any() or (np.diff(cols) != 1).any():
        raise ValueError("dense_offsets: rows and cols must be contiguous")
    ref_reader = ref.reader(frequency, polarization)
    sec_reader = sec.reader(frequency, polarization)

    tr, tc = tile
    tiles = [(rows[a:a + tr], cols[b:b + tc]) for a in range(0, rows.size, tr)
             for b in range(0, cols.size, tc)]
    nL, ni, nj = len(layers), rows.size, cols.size
    arr = {k: np.full((nL, ni, nj), np.nan, np.float32) for k in ("down", "across", "snr", "peak")}
    arr["cov"] = np.full((nL, ni, nj, 3), np.nan, np.float32)
    gl = np.full((ni, nj), np.nan, np.float32)
    gs = np.full((ni, nj), np.nan, np.float32)
    vg = np.full((ni, nj), np.nan)

    def run(t):
        return t, _tile_offsets(ref, sec, ref_reader, sec_reader, dem, lat, layers, t[0], t[1],
                                frequency, geometry_step, batch)

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        for (trows, tcols), res in ex.map(run, tiles):
            a = slice(trows[0] - rows[0], trows[-1] - rows[0] + 1)
            b = slice(tcols[0] - cols[0], tcols[-1] - cols[0] + 1)
            for k, name in enumerate(layers):
                for v in ("down", "across", "snr", "peak", "cov"):
                    arr[v][k, a, b] = res[name][v]
            gl[a, b], gs[a, b], vg[a, b] = (res["geometry_line"], res["geometry_sample"],
                                            res["groundTrackVelocity"])
            done += 1
            if progress is not None:
                progress(done, len(tiles))

    # metres, as offsets_product writes them: float32 data *= float64 factor
    dt_az = 1.0 / g.prf
    dr = g.range_pixel_spacing
    s_az = vg * dt_az
    along = (arr["down"] * s_az).astype(np.float32)
    srange = (arr["across"] * dr).astype(np.float32)
    var_a = (arr["cov"][..., 0] * s_az ** 2).astype(np.float32)
    var_r = (arr["cov"][..., 1] * dr ** 2).astype(np.float32)
    var_x = (arr["cov"][..., 2] * s_az * dr).astype(np.float32)
    lab_l = lat.label_lines(rows)
    lab_s = lat.label_samples(cols)
    dims = ("layer", "row", "col")
    names = list(layers)
    ds = xr.Dataset(
        {"alongTrackOffset": (dims, along), "slantRangeOffset": (dims, srange),
         "alongTrackOffsetVariance": (dims, var_a), "slantRangeOffsetVariance": (dims, var_r),
         "crossOffsetVariance": (dims, var_x), "snr": (dims, arr["snr"]),
         "correlationSurfacePeak": (dims, arr["peak"]),
         "down": (dims, arr["down"]), "across": (dims, arr["across"]),
         "groundTrackVelocity": (("row", "col"), vg),
         "geometryLineOffset": (("row", "col"), gl),
         "geometrySampleOffset": (("row", "col"), gs)},
        coords={"layer": names, "row": rows, "col": cols,
                "line": ("row", lab_l), "sample": ("col", lab_s),
                "zeroDopplerTime": ("row", g.sensing_start + lab_l / g.prf),
                "slantRange": ("col", g.starting_range + lab_s * g.range_pixel_spacing),
                "centre_shift_azimuth": ("layer", [p.window_azimuth // 2 - lat.label_azimuth
                                                   for p in layers.values()]),
                "centre_shift_range": ("layer", [p.window_range // 2 - lat.label_range
                                                 for p in layers.values()])},
        attrs={"reference": ref.granule_id or ref.path, "secondary": sec.granule_id or sec.path,
               "frequency": frequency, "polarization": polarization, "margin": margin,
               "skip_azimuth": skip_azimuth, "skip_range": skip_range,
               "az_start": lat.az_start, "rg_start": lat.rg_start,
               "lattice_shape": (lat.length, lat.width), "geometry_step": geometry_step,
               "zeroDopplerTime_epoch": str(g.ref_epoch),
               "layers": repr(layers),
               "convention": "secondary minus reference, after geometry coregistration "
                             "(ISCE3 ROFF); along-track m = down * groundTrackVelocity / prf"})
    return ds


def open_roff(path, rows=None, cols=None, frequency="A", polarization="HH"):
    """A NISAR ROFF's pixel offsets in :func:`dense_offsets`' layout (``rows``/``cols`` slices)."""
    import h5py
    import xarray as xr

    rows = slice(None) if rows is None else rows
    cols = slice(None) if cols is None else cols
    with h5py.File(path, "r") as f:
        po = f[ROFF_PATH.format(freq=frequency)]
        layers = sorted(k for k in po[polarization] if k.startswith("layer"))
        data = {v: (("layer", "row", "col"),
                    np.stack([po[f"{polarization}/{L}/{v}"][rows, cols] for L in layers]))
                for v in OFFSET_VARIABLES}
        t = po["zeroDopplerTime"][rows]
        r = po["slantRange"][cols]
        n_t, n_r = po["zeroDopplerTime"].shape[0], po["slantRange"].shape[0]
        ri = np.arange(n_t)[rows]
        ci = np.arange(n_r)[cols]
        epoch = po["zeroDopplerTime"].attrs.get("units", b"")
        # each layer's window size, for its centre shift from ISCE3's common label
        par = f"science/LSAR/ROFF/metadata/processingInformation/parameters/pixelOffsets/frequency{frequency}"
        win_a, win_r = [], []
        for L in layers:
            g = f.get(f"{par}/{L}")
            win_a.append(int(g["alongTrackWindowSize"][()]) if g is not None else 0)
            win_r.append(int(g["slantRangeWindowSize"][()]) if g is not None else 0)
    epoch = epoch.decode() if isinstance(epoch, bytes) else str(epoch)
    coords = {"layer": layers, "row": ri, "col": ci, "zeroDopplerTime": ("row", t),
              "slantRange": ("col", r)}
    if all(win_a) and all(win_r):
        coords["centre_shift_azimuth"] = ("layer", [w // 2 - min(win_a) // 2 for w in win_a])
        coords["centre_shift_range"] = ("layer", [w // 2 - min(win_r) // 2 for w in win_r])
    return xr.Dataset(data, coords=coords,
                      attrs={"source": str(path), "zeroDopplerTime_units": epoch})


def goff_geogrid(path, frequency="A"):
    """``(x, y, epsg)`` pixel-centre coordinates of a NISAR GOFF, to geocode onto the same grid."""
    import h5py

    with h5py.File(path, "r") as f:
        g = f[f"science/LSAR/GOFF/grids/frequency{frequency}/pixelOffsets"]
        return g["xCoordinates"][()], g["yCoordinates"][()], int(g["projection"][()])


def geocode_offsets(ds, reference, dem, x, y, epsg, *, variables=OFFSET_VARIABLES,
                    window_centres=False, frequency="A", block_rows=256):
    """ROFF -> GOFF (``geocode_insar``): each map pixel bilinear from the offsets lattice.

    ``ds`` is :func:`dense_offsets` output or :func:`open_roff` (same reference). ``x``/``y``
    are 1-D pixel-centre coordinates in ``epsg`` (:func:`goff_geogrid` reads a GOFF's).
    Each pixel: DEM height (biquintic), zero-Doppler ``geo2rdr`` on the reference orbit,
    fractional lattice row and column from the labels, ``BilinearInterpolator`` -- NaN
    outside the lattice (or the subset ``ds`` holds) and wherever a neighbour is NaN.

    ``window_centres=True`` places each layer at its windows' true centre instead of ISCE3's
    label (which is off by ``centre_shift_*`` pixels for layers larger than the smallest:
    32 px for GOFF's layer2). Returns an :class:`xarray.Dataset` on ``(layer, y, x)``.
    """
    import xarray as xr
    from pyproj import Transformer

    reference, dem = _rslc(reference), _dem(dem)
    g = reference.radar_grid(frequency)
    t_lab = np.asarray(ds["zeroDopplerTime"].values, float)
    r_lab = np.asarray(ds["slantRange"].values, float)
    dt = (t_lab[-1] - t_lab[0]) / (t_lab.size - 1)
    dr = (r_lab[-1] - r_lab[0]) / (r_lab.size - 1)
    layers = list(ds["layer"].values)
    shift_a = (ds["centre_shift_azimuth"].values if "centre_shift_azimuth" in ds.coords
               else np.zeros(len(layers)))
    shift_r = (ds["centre_shift_range"].values if "centre_shift_range" in ds.coords
               else np.zeros(len(layers)))
    to_ll = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    out = {v: np.full((len(layers), y.size, x.size), np.nan, np.float32) for v in variables}
    data = {v: np.asarray(ds[v].values, np.float32) for v in variables}
    for a in range(0, y.size, block_rows):
        yy = y[a:a + block_rows]
        X, Y = np.meshgrid(x, yy)
        lon, lat = to_ll.transform(X.ravel(), Y.ravel())
        lon, lat = np.deg2rad(lon), np.deg2rad(lat)
        h = dem.heights(lon, lat)
        t, r, ok = radar.geo2rdr(np.stack([lon, lat, h], axis=1), reference.orbit, None,
                                 g.wavelength, g.look_side, t0=g.sensing_mid,
                                 threshold=1e-8, maxiter=50)
        for k in range(len(layers)):
            sa = shift_a[k] / g.prf if window_centres else 0.0
            sr = shift_r[k] * g.range_pixel_spacing if window_centres else 0.0
            fr = (t - (t_lab[0] + sa)) / dt
            fc = (r - (r_lab[0] + sr)) / dr
            keep = ok & (fr >= 0) & (fc >= 0) & (fr <= t_lab.size - 1) & (fc <= r_lab.size - 1)
            idx = np.flatnonzero(keep)
            if idx.size == 0:
                continue
            for v in variables:
                vals = radar.bilinear_interp(data[v][k], fc[idx], fr[idx])
                o = out[v][k, a:a + block_rows].reshape(-1)
                o[idx] = vals
                out[v][k, a:a + block_rows] = o.reshape(yy.size, x.size)
    dims = ("layer", "y", "x")
    return xr.Dataset({v: (dims, out[v]) for v in variables},
                      coords={"layer": layers, "x": x, "y": y},
                      attrs={"epsg": int(epsg), "window_centres": bool(window_centres),
                             **{k: v for k, v in ds.attrs.items() if isinstance(v, (str, int, float))}})


def offset_los_stacks(ds, reference_gslc, reference, dem, x, y, epsg, *, layer="layer1",
                      window_centres=False, frequency="A", geocoded=None):
    """A ROFF as two LOS-style stacks a slip inversion reads like InSAR: ``{"range", "along"}``.

    Each is an :class:`~nisar_tools.los.LOSStack` whose ``los`` (metres, ``(pair, y, x)``) is
    the ground displacement projected on that stack's own unit vector, ``los_east`` /
    ``los_north`` / ``los_up``:

    - ``"range"``: ``-slantRangeOffset`` on the target->sensor LOS unit vector. A slant-range
      offset is secondary minus reference range, + away from the sensor; this package's LOS is
      + toward it, so it is the same quantity as an interferogram's LOS, with no 2 pi ambiguity;
    - ``"along"``: ``+alongTrackOffset`` on the along-track unit vector, which is horizontal and
      points along the flight direction (+ = the ground moved that way).

    Both vectors come from ``reference_gslc``'s geometry cube (the ROFF's reference date, the
    same cube :class:`~nisar_tools.los.LOSStack` uses), sampled at ``dem``'s heights.
    ``reference`` is that date's RSLC and ``dem`` a GeoTIFF path; both geocode the offsets
    (:func:`geocode_offsets`, or pass its output as ``geocoded``). Each stack also carries
    ``variance`` (m**2, the correlator's), ``snr`` and ``correlationSurfacePeak``, for masking
    and weighting -- per-window sigmas are ~0.2 m in range and ~0.3 m along-track on D141.

    Signs checked on D141 Panama (left-lateral, the north block moving WNW): the north-minus-
    south step 0.5-1.5 km from the trace is -4.29 m in slant range (toward a sensor looking ESE)
    and -1.50 m along-track (against the SSW flight direction). The along-track vector is
    orthogonal to the LOS to 0.018 there, as zero-Doppler geometry requires.
    """
    import xarray as xr

    from . import geometry
    from .los import LOSStack

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if geocoded is None:
        variables = ("alongTrackOffset", "slantRangeOffset", "alongTrackOffsetVariance",
                     "slantRangeOffsetVariance", "snr", "correlationSurfacePeak")
        geocoded = geocode_offsets(ds.sel(layer=[layer]), reference, dem, x, y, epsg,
                                   variables=variables, window_centres=window_centres,
                                   frequency=frequency)
    geo = geocoded.sel(layer=layer)
    heights = geometry.dem_heights_on_grid(dem, x, y, epsg)
    cube = geometry.read_geometry_cube(reference_gslc, frequency, along_track=True)
    look = geometry.sample_look_geometry(cube, x, y, epsg, heights)
    attrs = {"epsg": int(epsg), "sign": 1, "wavelength": float(cube.attrs["wavelength"]),
             "look_direction": cube.attrs.get("look_direction"), "frequency": frequency,
             "layer": str(layer), "source": "ROFF pixel offsets",
             **{k: v for k, v in ds.attrs.items() if k in ("reference", "secondary")}}
    stacks = {}
    for name, value, var, vector, extra in (
            ("range", -geo["slantRangeOffset"], "slantRangeOffsetVariance",
             ("los_east", "los_north", "los_up"), ("incidence_angle", "look_angle")),
            ("along", geo["alongTrackOffset"], "alongTrackOffsetVariance",
             ("along_east", "along_north", "along_up"), ())):
        out = xr.Dataset(
            {"los": (("pair", "y", "x"), np.asarray(value.values, np.float32)[None]),
             **{f"los_{c}": (("y", "x"), np.asarray(look[v].values, np.float32))
                for c, v in zip(("east", "north", "up"), vector)},
             **{k: (("y", "x"), np.asarray(look[k].values, np.float32)) for k in extra},
             "height": (("y", "x"), np.asarray(look["height"].values, np.float32)),
             "variance": (("y", "x"), np.asarray(geo[var].values, np.float32)),
             "snr": (("y", "x"), np.asarray(geo["snr"].values, np.float32)),
             "correlationSurfacePeak": (("y", "x"),
                                        np.asarray(geo["correlationSurfacePeak"].values, np.float32))},
            coords={"pair": [0], "y": y, "x": x})
        out = out.rio.write_crs(f"EPSG:{int(epsg)}")
        out.attrs.update(attrs, kind="range_offset" if name == "range" else "along_track_offset",
                         convention=("-slantRangeOffset: + toward the sensor" if name == "range"
                                     else "+alongTrackOffset: + along the flight direction"))
        stacks[name] = LOSStack(out)
    return stacks
