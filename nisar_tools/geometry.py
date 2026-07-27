"""Radar-geometry helpers: per-pixel look angles from a GSLC's built-in cube,
plus the phase -> line-of-sight-displacement conversion.

A NISAR GSLC granule embeds an ISCE3-computed geometry cube under
``metadata/radarGrid``: the incidence angle and the target->sensor line-of-sight
(LOS) unit vector (East/North components) tabulated on a coarse map grid at a
stack of reference heights above the WGS84 ellipsoid. Per output pixel we
trilinearly interpolate that cube in ``(x, y, terrain-height)`` -- the terrain
height from a user-supplied DEM -- to get the incidence angle and the full ENU
LOS unit vector. The cube already encodes the zero-Doppler geometry (from the
product's own orbit), so the orbit ephemeris is not needed for this path.

Sign convention for the phase conversion: LOS displacement is **positive toward
the sensor** (range decrease), ``d = +(lambda / 4pi) * unwrapped_phase``, under
this package's ``ref * conj(sec)`` interferogram convention. Pass ``sign=-1`` to
:func:`phase_to_los` if your fringe sense is inverted.
"""

import os
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr
from pyproj import Transformer
from rasterio.warp import Resampling
from scipy.interpolate import RegularGridInterpolator

SPEED_OF_LIGHT = 299_792_458.0  # m/s

_RADAR_GRID = "science/LSAR/{product}/metadata/radarGrid"
_FREQ_GRID = "science/LSAR/{product}/grids/frequency{f}"
_LOOK_DIR = "science/LSAR/identification/lookDirection"

_GEOM_VARS = ("incidence_angle", "look_angle", "los_east", "los_north")
# HDF5 cube dataset names for each output variable.
#
# ``look_angle`` is the product's ``elevationAngle``, which it defines as "the
# angle between the LOS vector and the normal to the ellipsoid at the sensor" --
# i.e. the off-nadir look angle, measured at the spacecraft. It is always
# smaller than the incidence angle (measured at the target, against the local
# vertical) by the Earth-curvature term: sin(look) = Re/(Re+h) * sin(incidence).
_CUBE_DATASETS = {
    "incidence_angle": "incidenceAngle",
    "look_angle": "elevationAngle",
    "los_east": "losUnitVectorX",
    "los_north": "losUnitVectorY",
}


def _decode(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def radar_wavelength(path, frequency="A", product="GSLC"):
    """Radar wavelength (m) = c / centerFrequency, read from a NISAR granule.

    ``product`` selects the L2 product group holding the ``grids`` tree:
    ``"GSLC"`` for a GSLC granule, ``"GUNW"`` for a NASA GUNW, which stores the
    same ``centerFrequency`` under its own group.
    """
    grid = _FREQ_GRID.format(product=product, f=frequency)
    with h5py.File(str(path), "r") as f:
        cf = float(f[grid + "/centerFrequency"][()])
    return SPEED_OF_LIGHT / cf


def read_geometry_cube(path, frequency="A", product="GSLC"):
    """Load a NISAR ``metadata/radarGrid`` geometry cube as an xarray Dataset.

    Returns dims ``(height, y, x)`` with ``incidence_angle`` (degrees) and the
    ``los_east`` / ``los_north`` LOS-unit-vector components; attrs carry the
    cube's ``epsg``, the ``wavelength``, and the ``look_direction``.

    ``product`` picks the product group: a ``"GSLC"`` granule or a ``"GUNW"``,
    which embeds a cube of the same layout (the two differ only in the number of
    reference heights, read here from the file, not assumed).
    """
    with h5py.File(str(path), "r") as f:
        rg = f[_RADAR_GRID.format(product=product)]
        data = {
            out: (("height", "y", "x"), rg[name][()])
            for out, name in _CUBE_DATASETS.items()
        }
        height = rg["heightAboveEllipsoid"][()].astype(float)
        y = rg["yCoordinates"][()].astype(float)
        x = rg["xCoordinates"][()].astype(float)
        proj = rg["projection"]
        epsg = int(proj.attrs.get("epsg_code", proj[()]))
        look = _decode(f[_LOOK_DIR][()]) if _LOOK_DIR in f else None

    ds = xr.Dataset(data, coords={"height": height, "y": y, "x": x})
    ds.attrs.update(
        epsg=epsg,
        wavelength=radar_wavelength(path, frequency, product),
        look_direction=look,
        frequency=frequency,
    )
    return ds


def _open_dem(dem):
    """Return a CRS-aware 2D DataArray for a DEM given a path or a DataArray."""
    if isinstance(dem, xr.DataArray):
        da = dem
    else:
        da = rioxarray.open_rasterio(str(dem), masked=True)
    if "band" in da.dims:
        da = da.squeeze("band", drop=True)
    if da.rio.crs is None:
        raise ValueError("DEM has no CRS; cannot reproject onto the output grid.")
    return da


def dem_heights_on_grid(dem, x, y, epsg):
    """Sample a DEM onto the ``(x, y)`` output grid. Returns a 2D height array.

    ``dem`` is a GeoTIFF path or a CRS-aware DataArray. It is reprojected and
    bilinearly resampled onto the output grid; pixels with no DEM coverage come
    back ``0.0`` (ellipsoid height). ``None`` yields all-zero heights (sea-level
    geometry), a reasonable fallback when no DEM is supplied.
    """
    ny, nx = len(y), len(x)
    if dem is None:
        return np.zeros((ny, nx), np.float32)

    da = _open_dem(dem)
    template = (
        xr.DataArray(
            np.zeros((ny, nx), np.float32),
            coords={"y": np.asarray(y), "x": np.asarray(x)},
            dims=("y", "x"),
        )
        .rio.write_crs(f"EPSG:{int(epsg)}")
        .rio.write_nodata(np.nan)
    )
    matched = da.rio.reproject_match(template, resampling=Resampling.bilinear)
    h = np.asarray(matched.values, dtype=np.float32)
    return np.where(np.isfinite(h), h, 0.0)


def sample_look_geometry(cube, x, y, epsg, height=None, workers=None):
    """Trilinearly interpolate a geometry ``cube`` onto the ``(x, y)`` grid.

    ``height`` is a scalar or a 2D ``(ny, nx)`` array of terrain heights above
    the ellipsoid (default 0). Returns a CRS-aware Dataset on dims ``(y, x)``
    with ``incidence_angle`` (deg), the ENU LOS unit vector ``los_east`` /
    ``los_north`` / ``los_up``, and the sampled ``height``. ``los_up`` is
    reconstructed as ``sqrt(1 - east^2 - north^2)`` (the LOS points up toward the
    sensor), which equals ``cos(incidence)``.

    Sampled a band of rows at a time on a thread pool rather than the whole grid
    at once. The interpolation is pointwise in ``(x, y, height)``, so banding
    cannot change the answer -- the result is bit-identical -- but it turns the
    largest single-threaded memory spike in the pipeline into a bounded one. The
    whole-grid version allocated two float64 meshgrids and an ``(ny*nx, 3)``
    float64 point list, then ran four ``RegularGridInterpolator`` calls over them
    back to back. Measured on a 3000x3000 grid: peak **202 -> 71 bytes/pixel**
    (1818 -> 636 MB), and 2.1x faster on ten cores, since both pyproj's transform
    and the interpolator release the GIL. ``workers=1`` runs the bands serially;
    the banding alone still bounds the memory.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ny, nx = len(y), len(x)
    height = np.broadcast_to(
        np.zeros(1) if height is None else np.asarray(height, float), (ny, nx)
    )

    cube_epsg = int(cube.attrs["epsg"])
    transformer = None
    if int(epsg) != cube_epsg:
        transformer = Transformer.from_crs(
            f"EPSG:{int(epsg)}", f"EPSG:{cube_epsg}", always_xy=True
        )

    # RegularGridInterpolator needs strictly-increasing axes; the cube's y runs
    # north-down, so sort every axis and reorder the values to match.
    ch, cy, cx = cube["height"].values, cube["y"].values, cube["x"].values
    hi, yi, xi = np.argsort(ch), np.argsort(cy), np.argsort(cx)
    axes = (ch[hi], cy[yi], cx[xi])
    h_lo, h_hi = axes[0].min(), axes[0].max()

    # Built once and shared: an interpolator only holds references to the cube's
    # (small) values, and its __call__ allocates nothing shared.
    interps = {
        var: RegularGridInterpolator(
            axes, cube[var].values[np.ix_(hi, yi, xi)],
            bounds_error=False, fill_value=np.nan,
        )
        for var in _GEOM_VARS
    }

    out = {var: np.empty((ny, nx), np.float32) for var in _GEOM_VARS}
    out["los_up"] = np.empty((ny, nx), np.float32)

    def _band(rows):
        xo, yo = np.meshgrid(x, y[rows])
        if transformer is not None:
            xc, yc = transformer.transform(xo, yo)
        else:
            xc, yc = xo, yo
        hcl = np.clip(height[rows], h_lo, h_hi)  # avoid extrapolation
        pts = np.stack([hcl.ravel(), yc.ravel(), xc.ravel()], axis=-1)
        shape = (rows.stop - rows.start, nx)

        sampled = {}
        for var, interp in interps.items():
            sampled[var] = interp(pts).reshape(shape)
            out[var][rows] = sampled[var]
        # From the float64 samples, as the whole-grid version did -- deriving it
        # from the float32 output would round twice.
        out["los_up"][rows] = np.sqrt(np.clip(
            1.0 - sampled["los_east"] ** 2 - sampled["los_north"] ** 2, 0.0, 1.0
        ))

    if workers is None:
        workers = os.cpu_count() or 1
    band = max(1, ny // max(1, 4 * workers))
    bands = [slice(r0, min(r0 + band, ny)) for r0 in range(0, ny, band)]
    if workers <= 1 or len(bands) == 1:
        for rows in bands:
            _band(rows)
    else:
        with ThreadPoolExecutor(workers, thread_name_prefix="geom") as pool:
            list(pool.map(_band, bands))

    ds = xr.Dataset(
        {
            "incidence_angle": (("y", "x"), out["incidence_angle"]),
            "look_angle": (("y", "x"), out["look_angle"]),
            "los_east": (("y", "x"), out["los_east"]),
            "los_north": (("y", "x"), out["los_north"]),
            "los_up": (("y", "x"), out["los_up"]),
            "height": (("y", "x"), np.asarray(height, np.float32)),
        },
        coords={"y": y, "x": x},
    ).rio.write_crs(f"EPSG:{int(epsg)}")
    ds.attrs["epsg"] = int(epsg)
    return ds


def phase_to_los(unwrapped, wavelength, sign=1):
    """Convert unwrapped phase (radians) to LOS displacement (metres).

    ``d = sign * (wavelength / 4pi) * phase``. The default ``sign=+1`` makes
    displacement positive toward the sensor under this package's
    ``ref * conj(sec)`` interferogram convention. Accepts numpy, dask, or
    xarray input and preserves its type (staying lazy for dask/xarray).
    """
    return sign * (wavelength / (4.0 * np.pi)) * unwrapped
