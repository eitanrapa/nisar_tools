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

import h5py
import numpy as np
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr
from pyproj import Transformer
from rasterio.warp import Resampling
from scipy.interpolate import RegularGridInterpolator

SPEED_OF_LIGHT = 299_792_458.0  # m/s

_RADAR_GRID = "science/LSAR/GSLC/metadata/radarGrid"
_FREQ_GRID = "science/LSAR/GSLC/grids/frequency{f}"
_LOOK_DIR = "science/LSAR/identification/lookDirection"

_GEOM_VARS = ("incidence_angle", "los_east", "los_north")
# HDF5 cube dataset names for each output variable.
_CUBE_DATASETS = {
    "incidence_angle": "incidenceAngle",
    "los_east": "losUnitVectorX",
    "los_north": "losUnitVectorY",
}


def _decode(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def radar_wavelength(gslc_path, frequency="A"):
    """Radar wavelength (m) = c / centerFrequency, read from the GSLC granule."""
    with h5py.File(str(gslc_path), "r") as f:
        cf = float(f[_FREQ_GRID.format(f=frequency) + "/centerFrequency"][()])
    return SPEED_OF_LIGHT / cf


def read_geometry_cube(gslc_path, frequency="A"):
    """Load a GSLC's ``metadata/radarGrid`` geometry cube as an xarray Dataset.

    Returns dims ``(height, y, x)`` with ``incidence_angle`` (degrees) and the
    ``los_east`` / ``los_north`` LOS-unit-vector components; attrs carry the
    cube's ``epsg``, the ``wavelength``, and the ``look_direction``.
    """
    with h5py.File(str(gslc_path), "r") as f:
        rg = f[_RADAR_GRID]
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
        wavelength=radar_wavelength(gslc_path, frequency),
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


def sample_look_geometry(cube, x, y, epsg, height=None):
    """Trilinearly interpolate a geometry ``cube`` onto the ``(x, y)`` grid.

    ``height`` is a scalar or a 2D ``(ny, nx)`` array of terrain heights above
    the ellipsoid (default 0). Returns a CRS-aware Dataset on dims ``(y, x)``
    with ``incidence_angle`` (deg), the ENU LOS unit vector ``los_east`` /
    ``los_north`` / ``los_up``, and the sampled ``height``. ``los_up`` is
    reconstructed as ``sqrt(1 - east^2 - north^2)`` (the LOS points up toward the
    sensor), which equals ``cos(incidence)``.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ny, nx = len(y), len(x)
    height = np.broadcast_to(
        np.zeros(1) if height is None else np.asarray(height, float), (ny, nx)
    )

    cube_epsg = int(cube.attrs["epsg"])
    xo, yo = np.meshgrid(x, y)
    if int(epsg) != cube_epsg:
        tr = Transformer.from_crs(
            f"EPSG:{int(epsg)}", f"EPSG:{cube_epsg}", always_xy=True
        )
        xc, yc = tr.transform(xo, yo)
    else:
        xc, yc = xo, yo

    # RegularGridInterpolator needs strictly-increasing axes; the cube's y runs
    # north-down, so sort every axis and reorder the values to match.
    ch, cy, cx = cube["height"].values, cube["y"].values, cube["x"].values
    hi, yi, xi = np.argsort(ch), np.argsort(cy), np.argsort(cx)
    axes = (ch[hi], cy[yi], cx[xi])

    hcl = np.clip(height, axes[0].min(), axes[0].max())  # avoid extrapolation
    pts = np.stack([hcl.ravel(), yc.ravel(), xc.ravel()], axis=-1)

    out = {}
    for var in _GEOM_VARS:
        vals = cube[var].values[np.ix_(hi, yi, xi)]
        interp = RegularGridInterpolator(
            axes, vals, bounds_error=False, fill_value=np.nan
        )
        out[var] = interp(pts).reshape(ny, nx)

    up = np.sqrt(np.clip(1.0 - out["los_east"] ** 2 - out["los_north"] ** 2, 0.0, 1.0))

    ds = xr.Dataset(
        {
            "incidence_angle": (("y", "x"), out["incidence_angle"].astype(np.float32)),
            "los_east": (("y", "x"), out["los_east"].astype(np.float32)),
            "los_north": (("y", "x"), out["los_north"].astype(np.float32)),
            "los_up": (("y", "x"), up.astype(np.float32)),
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
