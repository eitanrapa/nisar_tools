"""Geographic helpers: bbox transforms, lat/lon reprojection, grid warping.

Kept free of any class so they can be reused by GSLC, the stacks, and the
plotting layer. Reprojection is intentionally eager (rioxarray loads its
input), so it must only ever be handed a single small 2D slice; grid warping
is likewise per-frame and is wrapped in dask tasks by the callers.
"""

import warnings

import numpy as np
import rioxarray  # noqa: F401  (registers the .rio accessor on xarray objects)
import xarray as xr
from affine import Affine
from pyproj import Transformer
from rasterio.errors import NotGeoreferencedWarning
from rasterio.warp import Resampling, reproject, transform_bounds

# rasterio's reproject() warns about bare numpy arrays even when explicit
# src/dst transforms are supplied, and warnings.catch_warnings() is not
# thread-safe under dask's threaded scheduler, so the spurious warning is
# silenced once here, narrowly by message.
warnings.filterwarnings(
    "ignore",
    category=NotGeoreferencedWarning,
    message="Dataset has no geotransform",
)


def bbox_to_native(lon_min, lon_max, lat_min, lat_max, epsg_code):
    """Transform a lon/lat bounding box to native-CRS x/y bounds.

    Returns ``(x_min, x_max, y_min, y_max)`` in the projection given by
    ``epsg_code``. All four corners are transformed (not just two) so the
    bounds are correct even when the box edges are not axis-aligned in the
    native projection.
    """
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg_code}", always_xy=True)

    corners_lon = [lon_min, lon_max, lon_max, lon_min]
    corners_lat = [lat_min, lat_min, lat_max, lat_max]
    x_corners, y_corners = transformer.transform(corners_lon, corners_lat)

    return min(x_corners), max(x_corners), min(y_corners), max(y_corners)


def native_bbox_to_lonlat(x_min, x_max, y_min, y_max, epsg_code):
    """Transform a native-CRS bounding box to lon/lat ``(lon_min, lon_max,
    lat_min, lat_max)``. Edge-densified, so the returned box covers the whole
    footprint even where the grid is rotated relative to lon/lat."""
    return transform_native_bbox(x_min, x_max, y_min, y_max, epsg_code, 4326)


def transform_native_bbox(x_min, x_max, y_min, y_max, src_epsg, dst_epsg):
    """Transform a native-CRS bbox into another projected CRS.

    Edge-densified (via :func:`rasterio.warp.transform_bounds`), so the
    returned bounds cover the whole warped footprint even though a projected
    rectangle's edges curve in the target CRS. Returns
    ``(x_min, x_max, y_min, y_max)``.
    """
    left, bottom, right, top = transform_bounds(
        f"EPSG:{src_epsg}", f"EPSG:{dst_epsg}", x_min, y_min, x_max, y_max
    )
    return left, right, bottom, top


def grid_transform(x_coords, y_coords):
    """Affine geotransform for a regular grid given its pixel-center coords."""
    dx = float(x_coords[1] - x_coords[0])
    dy = float(y_coords[1] - y_coords[0])
    return Affine(
        dx, 0.0, float(x_coords[0]) - dx / 2,
        0.0, dy, float(y_coords[0]) - dy / 2,
    )


def resampling_from_name(name):
    """Resolve a rasterio :class:`Resampling` mode by name (e.g. "bilinear")."""
    try:
        return Resampling[name]
    except KeyError:
        valid = ", ".join(r.name for r in Resampling)
        raise ValueError(
            f"Unknown resampling {name!r}; expected one of: {valid}"
        ) from None


def warp_to_grid(arr, src_transform, src_epsg, dst_transform, dst_epsg,
                 dst_shape, resampling="bilinear"):
    """Warp one 2D array between projected grids. Eager, whole-frame.

    Complex input is resampled on its real and imaginary parts separately
    (for linear kernels this is exactly complex-valued interpolation) and
    recombined as complex64; use ``resampling="nearest"`` to preserve exact
    sample values. Pixels with no source coverage come back NaN.
    """
    resampling = resampling_from_name(resampling)

    def _warp_band(band):
        dst = np.full(dst_shape, np.nan, dtype=np.float32)
        reproject(
            np.ascontiguousarray(band, dtype=np.float32),
            dst,
            src_transform=src_transform,
            src_crs=f"EPSG:{src_epsg}",
            dst_transform=dst_transform,
            dst_crs=f"EPSG:{dst_epsg}",
            src_nodata=np.nan,
            dst_nodata=np.nan,
            resampling=resampling,
        )
        return dst

    if np.iscomplexobj(arr):
        return (_warp_band(arr.real) + 1j * _warp_band(arr.imag)).astype(
            np.complex64
        )
    return _warp_band(arr)


def project_to_latlon(data, x_coords=None, y_coords=None, epsg_code=None):
    """Reproject a 2D grid from its native CRS to WGS84 lon/lat.

    Accepts either a georeferenced :class:`xarray.DataArray` (with a written
    CRS) or a raw 2D array plus ``x_coords``/``y_coords``/``epsg_code``.
    Returns a lon/lat :class:`xarray.DataArray`. Eager: computes its input.
    """
    if isinstance(data, xr.DataArray) and data.rio.crs is not None:
        da = data
    else:
        if x_coords is None or y_coords is None or epsg_code is None:
            raise ValueError(
                "Provide a CRS-aware DataArray, or data with x_coords, "
                "y_coords and epsg_code."
            )
        da = xr.DataArray(
            np.asarray(data), coords={"y": y_coords, "x": x_coords}, dims=["y", "x"]
        ).rio.write_crs(f"EPSG:{epsg_code}")

    return da.rio.reproject("EPSG:4326")
