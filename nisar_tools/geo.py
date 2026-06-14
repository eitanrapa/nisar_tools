"""Geographic helpers: bbox transforms and lat/lon reprojection.

Kept free of any class so they can be reused by GSLC, the stacks, and the
plotting layer. Reprojection is intentionally eager (rioxarray loads its
input), so it must only ever be handed a single small 2D slice.
"""

import numpy as np
import rioxarray  # noqa: F401  (registers the .rio accessor on xarray objects)
import xarray as xr
from pyproj import Transformer


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
    lat_min, lat_max)``."""
    transformer = Transformer.from_crs(f"EPSG:{epsg_code}", "EPSG:4326", always_xy=True)
    lon0, lat0 = transformer.transform(x_min, y_min)
    lon1, lat1 = transformer.transform(x_max, y_max)
    return min(lon0, lon1), max(lon0, lon1), min(lat0, lat1), max(lat0, lat1)


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
