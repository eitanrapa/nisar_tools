"""Water masking via the GMT land/sea mask.

The mask is computed once per grid (it is small at the multilooked
resolution), optionally cached in the workspace, and applied lazily by
broadcasting over the pair/time dimension.

``pygmt`` is an optional dependency (it needs the GMT native library), so it is
imported lazily inside the function rather than at module load.
"""

import numpy as np
import rioxarray  # noqa: F401
import xarray as xr

from . import geo


def make_water_mask(x_coords, y_coords, epsg_code, buffer=0.05):
    """Build a land=1 / water=NaN mask aligned to the given native grid.

    Returns a 2D :class:`xarray.DataArray` on ``(y, x)``.
    """
    import pygmt

    x_min, x_max = float(np.min(x_coords)), float(np.max(x_coords))
    y_min, y_max = float(np.min(y_coords)), float(np.max(y_coords))
    x_spacing = float(np.abs(x_coords[1] - x_coords[0]))
    y_spacing = float(np.abs(y_coords[1] - y_coords[0]))

    lon_min, lon_max, lat_min, lat_max = geo.native_bbox_to_lonlat(
        x_min, x_max, y_min, y_max, epsg_code
    )
    region_latlon = [
        lon_min - buffer,
        lon_max + buffer,
        lat_min - buffer,
        lat_max + buffer,
    ]

    mask_latlon = pygmt.grdlandmask(
        region=region_latlon,
        spacing="5e",
        mask_values=[np.nan, 1, np.nan, 1, np.nan],
        resolution="f",
        registration="p",
    )
    mask_latlon = mask_latlon.rio.write_crs("EPSG:4326")
    mask_xy = mask_latlon.rio.reproject(
        f"EPSG:{epsg_code}", resolution=(x_spacing, y_spacing)
    )

    grid = xr.DataArray(
        np.zeros((len(y_coords), len(x_coords))),
        coords={"y": y_coords, "x": x_coords},
        dims=["y", "x"],
    )
    return mask_xy.interp_like(grid, method="nearest")


def water_mask_for_grid(x_coords, y_coords, epsg_code, workspace=None,
                        name="water_mask"):
    """Return a water mask, computing and caching it in the workspace if given."""
    if workspace is not None and workspace.exists(name):
        cached = workspace.load(name)["mask"]
        if (
            cached.sizes.get("x") == len(x_coords)
            and cached.sizes.get("y") == len(y_coords)
        ):
            return cached

    mask = make_water_mask(x_coords, y_coords, epsg_code)

    if workspace is not None:
        ds = mask.to_dataset(name="mask")
        workspace.store(
            name,
            ds,
            {"stage": name, "epsg": epsg_code, "nx": len(x_coords), "ny": len(y_coords)},
            overwrite=True,
        )
    return mask
