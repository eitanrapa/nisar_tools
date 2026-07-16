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


def make_water_mask(x_coords, y_coords, epsg_code, buffer=0.05,
                    resolution="f", spacing="5e"):
    """Build a land=1 / water=NaN mask aligned to the given native grid.

    Returns a 2D :class:`xarray.DataArray` on ``(y, x)``.

    ``resolution`` is the GMT/GSHHG coastline resolution
    (``"f"``/``"h"``/``"i"``/``"l"``/``"c"``, full→crude). ``"f"`` is the most
    accurate but needs the large full-resolution GSHHG dataset downloaded; if
    that download is unavailable or its cache is corrupt, pass a coarser
    resolution such as ``"i"`` (intermediate), which is more than adequate for
    masking geocoded SAR and downloads reliably. ``spacing`` is the mask grid
    spacing in lon/lat (GMT units, e.g. ``"5e"`` = 5 arc-seconds).
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

    # maskvalues=[wet, dry]: ocean -> NaN, land -> 1, so multiplying a raster
    # by this mask keeps land and blanks water. (pygmt >= 0.12 renamed
    # mask_values -> maskvalues and split border handling into bordervalues;
    # the default bordervalues treats coastline nodes as land.)
    mask_latlon = pygmt.grdlandmask(
        region=region_latlon,
        spacing=spacing,
        maskvalues=[np.nan, 1],
        resolution=resolution,
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
                        name="water_mask", resolution="f", spacing="5e"):
    """Return a water mask, computing and caching it in the workspace if given.

    ``resolution``/``spacing`` are forwarded to :func:`make_water_mask`. The
    cache is keyed on the full grid identity (EPSG, shape, origin) and the
    mask parameters, so a different crop or resolution never reuses a stale
    mask; a parameter change recomputes and overwrites.
    """
    params = {
        "stage": name,
        "epsg": int(epsg_code),
        "nx": len(x_coords),
        "ny": len(y_coords),
        "x0": float(x_coords[0]),
        "y0": float(y_coords[0]),
        "resolution": resolution,
        "spacing": spacing,
    }
    if workspace is not None and workspace.has(name, params):
        return workspace.load(name)["mask"]

    mask = make_water_mask(
        x_coords, y_coords, epsg_code, resolution=resolution, spacing=spacing
    )

    if workspace is not None:
        ds = mask.to_dataset(name="mask")
        workspace.store(name, ds, params, overwrite=True)
    return mask
