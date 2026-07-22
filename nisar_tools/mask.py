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

# Build the coastline this many times finer than the target pixel. Sampling it
# back down is nearest-neighbour, so a source slightly finer than the
# destination keeps the reprojection well posed; beyond ~2x the coastline moves
# by well under a pixel and only costs time.
_MASK_OVERSAMPLE = 2


def grid_spacing_arg(x_coords, y_coords, epsg_code, oversample=_MASK_OVERSAMPLE):
    """GMT ``-I`` increment that resolves the coastline at this grid's pixel.

    **GMT's ``e`` suffix means metres, not arc-seconds** — a plain number is
    degrees, ``s`` is arc-seconds. Getting that wrong is expensive: the old
    default of ``"5e"`` read as "5 arc-seconds" but asked GMT for a *5 metre*
    coastline, so the mask was built at the GSLC's native resolution however
    much the interferogram had been multilooked. On a coastal crop that was
    ~1000x more nodes than the grid could represent, and about a minute of
    GMT time per call.
    """
    import pyproj

    step = min(
        abs(float(x_coords[1] - x_coords[0])),
        abs(float(y_coords[1] - y_coords[0])),
    ) / oversample
    # Trimmed: coordinate arithmetic leaves float noise, and this string goes
    # into the mask cache key, where 0.0009999999999976694 and 0.001 would
    # otherwise look like different masks.
    step = f"{step:.6g}"
    # Geographic grids are already in degrees, which is GMT's unit-less default.
    if pyproj.CRS.from_epsg(int(epsg_code)).is_geographic:
        return step
    return f"{step}e"


def make_water_mask(x_coords, y_coords, epsg_code, buffer=0.05,
                    resolution="f", spacing=None):
    """Build a land=1 / water=NaN mask aligned to the given native grid.

    Returns a 2D :class:`xarray.DataArray` on ``(y, x)``.

    ``resolution`` is the GMT/GSHHG coastline resolution
    (``"f"``/``"h"``/``"i"``/``"l"``/``"c"``, full→crude). ``"f"`` is the most
    accurate but needs the large full-resolution GSHHG dataset downloaded; if
    that download is unavailable or its cache is corrupt, pass a coarser
    resolution such as ``"i"`` (intermediate), which is more than adequate for
    masking geocoded SAR and downloads reliably.

    ``spacing`` is the GMT increment for the coastline grid. Leave it ``None``
    to track the target grid (see :func:`grid_spacing_arg`) — the mask only has
    to resolve the coastline at the pixel size it will be sampled onto, so a
    multilooked stack needs a far coarser mask than a full-resolution one.
    """
    import pygmt

    if spacing is None:
        spacing = grid_spacing_arg(x_coords, y_coords, epsg_code)

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
    mask = mask_xy.interp_like(grid, method="nearest")
    # Drop rio's CRS bookkeeping (the scalar ``spatial_ref`` coordinate):
    # applying the mask via ``.where`` would propagate it onto the result,
    # where it collides with the ``spatial_ref`` variable already in the
    # zarr-backed stacks and makes xarray's merge fail as ambiguous.
    return mask.reset_coords(drop=True)


def water_mask_for_grid(x_coords, y_coords, epsg_code, workspace=None,
                        name="water_mask", resolution="f", spacing=None):
    """Return a water mask, computing and caching it in the workspace if given.

    ``resolution``/``spacing`` are forwarded to :func:`make_water_mask`. The
    cache is keyed on the full grid identity (EPSG, shape, origin) and the
    mask parameters, so a different crop or resolution never reuses a stale
    mask; a parameter change recomputes and overwrites.
    """
    # Resolve before hashing: the grid-derived spacing has to be part of the
    # key, or two grids with the same shape and origin but different pixel
    # sizes would share a cached mask.
    if spacing is None:
        spacing = grid_spacing_arg(x_coords, y_coords, epsg_code)

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
        # ``reset_coords`` also cleans masks cached before ``make_water_mask``
        # stripped the ``spatial_ref`` coordinate.
        return workspace.load(name)["mask"].reset_coords(drop=True)

    mask = make_water_mask(
        x_coords, y_coords, epsg_code, resolution=resolution, spacing=spacing
    )

    if workspace is not None:
        ds = mask.to_dataset(name="mask")
        workspace.store(name, ds, params, overwrite=True)
    return mask
