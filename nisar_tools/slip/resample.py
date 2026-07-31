"""Putting every scene on one lattice, in the frame the inversion works in.

Tracks arrive on whatever grid their processor chose -- a NISAR product at 25-50 m
in its own UTM zone, a GMTSAR ``.grd`` at 10 arc-seconds in lon/lat. Sampling them
as they are is not neutral, because a quadtree cell is an integer number of
pixels halved at its midpoint: the reachable cell sizes form a **dyadic ladder set
by the pixel size**, per axis and per scene (see
:func:`~nisar_tools.slip.diagnostics.cell_size_ladder`). Two scenes at different
resolutions land on different ladders, so one ``width_min`` means two different
things, the per-track sample counts diverge for a reason that has nothing to do
with the data, and that reaches the inversion as an unintended reweighting
through :meth:`~nisar_tools.slip.sampling.Observations.concat`.

So: resample everything onto one lattice first. The lattice is built in the
shared :class:`~nisar_tools.slip.frame.LocalFrame` rather than in a UTM zone,
because the frame is what the inversion measures in anyway and it is the same
grid for every track -- a study area spanning a zone boundary has no single UTM
grid, which is the whole reason :class:`LocalFrame` exists.

The default spacing is 10 arc-seconds, which is what ALOS-2 interferograms are
usually delivered at, and is already ~20x finer than a fault element. Going finer
buys the inversion nothing and costs it noise.
"""

import numpy as np

from .. import geo
from ..los import LOSStack
from ..stack import warp_onto_lattice

#: 10 arc-seconds of latitude, in metres -- the conventional InSAR posting, and
#: the exact spacing of the ALOS-2 grids this was written for (measured
#: ``0.00277836 deg`` = 10.0021 arcsec).
ARCSEC_10 = 10.0 / 3600.0 * 111320.0


def frame_lattice(stacks, frame, spacing=ARCSEC_10, bounds=None, pad=0.0):
    """One north-up ``(x, y)`` lattice in ``frame`` covering every stack.

    Anchored on multiples of ``spacing`` from the frame origin, so the lattice
    depends only on the frame and the spacing -- not on which stacks were passed,
    nor on the order they came in. Two runs over different subsets of the same
    tracks therefore produce grids that line up exactly.

    ``bounds`` is ``(x_min, x_max, y_min, y_max)`` in local frame metres and
    overrides the union of the stacks' footprints; ``pad`` widens that union.

    The returned coordinates **are** local frame metres, because the grid is
    built in :attr:`~nisar_tools.slip.frame.LocalFrame.local_crs` -- the frame's
    projection with the origin folded into its false easting. That is what lets
    the sampler read ``x``/``y`` straight off the raster while the raster stays
    correctly georeferenced for export.
    """
    stacks = [stacks] if isinstance(stacks, LOSStack) else list(stacks)
    if not stacks:
        raise ValueError("Need at least one LOSStack to build a lattice for")

    if bounds is None:
        boxes = [_footprint(stack, frame) for stack in stacks]
        bounds = (min(b[0] for b in boxes), max(b[1] for b in boxes),
                  min(b[2] for b in boxes), max(b[3] for b in boxes))
    x_min, x_max, y_min, y_max = (float(v) for v in bounds)
    x_min, y_min = x_min - pad, y_min - pad
    x_max, y_max = x_max + pad, y_max + pad

    spacing = float(spacing)
    if spacing <= 0:
        raise ValueError(f"spacing must be positive, not {spacing}")
    # Snap outward to whole multiples of the spacing **in local metres**, so the
    # grid phase is tied to the study area rather than to a false easting.
    x = np.arange(np.floor(x_min / spacing), np.ceil(x_max / spacing) + 1) * spacing
    y = np.arange(np.ceil(y_max / spacing), np.floor(y_min / spacing) - 1, -1) * spacing
    return x, y


def resample_to_frame(stack, frame, x=None, y=None, spacing=ARCSEC_10,
                      resampling="bilinear", name=None):
    """Warp one :class:`~nisar_tools.los.LOSStack` onto a lattice in ``frame``.

    Pass the ``x``/``y`` from :func:`frame_lattice` to put several tracks on the
    *same* grid; without them a lattice is built for this stack alone, which is
    only useful when there is one track.

    The result is a normal ``LOSStack`` except that it carries
    ``attrs["frame"]`` and **no** ``attrs["epsg"]`` -- the frame's transverse
    Mercator has no EPSG code. :meth:`~nisar_tools.slip.sampling.Observations.from_los`
    recognises the frame attribute and skips its usual reprojection, and anything
    that only needs the projection should read
    :attr:`~nisar_tools._base.RasterStackMixin.crs`.
    """
    if x is None or y is None:
        x, y = frame_lattice([stack], frame, spacing=spacing)

    src = stack.crs
    dst = frame.local_crs
    out = {}
    for var in stack.ds.data_vars:
        field = stack.ds[var]
        if "y" not in field.dims or "x" not in field.dims:
            continue
        if field.dims[0] in ("y",):                    # a shared 2-D geometry layer
            work = field.expand_dims("pair")
            out[var] = warp_onto_lattice(
                work, x, y, src, dst, resampling=resampling
            ).isel(pair=0, drop=True)
        else:
            out[var] = warp_onto_lattice(field, x, y, src, dst, resampling=resampling)

    import xarray as xr

    ds = xr.Dataset(out)
    ds = ds.rio.write_crs(frame.local_crs)
    ds.attrs.update(stack.ds.attrs)
    ds.attrs.pop("epsg", None)          # the frame's tmerc has no EPSG code
    ds.attrs.update(
        frame=frame.to_dict(),
        resampled={"spacing": float(x[1] - x[0]), "resampling": str(resampling),
                   "source_epsg": stack.ds.attrs.get("epsg")},
    )
    if name is not None:
        ds.attrs["direction"] = ds.attrs.get("direction") or name
        ds.attrs["track"] = name
    return LOSStack(ds)


def resample_all(scenes, frame, spacing=ARCSEC_10, bounds=None, pad=0.0,
                 resampling="bilinear"):
    """Put a ``{name: LOSStack}`` mapping on one shared lattice.

    Returns a mapping of the same keys. Every output shares bit-identical
    ``x``/``y``, which is what makes their quadtree cell-size ladders -- and so
    their sample counts -- comparable.
    """
    scenes = dict(scenes)
    x, y = frame_lattice(list(scenes.values()), frame, spacing=spacing,
                         bounds=bounds, pad=pad)
    return {
        name: resample_to_frame(stack, frame, x, y, resampling=resampling, name=name)
        for name, stack in scenes.items()
    }


def _footprint(stack, frame):
    """A stack's bounding box in local frame metres, edge-densified."""
    sx, sy = stack.x, stack.y
    dx = abs(float(sx[1] - sx[0])) / 2.0
    dy = abs(float(sy[1] - sy[0])) / 2.0
    x_min, x_max, y_min, y_max = geo.transform_native_bbox(
        min(sx) - dx, max(sx) + dx, min(sy) - dy, max(sy) + dy,
        stack.crs, frame.local_crs,
    )
    return x_min, x_max, y_min, y_max
