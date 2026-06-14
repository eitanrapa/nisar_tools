"""The :class:`GSLCStack`: a set of co-registered GSLCs on a common grid.

Backed by a lazy ``xarray`` Dataset with a single ``slc`` variable of dims
``(time, y, x)``. Cropping and merging stay lazy; interferogram formation
produces an :class:`~nisar_tools.interferogram.InterferogramStack`.
"""

import numpy as np
import rioxarray  # noqa: F401
import xarray as xr

from . import geo
from ._base import SPATIAL_CHUNK, RasterStackMixin
from .interferogram import InterferogramStack, make_pairs


class GSLCStack(RasterStackMixin):
    """A time stack of aligned GSLCs."""

    STAGE = "slc_stack"

    def __init__(self, ds):
        self.ds = ds

    # -- construction ------------------------------------------------------
    @classmethod
    def from_gslcs(cls, gslcs, bbox=None):
        """Build a stack from open :class:`GSLC` objects, optionally cropped.

        ``bbox`` is ``(lon_min, lon_max, lat_min, lat_max)``. All granules
        must share one EPSG and pass direction and lie on the same grid (e.g.
        the same frame across dates).
        """
        if len(gslcs) == 0:
            raise ValueError("Need at least one GSLC")

        epsgs = {g.epsg for g in gslcs}
        if len(epsgs) > 1:
            raise ValueError(f"GSLCs span multiple EPSG codes: {epsgs}")
        directions = {g.direction for g in gslcs}
        if len(directions) > 1:
            raise ValueError(f"GSLCs span multiple pass directions: {directions}")

        arrays = []
        times = []
        for i, g in enumerate(gslcs):
            da = g.crop(*bbox) if bbox is not None else g.data
            t = g.datetime if g.datetime is not None else np.datetime64(i, "s")
            arrays.append(da)
            times.append(t)

        order = np.argsort(np.asarray(times))
        arrays = [arrays[i] for i in order]
        times = [times[i] for i in order]

        slc = xr.concat(arrays, dim="time", join="exact")
        slc = slc.assign_coords(time=("time", np.asarray(times)))

        ds = slc.to_dataset(name="slc")
        ds = ds.rio.write_crs(f"EPSG:{gslcs[0].epsg}")
        ds.attrs.update(
            epsg=int(gslcs[0].epsg),
            direction=gslcs[0].direction,
            x_spacing=float(gslcs[0].x_spacing),
            y_spacing=float(gslcs[0].y_spacing),
        )
        return cls(ds)

    @classmethod
    def from_zarr(cls, path):
        return cls(xr.open_zarr(path))

    # -- lazy operations ---------------------------------------------------
    def crop(self, lon_min, lon_max, lat_min, lat_max):
        """Return a new, lazily cropped stack."""
        x_min, x_max, y_min, y_max = geo.bbox_to_native(
            lon_min, lon_max, lat_min, lat_max, self.epsg
        )
        x = self.x
        y = self.y
        x_slice = slice(x_min, x_max) if x[0] <= x[-1] else slice(x_max, x_min)
        y_slice = slice(y_min, y_max) if y[0] <= y[-1] else slice(y_max, y_min)
        out = self.ds.sel(x=x_slice, y=y_slice)
        out.attrs.update(self.ds.attrs)
        return GSLCStack(out)

    def merge(self, other):
        """Merge an adjacent-frame stack (same dates) onto the union grid.

        Implemented as an explicit outer-join align + ``fillna``, then a
        re-chunk to undo the ragged chunk layout that the outer reindex
        produces. Self takes precedence where the two overlap.
        """
        if self.epsg != other.epsg:
            raise ValueError("Cannot merge stacks with different EPSG codes")

        a, b = xr.align(self.ds["slc"], other.ds["slc"], join="outer")
        merged = a.fillna(b)
        merged = merged.chunk(
            {"time": 1, "y": SPATIAL_CHUNK, "x": SPATIAL_CHUNK}
        )
        ds = merged.to_dataset(name="slc")
        ds = ds.rio.write_crs(f"EPSG:{self.epsg}")
        ds.attrs.update(self.ds.attrs)
        return GSLCStack(ds)

    def form_interferograms(
        self, pairs="sequential", looks=5, downsample=True, convolution="Uniform"
    ):
        """Form an :class:`InterferogramStack` from pairs of acquisitions."""
        return InterferogramStack.from_slc_stack(
            self,
            pairs=pairs,
            looks=looks,
            downsample=downsample,
            convolution=convolution,
        )

    # -- persistence -------------------------------------------------------
    def persist(self, workspace, name=None, overwrite=False, **params):
        """Write the stack to the workspace and return the reopened lazy stack."""
        name = name or self.STAGE
        ds = self.ds.chunk(self.disk_chunks("time"))
        full_params = {"stage": name, "epsg": self.epsg, **params}
        reopened = workspace.store(name, ds, full_params, overwrite=overwrite)
        return GSLCStack(reopened)

    def make_pairs(self, pairs="sequential"):
        """Resolve a pairs spec into an explicit list against this stack."""
        return make_pairs(pairs, self.sizes["time"])

    def __repr__(self):
        s = self.sizes
        return (
            f"<GSLCStack EPSG:{self.epsg} "
            f"time={s.get('time')} y={s.get('y')} x={s.get('x')}>"
        )
