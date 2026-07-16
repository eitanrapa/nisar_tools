"""The :class:`InterferogramStack`: multilooked interferograms + coherence.

Backed by a lazy ``xarray`` Dataset with ``igram`` (complex) and ``coherence``
(float32) variables of dims ``(pair, y, x)``. Each pair carries ``ref_time``
and ``sec_time`` auxiliary coordinates (plain coords rather than a MultiIndex,
which would not serialise to Zarr).
"""

from itertools import combinations

import numpy as np
import rioxarray  # noqa: F401
import xarray as xr

from . import _kernels
from ._base import RasterStackMixin


def make_pairs(spec, n):
    """Resolve a pairs specification into a list of ``(ref, sec)`` indices.

    - ``"sequential"``: consecutive acquisitions ``(0,1), (1,2), ...``.
    - ``"all"``: every combination ``(i, j)`` with ``i < j``.
    - an explicit iterable of ``(i, j)`` pairs: validated and returned as-is.
    """
    if spec == "sequential":
        return [(i, i + 1) for i in range(n - 1)]
    if spec == "all":
        return list(combinations(range(n), 2))
    pairs = [tuple(p) for p in spec]
    for i, j in pairs:
        if not (0 <= i < n and 0 <= j < n):
            raise ValueError(f"pair index out of range for stack of {n}: {(i, j)}")
    return pairs


class InterferogramStack(RasterStackMixin):
    """A stack of multilooked interferograms with coherence."""

    STAGE = "igrams"

    def __init__(self, ds):
        self.ds = ds

    # -- construction ------------------------------------------------------
    @classmethod
    def from_slc_stack(
        cls, stack, pairs="sequential", looks=5, downsample=True, convolution="Uniform"
    ):
        if convolution not in _kernels.VALID_CONVOLUTIONS:
            raise ValueError("convolution must be Uniform or Gaussian")

        pair_list = make_pairs(pairs, stack.sizes["time"])
        if len(pair_list) == 0:
            raise ValueError("No pairs to form (need >= 2 acquisitions)")

        ref_idx = [i for i, _ in pair_list]
        sec_idx = [j for _, j in pair_list]

        slc = stack.ds["slc"]
        ref = slc.isel(time=ref_idx)
        sec = slc.isel(time=sec_idx)
        ref_times = np.asarray(ref["time"].values)
        sec_times = np.asarray(sec["time"].values)

        x = stack.x
        y = stack.y
        max_x = len(x) // looks * looks
        max_y = len(y) // looks * looks

        # The kernel batches over the leading (pair) axis and multilooks the
        # trailing spatial axes, so 3D dask arrays go straight through.
        igram, coherence = _kernels.igram_coherence(
            ref.data, sec.data, max_x, max_y, looks, downsample, convolution
        )

        if downsample:
            new_x = _kernels.downsampled_coords(x, looks, max_x)
            new_y = _kernels.downsampled_coords(y, looks, max_y)
        else:
            new_x, new_y = x, y

        npairs = len(pair_list)
        ds = xr.Dataset(
            {
                "igram": (("pair", "y", "x"), igram),
                "coherence": (("pair", "y", "x"), coherence),
            },
            coords={
                "pair": np.arange(npairs),
                "y": new_y,
                "x": new_x,
                "ref_time": ("pair", ref_times),
                "sec_time": ("pair", sec_times),
            },
        )
        ds = ds.rio.write_crs(f"EPSG:{stack.epsg}")
        ds.attrs.update(
            epsg=stack.epsg,
            direction=stack.direction,
            looks=int(looks),
            downsample=bool(downsample),
            convolution=convolution,
            x_spacing=float(stack.ds.attrs.get("x_spacing", np.nan)),
            y_spacing=float(stack.ds.attrs.get("y_spacing", np.nan)),
            pairs=[list(p) for p in pair_list],
        )
        return cls(ds)

    @classmethod
    def from_zarr(cls, path):
        return cls(xr.open_zarr(path))

    # -- operations --------------------------------------------------------
    def mask_water(self, workspace=None, resolution="f", spacing="5e"):
        """Lazily mask water on both igram and coherence. Returns a new stack.

        ``resolution`` is the GMT coastline resolution; use a coarser value
        (e.g. ``"i"``) if the full-resolution GSHHG dataset is unavailable.
        """
        from .mask import water_mask_for_grid

        mask = water_mask_for_grid(
            self.x, self.y, self.epsg, workspace=workspace,
            resolution=resolution, spacing=spacing,
        )
        # The mask is land=1 / water=NaN; ``where`` needs a boolean condition
        # (NaN is truthy, so passing the raw mask would keep water pixels).
        keep = mask.notnull()
        ds = self.ds.copy()
        ds["igram"] = self.ds["igram"].where(keep)
        ds["coherence"] = self.ds["coherence"].where(keep)
        ds.attrs.update(self.ds.attrs)
        return InterferogramStack(ds)

    def unwrap(self, workspace, name="unwrapped", nproc=1, res_az=8, res_rg=3,
               overwrite=False):
        """Unwrap every pair with SNAPHU. See :class:`UnwrappedStack`."""
        from .unwrap import UnwrappedStack

        return UnwrappedStack.from_interferograms(
            self,
            workspace,
            name=name,
            nproc=nproc,
            res_az=res_az,
            res_rg=res_rg,
            overwrite=overwrite,
        )

    # -- persistence -------------------------------------------------------
    def persist(self, workspace, name=None, overwrite=False, **params):
        name = name or self.STAGE
        ds = self.ds.chunk(self.disk_chunks("pair"))
        full = {
            "stage": name,
            "epsg": self.epsg,
            "looks": self.ds.attrs.get("looks"),
            "downsample": self.ds.attrs.get("downsample"),
            "convolution": self.ds.attrs.get("convolution"),
            "pairs": self.ds.attrs.get("pairs"),
            **params,
        }
        reopened = workspace.store(name, ds, full, overwrite=overwrite)
        return InterferogramStack(reopened)

    # -- plotting ----------------------------------------------------------
    def plot_wrapped(self, pair=0):
        from .plot import plot_wrapped_phase

        return plot_wrapped_phase(
            self.ds["igram"].isel(pair=pair), epsg_code=self.epsg
        )

    def __repr__(self):
        s = self.sizes
        return (
            f"<InterferogramStack EPSG:{self.epsg} "
            f"pair={s.get('pair')} y={s.get('y')} x={s.get('x')}>"
        )
