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
from ._base import (
    RasterStackMixin,
    compute_chunks,
    open_stage,
    plane_kernel,
    wrapped_phase,
)


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
    # Wrapped phase and coherence by default; the interferogram amplitude is
    # available via ``fields=["amplitude", ...]``.
    GRD_DEFAULT_FIELDS = ("phase", "coherence")

    def __init__(self, ds):
        self.ds = ds

    # -- construction ------------------------------------------------------
    @classmethod
    def from_slc_stack(
        cls, stack, pairs="sequential", looks=5, downsample=True,
        convolution="Uniform", nan_aware=True, min_valid_fraction=0.5,
        align_looks=True,
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
        # Drop the few leading samples that would put the multilook blocks off
        # the absolute lattice, so this frame's multilooked grid depends only on
        # the native grid and ``looks`` -- never on the crop it came from. Two
        # frames of one track then land on a shared lattice and merge exactly.
        lead_x = _kernels.lattice_lead(x, looks) if (align_looks and downsample) else 0
        lead_y = _kernels.lattice_lead(y, looks) if (align_looks and downsample) else 0
        if lead_x or lead_y:
            ref = ref.isel(x=slice(lead_x, None), y=slice(lead_y, None))
            sec = sec.isel(x=slice(lead_x, None), y=slice(lead_y, None))
            x = x[lead_x:]
            y = y[lead_y:]

        max_x = len(x) // looks * looks
        max_y = len(y) // looks * looks
        if downsample and (max_x == 0 or max_y == 0):
            raise ValueError(
                f"Not enough samples to multilook by {looks} after aligning to "
                f"the lattice (y: {len(y)}, x: {len(x)}); use a smaller looks "
                "or align_looks=False"
            )

        # The kernel batches over the leading (pair) axis and multilooks the
        # trailing spatial axes, so 3D dask arrays go straight through.
        igram, coherence = _kernels.igram_coherence(
            ref.data, sec.data, max_x, max_y, looks, downsample, convolution,
            nan_aware=nan_aware, min_valid_fraction=min_valid_fraction,
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
            align_looks=bool(align_looks),
            convolution=convolution,
            nan_aware=bool(nan_aware),
            min_valid_fraction=float(min_valid_fraction),
            x_spacing=float(stack.ds.attrs.get("x_spacing", np.nan)),
            y_spacing=float(stack.ds.attrs.get("y_spacing", np.nan)),
            pairs=[list(p) for p in pair_list],
        )
        return cls(ds)

    @classmethod
    def from_zarr(cls, path):
        return cls(open_stage(path))

    # -- operations --------------------------------------------------------
    def mask_water(self, mask_cache=None, resolution="f", spacing=None,
                   mask_name=None):
        """Lazily mask water on both igram and coherence. Returns a new stack.

        Lazy: the masked values are **not** written anywhere. Call
        :meth:`persist` (under a new stage name) if you want them on disk.

        ``mask_cache`` is a :class:`~nisar_tools.workspace.Workspace` used to
        cache the *coastline mask itself*, keyed on the grid, so GMT is not
        re-run for the same crop. It is not where the masked data goes.

        ``resolution`` is the GMT coastline resolution; use a coarser value
        (e.g. ``"i"``) if the full-resolution GSHHG dataset is unavailable.
        ``spacing`` defaults to tracking this stack's own pixel size, so a
        multilooked stack builds a correspondingly coarse coastline.
        ``mask_name`` overrides the cache store's name, which otherwise is
        derived from the grid so masks for different grids coexist.
        """
        from .mask import grid_spacing_arg, water_mask_for_grid

        # Resolve here so the recorded value (which feeds the stage hash) is
        # the increment actually used, not a placeholder None.
        if spacing is None:
            spacing = grid_spacing_arg(self.x, self.y, self.epsg)

        mask = water_mask_for_grid(
            self.x, self.y, self.epsg, workspace=mask_cache, name=mask_name,
            resolution=resolution, spacing=spacing,
        )
        # The mask is land=1 / water=NaN; ``where`` needs a boolean condition
        # (NaN is truthy, so passing the raw mask would keep water pixels).
        keep = mask.notnull()
        ds = self.ds.copy()
        ds["igram"] = self.ds["igram"].where(keep)
        ds["coherence"] = self.ds["coherence"].where(keep)
        ds.attrs.update(self.ds.attrs)
        ds.attrs["water_mask"] = {"resolution": resolution, "spacing": spacing}
        return InterferogramStack(ds)

    def remove_unconnected_regions(self, min_size=None, connectivity=1,
                                   max_drop_fraction=0.01, target_blocks=None):
        """Drop the regions the mainland cannot reach. Returns a new stack.

        Run this between :meth:`mask_water` and :meth:`unwrap`. A coastline mask
        leaves islets, ships, platforms and decorrelated fragments stranded
        offshore, and an unwrapper propagates phase along arcs between adjacent
        pixels -- so a region with no path to the main body carries no recoverable
        ambiguity, however large. Left in place they produce artifacts that appear
        to bridge open water. The criterion is **connectivity, not size**.

        ``min_size=None`` (default) keeps only the largest component. Pass
        ``min_size=N`` to keep every component of more than ``N`` pixels, which is
        what a scene that genuinely *is* two landmasses wants: both real bodies
        survive and the speckle still goes.

        ``max_drop_fraction`` refuses to blank a real landmass -- see
        :func:`~nisar_tools._kernels.remove_unconnected_regions` for why the guard
        is on the largest single dropped component rather than the total, and for
        the ``connectivity`` convention.

        Both ``igram`` and ``coherence`` are masked, so the two stay consistent
        (the same thing :meth:`mask_water` does). Lazy: nothing is written until
        :meth:`persist`. Note the guard raises when the graph *computes*, not when
        this is called -- an eager check would mean a whole extra pass over the
        stack.
        """
        igram = self.ds["igram"]
        depth = _kernels.unconnected_regions_depth(
            min_size, igram.sizes["y"], igram.sizes["x"]
        )
        cleaned = plane_kernel(
            _kernels.remove_unconnected_regions_planes, igram, depth=depth,
            target_blocks=target_blocks, min_size=min_size,
            connectivity=connectivity, max_drop_fraction=max_drop_fraction,
        )

        ds = self.ds.copy()
        ds["igram"] = cleaned
        # Coherence carries no NaN of its own (it is exactly 0.0 outside the
        # swath), so it follows the igram's footprint rather than being labelled
        # separately -- which also means one labelling pass, not two.
        ds["coherence"] = self.ds["coherence"].where(cleaned.notnull())
        ds.attrs.update(self.ds.attrs)
        ds.attrs["unconnected_removed"] = {
            "min_size": None if min_size is None else int(min_size),
            "connectivity": int(connectivity),
            "max_drop_fraction": (None if max_drop_fraction is None
                                  else float(max_drop_fraction)),
        }
        return InterferogramStack(ds)

    def filter_goldstein(self, alpha=0.5, patch_size=32, overlap=0.75, psd_smooth=3,
                         target_blocks=None):
        """Goldstein-Werner phase-filter every pair's igram. Returns a new stack.

        A lazy, per-pair adaptive spectral filter applied after multilooking and
        before unwrapping: it sharpens fringes and suppresses phase noise, which
        greatly reduces the residues SNAPHU must resolve. See
        :func:`nisar_tools._kernels.goldstein_filter` for the algorithm and
        parameters.

        ``alpha`` is either a float in ``[0, 1]`` (constant strength; ``0`` is a
        no-op) or ``"adaptive"`` for the Baran et al. (2003) coherence-adaptive
        strength ``1 - coherence`` per patch, matching GMTSAR's ``phasefilt``
        with ``-amp1/-amp2``. The adaptive mode reads this stack's ``coherence``.

        Only ``igram`` is filtered; ``coherence`` (a separate quality measure) is
        left untouched. The filter's support is one ``patch_size``, so it runs chunk
        by chunk with a matching halo rather than a whole plane at a time -- the
        patch lattice stays global, so the result is identical either way (see
        :func:`nisar_tools._kernels.goldstein_filter_dask`).
        """
        adaptive = isinstance(alpha, str)
        igram = self.ds["igram"]
        kwargs = dict(
            patch_size=int(patch_size), overlap=float(overlap),
            psd_smooth=int(psd_smooth),
        )

        # A persisted interferogram arrives on the 2048-px *disk* chunk, and a
        # multilooked one is usually smaller than that in both directions -- so it
        # is one spatial chunk and there is nothing to spread over the cores.
        # Rechunk to a working size first; the halo is one patch_size.
        if _kernels._is_dask(igram.data):
            working = compute_chunks(
                igram.sizes["y"], igram.sizes["x"], patch_size, target_blocks
            )
            if working is not None:
                igram = igram.chunk(
                    {"pair": 1, "y": working[0], "x": working[1]}
                )

        if adaptive:
            # Overlapped alongside the igram, so each block filters against the
            # coherence of the same window.
            coherence = self.ds["coherence"]
            if _kernels._is_dask(igram.data):
                coherence = coherence.chunk(
                    dict(zip(igram.dims, igram.chunks))
                )
            data = _kernels.goldstein_filter_dask(
                igram.data, coherence.data, alpha=alpha, **kwargs
            )
            alpha_attr = alpha
        else:
            data = _kernels.goldstein_filter_dask(
                igram.data, alpha=float(alpha), **kwargs
            )
            alpha_attr = float(alpha)

        filtered = xr.DataArray(
            data, dims=igram.dims, coords=igram.coords, attrs=igram.attrs,
            name=igram.name,
        )

        ds = self.ds.copy()
        ds["igram"] = filtered
        ds.attrs.update(self.ds.attrs)
        ds.attrs["goldstein"] = {
            "alpha": alpha_attr,
            "patch_size": int(patch_size),
            "overlap": float(overlap),
            "psd_smooth": int(psd_smooth),
        }
        return InterferogramStack(ds)

    def unwrap(self, workspace, name="unwrapped", nproc=1, res_az=8, res_rg=3,
               overwrite=False, **kwargs):
        """Unwrap every pair with SNAPHU. See
        :meth:`~nisar_tools.unwrap.UnwrappedStack.from_interferograms` for the
        tiling (``ntiles``, ``tile_overlap``, ``max_tile_pixels``) and concurrency
        (``nproc``, ``pairs_in_flight``) knobs.
        """
        from .unwrap import UnwrappedStack

        return UnwrappedStack.from_interferograms(
            self,
            workspace,
            name=name,
            nproc=nproc,
            res_az=res_az,
            res_rg=res_rg,
            overwrite=overwrite,
            **kwargs,
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
            "align_looks": self.ds.attrs.get("align_looks"),
            "convolution": self.ds.attrs.get("convolution"),
            "nan_aware": self.ds.attrs.get("nan_aware"),
            "min_valid_fraction": self.ds.attrs.get("min_valid_fraction"),
            "pairs": self.ds.attrs.get("pairs"),
            **params,
        }
        # Only record filter params once filtered, so an unfiltered igrams stage
        # keeps its original hash (and re-running with a new alpha re-computes).
        if self.ds.attrs.get("goldstein") is not None:
            full["goldstein"] = self.ds.attrs["goldstein"]
        if self.ds.attrs.get("water_mask") is not None:
            full["water_mask"] = self.ds.attrs["water_mask"]
        if self.ds.attrs.get("unconnected_removed") is not None:
            full["unconnected_removed"] = self.ds.attrs["unconnected_removed"]
        reopened = workspace.store(name, ds, full, overwrite=overwrite)
        return InterferogramStack(reopened)

    # -- export ------------------------------------------------------------
    def _grd_specs(self):
        """Wrapped ``phase`` and ``amplitude`` of the complex interferogram
        plus ``coherence``, per pair."""
        ig = self.ds["igram"]
        return [
            ("phase", wrapped_phase(ig), True),
            ("amplitude", np.abs(ig), True),
            ("coherence", self.ds["coherence"], True),
        ]

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
