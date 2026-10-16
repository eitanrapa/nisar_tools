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
    masked_edges,
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



def _phase_model_rotation(model, x, y, like):
    """``(rotation, description, at)`` for ``from_slc_stack(phase_model=)``: ``exp(i model)`` at the native pixels
    ``x``/``y`` as a dask array chunked like ``like``'s trailing (y, x) axes (complex64), a description carrying a hash of
    the model (for the stage's params), and ``at(x, y)`` -> the model on any grid (float32)."""
    import hashlib

    import dask.array as da
    from scipy.interpolate import RegularGridInterpolator

    mx = np.asarray(model["x"].values, float)
    my = np.asarray(model["y"].values, float)
    mv = np.asarray(model.values, float)
    if mv.ndim != 2:
        raise ValueError("phase_model must be a (y, x) DataArray")
    if my[0] > my[-1]:
        my, mv = my[::-1], mv[::-1]
    mv = np.where(np.isfinite(mv), mv, 0.0)
    f = RegularGridInterpolator((my, mx), mv, bounds_error=False, fill_value=0.0)

    def at(xx, yy):
        Y, X = np.meshgrid(np.asarray(yy, float), np.asarray(xx, float), indexing="ij")
        return f(np.column_stack([Y.ravel(), X.ravel()])).reshape(Y.shape).astype(np.float32)

    ychunks, xchunks = like.chunks[-2], like.chunks[-1]
    zero = da.zeros((len(y), len(x)), chunks=(ychunks, xchunks), dtype=np.float32)

    def block(z, block_info=None):
        (r0, r1), (c0, c1) = block_info[0]["array-location"]
        return np.exp(1j * at(x[c0:c1], y[r0:r1])).astype(np.complex64)

    rot = da.map_blocks(block, zero, dtype=np.complex64)
    digest = hashlib.sha256(np.ascontiguousarray(mv, dtype=np.float32).tobytes()).hexdigest()[:16]
    desc = str(model.attrs.get("description", "phase model")) + f" [sha256 {digest}, {mv.shape[0]} x {mv.shape[1]}]"
    return rot, desc, at


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
        align_looks=True, sigma=None, phase_model=None,
    ):
        """``phase_model`` (optional): a ``(y, x)`` DataArray of phase (rad, this band) on a regular map grid in
        the stack's CRS -- e.g. a near-fault LOS model from pixel offsets. It is taken out of every pair at full
        resolution, ``ref * conj(sec) * exp(-i model)``, BEFORE multilooking, so steep coseismic fringes do not
        cancel inside the looks, and is kept on the multilooked grid as ``phase_model`` for
        :meth:`~nisar_tools.unwrap.UnwrappedStack.from_interferograms` to add back after unwrapping. Bilinear
        at each pixel; 0 outside the model's grid and where it is NaN.
        """
        if convolution not in _kernels.VALID_CONVOLUTIONS:
            raise ValueError("convolution must be Uniform or Gaussian")
        if sigma is not None and convolution != "Gaussian":
            raise ValueError("sigma applies to the Gaussian multilook only")

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
        sec_data = sec.data
        model_desc = None
        if phase_model is not None:
            rot, model_desc, model_at = _phase_model_rotation(phase_model, np.asarray(x), np.asarray(y), sec.data)
            sec_data = sec_data * rot[None]                 # conj(sec * e^{i m}) = conj(sec) e^{-i m}
        igram, coherence = _kernels.igram_coherence(
            ref.data, sec_data, max_x, max_y, looks, downsample, convolution,
            nan_aware=nan_aware, min_valid_fraction=min_valid_fraction, sigma=sigma,
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
        if sigma is not None:
            ds.attrs["sigma"] = float(sigma)
        if phase_model is not None:
            ds["phase_model"] = (("y", "x"), model_at(np.asarray(new_x, float), np.asarray(new_y, float)))
            ds.attrs["phase_model"] = model_desc
        # Which sub-band this is, carried so the dual-frequency ionosphere
        # correction can read both carriers off the stacks instead of being told
        # them. Not in the persist hash: it is provenance, not a parameter, and
        # folding it in would stale every existing igrams store.
        for key in ("frequency", "center_frequency", "wavelength"):
            if stack.ds.attrs.get(key) is not None:
                ds.attrs[key] = stack.ds.attrs[key]
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

    def mask_edges(self, edge_pixels=8, min_coherence=None, edges="along_track",
                   target_blocks=None):
        """Mask swath-edge effects on the interferogram. Returns a new stack (lazy).

        The same trim :meth:`UnwrappedStack.mask_edges
        <nisar_tools.unwrap.UnwrappedStack.mask_edges>` applies, run before the
        unwrap instead of after it: the decorrelated near/far-range fringe is
        residues for SNAPHU to resolve, and resolving them wrongly is what puts a
        2*pi seam into the interior. ``min_coherence`` additionally nulls pixels
        below that coherence, which is the other standard way to keep residues
        away from the solver.

        ``edges="along_track"`` (default) trims only the near/far-range
        boundaries, scanning each row for its first and last valid sample rather
        than eroding the footprint -- so a coastline, a lake blanked by
        :meth:`mask_water` and the frame's azimuth ends are all spared.
        ``edges="all"`` erodes the whole finite footprint instead. See
        :func:`~nisar_tools._kernels.along_track_edge_mask`. The sides can also
        be named one at a time -- ``"left"``, ``"right"``, ``"top"``,
        ``"bottom"``, a sequence of them, or a ``{side: pixels}`` mapping for
        different depths; see :meth:`UnwrappedStack.mask_edges
        <nisar_tools.unwrap.UnwrappedStack.mask_edges>`.

        ⚠️ **Prefer :meth:`GSLCStack.mask_edges
        <nisar_tools.stack.GSLCStack.mask_edges>` where the SLCs are still to
        hand.** Trimming here blanks a value the multilook has *already* averaged
        the decorrelated samples into -- including into the coherence that is
        supposed to flag them. Masking before multilooking drops them from that
        average instead. This method is for a stack already persisted past the SLC
        stage. Note ``edge_pixels`` is in multilooked pixels here, and in native
        ones there.

        Both ``igram`` and ``coherence`` end up on one footprint, as
        :meth:`mask_water` does. The edge scan runs on ``igram`` alone: coherence
        is exactly ``0.0`` outside the swath rather than NaN, so scanning it for a
        row's first valid sample would find column 0 on every row.
        """
        igram = self.ds["igram"]
        coherence = self.ds["coherence"]

        masked, provenance = masked_edges(
            igram, edge_pixels, edges, target_blocks=target_blocks
        )

        if min_coherence is not None:
            masked = masked.where(coherence >= min_coherence)

        ds = self.ds.copy()
        ds["igram"] = masked
        ds["coherence"] = coherence.where(masked.notnull())
        ds.attrs.update(self.ds.attrs)
        ds.attrs["edges_masked"] = {**provenance, "min_coherence": min_coherence}
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

    def correct_ionosphere(self, screen):
        """Rotate the wrapped interferogram by an ionospheric screen (lazy).

        ``screen`` is what
        :func:`~nisar_tools.ionosphere.dual_frequency_screen` returns -- the
        dispersive phase measured from two *unwrapped* sub-band stacks -- or an
        :class:`~nisar_tools.unwrap.UnwrappedStack` carrying one as its
        ``phase_screen`` layer. It is removed here as a phase rotation,
        ``igram * exp(-i * screen)``, so the pair goes into SNAPHU already
        flattened.

        **Correcting before the unwrap is the point of this method.** A large
        ionospheric ramp is fringes, and every one of them is an opportunity for
        the unwrapper to lay a 2*pi seam in the wrong place; removing them first
        is cheaper than unwrapping them and then subtracting. The screen itself
        still has to come from unwrapped sub-band phases -- there is no
        wrapped-domain estimate, only a wrapped-domain *application* (see
        :mod:`nisar_tools.ionosphere`) -- but those can be unwrapped from a
        coarsely multilooked pair, which is both easier and, for a
        kilometre-scale screen, no less accurate. The screen is interpolated onto
        this stack's grid, so the two need not match.

        Coherence is untouched: a phase rotation changes no amplitude.
        """
        from . import ionosphere

        if self.ds.attrs.get("ionosphere") is not None:
            raise ValueError(
                "An ionospheric screen has already been removed from this "
                "stack; correcting again would subtract a second one."
            )
        field, provenance = ionosphere.as_screen(
            screen, np.asarray(self.ds["y"].values),
            np.asarray(self.ds["x"].values), self.sizes["pair"],
        )
        field = field.assign_coords(pair=self.ds["pair"].values)

        igram = self.ds["igram"]
        ds = self.ds.copy()
        ds["igram"] = (igram * np.exp(-1j * field)).astype(igram.dtype)
        ds.attrs.update(self.ds.attrs)
        ds.attrs["ionosphere"] = {**provenance, "applied": "wrapped"}
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
        # sigma only when set, so default-sigma stores keep their existing hashes
        if self.ds.attrs.get("sigma") is not None:
            full["sigma"] = float(self.ds.attrs["sigma"])
        # Only record filter params once filtered, so an unfiltered igrams stage
        # keeps its original hash (and re-running with a new alpha re-computes).
        if self.ds.attrs.get("goldstein") is not None:
            full["goldstein"] = self.ds.attrs["goldstein"]
        if self.ds.attrs.get("water_mask") is not None:
            full["water_mask"] = self.ds.attrs["water_mask"]
        if self.ds.attrs.get("edges_masked") is not None:
            full["edges_masked"] = self.ds.attrs["edges_masked"]
        if self.ds.attrs.get("unconnected_removed") is not None:
            full["unconnected_removed"] = self.ds.attrs["unconnected_removed"]
        if self.ds.attrs.get("ionosphere") is not None:
            full["ionosphere"] = self.ds.attrs["ionosphere"]
        if self.ds.attrs.get("phase_model") is not None:
            full["phase_model"] = self.ds.attrs["phase_model"]
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
