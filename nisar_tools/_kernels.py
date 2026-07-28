"""Numeric kernels for multilooking, interferogram formation, Goldstein phase
filtering, and SNAPHU sizing.

The multilook/interferogram numerics are ported verbatim from the original
procedural module (kept as ``tests/legacy_reference.py``, the equivalence-test
oracle). The ``*_dask`` helpers wrap the same math so it can
run lazily over chunked arrays without ever materializing a full-resolution
stack. The plain-numpy versions are kept as the reference implementation that
the dask path is equivalence-tested against.

:func:`igram_coherence` is the one place that departs from the oracle: it
defaults to a NaN-aware normalized convolution, because the legacy formula lets
scipy's non-NaN-aware filters spread every invalid sample across the whole
filter footprint. ``nan_aware=False`` still reproduces the legacy result
exactly, and that is what the equivalence tests pin.

:func:`goldstein_filter` is new (it has no legacy counterpart); it is a 2D
whole-plane operation validated by its own properties -- an exact ``alpha=0``
round-trip, phase-noise reduction, and NaN preservation -- rather than against
the legacy oracle.
"""

import math
import warnings

import numpy as np
from scipy.ndimage import (
    binary_erosion,
    gaussian_filter,
    gaussian_filter1d,
    uniform_filter,
    uniform_filter1d,
)

VALID_CONVOLUTIONS = ("Uniform", "Gaussian")


def _is_dask(arr):
    """True if ``arr`` is a dask array, without importing dask eagerly."""
    return type(arr).__module__.split(".")[0] == "dask"


def _multilook_many(arrays, max_x, max_y, looks, downsample, convolution):
    """Multilook several same-shaped real arrays, fusing the dask passes.

    On dask the arrays are stacked on a new leading axis and pushed through a
    single :func:`multilook_dask`. They share an input footprint and a halo, so as
    separate calls each one materialises the same overlap again;
    ``multilook_dask`` only ever filters the trailing two axes, so the stacked
    axis rides through untouched exactly as ``pair`` does.

    Numpy input is looped instead. :func:`multilook` is the verbatim 2-D legacy
    kernel -- it filters *every* axis and slices the first two -- so a stacked
    axis would be smoothed along and mis-sliced. There is nothing to save there
    anyway: without chunks there is no halo to materialise twice.
    """
    if _is_dask(arrays[0]):
        import dask.array as da

        stacked = multilook_dask(
            da.stack(arrays), max_x, max_y, looks, downsample, convolution
        )
        return [stacked[i] for i in range(len(arrays))]
    return [
        multilook(arr, max_x, max_y, looks, downsample, convolution)
        for arr in arrays
    ]


def _fillna_zero(arr):
    """Replace NaNs with 0.0 on either a numpy or a dask array.

    ``np.nan_to_num(arr, nan=0.0)`` does not dispatch onto dask (its
    ``nan_to_num`` rejects the ``nan=`` keyword), so dask is handled
    explicitly. The numpy branch is kept identical to the legacy call.
    """
    if _is_dask(arr):
        import dask.array as dask_array

        return dask_array.where(dask_array.isnan(arr), 0.0, arr)
    return np.nan_to_num(arr, nan=0.0)


def _overlap_depth(looks, convolution):
    """Chunk-overlap depth needed so a windowed filter matches the global one.

    The depth must be at least the filter's radius, otherwise contributions
    from neighbouring chunks are silently dropped and chunk seams appear.

    - Uniform filter of ``size=looks`` reaches at most ``looks // 2`` samples.
    - Gaussian filter of ``sigma=looks`` reaches ``int(truncate*sigma + 0.5)``
      samples, where scipy's default ``truncate`` is 4.0.
    """
    if convolution == "Gaussian":
        return int(4.0 * looks + 0.5)
    if convolution == "Uniform":
        return looks // 2 + 1
    raise ValueError("convolution must be Uniform or Gaussian")


def multilook(arr, max_x, max_y, looks, downsample, convolution):
    """Convolve and (optionally) downsample a 2D array. Verbatim numpy kernel.

    Mirrors the original ``_multilook_array``. Operates on real or complex
    numpy arrays via scipy's filters directly.
    """
    if convolution == "Gaussian":
        smoothed = gaussian_filter(arr, sigma=looks, mode="constant", cval=0.0)
    elif convolution == "Uniform":
        smoothed = uniform_filter(arr, size=looks, mode="constant", cval=0.0)
    else:
        raise ValueError("convolution must be Uniform or Gaussian")

    if downsample:
        smoothed_truncated = smoothed[:max_y, :max_x]
        smoothed = smoothed_truncated[looks // 2 :: looks, looks // 2 :: looks]

    return smoothed


def multilook_dask(arr, max_x, max_y, looks, downsample, convolution):
    """Lazy multilook over a chunked dask array, matching :func:`multilook`.

    Filters real and imaginary parts separately (linear filters make this
    identical to filtering the complex array, and avoids version-dependent
    complex support inside ``map_overlap``). The smoothing uses
    ``map_overlap`` with a mode-dependent depth so the result is independent
    of the chunk layout.
    """
    depth = _overlap_depth(looks, convolution)
    ndim = arr.ndim
    n_lead = ndim - 2  # leading non-spatial (stack/pair) axes, if any

    # Filter only the trailing two spatial axes. A per-axis parameter of
    # 0 (Gaussian sigma) or 1 (uniform size) leaves the leading stack axes
    # untouched, so pairs/dates never mix even when chunked together.
    if convolution == "Gaussian":
        sigma = (0.0,) * n_lead + (looks, looks)

        def _f(block):
            return gaussian_filter(block, sigma=sigma, mode="constant", cval=0.0)
    else:
        size = (1,) * n_lead + (looks, looks)

        def _f(block):
            return uniform_filter(block, size=size, mode="constant", cval=0.0)

    # Overlap only the trailing spatial axes; never the leading stack axis.
    spatial_depth = {ndim - 2: depth, ndim - 1: depth}

    if arr.dtype.kind == "c":
        real = arr.real.map_overlap(_f, depth=spatial_depth, boundary=0.0)
        imag = arr.imag.map_overlap(_f, depth=spatial_depth, boundary=0.0)
        smoothed = (real + 1j * imag).astype(arr.dtype)
    else:
        smoothed = arr.map_overlap(_f, depth=spatial_depth, boundary=0.0)

    if downsample:
        smoothed = smoothed[..., :max_y, :max_x]
        smoothed = smoothed[..., looks // 2 :: looks, looks // 2 :: looks]

    return smoothed


def boundary_response(ny, nx, max_x, max_y, looks, downsample, convolution):
    """The filter's response to an all-valid raster, i.e. ``multilook(ones)``.

    ``mode="constant", cval=0.0`` treats everything beyond the raster as zero,
    so the response decays toward the edges (to 0.5 at a straight edge, 0.25 at
    a corner) instead of staying at 1. Dividing a smoothed valid-mask by this
    turns it into a true valid *fraction*, which is what distinguishes a real
    data boundary from the arbitrary edge where the raster was cropped.

    Both filters are separable, so this is the outer product of two 1D
    profiles rather than a full 2D filter pass -- two vectors of length ``ny``
    and ``nx``, negligible next to the arrays being multilooked.
    """
    prof_y, prof_x = boundary_response_profiles(
        ny, nx, max_x, max_y, looks, downsample, convolution
    )
    return np.outer(prof_y, prof_x)


def boundary_response_profiles(ny, nx, max_x, max_y, looks, downsample,
                               convolution):
    """The two 1-D profiles whose outer product is :func:`boundary_response`.

    Exposed separately so the nan-aware normalisation can divide a dask array by
    each vector in turn: the outer product is a full-grid float64 array, and
    multiplying it into a dask expression embeds it in every task's payload.
    """
    if convolution == "Gaussian":
        def _f(n):
            return gaussian_filter1d(
                np.ones(n), looks, mode="constant", cval=0.0
            )
    elif convolution == "Uniform":
        def _f(n):
            return uniform_filter1d(
                np.ones(n), looks, mode="constant", cval=0.0
            )
    else:
        raise ValueError("convolution must be Uniform or Gaussian")

    prof_y, prof_x = _f(ny), _f(nx)
    if downsample:
        # Same truncate-then-stride as ``multilook``, so this lands on the
        # downsampled grid.
        prof_y = prof_y[:max_y][looks // 2 :: looks]
        prof_x = prof_x[:max_x][looks // 2 :: looks]
    return prof_y, prof_x


def igram_coherence(c1, c2, max_x, max_y, looks, downsample, convolution,
                    nan_aware=True, min_valid_fraction=0.5):
    """Form a multilooked interferogram and its coherence.

    Works on either numpy or dask arrays: the only backend-specific step is
    the multilook, which is dispatched on the array type. Every other
    operation (``*``, ``conj``, ``abs``, ``sqrt``, ``isfinite``, ``where``,
    ``clip``) is supported identically by numpy and dask.

    ``nan_aware=False`` reproduces the legacy formula verbatim. That path feeds
    NaN straight into scipy's filters, which are not NaN-aware, so every
    invalid sample spreads over the whole filter footprint: a radius of
    ``4 * looks`` for Gaussian, and -- because ``uniform_filter`` is a running
    sum -- everything downstream along both axes for Uniform. On real GSLCs,
    which are NaN outside the swath (~45% of a granule) and on every merged
    union grid, that erodes or wipes out the interferogram.

    ``nan_aware=True`` (the default) instead never lets NaN reach scipy. It
    zero-fills the invalid samples, multilooks the validity mask alongside the
    data, and divides it back out -- a normalized convolution. An output pixel
    is kept when at least ``min_valid_fraction`` of its filter weight came from
    valid input, so the NaN footprint neither dilates nor grows: at a straight
    edge the 0.5 default lands exactly on the true boundary.

    Relative to the legacy path this leaves the phase and the coherence
    unchanged -- the smoothed mask cancels exactly in the coherence ratio, and
    dividing by a positive real does not move the argument. The one difference
    is the interferogram *amplitude* within a filter radius of a boundary,
    where the normalization removes the zero-padding bias the legacy path
    carries.

    Returns ``(interferogram, coherence)`` as ``(complex, float32)``.
    """
    if not nan_aware:
        raw_interf = c1 * np.conj(c2)
        raw_int1 = np.abs(c1) ** 2
        raw_int2 = np.abs(c2) ** 2

        ml = multilook_dask if _is_dask(raw_interf) else multilook
        ml_interf = ml(raw_interf, max_x, max_y, looks, downsample, convolution)
        ml_int1 = ml(raw_int1, max_x, max_y, looks, downsample, convolution)
        ml_int2 = ml(raw_int2, max_x, max_y, looks, downsample, convolution)

        ml_corr = np.abs(ml_interf) / (np.sqrt(ml_int1 * ml_int2) + 1e-8)

        # Force areas completely outside the valid swath back to exactly 0.0.
        ml_corr = _fillna_zero(ml_corr)
        ml_corr = ml_corr.clip(0.0, 1.0)  # .clip works on numpy and dask alike

        return ml_interf, ml_corr.astype(np.float32)

    if not (0.0 <= min_valid_fraction <= 1.0):
        raise ValueError("min_valid_fraction must be in [0, 1]")

    valid = np.isfinite(c1) & np.isfinite(c2)
    mask = valid.astype(np.float32)

    ml = multilook_dask if _is_dask(c1) else multilook

    def _ml(arr):
        return ml(arr, max_x, max_y, looks, downsample, convolution)

    # Zero-fill before filtering: scipy never sees a NaN, so nothing spreads.
    #
    # The three real fields go through one fused multilook pass rather than three
    # (see _multilook_many). The complex interferogram sum stays its own pass: the
    # complex path splits into real and imaginary internally, so folding it in
    # would put two dtypes in one stack.
    sum_interf = _ml(np.where(valid, c1 * np.conj(c2), 0.0))
    sum_int1, sum_int2, sum_mask = _multilook_many(
        [
            np.where(valid, np.abs(c1) ** 2, 0.0),
            np.where(valid, np.abs(c2) ** 2, 0.0),
            mask,
        ],
        max_x, max_y, looks, downsample, convolution,
    )

    # ``sum_mask`` carries both effects we care about: how much of the
    # footprint was valid, and how much of it fell outside the raster. Only the
    # first is a data boundary, so divide the second out before thresholding.
    ny, nx = c1.shape[-2:]
    prof_y, prof_x = boundary_response_profiles(
        ny, nx, max_x, max_y, looks, downsample, convolution
    )
    # Divided out one axis at a time: the response is separable, so this keeps two
    # vectors in the graph instead of a materialised full-grid float64 array.
    valid_fraction = sum_mask / prof_y[:, None] / prof_x[None, :]
    # The ``> 0`` term also covers min_valid_fraction=0, where a pixel with no
    # valid contribution at all would otherwise be kept as a meaningless zero.
    keep = (sum_mask > 0.0) & (valid_fraction >= min_valid_fraction)

    # Normalized convolution: recover the mean over the valid samples alone.
    # ``keep`` is false wherever ``sum_mask`` is 0, so the guard here only
    # avoids a divide-by-zero warning on values that get discarded anyway.
    norm = np.where(sum_mask > 0.0, sum_mask, 1.0)
    ml_interf = sum_interf / norm
    ml_int1 = sum_int1 / norm
    ml_int2 = sum_int2 / norm

    ml_corr = np.abs(ml_interf) / (np.sqrt(ml_int1 * ml_int2) + 1e-8)

    ml_interf = np.where(keep, ml_interf, np.nan).astype(c1.dtype)
    # Outside the valid swath the coherence is exactly 0.0, as in the legacy
    # path; the interferogram carries the NaN footprint instead.
    ml_corr = np.where(keep, _fillna_zero(ml_corr), 0.0)
    ml_corr = ml_corr.clip(0.0, 1.0)

    return ml_interf, ml_corr.astype(np.float32)


def downsampled_coords(coords, looks, max_n):
    """Coordinate values for a downsampled axis (block-mean of ``looks``)."""
    return coords[:max_n].reshape(-1, looks).mean(axis=1)


def _patch_origins(n, ps, step):
    """Start indices of length-``ps`` patches tiling ``[0, n)`` with stride ``step``.

    Always includes ``0`` and ``n - ps`` so every sample is covered by at least
    one full patch (the final patch is flush with the far edge, overlapping its
    neighbour a little more than ``step``). If ``n <= ps`` a single patch suffices.
    """
    if n <= ps:
        return [0]
    origins = list(range(0, n - ps + 1, step))
    if origins[-1] != n - ps:
        origins.append(n - ps)
    return origins


def goldstein_filter(igram, alpha=0.5, patch_size=32, overlap=0.75, psd_smooth=3,
                     coherence=None, y_index=None, x_index=None,
                     grid_shape=None):
    """Goldstein-Werner adaptive spectral filter on a 2D complex interferogram.

    The interferogram is tiled into overlapping ``patch_size`` windows. Each
    patch is tapered, FFT'd, and its spectrum is scaled by
    ``(smooth(|Z|) / max)**alpha`` -- an adaptive low-pass that attenuates
    low-power (noisy) frequencies while preserving the dominant fringe -- then
    inverse-FFT'd and accumulated with a weighted overlap-add (Welch-style, a
    strictly-positive Hamming taper doubling as both the analysis window and the
    blend weight, with a final divide by the summed squared weights).

    ``alpha`` is the filter strength, either:

    - a float in ``[0, 1]``: constant strength; ``0`` is an exact identity (no
      filtering), larger values filter more aggressively; or
    - ``"adaptive"``: the Baran et al. (2003) modification -- each patch's
      strength is ``1 - (window-weighted mean coherence over the patch)``, so
      incoherent areas are filtered hard and coherent ones barely touched. This
      is what GMTSAR's ``phasefilt`` does when given ``-amp1/-amp2``, and
      requires a ``coherence`` array (same shape as ``igram``, values in
      ``[0, 1]``).

    ``patch_size`` is the FFT window (clipped to the raster if smaller);
    ``overlap`` is the fractional patch overlap in ``[0, 1)``; ``psd_smooth`` is
    the boxcar size used to smooth the magnitude spectrum (wrapped, since a DFT
    spectrum is periodic).

    NaNs (e.g. the interferogram's out-of-swath fill) are treated as zero for the
    transforms and restored to NaN in the output. Returns a complex array of the
    same shape and dtype as ``igram``.

    ``y_index``/``x_index``/``grid_shape`` let this run on a *window* of a bigger
    raster while keeping the patch lattice and the overlap-add normalization
    global, so the window's interior matches the whole-plane answer exactly. They
    are what :func:`goldstein_filter_dask` uses to decompose the filter spatially;
    leave them unset for a plain whole-plane call.
    """
    adaptive = isinstance(alpha, str)
    if adaptive:
        if alpha != "adaptive":
            raise ValueError("alpha must be a float in [0, 1] or 'adaptive'")
        if coherence is None:
            raise ValueError("alpha='adaptive' requires a coherence array")
    elif not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0, 1] or 'adaptive'")
    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must be in [0, 1)")
    if psd_smooth < 1:
        raise ValueError("psd_smooth must be >= 1")

    igram = np.asarray(igram)
    dtype = igram.dtype
    bny, bnx = igram.shape

    if adaptive:
        coh = np.clip(np.nan_to_num(np.asarray(coherence, float), nan=0.0), 0.0, 1.0)
        if coh.shape != igram.shape:
            raise ValueError("coherence must have the same shape as igram")

    # ``igram`` may be a *window* of a larger raster (see goldstein_filter_dask).
    # The patch lattice and the overlap-add normalization are properties of the
    # whole raster, so they are always computed on the global grid and then
    # restricted to this window -- that is what makes a windowed call return
    # exactly what the whole-plane call would.
    ny, nx = (bny, bnx) if grid_shape is None else (int(grid_shape[0]),
                                                   int(grid_shape[1]))
    y_offset = 0 if y_index is None else int(np.asarray(y_index).ravel()[0])
    x_offset = 0 if x_index is None else int(np.asarray(x_index).ravel()[0])

    ps = int(min(patch_size, ny, nx))
    if ps < 2:  # nothing meaningful to transform
        return igram.copy()

    step = max(1, int(round(ps * (1.0 - overlap))))

    nan_mask = np.isnan(igram)
    # Compute in double precision so the alpha=0 round-trip is exact to ~1e-12
    # regardless of the (typically complex64) storage dtype.
    work = np.where(nan_mask, 0.0, igram).astype(np.complex128)

    # A Hamming taper (strictly > 0, min 0.08) reduces spectral leakage and
    # blends overlapping patches. Using the same window for analysis and blend,
    # then dividing by the summed squared weights, makes alpha=0 reconstruct the
    # input exactly for any window and any overlap.
    win1d = np.hamming(ps)
    window = np.outer(win1d, win1d)
    window_sum = window.sum()

    out = np.zeros((bny, bnx), dtype=np.complex128)

    # The summed squared weights depend only on the lattice, not on the data, and
    # the sum separates: sum_{y0,x0} w2[y-y0, x-x0] = (sum_y0 h2[y-y0]) *
    # (sum_x0 h2[x-x0]). So it is two 1-D accumulations and an outer product
    # instead of one 2-D slice-add per patch -- half the overlap-add traffic.
    w2_1d = win1d ** 2

    def _lattice(n, offset, extent):
        """Global patch origins, in window coordinates, plus the global weight
        profile restricted to the window."""
        origins = _patch_origins(n, ps, step)
        profile = np.zeros(n)
        for o in origins:
            profile[o:o + ps] += w2_1d
        # Only patches that lie wholly inside the window can be evaluated here;
        # by construction (see goldstein_filter_dask) those are exactly the ones
        # that touch the region the caller keeps.
        local = [o - offset for o in origins
                 if o >= offset and o + ps <= offset + extent]
        return local, profile[offset:offset + extent]

    y_origins, prof_y = _lattice(ny, y_offset, bny)
    x_origins, prof_x = _lattice(nx, x_offset, bnx)
    wsum = np.outer(prof_y, prof_x)

    # The patches are gathered and transformed in one batch rather than one at a
    # time. At the defaults (ps=32, overlap=0.75 -> step=8) a 900x1100 raster has
    # ~15 700 patches, and a per-patch loop spends nearly all its time in numpy
    # call overhead on 32x32 arrays. Batched, each of fft2/uniform_filter/ifft2 is
    # a single call over an (npatch, ps, ps) stack -- and pocketfft releases the
    # GIL, so this stops being a serialisation point inside its dask task too.
    #
    # Rows are batched one band of patches at a time: a whole-raster batch would
    # cost npatch * ps^2 complex128 (~8x the raster at these defaults), while a
    # single row of patches is ~ps/step times a raster row. It also makes the
    # overlap-add safe -- patches within a row write disjoint column spans only
    # after the accumulation is folded per row, so the adds are done row by row.
    for y0 in y_origins:
        ysl = slice(y0, y0 + ps)
        # (npatch_x, ps, ps), one band of patches side by side.
        patches = np.stack([work[ysl, x0:x0 + ps] for x0 in x_origins])
        patches *= window
        spec = np.fft.fft2(patches, axes=(-2, -1))
        # size=(1, psd_smooth, psd_smooth) leaves the batch axis untouched; the
        # spatial axes still wrap, as a DFT spectrum is periodic.
        psd = uniform_filter(
            np.abs(spec), size=(1, psd_smooth, psd_smooth), mode="wrap"
        )
        if adaptive:
            # Baran: strength = 1 - mean coherence over the (windowed) patch.
            strength = 1.0 - np.stack([
                (window * coh[ysl, x0:x0 + ps]).sum() / window_sum
                for x0 in x_origins
            ])
        else:
            strength = np.full(len(x_origins), float(alpha))
        peak = psd.reshape(len(x_origins), -1).max(axis=1) + 1e-20
        # 0**0 == 1, so alpha=0 gives H==1 (identity) even at zero-power bins.
        response = np.power(
            psd / peak[:, None, None], strength[:, None, None]
        )
        filtered = np.fft.ifft2(spec * response, axes=(-2, -1))
        filtered *= window

        for k, x0 in enumerate(x_origins):
            out[ysl, x0:x0 + ps] += filtered[k]

    # wsum > 0 everywhere (Hamming is strictly positive and the patches cover the
    # whole raster); the guard is belt-and-braces against a degenerate tiling.
    out /= np.where(wsum > 0.0, wsum, 1.0)
    out = out.astype(dtype)
    out[nan_mask] = np.nan
    return out


def goldstein_filter_dask(igram, coherence=None, *, alpha=0.5, patch_size=32,
                          overlap=0.75, psd_smooth=3):
    """Goldstein filter over a dask ``(pair, y, x)`` array, chunk by chunk.

    The filter is local: a pixel's value depends only on the patches containing
    it, so nothing further than ``patch_size`` away matters and a ``patch_size``
    halo suffices. What a plain ``map_overlap`` would get *wrong* is the patch
    lattice -- recomputed per block, it would land at different absolute positions
    than the whole-plane run and give different spectra. So each block is handed
    its global origin and the global grid shape, and evaluates the *global* lattice
    restricted to its own padded window. Every patch touching the kept region lies
    wholly inside that window, so the trimmed result is exactly the whole-plane
    result.

    The padded windows are sliced explicitly rather than via ``map_overlap``,
    because that aligns extra arrays by broadcasting from the right and cannot be
    handed a per-axis index array alongside a ``(pair, y, x)`` stack.

    Numpy input falls through to the whole-plane batch wrapper.
    """
    if not _is_dask(igram):
        return goldstein_filter_planes(
            igram, coherence, alpha=alpha, patch_size=patch_size,
            overlap=overlap, psd_smooth=psd_smooth,
        )

    import dask.array as da

    ny, nx = igram.shape[-2:]
    halo = int(min(patch_size, ny, nx))
    kwargs = dict(
        alpha=alpha, patch_size=int(patch_size), overlap=float(overlap),
        psd_smooth=int(psd_smooth), grid_shape=(ny, nx),
    )

    ychunks, xchunks = igram.chunks[-2], igram.chunks[-1]
    ystarts = np.cumsum((0,) + tuple(ychunks[:-1]))
    xstarts = np.cumsum((0,) + tuple(xchunks[:-1]))

    def _block(block, coh=None, y0=0, x0=0):
        return goldstein_filter_planes(
            block, coh, y_index=np.array([y0]), x_index=np.array([x0]), **kwargs
        )

    rows = []
    for y0, hy in zip(ystarts, ychunks):
        cols = []
        for x0, wx in zip(xstarts, xchunks):
            py0, py1 = max(0, y0 - halo), min(ny, y0 + hy + halo)
            px0, px1 = max(0, x0 - halo), min(nx, x0 + wx + halo)
            # One spatial chunk per window; the stack axis keeps its own chunking,
            # so a task holds one padded plane and the pairs stay independent.
            spatial = ((py1 - py0,), (px1 - px0,))
            window = igram[..., py0:py1, px0:px1].rechunk(
                igram.chunks[:-2] + spatial
            )
            args = [window]
            if coherence is not None:
                args.append(
                    coherence[..., py0:py1, px0:px1].rechunk(window.chunks)
                )
            filtered = da.map_blocks(
                _block, *args, y0=py0, x0=px0,
                dtype=igram.dtype, meta=igram._meta,
            )
            iy, ix = y0 - py0, x0 - px0
            cols.append(filtered[..., iy:iy + hy, ix:ix + wx])
        rows.append(cols)
    # np.block semantics: a depth-2 nesting concatenates the last two axes.
    return da.block(rows)


def goldstein_filter_planes(arr, coherence=None, *, alpha=0.5, patch_size=32,
                            overlap=0.75, psd_smooth=3, y_index=None,
                            x_index=None, grid_shape=None):
    """Apply :func:`goldstein_filter` to each trailing 2D plane of ``arr``.

    Leading axes (e.g. a stacked ``pair`` dimension) are looped over so a whole
    3D block goes through in one call without the planes ever mixing. Kept as a
    thin batch wrapper so the 2D kernel stays the single reference implementation.
    ``coherence`` (for ``alpha="adaptive"``) is indexed plane-by-plane alongside
    ``arr``; it is the optional second positional input so ``xr.apply_ufunc`` can
    pass it as a second core-dims array.
    """
    arr = np.asarray(arr)
    coh = None if coherence is None else np.asarray(coherence)
    window = dict(y_index=y_index, x_index=x_index, grid_shape=grid_shape)
    if arr.ndim == 2:
        return goldstein_filter(arr, alpha, patch_size, overlap, psd_smooth,
                                coherence=coh, **window)

    flat = arr.reshape((-1,) + arr.shape[-2:])
    coh_flat = None if coh is None else coh.reshape((-1,) + coh.shape[-2:])
    out = np.empty_like(flat)
    for k in range(flat.shape[0]):
        ck = None if coh_flat is None else coh_flat[k]
        out[k] = goldstein_filter(flat[k], alpha, patch_size, overlap, psd_smooth,
                                  coherence=ck, **window)
    return out.reshape(arr.shape)


# -- unwrapped-phase cleaning: edge masking, spline outlier rejection, deramp --
#
# Ports the recipe of the user's ``filt_gunw.csh`` (GMT ``surface -T`` tension
# spline + residual outlier masking + a deramp) into pure numpy/scipy. GMT's
# adjustable-tension surface has no direct scipy equivalent; a NaN-aware smooth
# surface (normalized Gaussian convolution, the same zero-fill/divide trick as
# ``igram_coherence``) stands in for the smooth trend, and a low-order 2D
# polynomial least-squares fit stands in for the ramp.


def smooth_surface(field, scale, exclude=None):
    """NaN-aware smooth surface of ``field`` (2D), Gaussian ``scale`` in pixels.

    Invalid pixels are zero-filled for the convolution and divided back out via
    a smoothed validity mask, so a NaN neither reads as 0 nor spreads -- the
    normalized-convolution trick used by the nan-aware multilook. Output is NaN
    where the smoothed validity mass is negligible (too little nearby data).

    ``exclude`` (optional boolean mask, same shape) drops those pixels from the
    fit while still evaluating the surface there: the normalized convolution
    fills them from surrounding non-excluded data. Used to keep a signal region
    out of a deramp's trend estimate. Its interior comes back NaN if the gap is
    wide relative to ``scale`` (no nearby data survives to interpolate from).
    """
    field = np.asarray(field, dtype=np.float64)
    finite = np.isfinite(field)
    if exclude is not None:
        finite = finite & ~np.asarray(exclude, dtype=bool)
    w = finite.astype(np.float64)
    filled = np.where(finite, field, 0.0)
    num = gaussian_filter(w * filled, sigma=scale, mode="constant", cval=0.0)
    den = gaussian_filter(w, sigma=scale, mode="constant", cval=0.0)
    return np.divide(num, den, out=np.full_like(num, np.nan), where=den > 1e-6)


def poly_ncoef(degree):
    """Number of coefficients in a total-degree-``degree`` 2D polynomial."""
    return (int(degree) + 1) * (int(degree) + 2) // 2


def _poly_columns(ny, nx, degree, y0=0, x0=0, bny=None, bnx=None):
    """Columns of the 2D polynomial design matrix on a normalized [-1, 1] grid.

    Normalizing keeps the least-squares system well conditioned regardless of
    the pixel coordinates' magnitude. Total-degree basis: 1, x, y, x^2, xy, ...

    ``(y0, x0)`` and ``(bny, bnx)`` select a window of the ``(ny, nx)`` grid while
    keeping the normalization *global*, so a chunk-wise fit builds the same basis
    the whole-plane fit does. Defaults cover the whole grid.
    """
    bny = ny if bny is None else bny
    bnx = nx if bnx is None else bnx
    yy, xx = np.mgrid[y0:y0 + bny, x0:x0 + bnx].astype(np.float64)
    xx = 2.0 * xx / max(nx - 1, 1) - 1.0
    yy = 2.0 * yy / max(ny - 1, 1) - 1.0
    cols = [(xx ** (d - i)) * (yy ** i)
            for d in range(degree + 1) for i in range(d + 1)]
    return np.stack([c.ravel() for c in cols], axis=1)  # (npix, ncoef)


def poly_surface(field, degree, exclude=None):
    """Least-squares 2D polynomial surface fit to ``field``'s finite pixels.

    Evaluated on the whole grid (a polynomial is defined everywhere). Returns
    all-NaN if there are fewer valid pixels than coefficients. ``exclude``
    (optional boolean mask, same shape) drops those pixels from the fit -- so a
    masked signal region does not bias the ramp -- while the fitted polynomial is
    still evaluated over the full grid.
    """
    field = np.asarray(field, dtype=np.float64)
    ny, nx = field.shape
    design = _poly_columns(ny, nx, int(degree))
    values = field.ravel()
    valid = np.isfinite(values)
    if exclude is not None:
        valid = valid & ~np.asarray(exclude, dtype=bool).ravel()
    if int(valid.sum()) < design.shape[1]:
        return np.full((ny, nx), np.nan)
    coef, *_ = np.linalg.lstsq(design[valid], values[valid], rcond=None)
    return (design @ coef).reshape(ny, nx)


def fit_surface(field, method="spline", scale=None, degree=1, exclude=None):
    """Fit a smooth surface to a 2D ``field`` -- the trend a deramp subtracts.

    The single source of truth for the smooth trend :func:`deramp` subtracts.
    ``method="spline"`` is a NaN-aware normalized-convolution Gaussian at sigma
    ``scale`` px (default a quarter of the smaller axis); ``method="poly"`` is a
    total-degree-``degree`` 2D polynomial. ``exclude`` (optional boolean mask)
    is forwarded to the underlying fit to keep a signal region out of it.
    """
    field = np.asarray(field, dtype=np.float64)
    if method == "spline":
        if scale is None:
            scale = 0.25 * min(field.shape)
        return smooth_surface(field, scale, exclude=exclude)
    if method == "poly":
        return poly_surface(field, int(degree), exclude=exclude)
    raise ValueError(f"method must be 'poly' or 'spline', got {method!r}")


def remove_outliers(field, scale, threshold, iterations):
    """Iteratively NaN pixels far from a NaN-aware smooth surface (2D).

    Mirrors ``filt_gunw.csh``: fit a smooth trend, flag ``|field - trend| >
    threshold`` as outliers, and refit on the survivors. ``threshold`` is in the
    field's own units (radians for unwrapped phase). Returns a copy with the
    rejected pixels set to NaN.
    """
    out = np.array(field, dtype=np.float64)
    for _ in range(int(iterations)):
        residual = np.abs(out - smooth_surface(out, scale))
        out = np.where(residual > threshold, np.nan, out)
    return out.astype(np.asarray(field).dtype)


def deramp(field, degree=1, method="poly", scale=None, exclude=None):
    """Estimate and subtract a long-wavelength surface (ramp) from ``field`` (2D).

    ``method="poly"`` subtracts a total-degree-``degree`` polynomial (the classic
    InSAR deramp; 1 = plane); ``method="spline"`` subtracts a NaN-aware smooth
    surface at Gaussian sigma ``scale`` (defaults to a quarter of the smaller
    axis), a high-pass that also removes gently curved ionosphere ramps. NaNs are
    preserved. The subtracted surface is exactly what :func:`fit_surface`
    produces, so ``deramp(spline) == field - fit_surface(spline)``.

    ``exclude`` (optional boolean mask) marks a signal region to leave out of the
    ramp *fit*; the ramp is still subtracted from those pixels, so the signal is
    kept but not allowed to bias the trend.
    """
    field = np.asarray(field, dtype=np.float64)
    surface = fit_surface(field, method=method, scale=scale, degree=degree,
                          exclude=exclude)
    return (field - surface).astype(np.asarray(field).dtype)


def along_track_edge_mask(valid, edge_pixels):
    """Pixels within ``edge_pixels`` of a row's first/last valid sample.

    The near- and far-range boundaries of a swath are the edges that run
    *along-track*, and they are the only ones carrying the decorrelated
    antenna-pattern fringe. On a north-up geocoded grid a near-polar orbit puts
    the track within a few degrees of grid north (measured 3-7 deg on the D126
    frame), so across-track is grid *east* and each raster row crosses the swath
    once: its first and last valid samples are the two range boundaries.

    Scanning rows rather than eroding the footprint is what keeps the trim off
    everything else that happens to border a NaN -- a coastline, a lake from
    :meth:`~nisar_tools.unwrap.UnwrappedStack.mask_water`, the azimuth ends of
    the frame -- none of which are range edges. The depth is measured in columns,
    so it is the perpendicular depth divided by cos(track tilt): 0.7% deep at 7
    degrees, which is far below one pixel of the erosion itself.

    Returns the boolean mask of pixels to *drop*. Rows are independent, so this
    decomposes freely along ``y`` but needs whole rows along ``x``.
    """
    valid = np.asarray(valid, dtype=bool)
    n = int(edge_pixels)
    drop = np.zeros_like(valid)
    if n <= 0:
        return drop
    any_valid = valid.any(axis=1)
    # argmax on a boolean row gives the first True; the same on the reversed row
    # gives the last. Rows with no data are excluded, so the degenerate argmax=0
    # never reaches the slicing below.
    first = valid.argmax(axis=1)
    last = valid.shape[1] - 1 - valid[:, ::-1].argmax(axis=1)
    cols = np.arange(valid.shape[1])
    lo = (cols[None, :] < (first + n)[:, None])
    hi = (cols[None, :] > (last - n)[:, None])
    drop[any_valid] = ((lo | hi) & valid)[any_valid]
    return drop


def mask_edges(field, edge_pixels, edges="along_track"):
    """Trim ``edge_pixels`` of swath edge off ``field`` (2D), NaN-ing the border.

    ``edges="along_track"`` (default) trims only the near/far-range boundaries --
    the edges that run along-track, and the only ones with a decorrelated fringe.
    See :func:`along_track_edge_mask`; it needs whole raster rows.

    ``edges="all"`` erodes the whole finite footprint by ``edge_pixels`` cross
    iterations (:func:`scipy.ndimage.binary_erosion`, also trimming the raster
    edge via ``border_value=0``). That reaches inward from *every* valid/NaN
    boundary, so it also eats a collar around coastlines, water-masked lakes and
    the azimuth ends of the frame; on the D126 frame that was 47.5% of everything
    it removed. Kept for the cases where the footprint really is all swath edge.

    A no-op when ``edge_pixels`` is 0.
    """
    field = np.asarray(field)
    valid = np.isfinite(field)
    if edge_pixels and edge_pixels > 0:
        if edges == "along_track":
            valid = valid & ~along_track_edge_mask(valid, edge_pixels)
        elif edges == "all":
            valid = binary_erosion(valid, iterations=int(edge_pixels),
                                   border_value=0)
        else:
            raise ValueError(
                f"edges must be 'along_track' or 'all', got {edges!r}"
            )
    return np.where(valid, field, np.asarray(np.nan, dtype=field.dtype))


def _batch_planes(func, arr, **kwargs):
    """Apply a 2D-plane ``func`` to each trailing plane of a possibly-3D ``arr``.

    Leading axes (a stacked ``pair`` dimension) are looped over so a whole block
    goes through one ``apply_ufunc`` call without the planes mixing -- the same
    batch pattern as :func:`goldstein_filter_planes`.
    """
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return func(arr, **kwargs)
    flat = arr.reshape((-1,) + arr.shape[-2:])
    out = np.empty_like(flat)
    for k in range(flat.shape[0]):
        out[k] = func(flat[k], **kwargs)
    return out.reshape(arr.shape)


def halo_planes(func, arr, depth, **kwargs):
    """Apply a 2D-plane ``func`` over ``arr`` with a ``depth``-pixel spatial halo.

    Decomposes a *local* plane kernel spatially instead of forcing one whole plane
    per dask task, which is what caps a short stack's parallelism at its pair
    count. ``depth`` has to cover the kernel's support or the chunk seams show, so
    each caller derives it from its own footprint. Only the trailing two axes are
    overlapped -- the leading stack axis never mixes.

    The global boundary is padded with NaN, i.e. "no data", which is exactly how
    these kernels already treat everything beyond the raster (``mask_edges`` erodes
    with ``border_value=0``; the surface fits give a NaN zero weight). Numpy input
    falls through to the plain whole-plane batch loop.

    A halo wider than the raster itself has no decomposition to offer, so that
    case collapses to one whole plane per task -- which is what a globally
    supported kernel (``deramp(method="spline")`` at its default scale) always
    wants anyway.
    """
    if not _is_dask(arr):
        return _batch_planes(func, arr, **kwargs)

    ndim = arr.ndim
    depth = int(depth)
    if depth >= min(arr.shape[-2:]):
        arr = arr.rechunk({ndim - 2: -1, ndim - 1: -1})
        return arr.map_blocks(
            lambda block: _batch_planes(func, block, **kwargs),
            dtype=arr.dtype, meta=arr._meta,
        )
    return arr.map_overlap(
        lambda block: _batch_planes(func, block, **kwargs),
        depth={ndim - 2: depth, ndim - 1: depth},
        boundary=np.nan, dtype=arr.dtype, meta=arr._meta,
    )


def row_planes(func, arr, **kwargs):
    """Apply a 2D-plane ``func`` that needs whole raster *rows*, chunk by chunk.

    A halo cannot serve a kernel whose support is the entire row -- a chunk in
    the middle of the raster has no way to know whether it holds the row's first
    valid sample. So ``x`` is gathered into one chunk and the decomposition runs
    along ``y`` only, where the rows are independent. No overlap is needed at
    all, which makes it exact rather than exact-to-a-halo.

    Used by ``mask_edges(edges="along_track")``. Numpy input falls through to the
    plain whole-plane batch loop.
    """
    if not _is_dask(arr):
        return _batch_planes(func, arr, **kwargs)
    ndim = arr.ndim
    arr = arr.rechunk({ndim - 1: -1})
    return arr.map_blocks(
        lambda block: _batch_planes(func, block, **kwargs),
        dtype=arr.dtype, meta=arr._meta,
    )


def remove_outliers_planes(arr, *, scale, threshold, iterations):
    return _batch_planes(remove_outliers, arr, scale=scale, threshold=threshold,
                         iterations=iterations)


def remove_outliers_depth(scale, iterations):
    """Halo needed to decompose :func:`remove_outliers` spatially.

    The Gaussian's support is nominally infinite, but past ``4*sigma`` its weight
    is ~1e-4 of the peak, so that is the working radius (the same rule
    :func:`_overlap_depth` uses for the multilook). Each iteration re-smooths the
    *previous* iteration's NaN pattern, so the error front advances one radius per
    pass and the halo has to cover all of them.
    """
    return int(math.ceil(4.0 * float(scale))) * max(1, int(iterations))


def deramp_planes(arr, *, degree, method, scale, exclude=None):
    # exclude is a single (y, x) signal mask shared across the stacked planes.
    return _batch_planes(deramp, arr, degree=degree, method=method, scale=scale,
                         exclude=exclude)


def mask_edges_planes(arr, *, edge_pixels, edges="along_track"):
    return _batch_planes(mask_edges, arr, edge_pixels=edge_pixels, edges=edges)


# -- deramp: the polynomial fit as a chunk-wise reduction ----------------------
#
# A whole-plane ``poly_surface`` builds an (npix, ncoef) float64 design matrix --
# 768 MB for a degree-2 fit on a 4000x4000 plane, and _poly_columns materialises
# the columns twice before stacking them. But a least-squares fit is a *sum* over
# pixels, so the normal-equation blocks (A^T A, A^T b) accumulate per chunk and the
# solve happens once on a 6x6 system. That decomposes the fit across chunks and
# drops peak memory to one chunk's worth of design columns.


def poly_normal_equations(field, degree, y_index, x_index, ny, nx, exclude=None):
    """``(A^T A, A^T b)`` for one block of a global polynomial fit.

    ``y_index``/``x_index`` are the block's *global* row and column indices and
    ``(ny, nx)`` the full grid shape, so every block builds the same globally
    normalized basis :func:`_poly_columns` uses. Summing the returned blocks over
    the spatial chunks reproduces the whole-plane system exactly.

    ``field`` is ``(nplane, bny, bnx)``; ``exclude`` is an optional ``(bny, bnx)``
    boolean window of the shared signal mask.
    """
    field = np.asarray(field, dtype=np.float64)
    planes = field.reshape((-1,) + field.shape[-2:])
    nplane, bny, bnx = planes.shape

    design = _poly_columns(
        ny, nx, int(degree),
        y0=int(y_index[0]), x0=int(x_index[0]), bny=bny, bnx=bnx,
    )
    ncoef = design.shape[1]
    drop = None if exclude is None else np.asarray(exclude, dtype=bool).ravel()

    ata = np.zeros((nplane, ncoef, ncoef))
    atb = np.zeros((nplane, ncoef))
    for k in range(nplane):
        values = planes[k].ravel()
        valid = np.isfinite(values)
        if drop is not None:
            valid = valid & ~drop
        if not valid.any():
            continue
        a = design[valid]
        ata[k] = a.T @ a
        atb[k] = a.T @ values[valid]
    return ata, atb


def poly_solve_parts(parts, ncoef):
    """Sum per-block normal equations and solve, per plane.

    Returns NaN coefficients for a plane the surviving pixels cannot constrain,
    matching :func:`poly_surface`'s all-NaN return in that case. ``lstsq`` (SVD)
    rather than ``solve``, so a deficient system degrades instead of raising.
    """
    ata = sum(part[0] for part in parts)
    atb = sum(part[1] for part in parts)
    out = np.full((ata.shape[0], ncoef), np.nan)
    for k in range(ata.shape[0]):
        if np.linalg.matrix_rank(ata[k]) < ncoef:
            continue  # fewer independent samples than coefficients
        out[k], *_ = np.linalg.lstsq(ata[k], atb[k], rcond=None)
    return out


def poly_subtract_block(block, coef, y_index, x_index, degree, ny, nx):
    """Subtract the fitted polynomial from one block. NaNs are preserved."""
    planes = block.reshape((-1,) + block.shape[-2:])
    bny, bnx = planes.shape[-2:]
    design = _poly_columns(
        ny, nx, int(degree),
        y0=int(y_index[0]), x0=int(x_index[0]), bny=bny, bnx=bnx,
    )
    coef = np.asarray(coef, dtype=np.float64).reshape((-1, design.shape[1]))
    surface = (design @ coef.T).T.reshape(planes.shape)
    return (planes - surface).reshape(block.shape).astype(block.dtype)


def deramp_poly_dask(arr, degree, exclude=None):
    """Chunk-wise ``deramp(method="poly")`` over a dask ``(..., y, x)`` array.

    Two lazy passes: accumulate the normal equations per chunk and solve the small
    ``ncoef x ncoef`` system once, then evaluate and subtract the surface chunk by
    chunk. Mathematically the whole-plane fit -- the fit is a sum over pixels -- but
    it never materialises a full ``(npix, ncoef)`` design matrix, and every chunk
    is an independent task.

    The global row/column indices ride along as ``arange`` arrays rather than
    coming from ``block_info``, so each block can build the globally normalized
    basis without depending on how dask reports its position.
    """
    import dask
    import dask.array as da

    ncoef = poly_ncoef(degree)
    ny, nx = arr.shape[-2:]
    work = arr if arr.ndim == 3 else arr[None]

    y_index = da.arange(ny, chunks=work.chunks[-2])
    x_index = da.arange(nx, chunks=work.chunks[-1])
    mask = None if exclude is None else np.asarray(exclude, dtype=bool)

    # Pass 1: one normal-equation contribution per chunk, grouped by stack block
    # so the solve stays aligned with the stack axis' chunking.
    y_offsets = np.cumsum((0,) + tuple(work.chunks[-2][:-1]))
    x_offsets = np.cumsum((0,) + tuple(work.chunks[-1][:-1]))
    blocks = work.to_delayed()
    parts = {}
    for (ik, iy, ix), block in np.ndenumerate(blocks):
        y0, x0 = int(y_offsets[iy]), int(x_offsets[ix])
        bny, bnx = work.chunks[-2][iy], work.chunks[-1][ix]
        window = None if mask is None else mask[y0:y0 + bny, x0:x0 + bnx]
        parts.setdefault(ik, []).append(
            dask.delayed(poly_normal_equations, pure=True)(
                block, degree, np.array([y0]), np.array([x0]), ny, nx,
                exclude=window,
            )
        )

    coef = da.concatenate(
        [
            da.from_delayed(
                dask.delayed(poly_solve_parts, pure=True)(parts[ik], ncoef),
                shape=(work.chunks[0][ik], ncoef), dtype=np.float64,
            )
            for ik in range(blocks.shape[0])
        ],
        axis=0,
    )

    # Pass 2: subtract. ``concatenate=True`` gathers the (single-chunk) coefficient
    # axis so each block sees the whole coefficient vector for its planes.
    out = da.blockwise(
        poly_subtract_block, "kyx",
        work, "kyx",
        coef, "kc",
        y_index, "y",
        x_index, "x",
        degree=degree, ny=ny, nx=nx,
        concatenate=True,
        dtype=arr.dtype,
        meta=work._meta,
    )
    return out if arr.ndim == 3 else out[0]


# -- SNAPHU tile sizing --------------------------------------------------------
#
# SNAPHU's tile geometry is a *correctness* parameter, not a performance one: the
# tiling decides where the 2pi seams fall, so two runs with different tile grids
# give different unwrapped phase. It therefore depends on the raster alone here,
# and ``nproc`` only says how many of the resulting tiles may be solved at once.
#
# The binding constraint is a hard ceiling inside SNAPHU. In tiled mode it stores
# the per-tile region labels as C ``short`` (snaphu_tile.c, ``regions`` array) and
# aborts with "Number of regions in tile exceeds max allowed" once a label passes
# LARGESHORT. A region has to reach MINREGIONSIZE pixels to be kept, so a tile can
# hold at most ~LARGESHORT * min_region_size pixels -- and fewer in practice,
# because a low-coherence pair fragments into smaller regions. Hence the tile-area
# budget below, which defaults to roughly half the ceiling.

# snaphu.h: #define LARGESHORT 32000
SNAPHU_LARGESHORT = 32000

# Half of the 3.2 M-pixel ceiling at snaphu-py's default min_region_size=100.
DEFAULT_MAX_TILE_PIXELS = 1_500_000


def snaphu_region_budget(min_region_size):
    """Largest tile area, in pixels, SNAPHU can label in tiled mode."""
    return SNAPHU_LARGESHORT * int(min_region_size)


def snaphu_tile_shape(igram_shape, ntiles, tile_overlap):
    """SNAPHU's own ``(ni, nj)`` tile size, and the size of the last tile.

    Mirrors ``CheckParams`` in snaphu_io.c: overlap *inflates* each tile rather
    than shrinking the stride, and the ceil means the final tile is up to
    ``ntiles - 1`` samples shorter than the nominal one.
    """
    nlines, linelen = int(igram_shape[0]), int(igram_shape[1])
    trow, tcol = int(ntiles[0]), int(ntiles[1])
    rov, cov = _overlap_pair(tile_overlap)

    ni = math.ceil((nlines + (trow - 1) * rov) / trow)
    nj = math.ceil((linelen + (tcol - 1) * cov) / tcol)
    last_ni = nlines - (trow - 1) * (ni - rov)
    last_nj = linelen - (tcol - 1) * (nj - cov)
    return (ni, nj), (last_ni, last_nj)


def _overlap_pair(tile_overlap):
    """Normalize a scalar-or-pair overlap the way ``snaphu.unwrap`` does."""
    try:
        rov, cov = tile_overlap
    except TypeError:
        rov = cov = tile_overlap
    return int(rov), int(cov)


def snaphu_params(igram_shape, nproc=1, *, ntiles=None, tile_overlap=None,
                  max_tile_pixels=DEFAULT_MAX_TILE_PIXELS, overlap_target=256,
                  min_region_size=100):
    """Pick SNAPHU tile geometry and overlap for a given raster.

    The grid is the coarsest one whose tiles fit inside ``max_tile_pixels`` while
    tracking the raster's aspect ratio, so it is a function of the raster and
    **not** of ``nproc`` -- changing how many processors you give SNAPHU no longer
    changes the answer it returns. Pass ``ntiles``/``tile_overlap`` to override
    either choice.

    ``nproc`` is accepted (and kept positional, as the legacy signature had it)
    only so a caller can be told when it has asked for more processors than there
    are tiles to hand them; it never feeds the geometry.

    Returns ``(ntiles, tile_overlap)``, with ``tile_overlap`` a scalar applied to
    both axes.
    """
    rows, cols = int(igram_shape[0]), int(igram_shape[1])
    if rows < 1 or cols < 1:
        raise ValueError(f"igram_shape must be positive, got {igram_shape!r}")

    if ntiles is None:
        ntiles, auto_overlap = _tile_grid(
            rows, cols, max_tile_pixels, overlap_target
        )
    else:
        ntiles = (int(ntiles[0]), int(ntiles[1]))
        if ntiles[0] < 1 or ntiles[1] < 1:
            raise ValueError(f"ntiles must be positive, got {ntiles!r}")
        auto_overlap = _tile_overlap(rows, cols, ntiles, overlap_target)

    if tile_overlap is None:
        tile_overlap = auto_overlap
    else:
        tile_overlap = int(tile_overlap)
        if tile_overlap < 0:
            raise ValueError(f"tile_overlap must be >= 0, got {tile_overlap}")

    return ntiles, tile_overlap


def _tile_grid(rows, cols, max_tile_pixels, overlap_target):
    """Coarsest ``(tiles_row, tiles_col)`` with tiles under the area budget.

    Grows whichever axis currently has the longer tile side, so the tiles stay as
    square as the raster allows -- SNAPHU's secondary (tile-assembly) network cost
    scales with total seam length, which square tiles minimise.

    The budget is measured against SNAPHU's *own* tile size, which the overlap
    inflates (see :func:`snaphu_tile_shape`) -- sizing on ``rows/trow * cols/tcol``
    instead would quietly overshoot by the overlap fraction. Growing a tile count
    shrinks both the stride and the overlap it permits, so this terminates.
    """
    budget = max(1, int(max_tile_pixels))
    trow = tcol = 1
    # Bounded by SNAPHU's own ntilerow^2 <= nlines rule, so a raster thin enough
    # to hit that keeps the coarsest tiling it can legally have.
    max_row, max_col = max(1, math.isqrt(rows)), max(1, math.isqrt(cols))
    while True:
        overlap = _tile_overlap(rows, cols, (trow, tcol), overlap_target)
        (ni, nj), _ = snaphu_tile_shape((rows, cols), (trow, tcol), overlap)
        if ni * nj <= budget:
            return (trow, tcol), overlap
        grow_row = (rows / trow) >= (cols / tcol)
        if grow_row and trow < max_row:
            trow += 1
        elif not grow_row and tcol < max_col:
            tcol += 1
        elif trow < max_row:
            trow += 1
        elif tcol < max_col:
            tcol += 1
        else:
            # Cannot subdivide further without breaking CheckParams. Hand back
            # the legal maximum; snaphu_params_check reports it properly.
            return (trow, tcol), overlap


def _tile_overlap(rows, cols, ntiles, overlap_target):
    """Overlap in pixels: capped at a quarter of the tile, and always legal.

    SNAPHU wants a generous overlap (it warns below TILEOVRLPWARNTHRESH = 400),
    but overlap inflates every tile, so it is capped at 25% of the smaller tile
    side. The floor is ``min(10, ...)``, never a bare ``max(10, ...)``: on a small
    raster a hard 10 can exceed what ``CheckParams`` permits and abort the run.

    A single tile gets 0. SNAPHU ignores the overlap there, but it says so on
    *stderr* ("only one tile--disregarding tile overlap values"), and since
    ``run_snaphu`` re-raises the whole stderr buffer as the error message, every
    stray line it prints becomes noise on top of a real failure.
    """
    if ntiles[0] == 1 and ntiles[1] == 1:
        return 0
    tile_h = max(1, rows // ntiles[0])
    tile_w = max(1, cols // ntiles[1])
    side = min(tile_h, tile_w)
    overlap = min(int(overlap_target), side // 4)
    return max(min(10, side // 4), overlap)


def snaphu_params_check(igram_shape, ntiles, tile_overlap, min_region_size=100):
    """Check a tiling against what SNAPHU will accept, before invoking it.

    Two kinds of problem, and they get different treatment because they carry
    different certainty:

    * The tile preconditions in ``CheckParams`` (snaphu_io.c) are **deterministic**
      -- SNAPHU compares the numbers and exits. Those **raise**.
    * The per-tile region ceiling is **probabilistic**. ``GrowRegions`` aborts once
      a region label passes ``LARGESHORT``, and the worst case is one region per
      ``min_region_size`` pixels, so ``tile_area / min_region_size`` bounds it --
      but the count that actually forms depends on how much the phase fragments,
      and is usually far lower. Measured: a 600x332 tile with a budget of 32000
      formed 6566 regions and unwrapped fine. So that one **warns**: refusing it
      outright would block a deliberately coarse tiling that would have worked.

    Either way the numbers are named here rather than arriving as a bare
    ``RuntimeError`` carrying only the subprocess's stderr.
    """
    nlines, linelen = int(igram_shape[0]), int(igram_shape[1])
    trow, tcol = int(ntiles[0]), int(ntiles[1])
    rov, cov = _overlap_pair(tile_overlap)

    if trow < 1 or tcol < 1:
        raise ValueError(f"ntiles must be positive, got {ntiles!r}")
    if rov < 0 or cov < 0:
        raise ValueError(f"tile_overlap must be >= 0, got {tile_overlap!r}")
    if trow == 1 and tcol == 1:
        # Single-tile mode: SNAPHU ignores the tiling entirely, and the region
        # labels are never squeezed into a short, so no ceiling applies.
        return

    prefix = (
        f"SNAPHU tiling is invalid for a {nlines}x{linelen} raster with "
        f"ntiles={(trow, tcol)}, tile_overlap={tile_overlap}: "
    )
    if trow + rov > nlines:
        raise ValueError(
            prefix + f"ntilerow + rowovrlp = {trow + rov} exceeds nlines = {nlines}"
        )
    if tcol + cov > linelen:
        raise ValueError(
            prefix + f"ntilecol + colovrlp = {tcol + cov} exceeds linelen = {linelen}"
        )
    if trow * trow > nlines:
        raise ValueError(
            prefix + f"ntilerow^2 = {trow * trow} exceeds nlines = {nlines}; "
            f"use at most {int(math.isqrt(nlines))} tile rows"
        )
    if tcol * tcol > linelen:
        raise ValueError(
            prefix + f"ntilecol^2 = {tcol * tcol} exceeds linelen = {linelen}; "
            f"use at most {int(math.isqrt(linelen))} tile columns"
        )

    (ni, nj), (last_ni, last_nj) = snaphu_tile_shape(
        (nlines, linelen), (trow, tcol), (rov, cov)
    )
    if int(min_region_size) > last_ni * last_nj:
        raise ValueError(
            prefix + f"the final tile is {last_ni}x{last_nj} = {last_ni * last_nj} "
            f"pixels, smaller than min_region_size = {int(min_region_size)}"
        )

    budget = snaphu_region_budget(min_region_size)
    if ni * nj > budget:
        warnings.warn(
            f"SNAPHU tiling for a {nlines}x{linelen} raster with "
            f"ntiles={(trow, tcol)}, tile_overlap={tile_overlap} gives "
            f"{ni}x{nj} = {ni * nj}-pixel tiles, over the per-tile region budget "
            f"of {budget} ({SNAPHU_LARGESHORT} regions x min_region_size="
            f"{int(min_region_size)}). If the phase fragments that far SNAPHU will "
            f"abort with 'Number of regions in tile exceeds max allowed'; a noisy "
            f"or low-coherence pair is the one that does. Lower max_tile_pixels, "
            f"pass a finer ntiles=, or raise min_region_size.",
            RuntimeWarning,
            stacklevel=2,
        )


def snaphu_nlooks(looks_az, looks_rg, spacing_az, spacing_rg, res_az, res_rg):
    """Equivalent number of independent looks for SNAPHU. Verbatim.

    From SNAPHU-py: ``n_e = k_r k_a (d_r d_a) / (rho_r rho_a)`` where ``k`` are
    the looks, ``d`` the single-look sample spacing, and ``rho`` the
    resolution, in range and azimuth.
    """
    n_e = np.abs(
        (looks_az * looks_rg) * (spacing_az / res_az) * (spacing_rg / res_rg)
    )
    # SNAPHU requires an integer number of looks.
    return round(n_e)
