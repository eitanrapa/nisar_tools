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

import numpy as np
from scipy.ndimage import (
    gaussian_filter,
    gaussian_filter1d,
    uniform_filter,
    uniform_filter1d,
)

VALID_CONVOLUTIONS = ("Uniform", "Gaussian")


def _is_dask(arr):
    """True if ``arr`` is a dask array, without importing dask eagerly."""
    return type(arr).__module__.split(".")[0] == "dask"


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
    return np.outer(prof_y, prof_x)


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
    sum_interf = _ml(np.where(valid, c1 * np.conj(c2), 0.0))
    sum_int1 = _ml(np.where(valid, np.abs(c1) ** 2, 0.0))
    sum_int2 = _ml(np.where(valid, np.abs(c2) ** 2, 0.0))
    sum_mask = _ml(mask)

    # ``sum_mask`` carries both effects we care about: how much of the
    # footprint was valid, and how much of it fell outside the raster. Only the
    # first is a data boundary, so divide the second out before thresholding.
    ny, nx = c1.shape[-2:]
    response = boundary_response(
        ny, nx, max_x, max_y, looks, downsample, convolution
    )
    # The ``> 0`` term also covers min_valid_fraction=0, where a pixel with no
    # valid contribution at all would otherwise be kept as a meaningless zero.
    keep = (sum_mask > 0.0) & ((sum_mask / response) >= min_valid_fraction)

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
                     coherence=None):
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
    ny, nx = igram.shape

    if adaptive:
        coh = np.clip(np.nan_to_num(np.asarray(coherence, float), nan=0.0), 0.0, 1.0)
        if coh.shape != igram.shape:
            raise ValueError("coherence must have the same shape as igram")

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
    w2 = window ** 2
    window_sum = window.sum()

    out = np.zeros((ny, nx), dtype=np.complex128)
    wsum = np.zeros((ny, nx), dtype=np.float64)

    for y0 in _patch_origins(ny, ps, step):
        for x0 in _patch_origins(nx, ps, step):
            sl = (slice(y0, y0 + ps), slice(x0, x0 + ps))
            patch = window * work[sl]
            spec = np.fft.fft2(patch)
            psd = uniform_filter(np.abs(spec), size=psd_smooth, mode="wrap")
            if adaptive:
                # Baran: strength = 1 - mean coherence over the (windowed) patch.
                a = 1.0 - float((window * coh[sl]).sum() / window_sum)
            else:
                a = alpha
            # 0**0 == 1, so alpha=0 gives H==1 (identity) even at zero-power bins.
            response = np.power(psd / (psd.max() + 1e-20), a)
            filtered = np.fft.ifft2(spec * response)
            out[sl] += window * filtered
            wsum[sl] += w2

    # wsum > 0 everywhere (Hamming is strictly positive and the patches cover the
    # whole raster); the guard is belt-and-braces against a degenerate tiling.
    out /= np.where(wsum > 0.0, wsum, 1.0)
    out = out.astype(dtype)
    out[nan_mask] = np.nan
    return out


def goldstein_filter_planes(arr, coherence=None, *, alpha=0.5, patch_size=32,
                            overlap=0.75, psd_smooth=3):
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
    if arr.ndim == 2:
        return goldstein_filter(arr, alpha, patch_size, overlap, psd_smooth,
                                coherence=coh)

    flat = arr.reshape((-1,) + arr.shape[-2:])
    coh_flat = None if coh is None else coh.reshape((-1,) + coh.shape[-2:])
    out = np.empty_like(flat)
    for k in range(flat.shape[0]):
        ck = None if coh_flat is None else coh_flat[k]
        out[k] = goldstein_filter(flat[k], alpha, patch_size, overlap, psd_smooth,
                                  coherence=ck)
    return out.reshape(arr.shape)


def snaphu_params(igram_shape, nproc, overlap_target=256):
    """Pick SNAPHU tile geometry and overlap for a given raster + processors.

    Ported verbatim from ``_calculate_snaphu_params``.
    """
    rows, cols = igram_shape
    aspect_ratio = rows / cols

    # Ideal number of columns given nproc and aspect ratio:
    # tiles_col^2 * aspect_ratio = nproc
    tiles_col = max(1, int(round(math.sqrt(nproc / aspect_ratio))))
    tiles_row = max(1, int(round(tiles_col * aspect_ratio)))

    # Ensure total tiles >= nproc so no processor sits idle.
    while (tiles_row * tiles_col) < nproc:
        if (tiles_row / tiles_col) < aspect_ratio:
            tiles_row += 1
        else:
            tiles_col += 1

    ntiles = (tiles_row, tiles_col)

    tile_h = rows // tiles_row
    tile_w = cols // tiles_col

    # Overlap should not exceed 25% of the tile size.
    max_overlap = int(min(tile_h, tile_w) * 0.25)
    tile_overlap = min(overlap_target, max_overlap)

    # Hard floor for tiny arrays.
    tile_overlap = max(10, tile_overlap)

    return ntiles, tile_overlap


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
