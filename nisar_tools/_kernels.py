"""Numeric kernels for multilooking, interferogram formation, and SNAPHU sizing.

These are the validated numerics ported verbatim from the original procedural
module (kept as ``tests/legacy_reference.py``, the equivalence-test oracle).
The ``*_dask`` helpers wrap the same math so it can
run lazily over chunked arrays without ever materializing a full-resolution
stack. The plain-numpy versions are kept as the reference implementation that
the dask path is equivalence-tested against.
"""

import math

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter

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


def igram_coherence(c1, c2, max_x, max_y, looks, downsample, convolution):
    """Form a multilooked interferogram and its coherence. Verbatim formula.

    Works on either numpy or dask arrays: the only backend-specific step is
    the multilook, which is dispatched on the array type. Every other
    operation (``*``, ``conj``, ``abs``, ``sqrt``, ``nan_to_num``, ``clip``)
    is supported identically by numpy and dask.

    Returns ``(interferogram, coherence)`` as ``(complex, float32)``.
    """
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


def downsampled_coords(coords, looks, max_n):
    """Coordinate values for a downsampled axis (block-mean of ``looks``)."""
    return coords[:max_n].reshape(-1, looks).mean(axis=1)


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
