"""Dense pixel offsets between two radar-geometry SLCs: ISCE3's PyCuAmpcor, in numpy.

This is the cross-correlator NISAR's ROFF/GOFF products are made with: ISCE3 v0.25.16
``nisar.workflows.offsets_product`` drives ``isce3.cuda.matchtemplate.PyCuAmpcor`` (GPU only --
"Currently ISCE3 supports only GPU cross-correlation") on the reference RSLC and the
geometry-coregistered secondary. Each window follows ``cuAmpcorChunk::run`` step for step:

1. **Raw stage.** Amplitudes; the reference window's mean removed (the secondary's is not --
   the normalisation takes care of it). FFT cross-correlation of the reference window over the
   ``window + 2*half_search`` secondary chip (``cuFreqCorrelator``: template zero-padded at the
   end, ``conj(T) * S``, the top-left ``2*half_search + 1`` lags kept). Normalised
   cross-correlation, ``corr / sqrt((S2 - S1**2/N) * sum(t**2) + FLT_EPSILON)`` with ``S1``/``S2``
   the secondary's sums over each lag's window (``cuCorrNormalize``). Integer peak (first maximum
   in row-major order). Offset variance from the surface's curvature at that peak
   (``cuEstimateVariance``; 99 where the peak touches the edge or the curvature is below 1e-2).
   SNR = peak**2 / mean of corr**2 over a ``correlation_statistics_zoom`` window around the peak,
   the peak excluded (``cuEstimateSnr``; samples outside the surface do not count).
2. **Re-centre.** A secondary chip of ``window + 2*h`` around the peak, ``h =
   correlation_surface_zoom / (2 * slc_oversampling_factor)``, its start clamped to the search
   window (``dev_adjustOffset``; the clamp is remembered as the *shift*).
3. **Oversampled stage.** Both chips deramped (``deramping_method="complex"``: the mean phase step
   along the chosen axis, ``angle(sum z * conj(z_next))``, removed as a linear ramp in double
   precision), FFT-oversampled by ``slc_oversampling_factor`` (``cuOverSamplerC2C``: inverse FFT,
   1/N, the quadrant copy of ``cuArraysPaddingMany`` -- the Nyquist bin goes to the negative
   side, unsplit -- forward FFT), amplitudes, the reference's mean removed, correlated and
   normalised again; the last row and column dropped to make the surface even.
4. **Sub-pixel.** That surface oversampled by ``correlation_surface_oversampling_factor``
   (``"sinc"``: ``cuSincOverSamplerR2R`` -- an 8-tap Hann-windowed sinc of bandwidth 0.75,
   tabulated at 1/4096 sample, normalised by its weight sum, evaluated only within +-2 surface
   samples of the re-centred peak, the rest left 0; ``"fft"``: ``cuOverSamplerR2R``). Argmax.
5. ``offset = start + argmax / (surface_factor * slc_factor) - half_search`` per axis; the
   oversampled maximum is the ``correlationSurfacePeak``.

**Departures.** float32 FFTs as cuFFT, but numpy's summation order: not bit-identical to the GPU,
agreement is to rounding. The separable sinc is evaluated as two small matrix products
(identical arithmetic up to rounding -- the CUDA kernel's 2-D sum and normalisation factor into
row and column weights). ``cuSincInterpolation_kernel`` wraps the *row* index with the
*column* count (``inx -= inNY``); the surfaces here are square, so it cannot differ.

Axes follow ISCE3: "down" = azimuth = rows, "across" = range = columns.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import scipy.fft as sfft

FLT_EPSILON = np.float32(1.1920929e-07)
NTHREADS = 512        # cudaUtil.h

# cuSincOverSamplerR2R constants
_SINC_WINDOW = 2
_SINC_BETA = 0.75
_SINC_RELFILTLEN = 6.0
_SINC_DECFACTOR = 4096


@dataclass(frozen=True)
class AmpcorParams:
    """One offsets layer, as ``offsets_product``'s runconfig gives it (NISAR GOFF defaults)."""

    window_range: int = 64
    window_azimuth: int = 64
    half_search_range: int = 16
    half_search_azimuth: int = 16
    slc_oversampling_factor: int = 2
    deramping_method: str = "complex"          # "complex" | "magnitude" | None
    deramping_axis: str = "azimuth"            # "azimuth" | "range" | "both"
    correlation_statistics_zoom: int = 21
    correlation_surface_zoom: int = 8
    correlation_surface_oversampling_factor: int = 32
    correlation_surface_oversampling_method: str = "sinc"   # "sinc" | "fft"

    @property
    def zoom_window(self):
        """``zoomWindowSize`` after ``setupParameters``."""
        actual = min(self.half_search_range, self.half_search_azimuth) * 2 * self.slc_oversampling_factor
        return min(self.correlation_surface_zoom, actual)

    @property
    def half_zoom_raw(self):
        return self.zoom_window // (2 * self.slc_oversampling_factor)


@lru_cache(maxsize=1)
def _sinc_filter_cached():
    return sinc_filter()


def sinc_filter():
    """``cuSetupSincKernel``: the 8-tap Hann-windowed sinc table, float32, length 8*4096 + 1."""
    intplength = int(_SINC_RELFILTLEN / _SINC_BETA + 0.5)
    ncoef = intplength * _SINC_DECFACTOR
    i = np.arange(ncoef + 1, dtype=np.float64)
    soff = (ncoef - 1.0) / 2.0
    wgthgt = 0.5                                # (1 - pedestal) / 2, pedestal 0
    wa = (i - soff).astype(np.float32)
    wgt = (1.0 - wgthgt) + wgthgt * np.cos(np.float32(np.pi) * wa * np.float32(1.0 / soff))
    s = wa * np.float32(_SINC_BETA) * np.float32(1.0 / _SINC_DECFACTOR) * np.float32(3.14159265359)
    fct = np.where(s != 0, np.sin(s) / np.where(s != 0, s, 1), 1.0)
    return (fct * wgt).astype(np.float32), intplength


@lru_cache(maxsize=64)
def _sinc_table(n_in, covs, rows):
    return _sinc_matrix_impl(n_in, covs, np.asarray(rows))


def _sinc_matrix(n_in, covs, rows):
    return _sinc_table(int(n_in), int(covs), tuple(int(r) for r in rows))


def _sinc_matrix_impl(n_in, covs, rows):
    """Normalised sinc weights mapping an ``n_in`` surface onto output indices ``rows``
    (``cuSincInterpolation_kernel``'s 1-D factor): ``(len(rows), n_in)``."""
    filt, ntap = _sinc_filter_cached()
    rows = np.asarray(rows)
    r_out = rows.astype(np.float32) / np.float32(covs)
    i_out = r_out.astype(np.int64)
    i_frac = ((r_out - i_out) * _SINC_DECFACTOR).astype(np.int64)
    K = np.zeros((rows.size, n_in), np.float64)
    for i in range(ntap):
        src = i_out - i + ntap // 2
        src = np.where(src < 0, src + n_in, src)
        src = np.where(src >= n_in, src - n_in, src)
        np.add.at(K, (np.arange(rows.size), src), filt[i * _SINC_DECFACTOR + i_frac])
    return K / K.sum(axis=1, keepdims=True)


def _fft_oversample(z, factor, workers):
    """``cuOverSamplerC2C``: ``fft(quadrant_pad(ifft(z)))`` per image, ``(N, h, w)`` -> ``(N, h*f, w*f)``."""
    n, h1, w1 = z.shape
    h2, w2 = h1 * factor, w1 * factor
    F = sfft.ifft2(z, axes=(-2, -1), workers=workers)
    G = np.zeros((n, h2, w2), F.dtype)
    a, b = h1 // 2, w1 // 2
    G[:, :a, :b] = F[:, :a, :b]
    G[:, h2 - a:, :b] = F[:, h1 - a:, :b]
    G[:, :a, w2 - b:] = F[:, :a, w1 - b:]
    G[:, h2 - a:, w2 - b:] = F[:, h1 - a:, w1 - b:]
    return sfft.fft2(G, axes=(-2, -1), workers=workers)


def _fft_oversample_real(s, factor, workers):
    """``cuOverSamplerR2R`` on a batch of real surfaces (the ``"fft"`` method)."""
    return _fft_oversample(s.astype(np.complex64), factor, workers).real.astype(np.float32)


def _deramp(z, method, axis):
    """``cuDeramp``: remove the mean phase step along ``axis`` (0 azimuth, 1 range, 2 both).

    The ramp ``exp(i (row*phase_x + col*phase_y))`` factors into a row and a column term,
    so it is applied as two broadcast multiplies (CUDA rotates each sample once, in double;
    the difference is float32 rounding).
    """
    if method != "complex":
        return z
    n, h, w = z.shape
    out = z
    if axis != 1:      # azimuth / rows: angle(sum z[r] * conj(z[r + 1]))
        sx = np.einsum("nij,nij->n", z[:, :-1, :], np.conj(z[:, 1:, :]))
        ramp = np.exp(1j * np.arange(h)[None, :] * np.angle(sx)[:, None]).astype(np.complex64)
        out = out * ramp[:, :, None]
    if axis != 0:      # range / columns
        sy = np.einsum("nij,nij->n", z[:, :, :-1], np.conj(z[:, :, 1:]))
        ramp = np.exp(1j * np.arange(w)[None, :] * np.angle(sy)[:, None]).astype(np.complex64)
        out = out * ramp[:, None, :]
    return out


def _xcorr(t, s, workers):
    """``cuFreqCorrelator``: correlation of templates ``t`` (N, ht, wt) over images ``s`` (N, hs, ws).

    ``corr[u, v] = sum t[i, j] s[i + u, j + v]`` for the ``(hs - ht + 1, ws - wt + 1)`` lags that
    keep the template inside the image. Those lags never wrap, so the transform may be any size
    of at least the image: a fast length is used instead of cuFFT's ``(hs, ws)``.
    """
    n, ht, wt = t.shape
    _, hs, ws = s.shape
    fh, fw = sfft.next_fast_len(hs, real=True), sfft.next_fast_len(ws, real=True)
    T = sfft.rfft2(t, s=(fh, fw), axes=(-2, -1), workers=workers)
    S = sfft.rfft2(s, s=(fh, fw), axes=(-2, -1), workers=workers)
    np.conj(T, out=T)
    T *= S
    c = sfft.irfft2(T, s=(fh, fw), axes=(-2, -1), workers=workers)
    return c[:, :hs - ht + 1, :ws - wt + 1].astype(np.float32)


def _window_sums(a, ht, wt, nu, nv):
    """Sums of ``a`` (N, H, W) over every ``ht x wt`` window at the ``nu x nv`` leading lags."""
    c = np.cumsum(a, axis=1, dtype=np.float64)
    r = c[:, ht - 1:ht - 1 + nu, :].copy()
    r[:, 1:, :] -= c[:, :nu - 1, :]
    c = np.cumsum(r, axis=2)
    w = c[:, :, wt - 1:wt - 1 + nv].copy()
    w[:, :, 1:] -= c[:, :, :nv - 1]
    return w


def _normalise(corr, t, s):
    """``cuCorrNormalize``: normalised cross-correlation (template mean already removed)."""
    n, ht, wt = t.shape
    nu, nv = corr.shape[1:]
    t2 = np.einsum("nij,nij->n", t, t, dtype=np.float64)
    s1 = _window_sums(s, ht, wt, nu, nv)
    s2 = _window_sums(np.square(s, dtype=np.float64), ht, wt, nu, nv)
    norm2 = (s2 - s1 * s1 / (ht * wt)) * t2[:, None, None]
    return (corr / np.sqrt(norm2 + FLT_EPSILON)).astype(np.float32)


def _argmax_cuda(values, gids, block=NTHREADS):
    """``cudaKernel_maxloc2D``'s choice among ``values`` (N, K) at flat indices ``gids``.

    Each of ``block`` threads keeps the first strict maximum of the flat indices congruent to
    its id; the tree reduction keeps the lower thread on a tie. So the winner is the maximum
    whose ``gid % block`` is smallest, then the smallest ``gid``. NaN never wins (every
    comparison with it is false). Returns ``(position in K, value)`` per row.
    """
    v = values
    if np.isnan(v).any():
        v = np.where(np.isnan(v), -np.inf, v)
    k = np.argmax(v, axis=1)
    vmax = v[np.arange(v.shape[0]), k]
    tied = np.flatnonzero(np.count_nonzero(v == vmax[:, None], axis=1) > 1)
    if tied.size:
        g = np.broadcast_to(gids, v.shape)[tied]
        key = np.where(v[tied] == vmax[tied, None], (g % block) * (g.max() + 1) + g,
                       np.iinfo(np.int64).max)
        k[tied] = np.argmin(key, axis=1)
    return k, vmax.astype(np.float32)


def _argmax2d(a):
    """``cuArraysMaxloc2D`` over a batch of 2-D surfaces: ``(row, col, value)`` per image."""
    n, h, w = a.shape
    k, v = _argmax_cuda(a.reshape(n, -1), np.arange(h * w))
    return k // w, k % w, v


def _variance(corr, px, py, peak, template_size):
    """``cuEstimateVariance``: (cov_down, cov_across, cov_cross) in raw pixels**2."""
    n, nx, ny = corr.shape
    out = np.full((n, 3), [99.0, 99.0, 0.0], np.float32)
    ok = (px - 1 >= 0) & (py - 1 >= 0) & (px + 1 < nx) & (py + 1 < ny)
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return out
    c = corr[idx].astype(np.float32)
    x, y = px[idx], py[idx]
    g = lambda dx, dy: c[np.arange(idx.size), x + dx, y + dy]    # noqa: E731
    # ``- (a + b - 2.0*c)``: the float sum a + b, then a double subtraction, rounded on assignment
    dxx = (-((g(1, 0) + g(-1, 0)).astype(np.float64) - 2.0 * g(0, 0))).astype(np.float32)
    dyy = (-((g(0, 1) + g(0, -1)).astype(np.float64) - 2.0 * g(0, 0))).astype(np.float32)
    dxy = (g(1, 1) + g(-1, -1) - g(1, -1) - g(-1, 1)) * np.float32(0.25)
    n2 = np.maximum(np.float32(1.0) - peak[idx], np.float32(0.0))
    dxx, dyy, dxy = dxx * template_size, dyy * template_size, dxy * template_size
    n4 = n2 * n2
    n2 = n2 * 2
    n4 = n4 * np.float32(0.5) * template_size
    u = dxy * dxy - dxx * dyy
    u2 = u * u
    good = np.abs(u) >= 1e-2
    with np.errstate(divide="ignore", invalid="ignore"):
        cxx = (-n2 * u * dyy + n4 * (dyy * dyy + dxy * dxy)) / u2
        cyy = (-n2 * u * dxx + n4 * (dxx * dxx + dxy * dxy)) / u2
        cxy = (n2 * u * dxy - n4 * (dxx + dyy) * dxy) / u2
    sel = idx[good]
    out[sel, 0], out[sel, 1], out[sel, 2] = cxx[good], cyy[good], cxy[good]
    return out


def _snr(corr, px, py, peak, zoom):
    """``cuArraysCopyExtractCorr`` + ``cuArraysSumCorr`` + ``cuEstimateSnr``."""
    n, nx, ny = corr.shape
    zx, zy = min(zoom, nx), min(zoom, ny)
    ox = np.arange(zx)[None, :] + px[:, None] - zx // 2
    oy = np.arange(zy)[None, :] + py[:, None] - zy // 2
    vx = (ox >= 0) & (ox < nx)
    vy = (oy >= 0) & (oy < ny)
    cx, cy = np.clip(ox, 0, nx - 1), np.clip(oy, 0, ny - 1)
    vals = corr[np.arange(n)[:, None, None], cx[:, :, None], cy[:, None, :]].astype(np.float32)
    valid = vx[:, :, None] & vy[:, None, :]
    s = np.sum(np.where(valid, vals * vals, 0.0), axis=(1, 2), dtype=np.float32)
    count = valid.sum(axis=(1, 2))
    mean = (s - peak * peak) / (count - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (peak * peak / mean).astype(np.float32)


def _adjust(maxloc, old_range, new_range):
    """``dev_adjustOffset``: zoom-chip start inside the search window, and the clamp shift."""
    start = maxloc - new_range
    rbound = 2 * (old_range - new_range)
    shift = np.where(start < 0, -start, np.where(start > rbound, start - rbound, 0))
    start = np.clip(start, 0, rbound)
    return start, shift


def ampcor_windows(ref, sec, params=AmpcorParams(), workers=1):
    """Offsets for a batch of windows.

    ``ref`` is ``(N, window_azimuth, window_range)`` complex reference windows and ``sec``
    ``(N, window_azimuth + 2*half_search_azimuth, window_range + 2*half_search_range)`` complex
    secondary chips, the chip's ``[half_search, half_search]`` sample aligned with the window's
    ``[0, 0]`` (zero gross offset). Returns a dict of per-window arrays: ``down`` and ``across``
    offsets (secondary minus reference, raw pixels), ``snr``, ``cov`` ``(N, 3)`` (down, across,
    cross; raw pixels**2) and ``peak`` (the oversampled correlation maximum).
    """
    p = params
    ref = np.asarray(ref, np.complex64)
    sec = np.asarray(sec, np.complex64)
    n, ht, wt = ref.shape
    hsd, hsa = p.half_search_azimuth, p.half_search_range
    assert sec.shape == (n, ht + 2 * hsd, wt + 2 * hsa), (sec.shape, ref.shape)
    axis = {"azimuth": 0, "range": 1, "both": 2}[p.deramping_axis]
    ovs = int(p.slc_oversampling_factor)

    # 1. raw stage
    r_ref = np.abs(ref).astype(np.float32)
    r_ref -= r_ref.mean(axis=(1, 2), keepdims=True, dtype=np.float32)
    r_sec = np.abs(sec).astype(np.float32)
    corr = _normalise(_xcorr(r_ref, r_sec, workers), r_ref, r_sec)
    px, py, peak_raw = _argmax2d(corr)
    cov = _variance(corr, px, py, peak_raw, np.float32(ht * wt))
    snr = _snr(corr, px, py, peak_raw, p.correlation_statistics_zoom)

    # 2. re-centre
    h = p.half_zoom_raw
    sx, shx = _adjust(px, hsd, h)
    sy, shy = _adjust(py, hsa, h)

    # 3. oversampled stage
    o_ref = np.abs(_fft_oversample(_deramp(ref, p.deramping_method, axis), ovs, workers)).astype(np.float32)
    o_ref -= o_ref.mean(axis=(1, 2), keepdims=True, dtype=np.float32)
    zh, zw = ht + 2 * h, wt + 2 * h
    rows = sx[:, None] + np.arange(zh)[None, :]
    cols = sy[:, None] + np.arange(zw)[None, :]
    zoom = sec[np.arange(n)[:, None, None], rows[:, :, None], cols[:, None, :]]
    o_sec = np.abs(_fft_oversample(_deramp(zoom, p.deramping_method, axis), ovs, workers)).astype(np.float32)
    corr_o = _normalise(_xcorr(o_ref, o_sec, workers), o_ref, o_sec)[:, :-1, :-1]

    # 4. sub-pixel
    covs = int(p.correlation_surface_oversampling_factor)
    m = corr_o.shape[1]
    out_n = p.zoom_window * covs
    if p.correlation_surface_oversampling_method == "sinc":
        # cuSincOverSamplerR2R evaluates only +-2 surface samples around the re-centred peak;
        # the rest of its (out_n, out_n) output stays 0 and still takes part in the argmax.
        rng = _SINC_WINDOW * covs
        start = out_n // 2 - rng
        size = 2 * rng + 1
        factor = covs * ovs
        qx = np.zeros(n, np.int64)
        qy = np.zeros(n, np.int64)
        peak = np.zeros(n, np.float32)
        for key in set(zip(shx.tolist(), shy.tolist())):
            sel = np.flatnonzero((shx == key[0]) & (shy == key[1]))
            ox = (np.arange(size) + start + key[0] * factor) % out_n
            oy = (np.arange(size) + start + key[1] * factor) % out_n
            Kx, Ky = _sinc_matrix(m, covs, ox), _sinc_matrix(m, covs, oy)
            surf = (Kx @ corr_o[sel].astype(np.float64) @ Ky.T).astype(np.float32)
            gids = (ox[:, None] * out_n + oy[None, :]).ravel()
            k, v = _argmax_cuda(surf.reshape(sel.size, -1), gids)
            g = gids[k]
            nonpos = v <= 0
            if nonpos.any():
                full = np.zeros((nonpos.sum(), out_n, out_n), np.float32)
                full[:, ox[:, None], oy[None, :]] = surf[nonpos]
                fk, fv = _argmax_cuda(full.reshape(nonpos.sum(), -1), np.arange(out_n * out_n))
                g[nonpos], v[nonpos] = fk, fv
            qx[sel], qy[sel], peak[sel] = g // out_n, g % out_n, v
    else:
        surf = _fft_oversample_real(corr_o, covs, workers)
        qx, qy, peak = _argmax2d(surf)

    # 5. combine
    osr = np.float32(1.0 / (covs * ovs))
    down = (osr * qx + sx - hsd).astype(np.float32)
    across = (osr * qy + sy - hsa).astype(np.float32)
    return dict(down=down, across=across, snr=snr, cov=cov, peak=peak.astype(np.float32),
                raw_down=(px - hsd).astype(np.int32), raw_across=(py - hsa).astype(np.int32))


def window_grid(length, width, layers, margin=250, skip_azimuth=15, skip_range=15,
                start_pixel_azimuth=None, start_pixel_range=None, gross_offset_azimuth=0,
                gross_offset_range=0):
    """``get_start_pixels`` + ``get_offsets_shape``: the window lattice common to all ``layers``.

    Returns ``(az_start, rg_start, n_down, n_across)``: window ``(i, j)`` covers reference rows
    ``az_start + i*skip_azimuth`` onward and columns ``rg_start + j*skip_range`` onward.
    """
    margin = max(margin, abs(gross_offset_range), abs(gross_offset_azimuth))
    hs_az = min(L.half_search_azimuth for L in layers)
    hs_rg = min(L.half_search_range for L in layers)
    win_az = min(L.window_azimuth for L in layers)
    win_rg = min(L.window_range for L in layers)
    az_start = margin + hs_az if start_pixel_azimuth is None else start_pixel_azimuth
    rg_start = margin + hs_rg if start_pixel_range is None else start_pixel_range
    n_down = (length - (2 * margin + 2 * hs_az + win_az)) // skip_azimuth
    n_across = (width - (2 * margin + 2 * hs_rg + win_rg)) // skip_range
    return az_start, rg_start, n_down, n_across
