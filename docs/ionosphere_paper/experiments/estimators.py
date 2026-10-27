"""The split-spectrum estimators the paper compares, all on one set of map cells.

Every estimator takes the same inputs, ``(y, x)`` arrays on the cells:

    phi_a          the main-band phase, unwrapped (the caller's fault-aware unwrapping averaged onto the cells)
    psi_a, psi_b   the main- and side-band phases, wrapped (the angles of the cells' complex sums)
    coh_a, coh_b   the cells' coherences
    use            the cells that enter the estimate (coherence, coverage, the fault zone left out)
    sigma          each cell's standard deviation of the raw dispersive estimate (the filter's weights)

and returns the filtered screen (the dispersive phase at f_A, rad) and the raw estimate. They differ only in what they
unwrap and in the whole-cycle corrections they apply; the final filter is the same for all of them
(`nisar_tools.split_spectrum.ionosphere_filter`, called with ``filter_kw``). With W the wrap and U an unwrapping:

    ours            d = U[W(psi_B - (f_B/f_A) phi_A)] / k                 -- the scaled main band out BEFORE unwrapping
    diff_smooth     d = c0 phi_A + c1 U[W(psi_A - psi_B)], U = ours's     -- the control: same unwrapper, other order
    diff_snaphu     the same with U = SNAPHU on the cells                 -- Wegmueller et al. 2018 Method 1, ISCE3
                                                                             main_diff_ms_band without corrections
    diff_trivial    the double difference taken as wrapped about its circular mean (Wegmueller's "lowest ambiguity")
    diff_isce3      diff_snaphu + ISCE3's two cycle-correction passes (unwrap_correction: True, as the PR GUNWs ran)
    side_snaphu     d = a phi_A + b U[psi_B], U = SNAPHU                  -- the classic two-band estimate
    side_isce2      side_snaphu + ISCE2's relative correction (B's cycles against a quadratic of phi_A - phi_B)
    side_isce3      side_snaphu + ISCE3's two cycle-correction passes (main_side_band)

with k = f_A/f_B - f_B/f_A (-0.0861), (c0, c1) = (0.511, 11.61) and (a, b) = (12.12, -11.61) for NISAR's 40 + 5 MHz
(1239 / 1293.5 MHz). All three forms are the same number when every unwrapping is right; they differ in what the
unwrapper sees. ISCE3's passes are ported from its source (isce3 develop, 2026-10-21):
`atmosphere/ionosphere_filter.unwrapping_correction_with_filter` (each unwrapped phase rounded against its own
truncated-Gaussian low-pass, one iteration) and `atmosphere/main_band_estimation.compute_unwrapp_error_*` (rounded
against the first pass's filtered dispersive and non-dispersive phases), with the dispersive filter's kernel (sigma 4 x
37 B-grid pixels, ~2.0 x 2.6 km, truncated at 1.75 and 1.49 sigma).
"""

import warnings

import numpy as np
from scipy import ndimage

from nisar_tools import split_spectrum as ss

TWO_PI = 2.0 * np.pi


def _nanmedian3(x):
    """``generic_filter(x, np.nanmedian, size=3, mode="constant", cval=nan)``, bit for bit, ~50x faster (the 9
    neighbours stacked)."""
    p = np.pad(np.asarray(x, float), 1, mode="constant", constant_values=np.nan)
    st = np.stack([p[i:i + x.shape[0], j:j + x.shape[1]] for i in range(3) for j in range(3)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(st, axis=0)


def _guide(x, sigma):
    med = _nanmedian3(x)
    ok = np.isfinite(med)
    num = ndimage.gaussian_filter(np.where(ok, med, 0.0), sigma)
    den = ndimage.gaussian_filter(ok.astype(float), sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 1e-6, num / den, np.nan)


def _plane_guide(x, sigma):
    med = _nanmedian3(x)
    return ss._plane_fill(med, np.isfinite(med).astype(float), sigma)


# the filter's two 3 x 3 nanmedians, replaced in this process by the identical stacked version (12 s -> 0.25 s per call
# on 400 x 400 cells; checked equal, NaNs included). The library on disk is unchanged.
ss._guide, ss._plane_guide = _guide, _plane_guide
METHODS = ("ours", "diff_smooth", "diff_snaphu", "diff_trivial", "diff_isce3", "side_snaphu", "side_isce2", "side_isce3")

#: v6, the screen recipe every NISAR track was rebuilt with (2026-10-14)
FILTER_V6 = dict(sigma_m=(2000.0, 2600.0), fill="plane", fill_m=10000.0, fill_reach_m=12000.0)
#: ISCE3's correction passes filter with ISCE3's own fill (the guide)
FILTER_ISCE3 = dict(sigma_m=(2000.0, 2600.0), fill="guide")


def coefficients(f_a, f_b):
    """The three forms' factors: ``r = k d``; ``d = c0 phi_A + c1 (phi_A - phi_B)``; ``d = a phi_A + b phi_B``."""
    k = ss.dispersive_factor(f_a, f_b)
    c0, c1 = ss.main_diff_coefficients(f_a, f_b)
    return dict(k=k, c0=c0, c1=c1, a=-(f_b / f_a) / k, b=1.0 / k)


def raw_sigma(coh_a, coh_b, n_frac_a, n_frac_b, f_a, f_b, cell, resolution_a=(6.2, 5.3), resolution_b=(50.0, 5.3)):
    """ISCE3's per-cell sigma of the raw dispersive estimate, as `split_spectrum_screen` computes it."""
    C = coefficients(f_a, f_b)
    looks_a = np.maximum(n_frac_a * cell * cell / (resolution_a[0] * resolution_a[1]), 1.0)
    looks_b = np.maximum(n_frac_b * cell * cell / (resolution_b[0] * resolution_b[1]), 1.0)
    return np.hypot(C["a"] * ss.phase_sigma(coh_a, looks_a), C["b"] * ss.phase_sigma(coh_b, looks_b))


def snaphu_unwrap(psi, coh, use, nlooks=50.0):
    """SNAPHU (smooth cost, MCF init) on the cells in ``use``; the result is ``psi`` plus whole cycles, NaN outside."""
    import snaphu

    z = np.where(use, np.exp(1j * np.where(use, psi, 0.0)), 0).astype(np.complex64)
    c = np.where(use, np.clip(np.nan_to_num(coh), 1e-3, 1.0), 0).astype(np.float32)
    unw = np.asarray(snaphu.unwrap(z, c, nlooks=nlooks, cost="smooth", init="mcf", mask=use)[0], float)
    unw = psi + TWO_PI * np.round((unw - psi) / TWO_PI)
    return np.where(use, unw, np.nan)


def _separable_gauss(a, sx, sy, truncate):
    rx, ry = max(int(truncate[0] * sx), 1), max(int(truncate[1] * sy), 1)
    kx = np.exp(-0.5 * (np.arange(-rx, rx + 1) / sx) ** 2)
    ky = np.exp(-0.5 * (np.arange(-ry, ry + 1) / sy) ** 2)
    b = ndimage.convolve1d(a, kx / kx.sum(), axis=1, mode="constant")
    return ndimage.convolve1d(b, ky / ky.sum(), axis=0, mode="constant")


def isce3_unwrap_correction(unw, cell, sigma_m=(2000.0, 2600.0), truncate=(1.75, 1.49), iterations=1):
    """ISCE3's `unwrapping_correction_with_filter` (filter_method "convolution"): each valid cell moved by the whole
    cycles that separate it from a NaN-aware truncated-Gaussian low-pass of the same field. Returns the corrected
    field and the cycles taken out."""
    u = np.array(unw, float)
    ok = np.isfinite(u)
    sx, sy = (s / float(cell) for s in sigma_m)
    taken = np.zeros(u.shape)
    for _ in range(iterations):
        num = _separable_gauss(np.where(ok, u, 0.0), sx, sy, truncate)
        den = _separable_gauss(ok.astype(float), sx, sy, truncate)
        with np.errstate(invalid="ignore", divide="ignore"):
            low = np.where(den > 1e-8, num / den, np.nan)
        n = np.where(ok & np.isfinite(low), np.round((u - low) / TWO_PI), 0.0)
        u = u - TWO_PI * n
        taken += n
    return np.where(ok, u, np.nan), taken


def _filt(d, sigma, use, cell, kw):
    return ss.ionosphere_filter(d, sigma, use & np.isfinite(d), cell, **kw)[0]


def _poly2(f, w, X, Y):
    ok = np.isfinite(f) & (w > 0)
    xs, ys = (X - X[ok].mean()) / 1e5, (Y - Y[ok].mean()) / 1e5
    A = np.stack([np.ones_like(xs), xs, ys, xs * xs, xs * ys, ys * ys], -1)
    sw = np.sqrt(w[ok])
    c, *_ = np.linalg.lstsq(A[ok] * sw[:, None], f[ok] * sw, rcond=None)
    return A @ c


def estimate(method, phi_a, psi_a, psi_b, coh_a, coh_b, use, sigma, f_a, f_b, cell, *, filter_kw=None,
             x=None, y=None, unwrap_sigma=3.0, nlooks=50.0, data=None):
    """One estimator (see the module docstring). ``data`` (optional) are the cells where ``phi_a`` is defined beyond
    ``use`` -- ISCE3's pre-correction runs over the whole unwrapped field, as it does on the RUNW. ``x``/``y`` (cell
    centres, m) are needed by side_isce2's quadratic. Returns a dict: ``screen``, ``raw`` (the unfiltered estimate on
    ``use``), and the cycle counts a correction moved (``cycles_a``, ``cycles_b``: fractions of ``use``)."""
    filter_kw = dict(FILTER_V6 if filter_kw is None else filter_kw)
    C = coefficients(f_a, f_b)
    use = np.asarray(use, bool) & np.isfinite(phi_a) & np.isfinite(psi_a) & np.isfinite(psi_b)
    data = use if data is None else (np.asarray(data, bool) & np.isfinite(phi_a))
    phi_a = phi_a + ss.wrap(psi_a - phi_a)                 # the cells' own phase, on the caller's integers
    out = dict(cycles_a=0.0, cycles_b=0.0)
    if method == "ours":
        r = ss.wrap(psi_b - (f_b / f_a) * phi_a)
        r_unw, _ = ss.unwrap_smooth_phase(r, use, sigma=unwrap_sigma)
        d = r_unw / C["k"]
    elif method.startswith("diff_"):
        delta = ss.wrap(psi_a - psi_b)
        if method == "diff_smooth":
            du, _ = ss.unwrap_smooth_phase(delta, use, sigma=unwrap_sigma)
        elif method == "diff_trivial":
            m = np.angle(np.nansum(np.exp(1j * delta[use])))
            du = np.where(use, m + ss.wrap(delta - m), np.nan)
        elif method in ("diff_snaphu", "diff_isce3"):
            du = snaphu_unwrap(delta, np.minimum(coh_a, coh_b), use, nlooks)
        else:
            raise ValueError(method)
        pa = phi_a
        if method == "diff_isce3":
            pa_all, na = isce3_unwrap_correction(np.where(data, phi_a, np.nan), cell)
            du, nd = isce3_unwrap_correction(du, cell)
            pa = np.where(use, pa_all, np.nan)
            d1 = C["c0"] * pa + C["c1"] * du
            n1 = pa - d1
            Fd, Fn = _filt(d1, sigma, use, cell, FILTER_ISCE3), _filt(n1, sigma, use, cell, FILTER_ISCE3)
            fd = f_b - f_a
            com = np.round((pa - (Fn + Fd)) / TWO_PI)
            dif = np.round((du + fd / f_a * Fn - fd / f_b * Fd) / TWO_PI)
            pa, du = pa - TWO_PI * com, du - TWO_PI * dif
            out["cycles_a"] = float(np.mean(((na != 0) | (com != 0))[use]))
            out["cycles_b"] = float(np.mean(((nd != 0) | (dif != 0))[use]))
        d = C["c0"] * pa + C["c1"] * du
    elif method.startswith("side_"):
        pb = snaphu_unwrap(psi_b, coh_b, use, nlooks)
        pa = np.where(use, phi_a, np.nan)
        if method == "side_isce2":
            if x is None or y is None:
                raise ValueError("side_isce2 needs the cell centres x, y")
            X, Y = np.meshgrid(x, y)
            D = pa - pb
            P = _poly2(D, np.where(use, np.minimum(coh_a, coh_b), 0.0), X, Y)
            n = np.round((D - P) / TWO_PI)
            pb = pb + TWO_PI * n
            out["cycles_b"] = float(np.mean((n != np.nanmedian(n[use]))[use]))
        elif method == "side_isce3":
            pa_all, na = isce3_unwrap_correction(np.where(data, phi_a, np.nan), cell)
            pb, nb = isce3_unwrap_correction(pb, cell)
            pa = np.where(use, pa_all, np.nan)
            d1 = C["a"] * pa + C["b"] * pb
            n1 = pa - d1
            Fd, Fn = _filt(d1, sigma, use, cell, FILTER_ISCE3), _filt(n1, sigma, use, cell, FILTER_ISCE3)
            dif = np.round(((1 - f_b / f_a) * Fn + (1 - f_a / f_b) * Fd + pb - pa) / TWO_PI)
            com = np.round((pa + pb - (1 + f_b / f_a) * Fn - (1 + f_a / f_b) * Fd - TWO_PI * dif) / (2 * TWO_PI))
            pa, pb = pa - TWO_PI * com, pb - TWO_PI * (com + dif)
            out["cycles_a"] = float(np.mean(((na != 0) | (com != 0))[use]))
            out["cycles_b"] = float(np.mean(((nb != 0) | (dif != 0))[use]))
        elif method != "side_snaphu":
            raise ValueError(method)
        d = C["a"] * pa + C["b"] * pb
    else:
        raise ValueError(f"unknown method {method!r}")
    d = np.where(use, d, np.nan)
    out["raw"] = d
    out["screen"] = _filt(d, sigma, use, cell, filter_kw)
    return out
