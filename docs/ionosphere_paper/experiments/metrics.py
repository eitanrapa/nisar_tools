"""Scores for the estimator comparison, on the cells.

- **leak**: the slope (and correlation) of a screen's 15 km high-pass on the deformation's, within 10 km of the trace.
  This is the metric of the earlier GUNW comparisons (quadratic out, then a NaN-aware 15 km Gaussian out). Against
  the truth's ``psi`` on synthetics; against the non-dispersive phase (phi_A minus our screen) on NISAR data.
- **error** (synthetics): screen minus the true ionosphere, less its median beyond 20 km. Reported as rms in bands of
  distance from the trace (0-2 km is the excluded fault zone, filled).
- **step**: the screen's (or its error's) median at 2-4 km north minus 2-4 km south of the trace.
"""

import numpy as np
from scipy.ndimage import gaussian_filter

BINS_KM = ((0, 2), (2, 5), (5, 10), (10, 20), (20, np.inf))
HP_KM, NEAR_KM, FAR_KM = 15.0, 10.0, 20.0


def nan_lowpass(a, sigma_cells):
    m = np.isfinite(a)
    num = gaussian_filter(np.where(m, a, 0.0), sigma_cells, mode="nearest")
    den = gaussian_filter(m.astype(float), sigma_cells, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(m & (den > 1e-3), num / den, np.nan)


def minus_poly(f, X, Y, deg=2):
    ok = np.isfinite(f)
    xs, ys = (X - X[ok].mean()) / 1e5, (Y - Y[ok].mean()) / 1e5
    A = np.stack([xs ** i * ys ** (k - i) for k in range(deg + 1) for i in range(k + 1)], -1)
    c, *_ = np.linalg.lstsq(A[ok], f[ok], rcond=None)
    return np.where(ok, f - A @ c, np.nan)


def highpass(f, X, Y, cell, km=HP_KM):
    q = minus_poly(f, X, Y, 2)
    return q - nan_lowpass(q, km * 1e3 / cell)


def _slope(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 20 or np.std(x[ok]) == 0:
        return np.nan, np.nan
    return float(np.polyfit(x[ok], y[ok], 1)[0]), float(np.corrcoef(x[ok], y[ok])[0, 1])


def step(f, dist_m, mask, a_km=2.0, b_km=4.0):
    n = mask & np.isfinite(f) & (dist_m >= a_km * 1e3) & (dist_m < b_km * 1e3)
    s = mask & np.isfinite(f) & (dist_m <= -a_km * 1e3) & (dist_m > -b_km * 1e3)
    if n.sum() < 5 or s.sum() < 5:
        return np.nan
    return float(np.median(f[n]) - np.median(f[s]))


def synthetic(screen, truth, cell):
    """Scores of one screen against a synthetic's truth (rad at f_A)."""
    land, d = truth["land"], truth["dist"]
    X, Y = np.meshgrid(truth["x"], truth["y"])
    e = np.where(land, screen - truth["iono"], np.nan)
    far = land & np.isfinite(e) & (np.abs(d) > FAR_KM * 1e3)
    e = e - np.median(e[far])
    row = {}
    for a, b in BINS_KM:
        m = land & np.isfinite(e) & (np.abs(d) >= a * 1e3) & (np.abs(d) < b * 1e3)
        row[f"rms_{a}_{'inf' if np.isinf(b) else int(b)}"] = float(np.sqrt(np.mean(e[m] ** 2))) if m.any() else np.nan
    near = land & (np.abs(d) < NEAR_KM * 1e3)
    row["max_near"] = float(np.nanmax(np.abs(e[near])))
    he = highpass(e, X, Y, cell)
    hp = highpass(np.where(land, truth["psi"], np.nan), X, Y, cell)
    row["leak"], row["leak_r"] = _slope(np.where(near, hp, np.nan), np.where(near, he, np.nan))
    row["step"] = step(e, d, land)
    return row, e


def nisar(screen, ours, phi_nd, dist, land, X, Y, cell):
    """Scores of one screen on NISAR data: the leak against the non-dispersive phase, the step, and the difference
    from our screen (both less their medians beyond 20 km), near (< 10 km) and far (> 20 km)."""
    near = land & (np.abs(dist) < NEAR_KM * 1e3)
    hs = highpass(np.where(land, screen, np.nan), X, Y, cell)
    hn = highpass(np.where(land, phi_nd, np.nan), X, Y, cell)
    row = {}
    row["leak"], row["leak_r"] = _slope(np.where(near, hn, np.nan), np.where(near, hs, np.nan))
    row["step"] = step(screen, dist, land)
    diff = np.where(land, screen - ours, np.nan)
    far = land & np.isfinite(diff) & (np.abs(dist) > FAR_KM * 1e3)
    diff = diff - np.median(diff[far])
    row["diff_near"] = float(np.sqrt(np.nanmean(diff[near] ** 2)))
    row["diff_far"] = float(np.sqrt(np.nanmean(diff[far] ** 2)))
    row["diff_max_near"] = float(np.nanmax(np.abs(diff[near])))
    return row
