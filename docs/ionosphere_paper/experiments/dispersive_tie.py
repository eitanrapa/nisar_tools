"""The side band ties the main band across a gap.

A block whose main-band integer is off by eps cycles carries exactly 2 pi c0 eps (= 3.21 eps rad, c0 = f_B / (f_A + f_B))
of extra raw dispersive estimate: the scaled main band is taken out of the wrapped side band, the whole cycles wrap away
and only the 4.4% that does not is left (``selftest.py``). The ionosphere it measures is continuous across the gap that
separates two blocks (a fault zone, a strait), so the raw estimate's step across the gap measures eps:

    d = P(x, y) + delta [cell in block B]        over cells of A and B near their common boundary, per window
    e = -delta / (2 pi c0)                        whole cycles to ADD to block B's main band

P is a low-order polynomial (quadratic by default) shared by both sides of the window, so the ionosphere's gradient and
curvature are fitted across the gap, not extrapolated from one side. The fit is weighted by each cell's variance and
iteratively reweighted (Tukey biweight at 4.685 scaled MADs), because a local cycle error moves a patch by 3.21 rad and
must not drive the step. Windows tile the boundary; their median is the answer, their scatter its uncertainty.

Nothing here knows about faults: the blocks are two labels on the cell grid, and the boundary is wherever they come
within ``reach_m`` of each other.
"""

import numpy as np
from scipy import ndimage

#: f_B / (f_A + f_B) for NISAR's 1239 / 1293.5 MHz
C0 = 1293.5e6 / (1239.0e6 + 1293.5e6)
#: raw dispersive estimate per main-band cycle on a block (rad at f_A)
STEP = 2.0 * np.pi * C0
TUKEY = 4.685


def irls(A, y, w0=None, iters=30, k=TUKEY):
    """Weighted least squares, iteratively reweighted with Tukey's biweight on the weighted residuals. Returns the
    coefficients, the final weights and the formal covariance (scaled by the robust residual variance)."""
    w0 = np.ones(y.size) if w0 is None else np.asarray(w0, float)
    s0 = np.sqrt(w0)
    coef = np.linalg.lstsq(A * s0[:, None], y * s0, rcond=None)[0]
    w = w0
    for _ in range(iters):
        r = (y - A @ coef) * s0
        mad = 1.4826 * np.median(np.abs(r - np.median(r)))
        if not mad > 0:
            break
        u = r / (k * mad)
        w = w0 * np.where(np.abs(u) < 1.0, (1.0 - u * u) ** 2, 0.0)
        sw = np.sqrt(w)
        new = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)[0]
        done = np.allclose(new, coef, atol=1e-7)
        coef = new
        if done:
            break
    sw = np.sqrt(w)
    r = (y - A @ coef) * sw
    keep = w > 0
    dof = max(int(keep.sum()) - A.shape[1], 1)
    s2 = float((r[keep] ** 2).sum() / dof)
    try:
        cov = s2 * np.linalg.inv((A * w[:, None]).T @ A)
    except np.linalg.LinAlgError:
        cov = np.full((A.shape[1], A.shape[1]), np.nan)
    return coef, w, cov


def design(x, y, deg):
    """Columns of a full 2-D polynomial of degree ``deg`` in (x, y)."""
    cols = [np.ones_like(x)]
    for p in range(1, deg + 1):
        for q in range(p + 1):
            cols.append(x ** (p - q) * y ** q)
    return np.stack(cols, 1)


def near_boundary(block_a, block_b, cell, reach_m):
    """Distance (m) of every cell to block B and to block A, and the cells of each within ``reach_m`` of the other."""
    to_b = ndimage.distance_transform_edt(~np.asarray(block_b, bool), sampling=cell)
    to_a = ndimage.distance_transform_edt(~np.asarray(block_a, bool), sampling=cell)
    return to_a, to_b, np.asarray(block_a, bool) & (to_b <= reach_m), np.asarray(block_b, bool) & (to_a <= reach_m)


def block_tie(d, use, block_a, block_b, x, y, *, sigma=None, reach_m=10000.0, window_m=16000.0, deg=2,
              robust=True, min_cells=60, offset=(0.0, 0.0), min_gap_m=0.0, step=STEP, along=None):
    """Whole cycles ``e`` to add to block B's main band so that its raw dispersive estimate continues block A's.

    ``d`` is the raw (unfiltered) dispersive estimate on the cells (rad at f_A), ``use`` the cells it is defined on,
    ``block_a``/``block_b`` the two blocks' extents on the cell grid (every cell of the region, used or not: the
    boundary is measured from them), ``x``/``y`` the cell-centre coordinates (1-D, m), ``sigma`` each cell's standard
    deviation of ``d`` (weights; uniform if None). Cells of each block between ``min_gap_m`` and ``reach_m`` of the
    other enter; square windows of ``window_m`` (shifted by ``offset``, m) tile them, and each window with
    ``min_cells`` on both sides is fitted with a degree-``deg`` polynomial shared by both sides plus a step on B.
    ``step`` is the field's change on B per cycle added to B's main band, negated (3.21 rad for the raw dispersive
    estimate; for ``m = 4 pi / lambda * range offset - phi_A`` pass ``-m`` and ``2 pi``). ``along`` (optional, m, on
    the cells) is a coordinate along the boundary, e.g. the arc length of each cell's nearest trace point: the
    windows are then segments of ``window_m`` of it (shifted by ``offset[0]``) instead of squares.

    Returns a dict: ``e`` (median over windows), ``e_wmean`` (inverse-variance mean), ``se`` (standard error of the
    median from the window scatter, bootstrap), ``spread`` (scaled MAD of the windows), ``windows`` (per-window dicts)
    and ``cells`` (a boolean map of the cells that entered).
    """
    X, Y = np.meshgrid(np.asarray(x, float), np.asarray(y, float))
    cell = float(abs(x[1] - x[0]))
    to_a, to_b, near_a, near_b = near_boundary(block_a, block_b, cell, reach_m)
    ok = np.asarray(use, bool) & np.isfinite(d)
    in_a = ok & near_a & (to_b >= min_gap_m)
    in_b = ok & near_b & (to_a >= min_gap_m)
    sel = in_a | in_b
    w0 = np.ones(d.shape) if sigma is None else 1.0 / np.maximum(np.asarray(sigma, float), 1e-3) ** 2
    empty = dict(e=np.nan, e_wmean=np.nan, se=np.nan, spread=np.nan, windows=[], cells=sel)
    if not sel.any():
        return empty
    if along is None:
        x0 = X[sel].min() - window_m + offset[0]
        y0 = Y[sel].min() - window_m + offset[1]
        ix = np.floor((X - x0) / window_m).astype(int)
        iy = np.floor((Y - y0) / window_m).astype(int)
    else:
        a = np.asarray(along, float)
        sel &= np.isfinite(a)
        ix = np.floor((np.where(np.isfinite(a), a, 0.0) - (np.nanmin(a[sel]) + offset[0])) / window_m).astype(int)
        iy = np.zeros(ix.shape, int)
    rows = []
    for tx, ty in sorted(set(zip(ix[sel], iy[sel]))):
        m = sel & (ix == tx) & (iy == ty)
        na, nb = int((m & in_a).sum()), int((m & in_b).sum())
        if na < min_cells or nb < min_cells:
            continue
        xc, yc = X[m].mean(), Y[m].mean()
        A = np.column_stack([design((X[m] - xc) / 1e4, (Y[m] - yc) / 1e4, deg), in_b[m].astype(float)])
        coef, w, cov = irls(A, d[m], w0[m], iters=30 if robust else 0)
        rows.append(dict(x=float(xc), y=float(yc), e=float(-coef[-1] / step), se=float(np.sqrt(cov[-1, -1]) / step),
                         cells_a=na, cells_b=nb, kept=float(np.mean(w > 0))))
    if not rows:
        return empty
    e = np.array([r["e"] for r in rows])
    se = np.array([r["se"] for r in rows])
    rng = np.random.default_rng(0)
    boot = np.median(rng.choice(e, (2000, e.size)), axis=1)
    wv = 1.0 / np.maximum(se, 1e-3) ** 2
    return dict(e=float(np.median(e)), e_wmean=float(np.sum(wv * e) / np.sum(wv)), se=float(np.std(boot)),
                spread=float(1.4826 * np.median(np.abs(e - np.median(e)))), windows=rows, cells=sel)


def approx_sigma(coherence_a, coherence_b, f_a, f_b, cell, resolution_a=(6.2, 5.3), resolution_b=(50.0, 5.3)):
    """Each cell's standard deviation of the raw estimate (split_spectrum's, with full cells: the saved screens do not
    keep the pixel counts)."""
    from nisar_tools import split_spectrum as ss

    c_a = f_b * f_b / (f_b * f_b - f_a * f_a)
    c_b = f_a * f_b / (f_b * f_b - f_a * f_a)
    la = cell * cell / (resolution_a[0] * resolution_a[1])
    lb = cell * cell / (resolution_b[0] * resolution_b[1])
    return np.hypot(c_a * ss.phase_sigma(coherence_a, la), c_b * ss.phase_sigma(coherence_b, lb))


# ---------------------------------------------------------------------------------------------------------------------
# The step by kriging: the ionosphere as a Gaussian process instead of a polynomial
# ---------------------------------------------------------------------------------------------------------------------
def matern32(r, ell):
    a = np.sqrt(3.0) * r / ell
    return (1.0 + a) * np.exp(-a)


def _superpixels(d, sigma, sel, side_b, X, Y, agg):
    """Median of ``d`` over agg x agg cells, its variance (pi/2 x mean sigma^2 / n), centre and side (cells of one side
    only: a superpixel holding both sides is dropped)."""
    iy, ix = np.nonzero(sel)
    key = (iy // agg) * (10 ** 6) + ix // agg
    order = np.argsort(key, kind="stable")
    key, iy, ix = key[order], iy[order], ix[order]
    cut = np.flatnonzero(np.diff(key)) + 1
    out = []
    for gy, gx in zip(np.split(iy, cut), np.split(ix, cut)):
        b = side_b[gy, gx]
        if b.any() and not b.all():
            continue
        v = d[gy, gx]
        out.append((np.median(v), 1.5708 * np.mean(sigma[gy, gx] ** 2) / v.size, X[gy, gx].mean(), Y[gy, gx].mean(),
                    float(b[0]), v.size))
    return np.array(out) if out else np.zeros((0, 6))


def gls_reml(y, F, P, v, ell0=10.0, eta0=0.1):
    """REML fit of y = F beta + f + e, f ~ s2 Matern-3/2(ell) on the points P (km), e ~ s2 eta v (v known up to eta).
    Returns beta, its covariance, (ell, eta, s2), the standardized leave-one-out residuals and the Cholesky factor."""
    from scipy.linalg import cho_factor, cho_solve
    from scipy.optimize import minimize
    from scipy.spatial.distance import cdist

    R = cdist(P, P)
    n, p = F.shape
    vr = v / np.median(v)

    def parts(th):
        ell, eta = np.exp(th)
        V = matern32(R, ell) + np.diag(eta * vr)
        c = cho_factor(V, lower=True, check_finite=False)
        Vi_F = cho_solve(c, F, check_finite=False)
        Vi_y = cho_solve(c, y, check_finite=False)
        A = F.T @ Vi_F
        ca = cho_factor(A, lower=True, check_finite=False)
        beta = cho_solve(ca, F.T @ Vi_y, check_finite=False)
        r = y - F @ beta
        q = float(r @ cho_solve(c, r, check_finite=False))
        return c, ca, beta, r, q, Vi_F

    def obj(th):
        if not (np.log(0.5) <= th[0] <= np.log(300.0) and np.log(1e-4) <= th[1] <= np.log(1e3)):
            return 1e12                                   # Nelder-Mead has no bounds
        try:
            c, ca, beta, r, q, _ = parts(th)
        except np.linalg.LinAlgError:
            return 1e12
        s2 = q / (n - p)
        return (n - p) * np.log(s2) + 2 * np.log(np.diag(c[0])).sum() + 2 * np.log(np.diag(ca[0])).sum()

    best = None
    for l0 in (ell0, ell0 * 4):
        res = minimize(obj, np.log([l0, eta0]), method="Nelder-Mead",
                       options=dict(xatol=0.02, fatol=0.01, maxiter=200))
        if best is None or res.fun < best.fun:
            best = res
    c, ca, beta, r, q, Vi_F = parts(best.x)
    s2 = q / (n - p)
    Ainv = cho_solve(ca, np.eye(p), check_finite=False)
    Vi = cho_solve(c, np.eye(n), check_finite=False)
    Pd = np.diag(Vi) - np.einsum("ij,jk,ik->i", Vi_F, Ainv, Vi_F)
    z = (Vi @ r) / np.sqrt(np.maximum(Pd, 1e-12) * s2)
    ell, eta = np.exp(best.x)
    return beta, s2 * Ainv, (float(ell), float(eta), float(s2)), z


def gp_tie(d, use, block_a, block_b, x, y, *, sigma, reach_m=12000.0, window_m=20000.0, along=None, agg=3,
           min_points=30, min_gap_m=0.0, z_cut=3.5, max_points=1500, step=STEP, trend=1, seed=0):
    """As :func:`block_tie`, with the ionosphere a Gaussian process (Matern 3/2, its length, signal and noise scales by
    REML per window) plus a polynomial trend of degree ``trend``, instead of a polynomial alone. The cells are reduced to
    medians over ``agg`` x ``agg`` (a cycle error on a minority of a superpixel's cells does not move it), and points
    whose leave-one-out residual exceeds ``z_cut`` standard deviations are dropped and the fit repeated (a superpixel
    off by a whole cycle). Returns the median over windows ``e``, the inverse-variance mean ``e_wmean`` with its
    standard error ``se_wmean`` and the windows' chi-square per degree of freedom ``chi2``, and the windows."""
    Xg, Yg = np.meshgrid(np.asarray(x, float), np.asarray(y, float))
    cell = float(abs(x[1] - x[0]))
    to_a, to_b, near_a, near_b = near_boundary(block_a, block_b, cell, reach_m)
    ok = np.asarray(use, bool) & np.isfinite(d)
    in_a = ok & near_a & (to_b >= min_gap_m)
    in_b = ok & near_b & (to_a >= min_gap_m)
    sel = in_a | in_b
    rng = np.random.default_rng(seed)
    if along is None:
        x0, y0 = Xg[sel].min() - window_m, Yg[sel].min() - window_m
        wid = (np.floor((Xg - x0) / window_m).astype(int) * 100000 + np.floor((Yg - y0) / window_m).astype(int))
    else:
        a = np.asarray(along, float)
        sel &= np.isfinite(a)
        wid = np.floor((np.where(np.isfinite(a), a, 0.0) - np.nanmin(a[sel])) / window_m).astype(int)
    rows = []
    for w in np.unique(wid[sel]):
        m = sel & (wid == w)
        sp = _superpixels(d, sigma, m, in_b, Xg, Yg, agg)
        if sp.shape[0] == 0 or (sp[:, 4] == 1).sum() < min_points or (sp[:, 4] == 0).sum() < min_points:
            continue
        if sp.shape[0] > max_points:
            sp = sp[rng.choice(sp.shape[0], max_points, replace=False)]
        keep = np.ones(sp.shape[0], bool)
        for it in range(4):
            s = sp[keep]
            xc, yc = s[:, 2].mean(), s[:, 3].mean()
            P = np.c_[(s[:, 2] - xc) / 1e3, (s[:, 3] - yc) / 1e3]
            F = np.column_stack([design(P[:, 0] / 10, P[:, 1] / 10, trend), s[:, 4]])
            beta, cov, hyp, z = gls_reml(s[:, 0], F, P, s[:, 1])
            bad = np.abs(z) > z_cut
            if not bad.any():
                break
            idx = np.flatnonzero(keep)
            keep[idx[bad]] = False
        rows.append(dict(x=float(xc), y=float(yc), e=float(-beta[-1] / step), se=float(np.sqrt(cov[-1, -1]) / step),
                         ell_km=hyp[0], eta=hyp[1], s2=hyp[2], points_a=int((s[:, 4] == 0).sum()),
                         points_b=int((s[:, 4] == 1).sum()), dropped=int((~keep).sum())))
    if not rows:
        return dict(e=np.nan, e_wmean=np.nan, se_wmean=np.nan, chi2=np.nan, windows=[])
    e = np.array([r["e"] for r in rows])
    se = np.array([r["se"] for r in rows])
    wv = 1.0 / se ** 2
    mean = float(np.sum(wv * e) / np.sum(wv))
    chi2 = float(np.sum(wv * (e - mean) ** 2) / max(e.size - 1, 1)) if e.size > 1 else np.nan
    return dict(e=float(np.median(e)), e_wmean=mean, se_wmean=float(np.sqrt(1.0 / np.sum(wv))), chi2=chi2, windows=rows)
