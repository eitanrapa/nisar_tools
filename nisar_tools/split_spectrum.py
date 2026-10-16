"""Split-spectrum ionosphere from NISAR's main and side bands, without the fault leakage.

The estimator is ISCE3's ``main_diff_ms_band`` (the PR GUNW's): with ``phi_A`` the main-band
(frequency A) and ``phi_B`` the side-band (frequency B) interferometric phase,

    d = 12.12 phi_A - 11.61 phi_B          (at f_A = 1.239 GHz, f_B = 1.2935 GHz)

is the dispersive (ionospheric) phase at ``f_A`` -- the GUNW writes it as
``c0 phi_A + c1 Delta`` with ``Delta = phi_A - phi_B``. What changes here is *how the two phases
are obtained*, because in the GUNW that is where the deformation gets in. ISCE3 (v0.25.16):

- multilooks B over 13 x 16 of its own pixels (325 m of slant range) and *decimates* A's
  13 x 16-look phase (40 m) to B's columns, a fixed ~9 m range offset and an 8x footprint apart;
- unwraps ``phi_A`` and ``Delta`` separately, where ``Delta = -0.044 psi + 0.042 d`` carries
  4.4% of the deformation ``psi`` -- steps included -- into a second, independent unwrapping;
- then "corrects" ``phi_A`` by whole cycles against a ~2 km low-pass of the *non-dispersive*
  field: near a rupture the deformation leaves its low-pass by more than pi, so cycles are
  removed where none were wrong, each worth 0.51 x 2 pi = 3.2 rad of screen.

Measured on the D141 (Panama) PR GUNW: within 10 km of the rupture the screen's 15 km high-pass
is +0.34..+0.48 x the unwrapped phase's on the south side (r +0.67..+0.81), -0.14..-0.19 on the
north, still +0.13..+0.17 at 10-20 km. The footprint mismatch alone predicts a tenth of it.

Here, instead:

1. both bands are summed over the **same map cells** (:func:`gslc_cells`: GSLC pairs on their own
   grids, cells on a lattice both grids share);
2. ``phi_A`` comes from the caller's own unwrapping (fault-aware, cut and tied), and only the
   **purely dispersive** residual is unwrapped::

       r = phi_B - (f_B/f_A) phi_A = (f_A/f_B - f_B/f_A) d = -0.0861 d

   which holds no deformation at all, and is continuous across the fault;
3. no cycle is moved against a smoothed non-dispersive field; cells near the trace (or of low
   coherence in either band) are masked and filled by the filter, from both sides;
4. the filter is ISCE3's ``IonosphereFilter.low_pass_filter`` (guide, outliers, fill, inverse-
   variance-weighted truncated Gaussian), sized in metres instead of B-grid pixels.

The phase convention is the package's interferogram, ``reference * conj(secondary)``.
"""

import numpy as np
import xarray as xr
from scipy import ndimage

GSLC_GRID = "science/LSAR/GSLC/grids/frequency{f}"


def wrap(a):
    """Phase to ``[-pi, pi)``."""
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def dispersive_factor(f_a, f_b):
    """``k`` in ``phi_B - (f_B/f_A) phi_A = k d``: ``f_A/f_B - f_B/f_A`` (-0.0861 for NISAR A/B)."""
    return f_a / f_b - f_b / f_a


def main_diff_coefficients(f_a, f_b):
    """ISCE3's ``d = c0 phi_A + c1 (phi_A - phi_B)``: ``(c0, c1)`` (0.511, 11.61 for NISAR A/B)."""
    return f_b / (f_a + f_b), f_a * f_b / (f_b * f_b - f_a * f_a)


def phase_sigma(coherence, looks):
    """ISCE3's per-band phase standard deviation, ``sqrt(1 - g^2) / (g sqrt(2 N))``."""
    g = np.clip(np.asarray(coherence, float), 1e-6, 1.0)
    return np.sqrt(1.0 - g * g) / (g * np.sqrt(2.0 * np.asarray(looks, float)))


# ---------------------------------------------------------------------------
# 1. Cells
# ---------------------------------------------------------------------------
def _gslc_grid(path, frequency):
    import h5py

    with h5py.File(path, "r") as f:
        g = f[GSLC_GRID.format(f=frequency)]
        freq = float(g["centerFrequency"][()]) if "centerFrequency" in g else None
        return g["xCoordinates"][()], g["yCoordinates"][()], freq


def cell_lattice(x, y, cell):
    """Index offset and size of square ``cell`` (m) on a pixel-centre grid ``x``, ``y``.

    Cells start on multiples of ``cell`` in map coordinates, so two grids whose pixel edges
    both fall on that lattice (A's 5 m and B's 40 x 5 m, for instance) get the same cells.
    Returns ``(ox, oy, kx, ky, ncx, ncy, x_centres, y_centres)``.
    """
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
    kx, ky = int(round(cell / abs(dx))), int(round(cell / abs(dy)))
    if abs(kx * abs(dx) - cell) > 1e-6 or abs(ky * abs(dy) - cell) > 1e-6:
        raise ValueError(f"cell {cell} m is not a whole number of pixels ({dx}, {dy})")
    x_edge0 = x[0] - dx / 2.0
    y_edge0 = y[0] - dy / 2.0
    first_x = np.ceil(round(x_edge0 / cell, 9)) * cell if dx > 0 else np.floor(round(x_edge0 / cell, 9)) * cell
    first_y = np.floor(round(y_edge0 / cell, 9)) * cell if dy < 0 else np.ceil(round(y_edge0 / cell, 9)) * cell
    ox = int(round(abs(first_x - x_edge0) / abs(dx)))
    oy = int(round(abs(first_y - y_edge0) / abs(dy)))
    ncx = (x.size - ox) // kx
    ncy = (y.size - oy) // ky
    xc = first_x + np.sign(dx) * (np.arange(ncx) + 0.5) * cell
    yc = first_y + np.sign(dy) * (np.arange(ncy) + 0.5) * cell
    return ox, oy, kx, ky, ncx, ncy, xc, yc


def gslc_cells(reference, secondary, frequency, cell=240.0, polarization="HH", bounds=None,
               strip_cells=8, subswath_mask=False, near_range_m=11000.0, near_range_side=None):
    """One band's interferogram summed over map cells, from two GSLCs on one grid.

    ``reference``/``secondary`` are GSLC paths whose ``frequency`` grids coincide (a pair made
    with ``geocode_rslc(like=)``). ``bounds = (x_min, y_min, x_max, y_max)`` limits the cells.
    Returns an :class:`xarray.Dataset` on cell centres ``(y, x)``: the complex sum ``ifg`` of
    ``ref * conj(sec)``, the powers ``p1``/``p2``, the valid-pixel count ``n``, and the
    ``coherence``. Pixels that are zero or non-finite in either date do not count, nor (``subswath_mask``)
    pixels the products' own ``mask`` layer marks partially focused (0) or outside the acquisition (255) in
    either date. NISAR's band A is partially focused over ~10 km at near range and band B over ~1 km
    (A004, 2026-10-14): finite there, and the split spectrum turns their phase error into a screen strip.
    ``subswath_mask="near_range"`` drops the 255s, and the 0s only within ``near_range_m`` of each row's near-range
    footprint edge: a datatake's partially focused start or end is kept (A004's rupture lies in one; dropping it
    extrapolated the screen 20-40 km). ``near_range_side`` ("east"/"west") defaults to :func:`near_range_side_of`
    the reference.
    """
    import h5py

    from .gslc import DirectChunkReader

    x, y, f_c = _gslc_grid(reference, frequency)
    x2, y2, _ = _gslc_grid(secondary, frequency)
    if x.shape != x2.shape or y.shape != y2.shape or not (np.allclose(x, x2) and np.allclose(y, y2)):
        raise ValueError("gslc_cells: the two GSLCs are not on the same grid")
    ox, oy, kx, ky, ncx, ncy, xc, yc = cell_lattice(x, y, cell)
    cols = np.arange(ncx)
    rows = np.arange(ncy)
    if bounds is not None:
        x0, y0, x1, y1 = bounds
        cols = cols[(xc >= x0) & (xc <= x1)]
        rows = rows[(yc >= y0) & (yc <= y1)]
    if cols.size == 0 or rows.size == 0:
        raise ValueError("gslc_cells: no cells inside bounds")
    c0, c1 = int(cols[0]), int(cols[-1]) + 1
    r0, r1 = int(rows[0]), int(rows[-1]) + 1
    shape = (r1 - r0, c1 - c0)
    ifg = np.zeros(shape, np.complex128)
    p1 = np.zeros(shape)
    p2 = np.zeros(shape)
    n = np.zeros(shape, np.int64)
    path = GSLC_GRID.format(f=frequency) + "/" + polarization
    mpath = GSLC_GRID.format(f=frequency) + "/mask"
    with h5py.File(reference, "r") as h1, h5py.File(secondary, "r") as h2:
        R1, R2 = DirectChunkReader(h1[path]), DirectChunkReader(h2[path])
        masks = []
        if subswath_mask not in (False, True, "near_range"):
            raise ValueError(f"gslc_cells: subswath_mask must be False, True or 'near_range', not {subswath_mask!r}")
        near = subswath_mask == "near_range"
        if near:
            near_range_side = near_range_side or near_range_side_of(reference)
            if near_range_side not in ("east", "west"):
                raise ValueError(f"gslc_cells: near_range_side must be 'east' or 'west', not {near_range_side!r}")
        if subswath_mask:
            masks = [h[mpath] for h in (h1, h2) if mpath in h]
            if not masks:
                raise ValueError(f"gslc_cells: subswath_mask={subswath_mask!r} but neither GSLC has {mpath}")
        for a in range(r0, r1, strip_cells):
            b = min(a + strip_cells, r1)
            ys = slice(oy + a * ky, oy + b * ky)
            xs = slice(ox + c0 * kx, ox + c1 * kx)
            g1 = np.asarray(R1[ys, xs], np.complex64)
            g2 = np.asarray(R2[ys, xs], np.complex64)
            ok = np.isfinite(g1) & np.isfinite(g2) & (g1 != 0) & (g2 != 0)
            for M in masks:
                if near:                                 # on to the grid's near-range end: the edge may lie past bounds
                    ext = slice(xs.start, x.size) if near_range_side == "east" else slice(0, xs.stop)
                    mf = np.asarray(M[ys, ext])
                    cols = slice(xs.start - ext.start, xs.stop - ext.start)
                    m = mf[:, cols]
                    strip = _near_range_strip(mf, x[ext], cols, near_range_side, near_range_m)
                    ok &= (m != 255) & ((m != 0) | ~strip)
                else:
                    m = np.asarray(M[ys, xs])
                    ok &= (m >= 1) & (m < 255)
            g1 = np.where(ok, g1, 0)
            g2 = np.where(ok, g2, 0)
            sh = (b - a, ky, c1 - c0, kx)
            sl = slice(a - r0, b - r0)
            ifg[sl] = (g1 * np.conj(g2)).reshape(sh).sum(axis=(1, 3), dtype=np.complex128)
            p1[sl] = (np.abs(g1) ** 2).reshape(sh).sum(axis=(1, 3), dtype=np.float64)
            p2[sl] = (np.abs(g2) ** 2).reshape(sh).sum(axis=(1, 3), dtype=np.float64)
            n[sl] = ok.reshape(sh).sum(axis=(1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        coh = np.abs(ifg) / np.sqrt(p1 * p2)
    return xr.Dataset(
        {"ifg": (("y", "x"), ifg), "p1": (("y", "x"), p1), "p2": (("y", "x"), p2),
         "n": (("y", "x"), n), "coherence": (("y", "x"), coh)},
        coords={"y": yc[r0:r1], "x": xc[c0:c1]},
        attrs={"frequency": frequency, "center_frequency": f_c, "cell": float(cell),
               "pixels_per_cell": kx * ky, "reference": str(reference),
               "secondary": str(secondary), "polarization": polarization,
               "subswath_mask": subswath_mask if near else bool(subswath_mask),
               **({"near_range_m": float(near_range_m), "near_range_side": near_range_side} if near else {})})


def near_range_side_of(path):
    """"east" or "west": the map side of a product's near-range swath edge, from its ``lookDirection`` and
    ``orbitPassDirection`` (left-looking and ascending: the swath lies west of the ground track, near range east)."""
    import h5py

    with h5py.File(path, "r") as h:
        idn = h["science/LSAR/identification"]
        look, orbit = (idn[k][()] for k in ("lookDirection", "orbitPassDirection"))
    look, orbit = ((v.decode() if isinstance(v, bytes) else str(v)).strip().lower() for v in (look, orbit))
    if look not in ("left", "right") or orbit not in ("ascending", "descending"):
        raise ValueError(f"near_range_side_of: lookDirection {look!r}, orbitPassDirection {orbit!r}")
    return "east" if (look == "left") == (orbit == "ascending") else "west"


def _near_range_strip(mask_rows, x, cols, side, width_m):
    """``(rows, columns cols)`` of ``mask_rows`` (columns at ``x``, reaching the near-range end of the grid): within
    ``width_m`` of each row's near-range footprint edge (``mask != 255``)."""
    fp = mask_rows != 255
    has = fp.any(axis=1)
    if side == "east":
        edge = mask_rows.shape[1] - 1 - np.argmax(fp[:, ::-1], axis=1)
    else:
        edge = np.argmax(fp, axis=1)
    return has[:, None] & (np.abs(x[cols][None, :] - x[edge][:, None]) <= width_m)


def merge_cells(*cells):
    """Sum cell datasets of one band from several frames onto their union (cells coincide)."""
    xs = np.unique(np.concatenate([c.x.values for c in cells]))
    ys = np.unique(np.concatenate([c.y.values for c in cells]))[::-1]
    out = {v: np.zeros((ys.size, xs.size), dt) for v, dt in
           (("ifg", np.complex128), ("p1", float), ("p2", float), ("n", np.int64))}
    for c in cells:
        iy = np.searchsorted(-ys, -c.y.values)
        ix = np.searchsorted(xs, c.x.values)
        for v in out:
            out[v][np.ix_(iy, ix)] += c[v].values
    with np.errstate(invalid="ignore", divide="ignore"):
        coh = np.abs(out["ifg"]) / np.sqrt(out["p1"] * out["p2"])
    ds = xr.Dataset({**{v: (("y", "x"), a) for v, a in out.items()},
                     "coherence": (("y", "x"), coh)}, coords={"y": ys, "x": xs},
                    attrs=dict(cells[0].attrs))
    ds.attrs["merged_from"] = len(cells)
    return ds


def cell_average(field, x, y, cells, weights=None):
    """Average a fine-grid field (``y`` x ``x`` pixel centres) over the cells of ``cells``.

    NaN-aware; pixels belong to the cell whose square contains their centre. ``weights``
    (same shape as ``field``) default to 1 on finite pixels.
    """
    field = np.asarray(field, float)
    w = np.isfinite(field).astype(float) if weights is None else np.where(
        np.isfinite(field), np.asarray(weights, float), 0.0)
    cell = float(cells.attrs["cell"])
    xc, yc = cells.x.values, cells.y.values
    ix = np.floor((np.asarray(x) - (xc[0] - cell / 2)) / cell).astype(int)
    iy = np.floor(((yc[0] + cell / 2) - np.asarray(y)) / cell).astype(int)
    okx = (ix >= 0) & (ix < xc.size)
    oky = (iy >= 0) & (iy < yc.size)
    num = np.zeros((yc.size, xc.size))
    den = np.zeros((yc.size, xc.size))
    sub_f = np.where(w > 0, field, 0.0)[np.ix_(oky, okx)]
    sub_w = w[np.ix_(oky, okx)]
    IY, IX = np.meshgrid(iy[oky], ix[okx], indexing="ij")
    np.add.at(num, (IY, IX), sub_f * sub_w)
    np.add.at(den, (IY, IX), sub_w)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan), den


# ---------------------------------------------------------------------------
# 2-3. The dispersive residual, and its unwrapping
# ---------------------------------------------------------------------------
def dispersive_residual(cells_a, cells_b, phase_a):
    """``(phi_A, r)`` on the cells: ``phi_A`` re-wrapped onto the caller's unwrapped ``phase_a``
    (same cells), ``r = wrap(phi_B - (f_B/f_A) phi_A)`` (= ``k d`` wrapped; no deformation)."""
    f_a = float(cells_a.attrs["center_frequency"])
    f_b = float(cells_b.attrs["center_frequency"])
    ref = np.asarray(phase_a, float)
    phi_a = ref + wrap(np.angle(cells_a.ifg.values) - ref)
    r = wrap(np.angle(cells_b.ifg.values) - (f_b / f_a) * phi_a)
    return phi_a, r


def _nan_gauss(values, valid, sigma):
    num = ndimage.gaussian_filter(np.where(valid, values, 0.0), sigma, mode="nearest")
    den = ndimage.gaussian_filter(valid.astype(float), sigma, mode="nearest")
    return num, den


def unwrap_smooth_phase(r, valid, sigma=3.0, coherence=None):
    """Unwrap a wrapped field that is smooth at the scale ``sigma`` (cells), through gaps.

    The complex field ``exp(i r)`` is low-passed (NaN-aware Gaussian; gaps filled by wider
    ones), the low-passed phase is unwrapped by SNAPHU (it holds no residues worth the name),
    and ``wrap(r - lowpass)`` is added back. Returns the unwrapped field (NaN where not
    ``valid``) and the low-passed unwrapped phase everywhere.
    """
    import snaphu

    valid = valid & np.isfinite(r)
    z = np.where(valid, np.exp(1j * np.where(valid, r, 0.0)), 0.0)
    re, den = _nan_gauss(z.real, valid, sigma)
    im, _ = _nan_gauss(z.imag, valid, sigma)
    s = sigma
    while (den < 1e-3).any() and s < 4 * max(r.shape):
        s *= 2.0
        re2, den2 = _nan_gauss(z.real, valid, s)
        im2, _ = _nan_gauss(z.imag, valid, s)
        gap = den < 1e-3
        re, im, den = np.where(gap, re2, re), np.where(gap, im2, im), np.where(gap, den2, den)
    smooth = np.exp(1j * np.angle(re + 1j * im)).astype(np.complex64)
    corr = np.full(r.shape, 0.95, np.float32)
    unw, _ = snaphu.unwrap(smooth, corr, nlooks=50.0, cost="smooth", init="mcf")
    unw = np.asarray(unw, float)
    out = np.where(valid, unw + wrap(r - unw), np.nan)
    return out, unw


# ---------------------------------------------------------------------------
# 4. ISCE3's IonosphereFilter.low_pass_filter, sized in metres
# ---------------------------------------------------------------------------
def _remove_singletons(valid):
    lab, n = ndimage.label(valid, structure=np.ones((3, 3)))
    if n == 0:
        return valid
    sizes = ndimage.sum(valid, lab, index=np.arange(1, n + 1))
    keep = np.zeros(n + 1, bool)
    keep[1:] = sizes >= 2
    return keep[lab]


def _guide(x, sigma):
    """``nanmedian 3x3`` then a NaN-aware Gaussian (``median_gaussian``)."""
    from scipy.ndimage import generic_filter

    med = generic_filter(x, np.nanmedian, size=3, mode="constant", cval=np.nan)
    ok = np.isfinite(med)
    num = ndimage.gaussian_filter(np.where(ok, med, 0.0), sigma)
    den = ndimage.gaussian_filter(ok.astype(float), sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 1e-6, num / den, np.nan)


def _fill_nearest(a):
    bad = ~np.isfinite(a)
    if not bad.any() or bad.all():
        return a
    idx = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
    return a[tuple(idx)]


def _plane_guide(x, sigma):
    """``_guide`` with a local plane in place of the Gaussian mean: nanmedian 3 x 3, then `_plane_fill`."""
    from scipy.ndimage import generic_filter

    med = generic_filter(x, np.nanmedian, size=3, mode="constant", cval=np.nan)
    return _plane_fill(med, np.isfinite(med).astype(float), sigma)


def _plane_fill(data, weight, sigma):
    """Local-plane estimate of ``data`` at every cell (first-order normalized convolution): at each cell, the plane
    through the cells with ``weight`` > 0, weighted by ``weight`` x a Gaussian of ``sigma`` cells (cut at 3 sigma).
    Exact for a linear field, at a one-sided edge and across a gap too, where the zero-order (weighted-mean) guide lags
    the gradient by ~ the gradient x the distance to its kernel's centroid of valid cells. NaN where the 3 x 3 system
    is degenerate (no data in reach, or data on a line)."""
    r = int(np.ceil(3.0 * sigma))
    t = np.arange(-r, r + 1, dtype=float)
    g = np.exp(-0.5 * (t / sigma) ** 2)
    k = {p: g * t ** p for p in (0, 1, 2)}

    def corr(f, p, q):                    # sum_j g(dx) g(dy) dx^p dy^q f_j, dx = x_j - x_i in cells
        return ndimage.correlate1d(ndimage.correlate1d(f, k[p], axis=1, mode="constant"), k[q], axis=0, mode="constant")

    ok_in = np.isfinite(data) & np.isfinite(weight) & (weight > 0)
    w = np.where(ok_in, weight, 0.0)
    wd = w * np.where(ok_in, data, 0.0)
    m00, m10, m01, m20, m11, m02 = (corr(w, p, q) for p, q in ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)))
    r00, r10, r01 = (corr(wd, p, q) for p, q in ((0, 0), (1, 0), (0, 1)))
    M = np.stack([np.stack([m00, m10, m01], -1), np.stack([m10, m20, m11], -1), np.stack([m01, m11, m02], -1)], -2)
    det = np.linalg.det(M)
    ok = (m00 > 1e-6 * m00.max()) & (det > 1e-9 * np.maximum(m00, 1e-300) ** 3 * sigma ** 4)
    out = np.full(np.shape(data), np.nan)
    if ok.any():
        rhs = np.stack([r00, r10, r01], -1)[ok][..., None]
        out[ok] = np.linalg.solve(M[ok], rhs)[:, 0, 0]
    return out


def ionosphere_filter(data, sigma, mask, cell, sigma_m=(2000.0, 2600.0), guide_m=None,
                      truncate=(1.75, 1.49), outlier_threshold=3.5, mad_scale=1.4826, fill="guide", fill_m=None,
                      fill_reach_m=None):
    """``IonosphereFilter.low_pass_filter`` (one iteration) on a map-cell grid.

    ``fill``: how masked cells are filled before the final Gaussian. ``"guide"`` is ISCE3's (the guide, a zero-order
    weighted mean, nearest beyond its reach); ``"plane"`` a local plane through the cleaned cells (`_plane_fill`, at the
    guide's sigma; the guide beyond its reach). At a one-sided data edge the guide lags the field's gradient: on
    D126 (Venezuela, the coast along the trace plus the 2 km exclusion) a noise-free smooth field came out +25.7 mm
    LOS high at the trace (2026-10-14). ``fill_m``: the plane's sigma in metres (default the guide's). The plane reaches
    3 sigma, and beyond it the guide's constant takes over: a wider plane follows the gradient across a wide gap (an
    excluded near-range strip, ~11 km on A004). ``fill_reach_m`` caps how far from the data the plane may extrapolate
    (beyond it the guide, nearest beyond the guide's reach): near its 3-sigma limit a wide plane is poorly constrained
    (A004's islands, 20-30 km offshore, came out 4-10 rad off at fill_m 10 km).

    ``data``/``sigma``/``mask`` are ``(y, x)`` cells of size ``cell`` m. ``sigma_m`` = Gaussian
    sigma ``(x, y)`` in metres -- ISCE3's 4 x 37 B-grid pixels are ~1.9-2.6 km of ground range
    by 2.6 km along track on NISAR's swath, and x/y are close to range/azimuth in a UTM grid of
    a polar orbit. ``truncate`` is ISCE3's (+-7 and +-55 taps: 1.75 and 1.49 sigma). The guide
    is isotropic at the larger sigma (ISCE3's ``max(sig_x, sig_y)``) unless ``guide_m`` is given.

    Steps (ISCE3's): masked cells NaN, singletons dropped from the support; guide = NaN-aware
    3 x 3 median then Gaussian; outliers ``|r - med| > 3.5 * 1.4826 MAD`` of the residual from
    the guide removed **from the guide only**; masked cells filled from the guide (nearest
    beyond its reach) with the nearest valid sigma; the original valid values restored; the
    inverse-variance-weighted truncated Gaussian; ISCE3's output sigma
    ``sqrt((sum K^2 W / sum K^2) / (sum K W / sum K)^2)``. Returns ``(screen, sigma_out)``.
    """
    sx, sy = (s / float(cell) for s in sigma_m)
    sg = (max(sigma_m) if guide_m is None else float(guide_m)) / float(cell)
    sf = sg if fill_m is None else float(fill_m) / float(cell)
    X = np.where(mask, np.asarray(data, float), np.nan)
    S = np.where(mask, np.asarray(sigma, float), np.nan)
    S = np.where(S > 0, S, np.nan)
    V = _remove_singletons(np.isfinite(X))
    O = np.where(V, X, np.nan)
    S_O = np.where(V, S, np.nan)
    G = _guide(X, sg)
    if fill == "plane":                 # judge outliers against a plane: the zero-order guide lags a gradient at an
        G_t = _plane_guide(X, sg)        # edge, and at low noise the lag itself would be flagged
    else:
        G_t = G
    R = X - G_t
    good = V & np.isfinite(R)
    if good.any():
        med = np.median(R[good])
        scale = mad_scale * np.median(np.abs(R[good] - med))
        if scale > 0:
            out = good & (np.abs(R - med) > outlier_threshold * scale)
            if out.any():
                X = np.where(out, np.nan, X)
                S = np.where(out, np.nan, S)
                G = _guide(X, sg)
    if fill == "plane":
        P = _plane_fill(np.where(V, X, np.nan), 1.0 / np.where(np.isfinite(S), S, np.inf) ** 2, sf)
        if fill_reach_m is not None and V.any():
            P = np.where(ndimage.distance_transform_edt(~V) * float(cell) > float(fill_reach_m), np.nan, P)
        Y = _fill_nearest(np.where(np.isfinite(P), P, G))
    elif fill == "guide":
        Y = _fill_nearest(G)
    else:
        raise ValueError(f"fill must be 'guide' or 'plane', not {fill!r}")
    S_Y = _fill_nearest(S)
    Y = np.where(V, O, Y)
    S_Y = np.where(V, S_O, S_Y)
    fallback = np.nanmedian(S_O) if np.isfinite(S_O).any() else 1.0
    S_Y = np.where(np.isfinite(S_Y) & (S_Y > 0), S_Y, fallback)
    W = 1.0 / S_Y ** 2
    rx, ry = int(truncate[0] * sx), int(truncate[1] * sy)
    kx = np.exp(-0.5 * (np.arange(-rx, rx + 1) / sx) ** 2)
    ky = np.exp(-0.5 * (np.arange(-ry, ry + 1) / sy) ** 2)

    def conv(a, kxx, kyy):
        b = ndimage.convolve1d(a, kxx, axis=1, mode="nearest")
        return ndimage.convolve1d(b, kyy, axis=0, mode="nearest")

    YW = np.where(np.isfinite(Y), Y, 0.0) * W
    W = np.where(np.isfinite(Y), W, 0.0)
    num = conv(YW, kx, ky)
    den = conv(W, kx, ky)
    den2 = conv(W, kx * kx, ky * ky)
    with np.errstate(invalid="ignore", divide="ignore"):
        screen = num / den
        sigma_out = np.sqrt((den2 / (kx * kx).sum() / (ky * ky).sum())
                            / (den / kx.sum() / ky.sum()) ** 2)
    return screen, sigma_out


# ---------------------------------------------------------------------------
# The estimate
# ---------------------------------------------------------------------------
def fault_distance(cells, trace_x, trace_y):
    """Distance (m) of every cell centre to a polyline ``trace_x``, ``trace_y`` (map metres)."""
    X, Y = np.meshgrid(cells.x.values, cells.y.values)
    P = np.stack([X.ravel(), Y.ravel()], axis=1)
    best = np.full(P.shape[0], np.inf)
    tx, ty = np.asarray(trace_x, float), np.asarray(trace_y, float)
    for k in range(tx.size - 1):
        a = np.array([tx[k], ty[k]])
        b = np.array([tx[k + 1], ty[k + 1]])
        ab = b - a
        t = np.clip(((P - a) @ ab) / max(ab @ ab, 1e-12), 0.0, 1.0)
        d = np.hypot(*(P - (a + t[:, None] * ab)).T)
        best = np.minimum(best, d)
    return best.reshape(X.shape)


def split_spectrum_screen(cells_a, cells_b, phase_a, *, valid=None, exclude=None,
                          min_coherence=0.2, min_pixels_fraction=0.5, sigma_m=(2000.0, 2600.0),
                          unwrap_sigma=3.0, resolution_a=(6.2, 5.3), resolution_b=(50.0, 5.3), fill="guide",
                          fill_m=None, fill_reach_m=None):
    """The dispersive (ionospheric) phase at ``f_A``, filtered, on the cells.

    ``cells_a``/``cells_b`` are :func:`gslc_cells` of the same cells for the main (A) and side
    (B) band; ``phase_a`` the caller's unwrapped A phase averaged onto those cells
    (:func:`cell_average`) -- this is where the fault-aware unwrapping enters. ``valid`` masks
    cells to use (e.g. land, and where ``phase_a`` is trusted); ``exclude`` cells to drop
    (e.g. ``fault_distance(...) < 2000``). Cells also need coherence > ``min_coherence`` in
    both bands and at least ``min_pixels_fraction`` of their pixels valid in both.
    ``resolution_a``/``_b`` are (ground range, azimuth) resolutions in metres for the looks of
    the uncertainty (NISAR 40 + 5 MHz near 37 deg incidence: 6.2 / 50 m by 5.3 m).

    Returns an :class:`xarray.Dataset` on the cells: ``screen`` (filtered dispersive phase at
    ``f_A``, rad -- subtract it from the A phase), ``sigma`` (ISCE3's output sigma), the raw
    ``dispersive`` estimate and ``nondispersive`` (= phi_A - dispersive) on the cells used,
    ``mask``, and ``residual`` (r, wrapped). ``fill`` is :func:`ionosphere_filter`'s (``"plane"`` for an
    unbiased fill at one-sided edges: a coast along a fault, an excluded fault zone, a swath edge), ``fill_m`` the
    plane's sigma (m) and ``fill_reach_m`` how far from the data it may extrapolate.
    """
    f_a = float(cells_a.attrs["center_frequency"])
    f_b = float(cells_b.attrs["center_frequency"])
    k = dispersive_factor(f_a, f_b)
    phi_a, r = dispersive_residual(cells_a, cells_b, phase_a)
    na = cells_a.n.values / float(cells_a.attrs["pixels_per_cell"])
    nb = cells_b.n.values / float(cells_b.attrs["pixels_per_cell"])
    ga, gb = cells_a.coherence.values, cells_b.coherence.values
    use = (np.isfinite(phi_a) & np.isfinite(r) & (ga > min_coherence) & (gb > min_coherence)
           & (na >= min_pixels_fraction) & (nb >= min_pixels_fraction))
    if valid is not None:
        use &= np.asarray(valid, bool)
    if exclude is not None:
        use &= ~np.asarray(exclude, bool)
    r_unw, _ = unwrap_smooth_phase(r, use, sigma=unwrap_sigma)
    d = r_unw / k
    # ISCE3's per-pixel sigma: each band's Cramer-Rao phase sigma through d = c_a phi_A - c_b phi_B,
    # with looks = independent resolution cells in the cell (GSLC pixels are interpolated)
    c_a = f_b * f_b / (f_b * f_b - f_a * f_a)
    c_b = f_a * f_b / (f_b * f_b - f_a * f_a)
    cell = float(cells_a.attrs["cell"])
    looks_a = np.maximum(na * cell * cell / (resolution_a[0] * resolution_a[1]), 1.0)
    looks_b = np.maximum(nb * cell * cell / (resolution_b[0] * resolution_b[1]), 1.0)
    sig = np.hypot(c_a * phase_sigma(ga, looks_a), c_b * phase_sigma(gb, looks_b))
    screen, sig_out = ionosphere_filter(d, sig, use, cell, sigma_m=sigma_m, fill=fill, fill_m=fill_m,
                                        fill_reach_m=fill_reach_m)
    dims = ("y", "x")
    return xr.Dataset(
        {"screen": (dims, screen), "sigma": (dims, sig_out),
         "dispersive": (dims, np.where(use, d, np.nan)),
         "nondispersive": (dims, np.where(use, phi_a - d, np.nan)),
         "phase_a": (dims, phi_a), "residual": (dims, r), "mask": (dims, use),
         "coherence_a": (dims, ga), "coherence_b": (dims, gb)},
        coords={"y": cells_a.y, "x": cells_a.x},
        attrs={"f_a": f_a, "f_b": f_b, "dispersive_factor": k, "cell": cell,
               "sigma_m": tuple(sigma_m), "min_coherence": min_coherence, "fill": fill,
               "fill_m": float(max(sigma_m) if fill_m is None else fill_m),
               "fill_reach_m": float(3.0 * (max(sigma_m) if fill_m is None else fill_m) if fill_reach_m is None
                                     else fill_reach_m),
               "method": "main/side band, matched cells, dispersive-only unwrapping "
                         "(ISCE3 main_diff_ms_band estimator, re-ordered)",
               "convention": "screen is the ionospheric phase at f_A in the "
                             "reference*conj(secondary) convention: subtract it"})
