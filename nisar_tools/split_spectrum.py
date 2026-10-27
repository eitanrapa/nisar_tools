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
   grids, cells on a lattice both grids share) -- and, given the caller's unwrapped main band as a
   ``phase_model``, with it taken out of **every pixel** of both bands first, so that steep fringes
   inside a cell neither decorrelate it nor weight the two bands' cell phases differently;
2. ``phi_A`` comes from the caller's own unwrapping (fault-aware, cut and tied), and only the
   **purely dispersive** residual is unwrapped::

       r = phi_B - (f_B/f_A) phi_A = (f_A/f_B - f_B/f_A) d = -0.0861 d

   which holds no deformation at all, and is continuous across the fault;
3. no cycle is moved against a smoothed non-dispersive field; cells near the trace (or of low
   coherence in either band) are masked and filled by the filter, from both sides;
4. the filter is ISCE3's ``IonosphereFilter.low_pass_filter`` (guide, outliers, fill, inverse-
   variance-weighted truncated Gaussian), sized in metres instead of B-grid pixels; optionally
   with each gap filled by the local plane that interpolates it (a ladder of widths), and with
   its final Gaussian made first order -- a weighted local plane, which does not lag the
   ionosphere's gradient where the weights change, as they do across a rupture's damage zone;
5. a block the unwrapper saw on its own (cut off by a rupture, an island) is tied to the main
   region by the side band itself (:func:`block_cycles`): its main-band integer moves its raw
   estimate by 3.21 rad per cycle, and the ionosphere is continuous across the gap;
6. so does a patch the unwrapper got a cycle wrong: where two unwrappings of the pair disagree
   (:func:`cycle_patches`), the side band reads which one is right (:func:`patch_cycles`,
   :func:`patch_verdict`) -- for the main band's own data too, at 2 pi a cycle.

The phase convention is the package's interferogram, ``reference * conj(secondary)``.
"""

import warnings

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


def _model_reader(model):
    """``at(xs, ys)`` -> a ``(y, x)`` DataArray ``model`` bilinear at the pixel centres ``ys`` x ``xs`` (1-D), NaN off its
    grid or next to a NaN; and a description holding a hash of the values."""
    import hashlib

    mx = np.asarray(model["x"].values, float)
    my = np.asarray(model["y"].values, float)
    mv = np.asarray(model.values, float)
    if mv.ndim != 2 or mx.size < 2 or my.size < 2:
        raise ValueError("phase_model must be a (y, x) DataArray of at least 2 x 2")
    if my[0] > my[-1]:
        my, mv = my[::-1], mv[::-1]
    if mx[0] > mx[-1]:
        mx, mv = mx[::-1], mv[:, ::-1]
    dx, dy = mx[1] - mx[0], my[1] - my[0]
    if not (np.allclose(np.diff(mx), dx) and np.allclose(np.diff(my), dy)):
        raise ValueError("phase_model must be on a regular grid")

    def axis(t, t0, dt, n):
        f = (np.asarray(t, float) - t0) / dt
        i = np.clip(np.floor(f).astype(np.int64), 0, n - 2)
        off = (f < -1e-9) | (f > n - 1 + 1e-9)
        return i, np.where(off, np.nan, f - i)

    def at(xs, ys):
        ix, wx = axis(xs, mx[0], dx, mx.size)
        iy, wy = axis(ys, my[0], dy, my.size)
        wx, wy = wx.astype(np.float32), wy.astype(np.float32)
        rows0, rows1 = mv[iy], mv[iy + 1]
        return ((rows0[:, ix] * (1 - wx) + rows0[:, ix + 1] * wx) * (1 - wy)[:, None]
                + (rows1[:, ix] * (1 - wx) + rows1[:, ix + 1] * wx) * wy[:, None])

    digest = hashlib.sha256(np.ascontiguousarray(mv, dtype=np.float32).tobytes()).hexdigest()[:16]
    desc = str(model.attrs.get("description", "phase model")) + f" [sha256 {digest}, {mv.shape[0]} x {mv.shape[1]}]"
    mv = mv.astype(np.float32)          # a strip of band A is ~10^7 pixels: float32 halves it (1e-4 rad at 10^3 rad)
    return at, desc


def gslc_cells(reference, secondary, frequency, cell=240.0, polarization="HH", bounds=None,
               strip_cells=8, subswath_mask=False, near_range_m=11000.0, near_range_side=None,
               phase_model=None, model_frequency=None):
    """One band's interferogram summed over map cells, from two GSLCs on one grid.

    ``reference``/``secondary`` are GSLC paths whose ``frequency`` grids coincide (a pair made
    with ``geocode_rslc(like=)``). ``bounds = (x_min, y_min, x_max, y_max)`` limits the cells.
    Returns an :class:`xarray.Dataset` on cell centres ``(y, x)``: the complex sum ``ifg`` of
    ``ref * conj(sec)``, the powers ``p1``/``p2``, the valid-pixel count ``n``, and the
    ``coherence``.

    ``phase_model`` (optional): the caller's unwrapped main-band phase (rad at ``model_frequency``, Hz; default its
    ``center_frequency`` attribute), a ``(y, x)`` DataArray on a regular map grid in the GSLCs' CRS. Each pixel's
    ``ref * conj(sec)`` is turned by ``exp(-i (f_c / model_frequency) model)``, bilinear at the pixel, before the sums,
    so a cell holds the phase relative to the model, scaled to its band: the deformation's fringes neither decorrelate
    a cell nor weight its two bands differently (the side band's pixels are 8 times wider in range, and its speckle is
    not the main band's). Pixels where the model is NaN or off its grid do not count. The cells then also carry
    ``model``, the model's mean over the counted pixels, and :func:`dispersive_residual` reads them as such.

    Pixels that are zero or non-finite in either date do not count, nor (``subswath_mask``)
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
    model_at = scale = model_desc = None
    if phase_model is not None:
        model_frequency = model_frequency if model_frequency is not None else phase_model.attrs.get("center_frequency")
        if model_frequency is None or f_c is None:
            raise ValueError("gslc_cells: phase_model needs model_frequency (or a center_frequency attribute) and the "
                             "GSLC's centerFrequency")
        model_at, model_desc = _model_reader(phase_model)
        scale = float(f_c) / float(model_frequency)
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
    msum = np.zeros(shape) if model_at is not None else None
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
            if model_at is not None:
                mod = model_at(x[xs], y[ys])
                ok &= np.isfinite(mod)
                mod = np.where(ok, mod, 0.0)
                g2 = g2 * np.exp(1j * scale * mod).astype(np.complex64)     # conj(sec e^{i s m}) = conj(sec) e^{-i s m}
            g1 = np.where(ok, g1, 0)
            g2 = np.where(ok, g2, 0)
            sh = (b - a, ky, c1 - c0, kx)
            sl = slice(a - r0, b - r0)
            ifg[sl] = (g1 * np.conj(g2)).reshape(sh).sum(axis=(1, 3), dtype=np.complex128)
            p1[sl] = (np.abs(g1) ** 2).reshape(sh).sum(axis=(1, 3), dtype=np.float64)
            p2[sl] = (np.abs(g2) ** 2).reshape(sh).sum(axis=(1, 3), dtype=np.float64)
            n[sl] = ok.reshape(sh).sum(axis=(1, 3))
            if msum is not None:
                msum[sl] = mod.reshape(sh).sum(axis=(1, 3), dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        coh = np.abs(ifg) / np.sqrt(p1 * p2)
    data = {"ifg": (("y", "x"), ifg), "p1": (("y", "x"), p1), "p2": (("y", "x"), p2),
            "n": (("y", "x"), n), "coherence": (("y", "x"), coh)}
    model_attrs = {}
    if msum is not None:
        with np.errstate(invalid="ignore", divide="ignore"):
            data["model"] = (("y", "x"), np.where(n > 0, msum / np.maximum(n, 1), np.nan))
        model_attrs = {"phase_model": model_desc, "model_frequency": float(model_frequency), "model_scale": scale}
    return xr.Dataset(
        data,
        coords={"y": yc[r0:r1], "x": xc[c0:c1]},
        attrs={"frequency": frequency, "center_frequency": f_c, "cell": float(cell),
               "pixels_per_cell": kx * ky, "reference": str(reference),
               "secondary": str(secondary), "polarization": polarization,
               "subswath_mask": subswath_mask if near else bool(subswath_mask),
               **({"near_range_m": float(near_range_m), "near_range_side": near_range_side} if near else {}),
               **model_attrs})


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
    """Sum cell datasets of one band from several frames onto their union (cells coincide). Cells made with a
    ``phase_model`` keep its mean over the merged pixels; they cannot be merged with cells made without it."""
    xs = np.unique(np.concatenate([c.x.values for c in cells]))
    ys = np.unique(np.concatenate([c.y.values for c in cells]))[::-1]
    models = {c.attrs.get("phase_model") for c in cells}
    if len(models) > 1:
        raise ValueError(f"merge_cells: the cells were made with different phase models: {sorted(map(str, models))}")
    has_model = models != {None}
    out = {v: np.zeros((ys.size, xs.size), dt) for v, dt in
           (("ifg", np.complex128), ("p1", float), ("p2", float), ("n", np.int64))}
    msum = np.zeros((ys.size, xs.size)) if has_model else None
    for c in cells:
        iy = np.searchsorted(-ys, -c.y.values)
        ix = np.searchsorted(xs, c.x.values)
        for v in out:
            out[v][np.ix_(iy, ix)] += c[v].values
        if has_model:
            msum[np.ix_(iy, ix)] += np.where(c.n.values > 0, c.model.values * c.n.values, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        coh = np.abs(out["ifg"]) / np.sqrt(out["p1"] * out["p2"])
    data = {**{v: (("y", "x"), a) for v, a in out.items()}, "coherence": (("y", "x"), coh)}
    if has_model:
        data["model"] = (("y", "x"), np.where(out["n"] > 0, msum / np.maximum(out["n"], 1), np.nan))
    ds = xr.Dataset(data, coords={"y": ys, "x": xs}, attrs=dict(cells[0].attrs))
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
def dispersive_residual(cells_a, cells_b, phase_a=None):
    """``(phi_A, r)`` on the cells: ``phi_A`` re-wrapped onto the caller's unwrapped ``phase_a``
    (same cells), ``r = wrap(phi_B - (f_B/f_A) phi_A)`` (= ``k d`` wrapped; no deformation).

    Cells made with a ``phase_model`` (:func:`gslc_cells`) hold each band's phase relative to the model, so
    ``r = wrap(rho_B - (f_B/f_A) rho_A)`` with ``rho`` the cells' angles, taken pixel by pixel, and ``phi_A`` is the
    model's mean plus ``rho_A``. ``phase_a`` is then optional: given, it only moves each cell's integer to the one
    nearest it (a block re-tied at the cells), at ``(f_B/f_A) 2 pi`` of ``r`` per cycle as before."""
    f_a = float(cells_a.attrs["center_frequency"])
    f_b = float(cells_b.attrs["center_frequency"])
    with_model = ("model" in cells_a, "model" in cells_b)
    if any(with_model):
        if not all(with_model) or cells_a.attrs.get("phase_model") != cells_b.attrs.get("phase_model"):
            raise ValueError("dispersive_residual: both bands' cells must be made with the same phase_model")
        rho_a, rho_b = np.angle(cells_a.ifg.values), np.angle(cells_b.ifg.values)
        base = (f_a / float(cells_a.attrs["model_frequency"])) * cells_a.model.values + rho_a
        phi_a = base if phase_a is None else np.asarray(phase_a, float) + wrap(base - np.asarray(phase_a, float))
        r = wrap(rho_b - (f_b / f_a) * (rho_a + (phi_a - base)))
        return phi_a, r
    if phase_a is None:
        raise ValueError("dispersive_residual: phase_a is needed unless the cells carry a phase_model")
    ref = np.asarray(phase_a, float)
    phi_a = ref + wrap(np.angle(cells_a.ifg.values) - ref)
    r = wrap(np.angle(cells_b.ifg.values) - (f_b / f_a) * phi_a)
    return phi_a, r


def _nan_gauss(values, valid, sigma):
    num = ndimage.gaussian_filter(np.where(valid, values, 0.0), sigma, mode="nearest")
    den = ndimage.gaussian_filter(valid.astype(float), sigma, mode="nearest")
    return num, den


def _lowpass_filled(r, valid, sigma):
    """The phase of ``exp(i r)`` low-passed (NaN-aware Gaussian of ``sigma`` cells, gaps filled by
    ones twice as wide, and so on), and the first Gaussian's reach (``den >= 1e-3``: ~2 km past a
    data edge at 240 m cells and sigma 3)."""
    z = np.where(valid, np.exp(1j * np.where(valid, r, 0.0)), 0.0)
    re, den = _nan_gauss(z.real, valid, sigma)
    im, _ = _nan_gauss(z.imag, valid, sigma)
    reach = den >= 1e-3
    s = sigma
    while (den < 1e-3).any() and s < 4 * max(r.shape):
        s *= 2.0
        re2, den2 = _nan_gauss(z.real, valid, s)
        im2, _ = _nan_gauss(z.imag, valid, s)
        gap = den < 1e-3
        re, im, den = np.where(gap, re2, re), np.where(gap, im2, im), np.where(gap, den2, den)
    return np.angle(re + 1j * im), reach


def _unwrap_guide(phase, reach, valid):
    """SNAPHU on the low-passed ``phase``: over the whole grid, then over ``reach`` alone, each piece
    of the reach holding ``valid`` cells shifted onto the whole-grid solution by its most common
    whole-cycle difference there. Outside the reach, the whole-grid solution."""
    import snaphu

    smooth = np.exp(1j * phase).astype(np.complex64)
    corr = np.full(phase.shape, 0.95, np.float32)
    unw = np.asarray(snaphu.unwrap(smooth, corr, nlooks=50.0, cost="smooth", init="mcf")[0], float)
    if reach.all():
        return unw
    local = np.asarray(snaphu.unwrap(smooth, corr, nlooks=50.0, cost="smooth", init="mcf", mask=reach)[0], float)
    lab, n = ndimage.label(reach)
    out = unw.copy()
    moved = 0
    for p in range(1, n + 1):
        on = lab == p
        sel = on & valid
        if not sel.any():
            continue
        cyc = np.round((unw[sel] - local[sel]) / (2.0 * np.pi)).astype(int)
        mode = int(np.bincount(cyc - cyc.min()).argmax() + cyc.min())
        out[on] = local[on] + 2.0 * np.pi * mode
        moved += int((cyc != mode).sum())
    if moved:
        warnings.warn(f"unwrap_smooth_phase: the whole-grid unwrapping cut across the data at {moved} of "
                      f"{int(valid.sum())} cells (a phase singularity in a gap's fill): they take the unwrapping "
                      "over the data's reach", RuntimeWarning, stacklevel=3)
    return out


def unwrap_smooth_phase(r, valid, sigma=3.0, coherence=None):
    """Unwrap a wrapped field that is smooth at the scale ``sigma`` (cells), through gaps.

    The complex field ``exp(i r)`` is low-passed (NaN-aware Gaussian; gaps filled by wider
    ones), the low-passed phase is unwrapped by SNAPHU, and ``wrap(r - lowpass)`` is added back.
    Returns the unwrapped field (NaN where not ``valid``) and the low-passed unwrapped phase
    everywhere.

    SNAPHU runs twice (2026-10-17). The wide Gaussians that fill a large gap (a bay, the sea past a
    swath edge) average data whose phases differ by up to pi, and can leave a phase singularity
    there, tens of km from any cell. Unwrapped over the whole grid, SNAPHU must cut from it to the
    grid's edge, and it takes the shortest route, which may cross the data. On D040 (Panama) a
    singularity in Parita Bay, 28 km from the nearest cell, put one cycle of ``r`` (73 rad of screen,
    1.4 m of LOS) across the 20 km neck joining the Azuero peninsula to the mainland. So the
    low-passed phase is also unwrapped over the first Gaussian's reach alone, which holds no such
    singularity, and each piece of the reach takes the whole-grid solution's integer: pieces further
    apart than the reach are still tied through the fill, as before, but within a piece the fill
    cannot cut. Without a singularity the two runs agree, so the result is unchanged (checked on
    D141, A004 and the four Venezuela tracks' v6 screens: identical).
    """
    valid = valid & np.isfinite(r)
    lp, reach = _lowpass_filled(r, valid, sigma)
    unw = _unwrap_guide(lp, reach, valid)
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


def _plane_fill(data, weight, sigma, lever=False):
    """Local-plane estimate of ``data`` at every cell (first-order normalized convolution): at each cell, the plane
    through the cells with ``weight`` > 0, weighted by ``weight`` x a Gaussian of ``sigma`` cells (cut at 3 sigma).
    Exact for a linear field, at a one-sided edge and across a gap too, where the zero-order (weighted-mean) guide lags
    the gradient by ~ the gradient x the distance to its kernel's centroid of valid cells. NaN where the 3 x 3 system
    is degenerate (no data in reach, or data on a line). ``lever``: also return how far that centroid lies from each
    cell, in sigmas -- ~0 where the plane interpolates (data around the cell), ~d / sigma where it extrapolates d past
    a one-sided edge."""
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
    if lever:
        with np.errstate(invalid="ignore", divide="ignore"):
            return out, np.where(ok, np.hypot(m10, m01) / m00 / sigma, np.nan)
    return out


def _plane_ladder(data, weight, sigmas, lever_band):
    """`_plane_fill` at each of ``sigmas`` (cells, ascending), each cell taking the narrowest plane whose data lie
    around it: rung i's weight is 1 where its data's centroid lies within ``lever_band[0]`` sigmas of the cell, 0 beyond
    ``lever_band[1]``, linear between, the rest going to the wider rungs (the widest takes what is left)."""
    out = None
    for s in sorted(sigmas, reverse=True):
        p, lev = _plane_fill(data, weight, s, lever=True)
        if out is None:
            out = p
            continue
        t = np.clip((lever_band[1] - lev) / (lever_band[1] - lever_band[0]), 0.0, 1.0)
        t = np.where(np.isfinite(p) & np.isfinite(t), t, 0.0)
        out = np.where(np.isfinite(out), t * np.where(np.isfinite(p), p, 0.0) + (1.0 - t) * out, p)
    return out


def ionosphere_filter(data, sigma, mask, cell, sigma_m=(2000.0, 2600.0), guide_m=None,
                      truncate=(1.75, 1.49), outlier_threshold=3.5, mad_scale=1.4826, fill="guide", fill_m=None,
                      fill_reach_m=None, fill_lever=(1.5, 3.0), order=0):
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

    ``fill_m`` may also be a ladder of sigmas (metres), each filled cell taking the narrowest plane that interpolates
    there -- whose data's weighted centroid lies within ``fill_lever`` sigmas of it (`_plane_ladder`; blended between
    the two). A wide plane fits a curved ionosphere worse: across a 2 km fault zone the 10 km plane misses it by
    14-16 mm of LOS rms on synthetics, a 1.5-2 km plane by 2-3 mm (8-12 rad at 8 km of roughness: 56-65 mm against
    10-14). Only a one-sided gap -- an edge strip, which no plane fills well -- calls for a wider one, and the
    centroid tells the two apart: it stays at a two-sided gap's cells and runs away from a one-sided one's. On
    synthetic ionospheres the ladder (1.5, 3, 6, 12) km with ``fill_lever`` (1.5, 3) fills a 1-2 km fault zone with
    the 1.5 km plane alone and comes within 10-40% of the best single width in wider gaps and edge strips. Real
    ionosphere (34 fake gaps cut out of eight NISAR screens, 2026-10-21) is smoother at 2-10 km and its cells noisier,
    and there the 1.5 km rung is too narrow: a ladder from the guide's own width, (2.6, 5, 10) km, is best or tied in
    fault zones 1-4 km wide (1.1-3.6 mm of LOS rms; the 10 km plane 1.7-4.7) and close to the 10 km plane at a
    one-sided edge. All of these differ by about a millimetre in a 2 km zone. But a narrow fill also follows the data
    at a gap's edge more closely, local main-band cycle errors included: next to D141's rupture, where the cut
    unwrapping is a cycle off in patches north of the trace, it took them up (+8-10 mm of screen), where the 10 km
    plane moved by 2 mm or less when they were repaired.

    ``order``: the final Gaussian's. ISCE3's (0) is an inverse-variance weighted mean, and where the weights change
    -- across a fault's damage zone, whose coherence falls toward the trace -- its centroid leaves the cell and it lags
    the ionosphere's gradient: on synthetic ruptures by 8-15 mm of LOS at 1-5 km from the trace, opposite on the two
    sides. 1 fits a weighted plane through the same kernel instead (`_local_plane`, a weighted Savitzky-Golay step, as
    Piromthong et al. 2026 filter for scene edges), exact for a linear field whatever the weights; where they are
    symmetric about a cell it is the weighted mean. Its sigma is ISCE3's, with the plane's variance in place of the
    mean's. How much it matters depends on the ionosphere's gradient across the weights' change: on real fake gaps
    (no damage zone) at most 0.4 mm; next to D141's rupture 2-4 mm.

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
    sf = [sg] if fill_m is None else [float(f) / float(cell) for f in np.atleast_1d(fill_m)]
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
        W_in = 1.0 / np.where(np.isfinite(S), S, np.inf) ** 2
        if len(sf) == 1:
            P = _plane_fill(np.where(V, X, np.nan), W_in, sf[0])
        else:
            P = _plane_ladder(np.where(V, X, np.nan), W_in, sf, fill_lever)
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
    norm = (kx.sum() * ky.sum()) ** 2 / ((kx * kx).sum() * (ky * ky).sum())
    if order == 1:
        screen, var = _local_plane(np.where(np.isfinite(Y), Y, 0.0), W, kx, ky)
        with np.errstate(invalid="ignore"):
            return screen, np.sqrt(var * norm)
    if order != 0:
        raise ValueError(f"order must be 0 or 1, not {order!r}")
    num = conv(YW, kx, ky)
    den = conv(W, kx, ky)
    den2 = conv(W, kx * kx, ky * ky)
    with np.errstate(invalid="ignore", divide="ignore"):
        screen = num / den
        sigma_out = np.sqrt(den2 / den ** 2 * norm)
    return screen, sigma_out


def _local_plane(Y, W, kx, ky):
    """The final Gaussian made first order: at each cell, the plane through ``Y`` weighted by ``W`` x the separable
    kernel ``kx`` x ``ky`` (centred, truncated), evaluated at the cell -- a weighted Savitzky-Golay step. And its variance
    factor e0' M^-1 N M^-1 e0 (M the weighted moments, N the same with the squared kernel), which is ISCE3's
    ``sum K^2 W / (sum K W)^2`` wherever the weights are symmetric about the cell. Cells beyond the grid weigh nothing."""
    rx, ry = (kx.size - 1) // 2, (ky.size - 1) // 2
    tx, ty = np.arange(-rx, rx + 1, dtype=float), np.arange(-ry, ry + 1, dtype=float)

    def corr(a, gx, gy):                  # sum_j g(dx) g(dy) a_j, dx = x_j - x_i in cells
        return ndimage.correlate1d(ndimage.correlate1d(a, gx, axis=1, mode="constant"), gy, axis=0, mode="constant")

    pq = ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2))
    m = {k: corr(W, kx * tx ** k[0], ky * ty ** k[1]) for k in pq}
    n = {k: corr(W, kx * kx * tx ** k[0], ky * ky * ty ** k[1]) for k in pq}
    r = [corr(W * Y, kx * tx ** k[0], ky * ty ** k[1]) for k in pq[:3]]

    def sym(d):
        return np.stack([np.stack([d[0, 0], d[1, 0], d[0, 1]], -1), np.stack([d[1, 0], d[2, 0], d[1, 1]], -1),
                         np.stack([d[0, 1], d[1, 1], d[0, 2]], -1)], -2)

    M, N = sym(m), sym(n)
    ok = m[0, 0] > 0
    screen = np.full(W.shape, np.nan)
    var = np.full(W.shape, np.nan)
    if ok.any():
        Mi = np.linalg.inv(M[ok])
        screen[ok] = np.einsum("nj,nj->n", Mi[:, 0, :], np.stack(r, -1)[ok])
        var[ok] = np.einsum("ni,nij,nj->n", Mi[:, 0, :], N[ok], Mi[:, 0, :])
    return screen, var


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


def split_spectrum_screen(cells_a, cells_b, phase_a=None, *, valid=None, exclude=None,
                          min_coherence=0.2, min_pixels_fraction=0.5, sigma_m=(2000.0, 2600.0),
                          unwrap_sigma=3.0, resolution_a=(6.2, 5.3), resolution_b=(50.0, 5.3), fill="guide",
                          fill_m=None, fill_reach_m=None, fill_lever=(1.5, 3.0), order=0):
    """The dispersive (ionospheric) phase at ``f_A``, filtered, on the cells.

    ``cells_a``/``cells_b`` are :func:`gslc_cells` of the same cells for the main (A) and side
    (B) band; ``phase_a`` the caller's unwrapped A phase averaged onto those cells
    (:func:`cell_average`) -- this is where the fault-aware unwrapping enters. Cells made with that phase as their
    ``phase_model`` (pixel by pixel; the better way near a rupture) need no ``phase_a``: see
    :func:`dispersive_residual`. ``valid`` masks
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
    plane's sigma (m, or a ladder of them, with ``fill_lever``) and ``fill_reach_m`` how far from the data it may
    extrapolate; ``order`` the final step's (1: a weighted local plane, which does not lag a gradient where the weights
    change).
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
                                        fill_reach_m=fill_reach_m, fill_lever=fill_lever, order=order)
    dims = ("y", "x")
    return xr.Dataset(
        {"screen": (dims, screen), "sigma": (dims, sig_out),
         "dispersive": (dims, np.where(use, d, np.nan)),
         "nondispersive": (dims, np.where(use, phi_a - d, np.nan)),
         "phase_a": (dims, phi_a), "residual": (dims, r), "mask": (dims, use),
         "coherence_a": (dims, ga), "coherence_b": (dims, gb)},
        coords={"y": cells_a.y, "x": cells_a.x},
        attrs={"f_a": f_a, "f_b": f_b, "dispersive_factor": k, "cell": cell,
               "sigma_m": tuple(sigma_m), "min_coherence": min_coherence, "fill": fill, "order": int(order),
               "fill_m": (float(max(sigma_m)) if fill_m is None else float(fill_m) if np.ndim(fill_m) == 0
                          else tuple(float(f) for f in fill_m)),
               "fill_reach_m": float(3.0 * (max(sigma_m) if fill_m is None else np.max(fill_m)) if fill_reach_m is None
                                     else fill_reach_m),
               "method": "main/side band, matched cells, dispersive-only unwrapping "
                         "(ISCE3 main_diff_ms_band estimator, re-ordered)",
               "main_band_removed": "per pixel" if "model" in cells_a else "per cell",
               **({"phase_model": cells_a.attrs["phase_model"]} if "model" in cells_a else {}),
               "convention": "screen is the ionospheric phase at f_A in the "
                             "reference*conj(secondary) convention: subtract it"})


# ---------------------------------------------------------------------------
# 5. A block's main-band integer, from the side band
# ---------------------------------------------------------------------------
def _poly_columns(x, y, degree):
    cols = [np.ones_like(x)]
    for p in range(1, degree + 1):
        for q in range(p + 1):
            cols.append(x ** (p - q) * y ** q)
    return np.stack(cols, axis=1)


def _robust_lstsq(A, y, w0, iterations=30, k=4.685):
    """Weighted least squares reweighted by Tukey's biweight on the weighted residuals (``k`` scaled MADs). Returns the
    coefficients, the weights and the formal covariance scaled by the robust residual variance."""
    s0 = np.sqrt(w0)
    coef = np.linalg.lstsq(A * s0[:, None], y * s0, rcond=None)[0]
    w = w0
    for _ in range(iterations):
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
    keep = w > 0
    r = (y - A @ coef) * np.sqrt(w)
    s2 = float((r[keep] ** 2).sum() / max(int(keep.sum()) - A.shape[1], 1))
    try:
        cov = s2 * np.linalg.inv((A * w[:, None]).T @ A)
    except np.linalg.LinAlgError:
        cov = np.full((A.shape[1], A.shape[1]), np.nan)
    return coef, w, cov


def block_cycles(screen, block, reference, *, along=None, reach_m=12000.0, window_m=20000.0, degree=2,
                 min_cells=60, min_gap_m=0.0):
    """Whole cycles to add to ``block``'s main band so that its raw dispersive estimate continues ``reference``'s.

    A region the unwrapper saw on its own -- the block a cut along a rupture isolates, an island -- carries an
    arbitrary integer in phi_A. Taking the scaled main band out of the wrapped side band wraps away all of each such
    cycle but ``f_B / (f_A + f_B)`` of it, so a block whose integer is off by ``eps`` carries ``2 pi eps f_B / (f_A +
    f_B)`` (3.21 eps rad at NISAR's A/B) of extra raw estimate. The ionosphere is continuous across the gap between two
    blocks, and nothing else in the raw estimate steps there (the deformation is not in it), so its step across the gap
    measures ``eps``.

    ``screen`` is :func:`split_spectrum_screen`'s output; ``block`` and ``reference`` are boolean ``(y, x)`` masks on
    its cells holding every cell of each region, used or not (the boundary is measured from them: e.g.
    ``cell_average`` of a label's indicator, > 0.5). Cells of each within ``reach_m`` of the other and at least
    ``min_gap_m`` from it are tiled into windows of ``window_m`` along the boundary -- segments of ``along`` (m, on the
    cells; e.g. the arc length of each cell's nearest trace point) if given, squares otherwise. Each window holding
    ``min_cells`` of both is fitted with one polynomial of ``degree`` over both blocks plus a constant on ``block``,
    weighted by each cell's Cramer-Rao variance (full cells assumed: the screen does not keep the pixel counts) and
    reweighted by Tukey's biweight, because a patch with a local cycle error is 3.21 rad off and must not move the step.

    Returns a dict: ``cycles`` (the median over the windows, rounded), ``estimate`` (that median), ``spread`` (the
    windows' scaled MAD), ``ambiguous`` (the median more than 0.25 from ``cycles``, a spread over 0.25, or fewer than
    two windows) and
    ``windows`` (per window: centre ``x``/``y``, ``estimate``, formal ``sigma``, ``cells_block``/``cells_reference``,
    the fraction of cells ``kept`` by the reweighting).

    Measured (2026-10-21, ``docs/ionosphere_paper/``, section 6):

    - On fake boundaries through one block of eight NISAR scenes, where the answer is 0, the windows read |e| 0.035 at
      the median and 0.26 at the 95th percentile (0.15 without D054's ionospheric waves). The median over a boundary's
      windows read no wrong integer on 565 boundaries.
    - Most windows past 0.5 sit on real main-band errors: patches 1-3 cycles off, which the side band shows.
    - Across a rupture whose main band runs around its end (Venezuela A162), it reads +0.02.
    - On three Panama pairs it finds the range-offset ties one cycle short.
    - It is biased by up to a few tenths when the ionosphere carries 10-20 km waves along the boundary, or when a third
      of the cells beside the gap carry local cycle errors. A kriging estimate is a check.
    """
    f_a, f_b = float(screen.attrs["f_a"]), float(screen.attrs["f_b"])
    cell = float(screen.attrs["cell"])
    step = 2.0 * np.pi * f_b / (f_a + f_b)
    d = np.asarray(screen["dispersive"].values, float)
    use = np.asarray(screen["mask"].values, bool) & np.isfinite(d)
    c_a = f_b * f_b / (f_b * f_b - f_a * f_a)
    c_b = f_a * f_b / (f_b * f_b - f_a * f_a)
    sig = np.hypot(c_a * phase_sigma(screen["coherence_a"].values, cell * cell / (6.2 * 5.3)),
                   c_b * phase_sigma(screen["coherence_b"].values, cell * cell / (50.0 * 5.3)))
    w0 = 1.0 / np.maximum(sig, 1e-3) ** 2
    block, reference = np.asarray(block, bool), np.asarray(reference, bool)
    to_block = ndimage.distance_transform_edt(~block, sampling=cell)
    to_ref = ndimage.distance_transform_edt(~reference, sampling=cell)
    in_b = use & block & (to_ref <= reach_m) & (to_ref >= min_gap_m)
    in_r = use & reference & (to_block <= reach_m) & (to_block >= min_gap_m)
    sel = in_b | in_r
    X, Y = np.meshgrid(screen.x.values.astype(float), screen.y.values.astype(float))
    if along is None:
        key = (np.floor((X - X[sel].min()) / window_m) * 1e6 + np.floor((Y - Y[sel].min()) / window_m)) if sel.any() \
            else np.zeros(X.shape)
    else:
        a = np.asarray(along, float)
        sel &= np.isfinite(a)
        key = np.floor((np.where(np.isfinite(a), a, 0.0) - (np.nanmin(a[sel]) if sel.any() else 0.0)) / window_m)
    windows = []
    for k in np.unique(key[sel]):
        m = sel & (key == k)
        nb, nr = int((m & in_b).sum()), int((m & in_r).sum())
        if nb < min_cells or nr < min_cells:
            continue
        xc, yc = X[m].mean(), Y[m].mean()
        A = np.column_stack([_poly_columns((X[m] - xc) / 1e4, (Y[m] - yc) / 1e4, degree), in_b[m].astype(float)])
        coef, w, cov = _robust_lstsq(A, d[m], w0[m])
        windows.append(dict(x=float(xc), y=float(yc), estimate=float(-coef[-1] / step),
                            sigma=float(np.sqrt(cov[-1, -1]) / step), cells_block=nb, cells_reference=nr,
                            kept=float(np.mean(w > 0))))
    if not windows:
        return dict(cycles=None, estimate=np.nan, spread=np.nan, ambiguous=True, windows=[])
    e = np.array([w["estimate"] for w in windows])
    med = float(np.median(e))
    cyc = int(np.round(med))
    spread = float(1.4826 * np.median(np.abs(e - med)))
    return dict(cycles=cyc, estimate=med, spread=spread,
                ambiguous=bool(abs(med - cyc) > 0.25 or spread > 0.25 or e.size < 2), windows=windows)



# ---------------------------------------------------------------------------
# 6. The main band's integers patch by patch, from the side band
# ---------------------------------------------------------------------------
def cycle_patches(base, alternative, blocks, min_block_pixels=1000):
    """Patches where ``alternative`` -- another unwrapping of the same interferogram, on the same grid -- differs from
    ``base`` by whole cycles, beyond each block's own modal difference.

    ``blocks`` is a list of boolean masks (the regions of the base, e.g. the north block and the tied south block); in
    each, the alternative's most common integer difference is its block offset (another reference or tie) and is taken
    out, so only local disagreements are left. Returns ``(labels, delta)``: ``labels`` > 0 numbers the patches
    (connected pixels of one difference), ``delta[p - 1]`` the alternative's cycles minus the base's in patch ``p``.
    """
    base = np.asarray(base, float)
    k = np.round((np.asarray(alternative, float) - base) / (2.0 * np.pi))
    D = np.zeros(base.shape, np.int32)
    for blk in blocks:
        m = np.asarray(blk, bool) & np.isfinite(k)
        if m.sum() < min_block_pixels:
            continue
        kk = k[m].astype(np.int64)
        mode = int(np.bincount(kk - kk.min()).argmax() + kk.min())
        D[m] = kk - mode
    labels = np.zeros(base.shape, np.int64)
    delta = []
    for v in np.unique(D[D != 0]):
        lab, n = ndimage.label(D == v)
        labels[lab > 0] = lab[lab > 0] + len(delta)
        delta += [int(v)] * n
    return labels, np.asarray(delta, np.int64)


def patch_cycles(screen, patches, x, y, valid, *, reference=None, min_pixels=40, min_fraction=0.1, k=4.685):
    """Whole cycles to add to the main band in each of ``patches``, read from the side band.

    A patch whose main-band integer is off by ``n`` moves the raw dispersive estimate of every cell it covers by
    ``-2 pi f_B / (f_A + f_B) n`` (3.21 rad per cycle at NISAR's A/B) times the share of the cell's main-band pixels it
    holds. The ionosphere is smooth across it, so the raw estimate's departure from a smooth ``reference`` (default the
    screen itself) over the patch's cells, regressed on those shares, reads ``n``: Cramer-Rao weights, Tukey's biweight.

    ``screen`` is :func:`split_spectrum_screen`'s output, made without a fault-zone exclusion so that its raw estimate
    covers the patches near a trace (its ``dispersive``, ``mask`` and coherences are used); ``reference`` a smooth screen
    on the same cells, e.g. the production one, made with the exclusion. ``patches`` are integer labels (> 0) on the main
    band's own grid ``y`` x ``x`` (pixel centres), e.g. from :func:`cycle_patches`; ``valid`` that grid's pixels the
    cells' main-band phase was averaged over. Patches under ``min_pixels`` are skipped, and cells holding less than
    ``min_fraction`` of a patch do not count. Returns one dict per patch read: ``patch``, ``pixels``, ``cells``,
    ``cycles`` (the estimate, to add to the base) and ``sigma`` (its standard error), NaN where no cell is usable.
    """
    from scipy import sparse

    f_a, f_b = float(screen.attrs["f_a"]), float(screen.attrs["f_b"])
    step = 2.0 * np.pi * f_b / (f_a + f_b)
    cell = float(screen.attrs["cell"])
    xc, yc = screen.x.values, screen.y.values
    raw = np.asarray(screen["dispersive"].values, float)
    ref = screen["screen"] if reference is None else reference
    ref = np.asarray(getattr(ref, "values", ref), float)
    use = np.asarray(screen["mask"].values, bool) & np.isfinite(raw) & np.isfinite(ref)
    e = raw - ref
    if use.any():
        e = e - np.median(e[use])
    if "coherence_a" in screen and "coherence_b" in screen:     # each cell's Cramer-Rao sigma, full cells (as block_cycles)
        c_a = f_b * f_b / (f_b * f_b - f_a * f_a)
        c_b = f_a * f_b / (f_b * f_b - f_a * f_a)
        sig = np.hypot(c_a * phase_sigma(screen["coherence_a"].values, cell * cell / (6.2 * 5.3)),
                       c_b * phase_sigma(screen["coherence_b"].values, cell * cell / (50.0 * 5.3)))
    else:
        sig = np.asarray(screen["sigma"].values, float)
    ix = np.floor((np.asarray(x, float) - (xc[0] - cell / 2)) / cell).astype(np.int64)
    iy = np.floor(((yc[0] + cell / 2) - np.asarray(y, float)) / cell).astype(np.int64)
    IX, IY = np.meshgrid(ix, iy)
    inside = (IX >= 0) & (IX < xc.size) & (IY >= 0) & (IY < yc.size)
    cidx = np.where(inside, IY * xc.size + IX, 0)
    valid = np.asarray(valid, bool) & inside
    nvalid = np.bincount(cidx[valid], minlength=xc.size * yc.size)
    P = np.asarray(patches, np.int64)
    sel = (P > 0) & valid
    npatch = int(P.max()) if P.size else 0
    sizes = np.bincount(P[sel], minlength=npatch + 1)
    A = sparse.coo_matrix((np.ones(int(sel.sum())), (P[sel], cidx[sel])),
                          shape=(npatch + 1, xc.size * yc.size)).tocsr()
    ef, uf, sf = e.ravel(), use.ravel(), sig.ravel()
    out = []
    for p in np.flatnonzero(sizes >= min_pixels):
        if p == 0:
            continue
        row = A.getrow(p)
        c = row.indices
        f = row.data / np.maximum(nvalid[c], 1)
        ok = uf[c] & (f >= min_fraction)
        rec = dict(patch=int(p), pixels=int(sizes[p]), cells=int(ok.sum()), cycles=np.nan, sigma=np.nan)
        if ok.sum() >= 3:
            f, ee = f[ok], ef[c][ok]
            w0 = 1.0 / np.maximum(sf[c][ok], 0.3) ** 2
            w = w0
            n = -np.sum(w * f * ee) / (step * np.sum(w * f * f))
            for _ in range(20):
                r = ee + step * n * f
                s = 1.4826 * np.median(np.abs(r)) + 1e-12
                u = r / (k * s)
                w = w0 * np.where(np.abs(u) < 1.0, (1.0 - u * u) ** 2, 0.0)
                if not w.any():
                    break
                new = -np.sum(w * f * ee) / (step * np.sum(w * f * f))
                done = abs(new - n) < 1e-4
                n = new
                if done:
                    break
            r = ee + step * n * f
            s = 1.4826 * np.median(np.abs(r - np.median(r)))
            rec.update(cycles=float(n), sigma=float(s / (step * np.sqrt(np.sum(f * f)))))
        out.append(rec)
    return out


def patch_verdict(cycles, sigma, delta, cells, *, min_cells=5, max_sigma=0.2, tol=0.4, z=2.5):
    """Which of two unwrappings the side band backs in a patch: ``"alternative"`` if the reading lies within ``tol`` of the
    alternative's ``delta`` and ``z`` standard errors past the midpoint toward it, ``"base"`` the same toward 0, else
    ``"undecided"`` (too few cells, too uncertain, or a reading that backs neither)."""
    if not (np.isfinite(cycles) and np.isfinite(sigma)) or cells < min_cells or sigma > max_sigma or delta == 0:
        return "undecided"
    toward = (cycles - 0.5 * delta) * np.sign(delta)
    if abs(cycles - delta) < tol and toward > z * sigma:
        return "alternative"
    if abs(cycles) < tol and -toward > z * sigma:
        return "base"
    return "undecided"
