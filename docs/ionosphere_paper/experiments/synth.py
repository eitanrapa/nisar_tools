"""Synthetic NISAR main- and side-band cells over a surface rupture.

The truth is a vertical strike-slip rupture seen in LOS by a left-looking descending pass (D141's geometry), plus an
ionosphere. Each 40 m pixel's interferogram is drawn from the complex Wishart distribution of its looks, with a
coherence that combines the far field, a damage zone along the trace and the fringe-rate loss. The pixels are then
summed over 240 m cells exactly as `nisar_tools.split_spectrum.gslc_cells` sums GSLC pixels.

- Rupture: Okada (1985) rectangles (`slipsolve._okada`), 10 segments along the trace by 2 in depth (0-10 and 10-15
  km). Slip is tapered along strike (sqrt(sin)) and is 0.6 in the lower layer. Left-lateral; ``slip`` is the peak. The
  trace runs N105E through the origin: ``length`` 50 km ends inside the 96 km scene ("finite"), and 160 km runs
  through it ("through"), so the two sides connect only across the fault zone, as at D141 and in Venezuela.
- LOS: target to sensor (-0.586, 0.135, 0.799) at 37 deg incidence, heading 193 deg. Phase at f_A is
  ``-4 pi f_A / c x LOS`` (the package's convention); band B carries ``f_B / f_A`` of it.
- Ionosphere (phase at f_A, rad): a linear gradient, a Gaussian-correlated random field and optional travelling waves.
  Band B carries ``f_A / f_B`` of it.
- Coherence per band: ``gamma0`` x the damage zone (``q + (1 - q)(1 - exp(-(d/w)^2))`` on each side) x
  ``|sinc|`` of the band's phase gradient over its own resolution cell (A 6.2 m, B 50 m of ground range, 5.3 m along
  track) and over the 40 m pixel.
- Looks per 40 m pixel: A 48, B 6 (40 + 5 MHz), i.e. 1728 / 216 per cell (`split_spectrum`'s own estimate: 1753 / 217).
- ``offset_b_m``: band B sampled that far down range, the GUNW's A-to-B decimation offset.
- ``coast``: water north of the trace on its eastern half, beyond ``coast_km`` from it.
- ``patches``: ``(fraction, size m, within m)``: that fraction of the ground within ``within`` of the trace in blobs of
  coherence 0.05 (a smooth random field thresholded; size its correlation length), as real near fields are patchy.
- ``nearfield``: rough near-field deformation (`nearfield_field`), added to the Okada field in both bands. The Okada
  field alone changes by ~4 rad/km at the trace at 4 m of slip; D141's main band north of its trace changes by a
  median 14-24 rad/km within 2 km (`NEARFIELD_D141` matches that), 3-6 rad across a cell.
- ``pixel_models``: cells with the main band taken out pixel by pixel, as `gslc_cells(phase_model=)` makes them, on
  the same draws: ``"oracle"`` the true A phase at each pixel, ``"multilook"`` the A pixels' own phase after a
  Gaussian of one pixel, on the true integers (the caller's unwrapped main band, without unwrapping errors).
"""

import os

import numpy as np
import xarray as xr
from scipy import ndimage

C_LIGHT = 299792458.0
F_A, F_B = 1239.0e6, 1293.5e6
PX, CELL = 40.0, 240.0
HALF = 48000.0
STRIKE_DEG = 105.0
LENGTH, DEPTH, TOP_LAYER = 50000.0, 15000.0, 10000.0
NSEG = 10
LOS = np.array([-0.5864, 0.1354, 0.7986])
RANGE_DIR = np.array([0.9744, -0.2250])       # ground range (sensor -> target), horizontal
AZIMUTH_DIR = np.array([-0.2250, -0.9744])    # along track (heading 193 deg)
RES_A, RES_B, RES_AZ = 6.2, 50.0, 5.3
LOOKS_A, LOOKS_B = 48, 6
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "synth_unit_slip.npz")
LENGTH_THROUGH = 160000.0


def grid():
    n = int(round(2 * HALF / PX))
    x = -HALF + (np.arange(n) + 0.5) * PX
    y = HALF - (np.arange(n) + 0.5) * PX          # north up, as the GSLC cells
    return x, y


def fault_frame(length=LENGTH):
    s = np.array([np.sin(np.deg2rad(STRIKE_DEG)), np.cos(np.deg2rad(STRIKE_DEG))])
    nrm = np.array([-s[1], s[0]])                 # left of the strike direction: the "north" side
    start = -0.5 * length * s
    return s, nrm, start


def trace_xy(length=LENGTH):
    s, _, start = fault_frame(length)
    end = start + length * s
    return np.array([start[0], end[0]]), np.array([start[1], end[1]])


def signed_distance(X, Y, length=LENGTH):
    """Distance (m) to the finite trace segment, + on the north (left) side."""
    s, nrm, start = fault_frame(length)
    px, py = X - start[0], Y - start[1]
    t = np.clip(px * s[0] + py * s[1], 0.0, length)
    qx, qy = px - t * s[0], py - t * s[1]
    side = np.sign(px * nrm[0] + py * nrm[1])
    return np.hypot(qx, qy) * np.where(side == 0, 1.0, side)


def unit_slip_los(length=LENGTH):
    """LOS displacement (m, + toward the sensor) per metre of peak slip, on the 40 m grid (cached per length)."""
    from slipsolve._okada import okada_disp_surface

    cache = CACHE.replace(".npz", f"_{int(length / 1e3)}km.npz")
    if os.path.exists(cache):
        with np.load(cache) as z:
            if z["los"].shape == (grid()[1].size, grid()[0].size) and float(z["strike"]) == STRIKE_DEG:
                return z["los"]
    x, y = grid()
    X, Y = np.meshgrid(x, y)
    s, nrm, start = fault_frame(length)
    xp = (X - start[0]) * s[0] + (Y - start[1]) * s[1]
    yp = (X - start[0]) * nrm[0] + (Y - start[1]) * nrm[1]
    seg = length / NSEG
    taper = np.sqrt(np.sin(np.pi * (np.arange(NSEG) + 0.5) / NSEG))
    ux = np.zeros(X.size); uy = np.zeros(X.size); uz = np.zeros(X.size)
    for i in range(NSEG):
        for depth, width, w_d in ((TOP_LAYER, TOP_LAYER, 1.0), (DEPTH, DEPTH - TOP_LAYER, 0.6)):
            e, n_, v = okada_disp_surface(xp.ravel() - i * seg, yp.ravel(), seg, width, depth, 90.0,
                                          taper[i] * w_d, 0.0, 0.0)
            ux += e; uy += n_; uz += v
    east = ux * s[0] + uy * nrm[0]
    north = ux * s[1] + uy * nrm[1]
    los = (east * LOS[0] + north * LOS[1] + uz * LOS[2]).reshape(X.shape)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez(cache, los=los.astype(np.float32), strike=STRIKE_DEG, length=length)
    return los


def ionosphere(X, Y, rng, gradient=0.6, gradient_az=30.0, rf_rms=4.0, rf_km=15.0, tid_amp=0.0, tid_km=25.0,
               tid_az=70.0):
    """Ionospheric phase at f_A (rad): a gradient (rad/km, toward azimuth deg), a Gaussian-correlated random field
    (rms rad, correlation length km), optional plane waves (amplitude rad, wavelength km, azimuth deg)."""
    a = np.deg2rad(gradient_az)
    io = gradient * (X * np.sin(a) + Y * np.cos(a)) / 1e3
    if rf_rms > 0:
        w = rng.standard_normal(X.shape)
        f = np.real(np.fft.ifft2(ndimage.fourier_gaussian(np.fft.fft2(w), sigma=rf_km * 1e3 / PX)))
        io = io + rf_rms * (f - f.mean()) / f.std()
    if tid_amp > 0:
        b = np.deg2rad(tid_az)
        io = io + tid_amp * np.sin(2 * np.pi * (X * np.sin(b) + Y * np.cos(b)) / (tid_km * 1e3) + rng.uniform(0, 6.28))
    return io


def _damage(d, gamma0, north=(0.12, 2000.0), south=(0.25, 800.0)):
    out = np.ones(d.shape)
    for side, (g, w) in ((d >= 0, north), (d < 0, south)):
        if g is None:
            continue
        q = min(g / gamma0, 1.0)
        out = np.where(side, q + (1 - q) * (1 - np.exp(-(np.abs(d) / w) ** 2)), out)
    return out


def _fringe_loss(phase, res_r):
    gy, gx = np.gradient(phase, -PX, PX)               # rows run south (y decreasing)
    gr = gx * RANGE_DIR[0] + gy * RANGE_DIR[1]
    ga = gx * AZIMUTH_DIR[0] + gy * AZIMUTH_DIR[1]
    res = np.abs(np.sinc(gr * res_r / (2 * np.pi)) * np.sinc(ga * RES_AZ / (2 * np.pi)))
    pix = np.abs(np.sinc(gx * PX / (2 * np.pi)) * np.sinc(gy * PX / (2 * np.pi)))
    return res * pix


def _wishart_cross(phase, gamma, looks, rng):
    """``(W12, W11, W22)`` of an L-look complex Wishart pair with correlation ``gamma exp(i phase)`` (Bartlett)."""
    g1 = rng.standard_gamma(looks, phase.shape)
    g2 = rng.standard_gamma(looks - 1, phase.shape) if looks > 1 else np.zeros(phase.shape)
    a21 = (rng.standard_normal(phase.shape) + 1j * rng.standard_normal(phase.shape)) / np.sqrt(2.0)
    c = np.sqrt(np.clip(1.0 - gamma * gamma, 0.0, 1.0))
    e = np.exp(1j * phase)
    sg1 = np.sqrt(g1)
    w12 = g1 * gamma * e + sg1 * np.conj(a21) * c
    w22 = gamma * gamma * g1 + 2 * np.real(gamma * np.conj(e) * sg1 * np.conj(a21) * c) + c * c * (np.abs(a21) ** 2 + g2)
    return w12, g1, w22


def _cells(a, k):
    ny, nx = a.shape
    return a.reshape(ny // k, k, nx // k, k).sum(axis=(1, 3))


#: rough near-field deformation as large as D141's (median |grad| 14-24 rad/km within 2 km north of its trace)
NEARFIELD_D141 = dict(amp=8.0, corr_m=300.0, decay_m=5000.0)


def nearfield_field(X, Y, d, seed, amp=6.0, corr_m=300.0, decay_m=4000.0):
    """Rough near-field deformation (rad at f_A; non-dispersive): a Gaussian-correlated random field of unit rms
    (Gaussian of ``corr_m``) times ``amp`` exp(-|d| / ``decay_m``). Drawn from its own stream (``seed``, 7), so adding
    it moves no other draw."""
    rng = np.random.default_rng([int(seed), 7])
    w = rng.standard_normal(X.shape)
    f = np.real(np.fft.ifft2(ndimage.fourier_gaussian(np.fft.fft2(w), sigma=corr_m / PX)))
    return amp * (f - f.mean()) / f.std() * np.exp(-np.abs(d) / decay_m)


def scene(slip=4.0, seed=0, gamma0=0.6, damage=dict(north=(0.12, 2000.0), south=(0.25, 800.0)), iono=None,
          coast=False, coast_km=2.0, offset_b_m=0.0, looks=(LOOKS_A, LOOKS_B), length=LENGTH, patches=None,
          nearfield=None, pixel_models=()):
    """One synthetic pair. Returns ``(cells_a, cells_b, truth)``: the two bands' cell datasets (as `gslc_cells`) and
    the truth on the cells: ``psi`` (deformation phase at f_A), ``iono`` (dispersive phase at f_A), ``phase_a`` (the
    oracle main-band unwrapping, the cell mean of the true A phase over land), ``dist`` (signed distance to the trace,
    m), ``land``, and the cell centres; with ``pixel_models``, ``pixel_cells``: ``{name: (cells_a, cells_b)}`` made with
    the main band taken out pixel by pixel."""
    rng = np.random.default_rng(seed)
    x, y = grid()
    X, Y = np.meshgrid(x, y)
    los = slip * unit_slip_los(length).astype(float)
    psi = -4 * np.pi * F_A / C_LIGHT * los
    io = ionosphere(X, Y, rng, **(iono or {}))
    d = signed_distance(X, Y, length)
    if nearfield:
        psi = psi + nearfield_field(X, Y, d, seed, **nearfield)
    land = np.ones(X.shape, bool)
    if coast:
        s, _, _ = fault_frame(length)
        along = X * s[0] + Y * s[1]
        land = ~((d > coast_km * 1e3) & (along > 0))
    phase_a = psi + io
    if offset_b_m:
        gy, gx = np.gradient(psi, -PX, PX)
        psi_b = psi + offset_b_m * (gx * RANGE_DIR[0] + gy * RANGE_DIR[1])
    else:
        psi_b = psi
    phase_b = (F_B / F_A) * psi_b + (F_A / F_B) * io
    dz = _damage(d, gamma0, **(damage or dict(north=(None, 1.0), south=(None, 1.0))))
    if patches:                       # (fraction, size m, within m): decorrelated blobs (coherence 0.05) near the trace
        frac, size_m, within_m = patches
        w = rng.standard_normal(X.shape)
        f = np.real(np.fft.ifft2(ndimage.fourier_gaussian(np.fft.fft2(w), sigma=size_m / PX)))
        near_t = np.abs(d) < within_m
        blob = near_t & (f < np.quantile(f[near_t], frac))
        dz = np.where(blob, min(0.05 / gamma0, 1.0), dz)
    out = []
    k = int(round(CELL / PX))
    xc = _cells(X, k)[0] / (k * k)
    yc = _cells(Y, k)[:, 0] / (k * k)
    draws = []
    for phase, res, L, f_c in ((phase_a, RES_A, looks[0], F_A), (phase_b, RES_B, looks[1], F_B)):
        g = gamma0 * dz * _fringe_loss(phase, res)
        w12, w11, w22 = _wishart_cross(phase, g, L, rng)
        draws.append(tuple(np.where(land, v, 0) for v in (w12, w11, w22)) + (f_c,))
    n = _cells(land.astype(np.int64), k)

    def cellset(model=None, name=None):
        sets = []
        for w12, w11, w22, f_c in draws:
            if model is not None:
                w12 = w12 * np.exp(-1j * (f_c / F_A) * model)
            ifg, p1, p2 = _cells(w12, k), _cells(w11, k), _cells(w22, k)
            with np.errstate(invalid="ignore", divide="ignore"):
                coh = np.abs(ifg) / np.sqrt(p1 * p2)
            v = {"ifg": (("y", "x"), ifg), "p1": (("y", "x"), p1), "p2": (("y", "x"), p2),
                 "n": (("y", "x"), n), "coherence": (("y", "x"), coh)}
            at = {"center_frequency": f_c, "cell": CELL, "pixels_per_cell": k * k}
            if model is not None:
                with np.errstate(invalid="ignore", divide="ignore"):
                    v["model"] = (("y", "x"), np.where(n > 0, _cells(np.where(land, model, 0.0), k) / np.maximum(n, 1),
                                                       np.nan))
                at.update(phase_model=name, model_frequency=F_A, model_scale=f_c / F_A)
            sets.append(xr.Dataset(v, coords={"y": yc, "x": xc}, attrs=at))
        return sets

    cells_a, cells_b = cellset()
    pixel_cells = {}
    for name in pixel_models:
        if name == "oracle":
            model = phase_a
        elif name == "multilook":
            sm = ndimage.gaussian_filter(draws[0][0].real, 1.0) + 1j * ndimage.gaussian_filter(draws[0][0].imag, 1.0)
            model = phase_a + np.angle(sm * np.exp(-1j * phase_a))
        else:
            raise ValueError(f"unknown pixel model {name!r}")
        pixel_cells[name] = tuple(cellset(np.where(land, model, 0.0), name))
    area = _cells(land.astype(float), k)
    with np.errstate(invalid="ignore", divide="ignore"):
        ref = np.where(area > 0, _cells(np.where(land, phase_a, 0.0), k) / area, np.nan)
    Xc, Yc = np.meshgrid(cells_a.x.values, cells_a.y.values)
    truth = dict(psi=_cells(psi, k) / (k * k), iono=_cells(io, k) / (k * k), phase_a=ref,
                 dist=signed_distance(Xc, Yc, length), land=area / (k * k) > 0.5, x=cells_a.x.values,
                 y=cells_a.y.values, length=length)
    if pixel_models:
        truth["pixel_cells"] = pixel_cells
    return cells_a, cells_b, truth
