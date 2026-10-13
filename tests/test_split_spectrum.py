"""``nisar_tools.split_spectrum``: the main/side-band ionosphere without the fault leakage.

The central property: with the A phase unwrapped correctly (the caller's job, fault-aware), the
only thing unwrapped is ``r = phi_B - (f_B/f_A) phi_A = k d``, which holds no deformation, so a
large near-fault deformation -- steps included -- does not reach the screen.
"""

import h5py
import numpy as np
import pytest
import xarray as xr

from nisar_tools import split_spectrum as ss

F_A, F_B = 1.239e9, 1.2935e9


def test_coefficients_match_isce3():
    c0, c1 = ss.main_diff_coefficients(F_A, F_B)
    assert c0 == pytest.approx(0.5108, abs=1e-4) and c1 == pytest.approx(11.61, abs=0.01)
    k = ss.dispersive_factor(F_A, F_B)
    psi, d = 3.7, -1.3
    phi_a = psi + d
    phi_b = F_B / F_A * psi + F_A / F_B * d
    assert phi_b - F_B / F_A * phi_a == pytest.approx(k * d)
    assert c0 * phi_a + c1 * (phi_a - phi_b) == pytest.approx(d)


def _cells(phase, coherence, f_c, x, y, kx=48, ky=48):
    n = np.full(phase.shape, kx * ky)
    return xr.Dataset({"ifg": (("y", "x"), coherence * np.exp(1j * phase) * n),
                       "p1": (("y", "x"), n * 1.0), "p2": (("y", "x"), n * 1.0),
                       "n": (("y", "x"), n), "coherence": (("y", "x"), np.full(phase.shape, coherence))},
                      coords={"y": y, "x": x},
                      attrs={"center_frequency": f_c, "cell": 240.0, "pixels_per_cell": kx * ky})


def _scene(n=180):
    x = 240.0 * (np.arange(n) + 0.5)
    y = 240.0 * (np.arange(n)[::-1] + 0.5) - 240.0 * n / 2
    X, Y = np.meshgrid(x, y)
    # a rupture along y = 0: an arctan (2 km locking depth, 70 rad) plus a sharp 10 rad step
    psi = 70.0 * (2 / np.pi) * np.arctan(Y / 2000.0) + 10.0 * np.sign(Y)
    d = 25.0 * np.sin(2 * np.pi * X / 60e3) + 0.4 * Y / 1e3        # smooth ionosphere, rad at f_A
    return x, y, X, Y, psi, d


def test_deformation_does_not_reach_the_screen():
    x, y, X, Y, psi, d = _scene()
    phi_a = psi + d
    phi_b = F_B / F_A * psi + F_A / F_B * d
    ca, cb = _cells(phi_a, 0.8, F_A, x, y), _cells(phi_b, 0.6, F_B, x, y, kx=6)
    near = np.abs(Y) < 2000.0
    out = ss.split_spectrum_screen(ca, cb, np.where(near, np.nan, phi_a), exclude=near)
    # against the true ionosphere through the same mask and filter (a 2 km Gaussian takes 2% off
    # the 60 km sine by itself), so what is left is leakage and unwrapping
    ref, _ = ss.ionosphere_filter(d, out.sigma.values * 0 + 1.0, out.mask.values, 240.0)
    err = out.screen.values - ref
    err -= np.nanmedian(err)                      # the screen is defined up to a constant
    inner = (np.abs(X - X.mean()) < 15e3)
    assert np.nanmax(np.abs(err[inner])) < 0.05       # 25 rad ionosphere, 80 rad deformation
    # no copy of the high-passed deformation
    hp = psi - ss.ionosphere_filter(psi, np.ones_like(psi), np.ones(psi.shape, bool), 240.0)[0]
    sel = inner & ~near
    slope = np.sum(hp[sel] * err[sel]) / np.sum(hp[sel] ** 2)
    assert abs(slope) < 0.01


def test_a_unwrapping_error_shows_as_a_step_not_a_copy():
    # a whole-cycle error in the caller's phi_A costs (f_B/f_A - 1) 2 pi / k = 3.2 rad of screen, locally
    x, y, X, Y, psi, d = _scene()
    phi_a = psi + d
    phi_b = F_B / F_A * psi + F_A / F_B * d
    ca, cb = _cells(phi_a, 0.8, F_A, x, y), _cells(phi_b, 0.6, F_B, x, y, kx=6)
    wrong = phi_a + 2 * np.pi * ((X > 30e3) & (Y > 5e3))
    out = ss.split_spectrum_screen(ca, cb, wrong)
    step = np.nanmedian(out.dispersive.values[(X > 32e3) & (Y > 7e3)] - d[(X > 32e3) & (Y > 7e3)]) \
        - np.nanmedian(out.dispersive.values[(X < 28e3) & (Y > 7e3)] - d[(X < 28e3) & (Y > 7e3)])
    c0, _ = ss.main_diff_coefficients(F_A, F_B)
    assert step == pytest.approx(c0 * 2 * np.pi, abs=0.05)


def test_filter_keeps_constants_and_fills_holes():
    a = np.full((60, 70), 3.0)
    m = np.ones(a.shape, bool)
    m[20:30, 25:40] = False
    scr, sig = ss.ionosphere_filter(a, np.full(a.shape, 0.5), m, 240.0)
    np.testing.assert_allclose(scr, 3.0, rtol=1e-12)
    np.testing.assert_allclose(sig, 0.5, rtol=1e-9)      # ISCE3's sigma_out: unchanged for constant sigma


def test_cells_of_both_bands_coincide():
    xa = 234002.5 + 5.0 * np.arange(4000)
    xb = 234020.0 + 40.0 * np.arange(500)
    y = 925197.5 - 5.0 * np.arange(3000)
    la, lb = ss.cell_lattice(xa, y, 240.0), ss.cell_lattice(xb, y, 240.0)
    n = min(la[4], lb[4])
    np.testing.assert_allclose(la[6][:n], lb[6][:n])
    np.testing.assert_allclose(la[7], lb[7])
    assert la[2] == 48 and lb[2] == 6 and la[3] == lb[3] == 48


def _write_gslc(path, x, y, data, freq):
    with h5py.File(path, "w") as f:
        g = f.create_group(f"science/LSAR/GSLC/grids/frequency{freq}")
        g["xCoordinates"] = x
        g["yCoordinates"] = y
        g["centerFrequency"] = F_A if freq == "A" else F_B
        g.create_dataset("HH", data=data.astype(np.complex64), chunks=(16, 16), compression="gzip")


def test_gslc_cells_sums_the_pairs(tmp_path):
    rng = np.random.default_rng(0)
    x = 1000.0 + 2.5 + 5.0 * np.arange(96)            # pixel edges on 5 m, cells of 240 m start at 1200
    y = 3000.0 - 2.5 - 5.0 * np.arange(96)
    g1 = rng.standard_normal((96, 96)) + 1j * rng.standard_normal((96, 96))
    g2 = g1 * np.exp(-1j * 0.7)                          # ifg phase +0.7 everywhere
    _write_gslc(tmp_path / "a.h5", x, y, g1, "A")
    _write_gslc(tmp_path / "b.h5", x, y, g2, "A")
    c = ss.gslc_cells(tmp_path / "a.h5", tmp_path / "b.h5", "A", cell=240.0)
    assert c.sizes["x"] == 1 and c.sizes["y"] == 1       # the only whole cell: x 1200-1440, y 2880-2640
    assert float(c.x[0]) == pytest.approx(1320.0) and float(c.y[0]) == pytest.approx(2760.0)
    assert np.angle(c.ifg.values[0, 0]) == pytest.approx(0.7, abs=1e-5)
    assert c.coherence.values[0, 0] == pytest.approx(1.0, abs=1e-6)
    assert int(c.n.values[0, 0]) == 48 * 48
