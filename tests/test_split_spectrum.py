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


def _write_gslc(path, x, y, data, freq, mask=None, look=None):
    with h5py.File(path, "w") as f:
        if look is not None:                             # (lookDirection, orbitPassDirection)
            f["science/LSAR/identification/lookDirection"] = look[0]
            f["science/LSAR/identification/orbitPassDirection"] = look[1]
        g = f.create_group(f"science/LSAR/GSLC/grids/frequency{freq}")
        g["xCoordinates"] = x
        g["yCoordinates"] = y
        g["centerFrequency"] = F_A if freq == "A" else F_B
        g.create_dataset("HH", data=data.astype(np.complex64), chunks=(16, 16), compression="gzip")
        if mask is not None:
            g["mask"] = mask.astype(np.uint8)


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


def _steep_pair(tmp_path, gradient=10.0, seed=3):
    """Band A (5 m) and band B (40 x 5 m) GSLC pairs over 10 x 10 cells of 240 m, perfectly coherent, with independent
    speckle amplitudes in the two bands: the deformation ``gradient`` (rad/km at f_A, along x, plus curvature along y)
    and a smooth ionosphere. Returns the paths and the analytic deformation and ionosphere as functions of (x, y)."""
    rng = np.random.default_rng(seed)
    y = 3000.0 - 2.5 - 5.0 * np.arange(552)
    psi = lambda X, Y: gradient * 1e-3 * X + 3e-6 * (Y - 1800.0) ** 2            # noqa: E731
    iono = lambda X, Y: 2.0 * np.sin(X / 5000.0) + 0.3 * Y / 1000.0               # noqa: E731
    paths = {}
    for band, x in (("A", 1000.0 + 2.5 + 5.0 * np.arange(552)), ("B", 1000.0 + 20.0 + 40.0 * np.arange(69))):
        X, Y = np.meshgrid(x, y)
        f = F_A if band == "A" else F_B
        phase = f / F_A * psi(X, Y) + F_A / f * iono(X, Y)
        amp = rng.rayleigh(1.0, X.shape)
        g1 = amp * np.exp(1j * rng.uniform(-np.pi, np.pi, X.shape))
        for date, g in (("1", g1), ("2", g1 * np.exp(-1j * phase))):
            paths[band + date] = tmp_path / f"{band}{date}.h5"
            _write_gslc(paths[band + date], x, y, g, band)
    return paths, psi, iono


def _model(fn_a, step=10.0):
    x = np.arange(990.0, 3800.0, step)
    y = np.arange(3010.0, 200.0, -step)
    X, Y = np.meshgrid(x, y)
    return xr.DataArray(fn_a(X, Y), coords={"y": y, "x": x}, dims=("y", "x"), attrs={"center_frequency": F_A})


def test_gslc_cells_with_the_main_band_taken_out_pixel_by_pixel(tmp_path):
    """Steep fringes inside a cell: the bands' cell phases are different weighted means of them (independent speckle,
    the side band's pixels 8x wider), so r = phi_B - (f_B/f_A) phi_A per cell carries a deformation-made error that
    the 11.6x of d = r / k amplifies. Taken out of each pixel first, r is k d to the interpolation's precision."""
    paths, psi, iono = _steep_pair(tmp_path)
    a = ss.gslc_cells(paths["A1"], paths["A2"], "A")
    b = ss.gslc_cells(paths["B1"], paths["B2"], "B")
    model = _model(lambda X, Y: psi(X, Y) + iono(X, Y))
    am = ss.gslc_cells(paths["A1"], paths["A2"], "A", phase_model=model)
    bm = ss.gslc_cells(paths["B1"], paths["B2"], "B", phase_model=model)
    assert am.attrs["model_scale"] == 1.0 and bm.attrs["model_scale"] == pytest.approx(F_B / F_A)
    assert am.attrs["phase_model"] == bm.attrs["phase_model"] and "sha256" in am.attrs["phase_model"]
    X, Y = np.meshgrid(a.x.values, a.y.values)
    k = ss.dispersive_factor(F_A, F_B)
    truth = ss.wrap(k * iono(X, Y))
    _, r_cell = ss.dispersive_residual(a, b, psi(X, Y) + iono(X, Y))
    phi_a, r_pix = ss.dispersive_residual(am, bm)
    err_cell = ss.wrap(r_cell - truth)
    err_pix = ss.wrap(r_pix - truth)
    assert np.sqrt(np.mean(err_cell ** 2)) > 0.02          # 0.03 rad of r: 0.35 rad of d
    assert np.max(np.abs(err_pix)) < 2e-3                    # the ionosphere's own curvature in the cell, and float32
    # the coherence the fringes cost the cells comes back
    assert np.median(a.coherence.values) < 0.85 and np.min(am.coherence.values) > 0.999
    # phi_A is the model's cell mean plus the cell's residual: the A phase on the model's integers
    np.testing.assert_allclose(phi_a, psi(X, Y) + iono(X, Y), atol=0.1)
    # a given phase_a only moves integers: one cycle on a block is (f_B/f_A) 2 pi of r, as without the model
    moved = np.where(X > 2000.0, phi_a + 2 * np.pi, phi_a)
    _, r_moved = ss.dispersive_residual(am, bm, moved)
    np.testing.assert_allclose(ss.wrap(r_moved - r_pix), np.where(X > 2000.0, ss.wrap(-F_B / F_A * 2 * np.pi), 0.0),
                               atol=1e-9)
    with pytest.raises(ValueError):
        ss.dispersive_residual(am, b)                        # one band with the model, the other without
    with pytest.raises(ValueError):
        ss.dispersive_residual(am, bm.assign_attrs(phase_model="another"))


def test_gslc_cells_counts_only_pixels_with_a_model(tmp_path):
    paths, psi, iono = _steep_pair(tmp_path, gradient=2.0)
    model = _model(lambda X, Y: psi(X, Y) + iono(X, Y))
    model = model.where(model.x < 2000.0)                    # no model east of x = 2000 (a cut, water, another region)
    am = ss.gslc_cells(paths["A1"], paths["A2"], "A", phase_model=model)
    plain = ss.gslc_cells(paths["A1"], paths["A2"], "A")
    full = am.x.values + 120.0 < 1990.0                     # cells wholly west of the model's edge (and its neighbour)
    none = am.x.values - 120.0 >= 2000.0
    assert np.all(am.n.values[:, full] == plain.n.values[:, full])
    assert np.all(am.n.values[:, none] == 0) and np.all(np.isnan(am.model.values[:, none]))
    merged = ss.merge_cells(am, am)                          # a cell seen by two frames: sums, and the model's mean
    np.testing.assert_allclose(merged.model.values, am.model.values, equal_nan=True)
    assert np.all(merged.n.values == 2 * am.n.values)
    with pytest.raises(ValueError):
        ss.merge_cells(am, plain)
    with pytest.raises(ValueError):
        ss.gslc_cells(paths["A1"], paths["A2"], "A", phase_model=model.assign_attrs(center_frequency=None))


def test_gslc_cells_can_drop_partially_focused_pixels(tmp_path):
    """The products' mask (0 = partially focused) keeps a near-range strip out of the cell sums (A004, 2026-10-14)."""
    rng = np.random.default_rng(0)
    x = 1000.0 + 2.5 + 5.0 * np.arange(96)
    y = 3000.0 - 2.5 - 5.0 * np.arange(96)
    g1 = rng.standard_normal((96, 96)) + 1j * rng.standard_normal((96, 96))
    ph = np.full((96, 96), 0.7)
    ph[:, 70:] = 2.0                                     # a partially focused strip with its own phase error
    m = np.ones((96, 96), np.uint8)
    m[:, 70:] = 0
    m[:, :10] = 255                                      # outside the acquisition
    _write_gslc(tmp_path / "a.h5", x, y, g1, "A", mask=m)
    _write_gslc(tmp_path / "b.h5", x, y, g1 * np.exp(-1j * ph), "A", mask=m)
    plain = ss.gslc_cells(tmp_path / "a.h5", tmp_path / "b.h5", "A", cell=240.0)
    masked = ss.gslc_cells(tmp_path / "a.h5", tmp_path / "b.h5", "A", cell=240.0, subswath_mask=True)
    assert abs(np.angle(plain.ifg.values[0, 0]) - 0.7) > 0.2      # the strip pulls the plain sum away
    assert np.angle(masked.ifg.values[0, 0]) == pytest.approx(0.7, abs=1e-5)
    assert int(masked.n.values[0, 0]) == 48 * (70 - 40)            # the cell's columns 40-87, valid to 69
    assert masked.attrs["subswath_mask"] is True


def test_gslc_cells_near_range_drops_only_the_near_range_strip(tmp_path):
    """``subswath_mask="near_range"`` drops partially focused pixels only near the swath's near-range edge and keeps a
    partially focused datatake start (A004, 2026-10-14: its rupture lies in one)."""
    rng = np.random.default_rng(0)
    nx = 240
    x = 1000.0 + 2.5 + 5.0 * np.arange(nx)             # whole cells: columns 40-87, 88-135, 136-183, 184-231
    y = 3000.0 - 2.5 - 5.0 * np.arange(96)
    g1 = rng.standard_normal((96, nx)) + 1j * rng.standard_normal((96, nx))
    ph = np.full((96, nx), 0.7)
    m = np.ones((96, nx), np.uint8)
    m[:, 232:] = 255                                    # beyond the near-range edge (left-looking ascending: east)
    m[:, 200:232] = 0
    ph[:, 200:232] = 2.0                                # the near-range strip, 160 m, with its own phase error
    m[:, 40:61] = 0
    ph[:, 40:61] = 1.3                                  # a partially focused datatake start, far from near range
    m[:, 150:184] = 0
    ph[:, 150:184] = 1.9                                # and another 240-410 m from the edge, beside clipped bounds
    for name, g in (("a.h5", g1), ("b.h5", g1 * np.exp(-1j * ph))):
        _write_gslc(tmp_path / name, x, y, g, "A", mask=m, look=("Left", "Ascending"))
    a, b = tmp_path / "a.h5", tmp_path / "b.h5"
    assert ss.near_range_side_of(a) == "east"
    full = ss.gslc_cells(a, b, "A", cell=240.0, subswath_mask=True)
    near = ss.gslc_cells(a, b, "A", cell=240.0, subswath_mask="near_range", near_range_m=200.0)
    assert near.attrs["subswath_mask"] == "near_range" and near.attrs["near_range_side"] == "east"
    assert int(full.n.values[0, 3]) == int(near.n.values[0, 3]) == 48 * 16          # the strip out of both
    assert np.angle(near.ifg.values[0, 3]) == pytest.approx(0.7, abs=1e-5)
    assert int(full.n.values[0, 0]) == 48 * 27                                     # True drops the start too
    assert int(near.n.values[0, 0]) == 48 * 48                                     # "near_range" keeps it
    west = ss.gslc_cells(a, b, "A", cell=240.0, subswath_mask="near_range", near_range_m=100.0, near_range_side="west")
    assert int(west.n.values[0, 3]) == 48 * 48 and int(west.n.values[0, 0]) == 48 * 48
    assert int(full.n.values[0, 2]) == 48 * 14 and int(near.n.values[0, 2]) == 48 * 48
    # bounds that stop short of the near-range edge (at column 183): the strip is measured from the swath's edge
    cut = ss.gslc_cells(a, b, "A", cell=240.0, bounds=(1200.0, 2500.0, 1920.0, 3000.0), subswath_mask="near_range",
                        near_range_m=200.0)
    assert cut.sizes["x"] == 3 and int(cut.n.values[0, 2]) == 48 * 48
    with pytest.raises(ValueError):
        ss.gslc_cells(a, b, "A", cell=240.0, subswath_mask="far_range")


def test_plane_fill_follows_a_gradient_to_a_one_sided_edge():
    """Data on one side only, then a gap (a coast along a fault plus an excluded zone): ISCE3's fill, a zero-order
    mean, lags the gradient there; the local-plane fill does not (D126, 2026-10-14: +20..26 mm LOS at the trace)."""
    ny, nx, cell = 120, 160, 240.0
    yy, xx = np.mgrid[0:ny, 0:nx]
    truth = 0.02 * yy                                    # rad per 240 m cell along y
    mask = yy >= 48                                      # the data; rows 40-47 a gap beside them, nothing above
    rng = np.random.default_rng(1)
    sig = np.full(truth.shape, 0.3)
    data = np.where(mask, truth + sig * rng.standard_normal(truth.shape), np.nan)
    gap = (yy >= 40) & (yy < 48)
    inner = (yy >= 70) & (yy < 100)
    s_guide, _ = ss.ionosphere_filter(data, sig, mask, cell)
    s_plane, _ = ss.ionosphere_filter(data, sig, mask, cell, fill="plane")
    assert np.median((s_guide - truth)[gap]) > 0.15      # the guide sits on the data's side of the gradient
    assert abs(np.median((s_plane - truth)[gap])) < 0.03
    assert abs(np.median((s_plane - truth)[inner])) < 0.01   # and nothing changes away from the edge
    assert abs(np.median((s_guide - truth)[inner])) < 0.01
    with pytest.raises(ValueError):
        ss.ionosphere_filter(data, sig, mask, cell, fill="nope")


def test_a_wider_plane_fill_crosses_a_wide_edge_gap():
    """An excluded near-range strip ~12 km wide (A004, 2026-10-14): the plane at the guide's sigma (2.6 km) reaches ~8 km
    and the guide's constant takes over beyond; ``fill_m`` widens the plane so it follows the gradient across."""
    ny, nx, cell = 100, 200, 240.0
    yy, xx = np.mgrid[0:ny, 0:nx]
    truth = 0.02 * xx                                    # rad per 240 m cell, across the strip
    mask = xx < 150                                      # columns 150-199 (12 km) excluded
    rng = np.random.default_rng(2)
    sig = np.full(truth.shape, 0.3)
    data = np.where(mask, truth + sig * rng.standard_normal(truth.shape), np.nan)
    far = xx >= 185                                      # 8.4-12 km into the strip
    inner = (xx > 20) & (xx < 120)
    s_plane, _ = ss.ionosphere_filter(data, sig, mask, cell, fill="plane")
    s_wide, _ = ss.ionosphere_filter(data, sig, mask, cell, fill="plane", fill_m=6000.0)
    assert np.median(np.abs((s_plane - truth)[far])) > 0.5      # 0.90: the guide's constant
    assert np.median(np.abs((s_wide - truth)[far])) < 0.15      # 0.095, most of it the final Gaussian at the array edge
    assert np.median(np.abs((s_wide - truth)[inner])) < 0.02     # nothing changes where there are data
    assert np.allclose(s_wide[inner], s_plane[inner], atol=1e-9)
    # a capped reach: the plane stops 2.4 km out and the far end falls back to the guide (A004's islands)
    s_cap, _ = ss.ionosphere_filter(data, sig, mask, cell, fill="plane", fill_m=6000.0, fill_reach_m=2400.0)
    assert np.median(np.abs((s_cap - truth)[far])) > 0.5
    assert np.allclose(s_cap[inner], s_plane[inner], atol=1e-9)


def test_a_first_order_filter_does_not_lag_where_the_weights_change():
    """ISCE3's final Gaussian is an inverse-variance weighted mean. Across a fault's damage zone the coherence, and with
    it the weight, falls toward the trace, so the mean's centroid moves away from the trace on both sides and the screen
    lags the ionosphere's gradient: oppositely on the two sides, a step. The weighted local plane does not."""
    ny, nx, cell = 140, 160, 240.0
    yy, xx = np.mgrid[0:ny, 0:nx]
    truth = 0.03 * yy + 0.01 * xx                                        # rad per cell: ~12 mm of LOS per km across
    d = np.abs(yy - 70.0) * cell                                         # distance to a trace along row 70
    sig = 0.3 / (0.15 + 0.85 * (1.0 - np.exp(-(d / 2000.0) ** 2)))       # 2 rad at the trace, 0.3 far away
    mask = np.ones(truth.shape, bool)
    s0, g0 = ss.ionosphere_filter(truth, sig, mask, cell)
    s1, g1 = ss.ionosphere_filter(truth, sig, mask, cell, order=1)
    inner = (xx > 30) & (xx < 130)
    north, south = inner & (yy < 70) & (yy > 55), inner & (yy > 70) & (yy < 85)        # 0-3.6 km either side
    assert np.median((s0 - truth)[north]) < -0.1 and np.median((s0 - truth)[south]) > 0.1   # 0.4 rad: ~8 mm each way
    assert np.max(np.abs((s1 - truth)[inner & (yy > 20) & (yy < 120)])) < 1e-8
    # uniform weights: the plane is the mean, away from the grid's edge; and so are their sigmas
    flat = np.full(truth.shape, 0.3)
    u0, h0 = ss.ionosphere_filter(truth, flat, mask, cell)
    u1, h1 = ss.ionosphere_filter(truth, flat, mask, cell, order=1)
    core = (xx > 20) & (xx < 140) & (yy > 20) & (yy < 120)
    np.testing.assert_allclose(u1[core], u0[core], atol=1e-9)
    np.testing.assert_allclose(h1[core], h0[core], rtol=1e-9)
    assert np.all(g1[north] > g0[north])                                # the plane's own variance where it leans
    with pytest.raises(ValueError):
        ss.ionosphere_filter(truth, sig, mask, cell, order=2)


def test_a_ladder_of_planes_fills_each_gap_with_the_plane_that_interpolates_it():
    """A curved field: a wide plane misses it across a narrow gap with data on both sides (a fault zone), a narrow one
    cannot reach deep into a one-sided strip (a swath edge). The ladder takes the narrow plane where its data lie
    around the cell and the wider ones where it would extrapolate."""
    ny, nx, cell = 160, 260, 240.0
    yy, xx = np.mgrid[0:ny, 0:nx]
    truth = 2e-4 * (xx - 90.0) ** 2 + 1.5e-4 * (yy - 70.0) ** 2 + 0.01 * xx      # rad; curvature ~4 rad over 15 km
    fault = np.abs(yy - 80) < 9                                          # a 4.3 km zone, data on both sides
    edge = xx >= 210                                                     # the last 12 km: nothing beyond
    mask = ~fault & ~edge
    sig = np.full(truth.shape, 0.3)
    data = np.where(mask, truth, np.nan)
    ladder = (1500.0, 3000.0, 6000.0, 12000.0)
    out = {f: ss.ionosphere_filter(data, sig, mask, cell, fill="plane", fill_m=f, fill_reach_m=12000.0)[0]
           for f in (1500.0, 10000.0, ladder)}
    err = {f: np.abs(o - truth) for f, o in out.items()}
    zone = fault & (xx > 20) & (xx < 190)
    deep = edge & (xx >= 245) & (yy > 20) & (yy < 140)                   # 8.4-12 km into the strip
    assert np.median(err[ladder][zone]) < 0.5 * np.median(err[10000.0][zone])
    assert np.median(err[ladder][zone]) < 1.2 * np.median(err[1500.0][zone])
    assert np.median(err[ladder][deep]) < 0.5 * np.median(err[1500.0][deep])
    # the blend's ends: cells whose narrow plane interpolates take it exactly, and where it would extrapolate far the
    # next rung decides
    W = np.where(mask, 1.0 / 0.09, 0.0)
    p0, lev0 = ss._plane_fill(np.where(mask, data, np.nan), W, 1500.0 / cell, lever=True)
    lad = ss._plane_ladder(np.where(mask, data, np.nan), W, [s / cell for s in ladder], (1.0, 2.0))
    close = np.isfinite(lev0) & (lev0 <= 1.0)
    np.testing.assert_allclose(lad[close], p0[close], rtol=0, atol=1e-9)
    assert (~mask & close).sum() > 1000                                  # the whole fault zone, and the strip's first cells
    p1 = ss._plane_ladder(np.where(mask, data, np.nan), W, [s / cell for s in ladder[1:]], (1.0, 2.0))
    off = ~np.isfinite(lev0) | (lev0 >= 2.0)
    np.testing.assert_allclose(lad[off & np.isfinite(p1)], p1[off & np.isfinite(p1)], rtol=0, atol=1e-9)


def test_a_singularity_in_a_gap_fill_does_not_cut_across_the_data():
    """D040 (Panama, 2026-10-17): the wide Gaussians filling Parita Bay left a phase singularity 28 km from any cell, and
    SNAPHU over the whole grid cut from it to the nearest grid edge -- across the 20 km neck joining Azuero to the
    mainland -- putting one cycle of r (73 rad of screen) between them. Planted here: land north and south of a bay,
    joined by a neck on the west, and a singularity in the bay nearer the west edge than any other."""
    import snaphu

    ny, nx = 200, 220
    yy, xx = np.mgrid[0:ny, 0:nx]
    north, south = (yy < 85) & (xx >= 20), (yy >= 115) & (xx >= 20)
    neck = (yy >= 85) & (yy < 115) & (xx >= 20) & (xx < 32)
    valid = north | south | neck
    theta = np.arctan2(yy - 100.0, xx - 52.0)            # its branch cut runs west, through the neck
    truth = 0.02 * yy + np.mod(theta, 2 * np.pi)          # continuous on the land: the cut is east, in the bay
    phase = ss.wrap(truth)
    _, reach = ss._lowpass_filled(phase, valid, 3.0)
    assert not reach[100, 52] and reach[100, 25]           # the singularity lies beyond the data's reach
    # the old way, SNAPHU over the whole grid, puts a cycle between north and south (the scene reproduces the failure)
    old = np.asarray(snaphu.unwrap(np.exp(1j * phase).astype(np.complex64), np.full(phase.shape, 0.95, np.float32),
                                   nlooks=50.0, cost="smooth", init="mcf")[0], float)
    jump = np.median((old - truth)[north]) - np.median((old - truth)[south])
    assert abs(abs(jump) - 2 * np.pi) < 1e-3
    with pytest.warns(RuntimeWarning, match="cut across the data"):
        new = ss._unwrap_guide(phase, reach, valid)
    n = (new - truth)[valid] / (2 * np.pi)
    assert np.ptp(np.round(n)) == 0                       # one integer over the whole land
    assert np.max(np.abs(n - np.round(n))) < 1e-4


def _two_blocks(seed=0, n=400, cell=240.0, noise=0.7):
    """A raw dispersive estimate on cells: a gradient plus a 15 km correlated field (rms 4 rad), white noise, and two
    blocks north and south of y = 0 with the 2 km either side of it left out (a fault zone)."""
    from scipy import ndimage

    rng = np.random.default_rng(seed)
    x = cell * (np.arange(n) + 0.5) - cell * n / 2
    y = cell * (np.arange(n)[::-1] + 0.5) - cell * n / 2
    X, Y = np.meshgrid(x, y)
    f = ndimage.gaussian_filter(rng.standard_normal(X.shape), 15e3 / cell)
    d = 0.6 * (0.5 * X + 0.87 * Y) / 1e3 + 4.0 * f / f.std() + noise * rng.standard_normal(X.shape)
    use = np.abs(Y) > 2000.0
    coh = np.full(X.shape, 0.6)
    ds = xr.Dataset({"dispersive": (("y", "x"), np.where(use, d, np.nan)), "mask": (("y", "x"), use),
                     "coherence_a": (("y", "x"), coh), "coherence_b": (("y", "x"), coh)},
                    coords={"y": y, "x": x}, attrs={"f_a": F_A, "f_b": F_B, "cell": cell})
    return ds, Y > 0, Y < 0, X, Y


def test_block_cycles_reads_how_far_a_block_is_off():
    c0, _ = ss.main_diff_coefficients(F_A, F_B)
    for seed, e in ((0, 1), (1, -2), (2, 0)):
        ds, north, south, X, Y = _two_blocks(seed)
        ds["dispersive"] = ds["dispersive"].where(~south, ds["dispersive"] - 2 * np.pi * c0 * e)
        out = ss.block_cycles(ds, south, north, along=X)
        assert out["cycles"] == e and not out["ambiguous"]
        assert abs(out["estimate"] - e) < 0.15 and len(out["windows"]) >= 4
        assert ss.block_cycles(ds, north, south, along=X)["cycles"] == -e          # the other way round


def test_block_cycles_ignores_patches_of_local_cycle_errors():
    """The near field of a rupture unwraps in patches (D141's north side): a third of the reference's cells 2-6 km from
    the gap one cycle up must not move the tie."""
    from scipy import ndimage

    c0, _ = ss.main_diff_coefficients(F_A, F_B)
    ds, north, south, X, Y = _two_blocks(3)
    near = north & (Y > 2000.0) & (Y < 6000.0)
    f = ndimage.gaussian_filter(np.random.default_rng(9).standard_normal(X.shape), 1000.0 / 240.0)
    patch = near & (f > np.quantile(f[near], 0.67))
    ds["dispersive"] = ds["dispersive"] + np.where(patch, 2 * np.pi * c0, 0.0)
    ds["dispersive"] = ds["dispersive"].where(~south, ds["dispersive"] - 2 * np.pi * c0)
    out = ss.block_cycles(ds, south, north, along=X)
    assert out["cycles"] == 1 and abs(out["estimate"] - 1) < 0.2


def test_block_cycles_through_the_screen_with_a_rupture_between():
    """The whole chain on `_scene`'s rupture (70 rad arctan plus a 10 rad step): the south block's main band one cycle
    short reads +1, the right one 0 -- the deformation does not enter the step."""
    x, y, X, Y, psi, d = _scene()
    phi_a = psi + d
    phi_b = F_B / F_A * psi + F_A / F_B * d
    ca, cb = _cells(phi_a, 0.8, F_A, x, y), _cells(phi_b, 0.6, F_B, x, y, kx=6)
    near = np.abs(Y) < 2000.0
    for e in (0, 1):
        wrong = np.where(Y < 0, phi_a - 2 * np.pi * e, phi_a)
        out = ss.split_spectrum_screen(ca, cb, np.where(near, np.nan, wrong), exclude=near)
        tie = ss.block_cycles(out, Y < 0, Y > 0, along=X, window_m=15000.0)
        assert tie["cycles"] == e and abs(tie["estimate"] - e) < 0.05


def test_the_side_band_picks_the_unwrapping_that_is_right_patch_by_patch():
    """Two unwrappings of one pair disagree by a cycle in two patches: in one the base is wrong (the alternative right),
    in the other the alternative is. Every cycle the main band is off moves the raw dispersive estimate by 3.21 rad
    times the share of a cell it covers, so the side band tells the two apart (D141, 2026-10-21: a 26 km2 patch 3 km
    north of the trace, a cycle off in the cut unwrapping and right in the uncut one, reads -0.8 +- 0.1)."""
    rng = np.random.default_rng(5)
    nc, k, cell = 100, 4, 240.0                         # 100 x 100 cells of 240 m, main band on 60 m pixels
    xp = 60.0 * (np.arange(nc * k) + 0.5)
    yp = 60.0 * (np.arange(nc * k)[::-1] + 0.5)
    xc, yc = xp.reshape(nc, k).mean(1), yp.reshape(nc, k).mean(1)
    XP, YP = np.meshgrid(xp, yp)
    truth = 0.002 * XP + 3.0 * np.sin(YP / 7000.0)       # the main band (rad); blocks: all of it
    wrong = (np.hypot(XP - 9000.0, YP - 15000.0) < 2300.0)                      # the base a cycle high: not on cells
    alt_wrong = (np.abs(XP - 18000.0) < 1900.0) & (np.abs(YP - 6000.0) < 1500.0)  # the alternative a cycle high
    base = truth + 2 * np.pi * wrong
    alt = truth + 2 * np.pi * alt_wrong
    labels, delta = ss.cycle_patches(base, alt, [np.ones(base.shape, bool)])
    assert sorted(delta.tolist()) == [-1, 1]
    # the raw estimate on the cells: a smooth ionosphere, noise, and 3.21 rad per cycle of the base, by share of cell
    step = 2 * np.pi * F_B / (F_A + F_B)
    share = wrong.reshape(nc, k, nc, k).mean(axis=(1, 3))
    Xc, Yc = np.meshgrid(xc, yc)
    iono = 5.0 * np.cos(Xc / 20000.0) + 0.0001 * Yc
    sig = np.full(iono.shape, 0.7)
    raw = iono + step * share + sig * rng.standard_normal(iono.shape)
    scr = xr.Dataset({"dispersive": (("y", "x"), raw), "mask": (("y", "x"), np.ones(raw.shape, bool)),
                      "sigma": (("y", "x"), sig), "screen": (("y", "x"), iono)},
                     coords={"y": yc, "x": xc}, attrs={"f_a": F_A, "f_b": F_B, "cell": cell})
    reads = {r["patch"]: r for r in ss.patch_cycles(scr, labels, xp, yp, np.ones(base.shape, bool))}
    for p, r in reads.items():
        verdict = ss.patch_verdict(r["cycles"], r["sigma"], int(delta[p - 1]), r["cells"])
        if delta[p - 1] == -1:                          # the base's error: add -1 there, the alternative is right
            assert r["cycles"] == pytest.approx(-1.0, abs=0.1) and verdict == "alternative"
        else:                                           # the alternative's error: the base is right
            assert r["cycles"] == pytest.approx(0.0, abs=0.1) and verdict == "base"
    assert ss.patch_verdict(np.nan, np.nan, 1, 0) == "undecided"
    assert ss.patch_verdict(0.5, 0.05, 1, 50) == "undecided"          # backs neither
    assert ss.patch_verdict(-1.0, 0.05, 1, 50) == "undecided"         # a cycle the other way: neither unwrapping
