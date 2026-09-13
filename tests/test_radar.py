"""Geometry primitives behind the RSLC -> GSLC geocoder (``nisar_tools.radar``).

The interpolators are checked against analytic truth or against a literal
transcription of ISCE3's formulas; the fast paths against the slow ones they
replace, bit for bit, since the geocoder's fidelity to ASF's products rests on
reproducing ISCE3's floating-point operations rather than just its maths.
"""

import numpy as np
import pytest

import radar_synthetic as rs
from nisar_tools import radar


def _bits_equal(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return (a.dtype == b.dtype and a.shape == b.shape
            and np.array_equal(a.reshape(-1).view(f"u{a.dtype.itemsize}"),
                               b.reshape(-1).view(f"u{b.dtype.itemsize}")))


def _isce3_hermite(orbit, t):
    """``interpolateOrbitHermite`` transcribed: Lagrange products and all."""
    idx = int(np.clip(orbit._search(np.array([t]))[0] - 2, 0, orbit.size - 4))
    T = orbit.first + orbit.spacing * np.arange(idx, idx + 4)
    P, V = orbit.position[idx:idx + 4], orbit.velocity[idx:idx + 4]
    pos, vel = np.zeros(3), np.zeros(3)
    for i in range(4):
        h, hdot, sum_inv = 1.0, 0.0, 0.0
        for j in range(4):
            if j == i:
                continue
            h *= (t - T[j]) / (T[i] - T[j])
            sum_inv += 1.0 / (T[i] - T[j])
            prod = 1.0 / (T[i] - T[j])
            for k in range(4):
                if k not in (i, j):
                    prod *= (t - T[k]) / (T[i] - T[k])
            hdot += prod
        f0 = 1.0 - 2.0 * sum_inv * (t - T[i])
        g0 = 2.0 * (f0 * hdot - sum_inv * h)
        g1 = h + 2.0 * hdot * (t - T[i])
        pos += h * h * (f0 * P[i] + (t - T[i]) * V[i])
        vel += h * (g0 * P[i] + g1 * V[i])
    return pos, vel


# -- Orbit ---------------------------------------------------------------------
def test_hermite_matches_the_isce3_product_form():
    orbit = rs.orbit()
    ts = np.concatenate([np.linspace(orbit.start_time, orbit.end_time, 157),
                         orbit.time[5:9] + 1e-9])
    pos, vel = orbit.interpolate(ts)
    for k, t in enumerate(ts):
        p_ref, v_ref = _isce3_hermite(orbit, t)
        assert np.abs(pos[k] - p_ref).max() < 2e-7
        assert np.abs(vel[k] - v_ref).max() < 2e-8


def test_hermite_reproduces_the_state_vectors_and_the_true_orbit():
    orbit = rs.orbit()
    pos, vel = orbit.interpolate(orbit.time)
    assert np.abs(pos - orbit.position).max() < 1e-6
    assert np.abs(vel - orbit.velocity).max() < 1e-7
    ts = np.linspace(orbit.start_time, orbit.end_time, 1001)
    pos, vel = orbit.interpolate(ts)
    p_true, v_true = rs.truth(ts)
    assert np.abs(pos - p_true).max() < 1e-4          # metres, 10 s lattice
    assert np.abs(vel - v_true).max() < 1e-5


def test_orbit_is_nan_outside_its_span_and_shape_is_kept():
    orbit = rs.orbit()
    t = np.array([[orbit.start_time - 1.0, orbit.mid_time], [orbit.end_time, np.nan]])
    pos, vel = orbit.interpolate(t)
    assert pos.shape == (2, 2, 3) and vel.shape == (2, 2, 3)
    assert np.isnan(pos[0, 0]).all() and np.isnan(pos[1, 1]).all()
    assert np.isfinite(pos[0, 1]).all() and np.isfinite(pos[1, 0]).all()


def test_hermite_result_does_not_depend_on_which_points_share_the_call():
    orbit = rs.orbit()
    rng = np.random.default_rng(0)
    one_window = orbit.mid_time + rng.uniform(0, 0.5, 1000)
    spread = rng.uniform(orbit.start_time, orbit.end_time, 1000)
    both = np.concatenate([one_window, spread])
    p_all, v_all = orbit.interpolate(both)
    p1, v1 = orbit.interpolate(one_window)
    p2, v2 = orbit.interpolate(spread)
    assert np.allclose(p_all, np.concatenate([p1, p2]), rtol=0, atol=1e-8)
    assert np.allclose(v_all, np.concatenate([v1, v2]), rtol=0, atol=1e-9)


def test_with_reference_epoch_is_the_same_orbit():
    orbit = rs.orbit()
    shifted = orbit.with_reference_epoch(rs.EPOCH + np.timedelta64(3600, "s"))
    t = orbit.mid_time + 0.123
    p0, _ = orbit.interpolate(t)
    p1, _ = shifted.interpolate(t - 3600.0)
    assert np.abs(p1 - p0).max() < 1e-6


# -- geo2rdr / rdr2geo ------------------------------------------------------------
@pytest.fixture(scope="module")
def geometry():
    orbit = rs.orbit()
    grid = rs.radar_grid(length=2000, width=4000)
    rng = np.random.default_rng(1)
    t = rng.uniform(grid.sensing_start, grid.sensing_stop, 300)
    r = rng.uniform(grid.starting_range, grid.end_range, 300)
    h = rng.uniform(-100, 4000, 300)
    llh, ok = radar.rdr2geo(t, r, orbit, height=h, wavelength=grid.wavelength,
                            look_side=grid.look_side, threshold=1e-8)
    assert ok.all()
    return orbit, grid, t, r, h, llh


def test_rdr2geo_then_geo2rdr_round_trips(geometry):
    orbit, grid, t, r, h, llh = geometry
    assert np.abs(llh[:, 2] - h).max() < 1e-3
    t2, r2, ok = radar.geo2rdr(llh, orbit, None, grid.wavelength, grid.look_side,
                               t0=grid.sensing_mid)
    assert ok.all()
    assert np.abs(t2 - t).max() < 1e-7
    assert np.abs(r2 - r).max() < 1e-4


def test_geo2rdr_solution_is_zero_doppler(geometry):
    orbit, grid, _, _, _, llh = geometry
    t, r, ok = radar.geo2rdr(llh, orbit, None, grid.wavelength, grid.look_side,
                             t0=grid.sensing_mid)
    xyz = radar.lonlat_to_xyz(llh[:, 0], llh[:, 1], llh[:, 2])
    pos, vel = orbit.interpolate(t)
    cos_squint = np.einsum("ij,ij->i", xyz - pos, vel) / (
        np.linalg.norm(xyz - pos, axis=1) * np.linalg.norm(vel, axis=1))
    assert ok.all() and np.abs(cos_squint).max() < 1e-9
    assert np.allclose(np.linalg.norm(xyz - pos, axis=1), r, rtol=0, atol=1e-6)


def test_geo2rdr_iterates_do_not_depend_on_the_batch(geometry):
    """Vectorised geo2rdr must give each point exactly its own ISCE3 iterates."""
    orbit, grid, _, _, _, llh = geometry
    kw = dict(wavelength=grid.wavelength, look_side=grid.look_side, t0=grid.sensing_mid)
    t_all, r_all, ok_all = radar.geo2rdr(llh, orbit, None, **kw)
    for k in (0, 17, 299):
        t1, r1, ok1 = radar.geo2rdr(llh[k:k + 1], orbit, None, **kw)
        assert ok1[0] == ok_all[k]
        assert abs(t1[0] - t_all[k]) < 1e-12 and abs(r1[0] - r_all[k]) < 1e-9


def test_geo2rdr_rejects_the_wrong_look_side_and_far_points(geometry):
    orbit, grid, _, _, _, llh = geometry
    _, _, ok = radar.geo2rdr(llh, orbit, None, grid.wavelength, "right", t0=grid.sensing_mid)
    assert not ok.any()
    # A target seeded far outside the orbit's span gets the coarse search, and
    # one the orbit never passes still comes back unconverged, not as garbage.
    far = llh.copy()[:5]
    far[:, 1] = np.deg2rad(-60.0)
    _, _, ok = radar.geo2rdr(far, orbit, None, grid.wavelength, grid.look_side,
                             t0=grid.sensing_mid, maxiter=10)
    assert not ok.any()


def test_geo2rdr_with_a_doppler_lut_finds_the_squinted_time(geometry):
    orbit, grid, _, _, _, llh = geometry
    fd = 950.0
    lut = rs.constant_doppler(grid, fd)
    t, r, ok = radar.geo2rdr(llh[:50], orbit, lut, grid.wavelength, grid.look_side,
                             t0=grid.sensing_mid)
    xyz = radar.lonlat_to_xyz(llh[:50, 0], llh[:50, 1], llh[:50, 2])
    pos, vel = orbit.interpolate(t)
    los = xyz - pos
    doppler = 2.0 / grid.wavelength * np.einsum("ij,ij->i", los, vel) / np.linalg.norm(los, axis=1)
    assert ok.all()
    assert np.abs(doppler - fd).max() < 1e-3


# -- 2-D interpolators ------------------------------------------------------------
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("order", [4, 6, 8])
def test_spline_table_path_is_bit_identical_to_the_pointwise_path(dtype, order):
    rng = np.random.default_rng(order)
    z = (rng.standard_normal((37, 41)) * 300 + 500).astype(dtype)
    x = rng.uniform(0, 44, 5000)
    y = rng.uniform(0, 40, 5000)
    x[:40] = np.floor(x[:40])
    y[40:80] = 0.0
    ref = radar._spline2d_pointwise(z, x, y, order, np.dtype(dtype))
    assert _bits_equal(radar.spline2d_interp(z, x, y, order), ref)
    assert _bits_equal(radar.spline2d_interp(z, x.reshape(50, 100), y.reshape(50, 100), order).ravel(), ref)


def test_spline_is_accurate_on_a_smooth_surface_and_exact_at_nodes():
    yy, xx = np.mgrid[0:30, 0:30].astype(float)
    surface = lambda x, y: 100.0 + 3.0 * x - 2.0 * y + 0.05 * x * y   # noqa: E731
    z = surface(xx, yy)
    rng = np.random.default_rng(5)
    x = rng.uniform(3, 26, 400)
    y = rng.uniform(3, 26, 400)
    # A natural cubic spline through samples of a (bi)linear function is that
    # function, so this is exact -- and pins the index convention: ISCE3's
    # z(i + 1, j + 1) window and its 1-based spline coordinate cancel.
    out = radar.spline2d_interp(z, x, y, 6)
    assert np.abs(out - surface(x, y)).max() < 1e-9
    nodes = radar.spline2d_interp(z, np.array([5.0, 10.0]), np.array([7.0, 12.0]), 6)
    assert np.allclose(nodes, [z[7, 5], z[12, 10]], atol=1e-9)


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.complex64])
def test_bilinear_is_linear_inside_cells_and_on_cell_edges(dtype):
    z = (np.arange(20 * 30).reshape(20, 30) * 0.5).astype(dtype)
    if np.iscomplexobj(z):
        z = z + 1j * z[::-1]
    x = np.array([3.25, 7.0, 7.5, 7.0, 29.0])
    y = np.array([4.5, 2.25, 11.0, 11.0, 19.0])
    x1, y1 = np.floor(x).astype(int), np.floor(y).astype(int)
    x2, y2 = np.minimum(x1 + 1, 29), np.minimum(y1 + 1, 19)
    fx, fy = x - x1, y - y1
    expected = ((1 - fx) * (1 - fy) * z[y1, x1] + fx * (1 - fy) * z[y1, x2]
                + (1 - fx) * fy * z[y2, x1] + fx * fy * z[y2, x2])
    out = radar.bilinear_interp(z, x, y)
    assert out.dtype == np.result_type(z.dtype, np.float32)
    assert np.allclose(out, expected, rtol=1e-6)
    assert np.ndim(radar.bilinear_interp(z, 3.0, 4.0)) == 0


def test_nearest_rounds_half_away_from_zero():
    z = np.arange(25.0).reshape(5, 5)
    assert radar.nearest_interp(z, np.array([1.5, 2.49]), np.array([0.5, 3.5])).tolist() == [7.0, 22.0]


# -- LUT2d ------------------------------------------------------------------------
def test_empty_lut_is_its_reference_value_everywhere():
    lut = radar.LUT2d()
    assert not lut.have_data
    assert lut.contains(np.array([1e9]), np.array([-5.0])).all()
    assert np.all(lut.eval(np.zeros(3), np.zeros(3)) == 0.0)


def test_lut_clamps_outside_and_reports_containment():
    t = np.array([10.0, 20.0, 30.0])
    r = np.array([100.0, 200.0])
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    lut = radar.LUT2d.from_coords(r, t, data)
    assert lut.eval(15.0, 150.0) == pytest.approx(2.5)
    assert lut.eval(-100.0, 1e6) == pytest.approx(2.0)          # clamped to (t0, r_end)
    assert lut.contains(np.array([10.0, 30.0, 31.0]), np.array([100.0, 200.0, 150.0])).tolist() == [True, True, False]


# -- RadarGrid ----------------------------------------------------------------------
def test_radar_grid_contains_half_a_pixel_beyond_its_edges():
    g = rs.radar_grid(length=100, width=50)
    dt, dr = 1.0 / g.prf, g.range_pixel_spacing
    assert g.contains(g.sensing_start - 0.49 * dt, g.starting_range)
    assert not g.contains(g.sensing_start - 0.51 * dt, g.starting_range)
    assert g.contains(g.sensing_stop, g.end_range + 0.49 * dr)
    assert not g.contains(g.sensing_stop, g.end_range + 0.51 * dr)
    coarse = g.resize_keep_start_stop(11, 6)
    assert coarse.sensing_stop == pytest.approx(g.sensing_stop)
    assert coarse.end_range == pytest.approx(g.end_range)


def test_geo_bounding_box_encloses_the_swath_at_both_heights():
    orbit, grid = rs.orbit(), rs.radar_grid(length=3000, width=2000)
    x0, x1, y0, y1 = radar.geo_bounding_box(grid, orbit, 4326)
    corners, ok = radar.rdr2geo(
        np.array([grid.sensing_start, grid.sensing_stop] * 2),
        np.repeat([grid.starting_range, grid.end_range], 2), orbit, height=4000.0,
        wavelength=grid.wavelength, look_side=grid.look_side)
    lon, lat = np.degrees(corners[:, 0]), np.degrees(corners[:, 1])
    assert ok.all()
    assert (x0 <= lon).all() and (lon <= x1).all() and (y0 <= lat).all() and (lat <= y1).all()
