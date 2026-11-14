"""Fixes of ISCE3 v0.25.16 defects the port reproduces by default.

Each is off unless asked for (`geocode_rslc(isce3_compatible=...)`), and the
ISCE3 behaviour stays pinned by the other tests; these pin the fixes.
"""

import h5py
import numpy as np
import pytest

import radar_synthetic as rs
from nisar_tools import radar
from nisar_tools._geocode_kernels import radar_indices
from nisar_tools.dem import interpolate_dem_window
from nisar_tools.radar import _spline2d_even_table, _spline2d_pointwise
from nisar_tools.rslc import RSLC


def _surface():
    Y, X = np.mgrid[0:40, 0:50].astype(float)
    f = lambda x, y: np.sin(0.31 * x) * np.cos(0.23 * y) + 0.002 * x * y  # noqa: E731
    return f, f(X, Y)


def test_centred_spline_reads_row_zero_and_fits_better():
    f, z = _surface()
    at_nodes = radar.spline2d_interp(z, np.array([0.0, 3.0]), np.array([0.0, 2.0]), centered=True)
    np.testing.assert_allclose(at_nodes, [z[0, 0], z[2, 3]], atol=1e-12)
    off = radar.spline2d_interp(z, np.array([0.0]), np.array([0.0]))       # ISCE3 never reads row/column 0
    assert abs(off[0] - z[0, 0]) > 0.1
    rng = np.random.default_rng(0)
    x, y = rng.uniform(4, 44, 4000), rng.uniform(4, 34, 4000)
    err = lambda c: np.sqrt(np.mean((radar.spline2d_interp(z, x, y, centered=c) - f(x, y)) ** 2))  # noqa: E731
    assert err(True) < 0.5 * err(False)


@pytest.mark.parametrize("centered", [False, True])
def test_spline_table_path_matches_pointwise(centered):
    _, z = _surface()
    rng = np.random.default_rng(1)
    x, y = rng.uniform(0, 49, 3000), rng.uniform(0, 39, 3000)
    a = _spline2d_pointwise(z, x, y, 6, np.dtype(np.float64), centered)
    b = _spline2d_even_table(z, x, y, 6, np.dtype(np.float64), centered)
    assert np.array_equal(a, b)


def test_centred_dem_window_needs_three_nodes_after_the_point():
    arr = np.arange(100, dtype=np.float32).reshape(10, 10)
    x = np.array([7.5, 6.5])                 # floor columns 7 and 6 in a 10-wide window
    isce3 = interpolate_dem_window(arr, 0.0, 0.0, 1.0, 1.0, x, np.array([4.5, 4.5]), ref_height=-1.0)
    centred = interpolate_dem_window(arr, 0.0, 0.0, 1.0, 1.0, x, np.array([4.5, 4.5]), ref_height=-1.0,
                                     centered=True)
    assert np.all(isce3 != -1.0)             # ISCE3: one node after is enough
    assert centred[0] == -1.0 and centred[1] != -1.0


def _flat_dem_llh(grid, orbit, lines, samples):
    t = grid.sensing_start + np.asarray(lines, float) / grid.prf
    r = grid.starting_range + np.asarray(samples, float) * grid.range_pixel_spacing
    llh, ok = radar.rdr2geo(t, r, orbit, height=0.0, wavelength=grid.wavelength,
                            look_side=grid.look_side, threshold=1e-9)
    assert ok.all()
    return llh[:, 0], llh[:, 1], np.zeros(len(t))


def test_chip_rows_catch_a_line_narrowed_inside_the_chip():
    orbit = rs.orbit()
    grid = rs.radar_grid(length=64, width=64)
    doppler = rs.constant_doppler(grid, 950.0)
    valid = [np.column_stack([np.zeros(64), np.full(64, 64)]).astype(np.int64)]
    valid[0][32] = (40, 64)                  # line 32 lost its first 40 samples
    lon, lat, h = _flat_dem_llh(grid, orbit, [30.5], [30.5])
    *_, corners = radar_indices(lon, lat, h, grid, orbit, doppler, valid)
    *_, rows = radar_indices(lon, lat, h, grid, orbit, doppler, valid, chip_rows=True)
    assert corners[0] == 1 and rows[0] == 0


def test_clamped_corrections_apply_outside_their_table():
    orbit = rs.orbit()
    grid = rs.radar_grid(length=64, width=64)
    doppler = rs.constant_doppler(grid, 950.0)
    lon, lat, h = _flat_dem_llh(grid, orbit, [32.0], [32.0])
    r_mid = grid.starting_range + 32.0 * grid.range_pixel_spacing
    # a range-correction table that stops just short of the pixel
    table = radar.LUT2d.from_coords(np.array([grid.starting_range, r_mid - 10.0]),
                                    np.array([grid.sensing_start, grid.sensing_stop]),
                                    np.full((2, 2), 2.0 * grid.range_pixel_spacing))
    _, g0, *_ = radar_indices(lon, lat, h, grid, orbit, doppler, [], rg_correction=table)
    _, g1, *_ = radar_indices(lon, lat, h, grid, orbit, doppler, [], rg_correction=table, clamp_corrections=True)
    assert abs(g0[0] - 32.0) < 1e-6 and abs(g1[0] - 34.0) < 1e-6


def test_doppler_table_re_indexed_to_zero_doppler(tmp_path):
    orbit = rs.orbit()
    grid = rs.radar_grid(length=300, width=200)
    t_axis = np.linspace(grid.sensing_start - 8, grid.sensing_stop + 8, 9)
    r_axis = np.linspace(grid.starting_range - 2000, grid.end_range + 2000, 5)
    data = 900.0 + 5.0 * (t_axis - t_axis[0])[:, None] + 0.0 * r_axis[None, :]   # 5 Hz per second along track
    path = tmp_path / "rslc.h5"
    epoch = radar.epoch_units(rs.EPOCH)
    with h5py.File(path, "w") as f:
        ident = f.create_group("science/LSAR/identification")
        ident["lookDirection"], ident["orbitPassDirection"] = np.bytes_("Left"), np.bytes_("Descending")
        ident["listOfFrequencies"] = np.array([b"A"])
        root = f.create_group("science/LSAR/RSLC")
        sw = root.create_group("swaths")
        sw["zeroDopplerTime"] = grid.sensing_start + np.arange(grid.length) / grid.prf
        sw["zeroDopplerTime"].attrs["units"] = np.bytes_(epoch)
        sw["zeroDopplerTimeSpacing"] = 1.0 / grid.prf
        fa = sw.create_group("frequencyA")
        fa["slantRange"] = grid.starting_range + np.arange(grid.width) * grid.range_pixel_spacing
        fa["processedCenterFrequency"] = radar.SPEED_OF_LIGHT / grid.wavelength
        o = root.create_group("metadata/orbit")
        o["time"] = orbit.time
        o["time"].attrs["units"] = np.bytes_(epoch)
        o["position"], o["velocity"] = orbit.position, orbit.velocity
        par = root.create_group("metadata/processingInformation/parameters")
        par["zeroDopplerTime"], par["slantRange"] = t_axis, r_axis
        par["frequencyA/dopplerCentroid"] = data
    with RSLC(path) as r:
        native = r.doppler_centroid("A")
        zero = r.doppler_centroid("A", zero_doppler=True)
        # one interior node, done by hand: its target, then where the beam centre crosses it
        t0, r0 = t_axis[4], r_axis[2]
        llh, _ = radar.rdr2geo([t0], [r0], r.orbit, height=0.0, wavelength=grid.wavelength,
                               look_side=grid.look_side, threshold=1e-6)
        tn, rn, _ = radar.geo2rdr(llh, r.orbit, native, grid.wavelength, grid.look_side, t0=t0)
        assert 1.0 < t0 - tn[0] < 3.5                     # the beam centre leads by ~2 s
        assert zero.eval(np.array([t0]), np.array([r0]))[0] == pytest.approx(native.eval(tn, rn)[0], abs=1e-6)
        assert zero.eval(np.array([t0]), np.array([r0]))[0] < native.eval(np.array([t0]), np.array([r0]))[0]


def test_cube_rdr2geo_tolerance_keeps_ground_velocity():
    orbit = rs.orbit()
    grid = rs.radar_grid(length=200, width=200)
    doppler = rs.constant_doppler(grid, 950.0)
    lon, lat, _ = _flat_dem_llh(grid, orbit, [50.0, 100.0], [60.0, 120.0])
    isce3 = radar.radar_grid_cube_layer(lon, lat, 0.0, grid, orbit, doppler)
    fixed = radar.radar_grid_cube_layer(lon, lat, 0.0, grid, orbit, doppler, rdr2geo_threshold=1e-5)
    v0, v1 = isce3["groundTrackVelocity"], fixed["groundTrackVelocity"]
    assert np.all(np.isfinite(v1)) and np.allclose(v0, v1, rtol=1e-3)
