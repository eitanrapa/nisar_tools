"""Tests for radar-geometry helpers (cube reading, interpolation, phase->LOS).

The interpolation is checked against analytically-known synthetic cubes, and the
GSLC readers against a synthetic granule that carries a ``metadata/radarGrid``
cube (``gslc_factory(write_geometry=True)``).
"""

import numpy as np
import pyproj
import pytest
import xarray as xr

from nisar_tools import GSLC
from nisar_tools import geometry as G


def _cube(inc_func, nx=6, ny=5, heights=(-500.0, 0.0, 500.0, 1000.0),
          epsg=32611, x0=4e5, y0=4e6, step=1000.0):
    """Build a synthetic geometry cube. ``inc_func(height, x) -> incidence deg``;
    the LOS unit vector is (east=-sin(inc), north=0, up=cos(inc))."""
    hs = np.asarray(heights, float)
    x = x0 + step * np.arange(nx)
    y = y0 - step * np.arange(ny)  # descending, like a real product
    inc = np.empty((len(hs), ny, nx), np.float32)
    for i, h in enumerate(hs):
        for k in range(nx):
            inc[i, :, k] = inc_func(h, x[k])
    losx = (-np.sin(np.deg2rad(inc))).astype(np.float32)
    losy = np.zeros_like(losx)
    # Off-nadir look angle for a ~747 km orbit over a spherical Earth, so it is
    # always the smaller of the two, as in a real product.
    look = np.degrees(
        np.arcsin(6_371_000.0 / (6_371_000.0 + 747_000.0) * np.sin(np.deg2rad(inc)))
    ).astype(np.float32)
    ds = xr.Dataset(
        {
            "incidence_angle": (("height", "y", "x"), inc),
            "look_angle": (("height", "y", "x"), look),
            "los_east": (("height", "y", "x"), losx),
            "los_north": (("height", "y", "x"), losy),
        },
        coords={"height": hs, "y": y, "x": x},
    )
    ds.attrs["epsg"] = epsg
    return ds


# -- sample_look_geometry ----------------------------------------------------
@pytest.mark.parametrize("workers", [1, 2, 4, 10])
def test_banded_sampling_is_bit_identical(workers):
    """The sampling is pointwise in (x, y, height), so splitting it into row
    bands across threads cannot move a single value. Guards the banding that
    replaced the whole-grid version (202 -> 71 bytes/pixel of peak)."""
    cube = _cube(lambda h, x: 32.0 + 0.002 * (x - 4e5) + 0.005 * h, nx=8, ny=7)
    ox = np.linspace(4e5 + 300, 4e5 + 6500, 37)
    oy = np.linspace(4e6 - 5500, 4e6 - 300, 29)
    rng = np.random.default_rng(0)
    height = rng.uniform(-400.0, 900.0, (len(oy), len(ox)))

    ref = G.sample_look_geometry(cube, ox, oy, 32611, height=height, workers=1)
    got = G.sample_look_geometry(cube, ox, oy, 32611, height=height,
                                 workers=workers)
    assert set(got.data_vars) == set(ref.data_vars)
    for var in ref.data_vars:
        np.testing.assert_array_equal(
            got[var].values, ref[var].values,
            err_msg=f"{var} moved at workers={workers}",
        )


def test_banded_sampling_is_bit_identical_across_epsg():
    """The cross-EPSG path transforms each band's grid with pyproj, so it needs
    its own check -- a same-EPSG call skips the transformer entirely."""
    # Cube in zone 11, target grid declared in zone 12: the transformer runs.
    cube = _cube(lambda h, x: 32.0 + 0.002 * (x - 4e5), nx=40, ny=35, step=5000.0)
    # Derive the target grid by mapping the cube's own extent into zone 12, so it
    # is guaranteed to land inside the cube rather than on hand-picked numbers.
    to_12 = pyproj.Transformer.from_crs("EPSG:32611", "EPSG:32612", always_xy=True)
    cx, cy = cube["x"].values, cube["y"].values
    (x0, y0), (x1, y1) = (
        to_12.transform(cx[2], cy[2]), to_12.transform(cx[-3], cy[-3])
    )
    ox = np.linspace(min(x0, x1), max(x0, x1), 31)
    oy = np.linspace(max(y0, y1), min(y0, y1), 23)
    ref = G.sample_look_geometry(cube, ox, oy, 32612, workers=1)
    got = G.sample_look_geometry(cube, ox, oy, 32612, workers=8)
    # Not vacuous: the transformed grid has to land inside the cube.
    assert np.isfinite(ref["incidence_angle"].values).any()
    for var in ref.data_vars:
        np.testing.assert_array_equal(got[var].values, ref[var].values)


def test_interpolates_linear_in_x_exactly():
    # Incidence linear in x, height/y-independent -> linear interp is exact.
    x0, step = 4e5, 1000.0
    cube = _cube(lambda h, x: 30.0 + 0.002 * (x - x0), x0=x0, step=step)
    ox = np.linspace(x0 + 500, x0 + 4000, 15)
    oy = np.linspace(4e6 - 3500, 4e6 - 500, 11)
    geom = G.sample_look_geometry(cube, ox, oy, 32611, height=0.0)

    expected = 30.0 + 0.002 * (ox - x0)  # per-x, same for every y
    got = geom["incidence_angle"].values
    np.testing.assert_allclose(got, np.broadcast_to(expected, got.shape),
                               rtol=1e-4, atol=1e-3)


def test_interpolates_linear_in_height_exactly():
    # Incidence linear in height -> value at h=250 is the midpoint interpolation.
    cube = _cube(lambda h, x: 30.0 + 0.01 * h)
    ox = np.array([4e5 + 1500.0, 4e5 + 2500.0])
    oy = np.array([4e6 - 1000.0, 4e6 - 2000.0])
    geom = G.sample_look_geometry(cube, ox, oy, 32611, height=250.0)
    np.testing.assert_allclose(geom["incidence_angle"].values, 32.5,
                               rtol=1e-4, atol=1e-3)


def test_los_up_and_unit_norm_consistency():
    cube = _cube(lambda h, x: 35.0 + 1e-3 * (x - 4e5))
    ox = np.linspace(4e5 + 500, 4e5 + 4000, 12)
    oy = np.linspace(4e6 - 3500, 4e6 - 500, 9)
    geom = G.sample_look_geometry(cube, ox, oy, 32611, height=100.0)
    inc = np.deg2rad(geom["incidence_angle"].values)
    e, n, u = geom["los_east"].values, geom["los_north"].values, geom["los_up"].values
    # Unit norm holds tightly (u is built as sqrt(1-e^2-n^2)); the match to
    # cos/sin(inc) is limited by linearly interpolating the nonlinear LOS field.
    np.testing.assert_allclose(e**2 + n**2 + u**2, 1.0, atol=1e-5)  # unit vector
    np.testing.assert_allclose(n, 0.0, atol=1e-6)
    np.testing.assert_allclose(u, np.cos(inc), atol=2e-4)          # up = cos(inc)
    np.testing.assert_allclose(e, -np.sin(inc), atol=2e-4)         # east = -sin(inc)


def test_height_clipped_to_cube_range():
    # A height far above the cube's top layer must clamp, not extrapolate/NaN.
    cube = _cube(lambda h, x: 30.0 + 0.01 * h)  # tops out at h=1000 -> 40 deg
    geom = G.sample_look_geometry(cube, np.array([4e5 + 1500.0]),
                                  np.array([4e6 - 1500.0]), 32611, height=50000.0)
    np.testing.assert_allclose(geom["incidence_angle"].values, 40.0, atol=1e-3)


# -- dem_heights_on_grid -----------------------------------------------------
def test_dem_none_is_zero_height():
    h = G.dem_heights_on_grid(None, np.arange(5.0), np.arange(4.0), 32611)
    assert h.shape == (4, 5)
    assert np.all(h == 0.0)


def test_dem_sampled_onto_grid():
    epsg, x0, y0 = 32611, 4e5, 4e6
    xd = x0 + 100.0 * np.arange(20)
    yd = y0 - 100.0 * np.arange(20)
    dem = xr.DataArray(
        np.full((20, 20), 750.0, np.float32),
        coords={"y": yd, "x": xd}, dims=("y", "x"),
    ).rio.write_crs(f"EPSG:{epsg}")

    ox = x0 + 100.0 * np.arange(5) + 50.0
    oy = y0 - 100.0 * np.arange(4) - 50.0
    h = G.dem_heights_on_grid(dem, ox, oy, epsg)
    assert h.shape == (4, 5)
    np.testing.assert_allclose(h, 750.0, atol=1e-3)


# -- phase_to_los ------------------------------------------------------------
def test_phase_to_los_formula_and_sign():
    rng = np.random.default_rng(0)
    unw = rng.standard_normal((4, 5)).astype(np.float32) * 5
    lam = 0.24
    np.testing.assert_allclose(G.phase_to_los(unw, lam), -lam / (4 * np.pi) * unw)
    np.testing.assert_allclose(G.phase_to_los(unw, lam, sign=-1),
                               lam / (4 * np.pi) * unw)


def test_phase_to_los_inverts_the_range_change_chain():
    """The absolute sign, derived forward from a displacement rather than assumed.

    Every other test in the slip path builds its data with the same convention it
    then inverts, so all of them pass under a global sign flip. This one goes the
    other way round -- ground motion -> range change -> interferometric phase --
    and only then calls ``phase_to_los``, so it is the one place a mirrored LOS
    field cannot hide.

    The assumption being pinned is ISCE3's ``exp(-j 4 pi r / lambda)`` SLC phase,
    which with ``ref * conj(sec)`` gives
    ``phi = phi_ref - phi_sec = (4 pi / lambda) (r_sec - r_ref)``.
    """
    lam = 0.242
    look = np.array([0.62, -0.11, 0.78])       # target -> sensor, ascending-ish
    look /= np.linalg.norm(look)

    for u in (np.array([0.05, 0.0, 0.0]),      # 5 cm east
              np.array([0.0, 0.0, -0.03]),     # 3 cm subsidence
              np.array([-0.02, 0.04, 0.01])):
        toward = float(u @ look)               # positive = toward the sensor
        r_sec_minus_ref = -toward              # moving toward it shortens range
        phi = (4 * np.pi / lam) * r_sec_minus_ref

        assert G.phase_to_los(phi, lam) == pytest.approx(toward, abs=1e-12)

    # Subsidence with an upward-looking LOS must read as motion *away*.
    down = np.array([0.0, 0.0, -0.03])
    assert G.phase_to_los((4 * np.pi / lam) * -float(down @ look), lam) < 0


# -- GSLC readers (synthetic granule with a cube) ----------------------------
def test_radar_wavelength_from_gslc(gslc_factory):
    p = gslc_factory(ny=40, nx=40, write_geometry=True)
    lam = G.radar_wavelength(p)
    np.testing.assert_allclose(lam, 299792458.0 / 1_239_000_000.0, rtol=1e-9)


def test_read_geometry_cube_from_gslc(gslc_factory):
    p = gslc_factory(ny=40, nx=40, epsg=32611, write_geometry=True)
    cube = G.read_geometry_cube(p)
    assert set(cube.data_vars) == {
        "incidence_angle", "look_angle", "los_east", "los_north"
    }
    assert cube.sizes["height"] == 4
    assert cube.attrs["epsg"] == 32611
    assert cube.attrs["look_direction"] == "Left"
    assert cube.attrs["wavelength"] == pytest.approx(0.24196, abs=1e-4)


def test_missing_cube_raises(gslc_factory):
    # A granule without the geometry cube must fail clearly, not silently.
    p = gslc_factory(ny=32, nx=32, write_geometry=False)
    with pytest.raises((KeyError, OSError)):
        G.read_geometry_cube(p)
