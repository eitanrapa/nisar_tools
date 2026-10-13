"""Secondary registration in RSLC -> GSLC geocoding (``RegistrationField``).

A registration field is the per-pixel, map-domain counterpart of the timing
corrections: it moves where each output pixel samples the RSLC and leaves the
flattening range alone. So a *constant* field must be indistinguishable from
constant azimuth/range correction LUTs, and the geocoded value must be the
RSLC's signal at the shifted position.
"""

import numpy as np
import pytest

import radar_synthetic as rs
from nisar_tools import RegistrationField, radar
from nisar_tools._geocode_kernels import interpolate_slc, radar_indices
from nisar_tools.gslc_geocode import pixel_radar_indices


class _FlatDEM:
    def __init__(self, height=0.0):
        self.height = height

    def heights(self, lon, lat):
        return np.full(np.shape(lon), self.height)


def _map_points(grid, orbit, lines, samples, height=0.0):
    """UTM 17N x, y of radar positions (line, sample) at a constant height."""
    from pyproj import Transformer

    t = grid.sensing_start + np.asarray(lines, float) / grid.prf
    r = grid.starting_range + np.asarray(samples, float) * grid.range_pixel_spacing
    llh, ok = radar.rdr2geo(t, r, orbit, height=height, wavelength=grid.wavelength,
                            look_side=grid.look_side, threshold=1e-9)
    assert ok.all()
    x, y = Transformer.from_crs("EPSG:4326", "EPSG:32617", always_xy=True).transform(
        np.degrees(llh[:, 0]), np.degrees(llh[:, 1]))
    return np.asarray(x), np.asarray(y)


def test_constant_registration_equals_constant_timing_corrections():
    orbit = rs.orbit()
    grid = rs.radar_grid(length=64, width=64)
    doppler = rs.constant_doppler(grid, 950.0)
    x, y = _map_points(grid, orbit, [20.3, 31.7, 40.1], [22.9, 30.2, 41.6])
    field = RegistrationField(np.linspace(x.min() - 500, x.max() + 500, 5),
                              np.linspace(y.max() + 500, y.min() - 500, 4),
                              np.full((4, 5), 2.0), np.full((4, 5), -3.0))
    dem = _FlatDEM()
    _, _, _, a1, g1, f1, m1 = pixel_radar_indices(x, y, 32617, dem, grid, orbit, doppler, [],
                                                  registration=field)

    class _Corr:
        ts = np.array([grid.sensing_start - 1, grid.sensing_stop + 1])
        rs_ = np.array([grid.starting_range - 100, grid.end_range + 100])
        azimuth = radar.LUT2d.from_coords(rs_, ts, np.full((2, 2), 2.0 / grid.prf))
        slant_range = radar.LUT2d.from_coords(rs_, ts, np.full((2, 2), -3.0 * grid.range_pixel_spacing))

    _, _, _, a2, g2, f2, m2 = pixel_radar_indices(x, y, 32617, dem, grid, orbit, doppler, [],
                                                  corrections=_Corr())
    np.testing.assert_allclose(a1, a2, atol=1e-6)
    np.testing.assert_allclose(g1, g2, atol=1e-6)
    np.testing.assert_array_equal(f1, f2)          # flattening never moves
    _, _, _, a0, g0, f0, _ = pixel_radar_indices(x, y, 32617, dem, grid, orbit, doppler, [])
    np.testing.assert_allclose(a1 - a0, 2.0, atol=1e-6)
    np.testing.assert_allclose(g1 - g0, -3.0, atol=1e-6)
    np.testing.assert_array_equal(f1, f0)


def test_registered_pixel_samples_the_signal_at_the_shifted_position():
    """A band-limited image with a Doppler carrier, registered by a fractional
    shift, geocodes to the continuous signal at the shifted radar position
    (times the unchanged flattening phase)."""
    orbit = rs.orbit()
    grid = rs.radar_grid(length=96, width=96)
    fd = 950.0
    doppler = rs.constant_doppler(grid, fd)
    rng = np.random.default_rng(1)
    freqs = [((rng.uniform(-0.3, 0.3) + fd / grid.prf), rng.uniform(-0.3, 0.3)) for _ in range(6)]
    amps = rng.normal(size=6) + 1j * rng.normal(size=6)
    signal = lambda l, m: sum(a * np.exp(2j * np.pi * (fa * l + fr * m))  # noqa: E731
                              for (fa, fr), a in zip(freqs, amps))
    lines, samples = np.mgrid[0:96, 0:96].astype(float)
    block = signal(lines, samples).astype(np.complex64)
    x, y = _map_points(grid, orbit, [40.2, 47.9, 55.5], [38.6, 50.3, 57.1])
    shift = (0.37, -0.81)
    field = RegistrationField(np.linspace(x.min() - 500, x.max() + 500, 3),
                              np.linspace(y.max() + 500, y.min() - 500, 3),
                              np.full((3, 3), shift[0]), np.full((3, 3), shift[1]))
    _, _, _, az, rg, flat, _ = pixel_radar_indices(x, y, 32617, _FlatDEM(), grid, orbit, doppler, [],
                                                   registration=field)
    got = interpolate_slc(block, 0, 0, az, rg, flat, grid, doppler, flatten=False)
    want = signal(az, rg)
    # the 8-tap kernel's error at these frequencies is ~1e-2 of the amplitude
    np.testing.assert_allclose(got, want, atol=0.03 * np.abs(want).max())
    assert np.allclose(az - shift[0], radar_indices(*_llh(x, y), grid, orbit, doppler, [])[0], atol=1e-6)


def _llh(x, y):
    from pyproj import Transformer

    lon, lat = Transformer.from_crs("EPSG:32617", "EPSG:4326", always_xy=True).transform(x, y)
    return np.deg2rad(lon), np.deg2rad(lat), np.zeros(np.size(x))


def test_field_is_bilinear_and_clamps_outside_its_grid():
    x = np.array([0.0, 10.0, 20.0])
    y = np.array([100.0, 90.0])                      # descending, as on a map grid
    X, Y = np.meshgrid(x, y)
    field = RegistrationField(x, y, 0.1 * X + 0.01 * Y, -0.2 * X)
    lines, samples = field(np.array([5.0, 15.0, -50.0, 99.0]), np.array([95.0, 92.5, 95.0, 80.0]))
    np.testing.assert_allclose(lines, [0.1 * 5 + 0.95, 0.1 * 15 + 0.925, 0.0 + 0.95, 2.0 + 0.90])
    np.testing.assert_allclose(samples, [-1.0, -3.0, 0.0, -4.0])


def test_from_offsets_converts_metres_to_lines_and_samples():
    x = np.array([0.0, 80.0]); y = np.array([80.0, 0.0])
    field = RegistrationField.from_offsets(x, y, np.full((2, 2), -1.8), np.full((2, 2), 0.6),
                                           along_track_spacing=4.5, range_spacing=3.0)
    lines, samples = field(np.array([40.0]), np.array([40.0]))
    assert lines[0] == pytest.approx(-0.4) and samples[0] == pytest.approx(0.2)


def test_field_must_be_finite_and_regular():
    x = np.array([0.0, 1.0]); y = np.array([1.0, 0.0])
    with pytest.raises(ValueError, match="NaN"):
        RegistrationField(x, y, np.array([[0.0, np.nan], [0.0, 0.0]]), np.zeros((2, 2)))
    with pytest.raises(ValueError, match="regular"):
        RegistrationField(np.array([0.0, 1.0, 3.0]), y, np.zeros((2, 3)), np.zeros((2, 3)))
