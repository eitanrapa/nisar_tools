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


# -- an RSLC's own range and timing errors ------------------------------------------------
def test_range_and_azimuth_shifts_equal_a_registration_and_leave_the_flattening():
    """`range_shift(t, r)` metres and `azimuth_shift(t)` seconds move the sampled radar
    position exactly as a registration of `r / dr` samples and `t * prf` lines would."""
    orbit = rs.orbit()
    grid = rs.radar_grid(length=64, width=64)
    doppler = rs.constant_doppler(grid, 950.0)
    x, y = _map_points(grid, orbit, [20.3, 31.7, 40.1], [22.9, 30.2, 41.6])
    field = RegistrationField(np.linspace(x.min() - 500, x.max() + 500, 3),
                              np.linspace(y.max() + 500, y.min() - 500, 3),
                              np.full((3, 3), -0.8), np.full((3, 3), 2.5))
    dem = _FlatDEM()
    _, _, _, a1, g1, f1, m1 = pixel_radar_indices(x, y, 32617, dem, grid, orbit, doppler, [],
                                                  registration=field)
    seen = []

    def range_shift(t, r):
        seen.append((np.asarray(t).copy(), np.asarray(r).copy()))
        return np.full(np.shape(t), 2.5 * grid.range_pixel_spacing)

    _, _, _, a2, g2, f2, m2 = pixel_radar_indices(
        x, y, 32617, dem, grid, orbit, doppler, [], range_shift=range_shift,
        azimuth_shift=lambda t: np.full(np.shape(t), -0.8 / grid.prf))
    np.testing.assert_allclose(a1, a2, atol=1e-9)
    np.testing.assert_allclose(g1, g2, atol=1e-9)
    np.testing.assert_array_equal(f1, f2)
    np.testing.assert_array_equal(m1, m2)
    # the hooks see the pixel's own zero-Doppler time and slant range
    _, _, _, a0, g0, _, _ = pixel_radar_indices(x, y, 32617, dem, grid, orbit, doppler, [])
    np.testing.assert_allclose(seen[0][0], grid.sensing_start + a0 / grid.prf, atol=1e-9)
    np.testing.assert_allclose(seen[0][1], grid.starting_range + g0 * grid.range_pixel_spacing, atol=1e-6)


def test_a_timing_report_zeroes_the_mask_in_a_slips_transition():
    from nisar_tools import l0b

    orbit = rs.orbit()
    grid = rs.radar_grid(length=64, width=64)
    doppler = rs.constant_doppler(grid, 950.0)
    x, y = _map_points(grid, orbit, [10.0, 30.0, 50.0], [30.0, 30.0, 30.0])
    t_mid = grid.sensing_start + 30.0 / grid.prf
    problem = dict(kind=l0b.KIND_NAMES[l0b.EARLY], code=l0b.EARLY, count=1, start=0.0, end=0.0,
                   full=(t_mid + 0.004, t_mid + 1.0), transitions=[(t_mid - 0.004, t_mid + 0.004)])
    report = l0b.TimingReport(granule="x", timings=[], problems=[problem], prf_product=1909.6,
                              prf_expected=1909.6, prf_commanded=1909.6, span=(0.0, 1e6), lead=(2.0, 2.0),
                              aperture=0.008)
    valid = [np.column_stack([np.zeros(grid.length), np.full(grid.length, grid.width)]).astype(np.int64)]
    _, _, _, a0, _, _, m0 = pixel_radar_indices(x, y, 32617, _FlatDEM(), grid, orbit, doppler, valid)
    _, _, _, a1, _, _, m1 = pixel_radar_indices(x, y, 32617, _FlatDEM(), grid, orbit, doppler, valid,
                                                azimuth_shift=report)
    assert list(m0) == [1, 1, 1] and list(m1) == [1, 0, 1]
    np.testing.assert_allclose(a1 - a0, [0.0, 0.0, -grid.prf / 1909.6], atol=1e-6)


def test_geocoding_repairs_the_mask_unless_isce3s_masks_are_kept():
    from nisar_tools.gslc_geocode import _valid_samples

    class Rslc:
        def valid_samples(self, frequency):
            v = np.tile(np.array([[0, 50]], np.int64), (10, 1))
            v[4] = (0, 10)                                    # a line that drops out of the swath
            return [v]

    class Report:
        def repair_mask(self, valid):
            return [v + 1 for v in valid]

    assert _valid_samples(Rslc(), "A", True, Report())[0][4].tolist() == [1, 51]
    assert _valid_samples(Rslc(), "A", True, lambda t: 0.0 * t)[0][4].tolist() == [0, 50]
    assert _valid_samples(Rslc(), "A", False, Report())[0][4].tolist() == [0, 10]


def _two_band_rslc(tmp_path):
    """A small two-band RSLC (band B's range samples 8 times A's) and a flat DEM covering it."""
    import h5py
    import rasterio
    from rasterio.transform import from_origin

    orbit = rs.orbit()
    grid = rs.radar_grid(length=120, width=160)
    path = tmp_path / "NISAR_L1_PR_RSLC_023_126_D_083_4005_DHDH_A_20260622T225416_20260622T225450_P05023_N_F_J_002.h5"
    epoch = radar.epoch_units(rs.EPOCH)
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        ident = f.create_group("science/LSAR/identification")
        ident["lookDirection"] = np.bytes_("Left")
        ident["orbitPassDirection"] = np.bytes_("Descending")
        ident["granuleId"] = np.bytes_(path.stem)
        ident["listOfFrequencies"] = np.array([b"A", b"B"])
        root = f.create_group("science/LSAR/RSLC")
        sw = root.create_group("swaths")
        sw["zeroDopplerTime"] = grid.sensing_start + np.arange(grid.length) / grid.prf
        sw["zeroDopplerTime"].attrs["units"] = np.bytes_(epoch)
        sw["zeroDopplerTimeSpacing"] = 1.0 / grid.prf
        par = root.create_group("metadata/processingInformation/parameters")
        par["zeroDopplerTime"] = np.linspace(grid.sensing_start - 1, grid.sensing_stop + 1, 6)
        par["slantRange"] = np.linspace(grid.starting_range - 100, grid.end_range + 100, 5)
        for band, factor, bandwidth in (("A", 1, 40e6), ("B", 8, 5e6)):
            width = grid.width // factor
            fb = sw.create_group(f"frequency{band}")
            fb["listOfPolarizations"] = np.array([b"HH"])
            fb["slantRange"] = grid.starting_range + np.arange(width) * grid.range_pixel_spacing * factor
            fb["processedCenterFrequency"] = radar.SPEED_OF_LIGHT / grid.wavelength
            fb["processedRangeBandwidth"] = bandwidth
            fb["numberOfSubSwaths"] = 1
            fb["validSamplesSubSwath1"] = np.column_stack(
                [np.zeros(grid.length), np.full(grid.length, width)]).astype(np.uint32)
            fb["HH"] = (rng.standard_normal((grid.length, width))
                        + 1j * rng.standard_normal((grid.length, width))).astype(np.complex64)
            par[f"frequency{band}/dopplerCentroid"] = np.full((6, 5), 950.0)
        o = root.create_group("metadata/orbit")
        o["time"] = orbit.time
        o["time"].attrs["units"] = np.bytes_(epoch)
        o["position"] = orbit.position
        o["velocity"] = orbit.velocity
        o["interpMethod"] = np.bytes_("Hermite")
        o["orbitType"] = np.bytes_("MOE")
    t = grid.sensing_start + np.array([0.0, 0.0, 1.0, 1.0]) * (grid.length - 1) / grid.prf
    r = grid.starting_range + np.array([0.0, 1.0, 0.0, 1.0]) * (grid.width - 1) * grid.range_pixel_spacing
    llh, ok = radar.rdr2geo(t, r, orbit, height=0.0, wavelength=grid.wavelength, look_side=grid.look_side,
                            threshold=1e-9)
    assert ok.all()
    lon, lat = np.degrees(llh[:, 0]), np.degrees(llh[:, 1])
    dem = tmp_path / "dem.tif"
    step = 0.001
    west, north = lon.min() - 0.1, lat.max() + 0.1
    nx, ny = int((lon.max() + 0.1 - west) / step) + 1, int((north - lat.min() + 0.1) / step) + 1
    with rasterio.open(dem, "w", driver="GTiff", width=nx, height=ny, count=1, dtype="float32",
                       crs="EPSG:4326", transform=from_origin(west, north, step, step)) as dst:
        dst.write(np.zeros((1, ny, nx), np.float32))
    return path, dem


def test_one_field_is_refused_for_two_bands(tmp_path):
    """The shifts are in one band's lines and samples; band B's samples are 8 times A's."""
    from nisar_tools import geocode_rslc
    rslc, dem = _two_band_rslc(tmp_path)
    field = RegistrationField(np.array([0.0, 1.0]), np.array([1.0, 0.0]), np.zeros((2, 2)), np.full((2, 2), 3.0))
    with pytest.raises(ValueError, match="one field per band"):
        geocode_rslc(rslc, tmp_path / "out.h5", dem, registration=field)


def test_each_band_is_registered_with_its_own_field(tmp_path, monkeypatch):
    from nisar_tools import geocode_rslc, gslc_geocode
    rslc, dem = _two_band_rslc(tmp_path)
    fields = {band: RegistrationField(np.array([0.0, 1.0]), np.array([1.0, 0.0]), np.zeros((2, 2)),
                                      np.full((2, 2), shift), description=band)
              for band, shift in (("A", 8.0), ("B", 1.0))}
    seen = {}

    class _Stop(Exception):
        pass

    def _record(f, rslc, freq, *args, registration=None, **kwargs):
        seen[freq] = registration
        if freq == "B":                                      # the last band: nothing further to see
            raise _Stop
        return {}

    monkeypatch.setattr(gslc_geocode, "_geocode_frequency", _record)
    with pytest.raises(_Stop):
        geocode_rslc(rslc, tmp_path / "out.h5", dem, registration=fields, solid_earth_tides=False)
    assert seen == fields
    seen.clear()
    with pytest.raises(_Stop):                              # one band, one field
        geocode_rslc(rslc, tmp_path / "out.h5", dem, frequencies=["B"], registration=fields["B"],
                     solid_earth_tides=False)
    assert seen == {"B": fields["B"]}


@pytest.mark.parametrize("mode, clamped", [("pixels", False), (True, False), (False, True)])
def test_only_the_pixel_fixes_clamp_the_timing_corrections(tmp_path, monkeypatch, mode, clamped):
    """A correction applied outside its table moves pixels near the swath edge, so the default, which keeps
    ISCE3's pixels, skips it there as ISCE3 does."""
    from nisar_tools import geocode_rslc, gslc_geocode
    rslc, dem = _two_band_rslc(tmp_path)
    seen = []

    class _Stop(Exception):
        pass

    def _record(*args, clamp_corrections=None, **kwargs):
        seen.append(clamp_corrections)
        raise _Stop

    monkeypatch.setattr(gslc_geocode, "pixel_radar_indices", _record)
    with pytest.raises(_Stop):
        geocode_rslc(rslc, tmp_path / "out.h5", dem, frequencies=["A"], solid_earth_tides=False,
                     isce3_compatible=mode, workers=1)
    assert seen and set(seen) == {clamped}
