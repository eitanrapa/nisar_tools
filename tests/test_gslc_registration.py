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


def _two_band_rslc(tmp_path, centres=None):
    """A small two-band RSLC (band B's range samples 8 times A's) and a flat DEM covering it.

    `centres` gives each band's centre frequency (default: both the grid's)."""
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
            fb["processedCenterFrequency"] = (centres or {}).get(band, radar.SPEED_OF_LIGHT / grid.wavelength)
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


@pytest.mark.parametrize("epsg, mode, cube, lut", [(None, "pixels", 500.0, 1000.0), (4326, "pixels", 0.005, 0.01),
                                                   (4326, True, 0.01, 0.01)])
def test_the_metadata_grids_are_posted_in_the_output_crs_units(tmp_path, monkeypatch, epsg, mode, cube, lut):
    """Metres on a projected grid, degrees on a geographic one (ISCE3's defaults, and its 0.01 cubes when asked)."""
    from nisar_tools import geocode_rslc, gslc_geocode
    rslc, dem = _two_band_rslc(tmp_path)
    seen = []
    real = gslc_geocode.metadata_geogrid

    class _Stop(Exception):
        pass

    def _record(grid, spacing, cube=False):
        seen.append((float(spacing), cube))
        if len(seen) == 2:
            raise _Stop
        return real(grid, spacing, cube=cube)

    monkeypatch.setattr(gslc_geocode, "metadata_geogrid", _record)
    kwargs = {} if epsg is None else dict(epsg=epsg, posting=(5e-5, 5e-5), snap=None)
    with pytest.raises(_Stop):
        geocode_rslc(rslc, tmp_path / "out.h5", dem, frequencies=["A"], solid_earth_tides=False,
                     isce3_compatible=mode, **kwargs)
    assert seen == [(cube, True), (lut, False)]


# -- the carrier convention ---------------------------------------------------------------
def _target(fd_over_prf, line, sample, n=96):
    """A band-limited point target at fractional (`line`, `sample`) whose azimuth spectrum sits
    on the Doppler carrier `fd_over_prf` (cycles per line); real and positive at its peak."""
    fa = fd_over_prf + np.linspace(-0.35, 0.35, 15)
    fr = np.linspace(-0.35, 0.35, 15)
    lines, samples = np.mgrid[0:n, 0:n].astype(float)
    az = np.exp(2j * np.pi * fa[:, None, None] * (lines - line)).sum(0)
    rg = np.exp(2j * np.pi * fr[:, None, None] * (samples - sample)).sum(0)
    return (az * rg).astype(np.complex64)


def test_a_geometric_carrier_is_taken_before_the_shift():
    """The signal at the shifted position, with the carrier re-applied `carrier_lines` earlier."""
    grid = rs.radar_grid(length=96, width=96)
    fd = 950.0
    doppler = rs.constant_doppler(grid, fd)
    rng = np.random.default_rng(2)
    freqs = [((rng.uniform(-0.3, 0.3) + fd / grid.prf), rng.uniform(-0.3, 0.3)) for _ in range(6)]
    amps = rng.normal(size=6) + 1j * rng.normal(size=6)
    signal = lambda l, m: sum(a * np.exp(2j * np.pi * (fa * l + fr * m))  # noqa: E731
                              for (fa, fr), a in zip(freqs, amps))
    lines, samples = np.mgrid[0:96, 0:96].astype(float)
    block = signal(lines, samples).astype(np.complex64)
    az = np.array([40.2, 47.9, 55.5]) + 1.37
    rg = np.array([38.6, 50.3, 57.1])
    zero = np.zeros(3)
    got = interpolate_slc(block, 0, 0, az, rg, zero, grid, doppler, flatten=False,
                          carrier_lines=np.full(3, 1.37))
    want = signal(az, rg) * np.exp(-2j * np.pi * fd / grid.prf * 1.37)
    np.testing.assert_allclose(got, want, atol=0.03 * np.abs(want).max())


def test_a_field_that_follows_motion_keeps_the_unregistered_phase():
    """A target moved 0.6 line along track in the secondary. Sampled where it was, its phase carries the
    carrier term (the native-Doppler look); registered with the default carrier it keeps that phase at the
    full amplitude; registered with the carrier at the shifted position it loses the term."""
    grid = rs.radar_grid(length=96, width=96)
    fd = 950.0
    doppler = rs.constant_doppler(grid, fd)
    moved = 0.6
    block = _target(fd / grid.prf, 48.0 + moved, 47.3)
    one, zero = np.ones(1), np.zeros(1)
    kw = dict(flatten=False)
    unregistered = interpolate_slc(block, 0, 0, 48.0 * one, 47.3 * one, zero, grid, doppler, **kw)[0]
    geometric = interpolate_slc(block, 0, 0, (48.0 + moved) * one, 47.3 * one, zero, grid, doppler,
                                carrier_lines=moved * one, **kw)[0]
    shifted = interpolate_slc(block, 0, 0, (48.0 + moved) * one, 47.3 * one, zero, grid, doppler, **kw)[0]
    term = 2 * np.pi * fd / grid.prf * moved
    assert np.angle(unregistered) == pytest.approx(-term, abs=0.02)
    assert np.angle(geometric * np.conj(unregistered)) == pytest.approx(0.0, abs=0.02)
    assert abs(geometric) > 1.2 * abs(unregistered)
    assert np.angle(shifted) == pytest.approx(0.0, abs=0.02)


def test_fields_add_and_each_part_keeps_its_carrier():
    x = np.array([0.0, 10.0]); y = np.array([10.0, 0.0])
    measured = RegistrationField(x, y, np.full((2, 2), 0.4), np.full((2, 2), 1.0), description="measured")
    reference = RegistrationField(x, y, np.full((2, 2), 0.25), np.full((2, 2), -2.0), description="reference",
                                  carrier="shifted")
    plain = lambda xx, yy: (np.full(np.shape(xx), 0.1), np.zeros(np.shape(xx)))  # noqa: E731
    total = measured + reference + plain
    lines, samples = total(np.array([5.0]), np.array([5.0]))
    assert lines[0] == pytest.approx(0.75) and samples[0] == pytest.approx(-1.0)
    assert total.carrier_lines(np.array([5.0]), np.array([5.0]))[0] == pytest.approx(0.4)
    assert len(total.parts) == 3
    left = plain + measured
    assert left(np.array([5.0]), np.array([5.0]))[0][0] == pytest.approx(0.5)
    assert left.carrier_lines(np.array([5.0]), np.array([5.0]))[0] == pytest.approx(0.4)
    assert "measured (carrier geometric)" in total.description
    assert "reference (carrier shifted)" in total.description
    assert reference.carrier_lines(np.array([5.0]), np.array([5.0]))[0] == 0.0
    with pytest.raises(ValueError, match="carrier"):
        RegistrationField(x, y, np.zeros((2, 2)), np.zeros((2, 2)), carrier="zero-doppler")


def test_carrier_phase_turns_a_shifted_pair_into_a_geometric_one():
    """With the field on the secondary, `reference * conj(secondary)` gains `carrier_phase`."""
    grid = rs.radar_grid(length=96, width=96)
    fd = 950.0
    doppler = rs.constant_doppler(grid, fd)
    block = _target(fd / grid.prf, 48.0, 47.3)
    x = np.array([0.0, 10.0]); y = np.array([10.0, 0.0])
    field = RegistrationField(x, y, np.full((2, 2), 0.83), np.zeros((2, 2)))
    one, zero = np.ones(1), np.zeros(1)
    az = (47.5 + 0.83) * one
    shifted = interpolate_slc(block, 0, 0, az, 47.3 * one, zero, grid, doppler, flatten=False)[0]
    geometric = interpolate_slc(block, 0, 0, az, 47.3 * one, zero, grid, doppler, flatten=False,
                                carrier_lines=0.83 * one)[0]
    ref = 1.0 + 0.0j
    gained = np.angle((ref * np.conj(geometric)) * np.conj(ref * np.conj(shifted)))
    phase = field.carrier_phase(np.array([5.0]), np.array([5.0]), fd, grid.prf)[0]
    assert np.angle(np.exp(1j * (gained - phase))) == pytest.approx(0.0, abs=1e-4)


@pytest.mark.parametrize("carrier, carried", [("geometric", 0.4), ("shifted", None)])
def test_geocoding_hands_the_geometric_lines_to_the_interpolator(tmp_path, monkeypatch, carrier, carried):
    """End to end: the same RSLC geocoded with each convention differs by exactly the carrier phase."""
    from nisar_tools import geocode_rslc, gslc_geocode
    rslc, dem = _two_band_rslc(tmp_path)
    real = gslc_geocode.interpolate_slc
    seen = []

    class _Stop(Exception):
        pass

    def _record(*args, carrier_lines=None, **kwargs):
        seen.append((carrier_lines, real(*args, carrier_lines=carrier_lines, **kwargs)))
        raise _Stop

    monkeypatch.setattr(gslc_geocode, "interpolate_slc", _record)
    field = RegistrationField(np.array([0.0, 1.0]), np.array([1.0, 0.0]), np.full((2, 2), 0.4),
                              np.full((2, 2), 0.3), carrier=carrier)     # constant: beyond its grid, the edge
    with pytest.raises(_Stop):
        geocode_rslc(rslc, tmp_path / "out.h5", dem, frequencies=["A"], registration=field,
                     solid_earth_tides=False, workers=1)
    lines, values = seen[0]
    if carried is None:
        assert lines is None
    else:
        np.testing.assert_allclose(lines, carried)
    other = "shifted" if carrier == "geometric" else "geometric"
    seen.clear()
    with pytest.raises(_Stop):
        geocode_rslc(rslc, tmp_path / "out2.h5", dem, frequencies=["A"], solid_earth_tides=False, workers=1,
                     registration=RegistrationField(field.x, field.y, field.lines, field.samples, carrier=other))
    _, values_other = seen[0]
    geometric, shifted = (values[0], values_other[0]) if carrier == "geometric" else (values_other[0], values[0])
    ok = np.isfinite(geometric) & np.isfinite(shifted) & (np.abs(shifted) > 0)
    assert ok.sum() > 100
    grid = rs.radar_grid(length=120, width=160)
    want = -2 * np.pi * 950.0 / grid.prf * 0.4
    np.testing.assert_allclose(np.angle(geometric[ok] * np.conj(shifted[ok])), want, atol=1e-5)
    np.testing.assert_allclose(np.abs(geometric[ok]), np.abs(shifted[ok]), rtol=1e-5)


# -- band B's measured range offset and the TEC correction ------------------------------------
def _band_offset(value=4.1):
    """A `BandOffset` of one cell: band B `value` m beyond A everywhere."""
    from nisar_tools import qa
    return qa.BandOffset(granule="g", blocks=[], times=np.array([0.0, 1e6]), ranges=np.array([0.0, 1e7]),
                         offset=np.array([[value]]), measured=np.array([[True]]))


class _Centres:
    """An RSLC's centre frequencies, as the 40 + 5 MHz modes have them."""

    def center_frequency(self, freq):
        return {"A": 1239e6, "B": 1293.5e6}[freq]


def _tec_json(tmp_path, sub_tec, start=82_300.0, n=41, spacing=10.0):
    """An IMAGEN-style TEC JSON with a constant sub-orbital TEC (TECU), as `test_tec` builds them."""
    import json
    t = start + spacing * np.arange(n)
    doc = {"utc": [str(rs.EPOCH + np.timedelta64(int(s), "s")) for s in t],
           "totTecNr": [sub_tec + 3.0] * n, "topTecNr": [3.0] * n,
           "totTecFr": [sub_tec + 3.5] * n, "topTecFr": [3.5] * n}
    path = tmp_path / "NISAR_ANC_TEC_test.json"
    path.write_text(json.dumps(doc))
    return path


def test_a_measured_band_offset_loses_its_dispersive_part_where_the_tec_correction_applies():
    """Measured on the RSLC, band B's offset holds B's dispersive delay against A, which the TEC correction removes
    on its own: taken out where the correction applies (ISCE3 skips it outside its table, `clamp` does not)."""
    from nisar_tools.corrections import TimingCorrections
    from nisar_tools.gslc_geocode import _range_shift_hook

    tec = radar.LUT2d.from_coords([800e3, 900e3], [100.0, 200.0], np.full((2, 2), 10.0))   # B's own delay, m
    corr = TimingCorrections(grid=None, slant_range=tec, ionosphere_range=tec)
    offset = _band_offset()
    expected = 4.1 + 10.0 * ((1293.5 / 1239.0) ** 2 - 1)          # A is delayed 0.90 m more than B: 5.0 m
    t = np.array([150.0, 150.0, 300.0, np.nan])                  # inside, beyond in range, beyond in time, no pixel
    r = np.array([850e3, 950e3, 850e3, np.nan])
    for given in (offset, offset.shift):
        np.testing.assert_allclose(_range_shift_hook(given, corr, _Centres(), "B")(t, r), [expected, 4.1, 4.1, 4.1])
        np.testing.assert_allclose(_range_shift_hook(given, corr, _Centres(), "B", clamp=True)(t, r)[:3], expected)
    no_tec = TimingCorrections(grid=None, slant_range=radar.LUT2d())
    np.testing.assert_allclose(_range_shift_hook(offset, no_tec, _Centres(), "B")(t, r), 4.1)
    level = lambda t, r: np.full(np.shape(t), 4.1)                # noqa: E731 -- an offset of B's own, as given
    assert _range_shift_hook(level, corr, _Centres(), "B") is level
    assert _range_shift_hook(None, corr, _Centres(), "B") is None


def test_geocoding_takes_the_tec_delay_out_of_a_measured_band_offset(tmp_path, monkeypatch):
    """With a TEC file, band B's measured offset loses B's dispersive delay against A at the two bands' own
    frequencies, and the run configuration says so; without one it is used as measured."""
    from nisar_tools import geocode_rslc, gslc_geocode
    from nisar_tools.corrections import tec_luts

    centres = {"A": 1239e6, "B": 1293.5e6}
    rslc, dem = _two_band_rslc(tmp_path, centres)
    tec = _tec_json(tmp_path, 40.0)                               # a dusk pass
    offset = _band_offset()
    seen = {}

    class _Stop(Exception):
        pass

    def _record(f, rslc_, freq, pols, geogrid, grid, *args, range_shift=None, **kwargs):
        seen["grid"], seen["shift"] = grid, range_shift
        return {}

    def _run_config(**kwargs):
        seen["label"] = kwargs["range_shift"]
        raise _Stop

    monkeypatch.setattr(gslc_geocode, "_geocode_frequency", _record)
    monkeypatch.setattr(gslc_geocode, "_write_cubes", lambda *args, **kwargs: None)
    monkeypatch.setattr(gslc_geocode._gslc_metadata, "describe_run", _run_config)
    with pytest.raises(_Stop):
        geocode_rslc(rslc, tmp_path / "out.h5", dem, frequencies=["B"], tec_file=tec, solid_earth_tides=False,
                     range_shift={"B": offset.shift}, workers=1)
    grid = seen["grid"]
    t = np.array([grid.sensing_start + 0.3 * (grid.sensing_stop - grid.sensing_start)])
    r = np.array([grid.starting_range + 0.6 * (grid.end_range - grid.starting_range)])
    tau = {band: tec_luts(tec, f0, rs.orbit(), grid)[1].eval(t, r) for band, f0 in centres.items()}
    assert 0.8 < (tau["A"] - tau["B"])[0] < 1.4                   # ~0.0217 m per slant TECU
    np.testing.assert_allclose(seen["shift"](t, r), 4.1 + (tau["A"] - tau["B"]), rtol=1e-9)
    assert seen["label"]["B"].startswith("BandOffset of g;") and "TEC correction" in seen["label"]["B"]
    seen.clear()
    with pytest.raises(_Stop):
        geocode_rslc(rslc, tmp_path / "out.h5", dem, frequencies=["B"], solid_earth_tides=False,
                     range_shift={"B": offset}, workers=1)
    np.testing.assert_allclose(seen["shift"](t, r), 4.1)
    assert seen["label"]["B"] == "BandOffset of g"
