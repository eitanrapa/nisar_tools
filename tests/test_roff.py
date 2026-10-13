"""``nisar_tools.roff`` end to end on two synthetic RSLCs: one orbit, one grid, the secondary
the reference's band-limited speckle moved by a known sub-pixel offset. The geometry offsets
must vanish and the ROFF must return the move."""

import h5py
import numpy as np
import pytest

import radar_synthetic as rs
from nisar_tools import radar
from nisar_tools.ampcor import AmpcorParams
from nisar_tools.roff import OffsetLattice, dense_offsets, geometry_offsets

SHIFT = (0.37, -0.81)                    # secondary lines, samples relative to the reference


class _FlatDEM:
    def heights(self, lon, lat):
        return np.zeros(np.shape(lon))


def _speckle(shape, seed, band=0.7):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    F = np.fft.fft2(z)
    fy = np.fft.fftfreq(shape[0])[:, None]
    fx = np.fft.fftfreq(shape[1])[None, :]
    return F * ((np.abs(fy) < band / 2) & (np.abs(fx) < band / 2)), fy, fx


def _write(path, grid, orbit, image, name):
    epoch = radar.epoch_units(rs.EPOCH)
    with h5py.File(path, "w") as f:
        ident = f.create_group("science/LSAR/identification")
        ident["lookDirection"] = np.bytes_("Left")
        ident["orbitPassDirection"] = np.bytes_("Descending")
        ident["granuleId"] = np.bytes_(name)
        ident["listOfFrequencies"] = np.array([b"A"])
        root = f.create_group("science/LSAR/RSLC")
        sw = root.create_group("swaths")
        sw["zeroDopplerTime"] = grid.sensing_start + np.arange(grid.length) / grid.prf
        sw["zeroDopplerTime"].attrs["units"] = np.bytes_(epoch)
        sw["zeroDopplerTimeSpacing"] = 1.0 / grid.prf
        fa = sw.create_group("frequencyA")
        fa["listOfPolarizations"] = np.array([b"HH"])
        fa["slantRange"] = grid.starting_range + np.arange(grid.width) * grid.range_pixel_spacing
        fa["processedCenterFrequency"] = radar.SPEED_OF_LIGHT / grid.wavelength
        fa["numberOfSubSwaths"] = 1
        fa["validSamplesSubSwath1"] = np.column_stack(
            [np.zeros(grid.length), np.full(grid.length, grid.width)]).astype(np.uint32)
        fa.create_dataset("HH", data=image, chunks=(64, 64), compression="gzip",
                          compression_opts=1, shuffle=True)
        o = root.create_group("metadata/orbit")
        o["time"] = orbit.time
        o["time"].attrs["units"] = np.bytes_(epoch)
        o["position"] = orbit.position
        o["velocity"] = orbit.velocity
        o["interpMethod"] = np.bytes_("Hermite")
        o["orbitType"] = np.bytes_("MOE")
        par = root.create_group("metadata/processingInformation/parameters")
        par["zeroDopplerTime"] = np.linspace(grid.sensing_start - 1, grid.sensing_stop + 1, 6)
        par["slantRange"] = np.linspace(grid.starting_range - 100, grid.end_range + 100, 5)
        par["frequencyA/dopplerCentroid"] = np.zeros((6, 5))   # baseband speckle


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    d = tmp_path_factory.mktemp("roff")
    orbit = rs.orbit()
    grid = rs.radar_grid(length=320, width=320)
    F, fy, fx = _speckle((grid.length, grid.width), 4)
    ref = np.fft.ifft2(F).astype(np.complex64)
    sec = np.fft.ifft2(F * np.exp(-2j * np.pi * (fy * SHIFT[0] + fx * SHIFT[1]))).astype(np.complex64)
    a, b = d / "ref.h5", d / "sec.h5"
    _write(a, grid, orbit, ref, "REF")
    _write(b, grid, orbit, sec, "SEC")
    return a, b


LAYERS = {"layer1": AmpcorParams(window_range=32, window_azimuth=32,
                                 half_search_range=8, half_search_azimuth=8)}


def test_geometry_offsets_vanish_for_one_orbit_and_grid(pair):
    dl, ds = geometry_offsets(pair[0], pair[1], _FlatDEM(), np.arange(0, 320, 40.0),
                              np.arange(0, 320, 40.0))
    assert np.isfinite(dl).all() and np.abs(dl).max() < 1e-4 and np.abs(ds).max() < 1e-4


def test_dense_offsets_return_the_shift(pair):
    ds = dense_offsets(pair[0], pair[1], _FlatDEM(), layers=LAYERS, margin=24, skip_azimuth=24,
                       skip_range=24, tile=(4, 4), workers=2, geometry_step=16)
    lat = OffsetLattice.for_grid(320, 320, LAYERS, 24, 24, 24)
    assert ds.sizes["row"] == lat.length and ds.sizes["col"] == lat.width
    down = ds.down.values[0]
    across = ds.across.values[0]
    assert np.abs(down - SHIFT[0]).max() < 0.03
    assert np.abs(across - SHIFT[1]).max() < 0.03
    assert np.abs(ds.geometryLineOffset.values).max() < 1e-4
    vg = ds.groundTrackVelocity.values
    np.testing.assert_allclose(ds.alongTrackOffset.values[0], down * vg / 1520.0, rtol=1e-6)
    np.testing.assert_allclose(ds.slantRangeOffset.values[0], across * 3.75, rtol=1e-6)
    # a subset is the same computation
    sub = dense_offsets(pair[0], pair[1], _FlatDEM(), layers=LAYERS, margin=24, skip_azimuth=24,
                        skip_range=24, rows=slice(2, 5), cols=slice(1, 4), workers=1, geometry_step=16)
    np.testing.assert_array_equal(sub.down.values[0], down[2:5, 1:4])
