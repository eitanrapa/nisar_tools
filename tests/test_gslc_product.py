"""The RSLC reader, the product grids, and the HDF5 writing helpers.

A synthetic RSLC with the real layout's names and types stands in for a 25 GB
granule; the grid tests pin the numbers of ASF's reference GSLC
(``NISAR_L2_PR_GSLC_022_162_A_006_..._P05023``), whose metadata grids the
workflow must reproduce exactly.
"""

import h5py
import numpy as np
import pytest

import radar_synthetic as rs
from nisar_tools import gslc_product as gp, radar
from nisar_tools.gslc_geocode import default_gslc_name, product_bounds
from nisar_tools.rslc import (
    RSLC, GeoGrid, geogrid_from_bounds, metadata_geogrid, snap_geogrid)

RSLC_NAME = "NISAR_L1_PR_RSLC_023_126_D_083_4005_DHDH_A_20260622T225416_20260622T225450_P05023_N_F_J_002"


@pytest.fixture
def synthetic_rslc(tmp_path):
    """A small RSLC file: orbit, one frequency, two subswaths, a Doppler LUT."""
    orbit = rs.orbit()
    grid = rs.radar_grid(length=300, width=200)
    path = tmp_path / f"{RSLC_NAME}.h5"
    epoch = radar.epoch_units(rs.EPOCH)
    with h5py.File(path, "w") as f:
        ident = f.create_group("science/LSAR/identification")
        ident["lookDirection"] = np.bytes_("Left")
        ident["orbitPassDirection"] = np.bytes_("Descending")
        ident["granuleId"] = np.bytes_(RSLC_NAME)
        ident["listOfFrequencies"] = np.array([b"A"])
        root = f.create_group("science/LSAR/RSLC")
        sw = root.create_group("swaths")
        t = grid.sensing_start + np.arange(grid.length) / grid.prf
        sw["zeroDopplerTime"] = t
        sw["zeroDopplerTime"].attrs["units"] = np.bytes_(epoch)
        sw["zeroDopplerTimeSpacing"] = 1.0 / grid.prf
        fa = sw.create_group("frequencyA")
        fa["listOfPolarizations"] = np.array([b"HH", b"HV"])
        fa["slantRange"] = grid.starting_range + np.arange(grid.width) * grid.range_pixel_spacing
        fa["processedCenterFrequency"] = radar.SPEED_OF_LIGHT / grid.wavelength
        fa["processedRangeBandwidth"] = 40e6
        fa["numberOfSubSwaths"] = 3                  # the third is missing: stop there
        fa["validSamplesSubSwath1"] = np.column_stack([np.full(grid.length, 10), np.full(grid.length, 90)]).astype(np.uint32)
        fa["validSamplesSubSwath2"] = np.column_stack([np.full(grid.length, 100), np.full(grid.length, 190)]).astype(np.uint32)
        rng = np.random.default_rng(0)
        for pol in ("HH", "HV"):
            fa.create_dataset(pol, data=(rng.standard_normal((grid.length, grid.width))
                                         + 1j * rng.standard_normal((grid.length, grid.width))).astype(np.complex64),
                              chunks=(64, 64), compression="gzip", compression_opts=1, shuffle=True)
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
        par["frequencyA/dopplerCentroid"] = 900.0 + np.arange(30.0).reshape(6, 5)
    return path, grid, orbit


# -- RSLC reader --------------------------------------------------------------------
def test_rslc_reader_builds_isce3s_radar_grid(synthetic_rslc):
    path, grid, orbit = synthetic_rslc
    with RSLC(path) as r:
        assert r.granule_id == RSLC_NAME and r.frequencies == ["A"]
        assert r.polarizations("A") == ["HH", "HV"] and r.look_side == "left"
        g = r.radar_grid("A")
        assert g.length == grid.length and g.width == grid.width
        assert g.prf == pytest.approx(grid.prf, rel=1e-12)
        assert g.range_pixel_spacing == pytest.approx(grid.range_pixel_spacing, abs=1e-9)
        assert g.wavelength == pytest.approx(grid.wavelength, rel=1e-12)
        assert r.orbit.reference_epoch == rs.EPOCH
        p, _ = r.orbit.interpolate(orbit.mid_time)
        assert np.allclose(p, rs.truth(orbit.mid_time)[0][0], atol=1e-4)
        assert r.default_posting("A") == (5.0, 5.0)


def test_rslc_reader_valid_samples_and_doppler(synthetic_rslc):
    path, grid, _ = synthetic_rslc
    with RSLC(path) as r:
        valid = r.valid_samples("A")
        assert len(valid) == 2 and valid[0].dtype == np.int64
        assert valid[1][0].tolist() == [100, 190]
        dop = r.doppler_centroid("A")
        t = np.linspace(grid.sensing_start - 1, grid.sensing_stop + 1, 6)
        rg = np.linspace(grid.starting_range - 100, grid.end_range + 100, 5)
        assert dop.eval(t[2], rg[3]) == pytest.approx(900.0 + 2 * 5 + 3)
        window = r.reader("A", "HV")[10:20, 30:45]
        assert np.array_equal(window, r.image("A", "HV")[10:20, 30:45])


def test_footprint_and_default_grid(synthetic_rslc):
    path, grid, orbit = synthetic_rslc
    with RSLC(path) as r:
        fp = r.footprint("A")
        assert fp.shape == (2, 40, 2)
        llh, _ = radar.rdr2geo([grid.sensing_mid], [grid.mid_range], orbit, height=0.0,
                               wavelength=grid.wavelength, look_side=grid.look_side)
        lon, lat = np.degrees(llh[0, :2])
        assert fp[..., 0].min() < lon < fp[..., 0].max()
        assert fp[..., 1].min() < lat < fp[..., 1].max()
        epsg = r.utm_epsg("A")
        assert 32601 <= epsg <= 32660
        g = r.default_geogrid("A", snap=240.0)
        assert g.start_x % 240 == 0 and g.start_y % 240 == 0
        assert g.end_x % 240 == 0 and g.end_y % 240 == 0
        raw = r.default_geogrid("A")
        assert g.start_x <= raw.start_x and g.end_x >= raw.end_x
        assert g.start_y >= raw.start_y and g.end_y <= raw.end_y


# -- grids ----------------------------------------------------------------------------
def test_metadata_grids_reproduce_asfs_reference_gslc():
    product = geogrid_from_bounds(462960.0, 1145520.0, 802080.0, 814320.0, 5.0, 5.0, 32619)
    assert (product.length, product.width) == (66240, 67824)
    cube = metadata_geogrid(product, 500.0, cube=True)
    assert (cube.length, cube.width) == (674, 690)
    assert cube.x_coordinates[[0, -1]].tolist() == [460250.0, 804750.0]
    assert cube.y_coordinates[[0, -1]].tolist() == [1148250.0, 811750.0]
    lut = metadata_geogrid(product, 1000.0)
    assert (lut.length, lut.width) == (332, 340)
    assert lut.x_coordinates[[0, -1]].tolist() == [462500.0, 801500.0]
    assert lut.y_coordinates[[0, -1]].tolist() == [1145500.0, 814500.0]


def test_snap_rounds_outward_and_rejects_off_lattice_snaps():
    g = GeoGrid(1001.0, 5007.0, 5.0, -5.0, 100, 80, 32619)
    s = snap_geogrid(g, 240.0, 240.0)
    assert (s.start_x, s.start_y) == (960.0, 5040.0)
    assert s.end_x == 1680.0 and s.end_y == 4560.0
    with pytest.raises(ValueError):
        snap_geogrid(g, 242.0, 240.0)
    with pytest.raises(ValueError):
        GeoGrid(0.0, 0.0, 5.0, 5.0, 10, 10, 32619)


def test_default_gslc_name_marks_the_product_on_demand():
    assert default_gslc_name(RSLC_NAME) == RSLC_NAME.replace("L1_PR_RSLC", "L2_OD_GSLC")
    with pytest.raises(ValueError):
        default_gslc_name("NISAR_L2_PR_GSLC_whatever")


@pytest.mark.parametrize("layout", ["GSLC", "GUNW"])
def test_product_bounds_reads_the_frame_box(tmp_path, layout):
    group = {"GSLC": "science/LSAR/GSLC/grids/frequencyA",
             "GUNW": "science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram"}[layout]
    spacing = {"GSLC": 5.0, "GUNW": 80.0}[layout]
    path = tmp_path / f"{layout}.h5"
    with h5py.File(path, "w") as f:
        g = f.create_group(group)
        g["xCoordinates"] = 618480.0 + spacing / 2 + spacing * np.arange(int(337680 / spacing))
        g["yCoordinates"] = 1365840.0 - spacing / 2 - spacing * np.arange(int(329760 / spacing))
        g["projection"] = np.uint32(32619)
    bounds, epsg = product_bounds(path)
    assert bounds == (618480.0, 1365840.0, 956160.0, 1036080.0) and epsg == 32619


# -- writing helpers ----------------------------------------------------------------------
def test_chunk_writer_reads_back_like_an_h5py_write(tmp_path):
    pytest.importorskip("osgeo")                     # the projection dataset's WKT
    grid = geogrid_from_bounds(0.0, 3000.0, 3050.0, 0.0, 5.0, 5.0, 32619)   # 600 x 610: edge chunks
    rng = np.random.default_rng(2)
    data = (rng.standard_normal((grid.length, grid.width))
            + 1j * rng.standard_normal((grid.length, grid.width))).astype(np.complex64)
    data[:512, 512:] = complex(np.nan, np.nan)                 # one chunk entirely fill
    data[512:, 512:] = complex(np.nan, 0.0)                    # NaN, but not the fill
    mask = rng.integers(0, 3, data.shape).astype(np.uint8)
    mask[512:, :512] = 255
    ours, ref = tmp_path / "ours.h5", tmp_path / "ref.h5"
    for path in (ours, ref):
        with gp.create_product_file(path) as f:
            gp.create_image_datasets(f, "A", grid, ["HH"])
            root = f[f"{gp.PRODUCT}/grids/frequencyA"]
            if path == ref:
                root["HH"][...] = data
                root["mask"][...] = mask
                continue
            hh, mk = gp.ChunkWriter(root["HH"]), gp.ChunkWriter(root["mask"])
            for r in range(0, grid.length, gp.CHUNK):
                for c in range(0, grid.width, gp.CHUNK):
                    hh.write(r, c, data[r:r + gp.CHUNK, c:c + gp.CHUNK])
                    mk.write(r, c, mask[r:r + gp.CHUNK, c:c + gp.CHUNK])
    with h5py.File(ours) as a, h5py.File(ref) as b:
        root = f"{gp.PRODUCT}/grids/frequencyA"
        assert a[f"{root}/HH"][()].tobytes() == b[f"{root}/HH"][()].tobytes()
        assert np.array_equal(a[f"{root}/mask"][()], b[f"{root}/mask"][()])
        # The all-NaN chunk was never allocated, yet reads back as fill.
        assert a[f"{root}/HH"].id.get_chunk_info_by_coord((0, 512)).byte_offset is None
        assert a[f"{root}/HH"].id.get_chunk_info_by_coord((512, 512)).byte_offset is not None
        assert a[f"{root}/mask"].id.get_chunk_info_by_coord((512, 0)).byte_offset is None
    with pytest.raises(ValueError, match="chunk-aligned"):
        with h5py.File(ours, "r+") as f:
            gp.ChunkWriter(f[f"{gp.PRODUCT}/grids/frequencyA/HH"]).write(1, 0, data[:5, :5])


def test_complex_stats_merge_chunks_into_whole_image_statistics():
    rng = np.random.default_rng(3)
    z = (rng.standard_normal((300, 200)) * 3 + 1 + 1j * rng.standard_normal((300, 200))).astype(np.complex64)
    z[rng.random(z.shape) < 0.2] = np.nan
    stats = gp.ComplexStats()
    for chunk in (z[:100], z[100:250, :50], z[100:250, 50:], z[250:]):
        stats.update(chunk)
    valid = z[np.isfinite(z)]
    parts = np.stack([valid.real.astype(float), valid.imag.astype(float)])
    assert stats.n == valid.size
    assert np.allclose(stats.mean, parts.mean(axis=1), rtol=1e-12)
    assert np.allclose(np.sqrt(stats.ssq / (stats.n - 1)), parts.std(axis=1, ddof=1), rtol=1e-10)
    assert np.array_equal(stats.min, parts.min(axis=1)) and np.array_equal(stats.max, parts.max(axis=1))


def test_projection_attributes_for_utm():
    pytest.importorskip("osgeo")
    attrs = gp.projection_attrs(32619)
    assert attrs["epsg_code"] == 32619 and attrs["utm_zone_number"] == 19
    assert attrs["longitude_of_central_meridian"] == -69.0
    assert attrs["false_easting"] == 500000.0 and attrs["scale_factor_at_central_meridian"] == 0.9996
    assert b"WGS 84 / UTM zone 19N" in attrs["spatial_ref"]
