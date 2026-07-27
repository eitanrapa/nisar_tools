"""Tests for the GSLC granule reader."""

import dask.array as da
import numpy as np
import pyproj
import pytest

from nisar_tools import GSLC


def test_metadata_and_lazy_data(gslc_factory):
    path = gslc_factory(ny=128, nx=96, epsg=32611, direction="Descending")
    g = GSLC(path)

    assert g.epsg == 32611
    assert g.direction == "Descending"
    assert g.shape == (128, 96)
    assert g.datetime is not None

    data = g.data
    # Lazy: backed by dask, not yet read.
    assert isinstance(data.data, da.Array)
    assert data.dims == ("y", "x")
    assert data.rio.crs.to_epsg() == 32611
    g.close()


def test_read_chunks_targets_a_block_count_and_stays_aligned(gslc_factory):
    """The read-block size is chosen for the *region*, so a small crop still has
    enough dask tasks to keep every core inflating -- it used to inherit the
    granule's fixed 2048, which made the notebook's bbox 2x2 blocks."""
    path = gslc_factory(ny=512, nx=512)   # conftest writes 64x64 HDF5 chunks
    g = GSLC(path)
    try:
        file_chunks = g._dset.chunks
        for ny, nx, target in [(512, 512, 16), (512, 512, 4), (128, 128, 64)]:
            cy, cx = g.read_chunks(ny, nx, target_blocks=target)
            # Aligned to the file's own chunking, so a block read decompresses
            # whole HDF5 chunks and never straddles one.
            assert cy % file_chunks[0] == 0 and cx % file_chunks[1] == 0
            assert cy >= file_chunks[0] and cx >= file_chunks[1]
            assert cy <= g.chunks[0] and cx <= g.chunks[1]
        # More blocks requested => blocks no larger.
        few = g.read_chunks(512, 512, target_blocks=4)
        many = g.read_chunks(512, 512, target_blocks=64)
        assert many[0] <= few[0] and many[1] <= few[1]
    finally:
        g.close()


def test_crop_sizes_its_blocks_from_the_crop(gslc_factory):
    path = gslc_factory(ny=512, nx=512)
    g = GSLC(path)
    try:
        tr = pyproj.Transformer.from_crs(
            f"EPSG:{g.epsg}", "EPSG:4326", always_xy=True
        )
        # A bbox covering roughly the middle half of the granule.
        x, y = g.x_coords, g.y_coords
        x0, x1 = x[len(x) // 4], x[3 * len(x) // 4]
        y0, y1 = y[3 * len(y) // 4], y[len(y) // 4]
        (lo0, la0), (lo1, la1) = tr.transform(x0, y0), tr.transform(x1, y1)
        crop = g.crop(min(lo0, lo1), max(lo0, lo1), min(la0, la1), max(la0, la1),
                      target_blocks=16)
        # The whole-granule default would leave this a single block.
        assert crop.data.numblocks[0] * crop.data.numblocks[1] > 1
        # And it still reads the right values.
        np.testing.assert_array_equal(
            crop.compute().values,
            g.data.sel(x=crop.x, y=crop.y).compute().values,
        )
    finally:
        g.close()


def test_data_handle_lifetime(gslc_factory):
    # Computing must work while the GSLC (and its file handle) is alive.
    path = gslc_factory(ny=64, nx=64)
    g = GSLC(path)
    arr = g.data.compute()
    assert arr.shape == (64, 64)
    assert arr.dtype == np.complex64
    g.close()


def test_descending_y_is_descending(gslc_factory):
    path = gslc_factory(direction="Descending")
    g = GSLC(path)
    y = g.y_coords
    assert y[0] > y[-1]  # descending
    g.close()


def test_crop_descending(gslc_factory):
    # A known geographic sub-box should crop a non-empty, smaller region.
    epsg = 32611
    path = gslc_factory(ny=200, nx=200, epsg=epsg, direction="Descending",
                        x0=400000.0, y0=4_000_000.0, dx=10.0, dy=10.0)
    g = GSLC(path)

    # Pick native coords well inside the grid and convert to lon/lat.
    tr = pyproj.Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    lon_a, lat_a = tr.transform(400500.0, 3_999_500.0)
    lon_b, lat_b = tr.transform(401000.0, 3_999_000.0)

    cropped = g.crop(min(lon_a, lon_b), max(lon_a, lon_b),
                     min(lat_a, lat_b), max(lat_a, lat_b))
    assert cropped.sizes["y"] > 0 and cropped.sizes["x"] > 0
    assert cropped.sizes["y"] < 200 and cropped.sizes["x"] < 200
    # Still descending in y.
    yv = cropped["y"].values
    assert yv[0] > yv[-1]
    g.close()


def test_context_manager(gslc_factory):
    path = gslc_factory()
    with GSLC(path) as g:
        assert g.shape[0] > 0
    assert g._file is None
