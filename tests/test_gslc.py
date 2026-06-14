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
