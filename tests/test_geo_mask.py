"""Tests for geographic helpers and (optionally) the water mask."""

import numpy as np
import pytest
import xarray as xr

from nisar_tools import geo


def test_bbox_roundtrip():
    epsg = 32611
    x_min, x_max, y_min, y_max = geo.bbox_to_native(-118.5, -118.0, 34.0, 34.5, epsg)
    assert x_max > x_min and y_max > y_min
    lon_min, lon_max, lat_min, lat_max = geo.native_bbox_to_lonlat(
        x_min, x_max, y_min, y_max, epsg
    )
    # Round-trips back to roughly the original box.
    assert lon_min == pytest.approx(-118.5, abs=0.05)
    assert lat_max == pytest.approx(34.5, abs=0.05)


def test_project_to_latlon_from_array():
    epsg = 32611
    x = 400000.0 + 10.0 * np.arange(20)
    y = 4_000_000.0 - 10.0 * np.arange(16)
    data = np.random.default_rng(0).standard_normal((16, 20))
    out = geo.project_to_latlon(data, x_coords=x, y_coords=y, epsg_code=epsg)
    assert isinstance(out, xr.DataArray)
    assert out.rio.crs.to_epsg() == 4326


def test_water_mask_if_gmt_available():
    pytest.importorskip("pygmt")
    from nisar_tools import mask

    epsg = 32611
    x = 400000.0 + 100.0 * np.arange(30)
    y = 4_000_000.0 - 100.0 * np.arange(30)
    try:
        m = mask.make_water_mask(x, y, epsg)
    except Exception as exc:  # GMT coastline data may be unavailable offline
        pytest.skip(f"GMT land mask unavailable: {exc}")
    assert m.sizes["y"] == 30 and m.sizes["x"] == 30
    # Land cells are 1, water cells are NaN.
    vals = np.unique(m.values[~np.isnan(m.values)])
    assert set(vals.tolist()) <= {1.0}
