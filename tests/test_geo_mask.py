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


def test_water_mask_cache_keyed_on_grid(tmp_path, monkeypatch):
    # The workspace cache must recompute when the grid origin or the mask
    # parameters change, not just when the shape changes.
    from nisar_tools import Workspace
    from nisar_tools import mask as mask_mod

    ws = Workspace(tmp_path / "ws")
    x = 400000.0 + 10.0 * np.arange(8)
    y = 4_000_000.0 - 10.0 * np.arange(6)

    calls = []

    def fake_make(x_coords, y_coords, epsg_code, buffer=0.05,
                  resolution="f", spacing="5e"):
        calls.append(1)
        return xr.DataArray(
            np.ones((len(y_coords), len(x_coords))),
            coords={"y": y_coords, "x": x_coords},
            dims=("y", "x"),
        )

    monkeypatch.setattr(mask_mod, "make_water_mask", fake_make)

    mask_mod.water_mask_for_grid(x, y, 32611, workspace=ws)
    mask_mod.water_mask_for_grid(x, y, 32611, workspace=ws)  # cache hit
    assert len(calls) == 1
    # Same shape, different origin: must recompute.
    mask_mod.water_mask_for_grid(x + 100.0, y, 32611, workspace=ws)
    assert len(calls) == 2
    # Same grid, different coastline resolution: must recompute.
    mask_mod.water_mask_for_grid(x + 100.0, y, 32611, workspace=ws,
                                 resolution="i")
    assert len(calls) == 3


def test_mask_water_applies_to_zarr_style_stack(monkeypatch):
    # Regression test for two bugs in mask_water:
    #  - the mask's rio ``spatial_ref`` coordinate collided with the
    #    ``spatial_ref`` data variable in zarr-backed stacks (MergeError);
    #  - ``.where(mask)`` on a 1/NaN mask was a no-op because NaN is truthy,
    #    so water pixels were silently kept.
    pytest.importorskip("rioxarray")
    from nisar_tools import mask as mask_mod
    from nisar_tools.unwrap import UnwrappedStack

    x = 470_000.0 + 500.0 * np.arange(40)
    y = 3_630_000.0 - 500.0 * np.arange(40)

    def fake_grdlandmask(region, spacing, maskvalues, resolution, registration):
        lon = np.linspace(region[0], region[1], 60)
        lat = np.linspace(region[2], region[3], 50)
        data = np.ones((50, 60))
        data[:, :20] = np.nan  # western third is water
        return xr.DataArray(data, coords={"y": lat, "x": lon}, dims=("y", "x"))

    pygmt = pytest.importorskip("pygmt")
    monkeypatch.setattr(pygmt, "grdlandmask", fake_grdlandmask)

    ds = xr.Dataset(
        {
            "unw": (("pair", "y", "x"), np.ones((2, 40, 40), np.float32)),
            "spatial_ref": ((), 0),  # as it comes back from zarr
        },
        coords={"y": y, "x": x, "pair": [0, 1]},
        attrs={"epsg": 32611},
    )
    masked = UnwrappedStack(ds).mask_water(resolution="i")

    frac = float(np.isfinite(masked.ds["unw"].isel(pair=0)).mean())
    assert 0.0 < frac < 1.0  # some water blanked, some land kept
    assert "spatial_ref" in masked.ds.data_vars


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

def test_mask_cache_only_caches_the_mask_not_the_data(tmp_path, monkeypatch):
    """``mask_cache`` reads like a persist target but is not one.

    It stores the coastline mask so GMT is not re-run for the same grid; the
    masked values stay lazy until the caller persists them.
    """
    pytest.importorskip("rioxarray")
    from nisar_tools import Workspace
    from nisar_tools.unwrap import UnwrappedStack

    pygmt = pytest.importorskip("pygmt")

    def fake_grdlandmask(region, spacing, maskvalues, resolution, registration):
        lon = np.linspace(region[0], region[1], 60)
        lat = np.linspace(region[2], region[3], 50)
        data = np.ones((50, 60))
        data[:, :20] = np.nan
        return xr.DataArray(data, coords={"y": lat, "x": lon}, dims=("y", "x"))

    monkeypatch.setattr(pygmt, "grdlandmask", fake_grdlandmask)

    ws = Workspace(tmp_path / "ws")
    x = 470_000.0 + 500.0 * np.arange(40)
    y = 3_630_000.0 - 500.0 * np.arange(40)
    ds = xr.Dataset(
        {"unw": (("pair", "y", "x"), np.ones((1, 40, 40), np.float32)),
         "conncomp": (("pair", "y", "x"), np.ones((1, 40, 40), np.uint32))},
        coords={"y": y, "x": x, "pair": [0]},
        attrs={"epsg": 32611},
    )
    ws.store("unwrapped", ds, {"stage": "unwrapped"})
    unw = UnwrappedStack.from_zarr(ws.path("unwrapped"))

    masked = unw.mask_water(mask_cache=ws, resolution="i")
    assert ws.exists("water_mask")          # the *mask* was cached
    # Spacing tracks the grid (500 m pixels, 2x oversampled), not a constant.
    assert masked.ds.attrs["water_mask"] == {"resolution": "i", "spacing": "250e"}

    # The masked values were not written: reloading gives the phase back whole.
    reloaded = UnwrappedStack.from_zarr(ws.path("unwrapped"))
    assert np.isfinite(reloaded.ds["unw"].compute().values).all()
    assert not np.isfinite(masked.ds["unw"].compute().values).all()


def test_unwrapped_stack_persists_a_masked_result(tmp_path, monkeypatch):
    """persist() is how a lazily-masked unwrapped stack reaches disk."""
    pytest.importorskip("rioxarray")
    from nisar_tools import Workspace, WorkspaceError
    from nisar_tools.unwrap import UnwrappedStack

    pygmt = pytest.importorskip("pygmt")

    def fake_grdlandmask(region, spacing, maskvalues, resolution, registration):
        lon = np.linspace(region[0], region[1], 60)
        lat = np.linspace(region[2], region[3], 50)
        data = np.ones((50, 60))
        data[:, :20] = np.nan
        return xr.DataArray(data, coords={"y": lat, "x": lon}, dims=("y", "x"))

    monkeypatch.setattr(pygmt, "grdlandmask", fake_grdlandmask)

    ws = Workspace(tmp_path / "ws")
    x = 470_000.0 + 500.0 * np.arange(40)
    y = 3_630_000.0 - 500.0 * np.arange(40)
    ds = xr.Dataset(
        {"unw": (("pair", "y", "x"), np.ones((1, 40, 40), np.float32)),
         "conncomp": (("pair", "y", "x"), np.ones((1, 40, 40), np.uint32))},
        coords={"y": y, "x": x, "pair": [0]},
        attrs={"epsg": 32611},
    )
    ws.store("unwrapped", ds, {"stage": "unwrapped"})
    unw = UnwrappedStack.from_zarr(ws.path("unwrapped"))
    masked = unw.mask_water(mask_cache=ws, resolution="i")

    out = masked.persist(ws, "unwrapped_masked")
    assert isinstance(out, UnwrappedStack)
    back = UnwrappedStack.from_zarr(ws.path("unwrapped_masked"))
    expected = masked.ds["unw"].compute().values
    np.testing.assert_array_equal(back.ds["unw"].compute().values, expected)

    # Masking is part of the stage identity, so an unmasked write differs.
    assert ws.stored_params_hash("unwrapped_masked") != ws.stored_params_hash(
        "unwrapped"
    )

    # Writing back over the store it reads from is refused, hence "new name".
    with pytest.raises(WorkspaceError, match="still an input"):
        masked.persist(ws, "unwrapped", overwrite=True)


def test_mask_spacing_tracks_the_grid_not_the_native_pixel():
    """The coastline only has to resolve at the pixel it is sampled onto.

    GMT's ``e`` suffix is *metres*: the old fixed ``"5e"`` default read like
    "5 arc-seconds" but asked for a 5 m coastline, so a multilooked stack still
    paid for a full-resolution mask -- ~1000x more nodes than its grid could
    represent, and about a minute of GMT time on a coastal crop.
    """
    from nisar_tools.mask import grid_spacing_arg

    for pixel_m, expected in ((150.0, "75e"), (300.0, "150e"), (5.0, "2.5e")):
        x = 470_000.0 + pixel_m * np.arange(50)
        y = 3_630_000.0 - pixel_m * np.arange(50)
        assert grid_spacing_arg(x, y, 32611) == expected

    # Anisotropic grids follow the finer axis.
    x = 470_000.0 + 200.0 * np.arange(20)
    y = 3_630_000.0 - 100.0 * np.arange(20)
    assert grid_spacing_arg(x, y, 32611) == "50e"

    # Geographic grids are already in degrees, GMT's unit-less default.
    lon = -118.0 + 0.002 * np.arange(20)
    lat = 34.0 + 0.002 * np.arange(20)
    assert grid_spacing_arg(lon, lat, 4326) == "0.001"


def test_mask_cache_key_includes_the_derived_spacing(tmp_path, monkeypatch):
    """Two grids of equal shape and origin but different pixel size must not
    share a cached mask -- the shape/origin key alone cannot tell them apart."""
    pytest.importorskip("rioxarray")
    from nisar_tools import Workspace, mask as mask_mod

    calls = []
    real = mask_mod.make_water_mask

    def counting(x_coords, y_coords, epsg_code, **kwargs):
        calls.append(kwargs.get("spacing"))
        return xr.DataArray(
            np.ones((len(y_coords), len(x_coords))),
            coords={"y": y_coords, "x": x_coords}, dims=["y", "x"],
        )

    monkeypatch.setattr(mask_mod, "make_water_mask", counting)
    ws = Workspace(tmp_path / "ws")

    # Same nx/ny/x0/y0, different pixel size.
    for pixel_m in (100.0, 200.0):
        x = 470_000.0 + pixel_m * np.arange(30)
        y = 3_630_000.0 - pixel_m * np.arange(30)
        mask_mod.water_mask_for_grid(x, y, 32611, workspace=ws, resolution="i")

    assert calls == ["50e", "100e"], calls   # recomputed, not reused
    assert real is not None
