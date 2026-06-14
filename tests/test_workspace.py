"""Tests for the Zarr workspace, including complex64 round-trips."""

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from nisar_tools import Workspace
from nisar_tools.workspace import WorkspaceError, hash_params


def _complex_ds():
    arr = (np.random.default_rng(0).standard_normal((2, 8, 8))
           + 1j * np.random.default_rng(1).standard_normal((2, 8, 8))).astype(np.complex64)
    return xr.Dataset(
        {"slc": (("time", "y", "x"), da.from_array(arr, chunks=(1, 8, 8)))},
        coords={"time": [0, 1], "y": np.arange(8), "x": np.arange(8)},
        attrs={"epsg": 32611},
    )


def test_complex64_roundtrip(tmp_path):
    ws = Workspace(tmp_path)
    ds = _complex_ds()
    out = ws.store("slc_stack", ds, {"a": 1})
    assert out["slc"].dtype == np.complex64
    np.testing.assert_array_equal(out["slc"].compute().values, ds["slc"].compute().values)


def test_has_matches_on_params(tmp_path):
    ws = Workspace(tmp_path)
    ws.store("slc_stack", _complex_ds(), {"bbox": [1, 2, 3, 4]})
    assert ws.has("slc_stack", {"bbox": [1, 2, 3, 4]})
    assert not ws.has("slc_stack", {"bbox": [9, 9, 9, 9]})
    assert not ws.has("other", {"bbox": [1, 2, 3, 4]})


def test_store_rejects_param_mismatch_without_overwrite(tmp_path):
    ws = Workspace(tmp_path)
    ws.store("slc_stack", _complex_ds(), {"v": 1})
    with pytest.raises(WorkspaceError):
        ws.store("slc_stack", _complex_ds(), {"v": 2})
    # Same params returns the cached store instead of rewriting.
    again = ws.store("slc_stack", _complex_ds(), {"v": 1})
    assert again is not None


def test_region_write_and_done_markers(tmp_path):
    ws = Workspace(tmp_path)
    npair, ny, nx = 3, 8, 8
    template = xr.Dataset(
        {"unw": (("pair", "y", "x"), da.zeros((npair, ny, nx), chunks=(1, ny, nx), dtype="float32"))},
        coords={"pair": np.arange(npair), "y": np.arange(ny), "x": np.arange(nx)},
    )
    ws.init_store("unwrapped", template, {"v": 1})
    assert ws.pairs_done("unwrapped") == set()

    for i in range(npair):
        slab = xr.Dataset(
            {"unw": (("pair", "y", "x"), np.full((1, ny, nx), i, dtype="float32"))}
        )
        ws.write_region("unwrapped", slab, region={"pair": slice(i, i + 1)})
        ws.mark_pair_done("unwrapped", i)

    assert ws.pairs_done("unwrapped") == {0, 1, 2}
    out = ws.load("unwrapped")
    for i in range(npair):
        assert np.all(out["unw"].isel(pair=i).values == i)


def test_hash_is_order_independent(tmp_path):
    assert hash_params({"a": 1, "b": 2}) == hash_params({"b": 2, "a": 1})
