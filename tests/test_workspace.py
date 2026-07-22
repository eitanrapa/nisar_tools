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


# -- overwriting a store that is still an input -----------------------------
#
# ``store`` deletes the target before ``to_zarr`` computes, so a dataset that
# still reads that store would race the delete. It does not fail loudly on its
# own -- it writes a correctly-shaped array with a varying fraction of pixels
# silently turned to NaN -- so it has to be refused up front.


def test_overwrite_from_the_store_itself_is_refused(tmp_path):
    ws = Workspace(tmp_path)
    ws.store("slc_stack", _complex_ds(), {"v": 1})

    reopened = ws.load("slc_stack")
    with pytest.raises(WorkspaceError, match="still an input"):
        ws.store("slc_stack", reopened, {"v": 2}, overwrite=True)

    # The store is intact: refusing must not have deleted anything.
    assert ws.exists("slc_stack")
    assert ws.stored_params_hash("slc_stack") == hash_params({"v": 1})


def test_refusal_survives_a_rebuilt_dataset(tmp_path):
    """``merge`` rebuilds the Dataset, which drops xarray's ``encoding``.

    The lazy data is what makes the write unsafe, so that is what is checked.
    """
    ws = Workspace(tmp_path)
    ws.store("slc_stack", _complex_ds(), {"v": 1})

    reopened = ws.load("slc_stack")
    rebuilt = (reopened["slc"] * 2).to_dataset(name="slc")
    assert not rebuilt.encoding.get("source")

    with pytest.raises(WorkspaceError, match="still an input"):
        ws.store("slc_stack", rebuilt, {"v": 2}, overwrite=True)


def test_unchunked_lazy_read_is_also_refused(tmp_path):
    """``chunks=None`` opens lazily without dask; it still reads on write."""
    ws = Workspace(tmp_path)
    ws.store("slc_stack", _complex_ds(), {"v": 1})

    lazy = xr.open_zarr(ws.path("slc_stack"), chunks=None)
    with pytest.raises(WorkspaceError, match="still an input"):
        ws.store("slc_stack", lazy, {"v": 2}, overwrite=True)


def test_overwrite_is_allowed_when_nothing_reads_the_target(tmp_path):
    """The guard must not block the legitimate overwrites."""
    ws = Workspace(tmp_path)
    ws.store("slc_stack", _complex_ds(), {"v": 1})

    # Independent data.
    ws.store("slc_stack", _complex_ds(), {"v": 2}, overwrite=True)
    assert ws.stored_params_hash("slc_stack") == hash_params({"v": 2})

    # Loaded from the target but fully in memory -- writing it reads no disk.
    computed = ws.load("slc_stack").compute()
    ws.store("slc_stack", computed, {"v": 3}, overwrite=True)
    assert ws.stored_params_hash("slc_stack") == hash_params({"v": 3})

    # Reads a *different* store than the one being written.
    ws.store("other", _complex_ds(), {"v": 1})
    ws.store("slc_stack", ws.load("other"), {"v": 4}, overwrite=True)
    assert ws.stored_params_hash("slc_stack") == hash_params({"v": 4})


def test_init_store_refuses_to_delete_its_own_source(tmp_path):
    """Region-written stores take the same path (e.g. unwrapping into igrams)."""
    ws = Workspace(tmp_path)
    ws.store("igrams", _complex_ds(), {"v": 1})

    source = ws.load("igrams")
    template = xr.Dataset(
        {"unw": (("time", "y", "x"), da.zeros((2, 8, 8), chunks=(1, 8, 8)))},
        coords={"time": [0, 1], "y": np.arange(8), "x": np.arange(8)},
    )
    with pytest.raises(WorkspaceError, match="still an input"):
        ws.init_store("igrams", template, {"v": 2}, source=source)

    assert ws.stored_params_hash("igrams") == hash_params({"v": 1})
    # Without a source to conflict with, the same call goes through.
    ws.init_store("igrams", template, {"v": 2})
    assert ws.stored_params_hash("igrams") == hash_params({"v": 2})


def test_source_detection_is_cheap_on_a_large_graph(tmp_path):
    """Sampling per dask layer, not walking every task."""
    import time

    from nisar_tools.workspace import _zarr_source_paths

    ws = Workspace(tmp_path)
    big = xr.Dataset(
        {"slc": (("time", "y", "x"),
                 da.zeros((4, 1024, 1024), chunks=(1, 64, 64), dtype=np.complex64))},
        coords={"time": np.arange(4), "y": np.arange(1024.), "x": np.arange(1024.)},
    )
    ws.store("big", big, {"v": 1})

    derived = (ws.load("big")["slc"] * 2 + 1).to_dataset(name="slc")
    assert len(dict(derived["slc"].data.dask)) > 1000

    start = time.perf_counter()
    found = _zarr_source_paths(derived)
    assert ws.path("big").resolve() in found
    assert time.perf_counter() - start < 1.0


def test_reopened_stage_keeps_its_crs(tmp_path):
    """Zarr has no coord/variable distinction, so spatial_ref comes back wrong.

    rioxarray then reports ``rio.crs is None`` on every field of a reopened
    stack, and anything that reprojects -- plotting, exporting to lon/lat --
    fails with "Provide a CRS-aware DataArray". Both persist() and from_zarr()
    have to restore it.
    """
    import rioxarray  # noqa: F401

    ws = Workspace(tmp_path)
    ds = _complex_ds().rio.write_crs("EPSG:32611")
    assert "spatial_ref" in ds.coords

    reopened = ws.store("slc_stack", ds, {"v": 1})
    assert "spatial_ref" in reopened.coords
    assert "spatial_ref" not in reopened.data_vars
    assert reopened["slc"].rio.crs.to_epsg() == 32611

    # load() is the other door into the same store.
    assert ws.load("slc_stack")["slc"].rio.crs.to_epsg() == 32611


def test_every_stage_class_restores_the_crs_from_zarr(tmp_path):
    import rioxarray  # noqa: F401

    from nisar_tools import GSLCStack, InterferogramStack, LOSStack, UnwrappedStack

    ws = Workspace(tmp_path)
    ws.store("s", _complex_ds().rio.write_crs("EPSG:32611"), {"v": 1})
    for cls in (GSLCStack, InterferogramStack, UnwrappedStack, LOSStack):
        stack = cls.from_zarr(ws.path("s"))
        assert stack.ds["slc"].rio.crs.to_epsg() == 32611, cls.__name__
