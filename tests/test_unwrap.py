"""Tests for SNAPHU unwrapping, region writes, and per-pair resume."""

import numpy as np
import pytest

from nisar_tools import GSLC, GSLCStack, Workspace
from nisar_tools.interferogram import InterferogramStack
from nisar_tools.unwrap import UnwrappedStack


def _igram_stack(gslc_factory, ws_dir, ndates=3, ny=80, nx=80):
    gslcs = []
    for k in range(ndates):
        p = gslc_factory(
            ny=ny, nx=nx, seed=k,
            datetime_str=f"2025-11-{10 + k:02d}T00:00:00.000000000",
        )
        gslcs.append(GSLC(p))
    # Persist the SLC stack first (reopens from Zarr) so the granule file
    # handles can be safely closed before the lazy downstream work runs.
    stack = GSLCStack.from_gslcs(gslcs).persist(Workspace(ws_dir), "slc_stack")
    for g in gslcs:
        g.close()
    return stack.form_interferograms(pairs="all", looks=5, downsample=True)


def test_unwrap_runs_and_shapes(gslc_factory, tmp_path):
    igrams = _igram_stack(gslc_factory, tmp_path / "ws0")
    igrams = igrams.persist(Workspace(tmp_path / "ws0"), "igrams")
    ws = Workspace(tmp_path / "ws1")

    unw = igrams.unwrap(ws, nproc=1)
    assert isinstance(unw, UnwrappedStack)
    assert unw.sizes["pair"] == igrams.sizes["pair"]
    assert unw.ds["unw"].dtype == np.float32
    assert unw.ds["conncomp"].dtype == np.uint32
    # All pairs marked done.
    assert ws.pairs_done("unwrapped") == set(range(igrams.sizes["pair"]))


def test_unwrap_resumes_after_interruption(gslc_factory, tmp_path, monkeypatch):
    import json
    import snaphu

    igrams = _igram_stack(gslc_factory, tmp_path / "ws0")
    igrams = igrams.persist(Workspace(tmp_path / "ws0"), "igrams")
    ws = Workspace(tmp_path / "ws1")
    npair = igrams.sizes["pair"]
    assert npair >= 3

    # Count SNAPHU invocations to prove which pairs are actually computed.
    calls = {"n": 0}
    real_unwrap = snaphu.unwrap

    def counting_unwrap(*args, **kwargs):
        calls["n"] += 1
        return real_unwrap(*args, **kwargs)

    monkeypatch.setattr("nisar_tools.unwrap.snaphu.unwrap", counting_unwrap)

    # First full run computes every pair.
    unw1 = igrams.unwrap(ws, nproc=1)
    first = unw1.ds["unw"].compute()
    assert calls["n"] == npair
    assert ws.pairs_done("unwrapped") == set(range(npair))

    # Simulate an interruption that only completed pair 0, then rerun with the
    # same parameters: resume must skip pair 0 and recompute the rest.
    ws._done_path("unwrapped").write_text(json.dumps({"pairs_done": [0]}))
    calls["n"] = 0
    unw2 = igrams.unwrap(ws, nproc=1)
    assert calls["n"] == npair - 1  # pair 0 skipped
    assert ws.pairs_done("unwrapped") == set(range(npair))
    # Deterministic: the recomputed result matches the original.
    np.testing.assert_array_equal(unw2.ds["unw"].compute().values, first.values)
