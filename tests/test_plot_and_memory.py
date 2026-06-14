"""Plotting smoke tests and a memory-bounded end-to-end smoke test."""

import gc
import tracemalloc

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # headless

from nisar_tools import GSLC, GSLCStack, Workspace


def _build_igrams(gslc_factory, ws_dir, ndates=2, ny=80, nx=80):
    gslcs = [
        GSLC(gslc_factory(ny=ny, nx=nx, seed=k,
                          datetime_str=f"2025-11-{10 + k:02d}T00:00:00.000000000"))
        for k in range(ndates)
    ]
    stack = GSLCStack.from_gslcs(gslcs).persist(Workspace(ws_dir), "slc_stack")
    for g in gslcs:
        g.close()
    return stack.form_interferograms(pairs="sequential", looks=5, downsample=True)


def test_plot_wrapped_returns_fig_ax(gslc_factory, tmp_path):
    igrams = _build_igrams(gslc_factory, tmp_path / "ws")
    fig, ax = igrams.plot_wrapped(pair=0)
    assert fig is not None and ax is not None


def _peak_pipeline_mb(gslc_factory, ws_dir, ndates, ny=256, nx=256):
    """Run the full persist -> interferogram -> persist pipeline; return the
    peak Python allocation in MB."""
    gslcs = [
        GSLC(gslc_factory(ny=ny, nx=nx, seed=k,
                          name=f"{ws_dir.name}_{k}.h5",
                          datetime_str=str(np.datetime64("2025-01-01") + np.timedelta64(12 * k, "D")) + "T00:00:00.000000000"))
        for k in range(ndates)
    ]
    ws = Workspace(ws_dir)

    gc.collect()
    tracemalloc.start()
    stack = GSLCStack.from_gslcs(gslcs).persist(ws, "slc_stack")
    for g in gslcs:
        g.close()
    igrams = stack.form_interferograms(pairs="sequential", looks=5, downsample=True)
    igrams = igrams.persist(ws, "igrams")
    _ = igrams.ds["coherence"].isel(pair=0).compute()  # force the graph
    peak_mb = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()
    return peak_mb


def test_memory_does_not_scale_with_stack_length(gslc_factory, tmp_path):
    """Out-of-core evidence: a 10x longer stack must not need ~10x the memory.

    Because every stage streams chunk-by-chunk to Zarr and only one
    pair/slice is ever in flight, peak allocation stays roughly flat as the
    number of acquisitions grows.
    """
    # Warm up caches so the first measured run isn't penalised for one-time
    # allocations.
    _peak_pipeline_mb(gslc_factory, tmp_path / "warmup", ndates=3)

    peak_small = _peak_pipeline_mb(gslc_factory, tmp_path / "small", ndates=4)
    peak_large = _peak_pipeline_mb(gslc_factory, tmp_path / "large", ndates=40)

    # 10x the dates. Linear (whole-stack-in-memory) growth would be ~10x; a
    # streaming pipeline grows only with the task graph, far slower. The bound
    # cleanly separates the two regimes.
    assert peak_large < peak_small * 6, (
        f"peak grew ~linearly with stack length: small={peak_small:.1f} MB, "
        f"large={peak_large:.1f} MB"
    )


def test_persisted_stores_stream_per_slice(gslc_factory, tmp_path):
    """Structural guarantee: stores chunk the stack dimension at 1, so the
    scheduler only ever processes one acquisition / pair at a time."""
    igrams = _build_igrams(gslc_factory, tmp_path / "ws", ndates=4)
    ws = Workspace(tmp_path / "ws")
    igrams = igrams.persist(ws, "igrams")

    slc = ws.load("slc_stack")["slc"]
    ig = ws.load("igrams")["igram"]
    # First element of the chunk tuple for the stack dim must be 1.
    assert slc.chunksizes["time"][0] == 1
    assert ig.chunksizes["pair"][0] == 1
