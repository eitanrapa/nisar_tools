"""End-to-end pipeline tests against the legacy module on synthetic data."""

import numpy as np
import pytest

import legacy_reference as legacy
from nisar_tools import GSLC, GSLCStack, InterferogramStack, Workspace


def _two_gslcs(gslc_factory, ny=120, nx=100, seed0=0, seed1=1):
    p1 = gslc_factory(ny=ny, nx=nx, seed=seed0,
                      datetime_str="2025-11-28T02:32:50.000000000")
    p2 = gslc_factory(ny=ny, nx=nx, seed=seed1,
                      datetime_str="2025-12-10T02:32:50.000000000")
    return GSLC(p1), GSLC(p2)


@pytest.mark.parametrize("convolution", ["Uniform", "Gaussian"])
@pytest.mark.parametrize("downsample", [False, True])
def test_interferogram_matches_legacy(gslc_factory, convolution, downsample):
    g1, g2 = _two_gslcs(gslc_factory)
    stack = GSLCStack.from_gslcs([g1, g2])

    igrams = stack.form_interferograms(
        pairs="sequential", looks=5, downsample=downsample, convolution=convolution
    )
    igram_new = igrams.ds["igram"].isel(pair=0).compute().values
    corr_new = igrams.ds["coherence"].isel(pair=0).compute().values

    # Legacy oracle on the same raw arrays.
    c1 = g1.data.compute().values
    c2 = g2.data.compute().values
    max_y = c1.shape[0] // 5 * 5
    max_x = c1.shape[1] // 5 * 5
    igram_ref, corr_ref = legacy._calculate_multilooked_interferograms(
        c1, c2, max_x, max_y, 5, downsample, convolution
    )

    np.testing.assert_allclose(igram_new, igram_ref, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(corr_new, corr_ref, rtol=1e-4, atol=1e-4)
    g1.close()
    g2.close()


def test_stack_time_sorted_and_pairs(gslc_factory):
    # Provide out of order; stack should sort by acquisition time.
    p_late = gslc_factory(seed=2, datetime_str="2025-12-10T00:00:00.000000000")
    p_early = gslc_factory(seed=3, datetime_str="2025-11-28T00:00:00.000000000")
    g_late, g_early = GSLC(p_late), GSLC(p_early)
    stack = GSLCStack.from_gslcs([g_late, g_early])
    times = stack.ds["time"].values
    assert times[0] < times[1]
    assert stack.make_pairs("sequential") == [(0, 1)]
    g_late.close()
    g_early.close()


def test_merge_union_grid(gslc_factory):
    # Two adjacent frames offset in x; merge should span the union.
    p1 = gslc_factory(ny=64, nx=64, x0=400000.0, seed=4)
    p2 = gslc_factory(ny=64, nx=64, x0=400000.0 + 64 * 10.0, seed=5)
    g1, g2 = GSLC(p1), GSLC(p2)
    s1 = GSLCStack.from_gslcs([g1])
    s2 = GSLCStack.from_gslcs([g2])
    merged = s1.merge(s2)
    # Union of two non-overlapping 64-wide frames => 128 columns.
    assert merged.sizes["x"] == 128
    assert merged.sizes["y"] == 64
    g1.close()
    g2.close()


def test_persist_resume_roundtrip(gslc_factory, tmp_path):
    g1, g2 = _two_gslcs(gslc_factory)
    ws = Workspace(tmp_path)
    params = {"files": ["a", "b"], "bbox": None}

    stack = GSLCStack.from_gslcs([g1, g2])
    stack = stack.persist(ws, "slc_stack", **params)
    g1.close()
    g2.close()  # safe: persist reopened from Zarr

    # Resume path: store is reused without recompute.
    assert ws.has("slc_stack", {"stage": "slc_stack", "epsg": stack.epsg,
                                "files": ["a", "b"], "bbox": None})
    reopened = GSLCStack.from_zarr(ws.path("slc_stack"))
    assert reopened.sizes["time"] == 2

    igrams = stack.form_interferograms(looks=5, downsample=True)
    igrams = igrams.persist(ws, "igrams")
    assert ws.exists("igrams")
    again = InterferogramStack.from_zarr(ws.path("igrams"))
    assert again.sizes["pair"] == 1
