"""End-to-end pipeline tests against the legacy module on synthetic data."""

import warnings

import numpy as np
import pyproj
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
        pairs="sequential", looks=5, downsample=downsample,
        convolution=convolution, nan_aware=False,
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


def test_merge_keeps_the_canonical_chunk_grid(gslc_factory):
    """Both sides are re-chunked before the combine, so nothing fragments.

    Crops start mid-chunk at their own phase; combining them directly makes
    dask split both into the union of their chunk boundaries, which multiplies
    the chunk count and the graph for the same work.
    """
    from nisar_tools._base import SPATIAL_CHUNK

    ny = nx = 5000  # > 2 chunks per axis, and a ragged final chunk
    p1 = gslc_factory(ny=ny, nx=nx, x0=400000.0, y0=4_000_000.0, seed=4)
    p2 = gslc_factory(ny=ny, nx=nx, x0=400000.0 + 10.0 * 1234,
                      y0=4_000_000.0 - 10.0 * 2345, seed=5)   # not chunk-aligned
    g1, g2 = GSLC(p1), GSLC(p2)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            merged = GSLCStack.from_gslcs([g1]).merge(GSLCStack.from_gslcs([g2]))
            chunks = merged.ds["slc"].chunks

        perf = [w for w in caught if "Increasing number of chunks" in str(w.message)]
        assert not perf, f"chunk grid fragmented: {[str(w.message) for w in perf]}"

        ny_u, nx_u = merged.sizes["y"], merged.sizes["x"]
        assert len(chunks[1]) == -(-ny_u // SPATIAL_CHUNK)
        assert len(chunks[2]) == -(-nx_u // SPATIAL_CHUNK)
        # Zarr requires the last chunk to be no larger than the first.
        assert chunks[1][-1] <= chunks[1][0] and chunks[2][-1] <= chunks[2][0]
    finally:
        g1.close()
        g2.close()


def _lazy_stack(x0, y0, ny=40000, nx=40000, ntime=3, step=10.0):
    """A granule-scale stack backed by lazy zeros — no file, no compute."""
    import dask.array as dsk
    import xarray as xr

    data = dsk.zeros((ntime, ny, nx), chunks=(1, 2048, 2048), dtype=np.complex64)
    ds = xr.Dataset(
        {"slc": (("time", "y", "x"), data)},
        coords={
            "time": np.arange(ntime).astype("datetime64[s]"),
            "y": y0 - step * np.arange(ny),
            "x": x0 + step * np.arange(nx),
        },
        attrs={"epsg": 32611, "direction": "Descending",
               "x_spacing": step, "y_spacing": -step},
    )
    return GSLCStack(ds)


def test_merge_does_not_warn_at_granule_scale():
    """Regression guard for dask's "Increasing number of chunks" warning.

    It only fires once the unified grid reaches ten times the largest input's
    partition count, so a small stack cannot exercise it -- hence the granule
    dimensions here (lazy zeros, so this costs nothing). Reaching the union by
    reindexing used to trip it: xarray builds a per-element boolean mask for
    the missing positions, and broadcasting that mask is what blew up the grid.
    """
    step = 10.0
    a = _lazy_stack(400000.0, 4_000_000.0)
    # Neighbouring frame, offset by a non-multiple of the chunk size in both
    # axes, so the two crops sit at different chunk phases.
    b = _lazy_stack(400000.0 + step * 12345, 4_000_000.0 - step * 30011)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        merged = a.merge(b)
        chunks = merged.ds["slc"].chunks

    perf = [str(w.message) for w in caught
            if "Increasing number of chunks" in str(w.message)]
    assert not perf, perf

    from nisar_tools._base import SPATIAL_CHUNK
    assert len(chunks[1]) == -(-merged.sizes["y"] // SPATIAL_CHUNK)
    assert len(chunks[2]) == -(-merged.sizes["x"] // SPATIAL_CHUNK)
    # Union spans both frames, and keeps this stack's descending y.
    assert merged.sizes["y"] == 40000 + 30011
    assert merged.sizes["x"] == 40000 + 12345
    assert merged.y[0] > merged.y[-1]
    assert merged.x[0] < merged.x[-1]


def test_merge_overlap_prefers_self(gslc_factory):
    """Where the two frames overlap, this stack's samples win."""
    ny = nx = 64
    step = 10.0
    p1 = gslc_factory(ny=ny, nx=nx, x0=400000.0, seed=4)
    # Half a frame to the east: columns 32..63 of self overlap 0..31 of other.
    p2 = gslc_factory(ny=ny, nx=nx, x0=400000.0 + step * 32, seed=5)
    g1, g2 = GSLC(p1), GSLC(p2)
    try:
        s1 = GSLCStack.from_gslcs([g1])
        s2 = GSLCStack.from_gslcs([g2])
        merged = s1.merge(s2)
        assert merged.sizes["x"] == nx + 32

        got = merged.ds["slc"].isel(time=0).compute()
        # Overlap column: must be self's value, not other's.
        overlap_x = s1.x[40]
        np.testing.assert_array_equal(
            got.sel(x=overlap_x).values,
            s1.ds["slc"].isel(time=0).sel(x=overlap_x).compute().values,
        )
        # A column only the other frame covers is filled from it.
        only_other = s2.x[-1]
        np.testing.assert_array_equal(
            got.sel(x=only_other).values,
            s2.ds["slc"].isel(time=0).sel(x=only_other).compute().values,
        )
    finally:
        g1.close()
        g2.close()


def test_merge_preserves_axis_direction(gslc_factory):
    """The reversing slice must land the axes the same way sortby did."""
    p1 = gslc_factory(ny=64, nx=64, x0=400000.0, seed=4)
    p2 = gslc_factory(ny=64, nx=64, x0=400000.0 + 64 * 10.0, seed=5)
    g1, g2 = GSLC(p1), GSLC(p2)
    try:
        s1 = GSLCStack.from_gslcs([g1])
        merged = s1.merge(GSLCStack.from_gslcs([g2]))
        # Descending pass: y descends, x ascends, as in the source granules.
        assert (s1.y[0] > s1.y[-1]) == (merged.y[0] > merged.y[-1])
        assert np.all(np.diff(merged.y) < 0)
        assert np.all(np.diff(merged.x) > 0)
        # Self's own samples survive unchanged at their own coordinates.
        got = merged.ds["slc"].sel(x=s1.x, y=s1.y).compute().values
        np.testing.assert_array_equal(got, s1.ds["slc"].compute().values)
    finally:
        g1.close()
        g2.close()


def test_merge_across_epsg(gslc_factory):
    # Two frames on one track straddling the UTM zone 10/11 boundary
    # (lon -120): frame 1 gridded in EPSG:32610, frame 2 in EPSG:32611,
    # acquired seconds apart on the same pass.
    ny = nx = 64
    dx = dy = 10.0
    to10 = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32610", always_xy=True)
    to11 = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32611", always_xy=True)
    x10, y10 = to10.transform(-120.0, 34.0)
    x11, y11 = to11.transform(-120.0, 34.0)

    p1 = gslc_factory(
        ny=ny, nx=nx, epsg=32610, dx=dx, dy=dy, seed=6,
        x0=round(x10) - nx * dx, y0=round(y10),
        datetime_str="2025-11-28T02:32:50.000000000",
    )
    # Constant-valued second frame: bilinear resampling must return the
    # constant wherever the warped frame has coverage.
    fill = np.full((ny, nx), 2.0 + 2.0j, dtype=np.complex64)
    p2 = gslc_factory(
        ny=ny, nx=nx, epsg=32611, dx=dx, dy=dy, data=fill,
        x0=round(x11), y0=round(y11),
        datetime_str="2025-11-28T02:33:05.000000000",
    )
    g1, g2 = GSLC(p1), GSLC(p2)
    s1 = GSLCStack.from_gslcs([g1])
    s2 = GSLCStack.from_gslcs([g2])

    merged = s1.merge(s2)

    # Result stays on self's CRS and lattice, with one paired acquisition.
    assert merged.epsg == 32610
    assert merged.sizes["time"] == 1
    assert np.allclose(np.diff(merged.x), dx)
    assert np.isin(s1.x, merged.x).all()
    assert np.isin(s1.y, merged.y).all()

    # Self's samples pass through untouched (exact coords, exact values).
    orig = s1.ds["slc"].isel(time=0).compute().values
    sub = merged.ds["slc"].isel(time=0).sel(x=s1.x, y=s1.y).compute().values
    np.testing.assert_array_equal(sub, orig)

    # The region east of frame 1 comes from the warped frame 2: valid pixels
    # exist and carry the constant fill.
    east = merged.ds["slc"].isel(time=0).sel(
        x=slice(s1.x.max() + 5 * dx, None)
    ).compute().values
    valid = east[~np.isnan(east)]
    assert valid.size > 0.3 * east.size
    np.testing.assert_allclose(valid, 2.0 + 2.0j, rtol=1e-5)
    g1.close()
    g2.close()


def test_merge_rejects_offset_lattice(gslc_factory):
    # Same CRS but grids offset by half a pixel: an outer join would silently
    # interleave near-duplicate coordinates, so merge must refuse.
    p1 = gslc_factory(ny=32, nx=32, x0=400000.0, seed=15)
    p2 = gslc_factory(ny=32, nx=32, x0=400000.0 + 5.0, seed=16)
    g1, g2 = GSLC(p1), GSLC(p2)
    s1 = GSLCStack.from_gslcs([g1])
    s2 = GSLCStack.from_gslcs([g2])
    with pytest.raises(ValueError, match="sub-pixel"):
        s1.merge(s2)
    g1.close()
    g2.close()


def test_merge_pairs_nearby_times(gslc_factory):
    # Adjacent frames on one pass differ by seconds; merge should pair them
    # and keep self's timestamps.
    p1 = gslc_factory(ny=32, nx=32, seed=10,
                      datetime_str="2025-11-28T02:32:50.000000000")
    p2 = gslc_factory(ny=32, nx=32, x0=400000.0 + 32 * 10.0, seed=11,
                      datetime_str="2025-11-28T02:33:02.000000000")
    g1, g2 = GSLC(p1), GSLC(p2)
    s1 = GSLCStack.from_gslcs([g1])
    s2 = GSLCStack.from_gslcs([g2])
    merged = s1.merge(s2)
    assert merged.sizes["time"] == 1
    assert merged.sizes["x"] == 64
    assert merged.ds["time"].values[0] == np.datetime64(
        "2025-11-28T02:32:50.000000000"
    )
    g1.close()
    g2.close()


def test_merge_rejects_unpairable_times(gslc_factory):
    # Different cycles (days apart) must not silently merge.
    p1 = gslc_factory(ny=32, nx=32, seed=12,
                      datetime_str="2025-11-28T02:32:50.000000000")
    p2 = gslc_factory(ny=32, nx=32, x0=400000.0 + 32 * 10.0, seed=13,
                      datetime_str="2025-12-10T02:32:50.000000000")
    g1, g2 = GSLC(p1), GSLC(p2)
    s1 = GSLCStack.from_gslcs([g1])
    s2 = GSLCStack.from_gslcs([g2])
    with pytest.raises(ValueError, match="No acquisition within"):
        s1.merge(s2)

    # Mismatched acquisition counts are rejected outright.
    p3 = gslc_factory(ny=32, nx=32, seed=14,
                      datetime_str="2025-12-10T02:32:50.000000000")
    g3 = GSLC(p3)
    with pytest.raises(ValueError, match="different numbers of acquisitions"):
        s1.merge(GSLCStack.from_gslcs([g1, g3]))
    g1.close()
    g2.close()
    g3.close()


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
