"""Tests for SNAPHU unwrapping, region writes, and per-pair resume."""

import warnings

import numpy as np
import pytest
import xarray as xr

from nisar_tools import GSLC, GSLCStack, Workspace
from nisar_tools import _kernels
from nisar_tools.interferogram import InterferogramStack
from nisar_tools.unwrap import UnwrappedStack, _snaphu_error_detail


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


def test_unwrap_carries_coherence_through(gslc_factory, tmp_path):
    """SNAPHU output keeps the interferogram's coherence, so mask_edges(
    min_coherence) works on a GSLC-derived stack and not only a GUNW."""
    igrams = _igram_stack(gslc_factory, tmp_path / "ws0")
    igrams = igrams.persist(Workspace(tmp_path / "ws0"), "igrams")
    ws = Workspace(tmp_path / "ws1")

    unw = igrams.unwrap(ws, nproc=1)
    assert "coherence" in unw.ds
    assert unw.ds["coherence"].dtype == np.float32
    # It is the interferogram's coherence, carried through unchanged.
    np.testing.assert_allclose(
        unw.ds["coherence"].values, igrams.ds["coherence"].values, equal_nan=True
    )

    # ...and it now drives mask_edges(min_coherence) on a SNAPHU stack.
    coh = unw.ds["coherence"].isel(pair=0).values
    thr = float(np.nanmedian(coh))
    out = unw.mask_edges(edge_pixels=0, min_coherence=thr).ds["unw"].isel(pair=0).values
    low = np.isfinite(coh) & (coh < thr)
    assert low.any() and np.all(np.isnan(out[low]))  # sub-threshold pixels nulled


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


# -- pair-level concurrency -------------------------------------------------

def test_unwrap_concurrent_pairs_match_the_serial_result(gslc_factory, tmp_path):
    """Pairs are independent, and each writes a Zarr region 1 deep along `pair`,
    so running them concurrently must be bit-identical to the serial loop."""
    igrams = _igram_stack(gslc_factory, tmp_path / "ws0")
    igrams = igrams.persist(Workspace(tmp_path / "ws0"), "igrams")
    npair = igrams.sizes["pair"]
    assert npair >= 3

    serial = igrams.unwrap(Workspace(tmp_path / "serial"), nproc=1)
    concurrent = igrams.unwrap(
        Workspace(tmp_path / "concurrent"), nproc=4, pairs_in_flight=3
    )

    for var in ("unw", "conncomp", "coherence"):
        np.testing.assert_array_equal(
            concurrent.ds[var].values, serial.ds[var].values
        )
    # Every marker survived the concurrent read-modify-write.
    assert Workspace(tmp_path / "concurrent").pairs_done("unwrapped") == set(
        range(npair)
    )


def test_unwrap_prefetches_the_next_pair(gslc_factory, tmp_path, monkeypatch):
    """At pairs_in_flight=1 the loads still overlap: pair i+1 must already be
    read by the time SNAPHU is asked for pair i+1."""
    import snaphu

    igrams = _igram_stack(gslc_factory, tmp_path / "ws0")
    igrams = igrams.persist(Workspace(tmp_path / "ws0"), "igrams")
    ws = Workspace(tmp_path / "ws1")

    events = []
    real_unwrap = snaphu.unwrap

    def tracking_unwrap(*args, **kwargs):
        events.append("unwrap")
        return real_unwrap(*args, **kwargs)

    monkeypatch.setattr("nisar_tools.unwrap.snaphu.unwrap", tracking_unwrap)

    real_isel = xr.DataArray.isel

    def tracking_isel(self, *args, **kwargs):
        if self.name == "igram":
            events.append("load")
        return real_isel(self, *args, **kwargs)

    monkeypatch.setattr(xr.DataArray, "isel", tracking_isel)
    igrams.unwrap(ws, nproc=1)
    monkeypatch.undo()

    # A strictly serial loop gives load,unwrap,load,unwrap,...; prefetching means
    # at least one load lands before the preceding unwrap has been requested.
    assert events.count("unwrap") == igrams.sizes["pair"]
    assert events[:2] == ["load", "load"]


def test_prefetch_bounds_its_window():
    """The lookahead is what keeps peak memory off the stack length."""
    from nisar_tools.unwrap import _prefetch

    started, live, peak = [], 0, 0
    import threading

    lock = threading.Lock()

    def load(i):
        nonlocal live, peak
        with lock:
            started.append(i)
            live += 1
            peak = max(peak, live)
        try:
            return i * 10
        finally:
            with lock:
                live -= 1

    got = list(_prefetch(load, range(20), lookahead=2))
    assert [item for item, _ in got] == list(range(20))
    assert [value for _, value in got] == [i * 10 for i in range(20)]
    # Never more than lookahead + 1 loads outstanding at once.
    assert peak <= 3
    assert sorted(started) == list(range(20))


def test_mark_pair_done_is_thread_safe(tmp_path):
    """An unsynchronised read-modify-write loses markers, and with them the
    per-pair resume."""
    from concurrent.futures import ThreadPoolExecutor

    ws = Workspace(tmp_path / "ws")
    ws._done_path("stage").write_text('{"pairs_done": []}')
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda i: ws.mark_pair_done("stage", i), range(200)))
    assert ws.pairs_done("stage") == set(range(200))


# -- SNAPHU tile sizing -----------------------------------------------------
#
# The rules these pin (see nisar_tools/_kernels.py):
#   * tile geometry depends on the raster, never on nproc;
#   * every tiling SNAPHU is handed satisfies its own CheckParams preconditions
#     AND stays under the per-tile region ceiling that made unwrap abort with
#     "Number of regions in tile exceeds max allowed" at small nproc.

SHAPES = [
    (11, 11), (64, 64), (300, 900), (522, 474), (903, 1112), (2000, 3000),
    (4000, 4000), (12000, 3000), (3000, 12000), (10000, 400), (2500, 40),
    (40000, 40000),
]


@pytest.mark.parametrize("shape", SHAPES)
def test_snaphu_params_is_always_legal(shape):
    """Whatever nproc asks for, SNAPHU can actually run it."""
    for nproc in (1, 2, 3, 4, 8, 10, 20, 40, 64):
        ntiles, overlap = _kernels.snaphu_params(shape, nproc)
        # Raises if SNAPHU would reject the tiling or choke on the tile size.
        _kernels.snaphu_params_check(shape, ntiles, overlap)


@pytest.mark.parametrize("shape", SHAPES)
def test_snaphu_params_is_independent_of_nproc(shape):
    """Tiling is part of the answer, so nproc must not move it.

    The old formula derived ntiles from nproc, which silently changed the
    unwrapped phase between runs while leaving the params hash untouched.
    """
    geometries = {_kernels.snaphu_params(shape, n) for n in range(1, 65)}
    assert len(geometries) == 1


@pytest.mark.parametrize("shape", SHAPES)
def test_snaphu_params_respects_the_tile_budget(shape):
    """The budget is measured on SNAPHU's own tile size, overlap included."""
    ntiles, overlap = _kernels.snaphu_params(shape)
    (ni, nj), _ = _kernels.snaphu_tile_shape(shape, ntiles, overlap)
    if ntiles != (1, 1):
        assert ni * nj <= _kernels.DEFAULT_MAX_TILE_PIXELS
    else:
        # A single tile has no region ceiling, and snaphu ignores the overlap --
        # so it is set to 0 to keep the "disregarding" line off stderr.
        assert overlap == 0


def test_snaphu_params_check_warns_about_the_region_ceiling():
    """The exact configuration that used to abort inside SNAPHU.

    A warning, not an error: the ceiling is on the region *count*, and
    ``tile_area / min_region_size`` is only the worst case -- a 600x332 tile with a
    32000 budget was measured forming 6566 regions and unwrapping fine. Refusing
    outright would block a coarse tiling that would have worked.
    """
    # 4000x4000 at the old nproc=2 mapping: one 4000x2128 tile, 8.5 M pixels.
    with pytest.warns(RuntimeWarning, match="Number of regions in tile exceeds"):
        _kernels.snaphu_params_check((4000, 4000), (1, 2), 256)
    # A finer grid on the same raster is silent.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _kernels.snaphu_params_check((4000, 4000), (4, 4), 250)


def test_snaphu_params_check_reports_checkparams_conditions():
    # ntilerow^2 > nlines
    with pytest.raises(ValueError, match=r"ntilerow\^2"):
        _kernels.snaphu_params_check((100, 4000), (11, 2), 4)
    # ntilecol + colovrlp > linelen -- the hole the old max(10, ...) floor left.
    with pytest.raises(ValueError, match="ntilecol \\+ colovrlp"):
        _kernels.snaphu_params_check((400, 11), (2, 2), 10)
    # A single tile is exempt: snaphu ignores the tiling entirely.
    _kernels.snaphu_params_check((11, 11), (1, 1), 10)


def test_snaphu_params_honours_explicit_overrides():
    assert _kernels.snaphu_params((4000, 4000), 8, ntiles=(3, 3))[0] == (3, 3)
    assert _kernels.snaphu_params((4000, 4000), 8, tile_overlap=400)[1] == 400
    # A tighter budget buys more tiles.
    coarse, _ = _kernels.snaphu_params((4000, 4000), max_tile_pixels=8_000_000)
    fine, _ = _kernels.snaphu_params((4000, 4000), max_tile_pixels=250_000)
    assert fine[0] * fine[1] > coarse[0] * coarse[1]


def test_snaphu_error_detail_strips_the_constant_overlap_warning():
    """SNAPHU writes warnings and errors to the same stream, and snaphu-py
    re-raises the whole buffer -- so the overlap warning always led the message
    and made every failure look like an overlap problem."""
    stderr = (
        "WARNING: Tile overlap is small (may give bad results)\n"
        "Number of regions in tile exceeds max allowed\n"
        "Abort\n"
    )
    detail = _snaphu_error_detail(stderr)
    assert "Tile overlap is small" not in detail
    assert detail.startswith("Number of regions in tile exceeds max allowed")
    # A message that is *only* noise still says something.
    assert _snaphu_error_detail(
        "WARNING: Tile overlap is small (may give bad results)"
    )


def test_unwrap_tiling_is_in_the_params_hash_but_nproc_is_not(
    gslc_factory, tmp_path
):
    igrams = _igram_stack(gslc_factory, tmp_path / "ws0")
    igrams = igrams.persist(Workspace(tmp_path / "ws0"), "igrams")
    ws = Workspace(tmp_path / "ws1")

    igrams.unwrap(ws, name="a", nproc=1)
    base = ws.stored_params_hash("a")

    # nproc no longer changes the result, so it must not invalidate the store.
    igrams.unwrap(ws, name="a", nproc=4)
    assert ws.stored_params_hash("a") == base

    # Tiling does change the result, so it must. (The test raster is 16x16, so
    # (2, 1) is the finest grid whose last tile still clears min_region_size.)
    igrams.unwrap(ws, name="b", nproc=1, ntiles=(2, 1))
    assert ws.stored_params_hash("b") != base


def test_unwrap_rejects_an_illegal_tiling_before_calling_snaphu(
    gslc_factory, tmp_path, monkeypatch
):
    import snaphu

    igrams = _igram_stack(gslc_factory, tmp_path / "ws0")
    igrams = igrams.persist(Workspace(tmp_path / "ws0"), "igrams")
    ws = Workspace(tmp_path / "ws1")

    def fail(*args, **kwargs):
        raise AssertionError("snaphu.unwrap should not have been reached")

    monkeypatch.setattr("nisar_tools.unwrap.snaphu.unwrap", fail)
    with pytest.raises(ValueError, match="SNAPHU tiling is invalid"):
        igrams.unwrap(ws, nproc=1, ntiles=(40, 40))


# -- 2*pi ambiguity ---------------------------------------------------------

def _cycle_stack(npair=2, ny=6, nx=6):
    """Two pairs, two connected components (top half = 1, bottom half = 2)."""
    unw = np.arange(npair * ny * nx, dtype=np.float32).reshape(npair, ny, nx)
    unw[0, 0, 0] = np.nan
    cc = np.zeros((npair, ny, nx), np.uint32)
    cc[:, : ny // 2, :] = 1
    cc[:, ny // 2 :, :] = 2
    ds = xr.Dataset(
        {"unw": (("pair", "y", "x"), unw.copy()),
         "conncomp": (("pair", "y", "x"), cc)},
        coords={"pair": np.arange(npair), "y": np.arange(float(ny)),
                "x": np.arange(float(nx))},
        attrs={"epsg": 32611},
    )
    return UnwrappedStack(ds), unw, cc


TWO_PI = 2.0 * np.pi


def test_add_cycles_shifts_whole_raster():
    stack, unw, _ = _cycle_stack()
    got = stack.add_cycles(1).ds["unw"].values
    finite = np.isfinite(unw)
    np.testing.assert_allclose(got[finite] - unw[finite], TWO_PI, rtol=1e-5)
    # The invalid footprint survives: NaN + anything is still NaN.
    assert np.isnan(got[0, 0, 0])
    assert got.dtype == np.float32


def test_add_cycles_is_signed_and_reversible():
    stack, unw, _ = _cycle_stack()
    back = stack.add_cycles(3).add_cycles(-3).ds["unw"].values
    finite = np.isfinite(unw)
    # float32 storage means "exact" is not on offer; the residue is ~1e-6 rad,
    # which at L-band is tens of nanometres of displacement.
    np.testing.assert_allclose(back[finite], unw[finite], atol=1e-5)


def test_add_cycles_selects_pairs_and_components():
    stack, unw, cc = _cycle_stack()

    only_pair0 = stack.add_cycles(2, pair=0).ds["unw"].values - unw
    np.testing.assert_allclose(np.nan_to_num(only_pair0[1]), 0.0, atol=1e-6)
    np.testing.assert_allclose(only_pair0[0][np.isfinite(only_pair0[0])],
                               2 * TWO_PI, rtol=1e-5)

    only_comp2 = stack.add_cycles(1, conncomp=2).ds["unw"].values - unw
    np.testing.assert_allclose(np.nan_to_num(only_comp2[cc == 1]), 0.0, atol=1e-6)
    np.testing.assert_allclose(only_comp2[cc == 2], TWO_PI, rtol=1e-5)

    # The two selectors intersect rather than union. NaN pixels stay NaN, so
    # they never register as shifted; exclude them from the "all touched" side.
    both = stack.add_cycles(1, pair=0, conncomp=1).ds["unw"].values - unw
    touched = ~np.isclose(np.nan_to_num(both), 0.0)
    valid = np.isfinite(unw)
    assert touched[0][(cc[0] == 1) & valid[0]].all()
    assert not touched[1].any()
    assert not touched[0][cc[0] == 2].any()


def test_add_cycles_rejects_fractional_shifts():
    stack, _, _ = _cycle_stack()
    for bad in (0.5, 1.25, -0.1):
        with pytest.raises(ValueError, match="whole number"):
            stack.add_cycles(bad)


def test_add_cycles_records_provenance_and_changes_the_hash(tmp_path):
    from nisar_tools import Workspace

    stack, _, _ = _cycle_stack()
    ws = Workspace(tmp_path / "ws")
    stack.persist(ws, "unw_plain")

    shifted = stack.add_cycles(2, pair=0)
    assert shifted.ds.attrs["cycle_shifts"] == [
        {"cycles": 2, "pair": [0], "conncomp": None}
    ]
    shifted.add_cycles(-1).persist(ws, "unw_shifted")
    assert len(
        UnwrappedStack.from_zarr(ws.path("unw_shifted")).ds.attrs["cycle_shifts"]
    ) == 2
    assert ws.stored_params_hash("unw_shifted") != ws.stored_params_hash("unw_plain")


def test_add_cycles_carries_into_los(gslc_factory):
    """One cycle is half a wavelength of range change."""
    from nisar_tools import geometry as G

    p = gslc_factory(ny=40, nx=32, dx=20.0, dy=20.0, write_geometry=True)
    g = GSLC(p)
    x, y, epsg = g.x_coords, g.y_coords, g.epsg
    g.close()
    ds = xr.Dataset(
        {"unw": (("pair", "y", "x"), np.zeros((1, len(y), len(x)), np.float32)),
         "conncomp": (("pair", "y", "x"), np.ones((1, len(y), len(x)), np.uint32))},
        coords={"pair": [0], "y": y, "x": x},
    ).rio.write_crs(f"EPSG:{epsg}")
    ds.attrs.update(epsg=epsg, direction="Descending")
    stack = UnwrappedStack(ds)

    base = stack.to_los(p, dem=None).ds["los"].values
    bumped = stack.add_cycles(1).to_los(p, dem=None).ds["los"].values
    # One cycle is 2*pi of phase, and d = -(lambda/4pi)*phase, so adding a cycle
    # moves the displacement half a wavelength *away* from the sensor.
    lam = G.radar_wavelength(p)
    np.testing.assert_allclose(bumped - base, -lam / 2.0, rtol=1e-5)
