"""Tests for SNAPHU unwrapping, region writes, and per-pair resume."""

import numpy as np
import pytest
import xarray as xr

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
    lam = G.radar_wavelength(p)
    np.testing.assert_allclose(bumped - base, lam / 2.0, rtol=1e-5)
