"""Tests for reading NASA GUNW products into :class:`UnwrappedStack`.

A GUNW arrives already unwrapped, so nothing here runs SNAPHU. GUNW and SNAPHU
ingestion now produce the *same* class (``UnwrappedStack``); this suite pins the
GUNW read schema (``unw`` / ``coherence`` / ``conncomp`` / ``phase_screen`` on
``(pair, y, x)``, grid, CRS, provenance attrs), out-of-swath handling (NaN phase,
65535 conncomp fill -> 0), the self-contained ``to_los`` (geometry + wavelength
from the product's own cube, no GSLC), multi-file stacking + grid validation, and
a Zarr persist round-trip.
"""

import numpy as np
import pytest

from nisar_tools import LOSStack, UnwrappedStack, Workspace

WAVELENGTH = 299_792_458.0 / 1_239_000_000.0  # ~0.242 m, from CENTER_FREQ_A


def test_from_gunw_file_schema(gunw_factory):
    u = UnwrappedStack.from_gunw_file(gunw_factory(ny=48, nx=64))

    assert isinstance(u, UnwrappedStack)
    assert u.sizes == {"pair": 1, "y": 48, "x": 64}
    assert set(u.ds.data_vars) >= {"unw", "coherence", "conncomp", "phase_screen"}
    assert u.ds["conncomp"].dtype == np.uint32
    assert u.epsg == 32611
    assert u.direction == "Descending"
    assert u.ds.attrs["source"] == "gunw"
    assert u.ds.attrs["look_direction"] == "Left"
    assert u.ds.attrs["wavelength"] == pytest.approx(WAVELENGTH, rel=1e-6)
    # provenance kept for reload + to_los
    assert len(u.ds.attrs["source_files"]) == 1
    assert u.ds.attrs["pairs"] == [
        ["2025-11-28T02:32:16.000000000", "2025-12-10T02:32:16.000000000"]
    ]
    assert u.ds["ref_time"].values[0] == np.datetime64("2025-11-28T02:32:16")
    assert u.ds["sec_time"].values[0] == np.datetime64("2025-12-10T02:32:16")


def test_invalid_footprint_and_conncomp_fill(gunw_factory):
    u = UnwrappedStack.from_gunw_file(gunw_factory(ny=40, nx=40, nan_border=4))
    unw = u.ds["unw"].isel(pair=0).values
    cc = u.ds["conncomp"].isel(pair=0).values

    invalid = ~np.isfinite(unw)
    assert invalid.any() and np.isfinite(unw).any()
    # The 65535 product fill and every out-of-swath pixel become label 0.
    assert np.all(cc[invalid] == 0)
    assert np.all(cc[~invalid] == 1)
    assert cc.max() == 1  # 65535 did not leak through


def test_grid_matches_file_and_has_crs(gunw_factory):
    u = UnwrappedStack.from_gunw_file(
        gunw_factory(nx=32, ny=32, x0=400000.0, dx=80.0)
    )
    assert u.x[0] == 400000.0
    assert np.allclose(np.diff(u.x), 80.0)
    assert u.y[0] > u.y[-1]  # descending
    assert u.ds.rio.crs is not None
    assert u.ds.rio.crs.to_epsg() == 32611


def test_to_los_self_contained(gunw_factory):
    """to_los needs no GSLC: geometry + wavelength come from the GUNW itself."""
    u = UnwrappedStack.from_gunw_file(gunw_factory(ny=48, nx=64, nan_border=4))
    los = u.to_los()  # no granule argument

    assert isinstance(los, LOSStack)
    assert los.ds.attrs["wavelength"] == pytest.approx(WAVELENGTH, rel=1e-6)

    unw = u.ds["unw"].isel(pair=0).values
    d = los.ds["los"].isel(pair=0).values
    valid = np.isfinite(unw)
    # d = +(lambda / 4pi) * phase
    assert np.allclose(d[valid], WAVELENGTH / (4 * np.pi) * unw[valid], atol=1e-6)
    assert np.all(np.isnan(d[~valid]))

    inc = los.ds["incidence_angle"].values
    up = los.ds["los_up"].values
    fin = np.isfinite(inc)
    # key invariant: los_up == cos(incidence); look angle < incidence
    assert np.allclose(up[fin], np.cos(np.deg2rad(inc[fin])), atol=1e-4)
    look = los.ds["look_angle"].values
    assert np.nanmax(look - inc) < 0
    assert 29.9 <= np.nanmin(inc) and np.nanmax(inc) <= 45.1


def test_snaphu_to_los_still_requires_gslc(gunw_factory):
    """A stack tagged source=snaphu has no embedded cube, so to_los needs a gslc."""
    u = UnwrappedStack.from_gunw_file(gunw_factory())
    u.ds.attrs["source"] = "snaphu"
    u.ds.attrs.pop("source_files", None)
    with pytest.raises(ValueError, match="needs a gslc"):
        u.to_los()


def test_multi_file_stack(gunw_factory):
    p1 = gunw_factory(sec_time="2025-12-10T02:32:16.000000000")
    p2 = gunw_factory(sec_time="2025-12-22T02:32:16.000000000")
    u = UnwrappedStack.from_gunw_files([p1, p2])

    assert u.sizes["pair"] == 2
    assert list(u.ds["sec_time"].values) == [
        np.datetime64("2025-12-10T02:32:16"),
        np.datetime64("2025-12-22T02:32:16"),
    ]
    assert len(u.ds.attrs["source_files"]) == 2


def test_stack_rejects_mismatched_grid(gunw_factory):
    p1 = gunw_factory(nx=64, ny=64)
    p2 = gunw_factory(nx=48, ny=64)  # different grid
    with pytest.raises(ValueError, match="different grid"):
        UnwrappedStack.from_gunw_files([p1, p2])


def test_persist_roundtrip(gunw_factory, tmp_path):
    u = UnwrappedStack.from_gunw_file(gunw_factory())
    ws = Workspace(tmp_path / "ws")

    saved = u.persist(ws, "gunw")
    assert isinstance(saved, UnwrappedStack)

    reopened = UnwrappedStack.from_zarr(ws.path("gunw"))
    assert reopened.ds.rio.crs is not None            # CRS coord restored
    assert reopened.ds.attrs.get("source") == "gunw"  # provenance survived
    assert reopened.ds.attrs.get("source_files")
    # to_los still works after a reload (reads the cube from the source file)
    los = reopened.to_los()
    assert isinstance(los, LOSStack)
    assert los.sizes == reopened.sizes
