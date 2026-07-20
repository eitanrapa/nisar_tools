"""Tests for LOSStack: phase->displacement + look geometry from the GSLC cube."""

import numpy as np
import pytest
import xarray as xr

from nisar_tools import GSLC, LOSStack, Workspace
from nisar_tools import geometry as G
from nisar_tools.unwrap import UnwrappedStack


def _synthetic_unwrapped(gslc_path, npair=2, seed=0):
    """A minimal UnwrappedStack on the granule's grid (no SNAPHU needed)."""
    g = GSLC(gslc_path)
    x, y, epsg, direction = g.x_coords, g.y_coords, g.epsg, g.direction
    g.close()
    rng = np.random.default_rng(seed)
    unw = (rng.standard_normal((npair, len(y), len(x))) * 3).astype(np.float32)
    ds = xr.Dataset(
        {"unw": (("pair", "y", "x"), unw)},
        coords={"pair": np.arange(npair), "y": y, "x": x},
    ).rio.write_crs(f"EPSG:{epsg}")
    ds.attrs.update(epsg=epsg, direction=direction)
    return UnwrappedStack(ds), unw, x, y, epsg


def _expected_incidence(x, dx):
    # Mirrors the synthetic cube: incidence 30->45 deg linearly across the cube
    # x-extent, which spans [x.min()-dx, x.max()+dx].
    lo, hi = x.min() - dx, x.max() + dx
    return 30.0 + 15.0 * (x - lo) / (hi - lo)


def test_to_los_displacement_and_geometry(gslc_factory):
    p = gslc_factory(ny=60, nx=48, dx=20.0, dy=20.0, write_geometry=True)
    unw_stack, unw, x, y, epsg = _synthetic_unwrapped(p)

    los = unw_stack.to_los(p, dem=None)
    assert isinstance(los, LOSStack)

    # Displacement: d = +(lambda / 4pi) * phase (positive toward sensor).
    lam = G.radar_wavelength(p)
    np.testing.assert_allclose(
        los.ds["los"].values, lam / (4 * np.pi) * unw, rtol=1e-5, atol=1e-7
    )

    # Geometry is 2D (shared across pairs) and matches the analytic cube.
    assert los.ds["incidence_angle"].dims == ("y", "x")
    exp_inc = _expected_incidence(np.asarray(x), 20.0)
    got_inc = los.ds["incidence_angle"].values
    np.testing.assert_allclose(got_inc, np.broadcast_to(exp_inc, got_inc.shape),
                               atol=5e-3)

    inc = np.deg2rad(got_inc)
    np.testing.assert_allclose(los.ds["los_up"].values, np.cos(inc), atol=1e-4)
    np.testing.assert_allclose(los.ds["los_east"].values, -np.sin(inc), atol=1e-4)
    np.testing.assert_allclose(los.ds["los_north"].values, 0.0, atol=1e-5)

    assert los.epsg == epsg
    assert los.ds.attrs["wavelength"] == pytest.approx(0.24196, abs=1e-4)
    assert los.ds.attrs["look_direction"] == "Left"
    assert los.ds.attrs["sign"] == 1


def test_to_los_sign_flip(gslc_factory):
    p = gslc_factory(ny=40, nx=40, write_geometry=True)
    unw_stack, unw, *_ = _synthetic_unwrapped(p)
    pos = unw_stack.to_los(p, sign=1).ds["los"].values
    neg = unw_stack.to_los(p, sign=-1).ds["los"].values
    np.testing.assert_allclose(neg, -pos, rtol=1e-6)


def test_to_los_wavelength_override(gslc_factory):
    p = gslc_factory(ny=32, nx=32, write_geometry=True)
    unw_stack, unw, *_ = _synthetic_unwrapped(p)
    los = unw_stack.to_los(p, wavelength=0.1)
    np.testing.assert_allclose(los.ds["los"].values, 0.1 / (4 * np.pi) * unw,
                               rtol=1e-5, atol=1e-7)


def test_to_los_with_dem_sets_height(gslc_factory):
    p = gslc_factory(ny=48, nx=48, dx=20.0, dy=20.0, write_geometry=True)
    unw_stack, unw, x, y, epsg = _synthetic_unwrapped(p)

    dem = xr.DataArray(
        np.full((len(y), len(x)), 600.0, np.float32),
        coords={"y": np.asarray(y), "x": np.asarray(x)}, dims=("y", "x"),
    ).rio.write_crs(f"EPSG:{epsg}")

    los = unw_stack.to_los(p, dem=dem)
    # The cube geometry is height-independent, so incidence is unchanged, but the
    # sampled DEM height must be carried through.
    np.testing.assert_allclose(los.ds["height"].values, 600.0, atol=1e-2)


def test_los_persist_roundtrip(gslc_factory, tmp_path):
    p = gslc_factory(ny=40, nx=40, write_geometry=True)
    unw_stack, *_ = _synthetic_unwrapped(p)
    ws = Workspace(tmp_path / "ws")

    los = unw_stack.to_los(p).persist(ws, "los")
    assert ws.exists("los")
    reopened = LOSStack.from_zarr(ws.path("los"))
    assert set(reopened.ds.data_vars) >= {
        "los", "incidence_angle", "los_east", "los_north", "los_up", "height"
    }
    assert reopened.ds["los"].dims == ("pair", "y", "x")
    assert reopened.ds["incidence_angle"].dims == ("y", "x")


def test_full_chain_form_unwrap_to_los(gslc_factory, tmp_path):
    # Exercise the real stage chain: SLCs -> igrams -> unwrap -> LOS.
    gslcs = []
    for k in range(2):
        gp = gslc_factory(ny=80, nx=80, seed=k, write_geometry=True,
                          datetime_str=f"2025-11-{10 + k:02d}T00:00:00.000000000")
        gslcs.append(gp)
    from nisar_tools import GSLCStack
    ws = Workspace(tmp_path / "ws")
    stack = GSLCStack.from_gslcs([GSLC(g) for g in gslcs]).persist(ws, "slc")
    igrams = stack.form_interferograms(pairs="sequential", looks=5, downsample=True)
    unw = igrams.unwrap(ws, nproc=1)

    los = unw.to_los(gslcs[0])   # reference granule supplies cube + lambda
    assert isinstance(los, LOSStack)
    assert los.sizes["pair"] == unw.sizes["pair"]
    assert los.ds["los"].dtype == np.float32
    assert np.isfinite(los.ds["incidence_angle"].values).all()
