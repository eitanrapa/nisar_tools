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


def _los_stack(gslc_factory):
    """A LOSStack on the synthetic geometry cube."""
    p = gslc_factory(ny=60, nx=48, dx=20.0, dy=20.0, write_geometry=True)
    unw_stack, _, _, _, _ = _synthetic_unwrapped(p)
    return unw_stack.to_los(p, dem=None)


def test_look_angle_is_smaller_than_incidence(gslc_factory):
    """Look angle is measured at the sensor, incidence at the target.

    Earth curvature makes the look (off-nadir) angle the smaller of the two,
    by ``sin(look) = Re/(Re+h) sin(incidence)``. They are distinct fields, so
    plotting one when you meant the other is a silent error.
    """
    los = _los_stack(gslc_factory)

    inc = los.ds["incidence_angle"].values
    look = los.ds["look_angle"].values
    assert np.isfinite(inc).all() and np.isfinite(look).all()
    assert (look < inc).all()

    # Recover the platform altitude the fixture used; it must be constant.
    ratio = np.sin(np.radians(look)) / np.sin(np.radians(inc))
    altitude = 6_371_000.0 * (1.0 / ratio - 1.0)
    np.testing.assert_allclose(altitude, 747_000.0, rtol=1e-3)


def test_look_angle_survives_persist(gslc_factory, tmp_path):
    los = _los_stack(gslc_factory)
    ws = Workspace(tmp_path / "ws_look")
    los.persist(ws, "los_look")
    back = LOSStack.from_zarr(ws.path("los_look"))
    np.testing.assert_array_equal(
        back.ds["look_angle"].values, los.ds["look_angle"].values
    )


def test_plot_look_angle_and_incidence_are_different_fields(gslc_factory):
    import matplotlib
    matplotlib.use("Agg")

    los = _los_stack(gslc_factory)
    fig_look, ax_look = los.plot_look_angle()
    fig_inc, ax_inc = los.plot_incidence()
    assert fig_look is not None and fig_inc is not None
    assert ax_look.get_title() != ax_inc.get_title()


def test_geometry_is_masked_to_the_data_footprint(gslc_factory):
    """The geometry cube covers the frame's rectangle, the radar does not.

    Interpolating the cube fills every pixel, so without masking the angle
    fields plot as a solid rectangle that does not match the swath -- and
    report an incidence angle for ground the pass never illuminated.
    """
    p = gslc_factory(ny=40, nx=32, dx=20.0, dy=20.0, write_geometry=True)
    unw_stack, unw, x, y, _ = _synthetic_unwrapped(p, npair=2)

    # Blank a corner of one pair only; the other pair still covers it.
    ds = unw_stack.ds.copy()
    blanked = ds["unw"].values.copy()
    blanked[0, :10, :8] = np.nan
    blanked[:, -6:, :] = np.nan          # this strip is missing from both pairs
    ds["unw"] = (("pair", "y", "x"), blanked)
    unw_stack = UnwrappedStack(ds)

    los = unw_stack.to_los(p, dem=None)
    any_pair = np.isfinite(los.ds["los"].values).any(axis=0)

    for name in ("incidence_angle", "look_angle", "los_east", "los_north",
                 "los_up", "height"):
        got = np.isfinite(los.ds[name].values)
        np.testing.assert_array_equal(got, any_pair, err_msg=name)

    # Union, not intersection: the corner blanked in one pair only is kept.
    assert np.isfinite(los.ds["look_angle"].values[:10, :8]).all()
    # The strip missing from both pairs is dropped.
    assert not np.isfinite(los.ds["look_angle"].values[-6:, :]).any()


def test_geometry_masking_can_be_disabled(gslc_factory):
    p = gslc_factory(ny=40, nx=32, dx=20.0, dy=20.0, write_geometry=True)
    unw_stack, _, _, _, _ = _synthetic_unwrapped(p, npair=1)
    ds = unw_stack.ds.copy()
    blanked = ds["unw"].values.copy()
    blanked[:, :10, :] = np.nan
    ds["unw"] = (("pair", "y", "x"), blanked)
    unw_stack = UnwrappedStack(ds)

    full = unw_stack.to_los(p, dem=None, mask_geometry=False)
    assert np.isfinite(full.ds["look_angle"].values).all()
    masked = unw_stack.to_los(p, dem=None)
    assert not np.isfinite(masked.ds["look_angle"].values[:10, :]).any()


def _two_frame_stack(gslc_factory):
    """A merged two-frame stack plus the granule path of each frame."""
    from nisar_tools import GSLCStack

    ny = nx = 64
    step = 20.0
    when = "2025-11-28T02:32:50.000000000"
    pa = gslc_factory(ny=ny, nx=nx, dx=step, dy=step, x0=400000.0,
                      y0=4_000_000.0, datetime_str=when, write_geometry=True)
    pb = gslc_factory(ny=ny, nx=nx, dx=step, dy=step, x0=400000.0 + step * nx,
                      y0=4_000_000.0 - step * (ny // 2), datetime_str=when,
                      write_geometry=True)
    ga, gb = GSLC(pa), GSLC(pb)
    merged = GSLCStack.from_gslcs([ga]).merge(GSLCStack.from_gslcs([gb]))
    slc = merged.ds["slc"].isel(time=0).compute().values   # before closing
    ga.close()
    gb.close()

    unw = np.where(np.isfinite(slc), 1.0, np.nan).astype(np.float32)[None]
    ds = xr.Dataset(
        {"unw": (("pair", "y", "x"), unw)},
        coords={"pair": [0], "y": merged.y, "x": merged.x},
    ).rio.write_crs(f"EPSG:{merged.epsg}")
    ds.attrs.update(epsg=merged.epsg, direction="Descending")
    return UnwrappedStack(ds), pa, pb


def test_merged_stack_needs_a_granule_per_frame(gslc_factory):
    """Each cube spans only its own frame.

    With one granule, half a merged stack comes back without geometry -- which
    is not obvious until the geometry is masked to the data footprint, at which
    point the angle plots show only one frame.
    """
    stack, pa, pb = _two_frame_stack(gslc_factory)
    data = np.isfinite(stack.ds["unw"].isel(pair=0).values)

    one = np.isfinite(stack.to_los(pa, dem=None).ds["look_angle"].values)
    both = np.isfinite(stack.to_los([pa, pb], dem=None).ds["look_angle"].values)

    # One granule covers its own frame only; both cover the whole footprint.
    assert 0.3 < one.sum() / data.sum() < 0.7
    np.testing.assert_array_equal(both, data)
    assert (one & ~data).sum() == 0          # never outside the data either


def test_granule_order_sets_precedence_in_the_overlap(gslc_factory):
    """Earlier granules win where cubes overlap, as merge does for data."""
    stack, pa, pb = _two_frame_stack(gslc_factory)
    ab = stack.to_los([pa, pb], dem=None)
    ba = stack.to_los([pb, pa], dem=None)

    assert ab.ds.attrs["granules"] == [str(pa), str(pb)]
    # Same coverage either way; the two orders differ only where cubes overlap.
    np.testing.assert_array_equal(
        np.isfinite(ab.ds["look_angle"].values),
        np.isfinite(ba.ds["look_angle"].values),
    )


def test_single_granule_path_still_works(gslc_factory):
    """A str/Path must stay a scalar -- iterating one would split it by character."""
    from pathlib import Path

    p = gslc_factory(ny=40, nx=32, dx=20.0, dy=20.0, write_geometry=True)
    stack, _, _, _, _ = _synthetic_unwrapped(p)
    for arg in (p, Path(p), [p]):
        los = stack.to_los(arg, dem=None)
        assert np.isfinite(los.ds["look_angle"].values).any()

    with pytest.raises(ValueError, match="at least one GSLC"):
        stack.to_los([], dem=None)
