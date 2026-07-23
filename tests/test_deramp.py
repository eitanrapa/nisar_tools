"""Tests for the unwrapped-phase cleaning pipeline and its kernels.

Covers the scipy port of ``filt_gunw.csh`` / ``h52grd.py``: edge masking,
tension-spline residual outlier rejection, polynomial/spline deramping, and
phase-screen (ionosphere) subtraction -- all as lazy ``UnwrappedStack`` methods,
plus the underlying numpy kernels.
"""

import numpy as np
import pytest

from nisar_tools import UnwrappedStack, Workspace
from nisar_tools._kernels import (
    deramp,
    fit_surface,
    mask_edges,
    poly_surface,
    remove_outliers,
    smooth_surface,
)


# -- kernels ---------------------------------------------------------------
def test_poly_surface_fits_plane_exactly():
    ny, nx = 20, 30
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    truth = 3.0 + 0.5 * xx - 0.2 * yy
    field = truth.copy()
    field[5, 5] = np.nan  # a hole must not change an exact plane fit
    assert np.allclose(poly_surface(field, degree=1), truth, atol=1e-6)


def test_smooth_surface_is_nan_aware():
    field = np.ones((30, 30), float)
    field[10:20, 10:20] = np.nan  # a NaN block must not drag the surface to 0
    s = smooth_surface(field, scale=3.0)
    assert np.nanmax(np.abs(s[np.isfinite(s)] - 1.0)) < 1e-6


def test_mask_edges_kernel_erodes_border():
    field = np.full((20, 20), 1.0, np.float32)
    out = mask_edges(field, edge_pixels=3)
    assert np.all(np.isnan(out[:3])) and np.all(np.isnan(out[-3:]))
    assert np.all(np.isfinite(out[3:-3, 3:-3]))


def test_remove_outliers_kernel_nulls_spike():
    yy, xx = np.mgrid[0:40, 0:40].astype(np.float32)
    field = 0.01 * xx  # gentle ramp
    field[20, 20] = 50.0  # one gross outlier
    out = remove_outliers(field, scale=5.0, threshold=1.0, iterations=3)
    assert np.isnan(out[20, 20])
    assert np.isfinite(out[5, 5])


def test_deramp_kernel_flattens_and_validates_method():
    yy, xx = np.mgrid[0:30, 0:40].astype(np.float64)
    field = 2.0 + 0.3 * xx - 0.1 * yy
    flat = deramp(field, degree=1, method="poly")
    assert np.nanstd(flat) < 1e-6
    with pytest.raises(ValueError, match="method must be"):
        deramp(field, method="nope")


def test_deramp_is_field_minus_fit_surface():
    """deramp subtracts exactly what fit_surface (the phase screen) returns."""
    rng = np.random.default_rng(0)
    field = np.cumsum(rng.normal(size=(40, 40)), axis=1)  # smooth-ish
    for kw in ({"method": "poly", "degree": 2}, {"method": "spline", "scale": 5.0}):
        assert np.allclose(deramp(field, **kw), field - fit_surface(field, **kw),
                           equal_nan=True)


def test_poly_surface_weight_uniform_equals_unweighted():
    rng = np.random.default_rng(1)
    field = rng.normal(size=(20, 25))
    w = np.full_like(field, 3.0)  # uniform weights -> same as OLS
    assert np.allclose(poly_surface(field, 2, weight=w), poly_surface(field, 2))


def test_smooth_surface_weight_zero_drops_pixel():
    field = np.ones((21, 21))
    field[10, 10] = 100.0            # a spike
    w = np.ones_like(field)
    w[10, 10] = 0.0                  # zero-weight it out
    s = smooth_surface(field, scale=3.0, weight=w)
    assert abs(s[10, 10] - 1.0) < 1e-6  # neighbours are 1.0, spike ignored


# -- UnwrappedStack methods (on ingested GUNW) -----------------------------
def test_remove_phase_screen_subtracts(gunw_factory):
    u = UnwrappedStack.from_gunw_file(gunw_factory(ny=40, nx=40, nan_border=2,
                                                   iono_amp=0.7))
    assert "phase_screen" in u.ds
    unw0 = u.ds["unw"].isel(pair=0).values
    screen = u.ds["phase_screen"].isel(pair=0).values

    out = u.remove_phase_screen()
    unw1 = out.ds["unw"].isel(pair=0).values
    valid = np.isfinite(unw0) & np.isfinite(screen)
    assert np.allclose(unw1[valid], (unw0 - screen)[valid], atol=1e-5)
    assert out.ds.attrs["phase_screen_removed"] == ["phase_screen"]

    with pytest.raises(ValueError, match="already been removed"):
        out.remove_phase_screen()


def test_remove_phase_screen_requires_layer(gunw_factory):
    u = UnwrappedStack.from_gunw_file(gunw_factory())
    u.ds = u.ds.drop_vars("phase_screen")
    with pytest.raises(ValueError, match="carries no"):
        u.remove_phase_screen()


def _snaphu_like(gunw_factory, **kw):
    """A GUNW read then stripped of its NASA screen, standing in for a SNAPHU
    stack that carries no phase_screen until one is estimated."""
    u = UnwrappedStack.from_gunw_file(gunw_factory(**kw))
    u.ds = u.ds.drop_vars("phase_screen")
    u.ds.attrs.pop("phase_screen_method", None)
    return u


def test_estimate_phase_screen_creates_layer(gunw_factory):
    u = _snaphu_like(gunw_factory, ny=40, nx=48, nan_border=2)
    assert "phase_screen" not in u.ds

    out = u.estimate_phase_screen(method="spline", scale=5.0)
    assert "phase_screen" in out.ds
    screen = out.ds["phase_screen"].isel(pair=0).values
    unw = out.ds["unw"].isel(pair=0).values
    # format parity with a GUNW screen: float32, radians grid, unw footprint
    assert out.ds["phase_screen"].dtype == np.float32
    assert np.array_equal(np.isnan(screen), np.isnan(unw))
    meth = out.ds.attrs["phase_screen_method"]
    assert meth["method"] == "spline" and meth["name"] == "phase_screen"

    # ...and it feeds remove_phase_screen just like NASA's
    corrected = out.remove_phase_screen()
    assert corrected.ds.attrs["phase_screen_removed"] == ["phase_screen"]


def test_estimate_then_remove_equals_deramp(gunw_factory):
    """estimate_phase_screen(spline) + remove == deramp(spline), same machinery."""
    u = _snaphu_like(gunw_factory, ny=40, nx=48, nan_border=2)
    est = u.estimate_phase_screen(method="spline", scale=6.0, weighted=False)
    via_screen = est.remove_phase_screen().ds["unw"].isel(pair=0).values
    via_deramp = u.deramp(method="spline", scale=6.0).ds["unw"].isel(pair=0).values
    valid = np.isfinite(via_deramp)
    assert np.allclose(via_screen[valid], via_deramp[valid], atol=1e-5)


def test_estimate_uses_coherence_weight(gunw_factory):
    u = _snaphu_like(gunw_factory, ny=32, nx=32, nan_border=2)
    # weighted defaults on when coherence is present
    out = u.estimate_phase_screen(method="spline", scale=5.0)
    assert out.ds.attrs["phase_screen_method"]["weighted"] is True
    # ...and off when coherence is absent
    u.ds = u.ds.drop_vars("coherence")
    out2 = u.estimate_phase_screen(method="spline", scale=5.0)
    assert out2.ds.attrs["phase_screen_method"]["weighted"] is False


def test_estimate_refuses_to_clobber_nasa_screen(gunw_factory):
    u = UnwrappedStack.from_gunw_file(gunw_factory())  # has NASA phase_screen
    with pytest.raises(ValueError, match="already exists"):
        u.estimate_phase_screen()

    # a distinct name keeps both NASA + spline for comparison
    out = u.estimate_phase_screen(name="phase_screen_spline", scale=5.0)
    assert "phase_screen" in out.ds and "phase_screen_spline" in out.ds

    # each can be removed independently, tracked by name
    two = (out.remove_phase_screen("phase_screen")
              .remove_phase_screen("phase_screen_spline"))
    assert two.ds.attrs["phase_screen_removed"] == [
        "phase_screen", "phase_screen_spline"
    ]


def test_nasa_screen_is_labelled(gunw_factory):
    u = UnwrappedStack.from_gunw_file(gunw_factory())
    assert u.ds.attrs["phase_screen_method"]["method"] == "nasa_split_spectrum"


def test_mask_edges_trims_footprint(gunw_factory):
    u = UnwrappedStack.from_gunw_file(gunw_factory(ny=40, nx=40, nan_border=2))
    before = np.isfinite(u.ds["unw"].isel(pair=0).values)
    out = u.mask_edges(edge_pixels=3)
    after = np.isfinite(out.ds["unw"].isel(pair=0).values)

    assert after.sum() < before.sum()
    row = after.shape[0] // 2  # centre row loses 3 px on each side
    assert before[row].sum() - after[row].sum() == 6
    assert out.ds.attrs["edges_masked"]["edge_pixels"] == 3


def test_mask_edges_min_coherence_needs_coherence(gunw_factory):
    u = UnwrappedStack.from_gunw_file(gunw_factory())
    u.ds = u.ds.drop_vars("coherence")
    with pytest.raises(ValueError, match="coherence"):
        u.mask_edges(min_coherence=0.5)


def test_mask_edges_builtin_mask_nulls_invalid_subswath(gunw_factory):
    # An interior stripe is finite in unw but flagged out-of-subswath by the
    # product mask: erosion alone keeps it, the built-in mask removes it.
    u = UnwrappedStack.from_gunw_file(
        gunw_factory(ny=40, nx=40, nan_border=2, mask_invalid_cols=3)
    )
    assert "subswath_mask" in u.ds
    unw = u.ds["unw"].isel(pair=0).values
    stripe = np.zeros_like(unw, bool)
    stripe[:, 20:23] = True
    assert np.isfinite(unw[stripe]).any()  # the stripe is real data in unw

    ero = u.mask_edges(edge_pixels=0).ds["unw"].isel(pair=0).values
    assert np.isfinite(ero[stripe]).any()  # erosion-only leaves it

    out = u.mask_edges(edge_pixels=0, use_builtin_mask=True)
    got = out.ds["unw"].isel(pair=0).values
    assert np.all(np.isnan(got[stripe]))                      # mask removes it
    assert np.isfinite(got[np.isfinite(unw) & ~stripe]).all()  # valid land kept
    assert out.ds.attrs["edges_masked"]["use_builtin_mask"] is True


def test_mask_edges_builtin_mask_requires_layer(gunw_factory):
    u = UnwrappedStack.from_gunw_file(gunw_factory())
    u.ds = u.ds.drop_vars("subswath_mask")
    with pytest.raises(ValueError, match="subswath"):
        u.mask_edges(use_builtin_mask=True)


def test_remove_outliers_nulls_spikes(gunw_factory):
    u = UnwrappedStack.from_gunw_file(
        gunw_factory(ny=48, nx=48, nan_border=2, spikes=8, seed=1)
    )
    unw = u.ds["unw"].isel(pair=0).values
    spikes = unw > 20.0  # injected +50 outliers tower over the [-3, 3] ramp
    assert spikes.sum() >= 1

    cleaned = u.remove_outliers(scale=6.0, threshold=2.0, iterations=3)
    out = cleaned.ds["unw"].isel(pair=0).values
    assert np.all(np.isnan(out[spikes]))                       # spikes gone
    interior = np.isfinite(unw) & ~spikes
    assert np.isfinite(out[interior]).mean() > 0.9             # ramp survives
    assert cleaned.ds.attrs["outliers_removed"]["threshold"] == 2.0


def test_deramp_poly_flattens_ramp(gunw_factory):
    # unwrappedPhase is a linear ramp in x; a degree-1 deramp flattens it.
    u = UnwrappedStack.from_gunw_file(gunw_factory(ny=48, nx=48, nan_border=2))
    before = u.ds["unw"].isel(pair=0).values
    out = u.deramp(degree=1).ds["unw"].isel(pair=0).values
    valid = np.isfinite(before)
    assert np.nanstd(before[valid]) > 1.0     # was a big ramp
    assert np.nanstd(out[valid]) < 0.05       # now flat


def test_deramp_enters_persist_hash(gunw_factory, tmp_path):
    """Applying deramp changes the persist identity (same stage name, two stores)."""
    u = UnwrappedStack.from_gunw_file(gunw_factory())
    ws_a = Workspace(tmp_path / "a")
    ws_b = Workspace(tmp_path / "b")

    u.persist(ws_a, "s")
    u.deramp(degree=2).persist(ws_b, "s")
    assert ws_a.stored_params_hash("s") != ws_b.stored_params_hash("s")


def test_pipeline_chains_and_preserves_type(gunw_factory):
    u = UnwrappedStack.from_gunw_file(gunw_factory(ny=48, nx=48, nan_border=2,
                                                  spikes=5, seed=2))
    out = (u.remove_phase_screen()
             .mask_edges(edge_pixels=2)
             .remove_outliers(threshold=2.0)
             .deramp(degree=1))
    assert isinstance(out, UnwrappedStack)
    # every step recorded its provenance
    for key in ("phase_screen_removed", "edges_masked", "outliers_removed", "deramp"):
        assert key in out.ds.attrs
    # still convertible to LOS from its own cube
    los = out.to_los()
    assert los.sizes == out.sizes
