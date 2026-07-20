"""Tests for the Goldstein-Werner phase filter (kernel + stack integration).

The kernel has no legacy oracle, so it is pinned by its defining properties:
an exact ``alpha=0`` round-trip, measurable phase-noise reduction, NaN
preservation, and no mixing across the stacked ``pair`` axis. The stack method
is then checked to be exactly the kernel applied per pair, with coherence left
untouched.
"""

import numpy as np
import pytest

from nisar_tools import GSLC, GSLCStack, Workspace
from nisar_tools import _kernels
from nisar_tools.interferogram import InterferogramStack
from nisar_tools.unwrap import UnwrappedStack
from nisar_tools.workspace import WorkspaceError


def _phase_resid_std(field, true_phase, edge=8):
    """Circular std of ``angle(field) - true_phase`` over the interior."""
    d = np.angle(np.exp(1j * (np.angle(field) - true_phase)))
    return np.std(d[edge:-edge, edge:-edge])


def _noisy_ramp(ny=64, nx=64, noise=0.8, seed=3):
    """A smooth phase ramp (unit amplitude) plus complex noise, and its truth."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:ny, 0:nx]
    true = 0.15 * xx + 0.10 * yy
    clean = np.exp(1j * true)
    n = rng.standard_normal((ny, nx)) + 1j * rng.standard_normal((ny, nx))
    return (clean + noise * n).astype(np.complex64), true


# -- kernel properties -----------------------------------------------------
def test_goldstein_alpha_zero_is_identity():
    rng = np.random.default_rng(0)
    z = (rng.standard_normal((64, 64)) + 1j * rng.standard_normal((64, 64))).astype(
        np.complex64
    )
    out = _kernels.goldstein_filter(z, alpha=0.0)
    np.testing.assert_allclose(out, z, rtol=1e-4, atol=1e-4)
    assert out.dtype == z.dtype
    assert out.shape == z.shape


@pytest.mark.parametrize("alpha", [0.5, 0.8, 1.0])
def test_goldstein_reduces_phase_noise(alpha):
    noisy, true = _noisy_ramp()
    filt = _kernels.goldstein_filter(noisy, alpha=alpha)
    # Filtering must move the phase closer to the underlying ramp.
    assert _phase_resid_std(filt, true) < _phase_resid_std(noisy, true)


def test_goldstein_stronger_alpha_smooths_more():
    noisy, true = _noisy_ramp()
    weak = _phase_resid_std(_kernels.goldstein_filter(noisy, alpha=0.3), true)
    strong = _phase_resid_std(_kernels.goldstein_filter(noisy, alpha=1.0), true)
    assert strong < weak


def test_goldstein_preserves_nan_and_shape():
    rng = np.random.default_rng(1)
    z = (rng.standard_normal((48, 40)) + 1j * rng.standard_normal((48, 40))).astype(
        np.complex64
    )
    z[10:20, 12:22] = np.nan
    out = _kernels.goldstein_filter(z, alpha=0.7)
    nan_in = np.isnan(z)
    # NaN exactly where the input was NaN; every other pixel finite (no leakage).
    np.testing.assert_array_equal(np.isnan(out), nan_in)
    assert np.isfinite(out[~nan_in]).all()
    assert out.shape == z.shape and out.dtype == z.dtype


def test_goldstein_clips_patch_to_small_raster():
    # patch_size larger than the raster must not raise and must stay finite.
    rng = np.random.default_rng(2)
    z = (rng.standard_normal((18, 12)) + 1j * rng.standard_normal((18, 12))).astype(
        np.complex64
    )
    out = _kernels.goldstein_filter(z, alpha=0.6, patch_size=32)
    assert out.shape == z.shape
    assert np.isfinite(out).all()


def test_goldstein_planes_matches_per_plane_no_mixing():
    rng = np.random.default_rng(4)
    stack = (
        rng.standard_normal((3, 40, 32)) + 1j * rng.standard_normal((3, 40, 32))
    ).astype(np.complex64)
    stack[1, 5:10, 5:10] = np.nan  # a NaN patch on one plane only

    out = _kernels.goldstein_filter_planes(stack, alpha=0.6)
    for k in range(stack.shape[0]):
        ref = _kernels.goldstein_filter(stack[k], alpha=0.6)
        np.testing.assert_array_equal(np.isnan(out[k]), np.isnan(ref))
        m = ~np.isnan(ref)
        np.testing.assert_allclose(out[k][m], ref[m], rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "kwargs",
    [dict(alpha=1.5), dict(alpha=-0.1), dict(overlap=1.0), dict(overlap=-0.1),
     dict(psd_smooth=0)],
)
def test_goldstein_rejects_bad_params(kwargs):
    z = np.ones((16, 16), dtype=np.complex64)
    with pytest.raises(ValueError):
        _kernels.goldstein_filter(z, **kwargs)


# -- Baran coherence-adaptive alpha (GMTSAR phasefilt with -amp1/-amp2) ------
@pytest.mark.parametrize("coh", [0.0, 0.3, 0.7, 1.0])
def test_goldstein_adaptive_matches_constant_for_uniform_coherence(coh):
    # Uniform coherence c must reproduce the constant-alpha result at alpha=1-c.
    rng = np.random.default_rng(5)
    z = (rng.standard_normal((64, 64)) + 1j * rng.standard_normal((64, 64))).astype(
        np.complex64
    )
    coherence = np.full((64, 64), coh, np.float32)
    adaptive = _kernels.goldstein_filter(z, alpha="adaptive", coherence=coherence)
    constant = _kernels.goldstein_filter(z, alpha=1.0 - coh)
    np.testing.assert_allclose(adaptive, constant, rtol=1e-5, atol=1e-5)


def test_goldstein_adaptive_full_coherence_is_identity():
    rng = np.random.default_rng(6)
    z = (rng.standard_normal((48, 48)) + 1j * rng.standard_normal((48, 48))).astype(
        np.complex64
    )
    out = _kernels.goldstein_filter(
        z, alpha="adaptive", coherence=np.ones((48, 48), np.float32)
    )
    np.testing.assert_allclose(out, z, rtol=1e-4, atol=1e-4)


def test_goldstein_adaptive_filters_low_coherence_more():
    rng = np.random.default_rng(7)
    z = (rng.standard_normal((64, 64)) + 1j * rng.standard_normal((64, 64))).astype(
        np.complex64
    )
    lo = _kernels.goldstein_filter(z, alpha="adaptive",
                                   coherence=np.full((64, 64), 0.1, np.float32))
    hi = _kernels.goldstein_filter(z, alpha="adaptive",
                                   coherence=np.full((64, 64), 0.9, np.float32))
    # Lower coherence -> stronger filtering -> larger departure from the input.
    assert np.mean(np.abs(lo - z)) > np.mean(np.abs(hi - z))


def test_goldstein_adaptive_requires_coherence():
    z = np.ones((16, 16), dtype=np.complex64)
    with pytest.raises(ValueError, match="requires a coherence"):
        _kernels.goldstein_filter(z, alpha="adaptive")


def test_goldstein_rejects_bad_alpha_string():
    z = np.ones((16, 16), dtype=np.complex64)
    with pytest.raises(ValueError, match="float in"):
        _kernels.goldstein_filter(z, alpha="strong")


def test_goldstein_adaptive_coherence_shape_mismatch():
    z = np.ones((32, 32), dtype=np.complex64)
    with pytest.raises(ValueError, match="same shape"):
        _kernels.goldstein_filter(
            z, alpha="adaptive", coherence=np.ones((16, 16), np.float32)
        )


def test_goldstein_planes_adaptive_matches_per_plane():
    rng = np.random.default_rng(8)
    stack = (
        rng.standard_normal((3, 40, 32)) + 1j * rng.standard_normal((3, 40, 32))
    ).astype(np.complex64)
    coh = np.stack([np.full((40, 32), c, np.float32) for c in (0.2, 0.5, 0.9)], 0)
    out = _kernels.goldstein_filter_planes(stack, coh, alpha="adaptive")
    for k in range(3):
        ref = _kernels.goldstein_filter(stack[k], alpha="adaptive", coherence=coh[k])
        np.testing.assert_allclose(out[k], ref, rtol=1e-6, atol=1e-6)


# -- stack integration -----------------------------------------------------
def _two_gslcs(gslc_factory, ny=120, nx=100):
    p1 = gslc_factory(ny=ny, nx=nx, seed=0,
                      datetime_str="2025-11-28T02:32:50.000000000")
    p2 = gslc_factory(ny=ny, nx=nx, seed=1,
                      datetime_str="2025-12-10T02:32:50.000000000")
    return GSLC(p1), GSLC(p2)


def test_filter_goldstein_matches_kernel_and_keeps_coherence(gslc_factory):
    g1, g2 = _two_gslcs(gslc_factory)
    igrams = GSLCStack.from_gslcs([g1, g2]).form_interferograms(looks=5)

    raw = igrams.ds["igram"].isel(pair=0).compute().values
    coh_before = igrams.ds["coherence"].compute().values

    filtered = igrams.filter_goldstein(alpha=0.6, patch_size=16)
    assert isinstance(filtered, InterferogramStack)

    got = filtered.ds["igram"].isel(pair=0).compute().values
    expected = _kernels.goldstein_filter(raw, alpha=0.6, patch_size=16)
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)

    # Coherence is a separate quality measure and must pass through untouched.
    np.testing.assert_array_equal(
        filtered.ds["coherence"].compute().values, coh_before
    )
    assert filtered.ds.attrs["goldstein"]["alpha"] == 0.6
    g1.close()
    g2.close()


def test_filter_goldstein_adaptive_matches_kernel(gslc_factory):
    g1, g2 = _two_gslcs(gslc_factory)
    igrams = GSLCStack.from_gslcs([g1, g2]).form_interferograms(looks=5)

    raw = igrams.ds["igram"].isel(pair=0).compute().values
    coh = igrams.ds["coherence"].isel(pair=0).compute().values
    coh_all = igrams.ds["coherence"].compute().values

    filtered = igrams.filter_goldstein(alpha="adaptive", patch_size=16)
    got = filtered.ds["igram"].isel(pair=0).compute().values
    expected = _kernels.goldstein_filter(
        raw, alpha="adaptive", patch_size=16, coherence=coh
    )
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)

    # Coherence passes through untouched; the adaptive mode is recorded.
    np.testing.assert_array_equal(filtered.ds["coherence"].compute().values, coh_all)
    assert filtered.ds.attrs["goldstein"]["alpha"] == "adaptive"
    g1.close()
    g2.close()


def test_filter_goldstein_adaptive_persist_roundtrip(gslc_factory, tmp_path):
    g1, g2 = _two_gslcs(gslc_factory, ny=60, nx=60)
    igrams = GSLCStack.from_gslcs([g1, g2]).form_interferograms(looks=5)
    ws = Workspace(tmp_path / "ws")

    igrams.filter_goldstein(alpha="adaptive").persist(ws, "igrams_filt")
    reopened = InterferogramStack.from_zarr(ws.path("igrams_filt"))
    assert reopened.ds.attrs["goldstein"]["alpha"] == "adaptive"
    # A different mode (constant) must not be mistaken for the same stage.
    with pytest.raises(WorkspaceError):
        igrams.filter_goldstein(alpha=0.5).persist(ws, "igrams_filt")
    g1.close()
    g2.close()


def test_filter_goldstein_alpha_zero_is_identity_on_stack(gslc_factory):
    g1, g2 = _two_gslcs(gslc_factory)
    igrams = GSLCStack.from_gslcs([g1, g2]).form_interferograms(looks=5)
    raw = igrams.ds["igram"].compute().values

    filtered = igrams.filter_goldstein(alpha=0.0)
    np.testing.assert_allclose(
        filtered.ds["igram"].compute().values, raw, rtol=1e-4, atol=1e-4
    )
    g1.close()
    g2.close()


def test_form_filter_unwrap_pipeline(gslc_factory, tmp_path):
    gslcs = []
    for k in range(3):
        p = gslc_factory(ny=80, nx=80, seed=k,
                         datetime_str=f"2025-11-{10 + k:02d}T00:00:00.000000000")
        gslcs.append(GSLC(p))
    ws = Workspace(tmp_path / "ws")
    stack = GSLCStack.from_gslcs(gslcs).persist(ws, "slc_stack")
    for g in gslcs:
        g.close()

    igrams = stack.form_interferograms(pairs="all", looks=5, downsample=True)
    filtered = igrams.filter_goldstein(alpha=0.7, patch_size=16)
    filtered = filtered.persist(ws, "igrams_filt")

    unw = filtered.unwrap(ws, nproc=1)
    assert isinstance(unw, UnwrappedStack)
    assert unw.sizes["pair"] == filtered.sizes["pair"]
    assert unw.ds["unw"].dtype == np.float32
    # The filter params survived persistence and are part of the stage hash.
    assert filtered.ds.attrs["goldstein"]["alpha"] == 0.7


def test_filter_params_change_stage_hash(gslc_factory, tmp_path):
    g1, g2 = _two_gslcs(gslc_factory, ny=60, nx=60)
    igrams = GSLCStack.from_gslcs([g1, g2]).form_interferograms(looks=5)
    ws = Workspace(tmp_path / "ws")

    igrams.filter_goldstein(alpha=0.5).persist(ws, "igrams_filt")
    # Same stage name, different alpha => different hash => must refuse to reuse.
    with pytest.raises(WorkspaceError):
        igrams.filter_goldstein(alpha=0.9).persist(ws, "igrams_filt")
    g1.close()
    g2.close()
