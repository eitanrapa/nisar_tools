"""Tests for the dual-frequency (split-spectrum) ionospheric correction.

The algebra is pinned against a *forward model*: a non-dispersive field and a
dispersive one are chosen, the two sub-band phases are synthesized from them
with the ``f`` / ``1/f`` scaling, and the estimator has to give the dispersive
one back. That is the only way to catch a sign or a reciprocal being flipped --
a test that inverts the same expression the code composes would pass either way,
which is exactly how the LOS sign stayed wrong for months.
"""

import warnings

import numpy as np
import pytest
import xarray as xr

from nisar_tools import GSLC, GSLCStack, InterferogramStack, UnwrappedStack
from nisar_tools import ionosphere as iono

FREQ_A = 1_239_000_000.0
FREQ_B = 1_293_500_000.0

REF_TIME = np.datetime64("2025-11-28T02:32:16")
SEC_TIME = np.datetime64("2025-12-10T02:32:16")


def _fields(ny=200, nx=400):
    """A sharp non-dispersive step plus a broad dispersive screen."""
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    nondispersive = 3.0 * np.tanh((xx - nx / 2) / 6.0) + 0.02 * yy
    dispersive = 5.0 * np.exp(-(((xx - 120) / 120.0) ** 2
                                + ((yy - 100) / 100.0) ** 2))
    return nondispersive, dispersive


def _sub_bands(nondispersive, dispersive):
    """The two sub-band phases implied by those fields, referred to band A."""
    phase_a = nondispersive + dispersive
    phase_b = (FREQ_B / FREQ_A) * nondispersive + (FREQ_A / FREQ_B) * dispersive
    return phase_a, phase_b


def _stack(phase, frequency, dx=50.0, dy=50.0, coherence=None, npair=1):
    """An ``UnwrappedStack`` around a ready-made phase field."""
    phase = np.asarray(phase, dtype=np.float32)
    if phase.ndim == 2:
        phase = np.broadcast_to(phase, (npair,) + phase.shape)
    npair, ny, nx = phase.shape
    data = {"unw": (("pair", "y", "x"), np.ascontiguousarray(phase))}
    if coherence is not None:
        data["coherence"] = (
            ("pair", "y", "x"),
            np.broadcast_to(np.asarray(coherence, np.float32),
                            phase.shape).copy(),
        )
    ds = xr.Dataset(
        data,
        coords={
            "pair": np.arange(npair),
            "y": 4.0e6 - dy * np.arange(ny),
            "x": 4.0e5 + dx * np.arange(nx),
            "ref_time": ("pair", np.array([REF_TIME] * npair)),
            "sec_time": ("pair", np.array([SEC_TIME] * npair)),
        },
    ).rio.write_crs("EPSG:32611")
    ds.attrs.update(epsg=32611, source="snaphu", direction="Descending",
                    center_frequency=float(frequency))
    return UnwrappedStack(ds)


# -- the algebra -----------------------------------------------------------

def test_split_recovers_the_two_parts_from_a_forward_model():
    """Both halves come back exactly, and they add up to band A's own phase."""
    rng = np.random.default_rng(0)
    nondispersive = rng.normal(size=64)
    dispersive = rng.normal(size=64)
    phase_a, phase_b = _sub_bands(nondispersive, dispersive)

    got_disp = iono.dispersive_phase(phase_a, phase_b, FREQ_A, FREQ_B)
    got_nond = iono.nondispersive_phase(phase_a, phase_b, FREQ_A, FREQ_B)
    np.testing.assert_allclose(got_disp, dispersive, atol=1e-12)
    np.testing.assert_allclose(got_nond, nondispersive, atol=1e-12)
    np.testing.assert_allclose(got_disp + got_nond, phase_a, atol=1e-12)


def test_argument_order_carries_no_hidden_convention():
    """Which band is the lower carrier is worked out, not assumed."""
    rng = np.random.default_rng(1)
    nondispersive, dispersive = rng.normal(size=32), rng.normal(size=32)
    phase_a, phase_b = _sub_bands(nondispersive, dispersive)
    swapped = iono.dispersive_phase(phase_b, phase_a, FREQ_B, FREQ_A,
                                    f_reference=FREQ_A)
    np.testing.assert_allclose(swapped, dispersive, atol=1e-12)


def test_nisar_coefficients_and_noise_gain():
    """The numbers the docs quote, pinned: 12.12 / -11.61, gain 16.8.

    They are what makes the low-pass mandatory, so a change in them is a change
    in the advice, not just in a constant.
    """
    (d_a, d_b), (n_a, n_b) = iono.split_coefficients(FREQ_A, FREQ_B)
    assert d_a == pytest.approx(12.122, abs=1e-3)
    assert d_b == pytest.approx(-11.612, abs=1e-3)
    assert d_a + n_a == pytest.approx(1.0)
    assert d_b + n_b == pytest.approx(0.0, abs=1e-12)
    assert iono.noise_gain(FREQ_A, FREQ_B) == pytest.approx(16.786, abs=1e-3)


def test_one_carrier_cannot_separate_two_unknowns():
    with pytest.raises(ValueError, match="same centre frequency"):
        iono.split_coefficients(FREQ_A, FREQ_A)


def test_the_screen_is_what_a_deramp_cannot_reach():
    """A dispersive screen and a non-dispersive ramp are different objects.

    The point of the module: a ramp scales with ``f`` like the deformation, so
    it survives the correction untouched -- which is why this is not a slower
    ``deramp``, and why a deramp is still needed afterwards.
    """
    ny, nx = 40, 60
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    ramp = 0.4 * xx - 0.2 * yy
    phase_a, phase_b = _sub_bands(ramp, np.zeros_like(ramp))
    screen = iono.dispersive_phase(phase_a, phase_b, FREQ_A, FREQ_B)
    assert np.abs(screen).max() < 1e-9


# -- the estimator ---------------------------------------------------------

def test_screen_recovered_from_two_unwrapped_stacks():
    nondispersive, dispersive = _fields()
    phase_a, phase_b = _sub_bands(nondispersive, dispersive)
    screen = iono.dual_frequency_screen(
        _stack(phase_a, FREQ_A), _stack(phase_b, FREQ_B),
        scale=0, remove_offset=False,
    )
    # float32 storage of an O(5) field through coefficients of ~12.
    np.testing.assert_allclose(screen.values[0], dispersive, atol=1e-4)
    assert screen.name == "phase_screen"
    assert screen.attrs["method"] == "dual_frequency"


def test_smoothing_commutes_with_the_combination_on_a_shared_grid():
    """Each band is low-passed before combining; linearity says that is the same
    answer as smoothing the screen afterwards, whenever the grids match."""
    from nisar_tools._kernels import smooth_surface

    nondispersive, dispersive = _fields(ny=96, nx=120)
    phase_a, phase_b = _sub_bands(nondispersive, dispersive)
    screen = iono.dual_frequency_screen(
        _stack(phase_a, FREQ_A), _stack(phase_b, FREQ_B),
        scale=6.0, remove_offset=False,
    )
    np.testing.assert_allclose(screen.values[0],
                               smooth_surface(dispersive, (6.0, 6.0)),
                               atol=1e-4)


def test_a_coarser_second_band_is_smoothed_before_it_is_resampled():
    """Band B is posted 8x coarser in x on a real product, so this is the normal
    path -- and interpolating its *raw* phase would put the resampling error
    through the 12x coefficient. Smoothing first keeps the shape error at the
    coarse band's own sampling floor.
    """
    from nisar_tools._kernels import smooth_surface

    nondispersive, dispersive = _fields()
    phase_a, phase_b = _sub_bands(nondispersive, dispersive)
    coarse = _stack(phase_b[:, ::8], FREQ_B, dx=400.0)
    screen = iono.dual_frequency_screen(
        _stack(phase_a, FREQ_A), coarse, scale=32.0,
    ).values[0]

    reference = smooth_surface(dispersive, (32.0, 32.0))
    reference = reference - np.nanmedian(np.where(np.isnan(screen), np.nan,
                                                  reference))
    rms = np.sqrt(np.nanmean((screen - reference) ** 2))
    assert rms < 0.25, rms
    # The two bands' sigmas are the same ground distance, not the same pixels.
    assert screen is not None
    attrs = iono.dual_frequency_screen(
        _stack(phase_a, FREQ_A), coarse, scale=32.0).attrs
    assert attrs["scale"] == [32.0, 32.0]
    assert attrs["other_scale"] == [32.0, 4.0]
    assert attrs["scale_metres"] == pytest.approx(1600.0)


def test_a_narrow_low_pass_against_a_coarse_band_warns():
    """Below twice the coarse posting the resampling dominates the screen
    (rms 0.284 vs 0.159 rad, measured); legitimate, so it warns rather than
    raising."""
    nondispersive, dispersive = _fields(ny=48, nx=96)
    phase_a, phase_b = _sub_bands(nondispersive, dispersive)
    with pytest.warns(RuntimeWarning, match="narrower than twice"):
        iono.dual_frequency_screen(
            _stack(phase_a, FREQ_A),
            _stack(phase_b[:, ::8], FREQ_B, dx=400.0),
            scale=4.0,
        )


def test_default_scale_is_the_same_ground_distance_at_any_looks():
    """A pixel default would mean a different filter on every multilook."""
    nondispersive, dispersive = _fields(ny=64, nx=64)
    phase_a, phase_b = _sub_bands(nondispersive, dispersive)
    for spacing, expected in ((50.0, 40.0), (200.0, 10.0)):
        attrs = iono.dual_frequency_screen(
            _stack(phase_a, FREQ_A, dx=spacing, dy=spacing),
            _stack(phase_b, FREQ_B, dx=spacing, dy=spacing),
        ).attrs
        assert attrs["scale_metres"] == pytest.approx(iono.DEFAULT_SMOOTH_METRES)
        assert attrs["scale"] == [expected, expected]


def test_offset_removal_leaves_the_shape_alone():
    nondispersive, dispersive = _fields(ny=48, nx=64)
    phase_a, phase_b = _sub_bands(nondispersive, dispersive)
    kwargs = dict(scale=0)
    a, b = _stack(phase_a, FREQ_A), _stack(phase_b, FREQ_B)
    kept = iono.dual_frequency_screen(a, b, remove_offset=False, **kwargs).values
    centred = iono.dual_frequency_screen(a, b, remove_offset=True, **kwargs).values
    spread = np.ptp(kept - centred)
    assert spread < 1e-4                      # a constant, per pair
    assert np.median(centred) == pytest.approx(0.0, abs=1e-4)


def test_low_coherence_pixels_are_filled_from_their_neighbours():
    """The gate runs before the low-pass, so a decorrelated patch is
    interpolated over instead of smeared across the screen."""
    nondispersive, dispersive = _fields(ny=64, nx=64)
    phase_a, phase_b = _sub_bands(nondispersive, dispersive)
    phase_a = phase_a.copy()
    phase_a[30:34, 30:34] += 40.0            # a decorrelated patch
    coherence = np.ones((64, 64), np.float32)
    coherence[30:34, 30:34] = 0.05

    a = _stack(phase_a, FREQ_A, coherence=coherence)
    b = _stack(phase_b, FREQ_B, coherence=np.ones((64, 64), np.float32))
    ungated = iono.dual_frequency_screen(a, b, scale=6.0, remove_offset=False)
    gated = iono.dual_frequency_screen(a, b, scale=6.0, min_coherence=0.3,
                                       remove_offset=False)
    # 40 rad of nonsense over 4x4 pixels, spread by the low-pass and amplified
    # 12x, still dominates the 5 rad screen; the gate takes it out entirely.
    contaminated = np.abs(ungated.values[0]).max()
    clean = np.abs(gated.values[0]).max()
    assert contaminated > 30.0
    assert clean < contaminated / 5


def test_min_coherence_without_a_coherence_layer_says_so():
    nondispersive, dispersive = _fields(ny=32, nx=32)
    phase_a, phase_b = _sub_bands(nondispersive, dispersive)
    with pytest.raises(ValueError, match="needs a coherence layer"):
        iono.dual_frequency_screen(_stack(phase_a, FREQ_A), _stack(phase_b, FREQ_B),
                                   min_coherence=0.3)


def test_a_stack_with_no_recorded_carrier_names_the_way_out():
    nondispersive, dispersive = _fields(ny=32, nx=32)
    phase_a, phase_b = _sub_bands(nondispersive, dispersive)
    a = _stack(phase_a, FREQ_A)
    del a.ds.attrs["center_frequency"]
    with pytest.raises(ValueError, match="does not record its centre frequency"):
        iono.dual_frequency_screen(a, _stack(phase_b, FREQ_B))
    # ...and the override works.
    screen = iono.dual_frequency_screen(a, _stack(phase_b, FREQ_B),
                                        frequency=FREQ_A, scale=0)
    np.testing.assert_allclose(screen.values[0], dispersive - np.median(dispersive),
                               atol=1e-3)


def test_mismatched_pair_counts_are_refused():
    nondispersive, dispersive = _fields(ny=32, nx=32)
    phase_a, phase_b = _sub_bands(nondispersive, dispersive)
    with pytest.raises(ValueError, match="different pair counts"):
        iono.dual_frequency_screen(_stack(phase_a, FREQ_A, npair=2),
                                   _stack(phase_b, FREQ_B, npair=1))


def test_different_crs_is_refused():
    nondispersive, dispersive = _fields(ny=32, nx=32)
    phase_a, phase_b = _sub_bands(nondispersive, dispersive)
    b = _stack(phase_b, FREQ_B)
    b.ds.attrs["epsg"] = 32612
    with pytest.raises(ValueError, match="different CRSs"):
        iono.dual_frequency_screen(_stack(phase_a, FREQ_A), b)


def test_the_estimate_stays_lazy_and_decomposes():
    da = pytest.importorskip("dask.array")
    nondispersive, dispersive = _fields(ny=256, nx=256)
    phase_a, phase_b = _sub_bands(nondispersive, dispersive)
    a, b = _stack(phase_a, FREQ_A), _stack(phase_b, FREQ_B)
    for stack in (a, b):
        stack.ds["unw"] = stack.ds["unw"].copy(
            data=da.from_array(stack.ds["unw"].values, chunks=(1, 256, 256))
        )
    screen = iono.dual_frequency_screen(a, b, scale=4.0, remove_offset=False)
    assert screen.chunks is not None
    assert np.prod(screen.data.numblocks[-2:]) > 1   # actually decomposed
    np.testing.assert_allclose(
        screen.values,
        iono.dual_frequency_screen(_stack(phase_a, FREQ_A), _stack(phase_b, FREQ_B),
                                   scale=4.0, remove_offset=False).values,
        atol=1e-4,
    )


def test_the_offset_median_survives_chunking_and_an_all_nan_pair():
    """``median(dim=("y", "x"))`` is not available on dask at all -- it refuses
    every two-axis reduction -- and ``np.nanmedian`` warns on a fully masked
    pair, which a water- or edge-masked stack legitimately has.
    """
    da = pytest.importorskip("dask.array")
    nondispersive, dispersive = _fields(ny=128, nx=128)
    phase_a, phase_b = _sub_bands(nondispersive, dispersive)
    phase_a = np.stack([phase_a, np.full_like(phase_a, np.nan)])
    phase_b = np.stack([phase_b, np.full_like(phase_b, np.nan)])
    a, b = _stack(phase_a, FREQ_A), _stack(phase_b, FREQ_B)
    for stack in (a, b):
        stack.ds["unw"] = stack.ds["unw"].copy(
            data=da.from_array(stack.ds["unw"].values, chunks=(1, 64, 64))
        )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        # The fixture gives every pair the same times, so pair by position.
        screen = iono.dual_frequency_screen(a, b, scale=0,
                                            time_tolerance=None).values
    assert np.nanmedian(screen[0]) == pytest.approx(0.0, abs=1e-4)
    assert np.all(np.isnan(screen[1]))


# -- applying it, unwrapped ------------------------------------------------

def test_correcting_unwrapped_phase_leaves_the_non_dispersive_part():
    nondispersive, dispersive = _fields(ny=96, nx=128)
    phase_a, phase_b = _sub_bands(nondispersive, dispersive)
    corrected = _stack(phase_a, FREQ_A).correct_ionosphere(
        _stack(phase_b, FREQ_B), scale=0,
    )
    got = corrected.ds["unw"].values[0]
    # Up to the arbitrary constant every unwrapped phase carries.
    assert np.ptp(got - nondispersive) < 1e-3
    assert "phase_screen" in corrected.ds
    assert corrected.ds.attrs["phase_screen_removed"] is True
    assert corrected.ds.attrs["ionosphere"]["applied"] == "unwrapped"
    assert corrected.ds.attrs["ionosphere"]["noise_gain"] == pytest.approx(16.786,
                                                                          abs=1e-3)


def test_correcting_twice_is_refused():
    nondispersive, dispersive = _fields(ny=32, nx=32)
    phase_a, phase_b = _sub_bands(nondispersive, dispersive)
    a, b = _stack(phase_a, FREQ_A), _stack(phase_b, FREQ_B)
    once = a.correct_ionosphere(b, scale=0)
    with pytest.raises(ValueError, match="already carries a 'phase_screen'"):
        once.correct_ionosphere(b, scale=0)


def test_an_existing_screen_is_only_replaced_on_request():
    """A GUNW ships NASA's own screen; overwriting it silently would lose which
    of the two the numbers came from."""
    nondispersive, dispersive = _fields(ny=32, nx=32)
    phase_a, phase_b = _sub_bands(nondispersive, dispersive)
    a, b = _stack(phase_a, FREQ_A), _stack(phase_b, FREQ_B)
    a.ds["phase_screen"] = a.ds["unw"] * 0.0
    with pytest.raises(ValueError, match="already carries a 'phase_screen'"):
        a.correct_ionosphere(b, scale=0)
    replaced = a.correct_ionosphere(b, scale=0, replace=True)
    assert replaced.ds.attrs["ionosphere"]["method"] == "dual_frequency"


def test_the_correction_folds_into_the_unwrapped_persist_hash(tmp_path):
    """Two workspaces, one stage name: the *only* difference is the correction,
    so the hashes have to differ -- otherwise a corrected store would be reused
    for an uncorrected run, silently."""
    from nisar_tools import Workspace

    nondispersive, dispersive = _fields(ny=32, nx=32)
    phase_a, phase_b = _sub_bands(nondispersive, dispersive)
    a, b = _stack(phase_a, FREQ_A), _stack(phase_b, FREQ_B)
    plain = Workspace(tmp_path / "plain")
    iono_ws = Workspace(tmp_path / "iono")
    a.persist(plain, "unw")
    a.correct_ionosphere(b, scale=0).persist(iono_ws, "unw")
    assert plain.stored_params_hash("unw") != iono_ws.stored_params_hash("unw")


# -- applying it, wrapped --------------------------------------------------

def _igram_stack(phase, coherence=0.8):
    phase = np.asarray(phase, dtype=float)
    ny, nx = phase.shape
    ds = xr.Dataset(
        {
            "igram": (("pair", "y", "x"),
                      np.exp(1j * phase)[None].astype(np.complex64)),
            "coherence": (("pair", "y", "x"),
                          np.full((1, ny, nx), coherence, np.float32)),
        },
        coords={
            "pair": [0],
            "y": 4.0e6 - 50.0 * np.arange(ny),
            "x": 4.0e5 + 50.0 * np.arange(nx),
            "ref_time": ("pair", np.array([REF_TIME])),
            "sec_time": ("pair", np.array([SEC_TIME])),
        },
    ).rio.write_crs("EPSG:32611")
    ds.attrs.update(epsg=32611, direction="Descending", looks=1)
    return InterferogramStack(ds)


def test_wrapped_correction_rotates_the_interferogram_by_the_screen():
    nondispersive, dispersive = _fields(ny=64, nx=96)
    phase_a, phase_b = _sub_bands(nondispersive, dispersive)
    screen = iono.dual_frequency_screen(
        _stack(phase_a, FREQ_A), _stack(phase_b, FREQ_B),
        scale=0, remove_offset=False,
    )
    igrams = _igram_stack(phase_a)
    out = igrams.correct_ionosphere(screen)

    got = np.angle(out.ds["igram"].values[0])
    residual = np.angle(np.exp(1j * (got - nondispersive)))
    assert np.abs(residual).max() < 1e-3
    # A phase rotation is not an amplitude change.
    np.testing.assert_array_equal(out.ds["coherence"].values,
                                  igrams.ds["coherence"].values)
    assert out.ds.attrs["ionosphere"]["applied"] == "wrapped"


def test_a_screen_estimated_on_a_coarse_grid_applies_to_a_fine_one():
    """The reason to correct before unwrapping: the screen can come from a
    cheaply multilooked pair and still be removed at full posting."""
    nondispersive, dispersive = _fields(ny=64, nx=96)
    phase_a, phase_b = _sub_bands(nondispersive, dispersive)
    coarse = iono.dual_frequency_screen(
        _stack(phase_a[::4, ::4], FREQ_A, dx=200.0, dy=200.0),
        _stack(phase_b[::4, ::4], FREQ_B, dx=200.0, dy=200.0),
        scale=4.0, remove_offset=False,
    )
    out = _igram_stack(phase_a).correct_ionosphere(coarse)
    assert out.ds["igram"].shape == (1, 64, 96)
    got = np.angle(out.ds["igram"].values[0])
    # Inside the coarse screen's own footprint the deformation is what is left.
    inner = np.angle(np.exp(1j * (got - nondispersive)))[8:-8, 8:-8]
    assert np.abs(inner).mean() < 0.35


def test_an_unwrapped_stack_can_stand_in_for_the_screen():
    nondispersive, dispersive = _fields(ny=32, nx=48)
    phase_a, phase_b = _sub_bands(nondispersive, dispersive)
    corrected = _stack(phase_a, FREQ_A).correct_ionosphere(
        _stack(phase_b, FREQ_B), scale=0, remove_offset=False)
    out = _igram_stack(phase_a).correct_ionosphere(corrected)
    got = np.angle(out.ds["igram"].values[0])
    residual = np.angle(np.exp(1j * (got - nondispersive)))
    assert np.abs(residual).max() < 1e-3


def test_a_stack_without_a_screen_is_refused():
    nondispersive, dispersive = _fields(ny=32, nx=32)
    phase_a, _ = _sub_bands(nondispersive, dispersive)
    with pytest.raises(ValueError, match="carries no 'phase_screen'"):
        _igram_stack(phase_a).correct_ionosphere(_stack(phase_a, FREQ_A))


def test_wrapped_correction_twice_is_refused():
    nondispersive, dispersive = _fields(ny=32, nx=32)
    phase_a, phase_b = _sub_bands(nondispersive, dispersive)
    screen = iono.dual_frequency_screen(_stack(phase_a, FREQ_A),
                                        _stack(phase_b, FREQ_B), scale=0)
    once = _igram_stack(phase_a).correct_ionosphere(screen)
    with pytest.raises(ValueError, match="already been removed"):
        once.correct_ionosphere(screen)


def test_the_wrapped_correction_folds_into_the_igram_persist_hash(tmp_path):
    from nisar_tools import Workspace

    nondispersive, dispersive = _fields(ny=32, nx=32)
    phase_a, phase_b = _sub_bands(nondispersive, dispersive)
    screen = iono.dual_frequency_screen(_stack(phase_a, FREQ_A),
                                        _stack(phase_b, FREQ_B), scale=0)
    plain = Workspace(tmp_path / "plain")
    iono_ws = Workspace(tmp_path / "iono")
    igrams = _igram_stack(phase_a)
    igrams.persist(plain, "igrams")
    igrams.correct_ionosphere(screen).persist(iono_ws, "igrams")
    assert plain.stored_params_hash("igrams") != iono_ws.stored_params_hash("igrams")


# -- the carrier follows the phase down the pipeline ------------------------

def test_the_sub_band_carrier_reaches_the_interferogram_stack(gslc_factory):
    """No argument should be needed on real data, which means the carrier has to
    survive GSLC -> GSLCStack -> InterferogramStack."""
    paths = [gslc_factory(ny=32, nx=32, frequency="B",
                          datetime_str=f"2025-11-2{i}T02:32:16.000000000")
             for i in (7, 8)]
    granules = [GSLC(p, frequency="B") for p in paths]
    try:
        assert granules[0].center_frequency == pytest.approx(FREQ_B)
        assert granules[0].wavelength == pytest.approx(299_792_458.0 / FREQ_B)
        stack = GSLCStack.from_gslcs(granules)
        assert stack.ds.attrs["frequency"] == "B"
        assert stack.ds.attrs["center_frequency"] == pytest.approx(FREQ_B)
        igrams = stack.form_interferograms(looks=2)
        assert igrams.ds.attrs["center_frequency"] == pytest.approx(FREQ_B)
    finally:
        for g in granules:
            g.close()


def test_a_stack_cannot_mix_the_two_sub_bands(gslc_factory):
    """Two carriers are not interferometric with each other -- that is the whole
    premise of the correction, so mixing them in one stack is an error."""
    a = GSLC(gslc_factory(ny=16, nx=16, frequency="A"), frequency="A")
    b = GSLC(gslc_factory(ny=16, nx=16, frequency="B",
                          datetime_str="2025-12-10T02:32:16.000000000"),
             frequency="B")
    try:
        with pytest.raises(ValueError, match="multiple sub-bands"):
            GSLCStack.from_gslcs([a, b])
    finally:
        a.close()
        b.close()
