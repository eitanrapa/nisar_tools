"""Pins the calibration harness: the guards, the ceiling, and the staging.

The module under test exists because a synthetic recovery experiment is easy to
run and easy to get silently wrong. Every test here corresponds to a specific way
it was got wrong first, so they are acceptance tests for the protocol rather than
coverage of the code.

⚠️ The expensive part of a calibration -- `run_case`, which builds Green's
functions and runs a sampling loop -- is deliberately **not** exercised here.
It is minutes per call even on a toy mesh, and everything it does beyond
orchestration is already pinned by `test_slip_predict.py` and
`test_slip_resolution.py`. What is tested is the part that is genuinely new: the
target's relationship to the bounds, the discretisation ceiling, the scoring
convention, and the plan's cost.
"""

import numpy as np
import pytest
from slip_synthetic import correlated_noise

from nisar_tools.slip import FaultMesh, FaultTrace
from nisar_tools.slip import calibrate as C
from nisar_tools.slip.covariance import CovarianceModel, correlated_field

_LON = np.array([-68.681, -68.300, -67.900, -67.500, -67.100, -66.700, -66.523])
_LAT = np.array([10.410, 10.490, 10.540, 10.565, 10.595, 10.620, 10.630])


@pytest.fixture(scope="module")
def mesh():
    trace = FaultTrace(_LON, _LAT, name="calib")
    frame = trace.local_frame()
    return FaultMesh.curved(trace, frame, uniform_dip=100.0, max_depth=40e3,
                            edge_length=8e3, down_dip_levels=11)


DEXTRAL = dict(polarity=(-1, 0, 0), strike=(-8.0, 8.0), dip=(-3.0, 3.0))


# -- the target must be reachable ------------------------------------------

def test_sign_alternating_target_is_refused_under_a_polarity_bound(mesh):
    """The bug this module exists to prevent.

    A textbook checkerboard alternates sign. Under `polarity=(-1, 0, 0)` almost
    half of it is in a region the solver cannot enter, so its recovery score
    measures the constraint -- and every summary statistic stays plausible while
    it does. Refusing to build it is the only reliable guard, because nothing
    downstream looks wrong.
    """
    with pytest.raises(ValueError, match="polarity bound"):
        C.plant_checkerboard(mesh, 40e3, DEXTRAL, base=0.0, swing=2.0)


def test_magnitude_target_is_entirely_inside_the_bound(mesh):
    target = C.plant_checkerboard(mesh, 40e3, DEXTRAL, base=1.25, swing=1.0)
    assert target.strike.max() < 0.0
    assert target.unreachable_fraction() == 0.0
    # The alternation survives: a factor of nine between weak and strong cells.
    assert np.isclose(target.strike.min(), -2.25, atol=0.05)
    assert np.isclose(target.strike.max(), -0.25, atol=0.05)


def test_without_a_polarity_bound_the_classic_form_is_restored(mesh):
    """A sign-free estimator should get the standard test, not a hobbled one."""
    target = C.plant_checkerboard(mesh, 40e3, None, base=0.0, swing=2.0)
    assert target.sign == 0
    assert target.strike.min() < -1.5 and target.strike.max() > 1.5
    assert target.unreachable_fraction() == 0.0


def test_bound_sign_reads_the_polarity_entry():
    assert C.bound_sign(DEXTRAL) == -1
    assert C.bound_sign(dict(polarity=(1, 0, 0))) == 1
    assert C.bound_sign(dict(strike=(-8.0, 8.0))) == 0
    assert C.bound_sign(None) == 0


# -- the discretisation ceiling --------------------------------------------

def test_ceiling_is_pure_geometry_and_flags_the_undersampled_wavelength(mesh):
    """A wavelength the mesh cannot carry must be excluded before it costs hours.

    On this mesh (about 8 km between node columns) a 20 km cell is only 2.5
    samples across -- above Nyquist, but with almost no margin, and its pattern
    correlation falls well below the longer cells'. A 10 km cell is below Nyquist
    outright and must be marked unusable.
    """
    rows = {r["cell"]: r for r in
            C.discretisation_ceiling(mesh, (10e3, 20e3, 40e3, 60e3))}
    assert rows[10e3]["samples_per_cell"] < 2.0
    assert not rows[10e3]["usable"]
    for cell in (20e3, 40e3, 60e3):
        assert rows[cell]["usable"]
        assert rows[cell]["amplitude_retained"] > 0.95
    # Longer wavelengths are carried more faithfully; that ordering is the point.
    assert (rows[20e3]["pattern_correlation"]
            < rows[40e3]["pattern_correlation"]
            < rows[60e3]["pattern_correlation"])


# -- the scoring convention -------------------------------------------------

def test_score_removes_the_mean_so_a_uniform_offset_cannot_flatter_it():
    """Amplitude ratio on the raw field is not a recovery statistic.

    Measured on real output: a model with pattern correlation 0.03 still returned
    an amplitude ratio of 0.88, because the uniform component of a magnitude
    checkerboard is recovered by anything at any damping. Scoring the oscillating
    part is what stops that reading as success.
    """
    rng = np.random.default_rng(0)
    want = -1.25 + 1.0 * np.sin(np.linspace(0, 8 * np.pi, 500))
    # An "estimate" that captures the base perfectly and the structure not at all.
    got = np.full_like(want, want.mean()) + 0.01 * rng.standard_normal(want.size)
    s = C.recovery_score(got, want)
    assert abs(s["correlation"]) < 0.2
    assert s["amplitude_ratio"] < 0.05        # the base is invisible to the score
    # An estimate that returns only the mean leaves the whole oscillation as
    # error, so rel_error is rms(sin) / rms(-1.25 + sin) = 0.707 / 1.436.
    # Pinning the value rather than a threshold makes the contrast with the
    # flattering 0.88 amplitude ratio explicit.
    assert s["rel_error"] == pytest.approx(0.492, abs=0.01)


def test_score_is_perfect_on_an_exact_recovery():
    want = -1.25 + np.sin(np.linspace(0, 6 * np.pi, 300))
    s = C.recovery_score(want.copy(), want)
    assert s["correlation"] == pytest.approx(1.0)
    assert s["amplitude_ratio"] == pytest.approx(1.0)
    assert s["rel_error"] == pytest.approx(0.0, abs=1e-12)


# -- reading a resolving length off the ladder ------------------------------

def test_resolving_wavelength_interpolates_between_rungs():
    table = [{"cell": 20e3, "rank": 79.0, "correlation": 0.17},
             {"cell": 40e3, "rank": 79.0, "correlation": 0.73},
             {"cell": 60e3, "rank": 79.0, "correlation": 0.75}]
    w = C.resolving_wavelength(table, rank=79.0, threshold=0.5)
    assert 20e3 < w < 40e3


def test_resolving_wavelength_is_none_when_nothing_clears_the_threshold():
    """The honest answer, and the one that should stop a resolution claim."""
    table = [{"cell": c, "rank": 21.0, "correlation": v}
             for c, v in ((20e3, 0.03), (40e3, 0.44), (60e3, 0.42))]
    assert C.resolving_wavelength(table, rank=21.0, threshold=0.5) is None


# -- the plan ---------------------------------------------------------------

def test_plan_is_staged_not_factorial_and_drops_unusable_wavelengths(mesh):
    plan = C.calibration_plan(mesh, cells=(10e3, 20e3, 40e3, 60e3),
                              budgets=(330, 660, 1320),
                              layouts=("rows", "unstructured"), seeds=5)
    assert 10e3 in plan["dropped_cells"]          # below Nyquist, never run
    assert plan["n_loops"] < plan["factorial_loops"] / 3
    stages = [c.stage for c in plan["cases"]]
    assert stages == sorted(stages), "stages must run in order; 2 and 3 need 1"
    # Stage 1 covers exactly the usable wavelengths, on one configuration.
    stage1 = [c for c in plan["cases"] if c.stage == 1]
    assert sorted(c.cell for c in stage1) == [20e3, 40e3, 60e3]
    assert len({(c.budget, c.layout, c.seed) for c in stage1}) == 1


def test_plan_case_names_are_unique(mesh):
    """Outputs are keyed by name, so a collision silently overwrites a result."""
    plan = C.calibration_plan(mesh)
    names = [c.name for c in plan["cases"]]
    assert len(names) == len(set(names))


# -- the noise synthesiser against its independent oracle -------------------

def test_library_noise_matches_the_test_oracle_bit_for_bit():
    """Pins the deliberate duplication between library and oracle.

    `covariance.correlated_field` and `slip_synthetic.correlated_noise` are two
    implementations of one algorithm, kept apart so the covariance round-trip
    tests verify the estimator against something independent of it. This test is
    what makes that duplication safe: if they diverge, one of them is wrong; if
    they are ever refactored onto a shared helper, this passes for the wrong
    reason and the round-trip tests quietly stop meaning anything.
    """
    model = CovarianceModel(sill=4e-4, range_m=22e3, nugget=1e-6)
    a = correlated_field((48, 64), 500.0, model, seed=11)
    b = correlated_noise((48, 64), 500.0, model.sill, model.range_m,
                         nugget=model.nugget, seed=11)
    assert np.array_equal(a, b)


def test_synthesised_noise_has_the_covariance_it_was_asked_for():
    """Forward check: the drawn field's variance and correlation length match.

    Not a round trip through the fitter -- that is `test_slip_covariance.py`'s
    job. This only asserts the draw is not silently mis-scaled, which is the
    failure that would make every calibration's noise level wrong.
    """
    model = CovarianceModel(sill=9e-4, range_m=10e3, nugget=0.0)
    field = correlated_field((256, 256), 1000.0, model, seed=3)
    assert field.std() == pytest.approx(np.sqrt(model.sill), rel=0.25)
    # Correlation at one range should sit near exp(-1); at three, near zero.
    lag = int(model.range_m / 1000.0)
    at_one = np.corrcoef(field[:, :-lag].ravel(), field[:, lag:].ravel())[0, 1]
    at_three = np.corrcoef(field[:, :-3 * lag].ravel(),
                           field[:, 3 * lag:].ravel())[0, 1]
    assert 0.2 < at_one < 0.6
    assert abs(at_three) < 0.25
