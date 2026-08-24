"""Is the nuisance ramp eating the earthquake?

Variance reduction cannot answer this and never could: the ramp columns are free
and unregularised, so adding them can only improve the fit. A ramp quietly
absorbing coseismic signal therefore shows up as a **better fit with a smaller
earthquake**, and that conjunction -- not either half alone -- is what these
tests pin.

Two complementary measures, and the pair matters. ``ramp_leakage`` is geometric
and needs no solve: how much of the fault's own predicted field lies in the ramp's
column space, i.e. what the ramp *could* absorb. ``ramp_sweep`` re-solves at
several degrees and reports what it *did*. The healthy fixture below shows why
both are needed -- its linear leakage is 0.55, which sounds alarming, while the
moment is stable to 0.3%.
"""

import numpy as np
import pytest

from nisar_tools.slip import FaultMesh, FaultTrace, Observations, SlipInversion
from nisar_tools.slip.diagnostics import ramp_leakage, ramp_sweep

from slip_synthetic import forward_los_stack, tapered_slip

_LON = np.array([-69.0, -68.6, -68.2, -67.8, -67.4, -67.0])
_LAT = np.array([10.55, 10.56, 10.57, 10.575, 10.585, 10.59])
_SOLVE = dict(smoothing=0.3, polarity=(-1, 0, 0), max_iter=200)


def _inversion(exclude_within):
    trace = FaultTrace.from_lonlat(_LON, _LAT, name="t")
    frame = trace.local_frame()
    mesh = FaultMesh.vertical(trace, frame, max_depth=20e3, edge_length=8e3)
    truth = tapered_slip(mesh, peak=-2.0)
    parts = [
        Observations.from_los(
            forward_los_stack(mesh, truth, trace, frame, geometry=geometry,
                              spacing=6000.0, half_x=120e3, half_y=90e3,
                              noise=0.004, seed=1),
            name=geometry, frame=frame, trace=trace, rms_min=0.006,
            width_min=8000.0, width_max=60000.0, exclude_within=exclude_within)
        for geometry in ("asc", "desc")
    ]
    return SlipInversion(mesh, Observations.concat(parts), ramp="offset")


@pytest.fixture(scope="module")
def healthy():
    """Full aperture with the near field kept -- the case that should be fine."""
    return _inversion(5e3)


@pytest.fixture(scope="module")
def starved():
    """Everything within 60 km of the trace thrown away.

    What is left is the smooth far-field tail of an arctangent step, which is very
    nearly a low-order polynomial over this aperture -- so the ramp can reproduce
    the earthquake, and does.
    """
    return _inversion(60e3)


# -- leakage: what the ramp *could* absorb ---------------------------------
def test_a_healthy_aperture_leaks_almost_no_curvature(healthy):
    leak = ramp_leakage(healthy)

    assert leak["curvature_only"] < 0.05, leak["curvature_only"]
    # A per-track offset always absorbs that track's mean displacement -- an
    # unwrapped interferogram has an arbitrary constant -- so this is small but
    # never zero, and is not a defect.
    assert 0.0 < leak["offset"]["leakage"] < 0.25
    assert set(leak["quadratic"]["per_track"]) == {"asc", "desc"}


def test_leakage_saturates_when_the_ramp_can_reproduce_the_field(starved):
    leak = ramp_leakage(starved)

    assert leak["quadratic"]["leakage"] > 0.95, leak["quadratic"]["leakage"]
    assert leak["quadratic"]["leakage"] > leak["offset"]["leakage"] + 0.5
    for kind in ("offset", "linear", "quadratic", "cubic"):
        assert 0.0 <= leak[kind]["leakage"] <= 1.0 + 1e-9


def test_leakage_rises_with_degree_and_needs_no_solved_model(healthy):
    """Nesting: a higher degree contains the lower one, so it cannot leak less."""
    leak = ramp_leakage(healthy)
    values = [leak[k]["leakage"] for k in ("offset", "linear", "quadratic", "cubic")]

    np.testing.assert_array_less(-1e-9, np.diff(values))
    # `slip=None` uses a generic unit pattern, so the number describes the
    # geometry and aperture rather than one particular answer.
    solved = healthy.solve(**_SOLVE)
    from_model = ramp_leakage(healthy, slip=solved)
    assert 0.0 <= from_model["quadratic"]["leakage"] <= 1.0 + 1e-9


# -- the sweep: what the ramp *did* ----------------------------------------
def test_the_sweep_clears_a_ramp_that_leaves_the_moment_alone(healthy):
    sweep = ramp_sweep(healthy, kinds=("offset", "linear", "quadratic"), **_SOLVE)

    assert list(sweep["ramp"].values) == ["offset", "linear", "quadratic"]
    assert sweep.attrs["suspect"] == []
    assert bool(sweep["converged"].all())
    np.testing.assert_allclose(sweep["moment_ratio"].values, 1.0, atol=0.05)
    # More free columns can only fit better; that is exactly why VR is no guide.
    np.testing.assert_array_less(-1e-9, np.diff(sweep["variance_reduction"].values))


def test_the_sweep_catches_a_ramp_eating_the_earthquake(starved):
    """The positive control. A test that can only ever pass proves nothing."""
    with pytest.warns(RuntimeWarning, match="smaller earthquake"):
        sweep = ramp_sweep(starved, kinds=("offset", "linear", "quadratic"), **_SOLVE)

    assert "quadratic" in sweep.attrs["suspect"]
    at = dict(ramp="quadratic")
    assert float(sweep["moment_ratio"].sel(**at)) < 0.5
    assert float(sweep["vr_change"].sel(**at)) > 0.0


def test_a_worse_fit_is_not_flagged_as_absorption(starved):
    """The discriminating case, and the reason the rule is a conjunction.

    On the starved fixture the *linear* ramp also loses most of the moment -- but
    it fits worse than the offset while doing so, which makes it a worse model
    rather than a ramp absorbing signal. Flagging on the moment alone would call
    it a false positive.
    """
    with pytest.warns(RuntimeWarning):
        sweep = ramp_sweep(starved, kinds=("offset", "linear", "quadratic"), **_SOLVE)

    linear = dict(ramp="linear")
    assert float(sweep["moment_ratio"].sel(**linear)) < 0.8
    assert float(sweep["vr_change"].sel(**linear)) < 0.0
    assert "linear" not in sweep.attrs["suspect"]


def test_the_sweep_reuses_the_greens_matrix(healthy):
    """The performance contract that makes this a routine check, not an hour."""
    other = healthy.with_ramp("quadratic")

    assert other.g is healthy.g
    assert other.n_ramp > healthy.n_ramp
    assert other.obs is healthy.obs
    # The original is untouched, or a sweep would corrupt its own baseline.
    assert healthy.ramp.shape[1] == healthy.n_ramp


def test_the_reference_degree_must_be_in_the_sweep(healthy):
    with pytest.raises(ValueError, match="reference"):
        ramp_sweep(healthy, kinds=("linear", "quadratic"), reference="offset")
