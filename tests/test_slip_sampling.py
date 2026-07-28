"""Pins the quadtree sampler and the ``Observations`` container.

Two properties matter beyond "it returns some points". First, the cells must
actually *adapt* -- fine where the field is rough, coarse where it is smooth --
or the decimation is just a stride. Second, the look vectors must be averaged
over exactly the pixels the displacement was averaged over; in the reference that
consistency is arranged by replaying stored cell index lists, and getting it
wrong would pair a displacement with a viewing geometry from somewhere else.
"""

import numpy as np
import pytest
from slip_synthetic import analytic_los_stack

from nisar_tools.slip import FaultTrace
from nisar_tools.slip.sampling import Observations

_LON = np.array([-68.681, -68.300, -67.900, -67.500, -67.100, -66.700, -66.523])
_LAT = np.array([10.410, 10.490, 10.540, 10.565, 10.595, 10.620, 10.630])


@pytest.fixture
def trace():
    return FaultTrace(_LON, _LAT, name="test_fault")


@pytest.fixture
def frame(trace):
    return trace.local_frame()


def _sample(trace, frame, **kwargs):
    stack = analytic_los_stack(trace, frame, spacing=kwargs.pop("spacing", 1000.0),
                              noise=kwargs.pop("noise", 0.0),
                              nan_rows=kwargs.pop("nan_rows", 0),
                              sign=kwargs.pop("sign", 1))
    kwargs.setdefault("rms_min", 0.008)
    kwargs.setdefault("width_min", 2500.0)
    kwargs.setdefault("width_max", 20000.0)
    kwargs.setdefault("exclude_within", 3000.0)
    return Observations.from_los(stack, frame=frame, trace=trace, **kwargs)


# -- adaptivity --------------------------------------------------------------

def test_cells_are_fine_near_the_fault_and_coarse_far_away(trace, frame):
    """The point of a quadtree: resolution follows the field, not a fixed stride."""
    obs = _sample(trace, frame)
    d = trace.distance(obs.ds["x"].values, obs.ds["y"].values, frame)
    near = obs.ds["cell_size"].values[d < 15e3]
    far = obs.ds["cell_size"].values[d > 60e3]
    assert near.size > 10 and far.size > 10
    assert near.mean() < 0.5 * far.mean()


def test_decimation_is_substantial(trace, frame):
    obs = _sample(trace, frame)
    assert obs.ds.attrs["n_raw_valid"] > 50 * obs.n


def test_a_smooth_field_needs_far_fewer_samples(trace, frame):
    """Raising the roughness threshold must coarsen the mesh of cells."""
    fine = _sample(trace, frame, rms_min=0.002)
    coarse = _sample(trace, frame, rms_min=0.05)
    assert coarse.n < fine.n


def test_cell_size_bounds_are_respected(trace, frame):
    obs = _sample(trace, frame, width_min=4000.0, width_max=16000.0)
    # cell_size is the mean of the two side lengths, so the bounds are on that.
    assert obs.ds["cell_size"].min() >= 3500.0
    assert obs.ds["cell_size"].max() <= 16500.0


# -- consistency -------------------------------------------------------------

def test_look_vectors_stay_unit_length(trace, frame):
    """Averaging three components of a unit vector shortens it; we renormalise.

    If this drifts, ``los_up == cos(incidence)`` stops holding and every
    projected Green's function is scaled by a slowly varying factor.
    """
    obs = _sample(trace, frame)
    norm = np.sqrt(obs.ds["look_e"] ** 2 + obs.ds["look_n"] ** 2 + obs.ds["look_u"] ** 2)
    np.testing.assert_allclose(norm.values, 1.0, atol=1e-12)


def test_samples_avoid_the_fault_trace(trace, frame):
    """Green's functions are singular on the fault, so samples must be kept off it."""
    obs = _sample(trace, frame, exclude_within=5000.0)
    d = trace.distance(obs.ds["x"].values, obs.ds["y"].values, frame)
    assert d.min() > 5000.0


def test_exclusion_needs_a_trace(trace, frame):
    stack = analytic_los_stack(trace, frame, spacing=2000.0)
    with pytest.raises(ValueError, match="needs a trace"):
        Observations.from_los(stack, frame=frame, exclude_within=1000.0)


def test_sign_attribute_restores_the_positive_toward_sensor_convention(trace, frame):
    """A stack that stores negated displacement, and says so, samples the same.

    ``LOSStack.attrs["sign"]`` records the convention *already applied* to the
    stored ``los``, so the sampler multiplies by it to get back to
    positive-toward-sensor. Two stacks describing the same ground motion -- one
    with ``sign=+1``, one with the data negated and ``sign=-1`` -- must therefore
    produce identical observations.

    Missing this would fit a slip model with the wrong sense of motion on every
    track at once, which the variance reduction cannot reveal.
    """
    positive = _sample(trace, frame, sign=1)
    negative = _sample(trace, frame, sign=-1)
    np.testing.assert_allclose(negative.ds["los"].values, positive.ds["los"].values,
                               rtol=1e-6)


def test_sign_attribute_is_actually_read(trace, frame):
    """Mislabel the convention and the sampled displacement flips.

    Guards against the attribute being ignored, which the test above would not
    catch on its own.
    """
    stack = analytic_los_stack(trace, frame, spacing=1500.0, sign=-1)
    honest = Observations.from_los(stack, frame=frame, trace=trace, rms_min=0.008,
                                   width_min=2500.0, width_max=20000.0,
                                   exclude_within=3000.0)
    stack.ds.attrs["sign"] = 1                        # now the label is wrong
    mislabelled = Observations.from_los(stack, frame=frame, trace=trace, rms_min=0.008,
                                        width_min=2500.0, width_max=20000.0,
                                        exclude_within=3000.0)
    np.testing.assert_allclose(mislabelled.ds["los"].values, -honest.ds["los"].values,
                               rtol=1e-9)

    stack.ds.attrs["sign"] = 0
    with pytest.raises(ValueError, match="sign attribute"):
        Observations.from_los(stack, frame=frame, trace=trace, exclude_within=3000.0)


def test_missing_swath_is_dropped_not_averaged(trace, frame):
    """NaN rows must not leak into a cell mean."""
    obs = _sample(trace, frame, nan_rows=40)
    assert np.all(np.isfinite(obs.ds["los"].values))
    assert obs.n > 100


def test_mostly_invalid_cells_are_discarded(trace, frame):
    strict = _sample(trace, frame, nan_rows=60, min_valid_fraction=0.95)
    loose = _sample(trace, frame, nan_rows=60, min_valid_fraction=0.05)
    assert strict.n <= loose.n


# -- container ---------------------------------------------------------------

def test_concat_weighting_modes(trace, frame):
    obs = _sample(trace, frame)
    before = obs.ds["weight"].values.copy()

    for mode, expected in (("none", 1.0),
                           ("sqrt_count", 1.0 / np.sqrt(obs.n)),
                           ("count", 1.0 / obs.n)):
        merged = Observations.concat([obs, obs], normalize=mode)
        assert merged.n == 2 * obs.n
        np.testing.assert_allclose(merged.ds["weight"].values, expected, rtol=1e-12)

    # concat must not reach back into its inputs: Dataset.copy() is shallow, and
    # the same object appearing twice would otherwise be scaled twice.
    np.testing.assert_array_equal(obs.ds["weight"].values, before)

    with pytest.raises(ValueError, match="normalize must be"):
        Observations.concat([obs], normalize="magic")


def test_concat_applies_named_track_weights(trace, frame):
    obs = _sample(trace, frame)
    merged = Observations.concat([obs], normalize="none", weights={"track": 3.0})
    np.testing.assert_allclose(merged.ds["weight"].values, 3.0)


def test_concat_refuses_mixed_frames(trace, frame):
    from nisar_tools.slip import LocalFrame

    a = _sample(trace, frame)
    b = Observations.from_los(
        analytic_los_stack(trace, frame, spacing=2000.0),
        frame=LocalFrame(-67.0, 10.0), trace=trace,
        rms_min=0.008, width_min=2500.0, width_max=20000.0, exclude_within=3000.0,
    )
    with pytest.raises(ValueError, match="same LocalFrame"):
        Observations.concat([a, b])


def test_persist_round_trip(trace, frame, tmp_path):
    from nisar_tools import Workspace

    obs = _sample(trace, frame)
    ws = Workspace(tmp_path / "ws")
    stored = obs.persist(ws, "obs")
    assert stored.n == obs.n
    np.testing.assert_allclose(stored.ds["los"].values, obs.ds["los"].values)
    assert stored.tracks == obs.tracks
    assert stored.frame.matches(obs.frame)


def test_empty_result_is_an_error_not_an_empty_set(trace, frame):
    with pytest.raises(ValueError, match="removed every sample"):
        _sample(trace, frame, exclude_within=1e7)
