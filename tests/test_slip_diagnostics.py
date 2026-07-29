"""Pins the measured-parameter helpers in :mod:`nisar_tools.slip.diagnostics`.

These exist because every sampling parameter used to be inherited from the
notebook example rather than derived from the scene, and each wrong value showed
up downstream as a symptom that looked like something else. The tests here are
mostly about *agreement with reality*: a predicted cell size that drifts from
what the quadtree actually does, or a noise floor that does not recover a planted
noise level, would be worse than having no helper at all.
"""

import numpy as np
import pytest
from slip_synthetic import analytic_los_stack

from nisar_tools.slip import FaultMesh, FaultTrace, Observations
from nisar_tools.slip import diagnostics as D

_LON = np.array([-68.681, -68.300, -67.900, -67.500, -67.100, -66.700, -66.523])
_LAT = np.array([10.410, 10.490, 10.540, 10.565, 10.595, 10.620, 10.630])


@pytest.fixture(scope="module")
def trace():
    return FaultTrace(_LON, _LAT, name="test_fault")


@pytest.fixture(scope="module")
def frame(trace):
    return trace.local_frame()


@pytest.fixture(scope="module")
def mesh(trace, frame):
    return FaultMesh.vertical(trace, frame, max_depth=20e3, edge_length=6e3)


# -- the cell-size ladder ----------------------------------------------------

@pytest.mark.parametrize("spacing", [1000.0, 1500.0])
@pytest.mark.parametrize("width_min", [1000.0, 1500.0, 2500.0, 4000.0])
def test_ladder_predicts_the_quadtree_terminal_cell(trace, frame, spacing, width_min):
    """The predicted terminal cell must equal what the sampler really produces.

    This is the test that matters. The dyadic ladder is not guessable -- cells
    are index rectangles halved at their midpoint, per axis, so ``width_min``
    only bites when it crosses a rung and identical-looking values can differ by
    2x in sample count. A prediction that drifts from ``_quadtree_cells`` would
    be actively misleading, so it is pinned against the real thing rather than
    against a formula.
    """
    stack = analytic_los_stack(trace, frame, spacing=spacing, noise=0.02)
    obs = Observations.from_los(stack, frame=frame, trace=trace, rms_min=0.001,
                                width_min=width_min, width_max=30000.0,
                                exclude_within=2000.0)

    predicted = D.cell_size_ladder(stack, width_min=width_min,
                                   width_max=30000.0)["terminal_cell"]
    assert float(obs.ds["cell_size"].values.min()) == pytest.approx(predicted)


def test_ladder_is_the_closure_of_midpoint_halving(trace, frame):
    """Every reachable extent must be a half of some other reachable extent.

    Not a single chain: an odd extent splits into two *different* sizes and both
    branches continue, so the reachable set is a tree's worth of values rather
    than repeated floor-halving. Following one branch is what made the first
    version overestimate the terminal cell by 25%.
    """
    stack = analytic_los_stack(trace, frame, spacing=1000.0)
    ladder = D.cell_size_ladder(stack)
    for key, dim in (("rows", "y"), ("cols", "x")):
        seq = ladder[key]
        assert seq.max() == stack.ds.sizes[dim]
        assert np.all(np.diff(seq) > 0)                    # sorted, unique
        reachable = set(seq.tolist())
        for value in reachable - {seq.max()}:
            assert any(p // 2 == value or p - p // 2 == value for p in reachable)


def test_nearby_width_min_values_can_share_a_rung(trace, frame):
    """Two different ``width_min`` values landing on one rung is the whole point."""
    stack = analytic_los_stack(trace, frame, spacing=1000.0)
    cells = {w: D.cell_size_ladder(stack, width_min=w)["terminal_cell"]
             for w in (1000.0, 1200.0, 1400.0, 1600.0, 2000.0, 3000.0, 5000.0)}
    assert len(set(cells.values())) < len(cells)


# -- noise floor -------------------------------------------------------------

@pytest.mark.parametrize("sigma", [0.004, 0.012])
def test_noise_floor_recovers_a_planted_level(trace, frame, sigma):
    stack = analytic_los_stack(trace, frame, spacing=500.0, noise=sigma)
    got = D.noise_floor(stack, trace, frame, block=2000.0, min_distance=40e3)
    assert got == pytest.approx(sigma, rel=0.15)


def test_noise_floor_is_biased_up_by_the_near_field(trace, frame):
    """Why ``min_distance`` exists: near the fault the scatter is signal.

    The analytic scene is an arctangent step across the trace, so blocks close to
    it contain a real gradient. Including them measures deformation and calls it
    noise, which would then set ``rms_min`` too high and throw away the very
    samples that resolve slip.
    """
    stack = analytic_los_stack(trace, frame, spacing=500.0, noise=0.006,
                               amplitude=0.4, width=4e3)
    near = D.noise_floor(stack, trace, frame, block=2000.0, min_distance=0.0)
    far = D.noise_floor(stack, trace, frame, block=2000.0, min_distance=40e3)
    assert near > far
    assert far == pytest.approx(0.006, rel=0.2)


def test_noise_floor_refuses_when_nothing_is_far_enough(trace, frame):
    stack = analytic_los_stack(trace, frame, spacing=2000.0)
    with pytest.raises(ValueError, match="No blocks survive"):
        D.noise_floor(stack, trace, frame, min_distance=10_000e3)


# -- scene report ------------------------------------------------------------

def test_scene_report_suggests_self_consistent_parameters(trace, frame, mesh):
    stack = analytic_los_stack(trace, frame, spacing=500.0, noise=0.009)
    rep = D.scene_report(stack, trace, frame, mesh=mesh, block=2000.0)

    a = rep.attrs
    assert a["rms_min"] == pytest.approx(D.RMS_MIN_MARGIN * a["noise_floor"])
    assert a["noise_floor"] == pytest.approx(0.009, rel=0.2)
    # A cell must not be able to straddle the trace.
    assert a["exclude_within"] >= 0.5 * a["terminal_cell"] - 1e-9
    # width_min has to land on a rung, not between them.
    ladder = D.cell_size_ladder(stack, width_min=a["width_min"])
    assert ladder["terminal_cell"] == pytest.approx(a["terminal_cell"])


def test_scene_report_detects_one_sided_coverage(trace, frame, mesh):
    """The failure a scalar coverage fraction hides.

    Blank everything on one side of the trace and the profile must say so along
    the whole strike, not merely report a smaller percentage.
    """
    stack = analytic_los_stack(trace, frame, spacing=1000.0, noise=0.008)
    tx, ty = trace.to_local(frame)
    yy = stack.ds["y"].values
    xx = stack.ds["x"].values
    fx, fy = frame.from_epsg(*np.meshgrid(xx, yy), int(stack.ds.attrs["epsg"]))
    north = trace.side(fx, fy, frame).reshape(fy.shape) > 0

    both = D.scene_report(stack, trace, frame, mesh=mesh)
    stack.ds["los"] = stack.ds["los"].where(~north)
    one = D.scene_report(stack, trace, frame, mesh=mesh)

    assert both.attrs["two_sided_fraction"] > 0.8
    assert one.attrs["two_sided_fraction"] < 0.2
    assert one.attrs["frac_left"] < both.attrs["frac_left"]
    assert float(one["valid_left"].sum()) < 0.05 * float(both["valid_left"].sum())

    # A coverage fraction that can exceed 1 is not a fraction: blocks are counted
    # whole while only their centres are tested against the band.
    for rep in (both, one):
        assert 0.0 <= rep.attrs["near_fault_coverage"] <= 1.0
    assert both.attrs["near_fault_coverage"] > 0.9
    assert one.attrs["near_fault_coverage"] == pytest.approx(0.5, abs=0.15)


def test_scene_report_checks_the_look_geometry(trace, frame, mesh):
    """The invariant that would have caught the inverted LOS sign."""
    stack = analytic_los_stack(trace, frame, spacing=1000.0, noise=0.008)
    stack.ds.attrs["direction"] = "Ascending"
    stack.ds.attrs["look_direction"] = "Left"

    rep = D.scene_report(stack, trace, frame, mesh=mesh, min_distance=40e3)
    east_sign = rep.attrs["los_east_sign"]
    assert rep.attrs["geometry_consistent"] == (east_sign > 0)

    # Same data, wrong label: the check must notice.
    stack.ds.attrs["direction"] = "Descending"
    flipped = D.scene_report(stack, trace, frame, mesh=mesh, min_distance=40e3)
    assert flipped.attrs["geometry_consistent"] != rep.attrs["geometry_consistent"]


def test_scene_report_geometry_check_is_none_without_attributes(trace, frame, mesh):
    stack = analytic_los_stack(trace, frame, spacing=1000.0)
    stack.ds.attrs.pop("direction", None)
    rep = D.scene_report(stack, trace, frame, mesh=mesh, min_distance=40e3)
    assert rep.attrs["geometry_consistent"] is None


def test_blocks_are_clamped_to_at_least_two_pixels(trace, frame):
    """A one-pixel block has no variance; ask for one and get two."""
    stack = analytic_los_stack(trace, frame, spacing=2000.0, noise=0.01)
    b = D._blocks(stack, trace, frame, block=1000.0)
    assert b["block_size"] == pytest.approx(4000.0)
    assert np.isfinite(b["std"]).any()


# -- ramp content ------------------------------------------------------------

def test_ramp_content_separates_an_offset_from_a_gradient(trace, frame):
    """An offset must show up in ``offset``; a gradient only in ``linear``."""
    stack = analytic_los_stack(trace, frame, spacing=1000.0, noise=0.002)
    obs = Observations.concat([
        Observations.from_los(stack, name="t", frame=frame, trace=trace,
                              rms_min=0.01, width_min=4000.0, width_max=30000.0,
                              exclude_within=3000.0)
    ], normalize="sqrt_count")

    base = np.zeros(obs.n)
    span = np.ptp(obs.ds["x"].values)

    obs.ds["los"] = ("obs", base + 0.25)                     # pure offset
    offset_only = D.ramp_content(obs)
    assert offset_only["offset"]["variance_reduction"] > 99.0
    assert offset_only["gradient_only"] < 1.0

    x = obs.ds["x"].values
    obs.ds["los"] = ("obs", 0.4 * (x - x.mean()) / span)     # pure gradient
    gradient = D.ramp_content(obs)
    assert gradient["offset"]["variance_reduction"] < 20.0
    assert gradient["linear"]["variance_reduction"] > 99.0
    assert gradient["gradient_only"] > 80.0
    # Columns are span-normalised, so the coefficient reads as metres of LOS
    # across the scene -- which is what makes "centimetres = orbit" a usable rule.
    assert abs(gradient["linear"]["coefficients"]["t:dx"]) == pytest.approx(0.4, rel=0.1)
