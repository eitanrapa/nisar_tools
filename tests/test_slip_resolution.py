"""Pins model resolution and the mesh and sampling it drives.

Variance reduction measures fit, and a slip model fits happily on a
discretisation the data never constrained -- the spare freedom is spent on noise.
Resolution measures the other thing, and these tests hold the two claims that
make it usable: that it behaves like a resolution (falls with the damping, and
falls with depth), and that the mesh built from it converges instead of running
away.

The runaway is the reason ``budget=`` exists, and it is pinned here rather than
described, because it was a real failure of the first working version: the loop
ran end to end, reported convergence, and returned a mesh of 44 elements with
0.74% variance reduction.
"""

import numpy as np
import pytest

from nisar_tools.slip import FaultTrace, FaultMesh, Observations, SlipInversion
from nisar_tools.slip import resolution as R

import slip_synthetic as ss

_LON = np.array([-69.0, -68.5, -68.0, -67.5, -67.0, -66.5, -66.0])
_LAT = np.array([10.55, 10.56, 10.57, 10.58, 10.585, 10.59, 10.60])


@pytest.fixture(scope="module")
def setup():
    trace = FaultTrace.from_lonlat(_LON, _LAT, name="t")
    frame = trace.local_frame()
    mesh = FaultMesh.vertical(trace, frame, max_depth=20e3, edge_length=10e3)
    slip = ss.tapered_slip(mesh, peak=-2.0)
    parts = [
        Observations.from_los(
            ss.forward_los_stack(mesh, slip, trace, frame, geometry=geometry,
                                 spacing=8000.0, noise=0.004, seed=1),
            name=geometry, frame=frame, trace=trace, rms_min=0.006,
            width_min=10000.0, width_max=60000.0, exclude_within=8000.0)
        for geometry in ("asc", "desc")
    ]
    obs = Observations.concat(parts)
    return trace, frame, mesh, SlipInversion(mesh, obs, ramp="offset")


# -- does it behave like a resolution? -------------------------------------
def test_resolution_falls_with_the_damping(setup):
    """The defining property: at zero damping the data decides everything."""
    _, _, _, inversion = setup
    ranks = [R.resolution_diagonal(inversion, w).attrs["effective_rank"]
             for w in (0.001, 0.1, 10.0, 1000.0)]

    assert ranks == sorted(ranks, reverse=True), ranks
    assert ranks[0] > 5.0 * ranks[-1]
    assert ranks[-1] < 1.0, "at heavy damping almost nothing is resolved"

    loose = R.resolution_diagonal(inversion, 1e-6)["resolution"].values
    tight = R.resolution_diagonal(inversion, 1e6)["resolution"].values
    assert loose.max() > 0.9
    assert tight.max() < 0.01
    for values in (loose, tight):
        np.testing.assert_array_less(-1e-9, values)
        np.testing.assert_array_less(values, 1.0 + 1e-9)


def test_shallow_elements_resolve_better_than_deep_ones(setup):
    """Surface data sees a shallow patch sharply and a deep one hardly at all."""
    _, _, mesh, inversion = setup
    per_element = R.element_resolution(inversion, R.resolution_diagonal(inversion, 0.3))
    depth = -mesh.centroids[:, 2]

    shallow = per_element[depth < 0.3 * depth.max()].mean()
    deep = per_element[depth > 0.7 * depth.max()].mean()
    assert shallow > 10.0 * deep, f"shallow {shallow:.4f} vs deep {deep:.4f}"


def test_the_ramp_columns_are_resolved_because_nothing_damps_them(setup):
    _, _, _, inversion = setup
    ramp = R.resolution_diagonal(inversion, 100.0).attrs["ramp_resolution"]

    assert len(ramp) == inversion.n_ramp
    np.testing.assert_array_less(0.9, np.asarray(ramp))


# -- the size field --------------------------------------------------------
def test_a_flat_resolution_field_asks_for_the_mesh_it_was_given(setup):
    """The identity the budget arithmetic has to satisfy, or every mesh drifts."""
    _, _, mesh, _ = setup
    size = R.target_element_size(mesh, np.full(mesh.n_elements, 0.5))

    np.testing.assert_allclose(size, R.element_size(mesh), rtol=1e-12)


def test_the_budget_sets_the_element_count_and_resolution_sets_the_places(setup):
    _, _, mesh, inversion = setup
    per_element = R.element_resolution(inversion, R.resolution_diagonal(inversion, 0.3))

    size = R.target_element_size(mesh, per_element)
    for factor in (0.5, 1.0, 2.0):
        wanted = int(factor * mesh.n_elements)
        got = R.adapt_layout(mesh, size, budget=wanted).n_elements
        # The 2:1 balance rule and integer node counts move it a little; the
        # point is that it tracks the request instead of drifting on its own.
        assert 0.8 * wanted <= got <= 1.25 * wanted, (factor, wanted, got)


def test_an_unbudgeted_step_drifts_and_a_budgeted_one_does_not(setup):
    """The failure ``budget=`` exists for, as a regression test.

    Targeting the median looks self-normalising and is not: most elements sit
    near zero resolution and hit the growth cap while few can shrink. Left to
    itself the loop coarsened 330 elements to 44 over three rounds while variance
    reduction fell to 0.74% -- converging, and returning nothing.
    """
    _, _, mesh, inversion = setup
    per_element = R.element_resolution(inversion, R.resolution_diagonal(inversion, 0.3))
    size = R.target_element_size(mesh, per_element)

    assert size.mean() > 1.2 * R.element_size(mesh).mean(), "the size field does drift"
    free = R.adapt_layout(mesh, size, budget=None).n_elements
    held = R.adapt_layout(mesh, size, budget="conserve").n_elements

    assert abs(free - mesh.n_elements) > 0.25 * mesh.n_elements, free
    assert abs(held - mesh.n_elements) <= 0.25 * mesh.n_elements, held


def test_max_ratio_bounds_a_single_step_even_where_resolution_underflows(setup):
    """Deep resolution is not merely small, it is exactly zero.

    ``target / r`` is then whatever the floor makes it and ``gain`` cannot temper
    an infinity, so the bound has to be on the ratio itself.
    """
    _, _, mesh, _ = setup
    resolution = np.zeros(mesh.n_elements)
    resolution[0] = 1.0

    ratio = R.target_element_size(mesh, resolution, target=1.0,
                                  max_ratio=2.0) / R.element_size(mesh)
    np.testing.assert_array_less(ratio, 2.0 + 1e-9)
    np.testing.assert_array_less(0.5 - 1e-9, ratio)


def test_smoothing_the_size_field_keeps_it_lipschitz(setup):
    _, _, mesh, _ = setup
    spike = np.full(mesh.n_elements, 5e3)
    spike[mesh.n_elements // 2] = 50e3

    smoothed = R.smooth_over_mesh(mesh, spike, rounds=3)
    assert smoothed.max() < 0.5 * spike.max()
    np.testing.assert_allclose(smoothed.min(), 5e3, rtol=0.5)


# -- the adapted layout ----------------------------------------------------
def test_the_adapted_layout_is_finer_where_the_data_resolves_better(setup):
    trace, frame, mesh, inversion = setup
    per_element = R.element_resolution(inversion, R.resolution_diagonal(inversion, 0.3))
    layout = R.adapt_layout(mesh, R.target_element_size(mesh, per_element))

    assert layout.counts is not None
    assert layout.counts[0] > layout.counts[-1], "the surface must be denser than the base"
    # Levels thicken downward, the same thing bias_w does by hand.
    thickness = np.diff(layout.levels)
    assert thickness[-1] > thickness[0]

    built = FaultMesh.from_layout(layout, trace, frame, uniform_dip=90.0,
                                  edge_length=10e3)
    built._check_winding(trace)
    for side in ("top", "bottom", "left", "right"):
        assert built.boundary_elements(side).any()


def test_the_adapted_mesh_spans_exactly_the_same_fault(setup):
    """A re-mesh must not quietly change the fault's extent."""
    _, _, mesh, inversion = setup
    per_element = R.element_resolution(inversion, R.resolution_diagonal(inversion, 0.3))
    layout = R.adapt_layout(mesh, R.target_element_size(mesh, per_element))

    np.testing.assert_allclose(layout.params[:, 0].max(), mesh.params[:, 0].max())
    np.testing.assert_allclose(layout.params[:, 1].min(), mesh.params[:, 1].min())
    assert layout.params[:, 0].min() == 0.0


def test_the_levels_never_jump_by_more_than_the_balance_factor(setup):
    _, _, mesh, _ = setup
    # A size field that saturates at depth is what produced a 17 km level under
    # a 2.5 km one before `_balance_levels` existed.
    depth = -mesh.centroids[:, 2]
    size = 1e3 + 40.0 * depth
    layout = R.adapt_layout(mesh, size, balance=2.0)

    thickness = np.diff(layout.levels)
    ratio = thickness[1:] / thickness[:-1]
    np.testing.assert_array_less(np.maximum(ratio, 1.0 / ratio), 2.0 + 1e-6)


# -- the cheap initialiser and the sampling field --------------------------
def test_the_column_norm_is_not_a_resolution_proxy(setup):
    """Why there is no cheap stand-in for ``R``, pinned so nobody adds one back.

    ``||G[:, k]||`` is the obvious way to avoid a factorisation and it is blind to
    the thing that matters: a deeper element radiates a broader but weaker pattern
    whose norm is roughly conserved, while resolution collapses because deep
    elements trade off with each other. A column norm cannot see a trade-off.
    """
    _, _, mesh, inversion = setup
    n = inversion.basis.n_basis
    norms = np.linalg.norm(inversion.g[:, :2 * n] * inversion.w[:, None], axis=0)
    column = np.maximum(norms[:n], norms[n:2 * n])
    proper = R.element_resolution(inversion, R.resolution_diagonal(inversion, 0.3))
    depth = -mesh.centroids[:, 2]

    assert np.corrcoef(proper, depth)[0, 1] < -0.5, "resolution must fall with depth"
    assert abs(np.corrcoef(column, depth)[0, 1]) < 0.2, "the column norm does not"

    deep = depth > 0.7 * depth.max()
    shallow = depth < 0.3 * depth.max()
    assert proper[shallow].mean() > 100.0 * proper[deep].mean()
    assert 0.5 < column[shallow].mean() / column[deep].mean() < 2.0


def test_resolution_gain_points_at_the_unresolved_part_of_the_fault(setup):
    """Lohman & Simons' criterion, as a field the existing quadtree can split on."""
    trace, frame, mesh, inversion = setup
    model = inversion.solve(smoothing=0.3, polarity=(-1, 0, 0))
    stack = ss.forward_los_stack(mesh, ss.tapered_slip(mesh, peak=-2.0), trace, frame,
                                 geometry="asc", spacing=8000.0, seed=2)

    resolution = R.resolution_diagonal(inversion, 0.3)
    # Declare the western third unresolved and everything else perfect, so the
    # gain has to follow the fault rather than the scene's own structure.
    values = np.ones(resolution.sizes["basis"])
    values[mesh.element_params[:, 0] < 0.35 * mesh.params[:, 0].max()] = 0.0
    resolution["resolution"].values[:] = values

    gain = R.resolution_gain(model, stack, resolution, frame=frame, spacing=8000.0)

    x = np.asarray(stack.ds["x"].values, dtype=float)
    assert gain.shape == (stack.ds["y"].size, x.size)
    columns = np.broadcast_to(x, gain.shape)
    # The frame's x runs west to east, and so does arc length along this trace.
    west = np.nanmean(np.where(columns < np.median(x), gain, np.nan))
    east = np.nanmean(np.where(columns > np.median(x), gain, np.nan))
    assert west > east, f"gain west {west:.3f} vs east {east:.3f}"


def test_a_wrongly_shaped_resolution_vector_is_refused_by_name(setup):
    _, _, mesh, _ = setup
    with pytest.raises(ValueError, match="element_resolution"):
        R.target_element_size(mesh, np.ones(3))
