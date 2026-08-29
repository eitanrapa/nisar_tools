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


# -- unstructured adaptation, where the depth levels stop being straight ------
#
# A row-structured layout reads its depth levels off ONE vertical, at the fault's
# midpoint, because a level is a single depth for the whole fault. When the size
# field varies along strike -- which is what uneven coverage along the fault
# produces -- that constraint is what limits how well the mesh can follow it.

def _along_strike_size_field(mesh, fine=3e3, coarse=9e3):
    """A target size that is fine in the west and coarse in the east.

    Deliberately *not* from a solved inversion: the property under test is what
    the builders do with a size field that varies along strike, and constructing
    it directly keeps the test fast and its intent legible.
    """
    s = mesh.element_params[:, 0]
    fraction = (s - s.min()) / max(np.ptp(s), 1.0)
    return fine + (coarse - fine) * fraction


def _follows(layout, want_at):
    """Median |relative error| of the achieved element size, global scale removed.

    The scale is divided out because the element budget sets it -- what is being
    measured is whether the *shape* of the size field survived.
    """
    from nisar_tools.slip.layout import signed_areas

    centre = layout.params[layout.triangles].mean(axis=1)
    got = np.sqrt(np.abs(signed_areas(layout.params, layout.triangles)))
    want = want_at(centre[:, 0], -centre[:, 1])
    return float(np.median(np.abs(got / (want * np.median(got / want)) - 1)))


def test_unstructured_follows_an_along_strike_size_field_better_than_rows(setup):
    """The measurement the ``structure="unstructured"`` mode exists for."""
    _, _, mesh, _ = setup
    sizes = _along_strike_size_field(mesh)
    want_at = R.size_field(mesh, sizes)

    rows = R.adapt_layout(mesh, sizes, budget="conserve", structure="rows")
    free = R.adapt_layout(mesh, sizes, budget="conserve", structure="unstructured")

    assert rows.counts is not None
    assert free.counts is None, "an unstructured layout has no depth levels"
    assert _follows(free, want_at) < _follows(rows, want_at)


def test_unstructured_respects_the_budget_and_the_boundaries(setup):
    _, _, mesh, _ = setup
    sizes = _along_strike_size_field(mesh)
    for budget in ("conserve", 120):
        layout = R.adapt_layout(mesh, sizes, budget=budget,
                                structure="unstructured")
        wanted = mesh.n_elements if budget == "conserve" else budget
        assert abs(layout.n_elements - wanted) <= 0.2 * wanted
        for side in ("top", "bottom", "left", "right"):
            from nisar_tools.slip.layout import boundary_node_mask
            assert int(boundary_node_mask(layout.params, side).sum()) >= 2


def test_unstructured_records_its_provenance_and_quality(setup):
    _, _, mesh, _ = setup
    layout = R.adapt_layout(mesh, _along_strike_size_field(mesh),
                            structure="unstructured")
    assert layout.attrs["layout"] == "adapted_unstructured"
    assert layout.attrs["quality"]["n_elements"] == layout.n_elements
    assert layout.attrs["quality"]["min_angle_deg"] > R.MIN_ANGLE_WARN_DEG


def test_an_unrelaxed_cloud_is_flagged_as_slivered(setup):
    """The adaptive loop runs unattended, so a degenerate triangulation has to
    say so; ``iterations=0`` is the rejection-sampled cloud, which is 0.4 deg."""
    _, _, mesh, _ = setup
    with pytest.warns(RuntimeWarning, match="minimum triangle angle"):
        R.adapt_layout(mesh, _along_strike_size_field(mesh), budget=None,
                       structure="unstructured", iterations=0)


def test_the_rows_structure_is_unchanged(setup):
    """The default must still produce exactly what it did before the mode split,
    or every existing run's mesh moves."""
    _, _, mesh, _ = setup
    sizes = _along_strike_size_field(mesh)
    default = R.adapt_layout(mesh, sizes, budget="conserve")
    explicit = R.adapt_layout(mesh, sizes, budget="conserve", structure="rows")
    np.testing.assert_array_equal(default.params, explicit.params)
    assert default.attrs["layout"] == "adapted"


def test_an_unknown_structure_is_refused(setup):
    _, _, mesh, _ = setup
    with pytest.raises(ValueError, match="structure must be"):
        R.adapt_layout(mesh, _along_strike_size_field(mesh),
                       budget=None, structure="quadtree")


def test_the_size_field_keeps_the_shape_it_was_asked_on(setup):
    """The row builder asks along a line and the unstructured one on a 2-D probe
    grid; a lookup that only handled 1-D broke the second with a broadcast error."""
    _, _, mesh, _ = setup
    at = R.size_field(mesh, _along_strike_size_field(mesh))
    line = at(np.linspace(0.0, 40e3, 7), np.full(7, 5e3))
    assert line.shape == (7,)
    grid_s, grid_d = np.meshgrid(np.linspace(0.0, 40e3, 5),
                                 np.linspace(0.0, 15e3, 4))
    assert at(grid_s, grid_d).shape == (4, 5)


# -- the full model/data resolution pair --------------------------------------
#
# The diagonal answers "how much of this estimate is data". It cannot answer
# "over what distance", which is the question that decides whether a mesh is
# finer than the data supports -- for that you need the off-diagonal structure.

def test_the_two_traces_are_the_same_number(setup):
    """``trace(A M^-1 A') == trace(M^-1 A'A)`` identically, so model and data
    resolution must agree on the effective parameter count. It is the one exact
    check available on either of them, and it catches the two halves being built
    from different factorizations."""
    _, _, _, inv = setup
    report = R.resolution_report(inv, 1.0)
    assert report.attrs["trace_identity_error"] < 1e-8
    assert report.attrs["trace_model_resolution"] == pytest.approx(
        float(R.data_importance(inv, 1.0).sum()), rel=1e-9)


def test_the_matrix_diagonal_is_the_diagonal_function(setup):
    _, _, _, inv = setup
    matrix = R.resolution_matrix(inv, 1.0)
    diag = R.resolution_diagonal(inv, 1.0)
    n = inv.basis.n_basis
    np.testing.assert_allclose(np.diag(matrix)[:n], diag["strike_slip"].values,
                               rtol=1e-10, atol=1e-12)


def test_kernel_width_survives_a_smoothing_change_where_the_diagonal_does_not(setup):
    """The measurement behind ``resolution_report``'s advice, and the reason the
    adaptive loop misbehaves: ``target_element_size`` sizes the mesh from the
    diagonal, which is mostly a function of lambda, rather than from the kernel
    width, which is set by the data geometry."""
    _, _, _, inv = setup
    widths, diagonals = [], []
    for lam in (0.1, 10.0):
        report = R.resolution_report(inv, lam)
        widths.append(report.attrs["median_kernel_width_m"])
        diagonals.append(float(np.median(report["resolution"].values)))
    width_ratio = max(widths) / min(widths)
    diagonal_ratio = max(diagonals) / min(diagonals)
    assert diagonal_ratio > 3.0, diagonal_ratio       # lambda moves the diagonal
    assert width_ratio < 1.5, width_ratio             # ...and barely the width
    assert width_ratio < diagonal_ratio


def test_the_kernel_is_wider_than_the_elements_it_averages(setup):
    """A mesh cannot claim more detail than its own averaging kernels carry."""
    _, _, mesh, inv = setup
    report = R.resolution_report(inv, 1.0)
    assert report.attrs["over_discretisation"] > 1.0
    assert report.attrs["median_kernel_width_m"] > np.median(R.element_size(mesh))


def test_data_importance_is_a_diagonal_of_a_projection(setup):
    """``N`` is a projection-like operator, so its diagonal lies in [0, 1] and
    sums to the effective rank -- a datum near 1 is one nothing else replaces."""
    _, _, _, inv = setup
    importance = R.data_importance(inv, 1.0)
    assert importance.shape == (inv.obs.n,)
    assert importance.min() >= -1e-12
    assert importance.max() <= 1.0 + 1e-9
    assert importance.sum() == pytest.approx(
        R.resolution_report(inv, 1.0).attrs["effective_rank"]
        + sum(R.resolution_diagonal(inv, 1.0).attrs["ramp_resolution"]), rel=1e-8)


def test_more_smoothing_resolves_fewer_parameters_and_less_data(setup):
    _, _, _, inv = setup
    ranks = [float(R.data_importance(inv, lam).sum()) for lam in (0.1, 1.0, 10.0)]
    assert ranks[0] > ranks[1] > ranks[2]


def test_kernels_refuse_a_matrix_from_a_different_basis(setup):
    _, _, mesh, inv = setup
    matrix = R.resolution_matrix(inv, 1.0)
    wrong = SlipInversion(mesh, inv.obs, basis="element"
                          if inv.basis.name == "node" else "node")
    with pytest.raises(ValueError, match="parameters"):
        R.resolution_kernels(wrong, matrix)


# -- sizing a mesh from the kernel width instead of the diagonal --------------

def test_the_diagonal_size_field_saturates_where_the_kernel_one_does_not(setup):
    """The reason the kernel criterion exists, and it is *not* lambda drift.

    ``target_element_size`` forms ``(target / r) ** gain`` on a diagonal that has
    underflowed over most of the fault, so it is a ratio of two nearly-zero
    numbers and runs to its clamp: measured here, a large fraction of elements
    are pinned at ``max_ratio`` and carry no information at all. The kernel width
    is a real length everywhere.
    """
    _, _, mesh, inv = setup
    per_el = R.element_resolution(inv, R.resolution_diagonal(inv, 1.0))
    ratio = (np.median(per_el) / np.maximum(per_el, 1e-12)) ** 0.5
    pinned = (ratio >= 3.0 - 1e-9) | (ratio <= 1 / 3 + 1e-9)
    assert pinned.mean() > 0.3, f"only {100 * pinned.mean():.0f}% pinned"

    width = R.kernel_element_size(inv, 1.0, size_min=1.0, size_max=1e9)
    assert np.all(np.isfinite(width)) and width.min() > 0.0
    # ...and it is not a constant either: it still varies across the mesh.
    assert np.ptp(width) > 0.05 * np.median(width)


def test_kernel_element_size_never_degenerates(setup):
    """Where resolution underflows, ``target / r`` saturates its clamp; the
    kernel width stays a real length everywhere, which is the whole point."""
    _, _, _, inv = setup
    for lam in (0.1, 1.0, 100.0):
        size = R.kernel_element_size(inv, lam)
        assert np.all(np.isfinite(size))
        assert size.min() > 0.0


def test_kernel_reduction_ordering(setup):
    """``min`` keeps the better-resolved direction and so asks for the finest
    mesh; ``max`` discards it. Their order cannot silently swap."""
    _, _, _, inv = setup
    report = R.resolution_report(inv, 1.0)
    sizes = {r: R.kernel_element_size(inv, 1.0, reduce=r, report=report,
                                      size_min=1.0, size_max=1e9)
             for r in R.KERNEL_REDUCTIONS}
    assert np.median(sizes["min"]) <= np.median(sizes["geometric"])
    assert np.median(sizes["geometric"]) <= np.median(sizes["max"])


def test_kernel_element_size_rejects_an_unknown_reduction(setup):
    _, _, _, inv = setup
    with pytest.raises(ValueError, match="reduce must be one of"):
        R.kernel_element_size(inv, 1.0, reduce="rms")


def test_kernel_element_size_scales_with_fraction(setup):
    """``fraction`` is elements per kernel width, so it multiplies straight
    through -- there is no hidden normalisation."""
    _, _, _, inv = setup
    report = R.resolution_report(inv, 1.0)
    kw = dict(report=report, size_min=1.0, size_max=1e9)
    half = R.kernel_element_size(inv, 1.0, fraction=0.5, **kw)
    quarter = R.kernel_element_size(inv, 1.0, fraction=0.25, **kw)
    np.testing.assert_allclose(quarter, 0.5 * half, rtol=1e-12)
