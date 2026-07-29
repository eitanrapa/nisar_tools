"""The curved-geometry machinery: the gridder, the surface, the dipping mesh.

The physics needs no new oracle here -- ``test_slip_tde.py`` already checks the
triangular dislocation against Okada at dips of 90, 85, 80, 70, 45, 30, 110 and
135 degrees, so a non-vertical element is already known to be right. What is new
is *geometry*: where the nodes go, which way the elements face, and whether a
fitted surface can quietly wander into the numerically hostile band beside
vertical.
"""

import numpy as np
import pytest

from nisar_tools.slip import FaultMesh, FaultSegment, FaultSurface, FaultTrace, gridfit
from nisar_tools.slip.mesh import _depth_levels, _snap_vertical_columns
from nisar_tools.slip.surface import centres
from nisar_tools.slip.trace import VERTICAL_TOL_DEG, dip_offset

# The real San Sebastian trace, decimated -- nearly east-west, the orientation
# that stresses winding and the reference's fixed-axis orientation test.
_LON = np.array([-68.681, -68.300, -67.900, -67.500, -67.100, -66.700, -66.523])
_LAT = np.array([10.410, 10.490, 10.540, 10.565, 10.595, 10.620, 10.630])


@pytest.fixture(scope="module")
def setup():
    trace = FaultTrace(_LON, _LAT, name="test_fault")
    return trace, trace.local_frame()


# -- the gridder -------------------------------------------------------------

def test_gridfit_is_exact_on_a_plane():
    """The strongest single check on the port.

    A plane is in the span of the triangle interpolant *and* in the null space of
    the second-difference regularizer, so an exact answer requires the design
    rows, the stencil offsets and the column ordering all to be right at once.
    Any transposition of the ``(ny, nx)`` layout, or an off-by-one in the
    ``ind + ny`` neighbour, breaks it immediately.
    """
    rng = np.random.default_rng(0)
    x_nodes, y_nodes = np.linspace(0, 10, 21), np.linspace(0, 5, 13)
    x, y = rng.uniform(0, 10, 2000), rng.uniform(0, 5, 2000)
    plane = lambda a, b: 3.0 - 1.5 * a + 2.0 * b  # noqa: E731

    fitted = gridfit(x, y, plane(x, y), x_nodes, y_nodes, smoothness=1.0)
    xx, yy = np.meshgrid(x_nodes, y_nodes)
    assert fitted.shape == (y_nodes.size, x_nodes.size)
    np.testing.assert_allclose(fitted, plane(xx, yy), atol=1e-7)


def test_gridfit_degenerates_to_a_plane_at_high_smoothness():
    rng = np.random.default_rng(1)
    x_nodes, y_nodes = np.linspace(0, 10, 21), np.linspace(0, 5, 13)
    x, y = rng.uniform(0, 10, 1500), rng.uniform(0, 5, 1500)
    v = np.sin(x) * np.cos(y)

    fitted = gridfit(x, y, v, x_nodes, y_nodes, smoothness=1e6)
    second = max(np.abs(np.diff(fitted, 2, axis=0)).max(),
                 np.abs(np.diff(fitted, 2, axis=1)).max())
    assert second < 1e-6, "an infinitely stiff surface must have no curvature left"


def test_gridfit_converges_to_the_interpolant_limit():
    """Low smoothness must approach the piecewise-linear representation error.

    Guards against a fit that is systematically biased rather than merely
    smoothed: the error has to stop falling at the interpolant's own limit, not
    at some larger floor of the fitting machinery.
    """
    rng = np.random.default_rng(2)
    x_nodes, y_nodes = np.linspace(0, 10, 41), np.linspace(0, 5, 21)
    x, y = rng.uniform(0, 10, 6000), rng.uniform(0, 5, 6000)
    f = lambda a, b: 2.0 + 0.5 * a - 0.3 * b + 0.02 * a ** 2  # noqa: E731
    xx, yy = np.meshgrid(x_nodes, y_nodes)

    errors = [np.abs(gridfit(x, y, f(x, y), x_nodes, y_nodes, smoothness=s)
                     - f(xx, yy)).max() for s in (1.0, 1e-2, 1e-4)]
    assert errors[0] > errors[-1]
    # dx^2 f'' / 8 for the quadratic term, with room for the fit.
    assert errors[-1] < 5e-3


def test_gradient_regularizer_is_not_the_laplacian():
    """The two differ by whether the directions are stacked or summed.

    ``diffusion`` penalises only the sum of the two second differences, so
    curvature along strike can pay for curvature down dip. On this problem, where
    the regularizer *is* the dip profile, that is not a cosmetic difference --
    hence the reference's choice of ``gradient``, and hence this test.
    """
    from nisar_tools.slip.surface import _regularizer

    dx, dy = np.full(9, 1.0), np.full(5, 1.0)
    grad = _regularizer(10, 6, dx, dy, 1.0, 1.0, "gradient")
    diff = _regularizer(10, 6, dx, dy, 1.0, 1.0, "diffusion")
    assert grad.shape == (2 * 60, 60)
    assert diff.shape == (60, 60)

    # Both annihilate a plane; only that is shared.
    xx, yy = np.meshgrid(np.arange(10.0), np.arange(6.0))
    plane = (1.0 + 2.0 * xx - 3.0 * yy).ravel(order="F")
    np.testing.assert_allclose(grad @ plane, 0.0, atol=1e-10)
    np.testing.assert_allclose(diff @ plane, 0.0, atol=1e-10)


def test_gridfit_rejects_unsorted_nodes():
    with pytest.raises(ValueError, match="strictly increasing"):
        gridfit([0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 2, 1], [0, 1, 2])


# -- down-dip projection -----------------------------------------------------

def test_dip_offset_snaps_near_vertical_to_exactly_zero():
    """The guard that keeps a fitted surface out of the catastrophic band.

    ``VERTICAL_TOL_DEG`` is a numerical tolerance, not a modelling preference:
    the triangular-dislocation solution's error against Okada is 2e-14 at exactly
    90 degrees and 1.9e+02 -- 190 times the signal -- a ten-thousandth of a degree
    away. See the constant's own docstring for the measured sweep.
    """
    assert dip_offset(20e3, 90.0) == 0.0
    assert dip_offset(20e3, 90.0 - VERTICAL_TOL_DEG / 2) == 0.0
    assert dip_offset(20e3, 90.0 + VERTICAL_TOL_DEG / 2) == 0.0
    # Just outside the tolerance the offset is real, and tiny -- 3.5 m over 20 km.
    assert 0 < dip_offset(20e3, 90.0 - 2 * VERTICAL_TOL_DEG) < 10.0


def test_segment_projection_matches_the_reference_formula():
    """Port check for ``project_segment_3d``: left normal, ``depth/tan(dip)``."""
    seg = FaultSegment(0.0, 0.0, 10e3, 0.0)          # due east
    x, y = seg.project(5e3, 60.0)
    offset = 5e3 / np.tan(np.radians(60.0))
    # Left normal of (+1, 0) is (0, +1), so the fault leans north.
    np.testing.assert_allclose(x, [0.0, 10e3])
    np.testing.assert_allclose(y, [offset, offset])


def test_dip_over_ninety_leans_the_other_way():
    """Dips above 90 are legal, and the reference's Myanmar run uses one.

    ``segmentDipDegrees = [75 75 70 80 85 90 100]`` -- so an overhanging segment
    is exercised, not hypothetical, and clamping into [0, 90] would silently
    change the geometry rather than reject it.
    """
    seg = FaultSegment(0.0, 0.0, 10e3, 0.0)
    _, north = seg.project(5e3, 80.0)
    _, south = seg.project(5e3, 100.0)
    assert north[0] > 0 and south[0] < 0
    np.testing.assert_allclose(north, -south)


def test_segment_file_round_trip(tmp_path):
    path = tmp_path / "Segment_001.txt"
    path.write_text("  1000.0 2000.0\n 3000.0 4000.0\n")
    seg = FaultSegment.from_file(path)
    assert (seg.x_begin, seg.y_begin, seg.x_end, seg.y_end) == (1000.0, 2000.0, 3000.0, 4000.0)
    assert seg.name == "Segment_001"

    (tmp_path / "short.txt").write_text("1 2 3\n")
    with pytest.raises(ValueError, match="needs four"):
        FaultSegment.from_file(tmp_path / "short.txt")


# -- the curvilinear frame ---------------------------------------------------

def test_curvilinear_round_trip(setup):
    """The two directions must agree on the trace the mesh actually uses.

    They are not exact inverses in general -- ``to_curvilinear`` measures
    perpendicular distance to the nearest *segment* while ``from_curvilinear``
    steps along the interpolated *vertex* normal, and at a bend those differ.
    That is why the mesh builds its curvilinear frame from the **resampled**
    trace: at the resampled spacing the normals turn slowly and the round trip is
    exact for most points and sub-element for the rest. Measured here the median
    error is 0 m on a resampled trace against 94 m on the raw 7-vertex one.
    """
    trace, frame = setup
    resampled = trace.resample(6e3, frame)
    x, y = resampled.to_local(frame)
    rng = np.random.default_rng(3)
    px = rng.uniform(x.min() + 20e3, x.max() - 20e3, 400)
    py = rng.uniform(y.min() - 8e3, y.max() + 8e3, 400)

    s, cross = resampled.to_curvilinear(px, py, frame)
    bx, by = resampled.from_curvilinear(s, cross, frame)
    error = np.hypot(bx - px, by - py)
    assert np.median(error) < 1e-6
    assert error.max() < 1000.0


def test_curvilinear_sign_matches_the_left_hand_normal(setup):
    trace, frame = setup
    x, y = trace.to_local(frame)
    mid = (x.size - 1) // 2
    nx, ny = trace.normals(frame)
    probe_x, probe_y = x[mid] + 5e3 * nx[mid], y[mid] + 5e3 * ny[mid]

    _, cross = trace.to_curvilinear(probe_x, probe_y, frame)
    assert cross[0] > 0, "positive cross must be the side normals() points to"
    assert trace.side(probe_x, probe_y, frame)[0] == 1


def test_min_curvature_radius_of_the_real_trace(setup):
    """The limit on how far the trace can be pushed down dip before it folds."""
    trace, frame = setup
    radius = trace.min_curvature_radius(frame)
    assert radius > 50e3
    # A straight trace has no curvature at all.
    straight = FaultTrace([-68.0, -67.0, -66.0], [10.0, 10.0, 10.0])
    assert straight.min_curvature_radius() > 1e6


# -- depth levels ------------------------------------------------------------

def test_depth_levels_bias():
    even = _depth_levels(0.0, 20e3, 6, 1.0)
    np.testing.assert_allclose(even, np.linspace(0, 20e3, 6))

    biased = _depth_levels(0.0, 20e3, 6, 1.15)
    assert biased[0] == 0.0
    np.testing.assert_allclose(biased[-1], 20e3)
    thickness = np.diff(biased)
    np.testing.assert_allclose(thickness[1:] / thickness[:-1], 1.15)
    assert thickness[0] < thickness[-1], "levels must thicken downward"


# -- the near-vertical snapper ----------------------------------------------

def test_snap_pulls_a_near_vertical_step_onto_exactly_vertical():
    depths = np.array([0.0, 3000.0, 6000.0])
    limit = 3000.0 * np.tan(np.radians(VERTICAL_TOL_DEG))
    raw = np.array([[0.0, 0.0], [limit / 2, 900.0], [limit / 2, 1800.0]])

    snapped = _snap_vertical_columns(raw, depths)
    assert snapped[1, 0] == snapped[0, 0], "a sub-tolerance step becomes exactly vertical"
    np.testing.assert_array_equal(snapped[:, 1], raw[:, 1]), "a real dip is untouched"


def test_no_element_lands_in_the_catastrophic_band(setup):
    """A surface that changes its sense of lean must cross vertical somewhere.

    Wherever it does, the elements there are not asked for by anybody -- they are
    an accident of the fit -- and without the snap they sit in the band where the
    triangular-dislocation solution loses every digit.
    """
    trace, frame = setup
    segments = FaultSegment.from_trace(trace, frame, 3)
    mesh = FaultMesh.curved(trace, frame, segments=segments, dips=[60.0, 90.0, 120.0],
                            max_depth=20e3, edge_length=6e3)
    offenders = np.abs(mesh.dip - 90.0) < VERTICAL_TOL_DEG
    assert not np.any(offenders & (mesh.dip != 90.0)), (
        "elements inside the tolerance must be exactly vertical, not nearly so"
    )


# -- the curved mesh ---------------------------------------------------------

def test_curved_at_ninety_is_bit_identical_to_vertical(setup):
    """The backward-compatibility gate.

    Not "agrees to a tolerance": identical, so every persisted phase-one mesh,
    params hash and stored model stays valid. This is why a uniform dip is
    projected in closed form instead of through the gridder, and why the offset
    at exactly 90 is a literal zero rather than ``depth / tan(pi/2)`` -- which in
    float64 is 1.2e-12 m, not 0.
    """
    trace, frame = setup
    plain = FaultMesh.vertical(trace, frame, max_depth=20e3, edge_length=6e3)
    curved = FaultMesh.curved(trace, frame, uniform_dip=90.0,
                              max_depth=20e3, edge_length=6e3)

    np.testing.assert_array_equal(curved.nodes, plain.nodes)
    np.testing.assert_array_equal(curved.triangles, plain.triangles)
    np.testing.assert_array_equal(curved.params, plain.params)
    assert curved.digest() == plain.digest()


@pytest.mark.parametrize("dip", [80.0, 70.0, 60.0, 45.0])
def test_uniform_dip_geometry(setup, dip):
    trace, frame = setup
    mesh = FaultMesh.curved(trace, frame, uniform_dip=dip,
                            max_depth=20e3, edge_length=6e3)

    # Dip is recovered; the excess is the trace's own curvature, which tilts an
    # element slightly out of the nominal plane. Measured over 45-80 degrees the
    # spread is 0.006 to 0.017 degrees, entirely at the bends.
    assert abs(mesh.dip.min() - dip) < 1e-9
    assert mesh.dip.max() < dip + 0.1
    # Area is the vertical fault's, stretched by the down-dip lengthening.
    expected = trace.length(frame) * 20e3 / np.sin(np.radians(dip))
    assert 0.98 < mesh.areas.sum() / expected < 1.06
    # Every node stays below the free surface and on the same side.
    assert mesh.nodes[:, 2].max() == 0.0
    assert mesh.has_centres()


def test_boundary_elements_still_flag_whole_rows_on_a_curved_mesh(setup):
    """The silent-failure risk that the (s, z) parameterization is chosen to avoid.

    ``boundary_elements`` selects on exact float equality of the parameters. That
    is safe only because a fitted surface moves a node in map view alone -- its
    parameter depth stays the literal level value shared by the whole row. If a
    builder ever computed depth per column, this would flag one node instead of
    the bottom row, ``zero_slip_boundary`` would emit two rows instead of two
    hundred, and slip would run off the bottom edge with no error at all.
    """
    trace, frame = setup
    mesh = FaultMesh.curved(trace, frame, uniform_dip=65.0,
                            max_depth=20e3, edge_length=6e3)
    n_along, n_down = mesh.attrs["n_along"], mesh.attrs["n_down"]

    # A whole row of quads along the bottom and top, a whole column at each end.
    for side in ("bottom", "top"):
        assert int(mesh.boundary_elements(side).sum()) == 2 * (n_along - 1)
    for side in ("left", "right"):
        assert int(mesh.boundary_elements(side).sum()) == 2 * (n_down - 1)


def test_dip_direction_distinguishes_the_two_leans(setup):
    """What ``dip`` throws away by taking an absolute value."""
    trace, frame = setup
    north = FaultMesh.curved(trace, frame, uniform_dip=70.0,
                             max_depth=20e3, edge_length=6e3)
    south = FaultMesh.curved(trace, frame, uniform_dip=110.0,
                             max_depth=20e3, edge_length=6e3)

    np.testing.assert_allclose(north.dip, south.dip, atol=0.2)
    separation = np.abs(np.median(north.dip_direction) - np.median(south.dip_direction))
    assert 170.0 < separation % 360.0 < 190.0


def test_segment_dips_bend_the_surface(setup):
    trace, frame = setup
    segments = FaultSegment.from_trace(trace, frame, 3)
    mesh = FaultMesh.curved(trace, frame, segments=segments, dips=[60.0, 75.0, 85.0],
                            max_depth=20e3, edge_length=6e3)

    assert mesh.attrs["kind"] == "curved"
    assert mesh.attrs["dips"] == [60.0, 75.0, 85.0]
    # The shallow-dipping end steps much further off the trace than the steep one.
    n_along = mesh.attrs["n_along"]
    bottom = mesh.nodes[-n_along:, :2]
    top = mesh.nodes[:n_along, :2]
    step = np.hypot(*(bottom - top).T)
    assert step[0] > step[-1] * 2, "60 degrees must step further than 85"


def test_curved_needs_exactly_one_geometry_description(setup):
    trace, frame = setup
    with pytest.raises(ValueError, match="exactly one of"):
        FaultMesh.curved(trace, frame)
    with pytest.raises(ValueError, match="exactly one of"):
        FaultMesh.curved(trace, frame, uniform_dip=70.0, segments=2, dips=[70.0, 80.0])


def test_folding_guard_refuses_a_hairpin():
    """An offset larger than the trace's radius of curvature folds the surface.

    The projected bottom line doubles back on itself, and the mesh built from it
    has inverted elements -- which would pass the winding check on one side and
    fail it on the other, with a message about winding rather than about the
    geometry that caused it.
    """
    angle = np.linspace(-np.pi / 2, np.pi / 2, 41)
    radius = 0.05                                   # ~5.5 km, degrees of latitude
    trace = FaultTrace(-68.0 + radius * np.cos(angle), 10.0 + radius * np.sin(angle))
    frame = trace.local_frame()

    with pytest.raises(ValueError, match="radius of curvature"):
        FaultMesh.curved(trace, frame, segments=1, dips=[30.0],
                         max_depth=20e3, edge_length=2e3)


def test_curved_mesh_survives_a_dataset_round_trip(setup):
    trace, frame = setup
    mesh = FaultMesh.curved(trace, frame, uniform_dip=70.0,
                            max_depth=20e3, edge_length=6e3)
    back = FaultMesh.from_dataset(mesh.to_dataset())

    np.testing.assert_allclose(back.nodes, mesh.nodes)
    np.testing.assert_allclose(back.params, mesh.params)
    np.testing.assert_allclose(back.centres, mesh.centres)
    np.testing.assert_allclose(back.centre_params, mesh.centre_params)
    assert back.digest() == mesh.digest()


def test_centre_grid_is_fitted_not_interpolated(setup):
    """The reference runs the fit twice rather than interpolating the node grid.

    Where the regularizer is doing the work -- which for this cloud is nearly
    everywhere, since only two depths are constrained -- the two answers differ,
    and the point sources belong on the fitted one.
    """
    trace, frame = setup
    x, y = trace.to_local(frame)
    s_nodes = np.linspace(0.0, np.hypot(*np.diff(np.column_stack([x, y]), axis=0).T).sum(), 21)
    z_nodes = np.linspace(-20e3, 0.0, 9)
    segments = FaultSegment.from_trace(trace, frame, 2)

    surface = FaultSurface.from_segments(trace, frame, segments, [55.0, 85.0],
                                         s_nodes=s_nodes, z_nodes=z_nodes)
    assert surface.cross_centres.shape == (z_nodes.size - 1, s_nodes.size - 1)
    np.testing.assert_allclose(surface.s_centres, centres(s_nodes))

    from scipy.interpolate import RegularGridInterpolator
    interpolated = RegularGridInterpolator(
        (surface.z_nodes, surface.s_nodes), surface.cross_nodes,
    )(np.stack(np.meshgrid(surface.s_centres, surface.z_centres)[::-1], axis=-1))
    assert np.abs(interpolated - surface.cross_centres).max() > 1.0


def test_vertical_surface_is_flat():
    surface = FaultSurface.vertical(np.linspace(0, 100e3, 11), np.linspace(-20e3, 0, 6))
    assert not np.any(surface.cross_nodes)
    assert not np.any(surface.cross_centres)
