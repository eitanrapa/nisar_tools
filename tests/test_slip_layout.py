"""Pins the freeform mesh layout: node placement, Delaunay, and the guards.

The load-bearing invariant is **winding**, as in ``test_slip_mesh.py``, but for a
harder case: an irregular node set has no quads, so the triangles come from
Delaunay, whose vertex order is arbitrary. The claim these tests exist to hold is
that orienting every simplex by its signed area *in the ``(s, z)`` parameter
plane* is equivalent to the lattice's own winding -- and therefore that a mesh
with a different number of nodes at each depth level carries the same dip-slip
sign as one without.

The second theme is **backward compatibility**. A uniform layout must reproduce
the classic lattice bit for bit, triangles and node coordinates alike, or every
persisted ``.slip.zip`` and workspace hash silently goes stale.
"""

import numpy as np
import pytest

from nisar_tools.slip import FaultTrace
from nisar_tools.slip import layout as L
from nisar_tools.slip.mesh import FaultMesh

# The same near-east-west trace the mesh tests use: the orientation that defeats
# a fixed-axis winding test, and so the one worth triangulating irregularly.
_LON = np.array([-68.681, -68.400, -68.000, -67.600, -67.200, -66.800, -66.523])
_LAT = np.array([10.410, 10.487, 10.541, 10.563, 10.593, 10.621, 10.630])


@pytest.fixture
def trace():
    return FaultTrace(_LON, _LAT, name="test_fault")


@pytest.fixture
def frame(trace):
    return trace.local_frame()


def _stations(trace, frame, spacing=3e3):
    x, y = trace.resample(spacing, frame).to_local(frame)
    return np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])


# -- winding ---------------------------------------------------------------
def test_the_lattice_split_is_clockwise_in_parameter_space():
    """The measured fact the whole Delaunay path rests on.

    ``PARAM_ORIENTATION`` is not a convention picked for tidiness; it is what
    ``_lattice_triangles`` happens to produce, and the 3-D normals that come out
    on the trace's left-hand side follow from it.
    """
    ns, nz = 6, 4
    s, z = np.meshgrid(np.arange(ns, dtype=float),
                       -np.arange(nz, dtype=float), indexing="xy")
    params = np.column_stack([s.ravel(), z.ravel()])
    areas = L.signed_areas(params, L._lattice_triangles(ns, nz))

    assert np.all(np.sign(areas) == L.PARAM_ORIENTATION)


def test_delaunay_reproduces_the_lattice_meshs_geometry(trace, frame):
    """Retriangulating a lattice's own nodes changes the diagonals, nothing else.

    The acceptance test for the winding rule. Delaunay picks different diagonals
    -- the four corners of a quad are cocircular, so its choice is arbitrary --
    but every normal must still land on the trace's left, the dips must match and
    the total area is a property of the node set, not the triangulation.
    """
    mesh = FaultMesh.vertical(trace, frame, max_depth=20e3, edge_length=5e3)
    triangles = L.triangulate(mesh.params)
    rebuilt = FaultMesh(mesh.nodes, triangles, mesh.params, frame=frame)

    assert rebuilt.n_elements == mesh.n_elements
    assert set(map(tuple, np.sort(triangles, axis=1))) != set(
        map(tuple, np.sort(mesh.triangles, axis=1))
    ), "a regular lattice is cocircular, so Delaunay should differ from the quad split"

    rebuilt._check_winding(trace)
    np.testing.assert_allclose(rebuilt.areas.sum(), mesh.areas.sum(), rtol=1e-12)
    np.testing.assert_allclose(rebuilt.dip, mesh.dip, atol=1e-9)
    np.testing.assert_array_less(0.0, np.abs(rebuilt.normals[:, 1]))


def test_orient_triangles_fixes_a_deliberately_reversed_triangulation():
    # Node (i, j) is index j * ns + i, so the second row is the deeper one.
    params = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, -1.0], [1.0, -1.0]])
    triangles = L._lattice_triangles(2, 2)
    flipped = triangles[:, [0, 2, 1]]

    np.testing.assert_array_equal(L.orient_triangles(params, flipped), triangles)
    # Idempotent: orienting an already-correct set changes nothing.
    np.testing.assert_array_equal(L.orient_triangles(params, triangles), triangles)


# -- backward compatibility ------------------------------------------------
def test_a_uniform_layout_is_the_classic_lattice_bit_for_bit(trace, frame):
    stations = _stations(trace, frame)
    depths = np.linspace(0.0, 20e3, 8)
    layout = L.graded_layout(stations[-1], depths, counts=stations.size,
                             stations=stations)

    assert layout.uniform_counts == (stations.size, depths.size)
    np.testing.assert_array_equal(layout.triangles,
                                  L._lattice_triangles(stations.size, depths.size))
    for j, sl in enumerate(layout.level_slices()):
        # Bit for bit, not to a tolerance: `np.interp` returns its own knots
        # exactly, which is what leaves `FaultMesh.digest()` unchanged.
        np.testing.assert_array_equal(layout.params[sl, 0], stations)
        np.testing.assert_array_equal(layout.params[sl, 1], -depths[j])


def test_curved_with_the_new_knobs_defaults_to_the_old_mesh(trace, frame):
    """The gate: adding the layout arguments must not move an existing mesh."""
    plain = FaultMesh.curved(trace, frame, uniform_dip=75.0,
                             max_depth=20e3, edge_length=4e3)
    explicit = FaultMesh.curved(
        trace, frame, uniform_dip=75.0, max_depth=20e3, edge_length=4e3,
        nodes_per_level=plain.attrs["n_along"],
    )
    assert explicit.digest() == plain.digest()
    assert FaultMesh.vertical(trace, frame, max_depth=20e3, edge_length=4e3).digest() \
        == FaultMesh.curved(trace, frame, uniform_dip=90.0,
                            max_depth=20e3, edge_length=4e3).digest()


# -- irregular layouts -----------------------------------------------------
def test_a_layout_with_a_different_count_at_every_level_builds_and_winds(trace, frame):
    counts = [40, 34, 28, 22, 16, 12]
    depths = np.array([0.0, 1.5e3, 4e3, 8e3, 14e3, 22e3])
    mesh = FaultMesh.curved(trace, frame, uniform_dip=80.0, depths=depths,
                            nodes_per_level=counts, edge_length=3e3)

    assert mesh.n_nodes == sum(counts)
    assert mesh.attrs["nodes_per_level"] == counts
    assert "n_along" not in mesh.attrs, "n_along means nothing once the levels differ"
    mesh._check_winding(trace)
    # Every node is in a triangle and every triangle has real area.
    assert set(np.unique(mesh.triangles)) == set(range(mesh.n_nodes))
    np.testing.assert_array_less(0.0, mesh.areas)


def test_the_boundary_sides_still_resolve_to_whole_node_rows(trace, frame):
    counts = [40, 30, 20, 12]
    depths = np.array([0.0, 3e3, 9e3, 20e3])
    mesh = FaultMesh.curved(trace, frame, uniform_dip=90.0, depths=depths,
                            nodes_per_level=counts, edge_length=3e3)

    assert int(mesh.boundary_nodes("top").sum()) == counts[0]
    assert int(mesh.boundary_nodes("bottom").sum()) == counts[-1]
    # One node per level at each end, whatever the level holds.
    assert int(mesh.boundary_nodes("left").sum()) == len(counts)
    assert int(mesh.boundary_nodes("right").sum()) == len(counts)
    for side in L.EDGE_SIDES:
        assert mesh.boundary_elements(side).any()


def test_along_spacing_may_be_a_callable_of_depth(trace, frame):
    depths = np.array([0.0, 5e3, 12e3, 25e3])
    mesh = FaultMesh.curved(trace, frame, uniform_dip=90.0, depths=depths,
                            along_spacing=lambda z: 2e3 + 0.4 * z, edge_length=3e3)
    counts = mesh.attrs["nodes_per_level"]

    assert counts == sorted(counts, reverse=True), "deeper levels must be coarser"
    assert counts[0] > 2 * counts[-1]


# -- guards ----------------------------------------------------------------
def test_a_collapsed_boundary_raises_instead_of_silently_unconstraining_it():
    """The silent failure this guard exists for.

    ``zero_slip_boundary`` selects whatever the boundary mask flags, so an edge
    that resolves to a single node stops constraining the mesh's rim -- and slip
    runs off it with nothing raised anywhere.
    """
    params = np.array([[0.0, 0.0], [1e3, 0.0], [2e3, 0.0],
                       [1.0, -1e3], [1e3, -1e3], [2e3 - 1.0, -1e3]])
    with pytest.raises(ValueError, match="exact equality"):
        L.require_boundaries(params)

    snapped = L.snap_boundary(params, length=2e3, tol=10.0)
    assert L.require_boundaries(snapped)["left"] == 2


def test_an_unbalanced_request_is_refined_rather_than_triangulated_badly():
    counts = [100, 4, 100]
    with pytest.warns(RuntimeWarning, match="within 2.0:1"):
        layout = L.graded_layout(50e3, np.array([0.0, 5e3, 20e3]), counts=counts)

    ratios = layout.counts[1:] / layout.counts[:-1]
    np.testing.assert_array_less(np.maximum(ratios, 1.0 / ratios), 2.0 + 1e-9)
    assert layout.counts[0] == 100 and layout.counts[-1] == 100


def test_a_duplicated_parameter_point_is_refused_by_name():
    params = np.array([[0.0, 0.0], [1e3, 0.0], [1e3, 0.0], [0.0, -1e3], [1e3, -1e3]])
    with pytest.raises(ValueError, match="no triangle"):
        L.from_points(params, length=1e3, snap=False)


def test_an_irregular_vertical_mesh_is_accurate_despite_looking_near_vertical():
    """Why there is no near-vertical guard, pinned so nobody adds one.

    An irregular triangulation leaves most of a vertical mesh's elements a hair
    off 90 degrees -- ``arccos(|n_z|)`` cannot return exactly 90 when the normal's
    vertical component is at machine epsilon -- which looks exactly like the
    catastrophic band recorded beside vertical. It is not: the band is a
    rectangle-as-two-TDEs disagreement with the *Okada* oracle, and against
    ``cutde``, an independent implementation of the same triangular dislocation,
    the whole design matrix is fine.

    Dev-only: ``cutde`` is in the ``dev`` extra and is never imported at runtime.
    """
    hs = pytest.importorskip("cutde.halfspace")
    from nisar_tools.slip.greens import HalfSpaceTDE

    trace = FaultTrace(_LON, _LAT, name="test_fault")
    frame = trace.local_frame()
    mesh = FaultMesh.curved(trace, frame, uniform_dip=90.0,
                            depths=np.array([0.0, 3e3, 9e3, 20e3]),
                            nodes_per_level=[40, 30, 20, 12], edge_length=3e3)

    offset = np.abs(mesh.dip - 90.0)
    assert ((offset > 0) & (offset < 1e-2)).sum() > 0.5 * mesh.n_elements

    rng = np.random.default_rng(3)
    x, y = rng.uniform(-60e3, 60e3, 300), rng.uniform(-60e3, 60e3, 300)
    reference = hs.disp_matrix(np.column_stack([x, y, np.zeros_like(x)]),
                               mesh.vertices, 0.25)
    up = np.ones_like(x)
    green = HalfSpaceTDE(0.25).los_matrix(mesh, x, y, 0 * up, 0 * up, up)
    for component in (0, 1):
        block = green[:, component * mesh.n_elements:(component + 1) * mesh.n_elements]
        slip = np.zeros(3)
        slip[component] = 1.0
        expected = (reference @ slip)[:, 2, :]
        assert np.abs(block - expected).max() / np.abs(expected).max() < 1e-12


def test_bias_l_still_raises_but_now_points_at_the_knob_that_works(trace, frame):
    with pytest.raises(ValueError, match="nodes_per_level"):
        FaultMesh.curved(trace, frame, uniform_dip=90.0, bias_l=1.2)


def test_depths_and_bias_w_are_refused_together(trace, frame):
    with pytest.raises(ValueError, match="bias_w"):
        FaultMesh.curved(trace, frame, uniform_dip=90.0,
                         depths=np.array([0.0, 5e3, 10e3]), bias_w=1.15)


def test_a_layout_and_the_knobs_that_would_contradict_it_are_refused(trace, frame):
    layout = L.graded_layout(100e3, np.array([0.0, 5e3, 10e3]), counts=20)
    with pytest.raises(ValueError, match="contradicting"):
        FaultMesh.from_layout(layout, trace, frame, uniform_dip=90.0,
                              nodes_per_level=10)


# -- freeform --------------------------------------------------------------
def test_lloyd_places_small_elements_where_the_size_field_is_small(trace, frame):
    layout = L.lloyd_layout(120e3, 20e3, lambda s, d: 2e3 + 0.5 * d, iterations=10)
    mesh = FaultMesh.from_layout(layout, trace, frame, uniform_dip=90.0,
                                 edge_length=3e3)

    assert layout.counts is None, "a relaxed cloud has no rows"
    mesh._check_winding(trace)
    depth = -mesh.centroids[:, 2]
    size = np.sqrt(2.0 * mesh.areas)
    shallow = size[depth < 5e3].mean()
    deep = size[depth > 12e3].mean()
    assert deep > 1.5 * shallow, f"shallow {shallow:.0f} m vs deep {deep:.0f} m"
    for side in L.EDGE_SIDES:
        assert int(L.boundary_node_mask(layout.params, side).sum()) >= 2
