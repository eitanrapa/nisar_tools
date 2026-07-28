"""Pins the fault trace reader, the local frame, and the vertical mesh.

The invariant that matters most here is **winding**: an element wound against its
neighbours has a negated dip-slip Green's-function column, which on a
strike-slip fault produces a model whose dominant component looks fine and whose
dip-slip field is randomly signed. The mesh is built so that cannot happen, and
these tests assert it rather than trusting it.
"""

import numpy as np
import pytest

from nisar_tools.slip import FaultTrace, LocalFrame
from nisar_tools.slip.mesh import FaultMesh

# A trace striking ~84 degrees, like the San Sebastian fault: nearly east-west,
# which is exactly the orientation that defeats a fixed-axis winding test.
_LON = np.array([-68.681, -68.400, -68.000, -67.600, -67.200, -66.800, -66.523])
_LAT = np.array([10.410, 10.487, 10.541, 10.563, 10.593, 10.621, 10.630])


@pytest.fixture
def trace():
    return FaultTrace(_LON, _LAT, name="test_fault")


@pytest.fixture
def frame(trace):
    return trace.local_frame()


def _kml(path, lon, lat):
    coords = " ".join(f"{a},{b},0" for a, b in zip(lon, lat))
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<kml xmlns="http://www.opengis.net/kml/2.2"><Document><Placemark>'
        f"<LineString><coordinates>{coords}</coordinates></LineString>"
        "</Placemark></Document></kml>"
    )
    return path


# -- trace -------------------------------------------------------------------

def test_reads_kml_and_ascii_identically(tmp_path):
    """A KML and the ASCII ``gmt kml2gmt`` would make of it give the same trace."""
    kml = _kml(tmp_path / "f.kml", _LON, _LAT)
    dat = tmp_path / "f.dat"
    # kml2gmt emits one '> -L"name"' header line before the coordinates.
    dat.write_text('> -L"f"\n' + "\n".join(f"{a} {b}" for a, b in zip(_LON, _LAT)))

    from_kml = FaultTrace.from_file(kml)
    from_dat = FaultTrace.from_file(dat)
    np.testing.assert_allclose(from_kml.lon, from_dat.lon)
    np.testing.assert_allclose(from_kml.lat, from_dat.lat)


def test_ascii_reader_skips_comments_and_segment_headers(tmp_path):
    dat = tmp_path / "f.dat"
    dat.write_text("# a comment\n> -L\"seg\"\n-68.0 10.5\n\n-67.0 10.6\n")
    t = FaultTrace.from_file(dat)
    np.testing.assert_allclose(t.lon, [-68.0, -67.0])


def test_resampling_is_uniform_in_arc_length(trace, frame):
    """The digitised vertices are unevenly spaced; the mesh lattice must not be.

    Uniform in *arc length* along the original polyline, which is what the mesh
    lattice needs. The straight-line distance between consecutive samples is then
    very slightly shorter wherever the trace bends -- a resampled step that spans
    a vertex cuts the corner -- so the chord spacing is uniform only to the
    trace's curvature, here 0.2%.
    """
    spacing = 3000.0
    r = trace.resample(spacing, frame)
    x, y = r.to_local(frame)
    step = np.hypot(np.diff(x), np.diff(y))

    assert (step.max() - step.min()) / step.mean() < 0.01
    np.testing.assert_allclose(step.mean(), spacing, rtol=0.01)
    # Length is preserved to the same corner-cutting error.
    assert abs(r.length(frame) - trace.length(frame)) / trace.length(frame) < 1e-3


def test_normals_are_left_handed(trace, frame):
    tx, ty = trace.tangents(frame)
    nx, ny = trace.normals(frame)
    np.testing.assert_allclose(tx * nx + ty * ny, 0.0, atol=1e-12)
    np.testing.assert_allclose(np.hypot(nx, ny), 1.0, atol=1e-12)
    # cross(tangent, normal) points +z for a left-hand normal.
    np.testing.assert_array_less(0.0, tx * ny - ty * nx)


def test_side_classification(trace, frame):
    x, y = trace.to_local(frame)
    nx, ny = trace.normals(frame)
    i = len(x) // 2
    assert trace.side(x[i] + 5e3 * nx[i], y[i] + 5e3 * ny[i], frame)[0] == 1
    assert trace.side(x[i] - 5e3 * nx[i], y[i] - 5e3 * ny[i], frame)[0] == -1
    assert trace.side(x[i], y[i], frame, tol=100.0)[0] == 0


def test_frame_round_trip_and_cross_projection(trace, frame):
    lon, lat = frame.to_lonlat(*trace.to_local(frame))
    np.testing.assert_allclose(lon, trace.lon, atol=1e-9)
    np.testing.assert_allclose(lat, trace.lat, atol=1e-9)

    # Coming in from a projected CRS must land in the same place as from lon/lat,
    # which is what lets two tracks in different UTM zones share one frame.
    from pyproj import Transformer

    t = Transformer.from_crs("EPSG:4326", "EPSG:32619", always_xy=True)
    ux, uy = t.transform(trace.lon, trace.lat)
    ax, ay = frame.from_epsg(ux, uy, 32619)
    bx, by = trace.to_local(frame)
    np.testing.assert_allclose(ax, bx, atol=1e-6)
    np.testing.assert_allclose(ay, by, atol=1e-6)


def test_frame_mismatch_is_refused(frame):
    other = LocalFrame(0.0, 0.0)
    assert not frame.matches(other)
    with pytest.raises(ValueError, match="different LocalFrame"):
        frame.require_match(other.to_dict(), "observations")


# -- mesh --------------------------------------------------------------------

def test_vertical_mesh_geometry(trace, frame):
    mesh = FaultMesh.vertical(trace, frame, max_depth=20e3, edge_length=3e3)

    # Exactly vertical -- not 89.999999, which sits in the precision band the
    # dislocation solutions lose digits in.
    np.testing.assert_array_equal(mesh.dip, 90.0)
    assert mesh.nodes[:, 2].max() == 0.0
    assert mesh.nodes[:, 2].min() == -20e3

    # A vertical extrusion's area is exactly the trace length times the depth.
    expected = trace.resample(3e3, frame).length(frame) * 20e3
    assert abs(mesh.areas.sum() - expected) / expected < 1e-9

    n_along, n_down = mesh.attrs["n_along"], mesh.attrs["n_down"]
    assert mesh.n_nodes == n_along * n_down
    assert mesh.n_elements == 2 * (n_along - 1) * (n_down - 1)


def test_every_element_is_wound_to_the_traces_left(trace, frame):
    """The winding invariant, asserted rather than assumed."""
    mesh = FaultMesh.vertical(trace, frame, max_depth=15e3, edge_length=2e3)
    nx, ny = trace.normals(frame)
    tx, ty = trace.to_local(frame)
    s_trace = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(tx), np.diff(ty)))])
    s_elem = mesh.element_params[:, 0]

    normals = mesh.normals
    dot = (normals[:, 0] * np.interp(s_elem, s_trace, nx)
           + normals[:, 1] * np.interp(s_elem, s_trace, ny))
    np.testing.assert_array_less(0.0, dot)
    # A vertical fault's normal is horizontal.
    np.testing.assert_allclose(normals[:, 2], 0.0, atol=1e-12)


def test_reversing_the_trace_flips_every_normal(trace, frame):
    """Vertex order is meaningful, and reversing it must be *consistently* meaningful.

    Areas are unchanged and every normal flips. If winding ever came from a
    triangulation library's arbitrary output instead of the lattice, this would
    fail with a mixture of signs.
    """
    forward = FaultMesh.vertical(trace, frame, max_depth=15e3, edge_length=3e3)
    reverse = FaultMesh.vertical(
        FaultTrace(trace.lon[::-1], trace.lat[::-1]), frame,
        max_depth=15e3, edge_length=3e3,
    )
    assert reverse.n_elements == forward.n_elements
    assert abs(reverse.areas.sum() - forward.areas.sum()) / forward.areas.sum() < 1e-12
    # Mean normal is dominated by the near-constant strike, so its sign flip is
    # the signal; both meshes are internally consistent by their own check.
    assert np.sign(reverse.normals[:, 1].mean()) == -np.sign(forward.normals[:, 1].mean())


def test_boundary_elements_are_the_along_strike_ends(trace, frame):
    """Resolved in (s, z), so a northward bend cannot masquerade as a fault tip.

    The trace here bends so that an interior vertex reaches further north than the
    eastern endpoint -- which is what defeats picking edges by ``argmax`` of the
    north coordinate.
    """
    bent = FaultTrace(
        np.array([-68.6, -68.2, -67.8, -67.4, -67.0]),
        np.array([10.40, 10.52, 10.95, 10.60, 10.62]),   # interior spike north
    )
    mesh = FaultMesh.vertical(bent, bent.local_frame(), max_depth=12e3, edge_length=3e3)

    left = mesh.boundary_elements("left")
    right = mesh.boundary_elements("right")
    s = mesh.element_params[:, 0]
    # Every flagged element is genuinely at an arc-length extreme.
    assert s[left].max() < s.mean()
    assert s[right].min() > s.mean()
    assert left.sum() > 0 and right.sum() > 0

    bottom = mesh.boundary_elements("bottom")
    assert np.all(mesh.element_params[bottom, 1] < mesh.element_params[:, 1].mean())
    with pytest.raises(ValueError, match="Unknown boundary"):
        mesh.boundary_elements("sideways")


def test_neighbours_are_symmetric(trace, frame):
    mesh = FaultMesh.vertical(trace, frame, max_depth=9e3, edge_length=3e3)
    nb = mesh.neighbors
    for t in range(mesh.n_elements):
        for other in nb[t]:
            if other >= 0:
                assert t in nb[other]
    # Interior elements of a lattice have three neighbours; edge ones fewer.
    assert (nb >= 0).sum(axis=1).max() == 3


def test_dataset_round_trip(trace, frame):
    mesh = FaultMesh.vertical(trace, frame, max_depth=12e3, edge_length=3e3)
    back = FaultMesh.from_dataset(mesh.to_dataset())
    np.testing.assert_array_equal(back.nodes, mesh.nodes)
    np.testing.assert_array_equal(back.triangles, mesh.triangles)
    assert back.digest() == mesh.digest()
    assert back.frame.matches(mesh.frame)


def test_digest_tracks_geometry(trace, frame):
    a = FaultMesh.vertical(trace, frame, max_depth=12e3, edge_length=3e3)
    b = FaultMesh.vertical(trace, frame, max_depth=12e3, edge_length=2e3)
    assert a.digest() != b.digest()


def test_units_are_guarded():
    nodes = np.array([[0.0, 0.0, 0.0], [1e3, 0.0, 0.0], [0.0, 0.0, -1e3]])
    tri = np.array([[0, 1, 2]])
    params = np.zeros((3, 2))

    with pytest.raises(ValueError, match="below the free surface"):
        FaultMesh(nodes + np.array([0.0, 0.0, 10.0]), tri, params)
    with pytest.raises(ValueError, match="kilometres"):
        FaultMesh(nodes * 1e5, tri, params)
