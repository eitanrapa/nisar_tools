"""Slip parameterization: element-constant against nodal tent functions.

Changing the basis changes what a parameter *means*, so the tests here are mostly
about invariants that must survive the change -- total area, moment, the shape of
the recovered field -- rather than about numbers that are allowed to differ.
"""

import numpy as np
import pytest
from slip_synthetic import forward_los_stack, tapered_slip

from nisar_tools.slip import FaultMesh, FaultTrace, Observations, SlipInversion
from nisar_tools.slip.basis import ElementBasis, NodeBasis, make_basis
from nisar_tools.slip.regularize import laplace_beltrami, neighbor_smoothing

_LON = np.array([-68.681, -68.300, -67.900, -67.500, -67.100, -66.700, -66.523])
_LAT = np.array([10.410, 10.490, 10.540, 10.565, 10.595, 10.620, 10.630])


@pytest.fixture(scope="module")
def mesh():
    trace = FaultTrace(_LON, _LAT, name="test_fault")
    return FaultMesh.curved(trace, trace.local_frame(), uniform_dip=75.0,
                            max_depth=20e3, edge_length=6e3)


@pytest.fixture(scope="module")
def scene(mesh):
    """One shared forward-model run; the rasters are slow to build."""
    trace = FaultTrace(_LON, _LAT, name="test_fault")
    frame = trace.local_frame()
    truth = tapered_slip(mesh, peak=-2.0, rake=20.0)
    obs = Observations.concat(
        [
            Observations.from_los(
                forward_los_stack(mesh, truth, trace, frame, geometry=g,
                                  spacing=5000.0, noise=0.004, nan_rows=3, seed=1),
                name=g, frame=frame, trace=trace,
                rms_min=0.006, width_min=6000.0, width_max=30000.0,
                exclude_within=5000.0,
            )
            for g in ("asc", "desc")
        ],
        normalize="sqrt_count",
    )
    return mesh, truth, obs


# -- the bases themselves ----------------------------------------------------

def test_node_basis_is_smaller_than_element_basis(mesh):
    node, element = NodeBasis(mesh), ElementBasis(mesh)
    assert node.n_basis < element.n_basis
    assert node.n_param == 2 * mesh.n_nodes
    assert element.n_param == 2 * mesh.n_elements


def test_lumped_areas_partition_the_fault(mesh):
    """Every square metre is accounted for exactly once, in either basis.

    The invariant that makes a moment comparable across parameterizations: a
    node's share is a third of its 1-ring, and the thirds sum to the whole.
    """
    for basis in (ElementBasis(mesh), NodeBasis(mesh)):
        assert basis.lumped_areas().sum() == pytest.approx(mesh.areas.sum())
        assert np.all(basis.lumped_areas() > 0)


def test_projection_preserves_each_basis_function_moment(mesh):
    """``P.T @ element_areas`` must be the lumped areas, exactly.

    This is what makes the mean-slip projection legitimate: it replaces a tent
    with its average over each triangle, and an average has the same integral.
    Get it wrong and the recovered moment is wrong by the same factor, silently.
    """
    basis = NodeBasis(mesh)
    projection = basis.projection()
    np.testing.assert_allclose(projection.T @ mesh.areas, basis.lumped_areas())
    np.testing.assert_allclose(np.asarray(projection.sum(axis=1)).ravel(), 1.0)


def test_tent_weights_integrate_to_the_lumped_area(mesh):
    """The definition the projection approximates, checked against it directly."""
    basis = NodeBasis(mesh)
    for node in (0, mesh.n_nodes // 2, mesh.n_nodes - 1):
        _, weights, _ = basis.tent_weights(node)
        assert weights.sum() == pytest.approx(basis.lumped_areas()[node], rel=1e-6)


def test_element_values_of_a_uniform_field(mesh):
    basis = NodeBasis(mesh)
    np.testing.assert_allclose(basis.element_values(np.full(basis.n_basis, 2.5)), 2.5)
    # And the projection agrees with it.
    coefficients = np.arange(basis.n_basis, dtype=float)
    np.testing.assert_allclose(basis.projection() @ coefficients,
                               basis.element_values(coefficients))


def test_node_boundary_picks_whole_edges(mesh):
    basis = NodeBasis(mesh)
    assert int(basis.boundary(("bottom",)).sum()) == mesh.attrs["n_along"]
    assert int(basis.boundary(("left",)).sum()) == mesh.attrs["n_down"]


def test_make_basis_by_name_and_passthrough(mesh):
    assert make_basis(mesh, "element").name == "element"
    assert make_basis(mesh, "node").name == "node"
    already = NodeBasis(mesh)
    assert make_basis(mesh, already) is already
    with pytest.raises(ValueError, match="Unknown slip basis"):
        make_basis(mesh, "quadratic")


# -- the nodal smoother ------------------------------------------------------

def test_laplace_beltrami_annihilates_a_constant_field(mesh):
    """A flat slip field has no curvature, so it must cost nothing.

    Catches a sign error in the edge weights and a mismatch between the degree
    diagonal and the weight matrix at once -- both of which leave an operator
    that looks reasonable and penalises uniform slip.
    """
    operator = laplace_beltrami(mesh)
    constant = np.ones(2 * mesh.n_nodes)
    assert np.abs(operator @ constant).max() < 1e-9


def test_laplace_beltrami_penalises_roughness(mesh):
    operator = laplace_beltrami(mesh)
    rng = np.random.default_rng(0)
    smooth = np.tile(mesh.params[:, 0] / mesh.params[:, 0].max(), 2)
    rough = rng.normal(size=2 * mesh.n_nodes)
    assert np.linalg.norm(operator @ rough) > 50 * np.linalg.norm(operator @ smooth)


def test_laplace_beltrami_weights_are_never_negative(mesh):
    """The reason for the cotangent/mean-value hybrid.

    A cotangent weight goes negative across an obtuse pair of triangles, which
    turns the smoother into an anti-smoother on part of the mesh: it would
    *reward* a jump between those two nodes. The fallback exists to prevent that,
    so the off-diagonal entries must all be non-positive in ``D - W`` form.
    """
    operator = laplace_beltrami(mesh, form="dirichlet").toarray()
    off_diagonal = operator - np.diag(np.diag(operator))
    assert off_diagonal.max() <= 1e-12


def test_nodal_and_element_smoothers_are_on_the_same_scale(mesh):
    """So one smoothing weight means the same thing in both parameterizations.

    Without the normalisation the nodal operator comes out about 1e4 times
    stronger on a 6 km mesh -- the reference's median-edge-length prefactor does
    not get there in metre units -- and a weight of 0.3 drives every parameter to
    zero.
    """
    element = neighbor_smoothing(mesh.neighbors)
    node = laplace_beltrami(mesh)
    ratio = (np.abs(node).sum(axis=1).max() / np.abs(element).sum(axis=1).max())
    assert 0.2 < float(ratio) < 5.0


# -- end to end --------------------------------------------------------------

def test_both_bases_recover_the_planted_patch(scene):
    """The acceptance test for the parameterization, run twice.

    A tent field should do at least as well as a piecewise-constant one on the
    same mesh while using fewer parameters, because a real slip distribution is
    continuous and a piecewise-constant model has to spend resolution
    representing edges that are not there.
    """
    mesh, truth, obs = scene
    n = mesh.n_elements
    truth_element = np.column_stack([truth[:n], truth[n:]])
    truth_moment = 30e9 * np.sum(mesh.areas * np.hypot(*truth_element.T))

    results = {}
    for basis in ("element", "node"):
        inversion = SlipInversion(mesh, obs, basis=basis)
        model = inversion.solve(smoothing=0.3, ds_ratio=3.0, polarity=(-1, 0, 0),
                                strike=(-6.0, 6.0), dip=(-3.0, 3.0))
        assert model.converged, basis
        assert model.variance_reduction > 95.0, basis
        assert 0.7 < model.moment() / truth_moment < 1.4, basis
        results[basis] = (model, inversion, np.corrcoef(
            model.element_slip[:, 0], truth_element[:, 0])[0, 1])

    assert results["node"][1].n_param < results["element"][1].n_param
    assert results["node"][2] > 0.9
    assert results["element"][2] > 0.9


def test_nodal_model_reports_slip_per_element(scene):
    """The output format must not depend on the parameterization."""
    mesh, _, obs = scene
    model = SlipInversion(mesh, obs, basis="node").solve(
        smoothing=0.3, polarity=(-1, 0, 0))

    assert model.strike_slip.size == mesh.n_nodes
    assert model.element_slip.shape == (mesh.n_elements, 2)
    ds = model.to_dataset()
    assert ds.sizes["element"] == mesh.n_elements
    assert ds.sizes["basis"] == mesh.n_nodes
    assert ds.attrs["basis"] == "node"


def test_nodal_model_round_trips_through_save(scene, tmp_path):
    mesh, _, obs = scene
    model = SlipInversion(mesh, obs, basis="node").solve(
        smoothing=0.3, polarity=(-1, 0, 0))
    from nisar_tools.slip import SlipModel

    path = model.save(tmp_path / "nodal.slip.zip")
    back = SlipModel.load(path)
    assert back.basis.name == "node"
    np.testing.assert_allclose(back.strike_slip, model.strike_slip)
    assert back.roughness == pytest.approx(model.roughness, rel=1e-9)
    assert back.moment() == pytest.approx(model.moment(), rel=1e-9)


def test_moment_accepts_a_velocity_model(scene):
    """Depth-dependent rigidity, the point of getting this far.

    A crust that is softer at the top gives the same shallow slip a smaller
    moment than a uniform 30 GPa does.
    """
    from nisar_tools.slip.edgrn import VelocityModel

    mesh, _, obs = scene
    model = SlipInversion(mesh, obs).solve(smoothing=0.3, polarity=(-1, 0, 0))
    soft_top = VelocityModel([0.0, 5e3, 30e3], [4.0e3, 6.0e3, 6.5e3],
                             [2.0e3, 3.5e3, 3.7e3], [2.4e3, 2.7e3, 2.9e3])

    layered = model.moment(soft_top)
    uniform = model.moment(30e9)
    assert layered != uniform
    assert 0.1 * uniform < layered < 2.0 * uniform
