"""Pins regularization, the solver, and end-to-end slip recovery.

The acceptance test is :func:`test_recovers_a_planted_slip_patch`: plant a known
slip distribution on the real fault geometry, forward-model it through two
viewing geometries, add noise, quadtree it, invert, and require the answer back.
Everything else here guards a specific way that can go quietly wrong.
"""

import numpy as np
import pytest
from slip_synthetic import forward_los_stack, tapered_slip

from nisar_tools.slip import FaultMesh, FaultTrace, Observations, SlipInversion
from nisar_tools.slip.inversion import _StackedOperator
from nisar_tools.slip.regularize import (
    neighbor_smoothing,
    ramp_columns,
    slip_bounds,
    zero_slip_boundary,
)

# The real San Sebastian trace, decimated: nearly east-west, which is the
# orientation that stresses the winding and boundary logic.
_LON = np.array([-68.681, -68.300, -67.900, -67.500, -67.100, -66.700, -66.523])
_LAT = np.array([10.410, 10.490, 10.540, 10.565, 10.595, 10.620, 10.630])


@pytest.fixture(scope="module")
def setup():
    trace = FaultTrace(_LON, _LAT, name="test_fault")
    frame = trace.local_frame()
    mesh = FaultMesh.vertical(trace, frame, max_depth=20e3, edge_length=6e3)
    return trace, frame, mesh


@pytest.fixture(scope="module")
def recovery(setup):
    """One shared forward-model-and-invert run; the rasters are slow to build."""
    trace, frame, mesh = setup
    truth = tapered_slip(mesh, peak=-2.0)
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
    inversion = SlipInversion(mesh, obs)
    return mesh, truth, obs, inversion


# -- regularization ----------------------------------------------------------

def test_smoothing_has_one_row_per_undirected_edge(setup):
    """Two rows per edge would be harmless but rescales the weight by sqrt(2)."""
    _, _, mesh = setup
    nb = mesh.neighbors
    n_edges = len({(min(i, int(j)), max(i, int(j)))
                   for i in range(mesh.n_elements) for j in nb[i] if j >= 0})
    smooth = neighbor_smoothing(nb)
    assert smooth.shape == (2 * n_edges, 2 * mesh.n_elements)
    # Each row is a single +1 / -1 difference.
    dense = smooth.toarray()
    np.testing.assert_allclose(dense.sum(axis=1), 0.0, atol=1e-12)
    assert np.all((dense != 0).sum(axis=1) == 2)


def test_smoothing_annihilates_uniform_slip(setup):
    """A constant slip field is perfectly smooth, so it must cost nothing."""
    _, _, mesh = setup
    smooth = neighbor_smoothing(mesh.neighbors, ss_ratio=1.0, ds_ratio=3.0)
    uniform = np.concatenate([np.full(mesh.n_elements, 2.0),
                              np.full(mesh.n_elements, -0.5)])
    np.testing.assert_allclose(smooth @ uniform, 0.0, atol=1e-12)


def test_dip_ratio_scales_only_the_dip_block(setup):
    _, _, mesh = setup
    smooth = neighbor_smoothing(mesh.neighbors, ss_ratio=1.0, ds_ratio=4.0)
    n_edges = smooth.shape[0] // 2
    ss_only = np.concatenate([np.arange(mesh.n_elements, dtype=float),
                              np.zeros(mesh.n_elements)])
    ds_only = np.concatenate([np.zeros(mesh.n_elements),
                              np.arange(mesh.n_elements, dtype=float)])
    a = smooth @ ss_only
    b = smooth @ ds_only
    np.testing.assert_allclose(b[n_edges:], 4.0 * a[:n_edges], atol=1e-12)


def test_zero_slip_boundary_is_sparse_and_targeted(setup):
    """Only the flagged elements get rows -- not a dense 2m x 2m diagonal per side."""
    _, _, mesh = setup
    w = zero_slip_boundary(mesh, sides=("bottom",), ratio=0.5)
    flagged = mesh.boundary_elements("bottom")
    assert w.shape == (2 * flagged.sum(), 2 * mesh.n_elements)
    dense = w.toarray()
    assert np.all((dense != 0).sum(axis=1) == 1)
    np.testing.assert_allclose(dense[dense != 0], 0.5)

    # Every constrained column belongs to a flagged element.
    touched = np.unique(np.nonzero(dense)[1]) % mesh.n_elements
    assert set(touched) == set(np.nonzero(flagged)[0])


def test_slip_bounds_polarity(setup):
    _, _, mesh = setup
    n = mesh.n_elements
    lo, hi = slip_bounds(n, strike=(-6.0, 6.0), dip=(-1.0, 1.0), polarity=(-1, 0, 0))
    # Right-lateral: strike-slip pinned non-positive, dip-slip untouched.
    np.testing.assert_allclose(hi[:n], 0.0)
    np.testing.assert_allclose(lo[:n], -6.0)
    np.testing.assert_allclose(lo[n:], -1.0)
    np.testing.assert_allclose(hi[n:], 1.0)

    lo, hi = slip_bounds(n, polarity=(1, 0))
    np.testing.assert_allclose(lo[:n], 0.0)

    with pytest.raises(ValueError, match="Bounds are empty"):
        slip_bounds(n, strike=(1.0, 6.0), polarity=(-1, 0))


def test_ramp_columns_are_per_track_and_scaled(recovery):
    """Nuisance columns must not leak between tracks or dominate conditioning."""
    _, _, obs, _ = recovery
    matrix, labels = ramp_columns(obs, "linear")
    assert matrix.shape == (obs.n, 3 * len(obs.tracks))
    assert labels[0].endswith(":offset")

    for i, name in enumerate(obs.tracks):
        block = matrix[:, 3 * i:3 * (i + 1)]
        other = ~obs.track_mask(name)
        np.testing.assert_allclose(block[other], 0.0)
    assert np.abs(matrix).max() <= 1.0

    assert ramp_columns(obs, "none")[0].shape == (obs.n, 0)
    with pytest.raises(ValueError, match="kind must be"):
        ramp_columns(obs, "quadratic")


# -- the acceptance test -----------------------------------------------------

def test_recovers_a_planted_slip_patch(recovery):
    """Forward-model a known slip distribution and get it back.

    This is the test the whole module exists to pass. It exercises the mesh, the
    dislocation solution, the quadtree, the two-track combination, the
    regularization and the solver together -- and it would fail on any of the sign
    errors the other tests guard individually.
    """
    mesh, truth, _, inversion = recovery
    model = inversion.solve(smoothing=0.3, ds_ratio=3.0, polarity=(-1, 0, 0),
                            strike=(-6.0, 6.0), dip=(-1.0, 1.0))

    assert model.converged, f"solver status {model.result.status}"
    assert model.variance_reduction > 95.0

    truth_ss = truth[:mesh.n_elements]
    assert np.corrcoef(model.strike_slip, truth_ss)[0, 1] > 0.9

    # Right-lateral sense preserved, and the polarity bound respected.
    assert np.all(model.strike_slip <= 1e-9)
    assert model.strike_slip.min() < 0.6 * truth_ss.min()

    # Moment is the robust summary statistic; regularization biases peak slip more.
    truth_moment = 30e9 * np.sum(mesh.areas * np.abs(truth_ss))
    assert 0.7 < model.moment() / truth_moment < 1.4

    # No spurious slip at the far ends, where the planted patch has none.
    s = mesh.element_params[:, 0]
    far = np.abs(s - s.mean()) > 0.4 * (s.max() - s.min())
    assert np.abs(model.strike_slip[far]).mean() < 0.1


def test_a_noise_only_inversion_stays_near_zero(recovery):
    """Given nothing to explain, the model must not invent slip.

    The bound is regularization, not the slip limits -- a solution pressed against
    ``strike=(-6, 6)`` would mean the null space is not being controlled.
    """
    mesh, _, obs, _ = recovery
    noisy = Observations(obs.ds.copy(deep=True))
    rng = np.random.default_rng(7)
    noisy.ds["los"] = ("obs", rng.normal(0.0, 0.004, obs.n))

    model = SlipInversion(mesh, noisy).solve(
        smoothing=0.3, polarity=(-1, 0, 0), strike=(-6.0, 6.0), dip=(-1.0, 1.0)
    )
    assert model.max_slip < 0.5


# -- solver behaviour --------------------------------------------------------

def test_more_smoothing_gives_a_smoother_model(recovery):
    _, _, _, inversion = recovery
    rough = inversion.solve(smoothing=0.02, polarity=(-1, 0, 0))
    smooth = inversion.solve(smoothing=3.0, polarity=(-1, 0, 0))
    assert smooth.roughness < rough.roughness
    assert smooth.variance_reduction <= rough.variance_reduction + 1e-6


def test_l_curve_is_monotone_in_roughness(recovery):
    """Roughness must rise as smoothing falls, and convergence must be reported.

    The 0.05 point is deliberately included: on this problem it is the awkward
    one, needing ~236 trust-region iterations where 2.0, 0.3 and 0.01 each need
    ~30. It converges under the default cap and would be silently truncated under
    a tighter one.
    """
    _, _, _, inversion = recovery
    ds, models = inversion.l_curve([0.05, 0.3, 2.0], polarity=(-1, 0, 0))

    # Returned largest-smoothing-first, which is also the cheap-to-converge order.
    assert list(ds["smoothing"].values) == [2.0, 0.3, 0.05]
    assert len(models) == 3
    r = ds["roughness"].values
    assert np.all(np.diff(r) > 0), r          # rougher as smoothing decreases
    assert np.all(ds["converged"].values), ds["iterations"].values
    assert ds.attrs["n_parameters"] == inversion.n_param


def test_l_curve_takes_no_worker_pool(recovery):
    """`l_curve` is serial on purpose -- see its docstring.

    A `workers=` argument was written, verified to give identical results, and
    removed: a sweep is dominated by the one weight that runs to the iteration
    cap (67% of an 8-weight sweep, capping any scheduler at 1.50x) and scipy's
    pure-Python `lsmr` holds the GIL anyway, so it measured 1.02x at best and
    slower above. This pins that it stays out of the signature rather than
    getting reintroduced on the same reasoning.
    """
    _, _, _, inversion = recovery
    with pytest.raises(TypeError, match="workers"):
        inversion.l_curve([0.3, 2.0], workers=2, polarity=(-1, 0, 0))


def _count_matvecs(monkeypatch):
    """Tally products through the stacked operator; returns a one-element list."""
    n = [0]
    for name in ("_matvec", "_rmatvec"):
        original = getattr(_StackedOperator, name)

        def counted(self, v, _original=original, _n=n):
            _n[0] += 1
            return _original(self, v)

        monkeypatch.setattr(_StackedOperator, name, counted)
    return n


def test_lsmr_tol_auto_is_the_default_and_cuts_work(recovery, monkeypatch):
    """The shipped default must be the adaptive inner tolerance, and must reduce
    work versus scipy's ``None`` -- for the same model.

    Counted in **matrix-vector products, not outer iterations**. ``"auto"`` solves
    each sub-problem loosely, so it can take *more* outer steps while doing far
    less total work; ``nit`` is the wrong meter, and pinning it pinned a fixture
    accident instead of the property -- it broke when the fixtures moved to the
    left-looking geometry NISAR actually flies, with no change to the solver.

    Summed over two weights, because the win is **not uniform in the weight**.
    Measured on this fixture, ``None``/``"auto"`` matvecs are 2.18x at lam=1.0,
    1.93x at 0.5 and 1.91x at 0.3, but **0.70x at lam=2.0 and 0.78x at 0.05** --
    the adaptive tolerance loses at both ends. Over a whole sweep it still comes
    out ahead (1.17x here), which is how :meth:`l_curve` uses it.
    """
    _, _, _, inversion = recovery
    weights = (1.0, 0.3)

    n = _count_matvecs(monkeypatch)
    work, models = {}, {}
    for tol in ("auto", None):
        n[0] = 0
        models[tol] = [inversion.solve(smoothing=lam, polarity=(-1, 0, 0),
                                       lsmr_tol=tol) for lam in weights]
        work[tol] = n[0]

    assert models["auto"][0].options["lsmr_tol"] == "auto"
    assert all(m.converged for ms in models.values() for m in ms)
    assert work[None] > 1.2 * work["auto"], work

    # Same answer: the speed-up is in how the sub-problem is solved, not what for.
    for fast, slow in zip(models["auto"], models[None]):
        assert fast.variance_reduction == pytest.approx(slow.variance_reduction,
                                                        abs=0.5)
        np.testing.assert_allclose(fast.strike_slip, slow.strike_slip, atol=0.05)


def test_ramp_absorbs_an_added_offset(recovery):
    """A constant added to one track must land in the nuisance column, not in slip."""
    mesh, _, obs, plain = recovery
    shifted = Observations(obs.ds.copy(deep=True))
    mask = shifted.track_mask("asc")
    shifted.ds["los"].values[mask] += 0.05

    without = SlipInversion(mesh, shifted).solve(smoothing=0.3, polarity=(-1, 0, 0))
    with_ramp = SlipInversion(mesh, shifted, ramp="offset").solve(
        smoothing=0.3, polarity=(-1, 0, 0)
    )
    assert with_ramp.variance_reduction > without.variance_reduction
    # The recovered offset should be close to what was injected.
    offsets = dict(zip(with_ramp.ramp_labels, with_ramp.ramp))
    assert abs(offsets["asc:offset"] - 0.05) < 0.03


def test_frame_mismatch_is_refused(setup, recovery):
    from nisar_tools.slip import LocalFrame

    _, _, mesh = setup
    _, _, obs, _ = recovery
    wrong = Observations(obs.ds.copy(deep=True))
    wrong.ds.attrs["frame"] = LocalFrame(0.0, 0.0).to_dict()
    with pytest.raises(ValueError, match="different LocalFrame"):
        SlipInversion(mesh, wrong)


# -- outputs -----------------------------------------------------------------

def test_to_dataset_and_text(recovery, tmp_path):
    mesh, _, _, inversion = recovery
    model = inversion.solve(smoothing=0.3, polarity=(-1, 0, 0))

    ds = model.to_dataset()
    assert ds.sizes["element"] == mesh.n_elements
    assert ds.attrs["converged"] == 1
    np.testing.assert_allclose(ds["slip"].values,
                               np.hypot(model.strike_slip, model.dip_slip))

    path = model.to_text(tmp_path / "model.txt")
    table = np.loadtxt(path, skiprows=1)
    assert table.shape == (mesh.n_elements, 10)
    np.testing.assert_allclose(table[:, 6], model.strike_slip, rtol=1e-6)
    np.testing.assert_allclose(table[:, 8], mesh.areas, rtol=1e-9)
    # Longitude and latitude, not local metres.
    assert -180.0 < table[:, 1].min() and table[:, 1].max() < 180.0


def test_vertex_tables_rebuild_the_element_table(recovery, tmp_path):
    """The load-bearing check on the vertex pair: the join has to close.

    Averaging each triangle's three nodal slips must give back the element value
    ``slip_model.txt`` reports. That single assertion pins the connectivity
    indexing, the 1-based offset and the column order all at once -- an off-by-one
    in any of them scatters slip onto the wrong corners and still writes a
    plausible-looking file.

    A **nodal** model, because there the identity is exact by construction: the
    element value simply *is* the mean of its three nodes. For an element model the
    node values come from a lossy scatter and the round trip smooths, so what
    survives there is the componentwise integral, tested separately below.
    """
    mesh, _, obs, _ = recovery
    model = SlipInversion(mesh, obs, basis="node").solve(
        smoothing=0.3, polarity=(-1, 0, 0))

    node_path, element_path = model.to_vertex_text(tmp_path)
    assert node_path.name == "vert_nodes.txt"
    assert element_path.name == "vert_elements.txt"

    nodes = np.loadtxt(node_path, skiprows=1)
    elements = np.loadtxt(element_path, skiprows=1, dtype=int)
    assert nodes.shape == (mesh.n_nodes, 9)
    assert elements.shape == (mesh.n_elements, 4)

    # 1-based throughout, and every index addresses a row of the node table.
    np.testing.assert_array_equal(elements[:, 0], np.arange(1, mesh.n_elements + 1))
    assert elements[:, 1:].min() == 1
    assert elements[:, 1:].max() == mesh.n_nodes

    connectivity = elements[:, 1:] - 1
    for column, component in ((5, 0), (6, 1)):
        np.testing.assert_allclose(nodes[:, column][connectivity].mean(axis=1),
                                   model.element_slip[:, component], atol=1e-9)

    # Element ids join row for row with the element table.
    table = np.loadtxt(model.to_text(tmp_path / "slip_model.txt"), skiprows=1)
    np.testing.assert_array_equal(elements[:, 0], table[:, 0].astype(int))
    # Geographic, not local metres; depth negative-down like `to_text`'s column 4.
    assert -180.0 < nodes[:, 1].min() and nodes[:, 1].max() < 180.0
    assert nodes[:, 3].max() == 0.0 and nodes[:, 3].min() < 0.0


def test_nodal_slip_conserves_the_slip_integral_componentwise(setup, recovery):
    """The element -> node scatter is area-weighted, so it conserves each component.

    Exactly, because ``P``'s rows sum to one and ``lumped_areas == P.T @ areas``.
    It does **not** conserve the sum of slip *magnitudes* -- an element value is
    the mean of three vectors, which is shorter than the mean of their lengths --
    so the two differ by about a percent and neither is a check on the other.
    """
    from nisar_tools.slip.basis import NodeBasis

    mesh, _, obs, _ = recovery
    model = SlipInversion(mesh, obs, ramp="linear", basis="element").solve(
        smoothing=0.3, polarity=(-1, 0, 0))

    nodal = model.nodal_slip()
    assert nodal.shape == (mesh.n_nodes, 2)
    lumped = NodeBasis(mesh).lumped_areas()
    for component in (0, 1):
        np.testing.assert_allclose((lumped * nodal[:, component]).sum(),
                                   (mesh.areas * model.element_slip[:, component]).sum(),
                                   rtol=1e-12)


def test_nodal_slip_is_the_parameter_vector_for_a_nodal_model(recovery):
    """No transformation at all when the nodes are what was solved for."""
    mesh, _, obs, _ = recovery
    model = SlipInversion(mesh, obs, ramp="linear", basis="node").solve(
        smoothing=0.3, polarity=(-1, 0, 0))

    nodal = model.nodal_slip()
    np.testing.assert_array_equal(nodal[:, 0], model.strike_slip)
    np.testing.assert_array_equal(nodal[:, 1], model.dip_slip)


def test_unconverged_models_refuse_to_export(recovery, tmp_path):
    """An iteration-capped solve has a meaningless fit; it must not be written out."""
    _, _, _, inversion = recovery
    model = inversion.solve(smoothing=0.3, polarity=(-1, 0, 0), max_iter=1)
    if model.converged:
        pytest.skip("solver converged in a single iteration; nothing to assert")
    assert "UNCONVERGED" in repr(model)
    with pytest.raises(ValueError, match="unconverged"):
        model.to_text(tmp_path / "bad.txt")
    with pytest.raises(ValueError, match="unconverged"):
        model.to_vertex_text(tmp_path)


def test_persist_round_trip(recovery, tmp_path):
    from nisar_tools import Workspace

    _, _, _, inversion = recovery
    model = inversion.solve(smoothing=0.3, polarity=(-1, 0, 0))
    ws = Workspace(tmp_path / "ws")
    stored = model.persist(ws, "slip")
    np.testing.assert_allclose(stored["strike_slip"].values, model.strike_slip)
    assert stored.attrs["mesh_digest"] == inversion.mesh.digest()


# -- recovery on a dipping fault ---------------------------------------------

@pytest.fixture(scope="module")
def dipping_recovery():
    """The phase-two acceptance run: a 70-degree fault with genuine dip-slip.

    Two things are new relative to :func:`recovery`, and both matter. The mesh
    dips, so every element's Green's function comes out of the general branch of
    the triangular-dislocation solution rather than the exactly-vertical one; and
    the planted patch has ``rake=25``, so the dip-slip half of the design matrix
    is actually excited. With a pure strike-slip patch on a vertical fault -- the
    phase-one fixture -- half the columns could be wrong and the recovery test
    would still pass.
    """
    trace = FaultTrace(_LON, _LAT, name="test_fault")
    frame = trace.local_frame()
    mesh = FaultMesh.curved(trace, frame, uniform_dip=70.0,
                            max_depth=20e3, edge_length=6e3)
    truth = tapered_slip(mesh, peak=-2.0, rake=25.0)
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
    return mesh, truth, SlipInversion(mesh, obs)


def test_recovers_a_planted_patch_on_a_dipping_fault(dipping_recovery):
    """The phase-two gate, held to the same bar as the vertical one."""
    mesh, truth, inversion = dipping_recovery
    model = inversion.solve(smoothing=0.3, ds_ratio=3.0, polarity=(-1, 0, 0),
                            strike=(-6.0, 6.0), dip=(-3.0, 3.0))

    assert model.converged, f"solver status {model.result.status}"
    assert model.variance_reduction > 95.0

    n = mesh.n_elements
    assert np.corrcoef(model.strike_slip, truth[:n])[0, 1] > 0.9
    truth_moment = 30e9 * np.sum(mesh.areas * np.hypot(truth[:n], truth[n:]))
    assert 0.7 < model.moment() / truth_moment < 1.4

    s = mesh.element_params[:, 0]
    far = np.abs(s - s.mean()) > 0.4 * (s.max() - s.min())
    assert np.abs(model.slip_magnitude[far]).mean() < 0.1


def test_dip_slip_is_recovered_not_just_absorbed(dipping_recovery):
    """The half of the model a vertical, pure-strike-slip fixture never tests.

    Correlation, not amplitude: regularization biases dip-slip more than
    strike-slip because it is the weaker of the two signals here and carries
    ``ds_ratio`` times the smoothing.
    """
    mesh, truth, inversion = dipping_recovery
    model = inversion.solve(smoothing=0.3, ds_ratio=3.0, polarity=(-1, 0, 0),
                            strike=(-6.0, 6.0), dip=(-3.0, 3.0))

    n = mesh.n_elements
    truth_ds = truth[n:]
    assert np.abs(truth_ds).max() > 0.5, "the fixture must actually plant dip-slip"
    assert np.corrcoef(model.dip_slip, truth_ds)[0, 1] > 0.8
    # Sign preserved: rake 25 on a negative peak puts dip-slip negative too.
    assert model.dip_slip[np.argmax(np.abs(truth_ds))] < 0


# -- surface displacement export ---------------------------------------------

def test_surface_displacement_grid(recovery):
    """The full three-component field, on the mesh's own frame.

    Line of sight collapses three components into one; what an inversion actually
    recovers is the vector, so being able to look at it is the point.
    """
    _, _, _, inversion = recovery
    model = inversion.solve(smoothing=0.3, polarity=(-1, 0, 0))
    grid = model.surface_displacement(spacing=5000.0, pad=40e3)

    assert set(grid.data_vars) == {"ux", "uy", "uz"}
    assert grid.rio.crs is not None, "the grid must be georeferenced to be exportable"
    assert grid.y[0] > grid.y[-1], "north-up, like every other raster here"
    for name in ("ux", "uy", "uz"):
        assert np.isfinite(grid[name].values).mean() > 0.8
        assert grid[name].attrs["units"] == "m"


def test_surface_displacement_is_nan_on_the_trace(recovery):
    """A dislocation solution is singular where the fault meets the surface.

    Returning a very large number there would be worse than returning nothing:
    it would set the colour scale of every plot of the field.
    """
    mesh, _, _, inversion = recovery
    model = inversion.solve(smoothing=0.3, polarity=(-1, 0, 0))
    grid = model.surface_displacement(spacing=4000.0, pad=20e3, exclude_within=8000.0)

    top = mesh.nodes[mesh.params[:, 1] == mesh.params[:, 1].max()]
    xx, yy = np.meshgrid(grid.x.values, grid.y.values)
    distance = np.sqrt(np.min((xx.ravel()[:, None] - top[None, :, 0]) ** 2
                              + (yy.ravel()[:, None] - top[None, :, 1]) ** 2, axis=1))
    near = (distance <= 8000.0).reshape(xx.shape)

    assert near.any(), "the test needs some grid points on the fault"
    assert np.all(np.isnan(grid["ux"].values[near]))
    assert np.isfinite(grid["ux"].values[~near]).all()


def test_surface_displacement_matches_forward_point_by_point(recovery):
    """The gridded field and ``forward`` must be the same calculation.

    They are not the same code path -- the grid evaluates in blocks and masks the
    near field -- so this is what pins the blocking as free.
    """
    _, _, _, inversion = recovery
    model = inversion.solve(smoothing=0.3, polarity=(-1, 0, 0))
    grid = model.surface_displacement(spacing=8000.0, pad=30e3, block=64)

    xx, yy = np.meshgrid(grid.x.values, grid.y.values)
    finite = np.isfinite(grid["ux"].values)
    pick = np.nonzero(finite.ravel())[0][::37][:20]
    expected = model.forward(xx.ravel()[pick], yy.ravel()[pick])

    for i, name in enumerate(("ux", "uy", "uz")):
        np.testing.assert_allclose(grid[name].values.ravel()[pick], expected[:, i],
                                   rtol=1e-5, atol=1e-9)


def test_strike_slip_moves_the_ground_sideways_not_up(recovery):
    """Physical sanity, and a check the components are not swapped.

    A pure strike-slip patch on this near-east-west fault produces horizontal
    displacement an order of magnitude larger than vertical, and mostly east-west.
    If ``ux``/``uy``/``uz`` were permuted this would fail immediately.
    """
    _, _, _, inversion = recovery
    model = inversion.solve(smoothing=0.3, polarity=(-1, 0, 0), dip=(-0.01, 0.01))
    grid = model.surface_displacement(spacing=5000.0, pad=40e3)

    east = np.nanmax(np.abs(grid["ux"].values))
    north = np.nanmax(np.abs(grid["uy"].values))
    up = np.nanmax(np.abs(grid["uz"].values))
    assert east > 3 * north, "an east-west fault slips mostly east-west"
    assert min(east, north) > 3 * up, "strike-slip is not a vertical motion"


def test_to_grd_writes_readable_single_variable_grids(recovery, tmp_path):
    import xarray as xr

    _, _, _, inversion = recovery
    model = inversion.solve(smoothing=0.3, polarity=(-1, 0, 0))
    paths = model.to_grd(tmp_path, spacing=8000.0, pad=30e3)

    assert [p.name for p in paths] == ["ux.grd", "uy.grd", "uz.grd"]
    for path in paths:
        field = xr.open_dataarray(path)
        assert field.name == "z"
        assert field.dims == ("lat", "lon")
        assert np.isfinite(field.values).any()
        field.close()

    with pytest.raises(KeyError, match="Unknown field"):
        model.to_grd(tmp_path, fields=("ue",), spacing=20000.0, pad=20e3)
