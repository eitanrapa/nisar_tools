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


def test_unconverged_models_refuse_to_export(recovery, tmp_path):
    """An iteration-capped solve has a meaningless fit; it must not be written out."""
    _, _, _, inversion = recovery
    model = inversion.solve(smoothing=0.3, polarity=(-1, 0, 0), max_iter=1)
    if model.converged:
        pytest.skip("solver converged in a single iteration; nothing to assert")
    assert "UNCONVERGED" in repr(model)
    with pytest.raises(ValueError, match="unconverged"):
        model.to_text(tmp_path / "bad.txt")


def test_persist_round_trip(recovery, tmp_path):
    from nisar_tools import Workspace

    _, _, _, inversion = recovery
    model = inversion.solve(smoothing=0.3, polarity=(-1, 0, 0))
    ws = Workspace(tmp_path / "ws")
    stored = model.persist(ws, "slip")
    np.testing.assert_allclose(stored["strike_slip"].values, model.strike_slip)
    assert stored.attrs["mesh_digest"] == inversion.mesh.digest()
