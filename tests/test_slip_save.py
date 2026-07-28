"""Saving a solved model to one file and getting it back.

The point of `SlipModel.save` is a long run under `nohup`: solve once, save,
and pick the result up later in a notebook without rebuilding the Green's
matrix. So `load` has to return something that still reports every statistic,
re-exports, plots, and forward-models -- not just a bag of arrays.
"""

import numpy as np
import pytest

from nisar_tools.slip import FaultMesh, FaultTrace, Observations, SlipInversion
from nisar_tools.slip import SlipModel
from slip_synthetic import analytic_los_stack

_LON = np.array([-68.681, -68.300, -67.900, -67.500, -67.100, -66.700, -66.523])
_LAT = np.array([10.410, 10.490, 10.540, 10.565, 10.595, 10.620, 10.630])


@pytest.fixture(scope="module")
def solved():
    trace = FaultTrace(_LON, _LAT, name="test_fault")
    frame = trace.local_frame()
    mesh = FaultMesh.vertical(trace, frame, max_depth=20e3, edge_length=8e3)
    obs = Observations.concat([
        Observations.from_los(
            analytic_los_stack(trace, frame, geometry=g, spacing=6000.0,
                               noise=0.004, seed=1),
            name=g, frame=frame, trace=trace, rms_min=0.01, width_min=12000.0,
            width_max=40000.0, exclude_within=5000.0)
        for g in ("asc", "desc")])
    model = SlipInversion(mesh, obs, ramp="linear").solve(
        smoothing=0.3, polarity=(-1, 0, 0), strike=(-6.0, 6.0), dip=(-1.0, 1.0))
    return trace, mesh, obs, model


def test_save_load_round_trip(solved, tmp_path):
    _, mesh, obs, model = solved
    path = model.save(tmp_path / "model.slip.zip")
    back = SlipModel.load(path)

    np.testing.assert_array_equal(back.x, model.x)
    np.testing.assert_array_equal(back.strike_slip, model.strike_slip)
    np.testing.assert_array_equal(back.dip_slip, model.dip_slip)
    np.testing.assert_array_equal(back.ramp, model.ramp)
    np.testing.assert_array_equal(back.data, model.data)
    np.testing.assert_array_equal(back.residual, model.residual)

    # Every derived statistic, including the ones needing the smoothing operator
    # and the mesh geometry -- neither of which is stored as such.
    for stat in ("variance_reduction", "rms_misfit", "roughness", "max_slip",
                 "moment_magnitude", "converged"):
        assert getattr(back, stat) == pytest.approx(getattr(model, stat)), stat
    assert back.ramp_labels == model.ramp_labels
    assert back.options == model.options
    assert back.smoothing == model.smoothing


def test_loaded_model_keeps_mesh_and_observations(solved, tmp_path):
    _, mesh, obs, model = solved
    back = SlipModel.load(model.save(tmp_path / "m.slip.zip"))

    # The mesh must be the *same* geometry, not merely the same size: digest is
    # a hash of nodes and triangles.
    assert back.mesh.digest() == mesh.digest()
    assert back.mesh.n_elements == mesh.n_elements
    assert back.obs.n == obs.n
    assert back.obs.tracks == obs.tracks
    # Per-track residuals need obs["track"] to have survived as strings.
    for name in obs.tracks:
        np.testing.assert_array_equal(back.track_residual(name),
                                      model.track_residual(name))


def test_loaded_model_can_forward_and_export(solved, tmp_path):
    trace, _, _, model = solved
    back = SlipModel.load(model.save(tmp_path / "m.slip.zip"))

    # forward() needs only the mesh and the engine, so it survives the Green's
    # matrix deliberately not being saved.
    x = np.array([20e3, -30e3]), np.array([15e3, 25e3])
    np.testing.assert_allclose(back.forward(*x), model.forward(*x), rtol=1e-10)
    assert back._inversion.g is None

    out = back.to_text(tmp_path / "model.txt")
    table = np.loadtxt(out, skiprows=1)
    assert table.shape == (back.mesh.n_elements, 10)


def test_saved_file_is_one_copyable_file(solved, tmp_path):
    """One file, copyable -- the whole point for a background run: solve under
    nohup on one machine, scp the result, open it in a notebook."""
    _, _, _, model = solved
    path = model.save(tmp_path / "m.slip.zip")
    assert (tmp_path / "m.slip.zip").is_file()

    moved = tmp_path / "elsewhere" / "renamed.zip"
    moved.parent.mkdir()
    moved.write_bytes(open(path, "rb").read())

    back = SlipModel.load(moved)
    assert back.variance_reduction == pytest.approx(model.variance_reduction)
    # Zarr attrs are JSON, so the provenance dicts survive as dicts.
    assert isinstance(back.options, dict)
    assert back.options["polarity"] == [-1, 0, 0]
    assert back.options["lsmr_tol"] == "auto"
