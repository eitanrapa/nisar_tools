"""Smoke tests for the slip figures.

Plots are checked for "runs, and puts the right number of things on the right
axes", not for appearance. The one substantive assertion is that the unrolled
slip panel draws one polygon per element -- if it silently dropped elements the
figure would still look plausible.
"""

import matplotlib
import numpy as np
import pytest
from slip_synthetic import forward_los_stack, tapered_slip

matplotlib.use("Agg")  # before pyplot is imported anywhere

from nisar_tools.slip import FaultMesh, FaultTrace, Observations, SlipInversion  # noqa: E402
from nisar_tools.slip import plot as slip_plot  # noqa: E402

_LON = np.array([-68.681, -68.200, -67.700, -67.200, -66.700, -66.523])
_LAT = np.array([10.410, 10.500, 10.550, 10.590, 10.620, 10.630])


@pytest.fixture(scope="module")
def solved():
    trace = FaultTrace(_LON, _LAT, name="test_fault")
    frame = trace.local_frame()
    mesh = FaultMesh.vertical(trace, frame, max_depth=18e3, edge_length=9e3)
    truth = tapered_slip(mesh, peak=-2.0)
    obs = Observations.concat([
        Observations.from_los(
            forward_los_stack(mesh, truth, trace, frame, geometry=g, spacing=9000.0),
            name=g, frame=frame, trace=trace,
            rms_min=0.01, width_min=9000.0, width_max=40000.0, exclude_within=6000.0,
        )
        for g in ("asc", "desc")
    ])
    inversion = SlipInversion(mesh, obs)
    return trace, mesh, obs, inversion, inversion.solve(smoothing=0.5, polarity=(-1, 0, 0))


@pytest.mark.parametrize("component", ["magnitude", "strike", "dip"])
def test_plot_slip_draws_every_element(solved, component):
    _, mesh, _, _, model = solved
    fig, ax = slip_plot.plot_slip(model, component=component)
    (collection,) = ax.collections
    assert len(collection.get_paths()) == mesh.n_elements
    assert ax.get_ylim()[1] == 0.0                      # surface at the top
    assert ax.get_xlabel().startswith("Distance along strike")
    matplotlib.pyplot.close(fig)


def test_plot_slip_rejects_an_unknown_component(solved):
    *_, model = solved
    with pytest.raises(ValueError, match="component must be"):
        slip_plot.plot_slip(model, component="tensile")


def test_plot_fit_shares_one_scale_between_data_and_model(solved):
    """An independently scaled model panel can make a poor fit look convincing."""
    trace, _, obs, _, model = solved
    fig, axes = slip_plot.plot_fit(model, track=obs.tracks[0], trace=trace)
    assert len(axes) == 3
    data_clim = axes[0].collections[0].get_clim()
    model_clim = axes[1].collections[0].get_clim()
    assert data_clim == model_clim
    # Each panel plots only the selected track.
    n_track = int(obs.track_mask(obs.tracks[0]).sum())
    assert axes[0].collections[0].get_offsets().shape[0] == n_track
    matplotlib.pyplot.close(fig)


def test_plot_fit_names_the_tracks_when_given_one_that_is_not_there(solved):
    """A wrong track name used to surface as a numpy reduction error.

    The empty selection reached the colour-scale line as `np.abs([]).max()`,
    raising "zero-size array to reduction operation maximum which has no
    identity" -- which names neither the track nor the model. The usual cause is
    a name read off a *different* Observations than the model carries.
    """
    *_, model = solved
    with pytest.raises(ValueError, match="No observations for track 'nope'"):
        slip_plot.plot_fit(model, track="nope")


def test_plot_samples(solved):
    trace, _, obs, _, _ = solved
    fig, ax = slip_plot.plot_samples(obs, trace=trace)
    assert ax.collections[0].get_offsets().shape[0] == obs.n
    matplotlib.pyplot.close(fig)


def test_plot_l_curve(solved):
    *_, inversion, _ = solved
    curve, _ = inversion.l_curve([0.2, 1.0, 5.0], polarity=(-1, 0, 0))
    fig, ax = slip_plot.plot_l_curve(curve)
    line = ax.lines[0]
    assert line.get_xydata().shape == (3, 2)
    matplotlib.pyplot.close(fig)


# -- the mesh, before any model exists ----------------------------------------

@pytest.mark.parametrize("color", ["area", "depth", "dip"])
def test_plot_mesh_draws_every_element_on_both_panels(solved, color):
    """``plot_mesh`` takes a bare mesh -- it is the figure to look at *before*
    committing to an inversion, which is while the mesh can still be changed."""
    trace, mesh, _, _, _ = solved
    fig, (map_ax, sec_ax) = slip_plot.plot_mesh(mesh, trace=trace, color=color)
    for ax in (map_ax, sec_ax):
        (collection,) = [c for c in ax.collections]
        assert len(collection.get_paths()) == mesh.n_elements
        assert collection.get_array().size == mesh.n_elements
    # The trace is drawn on the map panel only; the section is in (s, z).
    assert len(map_ax.lines) == 1
    assert not sec_ax.lines


def test_plot_mesh_panels_use_different_coordinates(solved):
    """Map view is ENU metres, the section is (arc length, depth) -- confusing the
    two would still draw a plausible-looking figure."""
    trace, mesh, _, _, _ = solved
    _, (map_ax, sec_ax) = slip_plot.plot_mesh(mesh)
    # Section: arc length against depth, both read straight off `params`.
    assert sec_ax.get_ylim() == (pytest.approx(mesh.params[:, 1].min() / 1e3), 0.0)
    assert sec_ax.get_xlim()[0] == pytest.approx(mesh.params[:, 0].min() / 1e3)

    # Map: east/north metres, which on a 238 km trace of 18 km depth spans a
    # visibly different range from the section's along-strike axis.
    east = mesh.vertices[:, :, 0] / 1e3
    assert map_ax.get_xlim()[0] <= east.min() and map_ax.get_xlim()[1] >= east.max()
    assert map_ax.get_ylim()[1] > 0.0        # northings straddle the frame origin
    assert map_ax.get_xlim()[0] < sec_ax.get_xlim()[0]


def test_plot_mesh_rejects_an_unknown_colour(solved):
    _, mesh, _, _, _ = solved
    with pytest.raises(ValueError, match="color must be one of"):
        slip_plot.plot_mesh(mesh, color="slip")


def test_param_vertices_matches_the_triangles(solved):
    _, mesh, _, _, _ = solved
    assert mesh.param_vertices.shape == (mesh.n_elements, 3, 2)
    np.testing.assert_array_equal(mesh.param_vertices, mesh.params[mesh.triangles])
