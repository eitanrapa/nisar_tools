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
    with pytest.raises(ValueError, match="component must be"):
        slip_plot.plot_slip_3d(model, component="tensile")


@pytest.mark.parametrize("component", ["magnitude", "strike", "dip"])
def test_plot_slip_3d_draws_every_element(solved, component):
    trace, mesh, _, _, model = solved
    fig, ax = slip_plot.plot_slip_3d(model, component=component, trace=trace)

    assert hasattr(ax, "add_collection3d"), "must be a 3-D axes"
    (collection,) = [c for c in ax.collections if hasattr(c, "do_3d_projection")]
    # A Poly3DCollection has no 2-D paths until it projects, so this also pins
    # that the figure actually renders rather than merely being assembled.
    fig.canvas.draw()
    assert len(collection.get_paths()) == mesh.n_elements
    assert ax.get_zlabel().startswith("Depth")
    # Depth runs from the base of the fault up to the free surface.
    assert ax.get_zlim()[1] == pytest.approx(mesh.nodes[:, 2].max() / 1e3)
    matplotlib.pyplot.close(fig)


def test_plot_slip_3d_matches_the_unrolled_colour_scale(solved):
    """The two views are meant to be compared by eye, so a signed component has
    to land on the same symmetric scale in both."""
    *_, model = solved
    fig2d, ax2d = slip_plot.plot_slip(model, component="strike")
    fig3d, ax3d = slip_plot.plot_slip_3d(model, component="strike")

    flat = ax2d.collections[0]
    solid = [c for c in ax3d.collections if hasattr(c, "do_3d_projection")][0]
    assert solid.get_clim() == flat.get_clim()
    assert solid.get_cmap().name == flat.get_cmap().name == "RdBu_r"
    matplotlib.pyplot.close(fig2d)
    matplotlib.pyplot.close(fig3d)


def test_plot_slip_3d_exaggerates_every_axis_but_the_longest(solved):
    """At true scale a 200 km fault 18 km deep is an unreadable sliver.

    The stretch is what makes the dip visible, and leaving the long axis alone is
    what keeps it from simply cancelling out.
    """
    *_, model = solved
    fig, ax = slip_plot.plot_slip_3d(model, exaggeration=4.0)
    aspect = np.asarray(ax.get_box_aspect(), dtype=float)

    assert aspect.argmax() == 0, "along strike (east here) stays the long axis"
    true_scale = slip_plot.plot_slip_3d(model, exaggeration=1.0)[1].get_box_aspect()
    # matplotlib renormalises what it is handed, so only the *relative* stretch
    # between the axes is observable -- which is the whole content of the knob.
    ratio = aspect / np.asarray(true_scale, dtype=float)
    np.testing.assert_allclose(ratio / ratio[0], [1.0, 4.0, 4.0])
    matplotlib.pyplot.close(fig)


def test_plot_slip_3d_refuses_a_flat_axes(solved):
    *_, model = solved
    _, flat = matplotlib.pyplot.subplots()
    with pytest.raises(TypeError, match="needs a 3-D axes"):
        slip_plot.plot_slip_3d(model, ax=flat)
    matplotlib.pyplot.close("all")


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


def test_plot_mesh_draws_an_arbitrary_per_element_field(solved):
    """How model resolution gets drawn -- the field worth seeing before solving."""
    _, mesh, _, _, _ = solved
    values = np.linspace(0.0, 1.0, mesh.n_elements)

    figure, axes = slip_plot.plot_mesh(mesh, color=values, label="model resolution")
    for axis in axes:
        collection = axis.collections[0]
        np.testing.assert_allclose(collection.get_array(), values)
    matplotlib.pyplot.close(figure)

    with pytest.raises(ValueError, match="entries for"):
        slip_plot.plot_mesh(mesh, color=values[:-1], label="short")


def test_param_vertices_matches_the_triangles(solved):
    _, mesh, _, _, _ = solved
    assert mesh.param_vertices.shape == (mesh.n_elements, 3, 2)
    np.testing.assert_array_equal(mesh.param_vertices, mesh.params[mesh.triangles])


# -- nodal slip along the edges, and where the quadrature spends -------------

@pytest.fixture(scope="module")
def solved_nodal(solved):
    """The same scene solved on the nodal basis, which is what has node values."""
    trace, mesh, obs, _, _ = solved
    inversion = SlipInversion(mesh, obs, basis="node")
    return inversion.solve(smoothing=0.5, polarity=(-1, 0, 0))


def test_slip_edges_draws_a_segment_per_subdivision(solved_nodal):
    """Every edge is subdivided, and the shading interpolates between its nodes.

    The segment count is the test that the subdivision happened at all: a
    per-edge mean would give one segment per edge and lose exactly the linear
    variation this view exists to show.
    """
    from matplotlib.collections import LineCollection

    model = solved_nodal
    fig, ax = slip_plot.plot_slip_edges(model, component="strike", segments=4)
    coll = [c for c in ax.collections if isinstance(c, LineCollection)][0]
    tri = np.asarray(model.mesh.triangles)
    n_edges = len({(min(a, b), max(a, b))
                   for e in tri
                   for a, b in ((e[0], e[1]), (e[1], e[2]), (e[2], e[0]))})
    assert len(coll.get_segments()) == 4 * n_edges

    value = np.asarray(model.strike_slip)
    shading = coll.get_array()
    assert shading.min() >= value.min() - 1e-9
    assert shading.max() <= value.max() + 1e-9
    matplotlib.pyplot.close(fig)


def test_slip_edges_refuses_an_element_basis(solved):
    """There are no nodal values to interpolate, so it must not invent some."""
    *_, model = solved
    with pytest.raises(ValueError, match="basis='node'"):
        slip_plot.plot_slip_edges(model)


def test_quadrature_order_refuses_an_engine_without_one(solved):
    """The half-space engine is a closed form -- it cuts nothing into points."""
    from nisar_tools.slip import HalfSpaceTDE

    trace, mesh, obs, _, _ = solved
    with pytest.raises(TypeError, match="no quadrature"):
        slip_plot.plot_quadrature_order(HalfSpaceTDE(), mesh, obs)


# -- the ramp, shown separately from the model it is bundled into -------------

@pytest.fixture(scope="module")
def solved_with_ramp():
    """The same problem solved with per-track linear nuisance columns.

    A real run almost always has them -- an unwrapped interferogram carries an
    arbitrary constant and usually a trend on top -- so this, not `solved`, is
    the layout `plot_fit` draws in practice.
    """
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
    inversion = SlipInversion(mesh, obs, ramp="linear")
    return trace, obs, inversion.solve(smoothing=0.5, polarity=(-1, 0, 0))


def test_ramp_prediction_and_fault_prediction_split_the_model(solved_with_ramp):
    """The two halves have to add back up to what the solver actually predicted,
    or the extra panels would be drawing a different model than the top row."""
    _, obs, model = solved_with_ramp
    assert model.ramp.size == 3 * len(obs.tracks)      # offset + two gradients
    np.testing.assert_allclose(
        model.ramp_prediction + model.fault_prediction, model.prediction,
        atol=1e-12,
    )
    # The ramp is a nuisance surface, not a copy of the fit.
    assert np.abs(model.ramp_prediction).max() > 0.0


def test_ramp_prediction_is_zero_without_ramp_columns(solved):
    *_, model = solved
    np.testing.assert_array_equal(model.ramp_prediction,
                                  np.zeros_like(model.data))
    np.testing.assert_allclose(model.fault_prediction, model.prediction)


def test_plot_fit_adds_a_ramp_row_when_there_is_a_ramp(solved_with_ramp):
    trace, obs, model = solved_with_ramp
    fig, axes = slip_plot.plot_fit(model, track=obs.tracks[0], trace=trace)
    assert axes.shape == (2, 3)
    titles = [ax.get_title() for ax in axes.ravel()]
    assert titles[:3] == ["Data", "Model", "Residual"]
    assert titles[3] == "Ramp"
    assert "ramp" in titles[4] and "slip only" in titles[5]

    n_track = int(obs.track_mask(obs.tracks[0]).sum())
    for ax in axes.ravel():
        assert ax.collections[0].get_offsets().shape[0] == n_track

    # The ramp shares the data's scale, so its size can be read off directly;
    # the two deramped panels share their own.
    assert axes[0, 0].collections[0].get_clim() == axes[1, 0].collections[0].get_clim()
    assert axes[1, 1].collections[0].get_clim() == axes[1, 2].collections[0].get_clim()
    assert "ramp spans" in fig._suptitle.get_text()
    matplotlib.pyplot.close(fig)


def test_plot_fit_keeps_the_three_panel_form_on_request(solved_with_ramp):
    trace, obs, model = solved_with_ramp
    fig, axes = slip_plot.plot_fit(model, track=obs.tracks[0], ramp=False)
    assert len(axes) == 3
    matplotlib.pyplot.close(fig)


def test_a_reloaded_model_still_separates_its_ramp(solved_with_ramp, tmp_path):
    """The Green's matrix is not saved, but the ramp columns are a pure function
    of the observations -- so the split survives a round trip."""
    from nisar_tools.slip import SlipModel

    _, _, model = solved_with_ramp
    path = model.save(tmp_path / "model.slip.zip")
    back = SlipModel.load(path)
    np.testing.assert_allclose(back.ramp_prediction, model.ramp_prediction,
                               atol=1e-9)
    np.testing.assert_allclose(back.fault_prediction, model.fault_prediction,
                               atol=1e-9)


def test_an_old_model_without_the_recorded_ramp_kind_still_splits(solved_with_ramp):
    """Models saved before the options carried ``ramp`` are read back by
    counting columns per track, which is one-to-one with the degree."""
    from nisar_tools.slip.regularize import ramp_kind_for

    _, obs, model = solved_with_ramp
    assert ramp_kind_for(model.ramp.size, len(obs.tracks)) == "linear"
    assert ramp_kind_for(0, len(obs.tracks)) == "none"
    with pytest.raises(ValueError, match="match no ramp degree"):
        ramp_kind_for(4 * len(obs.tracks), len(obs.tracks))


def test_plot_slip_can_draw_the_resolving_length(solved):
    """A slip map drawn at element scale shows more detail than the data
    supports and nothing in the picture says so; ``kernels=`` is what says it."""
    from nisar_tools.slip import resolution as R

    _, _, obs, inversion, model = solved
    report = R.resolution_report(inversion, 0.5)
    fig, ax = slip_plot.plot_slip(model, kernels=report)
    assert "resolving length" in ax.get_title()
    # the bar itself, one kernel width long
    (line,) = [ln for ln in ax.lines if ln.get_linewidth() == 3]
    span = np.ptp(line.get_xdata()) * 1e3
    assert span == pytest.approx(
        float(np.median(report["width_along"].values)), rel=1e-6)
    matplotlib.pyplot.close(fig)


# -- a nodal model must not be drawn as a piecewise-constant one -------------

def test_a_nodal_model_is_drawn_as_the_linear_field_it_is(solved_nodal):
    """⚠️ Flat shading draws a *different model* -- the user's objection.

    A nodal basis expands slip in piecewise-linear tents, so the field varies
    linearly across every triangle. Colouring each triangle one colour draws the
    piecewise-constant field a constant-slip TDE would give, and a reader cannot
    tell the two apart by looking. Gouraud interpolation from the three nodal
    values is not a smoothing convenience; it is the exact field.
    """
    from matplotlib.collections import PolyCollection
    from matplotlib.collections import TriMesh

    model = solved_nodal
    assert model.basis.name == "node"

    fig, ax = slip_plot.plot_slip(model)
    # `tripcolor(shading="gouraud")` yields a TriMesh; flat shading a PolyCollection.
    assert any(isinstance(c, TriMesh) for c in ax.collections), (
        "a nodal model was drawn with flat per-triangle shading")
    matplotlib.pyplot.close(fig)

    fig, ax = slip_plot.plot_slip(model, shading="flat")
    assert any(isinstance(c, PolyCollection) and not isinstance(c, TriMesh)
               for c in ax.collections), "shading='flat' did not fall back"
    matplotlib.pyplot.close(fig)


def test_the_title_quotes_the_value_the_panel_actually_draws(solved_nodal):
    """Title and colourbar must agree. They differed by 51% on the real model.

    Flat shading reduces the peak to the element mean while `model.max_slip` is
    the peak of the continuous field -- 3.04 m against 4.59 m on the six-scene
    run, the larger number printed above a colourbar that stopped at the smaller.
    """
    model = solved_nodal
    nodal = float(np.hypot(model.strike_slip, model.dip_slip).max())

    fig, ax = slip_plot.plot_slip(model)                      # gouraud: draws the nodal field
    assert f"{nodal:.2f}" in ax.get_title()
    matplotlib.pyplot.close(fig)

    fig, ax = slip_plot.plot_slip(model, shading="flat")      # flat: draws the element mean
    element = float(np.linalg.norm(model.element_slip, axis=1).max())
    title = ax.get_title()
    assert f"{element:.2f}" in title
    if abs(nodal - element) > 0.01 * nodal:
        assert "nodal" in title, "the nodal peak must be labelled when it differs"
    matplotlib.pyplot.close(fig)


def test_an_element_model_is_still_drawn_flat(solved):
    """An element basis really is piecewise constant, so flat is correct there."""
    from matplotlib.collections import TriMesh

    _, mesh, obs, _, _ = solved
    model = SlipInversion(mesh, obs).solve(smoothing=0.5, polarity=(-1, 0, 0))
    assert model.basis.name == "element"
    fig, ax = slip_plot.plot_slip(model)
    assert not any(isinstance(c, TriMesh) for c in ax.collections)
    matplotlib.pyplot.close(fig)
