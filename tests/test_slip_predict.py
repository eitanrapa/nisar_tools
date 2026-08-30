"""Pins model-based downsampling (Wang & Fialko 2015, section 2).

The claim being tested is the paper's own and no larger: choosing cells from a
*predicted* field stops the sampler chasing noise, because the observed field's
within-cell scatter is dominated by atmosphere and does not shrink as the cell
shrinks. So the acceptance test plants a patch of noise away from the fault and
requires the data-driven sampler to pile samples into it while the model-driven
one does not.

It is tempting to assert something stronger -- better model, fewer samples -- and
that was tried and does **not** hold unconditionally. Measured here at 12 mm of
noise, data-driven vs model-driven correlation with the truth: 0.944/0.934 white,
0.895/0.874 correlated over 15 km, 0.852/**0.918** at 30 km. The advantage
appears only once the noise is long-wavelength enough that extra samples inside
one atmospheric cell carry no independent information. A synthetic with a single
correlation length cannot settle that, so it is recorded rather than asserted.

The near-fault retention gets its own test because the failure it prevents is
invisible here: an initial model with little shallow slip predicts a smooth near
field, the quadtree coarsens exactly there, and the *next* iteration invents
shallow slip that nothing contradicts.
"""

import numpy as np
import pytest
from slip_synthetic import forward_los_stack, tapered_slip

from nisar_tools.slip import (
    FaultMesh,
    FaultTrace,
    Observations,
    SlipInversion,
    iterate_sampling,
    model_rms_min,
    predicted_los,
    resample_all,
)

_LON = np.array([-68.681, -68.300, -67.900, -67.500, -67.100, -66.700, -66.523])
_LAT = np.array([10.410, 10.490, 10.540, 10.565, 10.595, 10.620, 10.630])

_SAMPLING = dict(rms_min=0.006, width_min=6000.0, width_max=40000.0,
                 exclude_within=5000.0)


@pytest.fixture(scope="module")
def setup():
    trace = FaultTrace(_LON, _LAT, name="test_fault")
    frame = trace.local_frame()
    mesh = FaultMesh.vertical(trace, frame, max_depth=20e3, edge_length=6e3)
    truth = tapered_slip(mesh, peak=-2.0)
    scenes = resample_all(
        {g: forward_los_stack(mesh, truth, trace, frame, geometry=g, spacing=5000.0,
                              noise=0.004, nan_rows=3, seed=1)
         for g in ("asc", "desc")},
        frame, spacing=5000.0,
    )
    return trace, frame, mesh, truth, scenes


@pytest.fixture(scope="module")
def coarse(setup):
    """A deliberately under-sampled first model -- the loop's round zero."""
    trace, frame, mesh, _, scenes = setup
    obs = Observations.concat([
        Observations.from_los(stack, name=name, frame=frame, trace=trace,
                              **{**_SAMPLING, "width_min": 18000.0})
        for name, stack in scenes.items()
    ])
    return obs, SlipInversion(mesh, obs).solve(smoothing=0.3)


# -- the predicted field -------------------------------------------------------

def test_predicted_los_is_on_the_stacks_grid_and_smooth(setup, coarse):
    trace, frame, mesh, _, scenes = setup
    _, model = coarse
    stack = scenes["asc"]
    field = predicted_los(model, stack, frame, spacing=5000.0)

    assert field.shape == stack.ds["los"].values.shape[1:]
    assert np.isfinite(field).mean() > 0.8
    # Noise-free by construction: the observed field carries 4 mm of noise on top
    # of the same signal, so its pixel-to-pixel differences are much larger.
    observed = np.asarray(stack.ds["los"].isel(pair=0).values, dtype=float)
    both = np.isfinite(field) & np.isfinite(observed)
    step_model = np.abs(np.diff(np.where(both, field, np.nan), axis=1))
    step_data = np.abs(np.diff(np.where(both, observed, np.nan), axis=1))
    assert np.nanmedian(step_model) < 0.5 * np.nanmedian(step_data)


def test_predicted_los_tracks_the_observed_field(setup, coarse):
    """It is a prediction of *this* scene, not a generic smooth surface.

    Measured away from the trace, where the comparison is clean: with the true
    slip substituted the far-field correlation is **1.0000** and the rms
    difference 0.1 mm, at coarse-grid spacings from 1 to 8 km.

    Over the whole raster it is much lower (~0.8 even for the true slip), and
    that is *not* a defect in the prediction: the observed raster has been
    resampled onto the shared lattice, which bilinearly smooths the displacement
    step across the fault, while the prediction evaluates it sharply. Evaluating
    the model directly at every pixel with no interpolation at all reproduces the
    same gap, which is what identifies the cause. ``exclude_within`` removes that
    band from the inversion anyway.
    """
    trace, frame, mesh, _, scenes = setup
    _, model = coarse
    for name, stack in scenes.items():
        field = predicted_los(model, stack, frame, spacing=5000.0)
        observed = np.asarray(stack.ds["los"].isel(pair=0).values, dtype=float)
        grid_x, grid_y = np.meshgrid(stack.ds["x"].values, stack.ds["y"].values)
        distance = trace.distance(grid_x.ravel(), grid_y.ravel(),
                                  frame).reshape(observed.shape)

        both = np.isfinite(field) & np.isfinite(observed)
        assert np.corrcoef(field[both], observed[both])[0, 1] > 0.6
        far = both & (distance > 20e3)
        assert np.corrcoef(field[far], observed[far])[0, 1] > 0.96


def test_predicted_los_uses_each_scenes_own_look_vectors(setup, coarse):
    """Ascending and descending see opposite LOS from the same ground motion, so
    projecting with the wrong geometry would be invisible in a shape comparison
    but wrong in sign."""
    trace, frame, mesh, _, scenes = setup
    _, model = coarse
    asc = predicted_los(model, scenes["asc"], frame, spacing=5000.0)
    desc = predicted_los(model, scenes["desc"], frame, spacing=5000.0)
    both = np.isfinite(asc) & np.isfinite(desc)
    assert np.corrcoef(asc[both], desc[both])[0, 1] < 0.0


def test_model_rms_min_scales_with_the_field(setup, coarse):
    trace, frame, mesh, _, scenes = setup
    _, model = coarse
    field = predicted_los(model, scenes["asc"], frame, spacing=5000.0)
    assert model_rms_min(field, 0.02) == pytest.approx(
        0.02 * np.nanstd(field[np.isfinite(field)]), rel=1e-9)
    assert model_rms_min(field, 0.04) == pytest.approx(2 * model_rms_min(field, 0.02))
    with pytest.raises(ValueError, match="empty"):
        model_rms_min(np.full(4, np.nan))


# -- splitting on it -----------------------------------------------------------

def test_field_chooses_cells_but_data_fills_them(setup, coarse):
    """``field`` must never leak into the values handed to the inversion."""
    trace, frame, mesh, _, scenes = setup
    _, model = coarse
    stack = scenes["asc"]
    field = predicted_los(model, stack, frame, spacing=5000.0)
    obs = Observations.from_los(stack, name="asc", frame=frame, trace=trace,
                                field=field, **_SAMPLING)

    observed = np.asarray(stack.ds["los"].isel(pair=0).values, dtype=float)
    finite = observed[np.isfinite(observed)]
    assert obs.ds["los"].values.min() >= finite.min() - 1e-9
    assert obs.ds["los"].values.max() <= finite.max() + 1e-9

    # `std` stays a *data* diagnostic. Reported over the multi-pixel cells only:
    # a single-pixel cell has no scatter by definition and `_reduce_cells` gives
    # it 0.0, and at `width_min` on this grid many cells are one pixel.
    scatter = obs.ds["std"].values
    multi = scatter > 0
    assert multi.any()
    # Consistent with the 4 mm of noise planted in these scenes; a `std` computed
    # from the noise-free model field would be orders of magnitude smaller.
    assert np.median(scatter[multi]) > 1e-3
    assert obs.ds.attrs["quadtree"]["field"] == "model"


def test_a_constant_field_stops_splitting_however_noisy_the_data(setup):
    """The whole point of the change, as a single assertion.

    With a flat model there is nothing to resolve, so every cell should come back
    at ``width_max`` -- even though the observed field underneath is pure noise,
    which is exactly what drives the data-driven sampler to ``width_min``.
    """
    from nisar_tools import LOSStack

    trace, frame, mesh, _, scenes = setup
    ds = scenes["asc"].ds.copy(deep=True)
    rng = np.random.default_rng(0)
    noisy = np.where(np.isfinite(ds["los"].values),
                     rng.normal(0.0, 0.05, ds["los"].shape), np.nan)
    ds["los"] = (ds["los"].dims, noisy.astype(np.float32))
    stack = LOSStack(ds)

    data_driven = Observations.from_los(stack, name="asc", frame=frame, trace=trace,
                                        **_SAMPLING)
    model_driven = Observations.from_los(
        stack, name="asc", frame=frame, trace=trace,
        field=np.zeros_like(noisy[0]), **_SAMPLING,
    )
    assert model_driven.n < data_driven.n / 10
    # Nothing split at all: every cell is at the top of the reachable ladder,
    # which on this raster tops out at 30-32.5 km rather than the 40 km bound.
    assert model_driven.ds["cell_size"].values.min() > 4 * _SAMPLING["width_min"]


def test_refine_within_holds_the_near_fault_density(setup):
    """A flat model near the trace must not be allowed to coarsen it.

    Without this the next iteration has no near-field data at all, and shallow
    slip becomes unconstrained rather than merely uncertain -- the paper's
    "spurious shallow slip".
    """
    trace, frame, mesh, _, scenes = setup
    stack = scenes["asc"]
    flat = np.zeros_like(np.asarray(stack.ds["los"].isel(pair=0).values, dtype=float))

    plain = Observations.from_los(stack, name="asc", frame=frame, trace=trace,
                                  field=flat, **_SAMPLING)
    refined = Observations.from_los(stack, name="asc", frame=frame, trace=trace,
                                    field=flat, refine_within=25e3, **_SAMPLING)
    assert refined.n > 5 * plain.n

    distance = trace.distance(refined.ds["x"].values, refined.ds["y"].values, frame)
    size = refined.ds["cell_size"].values
    # Compared well inside and well outside the band, not either side of its edge:
    # a cell is refined when it *overlaps* the band, so a sample whose centroid
    # sits just beyond 25 km may legitimately still be small.
    near, far = distance < 15e3, distance > 40e3
    assert near.sum() > 0 and far.sum() > 0
    assert np.median(size[near]) <= 1.5 * _SAMPLING["width_min"]
    assert np.median(size[far]) > 3 * np.median(size[near])
    assert refined.ds.attrs["quadtree"]["refine_within"] == 25e3


def test_field_shape_is_checked(setup, coarse):
    trace, frame, mesh, _, scenes = setup
    with pytest.raises(ValueError, match="but the stack's grid is"):
        Observations.from_los(scenes["asc"], name="asc", frame=frame, trace=trace,
                              field=np.zeros((3, 4)), **_SAMPLING)


def test_data_driven_sampling_is_untouched(setup):
    """``field=None`` must reproduce the old behaviour exactly, attrs included --
    the persist hash keys off ``quadtree``."""
    trace, frame, mesh, _, scenes = setup
    obs = Observations.from_los(scenes["asc"], name="asc", frame=frame, trace=trace,
                                **_SAMPLING)
    assert "field" not in obs.ds.attrs["quadtree"]
    assert "refine_within" not in obs.ds.attrs["quadtree"]


# -- the loop ------------------------------------------------------------------

def test_iterate_sampling_converges_and_records_why(setup):
    trace, frame, mesh, truth, scenes = setup
    obs, model, history = iterate_sampling(
        scenes, mesh, trace, frame, {n: dict(_SAMPLING) for n in scenes},
        max_rounds=4, spacing=4000.0, solve_kwargs={"smoothing": 0.3}, verbose=False,
    )
    assert history[0]["field"] == "data"
    assert all(h["field"] == "model" for h in history[1:])
    assert len(history) <= 5
    # It stopped because the model stopped moving, not because it ran out of rounds.
    assert history[-1]["max_change"] <= 0.01
    assert model.converged
    assert np.corrcoef(truth[:mesh.n_elements], model.strike_slip)[0, 1] > 0.95


def test_the_history_records_the_mesh_because_it_can_now_move(setup):
    """A sample count read beside a silently changed element count means nothing."""
    trace, frame, mesh, _, scenes = setup
    _, _, history = iterate_sampling(
        scenes, mesh, trace, frame, {n: dict(_SAMPLING) for n in scenes},
        max_rounds=1, spacing=4000.0, solve_kwargs={"smoothing": 0.3}, verbose=False,
    )
    assert all(h["n_elements"] == mesh.n_elements for h in history)
    assert all(h["n_nodes"] == mesh.n_nodes for h in history)


def test_the_two_way_loop_remeshes_and_still_recovers_the_slip(setup):
    """Sample from the model, size the mesh from the data, and re-solve.

    The claim is not that adapting beats a fixed mesh -- on a fixture whose
    uniform mesh is already well matched to its data it need not. It is that the
    loop *converges*: the mesh stays the size it was asked to be, the recovery
    holds, and the run does not drift into a degenerate model while reporting
    success, which is exactly what it did before ``budget`` existed.
    """
    trace, frame, mesh, truth, scenes = setup
    _, model, history = iterate_sampling(
        scenes, mesh, trace, frame, {n: dict(_SAMPLING) for n in scenes},
        max_rounds=3, spacing=4000.0, solve_kwargs={"smoothing": 0.3}, verbose=False,
        mesh_adapt={"geometry": {"uniform_dip": 90.0, "edge_length": 6e3}},
    )
    counts = [h["n_elements"] for h in history]
    assert counts[0] == mesh.n_elements
    assert counts[-1] != counts[0], "the mesh was supposed to be rebuilt"
    for count in counts[1:]:
        assert 0.6 * mesh.n_elements <= count <= 1.7 * mesh.n_elements, counts

    assert model.variance_reduction > 80.0
    assert np.abs(model.element_slip).max() > 0.5, "not a degenerate model"
    assert model.mesh.n_elements == counts[-1]


def test_the_loop_runs_on_an_unstructured_mesh(setup):
    """``structure="unstructured"`` reaches the loop and the result stays sane.

    The mesh it builds has no depth levels at all, so this also checks that
    everything downstream of the layout -- the basis, the boundary constraint,
    the probe-lattice convergence test -- is indifferent to losing the rows.
    """
    trace, frame, mesh, truth, scenes = setup
    _, model, history = iterate_sampling(
        scenes, mesh, trace, frame, {n: dict(_SAMPLING) for n in scenes},
        max_rounds=3, spacing=4000.0, solve_kwargs={"smoothing": 0.3}, verbose=False,
        mesh_adapt={"geometry": {"uniform_dip": 90.0, "edge_length": 6e3},
                    "structure": "unstructured"},
    )
    assert model.mesh.attrs["layout"] == "adapted_unstructured"
    counts = [h["n_elements"] for h in history]
    for count in counts[1:]:
        assert 0.6 * mesh.n_elements <= count <= 1.7 * mesh.n_elements, counts
    assert model.variance_reduction > 80.0
    assert np.abs(model.element_slip).max() > 0.5, "not a degenerate model"


def test_a_remesh_switches_the_convergence_test_to_the_probe_lattice(setup):
    """``max |element_slip - previous|`` cannot survive a change of mesh.

    Two rounds on different meshes give different-length vectors, so the number
    is measured on a fixed ``(s, z)`` lattice belonging to neither. With the mesh
    held fixed the element-space number is kept verbatim, so an existing run
    prints exactly what it printed before -- that equivalence is what is checked.
    """
    trace, frame, mesh, _, scenes = setup
    kwargs = dict(max_rounds=1, spacing=4000.0, solve_kwargs={"smoothing": 0.3},
                  verbose=False)
    _, model, plain = iterate_sampling(
        scenes, mesh, trace, frame, {n: dict(_SAMPLING) for n in scenes}, **kwargs)
    # `sampling_field` alone builds the probe lattice without moving the mesh.
    _, probed, both = iterate_sampling(
        scenes, mesh, trace, frame, {n: dict(_SAMPLING) for n in scenes},
        sampling_field="model", **kwargs)

    assert plain[-1]["max_change"] == pytest.approx(both[-1]["max_change"])
    np.testing.assert_allclose(model.element_slip, probed.element_slip)


def test_sampling_on_resolution_concentrates_samples_near_the_fault(setup):
    """Lohman & Simons' field, through the sampler that already accepts one.

    What the knob does is move samples, not add them, and the measurement says so
    on this fixture: ``"model"`` 1974 samples with 33.5% inside 20 km of the
    trace, ``"resolution"`` 890 with **74.4%**, ``"both"`` 1663 with 39.8%. The
    resolution field is largest where surface data sees the deep, unresolved
    elements best, which is close in.

    ⚠️ Deliberately **not** asserted: that ``"both"`` samples more than either.
    ``rms_min`` is a fraction of the field's own global scatter, so adding a
    field that is large but smooth raises the split threshold everywhere without
    raising a cell's local scatter -- the count can go either way, and it does.
    """
    trace, frame, mesh, _, scenes = setup
    near, counts = {}, {}
    for field in ("model", "resolution", "both"):
        obs, _, _ = iterate_sampling(
            scenes, mesh, trace, frame, {n: dict(_SAMPLING) for n in scenes},
            max_rounds=1, spacing=4000.0, solve_kwargs={"smoothing": 0.3},
            sampling_field=field, verbose=False)
        distance = trace.distance(obs.ds["x"].values, obs.ds["y"].values, frame)
        near[field] = float((distance < 20e3).mean())
        counts[field] = obs.n
        assert obs.n > 0

    assert near["resolution"] > 2.0 * near["model"], near
    assert near["model"] < near["both"] < near["resolution"], near
    assert len(set(counts.values())) == 3, counts


def test_an_unknown_sampling_field_is_refused_by_name(setup):
    trace, frame, mesh, _, scenes = setup
    with pytest.raises(ValueError, match="sampling_field"):
        iterate_sampling(scenes, mesh, trace, frame,
                         {n: dict(_SAMPLING) for n in scenes},
                         sampling_field="curvature", verbose=False)


def test_mesh_adapt_without_a_geometry_is_refused_by_name(setup):
    """A layout says where the nodes go, not what the fault is."""
    trace, frame, mesh, _, scenes = setup
    with pytest.raises(ValueError, match="geometry"):
        iterate_sampling(scenes, mesh, trace, frame,
                         {n: dict(_SAMPLING) for n in scenes},
                         mesh_adapt={"gain": 0.5}, verbose=False)


def test_model_sampling_does_not_chase_a_patch_of_noise(setup, coarse):
    """The acceptance test, and it is the paper's own claim -- no more.

    Wang & Fialko motivate the scheme as avoiding "over-sampling in areas with
    large phase gradients due to noise (atmospheric delays, decorrelation,
    unwrapping errors, etc.)". So that is what is measured: plant a rough patch
    far from the fault, and require the data-driven sampler to pile samples into
    it while the model-driven one does not.

    ⚠️ Deliberately **not** asserted here: that the model-driven sampler gives a
    better slip model, or uses fewer samples. Measured on this fixture at
    2 tracks with 12 mm of noise, data-driven vs model-driven correlation with
    the truth came out 0.944/0.934 for white noise, 0.895/0.874 for noise
    correlated over 15 km, and 0.852/**0.918** at 30 km. It wins only once the
    noise is long-wavelength enough that extra samples inside one atmospheric
    cell stop carrying independent information -- which is the regime real
    interferograms are in, but is not something a synthetic with a single
    correlation length can be trusted to prove. The sample count likewise depends
    entirely on ``rms_fraction`` against the scene's noise, in either direction.
    """
    from nisar_tools import LOSStack

    trace, frame, mesh, _, scenes = setup
    _, model = coarse
    stack = scenes["asc"]

    grid_x, grid_y = np.meshgrid(stack.ds["x"].values, stack.ds["y"].values)
    distance = trace.distance(grid_x.ravel(), grid_y.ravel(), frame).reshape(grid_x.shape)
    patch = distance > 60e3                      # well outside any real signal
    assert patch.sum() > 200

    ds = stack.ds.copy(deep=True)
    values = np.asarray(ds["los"].values, dtype=float)
    rng = np.random.default_rng(3)
    values[0][patch] += rng.normal(0.0, 0.05, int(patch.sum()))
    ds["los"] = (ds["los"].dims, values.astype(np.float32))
    dirty = LOSStack(ds)

    field = predicted_los(model, dirty, frame, spacing=4000.0)
    data_driven = Observations.from_los(dirty, name="asc", frame=frame, trace=trace,
                                        **_SAMPLING)
    model_driven = Observations.from_los(dirty, name="asc", frame=frame, trace=trace,
                                         field=field, **_SAMPLING)

    def share_in_patch(obs):
        far = trace.distance(obs.ds["x"].values, obs.ds["y"].values, frame) > 60e3
        return float(far.mean())

    # The patch is pure noise, so every sample spent there is wasted.
    assert share_in_patch(data_driven) > 3 * share_in_patch(model_driven)


def test_iterate_sampling_needs_kwargs_for_every_scene(setup):
    trace, frame, mesh, _, scenes = setup
    with pytest.raises(ValueError, match="no entry for"):
        iterate_sampling(scenes, mesh, trace, frame, {"asc": dict(_SAMPLING)},
                         max_rounds=1, verbose=False)


# -- residual convention -------------------------------------------------------

def test_residual_is_observed_minus_modelled(setup, coarse):
    obs, model = coarse
    np.testing.assert_allclose(model.residual, model.data - model.prediction)
    assert model.rms_misfit == pytest.approx(np.sqrt(np.mean(model.residual ** 2)))


def test_iterate_sampling_hands_back_a_reusable_inversion(setup, tmp_path):
    """``model.inversion`` is what lets the notebook sweep an L-curve over the
    *refined* sampling without paying for a second Green's matrix.

    Without it the caller has only ``obs``, and rebuilding ``SlipInversion`` from
    it re-assembles G -- measured at ~25 s for 1106 elements against 8000
    observations, and the loop has just built exactly that matrix internally.
    """
    trace, frame, mesh, _, scenes = setup
    obs, model, _ = iterate_sampling(
        scenes, mesh, trace, frame, {n: dict(_SAMPLING) for n in scenes},
        max_rounds=2, spacing=4000.0, solve_kwargs={"smoothing": 0.3}, verbose=False,
    )
    inversion = model.inversion
    assert inversion.obs.n == obs.n
    assert inversion.g is not None
    assert inversion.g.shape[0] == obs.n

    # Re-solving through the handle costs no assembly and agrees with a fresh solve.
    again = inversion.solve(smoothing=0.3)
    np.testing.assert_allclose(again.strike_slip, model.strike_slip, atol=1e-9)


def test_a_loaded_model_has_no_matrix_to_reuse(setup, tmp_path):
    """``G`` is deliberately not saved, so the handle comes back without one --
    readable, but not re-solvable. Better to say so than to hand back a stub that
    fails somewhere further in."""
    from nisar_tools.slip import SlipModel

    trace, frame, mesh, _, scenes = setup
    obs = Observations.from_los(scenes["asc"], name="asc", frame=frame, trace=trace,
                                **_SAMPLING)
    model = SlipInversion(mesh, obs).solve(smoothing=0.3)
    model.save(tmp_path / "m.slip.zip")

    back = SlipModel.load(tmp_path / "m.slip.zip")
    assert back.inversion.g is None
    np.testing.assert_allclose(back.residual, back.data - back.prediction)


def test_a_degenerate_model_warns_instead_of_quietly_converging(setup):
    """Over-smoothing makes the loop 'converge' on nothing, and it looks fine.

    A flat model predicts a flat field, ``model_rms_min`` then returns ~0, the
    quadtree splits on floating-point noise, and nothing changes between rounds
    because nothing is there -- so ``max_change`` goes to zero and the loop stops
    as if it had succeeded. Hit for real on the D134 scene at ``smoothing=200``:
    peak slip 0.004 m at a respectable-looking 22% variance reduction.
    """
    trace, frame, mesh, _, scenes = setup
    with pytest.warns(RuntimeWarning, match="effectively zero"):
        _, model, history = iterate_sampling(
            scenes, mesh, trace, frame, {n: dict(_SAMPLING) for n in scenes},
            max_rounds=1, spacing=8000.0,
            solve_kwargs={"smoothing": 1e6}, verbose=False,
        )
    assert np.abs(model.element_slip).max() < 1e-3
    # The trap: it still reports as converged, which is why the warning exists.
    assert history[-1]["converged"]


def test_an_unknown_mesh_structure_is_refused_before_the_first_solve(setup):
    """A typo in NISAR_ADAPT_STRUCTURE should not cost a round of sampling and
    solving before it surfaces."""
    trace, frame, mesh, truth, scenes = setup
    with pytest.raises(ValueError, match="must be 'rows' or 'unstructured'"):
        iterate_sampling(
            scenes, mesh, trace, frame, {n: dict(_SAMPLING) for n in scenes},
            max_rounds=2, verbose=False,
            mesh_adapt={"geometry": {"uniform_dip": 90.0}, "structure": "quadtree"},
        )


def test_the_loop_can_size_the_mesh_from_the_kernel_width(setup):
    """``size_from="kernel"`` reaches the re-mesh and produces a usable mesh."""
    trace, frame, mesh, truth, scenes = setup
    _, model, history = iterate_sampling(
        scenes, mesh, trace, frame, {n: dict(_SAMPLING) for n in scenes},
        max_rounds=2, spacing=4000.0, solve_kwargs={"smoothing": 0.3}, verbose=False,
        mesh_adapt={"geometry": {"uniform_dip": 90.0, "edge_length": 6e3},
                    "structure": "unstructured", "size_from": "kernel"},
    )
    assert model.mesh.n_elements > 0
    assert model.variance_reduction > 50.0
    assert np.abs(model.element_slip).max() > 0.0


def test_an_unknown_size_source_is_refused_before_the_first_solve(setup):
    trace, frame, mesh, truth, scenes = setup
    with pytest.raises(ValueError, match="must be 'diagonal' or 'kernel'"):
        iterate_sampling(
            scenes, mesh, trace, frame, {n: dict(_SAMPLING) for n in scenes},
            max_rounds=2, verbose=False,
            mesh_adapt={"geometry": {"uniform_dip": 90.0}, "size_from": "resolution"},
        )


def test_conserve_holds_the_element_count_over_a_long_loop(setup):
    """⚠️ The drift that three rounds hides and eight rounds shows.

    ``adapt_layout`` resolves ``budget="conserve"`` against whatever mesh it is
    handed, and ``_fit_budget`` accepts any build within 5% of the target without
    bisecting. So round N aimed at round N-1's *result*: a +/-5% random walk with
    no restoring force, measured on the real problem as 1696 -> 1759 -> ... ->
    1893, **+11.6% compounded over 8 rounds**, every single step inside tolerance
    and therefore invisible to any per-round check.

    Pinning "conserve" to the *starting* mesh once is the fix, and this is the
    test that would have caught it: it looks at the trend across many rounds
    rather than at any one step.
    """
    trace, frame, mesh, _, scenes = setup
    _, _, history = iterate_sampling(
        scenes, mesh, trace, frame, {n: dict(_SAMPLING) for n in scenes},
        max_rounds=8, spacing=4000.0, solve_kwargs={"smoothing": 0.3}, verbose=False,
        mesh_adapt={"geometry": {"uniform_dip": 90.0, "edge_length": 6e3},
                    "rounds": 8},
    )
    counts = np.array([h["n_elements"] for h in history[1:]], dtype=float)
    assert counts.size >= 6, counts

    # Every round within tolerance of the ORIGINAL count, not of its predecessor.
    assert np.all(np.abs(counts - mesh.n_elements) <= 0.12 * mesh.n_elements), counts
    # And no trend: a random walk would show one, a pinned target should not.
    slope = np.polyfit(np.arange(counts.size), counts, 1)[0]
    assert abs(slope) < 0.02 * mesh.n_elements, (slope, counts)


def test_the_size_clamps_do_not_follow_the_mesh_they_are_bounding(setup):
    """``size_min``/``size_max`` default from the *current* mesh, so they drift too.

    ``max_ratio`` bounds the per-step ratio, not the cumulative one, so the size
    spread widened without bound -- 6.5x after 3 rounds, 30x after 8. Resolved
    against the starting mesh unless the caller was explicit.
    """
    from nisar_tools.slip.predict import _adapt_spec

    _, _, mesh, _, _ = setup
    spec = _adapt_spec({"geometry": {}}, 4, mesh=mesh)

    from nisar_tools.slip.resolution import element_size
    current = element_size(mesh)
    assert spec["size_min"] == pytest.approx(current.min() / 3.0)
    assert spec["size_max"] == pytest.approx(current.max() * 3.0)
    assert spec["budget"] == mesh.n_elements

    # An explicit value is never overridden.
    explicit = _adapt_spec({"geometry": {}, "size_min": 123.0, "budget": 40}, 4,
                           mesh=mesh)
    assert explicit["size_min"] == 123.0
    assert explicit["budget"] == 40
