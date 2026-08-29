"""Model-based downsampling: let the model decide where the data are worth having.

A quadtree run on observed interferograms splits wherever the phase is rough, and
atmosphere is rough. Measured on the Venezuela scenes, the median within-cell
scatter was **11.8 mm** against an ``rms_min`` of 6 mm, leaving **47% of cells**
above threshold -- and because that scatter is a *pixel* statistic it does not
shrink as the cell shrinks, so the recursion could not stop on information and
simply ran down to ``width_min``. The result was 18 631 samples of mostly
atmosphere, densest exactly where the interferogram was worst.

Wang & Fialko (2015, §2) break the loop by quadtreeing a **synthetic**
interferogram computed from a preliminary slip model, then filling the resulting
cells from the observed data:

    An initial slip model was estimated from inversion of coarsely sampled LOS
    displacement maps. Synthetic interferograms were computed using the slip
    model, and down-sampled using the quad-tree curvature-based algorithm. The
    bounding coordinates of each resolution cell (bin) were then used to compute
    the average LOS displacements from the observed interferograms. A new slip
    model was then derived from the updated dataset. Usually, a few iterations
    are sufficient to achieve a solution that stops changing with subsequent
    iterations. To avoid spurious shallow slip, a relatively dense sampling
    around the fault trace was retained through all iterations.

Both halves of that last sentence are load-bearing. It is a **loop**, not one
pass; and the near-fault density has to be *held*, because an initial model with
little shallow slip predicts a smooth near field, the quadtree then coarsens
precisely where shallow slip needs constraining, and the next round is free to
invent it. That is :func:`~nisar_tools.slip.sampling.Observations.from_los`'s
``refine_within``.

The prediction is evaluated on a coarse grid and interpolated up, never on the
raster directly. Timed on the real trace, forward-modelling all 1.58 M pixels of
one ALOS scene takes **8.6 min at 124 elements and 80 min at 1148**; a 2 km grid
is ~90 s, and the predicted field's shortest wavelength is set by the element
size, so nothing is lost.
"""

import numpy as np

from .layout import MAX_COUNT_RATIO
from .sampling import Observations, _to_frame, stack_epsg

#: Coarse grid on which the elastic forward model is evaluated, in metres.
DEFAULT_SPACING = 2000.0

#: Near-fault band held at ``width_min`` through every round, as a multiple of
#: ``exclude_within`` -- which is itself half a terminal cell, so this reaches a
#: couple of cells either side of the trace.
REFINE_MULTIPLE = 4.0

#: Stop when no parameter moves more than this between rounds (metres of slip).
DEFAULT_TOL = 0.01


def predicted_los(model, los_stack, frame=None, spacing=DEFAULT_SPACING,
                  block=4096, exclude_within=None):
    """The LOS field ``model`` predicts on ``los_stack``'s own grid.

    Returns an ``(ny, nx)`` float array in the package's positive-toward-sensor
    convention, ready to hand to
    :meth:`~nisar_tools.slip.sampling.Observations.from_los` as ``field``.

    Evaluated on a ``spacing``-metre grid via
    :meth:`~nisar_tools.slip.inversion.SlipModel.surface_displacement` and
    interpolated up. The three ENU components are interpolated, **not** the scalar
    LOS: the look vectors already sit on the raster at full resolution, so
    projecting after interpolation keeps whatever across-swath variation the
    geometry has -- which for the wide-swath ALOS scenes here is an incidence
    angle running 25.7 to 49.1 degrees.
    """
    from scipy.interpolate import RegularGridInterpolator

    frame = frame or model.mesh.frame
    if frame is None:
        raise ValueError("predicted_los needs a LocalFrame; the mesh carries none")

    ds = los_stack.ds
    epsg = stack_epsg(ds)
    x_native = np.asarray(ds["x"].values, dtype=float)
    y_native = np.asarray(ds["y"].values, dtype=float)
    gx, gy = np.meshgrid(x_native, y_native)
    fx, fy = _to_frame(gx.ravel(), gy.ravel(), frame, epsg)

    pad = 2.0 * spacing
    bounds = (fx.min() - pad, fx.max() + pad, fy.min() - pad, fy.max() + pad)
    if exclude_within is None:
        # Only the singularity itself has to be avoided, not the near field: the
        # band excluded here becomes a NaN hole that `refine_within` would then be
        # unable to fill. Half a coarse cell is enough.
        exclude_within = 0.5 * spacing
    field = model.surface_displacement(
        spacing=spacing, bounds=bounds, exclude_within=exclude_within, block=block
    )

    cy = np.asarray(field["y"].values, dtype=float)
    cx = np.asarray(field["x"].values, dtype=float)
    order = np.argsort(cy)                       # the grid is north-up; interp wants ascending

    points = np.column_stack([fy, fx])
    enu = np.empty((fx.size, 3))
    for i, name in enumerate(("ux", "uy", "uz")):
        values = _fill_nearest(np.asarray(field[name].values, dtype=float))
        interp = RegularGridInterpolator(
            (cy[order], cx), values[order], bounds_error=False, fill_value=np.nan
        )
        enu[:, i] = interp(points)

    look = np.stack([
        np.asarray(ds[f"los_{c}"].values, dtype=float).ravel()
        for c in ("east", "north", "up")
    ], axis=1)
    return np.einsum("ij,ij->i", enu, look).reshape(gx.shape)


def iterate_sampling(scenes, mesh, trace, frame, sample_kwargs, coarse_kwargs=None,
                     inversion_kwargs=None, solve_kwargs=None, max_rounds=5,
                     tol=DEFAULT_TOL, spacing=DEFAULT_SPACING, refine_within=None,
                     rms_fraction=0.02, normalize="sqrt_count", mesh_adapt=None,
                     sampling_field="model", verbose=True):
    """Run the Wang & Fialko loop: sample, invert, re-sample from the model.

    ``scenes`` is a ``{name: LOSStack}`` mapping -- put them on one lattice first
    with :func:`nisar_tools.slip.resample.resample_all`, or their cell-size
    ladders are not comparable. ``sample_kwargs`` is ``{name: kwargs}`` for
    :meth:`~nisar_tools.slip.sampling.Observations.from_los`, normally built from
    :func:`~nisar_tools.slip.diagnostics.scene_report`.

    Round 0 is deliberately coarse and data-driven -- the paper's "coarsely
    sampled LOS displacement maps" -- and ``coarse_kwargs`` overrides
    ``sample_kwargs`` for it alone (default: ``width_min`` tripled, which costs
    roughly an order of magnitude in samples and is plenty for a first geometry).
    Every later round predicts each scene's LOS from the current model, re-samples
    on that, and re-solves.

    Stops when the slip field moves by no more than ``tol`` between rounds, or at
    ``max_rounds``. Returns ``(observations, model, history)``; ``history`` is a
    list of per-round dicts carrying ``n``, ``variance_reduction`` and
    ``max_change``, so that the "stops changing" criterion can be checked rather
    than assumed.

    **Two-way optimization.** ``mesh_adapt`` turns the loop from
    sample-invert-sample into sample-**remesh**-invert: each round re-sizes the
    elements from the model resolution the current data supports
    (:func:`~nisar_tools.slip.resolution.target_element_size`) and rebuilds the
    mesh on the new layout. It is a dict, and it must carry ``geometry`` --
    keyword arguments for :meth:`~nisar_tools.slip.mesh.FaultMesh.from_layout`
    naming what the fault *is* (``bottom_trace=``, ``uniform_dip=``, ...), since a
    layout says only where the nodes go. Everything else in the dict is passed to
    ``target_element_size``, ``structure`` (``"rows"``, the default, or
    ``"unstructured"`` for a layout with no depth levels at all -- see
    :func:`~nisar_tools.slip.resolution.adapt_layout`), plus ``rounds`` to stop
    adapting after N rounds and
    ``smoothing`` to evaluate resolution at a weight other than ``solve_kwargs``'.

    ``sampling_field`` chooses what the quadtree splits on: ``"model"`` (the
    default, Wang & Fialko's predicted displacement), ``"resolution"`` (Lohman &
    Simons -- where a sample would most improve model resolution), or ``"both"``,
    the sum of the two standardised fields. The two answer different questions:
    the first avoids spending samples on atmosphere, the second spends them where
    the model is unconstrained. It **moves** samples rather than adding them --
    measured on the two-scene fixture, ``"model"`` put 1974 samples with 33.5%
    inside 20 km of the trace and ``"resolution"`` put 890 with **74.4%**.

    ⚠️ ``"both"`` is not "at least as many samples as either", and cannot be:
    ``rms_min`` is a fraction of the field's *own global* scatter, so a field that
    is large but smooth raises the split threshold everywhere without raising any
    cell's local scatter.

    ⚠️ **A changing mesh invalidates the obvious convergence test.** ``max_change``
    was ``max |element_slip - previous|``, and two rounds on different meshes give
    different-length vectors. It is measured on a fixed ``(s, z)`` probe lattice
    instead whenever the mesh moves, which is mesh- and basis-independent; with
    the mesh held fixed the element-space number is kept verbatim, so existing runs
    print exactly what they printed before.

    ⚠️ **That probe number has an irreducible floor, which is why ``mesh_adapt``
    stops before the last round.** Resampling a piecewise-constant field onto a
    fixed lattice differs by roughly the element-boundary contrast however little
    the model moved: measured, two models fitting the *same* data on 240 and 232
    elements agree to a **2 mm median** and still differ by **0.97 m** at the worst
    probe point. So ``tol`` cannot be met while the mesh is moving, and
    ``mesh_adapt["rounds"]`` defaults to ``max_rounds - 1`` so the run finishes on a
    fixed mesh and converges on the exact criterion. Set it to ``max_rounds`` to
    keep adapting throughout, and expect the loop to run out of rounds rather than
    converge.
    """
    from .inversion import SlipInversion

    scenes = dict(scenes)
    if not scenes:
        raise ValueError("Need at least one scene to sample")
    missing = set(scenes) - set(sample_kwargs)
    if missing:
        raise ValueError(f"sample_kwargs has no entry for {sorted(missing)}")

    inversion_kwargs = dict(inversion_kwargs or {})
    solve_kwargs = dict(solve_kwargs or {})
    if sampling_field not in ("model", "resolution", "both"):
        raise ValueError(
            f"Unknown sampling_field {sampling_field!r}; expected 'model', "
            "'resolution' or 'both'"
        )
    adapt = _adapt_spec(mesh_adapt, max_rounds)
    probe = _probe_lattice(mesh) if (adapt or sampling_field != "model") else None
    if refine_within is None:
        refine_within = REFINE_MULTIPLE * max(
            float(kw.get("exclude_within", 0.0)) for kw in sample_kwargs.values()
        )

    def sample(name, stack, field=None, **extra):
        kwargs = dict(sample_kwargs[name])
        kwargs.update(extra)
        return Observations.from_los(
            stack, name=name, frame=frame, trace=trace, field=field,
            refine_within=refine_within if field is not None else 0.0, **kwargs
        )

    coarse = dict(coarse_kwargs) if coarse_kwargs is not None else None
    parts = []
    for name, stack in scenes.items():
        extra = coarse if coarse is not None else {
            "width_min": 3.0 * float(sample_kwargs[name].get("width_min", 1000.0))
        }
        parts.append(sample(name, stack, **extra))

    obs = Observations.concat(parts, normalize=normalize)
    inversion = SlipInversion(mesh, obs, **inversion_kwargs)
    model = inversion.solve(**solve_kwargs)
    history = [_entry(0, obs, model, mesh, "data", float("nan"))]
    if verbose:
        _report(history[-1])

    for round_index in range(1, int(max_rounds) + 1):
        _warn_if_degenerate(model, round_index)
        previous = model.element_slip.copy()
        previous_probe = None if probe is None else _probe_slip(mesh, model, probe)

        fields = {}
        for name, stack in scenes.items():
            fields[name] = _split_field(model, inversion, stack, frame, spacing,
                                        sampling_field, adapt, solve_kwargs)
        parts = [sample(name, stack, field=fields[name],
                        rms_min=model_rms_min(fields[name], rms_fraction))
                 for name, stack in scenes.items()]

        if adapt is not None and round_index <= adapt["rounds"]:
            mesh = _adapt_mesh(mesh, inversion, trace, frame, adapt, solve_kwargs)

        obs = Observations.concat(parts, normalize=normalize)
        inversion = SlipInversion(mesh, obs, **inversion_kwargs)
        model = inversion.solve(**solve_kwargs)

        if previous_probe is not None and model.element_slip.size != previous.size:
            change = float(np.abs(
                _probe_slip(mesh, model, probe) - previous_probe).max())
        else:
            change = float(np.abs(model.element_slip - previous).max())
        history.append(_entry(round_index, obs, model, mesh,
                              sampling_field if sampling_field != "model" else "model",
                              change))
        if verbose:
            _report(history[-1])
        if change <= tol:
            break

    return obs, model, history


def _entry(round_index, obs, model, mesh, field, change):
    """One row of the loop's history, including what the mesh was that round."""
    return {"round": round_index, "n": obs.n, "field": field,
            "variance_reduction": model.variance_reduction,
            "max_change": change, "converged": model.converged,
            "n_elements": int(mesh.n_elements), "n_nodes": int(mesh.n_nodes)}


def _adapt_spec(mesh_adapt, max_rounds):
    """Validate the mesh-adaptation request, or ``None`` when there is none."""
    if mesh_adapt is None:
        return None
    spec = dict(mesh_adapt)
    if "geometry" not in spec:
        raise ValueError(
            "mesh_adapt needs a 'geometry' entry: a layout says where the nodes "
            "go, not what the fault is, so rebuilding the mesh needs the same "
            "keywords FaultMesh.curved took (bottom_trace=, uniform_dip=, "
            "segments=/dips=, edge_length=, ...)"
        )
    # Adapt for all but the last round by default, so the loop ends on a fixed
    # mesh where `max |element_slip - previous|` is exact. Re-meshing perturbs a
    # piecewise-constant slip field by ~0.1-1 m at element boundaries whatever
    # the model does -- measured, two models fitting the *same* data on 240 and
    # 232 elements agree to a 2 mm median and still differ by 0.97 m at the
    # worst probe point -- so a centimetre tolerance is unreachable while the
    # mesh is still moving, and a loop that adapts every round can only ever
    # stop on `max_rounds`.
    spec.setdefault("rounds", max(1, int(max_rounds) - 1))
    spec.setdefault("smoothing", None)
    # Checked here rather than at the first re-mesh: that happens after round
    # zero's sampling and solve, which on a real run is minutes in, and a typo
    # in an environment variable should not cost that.
    if spec.get("size_from", "diagonal") not in ("diagonal", "kernel"):
        raise ValueError(
            f"mesh_adapt['size_from'] must be 'diagonal' or 'kernel', got "
            f"{spec['size_from']!r}"
        )
    if spec.get("structure", "rows") not in ("rows", "unstructured"):
        raise ValueError(
            f"mesh_adapt['structure'] must be 'rows' or 'unstructured', got "
            f"{spec['structure']!r}"
        )
    return spec


def _resolution_of(inversion, adapt, solve_kwargs):
    """Model resolution at the weight the run actually solves at."""
    from .resolution import resolution_diagonal

    smoothing = adapt.get("smoothing") if adapt else None
    if smoothing is None:
        smoothing = solve_kwargs.get("smoothing", 0.3)
    keys = ("ss_ratio", "ds_ratio", "boundary_ratio", "sides", "smooth_weighting")
    passed = {k: solve_kwargs[k] for k in keys if k in solve_kwargs}
    return resolution_diagonal(inversion, smoothing, **passed)


def _split_field(model, inversion, stack, frame, spacing, sampling_field, adapt,
                 solve_kwargs):
    """What the quadtree splits on this round."""
    from .resolution import resolution_gain

    if sampling_field == "model":
        return predicted_los(model, stack, frame, spacing=spacing)

    resolution = _resolution_of(inversion, adapt, solve_kwargs)
    # No `spacing=`: the gain has its own, much coarser default. It is a smooth
    # field used only to rank locations, and it costs a full design matrix at
    # every probe point where the predicted field costs one forward model.
    gain = resolution_gain(model, stack, resolution, frame=frame)
    if sampling_field == "resolution":
        return gain
    # Standardise both before combining: one is metres of LOS and the other a sum
    # of squared Green's functions, so their raw magnitudes are unrelated.
    #
    # Added, not maximised. A pointwise max reads like "split where either field
    # would", and it is not: the quadtree splits on a cell's *scatter*, and the
    # max of two fields can be flatter than both where one of them clips the
    # other's variation. Measured on the two-scene fixture, `fmax` gave 1432
    # samples against 1974 for the predicted field alone. A sum's within-cell
    # variance is var(a) + var(b) + 2cov(a, b), so it loses only to a strong
    # anticorrelation.
    predicted = predicted_los(model, stack, frame, spacing=spacing)
    finite = np.isfinite(predicted)
    scale = predicted[finite].std() if finite.any() else 0.0
    if scale == 0:
        return gain
    return (predicted - predicted[finite].mean()) / scale + gain


def _adapt_mesh(mesh, inversion, trace, frame, adapt, solve_kwargs):
    """Re-size the elements from model resolution and rebuild on the new layout."""
    from .mesh import FaultMesh
    from .resolution import (adapt_layout, element_resolution,
                             kernel_element_size, target_element_size)

    if adapt.get("size_from", "diagonal") == "kernel":
        # The averaging-kernel width: a lambda-stable length that never
        # degenerates, unlike (target / r) on the diagonal. See
        # `kernel_element_size` for the measurement behind that choice.
        smoothing = adapt.get("smoothing")
        if smoothing is None:
            smoothing = solve_kwargs.get("smoothing", 1.0)
        sizes = kernel_element_size(
            inversion, smoothing,
            **{k: v for k, v in adapt.items()
               if k in ("fraction", "reduce", "size_min", "size_max")}
        )
    else:
        resolution = _resolution_of(inversion, adapt, solve_kwargs)
        sizes = target_element_size(
            mesh, element_resolution(inversion, resolution),
            **{k: v for k, v in adapt.items()
               if k in ("target", "gain", "max_ratio", "size_min", "size_max")}
        )
    layout = adapt_layout(
        mesh, sizes, budget=adapt.get("budget", "conserve"),
        balance=adapt.get("balance", MAX_COUNT_RATIO),
        **{k: v for k, v in adapt.items()
           if k in ("structure", "iterations", "seed")},
    )
    return FaultMesh.from_layout(layout, trace, frame, **adapt["geometry"])


def _probe_lattice(mesh, n_along=200, n_down=40):
    """A fixed ``(s, z)`` grid to compare two slip fields on.

    The loop's convergence test used to difference ``element_slip`` directly,
    which only works while the mesh is the same one. Sampling both fields on a
    lattice that belongs to neither makes the number mean the same thing across
    a re-mesh -- and it stays in metres of slip, so ``tol`` does not change
    meaning either.
    """
    params = mesh.params
    s = np.linspace(params[:, 0].min(), params[:, 0].max(), int(n_along))
    z = np.linspace(params[:, 1].min(), params[:, 1].max(), int(n_down))
    ss, zz = np.meshgrid(s, z)
    return np.column_stack([ss.ravel(), zz.ravel()])


def _probe_slip(mesh, model, probe):
    """``model``'s slip at fixed ``(s, z)`` points, ``(n_probe, 2)``.

    Both components, because ``element_slip`` carries both -- which keeps this
    the same quantity the fixed-mesh path differences, so ``tol`` means "no
    component moved more than this" in either mode rather than two different
    things.

    Nearest centroid rather than a barycentric lookup: the slip is piecewise
    constant per element anyway, so this *is* the field, not an approximation of
    it.
    """
    from scipy.spatial import cKDTree

    index = cKDTree(mesh.element_params).query(probe)[1]
    return model.element_slip[index]


def model_rms_min(field, fraction=0.02):
    """A split threshold for a *predicted* field, as a fraction of its own scatter.

    ``rms_min`` means something different once
    :meth:`~nisar_tools.slip.sampling.Observations.from_los` is splitting on a
    model: it is no longer a noise level -- there is no noise -- but a statement
    about how finely the predicted shape should be resolved. So
    :func:`~nisar_tools.slip.diagnostics.scene_report`'s value, which is measured
    from the data's noise floor, is the wrong number and must not be reused.
    """
    values = np.asarray(field, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("The predicted field is empty; nothing to set a threshold from")
    return float(fraction) * float(finite.std())


#: Below this peak slip the model that drives the sampling is not a model.
_DEGENERATE_SLIP = 1e-3


def _warn_if_degenerate(model, round_index):
    """Warn when the model about to steer the sampling is essentially zero.

    A flat model predicts a flat field, ``model_rms_min`` then returns roughly
    zero, and the quadtree splits on floating-point noise -- so the loop *appears*
    to converge (nothing changes between rounds, because nothing is there) on a
    model that explains none of the data. It is the smoothing weight almost every
    time: measured on the real D134 scene, anything above about 30 flattened the
    answer to a peak of 0.004 m while variance reduction sat at a
    respectable-looking 22%.
    """
    import warnings

    peak = float(np.abs(model.element_slip).max())
    if peak < _DEGENERATE_SLIP:
        warnings.warn(
            f"round {round_index} is being sampled from a model with a peak slip of "
            f"{peak:.2e} m and VR {model.variance_reduction:.1f}% -- effectively zero, "
            "so the predicted field carries no signal to sample. Lower the smoothing "
            "weight; the loop will otherwise 'converge' on nothing.",
            RuntimeWarning, stacklevel=3,
        )


def _fill_nearest(values):
    """Fill NaN holes from the nearest finite sample.

    The forward model returns NaN in a narrow band along the outcrop, where a
    dislocation solution is singular. Left alone, that band would be interpolated
    into a much wider hole on the fine raster, invalidating exactly the near-fault
    pixels ``refine_within`` exists to keep.
    """
    from scipy.ndimage import distance_transform_edt

    missing = ~np.isfinite(values)
    if not missing.any():
        return values
    if missing.all():
        raise ValueError("The forward model returned no finite values")
    index = distance_transform_edt(missing, return_distances=False, return_indices=True)
    return values[tuple(index)]


def _report(entry):
    change = entry["max_change"]
    change = "     -- " if not np.isfinite(change) else f"{change:7.4f} m"
    flag = "" if entry["converged"] else "   (solver did not converge)"
    # The mesh is in the line because it can now move between rounds, and a
    # sample count read beside a silently changed element count means nothing.
    mesh = (f"  mesh={entry['n_elements']}el/{entry['n_nodes']}n"
            if "n_elements" in entry else "")
    print(f"  round {entry['round']}  {entry['field']:>10}  "
          f"n={entry['n']:6d}{mesh}  VR={entry['variance_reduction']:6.2f}%  "
          f"max change {change}{flag}", flush=True)
