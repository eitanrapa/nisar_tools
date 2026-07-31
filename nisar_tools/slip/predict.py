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
                     rms_fraction=0.02, normalize="sqrt_count", verbose=True):
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

    Stops when no slip parameter moves by more than ``tol`` between rounds, or at
    ``max_rounds``. Returns ``(observations, model, history)``; ``history`` is a
    list of per-round dicts carrying ``n``, ``variance_reduction`` and
    ``max_change``, so that the "stops changing" criterion can be checked rather
    than assumed.
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
    model = SlipInversion(mesh, obs, **inversion_kwargs).solve(**solve_kwargs)
    history = [{"round": 0, "n": obs.n, "field": "data",
                "variance_reduction": model.variance_reduction,
                "max_change": float("nan"), "converged": model.converged}]
    if verbose:
        _report(history[-1])

    for round_index in range(1, int(max_rounds) + 1):
        _warn_if_degenerate(model, round_index)
        previous = model.element_slip.copy()
        parts = []
        for name, stack in scenes.items():
            field = predicted_los(model, stack, frame, spacing=spacing)
            parts.append(sample(name, stack, field=field,
                                rms_min=model_rms_min(field, rms_fraction)))

        obs = Observations.concat(parts, normalize=normalize)
        model = SlipInversion(mesh, obs, **inversion_kwargs).solve(**solve_kwargs)

        change = float(np.abs(model.element_slip - previous).max())
        history.append({"round": round_index, "n": obs.n, "field": "model",
                        "variance_reduction": model.variance_reduction,
                        "max_change": change, "converged": model.converged})
        if verbose:
            _report(history[-1])
        if change <= tol:
            break

    return obs, model, history


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
    print(f"  round {entry['round']}  {entry['field']:>5}  "
          f"n={entry['n']:6d}  VR={entry['variance_reduction']:6.2f}%  "
          f"max change {change}{flag}", flush=True)
