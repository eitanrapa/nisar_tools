"""Stage 2: sweep the smoothing weight over the observations stage 1 wrote.

    cd ~/Documents/GitHub/nisar_tools
    KMP_DUPLICATE_LIB_OK=TRUE \
    nohup ~/miniforge3/envs/remote_sensing/bin/python -u scripts/run_lcurve.py \
          > workdir/lcurve.log 2>&1 &
    tail -f workdir/lcurve.log

Reads only ``<ws>/slip_observations.zarr``; weights come from
``slip_config.LCURVE_WEIGHTS``. Writes:

    model_sampling/l_curve.csv        one row per weight
    model_sampling/figures/l_curve.png

Then put the corner into ``slip_config.SMOOTHING`` and run ``run_inversion.py``.

**Both files gain a ``_bootstrap`` suffix when the observations came from the
coarse round-0 sampling** (``NISAR_MAX_ROUNDS=0``), which is the L-curve-first
route. The label is read off the observations, not passed in, so the two curves
cannot overwrite each other or be confused later -- and they do mean different
things. Round 0 is deliberately under-sampled; where there are fewer observations
than parameters the smoothing supplies missing rank rather than trading misfit
against roughness, so its corner sits at more smoothing than the refined sampling
wants. Treat it as a starting point for stage 1, not as the answer.

The corner is deliberately left to the eye: automatic detection on a short
discrete sweep is unreliable, and the two neighbouring weights usually differ
less than the choice of mesh does. What this script *will* do is refuse to
recommend a weight whose solve hit the iteration cap.

⚠️ Runtime is dominated by the smooth end, not shared evenly. Measured on an
8-weight sweep: 0.78/0.75/0.54/0.84/0.61/0.91/**12.15**/1.59 s -- one weight was
67% of the sweep, and it was the one that did not converge. If this is slow, drop
the roughest weights rather than trying to parallelise it (``l_curve`` is serial
on purpose; threads measured 1.02x at 2 workers and 0.90x at 8).
"""

import numpy as np

from nisar_tools.slip import SlipInversion
from nisar_tools.slip.plot import plot_l_curve
from slip_config import (
    BOUNDS, INVERSION, LCURVE_WEIGHTS, OUT_DIR, banner, geometry,
    load_observations, mesh_summary, sampling_kind, save_figure, workspace,
)


def main():
    banner("stage 2: L-curve")
    ws = workspace(create=False)
    trace, frame, mesh = geometry()
    obs = load_observations(ws, frame)
    ratio = obs.n / (2 * mesh.n_elements)

    # The observations say which sampling they came from, so the outputs are named
    # for what they actually describe. A curve from the coarse round-0 sampling and
    # one from the refined sampling answer different questions and would otherwise
    # overwrite each other under the same filename.
    kind = sampling_kind(obs)
    suffix = "" if kind == "model" else "_bootstrap"
    print(f"    {obs!r}\n    {mesh!r}\n    {mesh_summary(mesh)}", flush=True)
    print(f"    sampling: {kind}-driven, {ratio:.1f}x the slip parameters\n", flush=True)

    if kind == "bootstrap":
        print("⚠️  this is the COARSE data-driven sampling, so the corner is only "
              "provisional: round 0 is deliberately under-sampled, and where there "
              "are fewer observations than parameters the smoothing is supplying "
              "missing rank rather than trading misfit against roughness. Expect the "
              "corner to sit at more smoothing than the refined sampling wants.",
              flush=True)
    if ratio < 1.0:
        print(f"⚠️  {obs.n} observations against {2 * mesh.n_elements} slip parameters "
              "-- the problem is under-determined at every weight on this curve",
              flush=True)

    # One Green's matrix for the whole sweep -- that is the point of doing this
    # here rather than re-solving from scratch per weight.
    inversion = SlipInversion(mesh, obs, **INVERSION)
    print(f"{inversion!r}\nsweeping {LCURVE_WEIGHTS}", flush=True)

    curve, models = inversion.l_curve(LCURVE_WEIGHTS, **BOUNDS)
    table = curve[["rms_misfit", "roughness", "variance_reduction",
                   "iterations", "converged"]].to_dataframe()
    # Peak slip is what makes both end-effects visible in the table itself, and
    # neither is visible in misfit or roughness alone.
    peak = np.array([float(np.abs(m.element_slip).max()) for m in models])
    table["max_slip_m"] = peak
    # Constant columns, so the file still says which sampling it describes after
    # it has been emailed to someone or opened a year later.
    table["sampling"] = kind
    table["n_observations"] = obs.n
    table["n_parameters"] = 2 * mesh.n_elements
    table.to_csv(OUT_DIR / f"l_curve{suffix}.csv")
    print("\n" + table.to_string(), flush=True)

    # Too much smoothing: the model goes flat while VR stays plausible-looking.
    # Measured on the real D134 scene, lambda >= 100 gave 0.004 m of slip at 22% VR.
    flat = peak < 1e-3
    if flat.any():
        print(f"\n⚠️  flat (zero-slip) models at smoothing "
              f"{list(curve['smoothing'].values[flat])} -- the smoothing is winning "
              "outright there, whatever the variance reduction says", flush=True)

    # Too little: the bound, not the data, is setting the answer.
    limit = max(abs(v) for v in BOUNDS["strike"])
    saturated = peak >= 0.999 * limit
    if saturated.any():
        print(f"⚠️  strike bound ±{limit:g} m saturated at smoothing "
              f"{list(curve['smoothing'].values[saturated])} -- those points are the "
              "bound, not a fit; ignore them or widen `BOUNDS`", flush=True)

    figure, axis = plot_l_curve(curve)
    axis.set_title(f"{axis.get_title()}\n{kind}-driven sampling, "
                   f"{obs.n} obs / {2 * mesh.n_elements} parameters", fontsize=9)
    save_figure(figure, f"l_curve{suffix}")

    converged = curve["converged"].values.astype(bool)
    if not converged.all():
        capped = curve["smoothing"].values[~converged]
        print(f"\n⚠️  hit the iteration cap at {list(np.round(capped, 4))} -- their "
              "statistics are meaningless; ignore those points on the curve",
              flush=True)
    if not converged.any():
        raise SystemExit("no weight converged; raise max_iter or widen the sweep")

    # Not the corner -- just the bracket the corner is inside, so the eye knows
    # where to look and a bad sweep range is obvious.
    weights = curve["smoothing"].values[converged]
    misfit = curve["rms_misfit"].values[converged]
    rough = curve["roughness"].values[converged]
    print(f"\nconverged range: smoothing {weights.min():g}..{weights.max():g}, "
          f"rms misfit {misfit.min():.4f}..{misfit.max():.4f} m, "
          f"roughness {rough.min():.3f}..{rough.max():.3f}", flush=True)
    if misfit.max() / max(misfit.min(), 1e-12) < 1.5:
        print("⚠️  misfit barely moves across the sweep -- the range is too narrow "
              "to show a corner. Widen LCURVE_WEIGHTS.", flush=True)

    where = OUT_DIR / "figures" / f"l_curve{suffix}.png"
    if kind == "bootstrap":
        print(f"\nread the provisional corner off {where}, then re-run stage 1 with "
              "NISAR_MAX_ROUNDS back at its default and NISAR_SMOOTHING set to it -- "
              "the refined sampling gets its own curve, written beside this one",
              flush=True)
    else:
        print(f"\nread the corner off {where}, put it in slip_config.SMOOTHING "
              "(or NISAR_SMOOTHING), then run run_inversion.py", flush=True)


if __name__ == "__main__":
    main()
