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
    load_observations, save_figure, workspace,
)


def main():
    banner("stage 2: L-curve")
    ws = workspace(create=False)
    trace, frame, mesh = geometry()
    obs = load_observations(ws, frame)
    print(f"    {obs!r}\n    {mesh!r}", flush=True)
    print(f"    {obs.n / (2 * mesh.n_elements):.1f}x the slip parameters\n", flush=True)

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
    table.to_csv(OUT_DIR / "l_curve.csv")
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

    save_figure(plot_l_curve(curve)[0], "l_curve")

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

    print(f"\nread the corner off {OUT_DIR / 'figures' / 'l_curve.png'}, put it in "
          "slip_config.SMOOTHING, then run run_inversion.py", flush=True)


if __name__ == "__main__":
    main()
