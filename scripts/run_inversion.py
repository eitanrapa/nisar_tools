"""Stage 3: solve once at the chosen weight, and write the result.

    cd ~/Documents/GitHub/nisar_tools
    KMP_DUPLICATE_LIB_OK=TRUE \
    nohup ~/miniforge3/envs/remote_sensing/bin/python -u scripts/run_inversion.py \
          > workdir/inversion.log 2>&1 &
    tail -f workdir/inversion.log

Reads only ``<ws>/slip_observations.zarr`` and solves at
``slip_config.SMOOTHING`` -- the weight read off stage 2's L-curve. Writes:

    model_sampling/slip_model.slip.zip   model + mesh + observations, self-contained
    model_sampling/slip_model.txt        GMT-ready element table
    model_sampling/summary.json          every scalar, without unzipping anything
    model_sampling/figures/              slip, and data/model/residual per track

The moment is computed with the depth-dependent rigidity from
``slip_config.VELOCITY_MODEL``, not the 30 GPa default -- for this crust that
default is ~25% low through the seismogenic zone, so Mw would be understated.

⚠️ The Green's matrix is **not** saved with the model: it is the largest object
in the problem and nothing downstream needs it. A reloaded model can be read,
plotted and forward-modelled, but not re-solved.
"""

import json

import numpy as np

from nisar_tools.slip import SlipInversion
from nisar_tools.slip.plot import plot_fit, plot_slip
from slip_config import (
    BOUNDS, OUT_DIR, SMOOTHING, VELOCITY_MODEL, banner, geometry,
    inversion_kwargs, load_observations, mesh_summary, save_figure,
    workspace,
)


def main():
    banner("stage 3: inversion")
    ws = workspace(create=False)
    trace, frame, mesh = geometry()
    obs = load_observations(ws, frame)
    print(f"    {obs!r}\n    {mesh!r}\n    {mesh_summary(mesh)}\n", flush=True)

    inversion = SlipInversion(mesh, obs, **inversion_kwargs())
    print(f"{inversion!r}  -- Green's matrix built", flush=True)

    model = inversion.solve(smoothing=SMOOTHING, **BOUNDS)
    print(f"{model!r}", flush=True)
    print(f"converged: {model.converged} | iterations: {model.result.nit}", flush=True)
    print(f"ramp terms: {dict(zip(model.ramp_labels, model.ramp.round(4)))}", flush=True)

    if not model.converged:
        # A capped solve has a meaningless variance reduction. Keep it, so the run
        # is not wasted, but under a name nothing will mistake for an answer.
        model.save(OUT_DIR / "slip_model_UNCONVERGED.slip.zip")
        raise SystemExit(
            f"solver hit the iteration cap after {model.result.nit} iterations; "
            f"raise max_iter, or raise SMOOTHING above {SMOOTHING:g}"
        )

    model.save(OUT_DIR / "slip_model.slip.zip")
    model.to_text(OUT_DIR / "slip_model.txt", shear_modulus=VELOCITY_MODEL)

    strike = model.strike_slip
    summary = {
        "n_observations": int(obs.n),
        "n_elements": int(mesh.n_elements),
        "n_parameters": int(inversion.n_param),
        "tracks": obs.tracks,
        "smoothing": float(SMOOTHING),
        "mesh": mesh_summary(mesh),
        "variance_reduction": float(model.variance_reduction),
        "rms_misfit_m": float(model.rms_misfit),
        "roughness": float(model.roughness),
        "max_slip_m": float(model.max_slip),
        "peak_strike_slip_m": float(strike.min()),
        "fraction_right_lateral": float((strike < 0).mean()),
        "moment_Nm": float(model.moment(VELOCITY_MODEL)),
        "moment_magnitude": float(model.moment_magnitude),
        "ramp": dict(zip(model.ramp_labels, model.ramp.round(6).tolist())),
        "velocity_model": VELOCITY_MODEL.to_dict(),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nMw {summary['moment_magnitude']:.2f}  "
          f"(M0 {summary['moment_Nm']:.3e} N m, depth-dependent rigidity)", flush=True)
    print(f"peak strike-slip {strike.min():+.2f} m; "
          f"{100 * summary['fraction_right_lateral']:.0f}% of elements right-lateral",
          flush=True)
    # A right-lateral fault solved with polarity=(-1,0,0) cannot come back
    # left-lateral, but it can come back ~zero, which looks like a fit and is not.
    if np.abs(strike).max() < 1e-3:
        print("⚠️  the model is essentially zero slip -- check the smoothing weight "
              "and that the LOS sign convention is right", flush=True)

    save_figure(plot_slip(model)[0], "slip")
    save_figure(plot_slip(model, component="strike")[0], "slip_strike")
    for track in obs.tracks:
        save_figure(plot_fit(model, track=track, trace=trace)[0], f"fit_{track}")

    print(f"\ndone -- everything in {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
