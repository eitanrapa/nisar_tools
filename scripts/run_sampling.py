"""Stage 1: put every scene on one lattice, then sample from the model.

    cd ~/Documents/GitHub/nisar_tools
    KMP_DUPLICATE_LIB_OK=TRUE \
    nohup ~/miniforge3/envs/remote_sensing/bin/python -u scripts/run_sampling.py \
          > workdir/sampling.log 2>&1 &
    echo $! > workdir/sampling.pid
    tail -f workdir/sampling.log

``-u`` is not optional: without it the log stays empty until the process exits,
which is exactly when you stop needing it. Configuration lives in
``slip_config.py``, shared with the other two stages.

Writes, as each is produced, so an interrupted run keeps what it finished:

    <ws>/los_<track>_frame.zarr   each scene on the shared 10-arcsec lattice
    <ws>/slip_observations.zarr   the downsampled observations -- stages 2 and 3
                                  read this and nothing else from here
    model_sampling/history.csv    per-round sample count, VR, max slip change
    model_sampling/bootstrap.slip.zip   the loop's own final model, for a look
    model_sampling/figures/       coverage per scene, the mesh, the samples
"""

import json

import numpy as np
import pandas as pd

from nisar_tools.slip import ARCSEC_10, iterate_sampling, resample_all
from nisar_tools.slip.plot import plot_mesh, plot_samples
from nisar_tools.slip.resolution import element_resolution, resolution_diagonal
from slip_config import (
    BOUNDS, LOOP, LOS_STAGE, OBS_STAGE, OUT_DIR, SAMPLING_FIELD, SCENES, SMOOTHING,
    adapt_kwargs, banner, geometry, inversion_kwargs, load_scene, mesh_summary,
    sampling_parameters, save_figure, save_mesh, workspace,
)


def main():
    banner("stage 1: sampling")
    ws = workspace()
    trace, frame, mesh = geometry()
    print(f"    {trace!r}\n    {mesh!r}\n    {mesh_summary(mesh)}\n", flush=True)

    scenes = {name: load_scene(spec, ws) for name, spec in SCENES.items()}
    for name, stack in scenes.items():
        print(f"{name}: {dict(stack.ds.sizes)} {stack.crs.to_string()[:30]}", flush=True)

    # One lattice for every track. A quadtree cell is an integer number of pixels
    # halved at its midpoint, so scenes at different resolutions reach different
    # cell-size ladders -- one `width_min` then means two different things and the
    # per-track sample counts diverge for a reason unrelated to the data.
    gridded = resample_all(scenes, frame, spacing=ARCSEC_10)
    for name, stack in gridded.items():
        stack.persist(ws, LOS_STAGE.format(name=name), overwrite=True)
        print(f"{name}: -> {dict(stack.ds.sizes)} @ {ARCSEC_10:.1f} m in the frame",
              flush=True)

    # Measured on the *resampled* stacks, and per scene rather than inherited:
    # `rms_min` is a noise level, and set below it the quadtree cannot stop
    # splitting and just runs down to `width_min`. `slip_config.SAMPLING` pins
    # values on top of the measurement, and the log says which.
    sampling = sampling_parameters(gridded, trace, frame, mesh)

    print("\nround 0 is data-driven and coarse; every later round is model-driven",
          flush=True)
    adapt = adapt_kwargs(trace, frame)
    if adapt is not None:
        print(f"mesh adaptation ON: {adapt['rounds']} round(s), "
              f"budget={adapt.get('budget')}, gain={adapt.get('gain')} -- the mesh "
              "below is the starting one, not the one stage 2 and 3 will use",
              flush=True)
    if SAMPLING_FIELD != "model":
        print(f"quadtree splits on {SAMPLING_FIELD!r}, not the predicted field",
              flush=True)
    obs, model, history = iterate_sampling(
        gridded, mesh, trace, frame, sampling,
        inversion_kwargs=inversion_kwargs(),
        solve_kwargs={"smoothing": SMOOTHING, **BOUNDS},
        mesh_adapt=adapt, sampling_field=SAMPLING_FIELD,
        **LOOP,
    )
    # The loop may have replaced it, and everything downstream -- the figures
    # here, `mesh_summary`, stages 2 and 3 -- has to describe what actually ran.
    mesh = model.mesh

    pd.DataFrame(history).set_index("round").to_csv(OUT_DIR / "history.csv")
    obs.persist(ws, OBS_STAGE, overwrite=True)
    print(f"\n{obs!r} -> {ws.path(OBS_STAGE)}", flush=True)

    if LOOP["max_rounds"] == 0:
        # NISAR_MAX_ROUNDS=0: round 0 only, so what was just written is the coarse
        # data-driven sampling, not the refined one. Say so here as well as in
        # stage 2 -- this overwrote whatever was in the observations stage before.
        print("NOTE: NISAR_MAX_ROUNDS=0, so these are the COARSE round-0 "
              "observations. Stage 2 will label its outputs `_bootstrap`; re-run "
              "this stage without the override to produce the refined sampling.",
              flush=True)

    # The loop's own model, kept for comparison. Stage 3 re-solves at whatever
    # weight the L-curve picks, so this is not the answer -- but if the two differ
    # much, the weight moved a long way and that is worth knowing.
    model.save(OUT_DIR / "bootstrap.slip.zip")
    print(f"{model!r}  (bootstrap, at smoothing={SMOOTHING:g})", flush=True)

    save_figure(plot_mesh(mesh, trace=trace, color="area")[0], "mesh")
    save_figure(plot_samples(obs, trace=trace)[0], "samples")

    # Model resolution is the reason an adaptive mesh looks the way it does, and
    # it is worth seeing even on a fixed one: it says which parts of the answer
    # the data paid for and which the smoothing supplied.
    resolution = resolution_diagonal(model.inversion, SMOOTHING,
                                     **{k: v for k, v in BOUNDS.items()
                                        if k in ("ss_ratio", "ds_ratio", "sides")})
    per_element = element_resolution(model.inversion, resolution)
    figure, axes = plot_mesh(mesh, trace=trace, color=per_element,
                             label="model resolution")
    save_figure(figure, "resolution")
    print(f"model resolution: median {np.median(per_element):.3f}, "
          f"{100.0 * float((per_element < 0.01).mean()):.0f}% of elements below "
          f"0.01 (effectively unconstrained), effective rank "
          f"{resolution.attrs['effective_rank']:.0f} of {2 * model.basis.n_basis}",
          flush=True)

    if adapt is not None:
        mesh_json = json.dumps(mesh_summary(mesh), indent=2, sort_keys=True)
        (OUT_DIR / "mesh_adapted.json").write_text(mesh_json + "\n")
        # Stages 2 and 3 rebuild the geometry from the config, which cannot
        # reproduce a mesh the data shaped -- so hand it over explicitly, or they
        # would invert these observations on a different discretisation.
        print(f"adapted mesh -> {save_mesh(mesh)}\n{mesh_json}", flush=True)

    print(f"\ndone -- next: run_lcurve.py, then run_inversion.py", flush=True)


if __name__ == "__main__":
    main()
