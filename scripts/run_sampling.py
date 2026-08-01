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

import pandas as pd

from nisar_tools.slip import ARCSEC_10, iterate_sampling, resample_all, scene_report
from nisar_tools.slip.plot import plot_coverage, plot_mesh, plot_samples
from slip_config import (
    BOUNDS, INVERSION, LOOP, LOS_STAGE, OBS_STAGE, OUT_DIR, SCENES, SMOOTHING,
    banner, geometry, load_scene, mesh_summary, save_figure, workspace,
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

    # Measured per scene, not inherited: `rms_min` is a noise level, and set below
    # it the quadtree cannot stop splitting and just runs down to `width_min`.
    sampling = {}
    for name, stack in gridded.items():
        report = scene_report(stack, trace, frame, mesh=mesh)
        sampling[name] = dict(
            rms_min=report.attrs["rms_min"], width_min=report.attrs["width_min"],
            width_max=30_000.0, exclude_within=report.attrs["exclude_within"],
        )
        print(f"{name}: noise {1e3 * report.attrs['noise_floor']:.1f} mm, "
              f"two-sided {100 * report.attrs['two_sided_fraction']:.0f}%, "
              f"geometry_ok={report.attrs['geometry_consistent']}", flush=True)
        print(f"      -> {sampling[name]}", flush=True)
        save_figure(plot_coverage(report, name=name)[0], f"coverage_{name}")

    print("\nround 0 is data-driven and coarse; every later round is model-driven",
          flush=True)
    obs, model, history = iterate_sampling(
        gridded, mesh, trace, frame, sampling,
        inversion_kwargs=INVERSION,
        solve_kwargs={"smoothing": SMOOTHING, **BOUNDS},
        **LOOP,
    )

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

    print(f"\ndone -- next: run_lcurve.py, then run_inversion.py", flush=True)


if __name__ == "__main__":
    main()
