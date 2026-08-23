"""Is the nuisance ramp eating the earthquake? Run after stage 1.

    cd ~/Documents/GitHub/nisar_tools
    KMP_DUPLICATE_LIB_OK=TRUE \
    nohup ~/miniforge3/envs/remote_sensing/bin/python -u scripts/run_ramp_check.py \
          > workdir/ramp_check.log 2>&1 &
    tail -f workdir/ramp_check.log

Reads only ``<ws>/slip_observations.zarr`` and the configured geometry. Writes:

    model_sampling/ramp_leakage.json   what each degree COULD absorb
    model_sampling/ramp_sweep.csv      what each degree DID
    model_sampling/figures/ramp_check.png

**Variance reduction cannot answer this question, which is the whole reason the
script exists.** The ramp columns are free and unregularised, so adding them can
only improve the fit -- a ramp quietly absorbing coseismic signal therefore shows
up as a *better fit with a smaller earthquake*, and nothing in a single solve's
output says so. The sweep flags that conjunction: moment down by more than
``slip_config.RAMP_TOLERANCE`` while variance reduction is up.

Two measures, and they answer different halves:

* **leakage** -- the fraction of the fault's own predicted field that lies in the
  ramp's column space. Geometric, no solve, so it describes the *aperture*. Read
  ``curvature_only``, not the totals: a per-track offset always absorbs that
  track's mean displacement, because an unwrapped interferogram carries an
  arbitrary constant.
* **the sweep** -- one solve per degree, sharing the Green's matrix, reporting
  what the slip model actually lost.

High leakage with a stable moment is fine and common: it means the ramp *could*
take the earthquake and the data still pinned it. It is the pair that decides.

⚠️ **The answer does not transfer between scene sets.** Measured on the full
six-scene Venezuela study a cubic ramp left the slip alone; on two descending
tracks from the same event it took 32% of the moment, because one-sided coverage
shows only the flank of the arctangent step and an odd cubic imitates a flank
well. Re-run this whenever the scene list changes.

⚠️ ``max_iter`` defaults higher here than ``BOUNDS`` uses elsewhere
(``NISAR_RAMP_MAX_ITER``, default 200): an iteration-capped solve makes every
column of the sweep meaningless, and one bad row invalidates the comparison
rather than just itself.
"""

import json
import os

import pandas as pd

from nisar_tools.slip import SlipInversion
from nisar_tools.slip.diagnostics import ramp_leakage, ramp_sweep
from slip_config import (
    BOUNDS, OUT_DIR, RAMP_KINDS, RAMP_TOLERANCE, SMOOTHING, banner, geometry,
    inversion_kwargs, load_observations, mesh_summary, sampling_kind,
    save_figure, workspace,
)


def _figure(leak, sweep, kinds):
    import matplotlib.pyplot as plt

    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    left.plot(kinds, [leak[k]["leakage"] for k in kinds], "o-", color="0.2")
    for track in sorted(leak[kinds[0]]["per_track"]):
        left.plot(kinds, [leak[k]["per_track"][track] for k in kinds], "o--",
                  lw=1, alpha=0.7, label=track)
    left.set_ylim(0, 1.02)
    left.set_ylabel("leakage (fraction of the earthquake the ramp can absorb)")
    left.set_title("What the ramp COULD absorb")
    left.legend(fontsize=8)
    left.grid(alpha=0.3)

    ratio = sweep["moment_ratio"].values
    right.plot(kinds, ratio, "o-", color="C3", label="moment / reference")
    right.axhline(1.0, color="0.5", lw=0.8)
    right.axhline(1.0 - RAMP_TOLERANCE, color="C3", ls=":", lw=1,
                  label=f"flag below {1 - RAMP_TOLERANCE:.2f}")
    twin = right.twinx()
    twin.plot(kinds, sweep["variance_reduction"].values, "s--", color="C0",
              label="variance reduction")
    twin.set_ylabel("variance reduction (%)", color="C0")
    right.set_ylabel("moment ratio", color="C3")
    right.set_title("What it DID")
    right.legend(fontsize=8, loc="lower left")
    right.grid(alpha=0.3)
    figure.tight_layout()
    return figure


def main():
    banner("ramp check: is the ramp eating the earthquake?")
    ws = workspace(create=False)
    trace, frame, mesh = geometry()
    obs = load_observations(ws, frame)

    kind = sampling_kind(obs)
    suffix = "" if kind == "model" else "_bootstrap"
    print(json.dumps(mesh_summary(mesh), indent=2, sort_keys=True), flush=True)
    print(f"\n{obs.n} observations ({kind}-driven sampling), "
          f"kinds {list(RAMP_KINDS)}, smoothing {SMOOTHING:g}", flush=True)

    inversion = SlipInversion(mesh, obs, **inversion_kwargs())
    print(f"G {inversion.g.shape} assembled", flush=True)

    leak = ramp_leakage(inversion, kinds=RAMP_KINDS)
    print("\nleakage -- what each degree COULD absorb "
          "(uniform unit strike-slip, so this is the aperture, not the answer):",
          flush=True)
    for name in RAMP_KINDS:
        per_track = {t: round(v, 3) for t, v in leak[name]["per_track"].items()}
        print(f"  {name:<10} {leak[name]['leakage']:.3f}   per track {per_track}",
              flush=True)
    marginal = {k: leak[k] for k in ("gradient_only", "curvature_only", "cubic_only")
                if k in leak}
    print("  marginal: " + "  ".join(f"{k} {v:+.3f}" for k, v in marginal.items()),
          flush=True)
    print("  ^ read the MARGINAL terms: a per-track offset always absorbs that "
          "track's mean displacement.", flush=True)
    (OUT_DIR / f"ramp_leakage{suffix}.json").write_text(json.dumps(leak, indent=2))

    solve = {**BOUNDS, "smoothing": SMOOTHING,
             "max_iter": int(os.environ.get("NISAR_RAMP_MAX_ITER", 200))}
    print(f"\nsweep -- one solve per degree at max_iter={solve['max_iter']}, "
          "sharing G ...", flush=True)
    sweep = ramp_sweep(inversion, kinds=RAMP_KINDS, tolerance=RAMP_TOLERANCE, **solve)

    table = sweep.to_dataframe()
    with pd.option_context("display.width", 200):
        print(table[["n_ramp", "variance_reduction", "rms_misfit", "max_slip",
                     "moment_magnitude", "moment_ratio", "vr_change",
                     "converged"]].to_string(), flush=True)
    table.to_csv(OUT_DIR / f"ramp_sweep{suffix}.csv")

    capped = [k for k in RAMP_KINDS if not bool(sweep["converged"].sel(ramp=k))]
    if capped:
        print(f"\n⚠️  {capped} hit the iteration cap, so their columns mean nothing "
              f"-- raise NISAR_RAMP_MAX_ITER above {solve['max_iter']} and re-run "
              "before reading the verdict.", flush=True)

    suspect = list(sweep.attrs["suspect"])
    if suspect:
        print(f"\n⚠️  VERDICT: {suspect} fit better than "
              f"{sweep.attrs['reference']!r} while recovering less moment -- that "
              "is a ramp absorbing coseismic signal. Prefer the highest degree "
              "NOT listed here unless something independent supports the ramp.",
              flush=True)
    else:
        print(f"\nVERDICT: no degree in {list(RAMP_KINDS)} lost more than "
              f"{100 * RAMP_TOLERANCE:.0f}% of the moment while fitting better. "
              "The ramp columns are absorbing nuisance, not earthquake.", flush=True)

    save_figure(_figure(leak, sweep, list(RAMP_KINDS)), f"ramp_check{suffix}")
    print("\ndone.", flush=True)


if __name__ == "__main__":
    main()
