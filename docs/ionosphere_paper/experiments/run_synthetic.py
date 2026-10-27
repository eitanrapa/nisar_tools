"""The estimator comparison on synthetic ruptures (`synth.py`), with the truth known.

usage: .../envs/remote_sensing/bin/python -u run_synthetic.py [workers=6] [only=finite,through,...]

Experiments (each scene runs the estimators of `estimators.py` on the same cells; 3 noise seeds each):

- ``finite``: a 50 km rupture inside the 96 km scene (its sides connect around its ends); peak slip 0, 1, 2, 4, 6 m;
  D141-like damage zone (north 0.12 over ~2 km, south 0.25 over ~0.8 km), far-field coherence 0.6; cells within 2 km of
  the trace excluded.
- ``through``: the same with a 160 km rupture running through the scene (D141's and Venezuela's case: the sides meet
  only across the fault zone).
- ``noexcl``: ``through`` with no fault-zone exclusion (every cell with coherence > 0.2 used, as ISCE3 runs).
- ``lowcoh``: ``through``, 4 m, far-field coherence 0.35.
- ``coast``: ``through``, 4 m, water north of the trace's eastern half beyond 2 km.
- ``patchy``: 4 m, finite and through-going, with 25% of the ground within 10 km of the trace in decorrelated blobs
  (coherence 0.05, ~0.5 km).
- ``offset``: ``finite``, 4 m, band B sampled 0, 3, 9 and 20 m down range (the GUNW decimates A to B's columns, ~9 m off).
- ``footprint``: 4 m, finite and through-going, the side band's cells widened in range by [0.365, 1, 0.365], the PR
  GUNW's footprint mismatch (`run_nisar.FOOTPRINT_W`).
- ``cycle``: ``through``, 4 m, seed 0, with the main band's south side one cycle off (a cut-off block's tie wrong by
  one), against the same scene without: how far each estimate moves, raw and filtered.
- ``example``: ``finite`` and ``through`` at 4 m, seed 0: the truth and every screen, for the figures.

Writes `../results/synthetic.csv`, `../results/synthetic_cycle.csv` and `../results/synthetic_example_<geometry>.npz`.
"""

import functools
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
print = functools.partial(print, flush=True)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import estimators as E  # noqa: E402
import metrics as M  # noqa: E402
import synth  # noqa: E402

RESULTS = os.path.join(HERE, "..", "results")
WEAK = dict(gradient=0.1, rf_rms=0.7)
EXCLUDE_M = 2000.0


def inputs(ca, cb, truth, exclude_m=EXCLUDE_M):
    psi_a, psi_b = np.angle(ca.ifg.values), np.angle(cb.ifg.values)
    ga, gb = ca.coherence.values, cb.coherence.values
    na, nb = ca.n.values / ca.attrs["pixels_per_cell"], cb.n.values / cb.attrs["pixels_per_cell"]
    base = truth["land"] & (ga > 0.2) & (gb > 0.2) & (na >= 0.5) & (nb >= 0.5)
    use = base & (np.abs(truth["dist"]) > exclude_m)
    sig = E.raw_sigma(ga, gb, na, nb, synth.F_A, synth.F_B, synth.CELL)
    return psi_a, psi_b, ga, gb, base, use, sig


def run_scene(job):
    name, kw, methods, exclude_m, keep = job
    t0 = time.time()
    kw = dict(kw)
    fw = kw.pop("footprint_w", None)
    ca, cb, tr = synth.scene(**kw)
    if fw:                                          # the side band's range footprint widened as the PR GUNW's
        from run_nisar import _range_kernel
        cb = _range_kernel(cb, fw)
        kw["footprint_w"] = fw
    psi_a, psi_b, ga, gb, base, use, sig = inputs(ca, cb, tr, exclude_m)
    rows, screens = [], {}
    for m in methods:
        out = E.estimate(m, tr["phase_a"], psi_a, psi_b, ga, gb, use, sig, synth.F_A, synth.F_B, synth.CELL,
                         x=tr["x"], y=tr["y"], data=tr["land"])
        row, err = M.synthetic(out["screen"], tr, synth.CELL)
        rows.append(dict(experiment=name, method=m, exclude_m=exclude_m, cycles_a=out["cycles_a"],
                         cycles_b=out["cycles_b"], seconds=0.0,
                         **{k: (v if not isinstance(v, dict) else str(v)) for k, v in kw.items()}, **row))
        if keep:
            screens[m] = out["screen"].astype(np.float32)
    north = (tr["dist"] > 0) & (tr["dist"] < 2e3)
    print(f"{name} {kw} excl {exclude_m:.0f} m [{time.time() - t0:.0f}s]: coherence A/B 0-2 km N "
          f"{np.nanmedian(ga[north]):.2f}/{np.nanmedian(gb[north]):.2f}; "
          + "; ".join(f"{r['method']} {r['rms_2_5']:.2f}/{r['leak']:+.3f}" for r in rows))
    return rows, ((name, screens, tr, dict(coh_a=ga, coh_b=gb, use=use)) if keep else None)


THROUGH = synth.LENGTH_THROUGH
SLIPS = (0.0, 1.0, 2.0, 4.0, 6.0)


def jobs(only):
    J = []
    seeds = range(3)
    if "finite" in only:
        J += [("finite", dict(slip=s, seed=k), E.METHODS, EXCLUDE_M, False) for s in SLIPS for k in seeds]
    if "through" in only:
        J += [("through", dict(slip=s, seed=k, length=THROUGH), E.METHODS, EXCLUDE_M, False) for s in SLIPS for k in seeds]
    if "noexcl" in only:
        J += [("noexcl", dict(slip=s, seed=k, length=THROUGH), E.METHODS, 0.0, False) for s in SLIPS[1:] for k in seeds]
    if "lowcoh" in only:
        J += [("lowcoh", dict(slip=4.0, seed=k, length=THROUGH, gamma0=0.35), E.METHODS, EXCLUDE_M, False) for k in seeds]
    if "coast" in only:
        J += [("coast", dict(slip=4.0, seed=k, length=THROUGH, coast=True), E.METHODS, EXCLUDE_M, False) for k in seeds]
    if "patchy" in only:
        J += [("patchy", dict(slip=4.0, seed=k, length=L, patches=(0.25, 500.0, 10000.0)), E.METHODS, EXCLUDE_M, False)
              for L in (synth.LENGTH, THROUGH) for k in seeds]
    if "footprint" in only:
        J += [("footprint", dict(slip=4.0, seed=k, length=L, footprint_w=0.365), ("ours", "diff_snaphu", "diff_isce3"),
               EXCLUDE_M, False) for L in (synth.LENGTH, THROUGH) for k in seeds]
    if "offset" in only:
        J += [("offset", dict(slip=4.0, seed=k, offset_b_m=o), ("ours", "diff_smooth", "diff_snaphu"), EXCLUDE_M, False)
              for o in (0.0, 3.0, 9.0, 20.0) for k in seeds]
    if "example" in only:
        J += [("example_finite", dict(slip=4.0, seed=0), E.METHODS, EXCLUDE_M, True),
              ("example_through", dict(slip=4.0, seed=0, length=THROUGH), E.METHODS, EXCLUDE_M, True)]
    return J


def cycle_experiment():
    """The main band's south side one cycle off: how much each estimate moves, raw and filtered."""
    ca, cb, tr = synth.scene(slip=4.0, seed=0, length=synth.LENGTH_THROUGH)
    psi_a, psi_b, ga, gb, base, use, sig = inputs(ca, cb, tr)
    south = tr["dist"] < 0
    moved = np.where(south, tr["phase_a"] + 2 * np.pi, tr["phase_a"])
    rows = []
    for m in E.METHODS:
        o0 = E.estimate(m, tr["phase_a"], psi_a, psi_b, ga, gb, use, sig, synth.F_A, synth.F_B, synth.CELL,
                        x=tr["x"], y=tr["y"], data=tr["land"])
        o1 = E.estimate(m, moved, psi_a, psi_b, ga, gb, use, sig, synth.F_A, synth.F_B, synth.CELL,
                        x=tr["x"], y=tr["y"], data=tr["land"])
        dr = o1["raw"] - o0["raw"]
        ds = o1["screen"] - o0["screen"]
        far_n = tr["land"] & (tr["dist"] > 20e3)
        ds = ds - np.nanmedian(ds[far_n])
        row = dict(method=m,
                   raw_south_median=float(np.nanmedian(dr[use & south])),
                   raw_north_median=float(np.nanmedian(dr[use & ~south])))
        for a, b in ((-8, -4), (-4, -2), (-2, 0), (0, 2), (2, 4), (4, 8), (20, 40), (-40, -20)):
            m_ = tr["land"] & (tr["dist"] >= a * 1e3) & (tr["dist"] < b * 1e3)
            row[f"screen_{a}_{b}km"] = float(np.nanmedian(ds[m_]))
        rows.append(row)
        print(f"cycle {m:13s} raw S {row['raw_south_median']:+7.3f} N {row['raw_north_median']:+7.3f} | screen "
              + " ".join(f"{k[7:]} {v:+6.2f}" for k, v in row.items() if k.startswith("screen_")))
    return pd.DataFrame(rows)


def main():
    opt = dict(t.split("=", 1) for t in sys.argv[1:] if "=" in t)
    only = opt.get("only", "finite,through,noexcl,lowcoh,coast,patchy,offset,example,cycle").split(",")
    workers = int(opt.get("workers", 6))
    os.makedirs(RESULTS, exist_ok=True)
    synth.unit_slip_los(synth.LENGTH)                            # build the Okada caches once, before the workers
    synth.unit_slip_los(synth.LENGTH_THROUGH)
    J = jobs(only)
    rows, examples = [], []
    if J:
        from multiprocessing import get_context

        with get_context("spawn").Pool(workers) as pool:
            for r, keep in pool.imap_unordered(run_scene, J):
                rows += r
                if keep is not None:
                    examples.append(keep)
        df = pd.DataFrame(rows)
        path = os.path.join(RESULTS, "synthetic.csv")
        if os.path.exists(path) and "only" in opt:                # a partial run replaces its own experiments only
            old = pd.read_csv(path)
            df = pd.concat([old[~old.experiment.isin(df.experiment.unique())], df], ignore_index=True)
        df.to_csv(path, index=False)
        print("wrote", path)
    for name, screens, tr, extra in examples:
        tx, ty = synth.trace_xy(float(tr["length"]))
        np.savez_compressed(os.path.join(RESULTS, f"synthetic_{name}.npz"),
                            **{f"screen_{k}": v for k, v in screens.items()},
                            **{f"truth_{k}": np.asarray(v) for k, v in tr.items()},
                            **{k: np.asarray(v) for k, v in extra.items()}, trace_x=tx, trace_y=ty)
    if "cycle" in only:
        cycle_experiment().to_csv(os.path.join(RESULTS, "synthetic_cycle.csv"), index=False)


if __name__ == "__main__":
    main()
