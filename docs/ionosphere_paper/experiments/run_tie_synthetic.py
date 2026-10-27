"""The side-band tie on synthetic through-going ruptures (`synth.py`), where the truth is known.

The reordered estimate is computed once per scene from the oracle main band (both blocks on their true integers), so
the right answer is 0 cycles. A wrong tie of the south block by e cycles moves its raw estimate by exactly -3.21 e rad
(`selftest.py`; the unwrapping of r does not change for |e| <= 2, checked below), and patches of local main-band
cycle errors near the trace are added the same way. Each scene is scored by both estimators (`dispersive_tie.block_tie`,
a robust polynomial, on cells 2-12 km from the gap and on 6-12 km only; `gp_tie`, kriging).

    .../envs/remote_sensing/bin/python -u run_tie_synthetic.py [workers=3]   -> ../results/tie_synthetic.csv
"""
import functools
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
for _n in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_n, "1")
print = functools.partial(print, flush=True)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import ndimage  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dispersive_tie as T  # noqa: E402
import estimators as E  # noqa: E402
import run_synthetic as R  # noqa: E402
import synth  # noqa: E402

RESULTS = os.path.join(HERE, "..", "results")
VARIANTS = {
    "default": dict(),
    "rough": dict(iono=dict(gradient=0.6, rf_rms=8.0, rf_km=8.0)),
    "tid": dict(iono=dict(gradient=0.6, rf_rms=4.0, rf_km=15.0, tid_amp=3.0, tid_km=25.0)),
    "lowcoh": dict(gamma0=0.35),
    "patchy": dict(patches=(0.25, 500.0, 10000.0)),
}
SLIP = (0.0, 4.0)
SEEDS = range(3)
PLANT = (0, 1, -2)                  # cycles the south block needs (its main band off by -e)
SLIPS_LOCAL = (0.0, 0.15, 0.35)     # fraction of north-side cells at 2-6 km in +1-cycle patches (~1 km blobs)


def local_patches(tr, frac, rng):
    """Cells on the north side 2-6 km from the trace, in ~1 km blobs covering ``frac`` of them."""
    near = (tr["dist"] > 2e3) & (tr["dist"] < 6e3)
    if frac <= 0:
        return np.zeros(near.shape, bool)
    f = ndimage.gaussian_filter(rng.standard_normal(near.shape), 1000.0 / synth.CELL)
    return near & (f > np.quantile(f[near], 1 - frac))


def run(job):
    variant, slip, seed = job
    t0 = time.time()
    kw = dict(VARIANTS[variant])
    ca, cb, tr = synth.scene(slip=slip, seed=seed, length=synth.LENGTH_THROUGH, **kw)
    psi_a, psi_b, ga, gb, base, use, sig = R.inputs(ca, cb, tr)
    out = E.estimate("ours", tr["phase_a"], psi_a, psi_b, ga, gb, use, sig, synth.F_A, synth.F_B, synth.CELL,
                     x=tr["x"], y=tr["y"], data=tr["land"])
    d0 = out["raw"]
    rows = []
    if variant == "default" and seed == 0:
        # the identity the planted ties rest on: the south block one cycle off through the whole chain
        pa1 = np.where(tr["dist"] < 0, tr["phase_a"] + 2 * np.pi, tr["phase_a"])
        d1 = E.estimate("ours", pa1, psi_a, psi_b, ga, gb, use, sig, synth.F_A, synth.F_B, synth.CELL,
                        x=tr["x"], y=tr["y"], data=tr["land"])["raw"]
        south = use & (tr["dist"] < 0)
        print(f"check: the south block one cycle off moves its raw estimate by {np.nanmedian((d1 - d0)[south]):+.4f} rad "
              f"(min {np.nanmin((d1 - d0)[south]):+.4f}, max {np.nanmax((d1 - d0)[south]):+.4f}; expected +{T.STEP:.4f}), "
              f"north {np.nanmax(np.abs((d1 - d0)[use & (tr['dist'] > 0)])):.1e}")
    X, Y = np.meshgrid(tr["x"], tr["y"])
    s_dir = np.array([np.sin(np.deg2rad(synth.STRIKE_DEG)), np.cos(np.deg2rad(synth.STRIKE_DEG))])
    along = X * s_dir[0] + Y * s_dir[1]
    bn, bs = tr["land"] & (tr["dist"] > 0), tr["land"] & (tr["dist"] < 0)
    rng = np.random.default_rng(1000 + seed)
    for frac in SLIPS_LOCAL:
        pat = local_patches(tr, frac, rng)
        for e_true in PLANT:
            d = d0 + np.where(bs, -T.STEP * e_true, 0.0) + np.where(pat, T.STEP, 0.0)
            for est, gap in (("polynomial", 0.0), ("polynomial 6-12 km", 6e3)):
                p = T.block_tie(d, use, bn, bs, tr["x"], tr["y"], sigma=sig, reach_m=12e3, window_m=20e3, along=along,
                                deg=2, min_cells=60, min_gap_m=gap)
                rows.append(dict(variant=variant, slip=slip, seed=seed, local=frac, e_true=e_true, estimator=est,
                                 e=p["e"], e_wmean=p["e_wmean"], spread=p["spread"], windows=len(p["windows"]),
                                 worst=max((abs(w["e"] - e_true) for w in p["windows"]), default=np.nan)))
            # kriging is slower: e = 0 only (the planted ties are exact shifts), without and with the most local errors
            if e_true == 0 and frac != SLIPS_LOCAL[1]:
                g = T.gp_tie(d, use, bn, bs, tr["x"], tr["y"], sigma=sig, reach_m=12e3, window_m=20e3, along=along)
                rows.append(dict(variant=variant, slip=slip, seed=seed, local=frac, e_true=e_true, estimator="kriging",
                                 e=g["e"], e_wmean=g["e_wmean"], spread=np.nan, windows=len(g["windows"]),
                                 worst=max((abs(w["e"]) for w in g["windows"]), default=np.nan), se=g["se_wmean"], chi2=g["chi2"]))
    print(f"{variant} slip {slip} seed {seed} [{time.time() - t0:.0f}s]: "
          + "; ".join(f"{r['estimator'][:4]} local {r['local']} e_true {r['e_true']:+d}: {r['e']:+.2f}" for r in rows
                      if r["e_true"] == 0))
    return rows


def main():
    workers = int(dict(a.split("=") for a in sys.argv[1:] if "=" in a).get("workers", 3))
    jobs = [(v, s, k) for v in VARIANTS for s in SLIP for k in SEEDS]
    if workers > 1:
        from multiprocessing import get_context
        with get_context("spawn").Pool(workers) as pool:
            res = pool.map(run, jobs, chunksize=1)
    else:
        res = [run(j) for j in jobs]
    df = pd.DataFrame([r for rows in res for r in rows])
    df.to_csv(os.path.join(RESULTS, "tie_synthetic.csv"), index=False)
    df["err"] = df.e - df.e_true
    print(df.groupby(["estimator", "variant", "local"]).err.agg(["count", "mean", "std", lambda v: np.abs(v).max()]).round(3))
    print("->", os.path.join(RESULTS, "tie_synthetic.csv"))


if __name__ == "__main__":
    main()
