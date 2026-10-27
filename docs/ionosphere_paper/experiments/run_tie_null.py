"""The side-band tie on FAKE boundaries: straight lines through one block (the Panama north blocks, the Venezuela main
regions), each with a 2 km gap cut out, as the fault zone is. Nothing jumps there, so every estimate should be 0 cycles:
their spread is the method's error on real ionosphere, noise and main-band errors.

    .../envs/remote_sensing/bin/python -u run_tie_null.py        -> results/tie_null.csv
"""
import os
import sys

for _n in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_n, "3")
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dispersive_tie as T                      # noqa: E402
from run_nisar import signed_distance           # noqa: E402

GAP_M = 2000.0
AWAY_M = 15000.0                                # cells this close to the real trace stay out
ORIENT = (0.0, 45.0, 90.0, 135.0)               # line directions, deg from east
CONFIGS = [dict(reach_m=8e3, deg=2), dict(reach_m=12e3, deg=2), dict(reach_m=16e3, deg=2), dict(reach_m=12e3, deg=1)]
WINDOW_M = 20000.0


def scenes():
    import panama_scenes as P
    import venezuela_scenes as V
    for tr, sc in (("D141", "v8"), ("D040", "v8"), ("A076", "v8"), ("A004", "v6")):
        S = P.load(tr, sc, roff=False)
        yield f"{tr} {sc}", S, S["block_n"]
    for tr in ("A061", "D126", "A162", "D054"):
        S = V.load(tr)
        yield f"{tr} v6", S, S["region"]


def main():
    rows = []
    for name, S, region in scenes():
        X, Y = np.meshgrid(S["x"], S["y"])
        tx, ty = S["trace"]
        far = np.abs(signed_distance(X, Y, tx, ty)) > AWAY_M
        sig = T.approx_sigma(S["coh_a"], S["coh_b"], S["f_a"], S["f_b"], S["cell"])
        base = region & far
        xm, ym = X[base].mean(), Y[base].mean()
        for th in ORIENT:
            u = np.array([np.cos(np.deg2rad(th)), np.sin(np.deg2rad(th))])
            v = np.array([-u[1], u[0]])
            proj = (X - xm) * v[0] + (Y - ym) * v[1]
            along = (X - xm) * u[0] + (Y - ym) * u[1]
            lo, hi = np.percentile(proj[base], [5, 95])
            for c in np.arange(np.ceil(lo / 12e3) * 12e3, hi, 12e3):
                nl = proj - c
                a = base & (nl > 0)
                b = base & (nl < 0)
                use = S["use"] & base & (np.abs(nl) >= GAP_M)
                for k, cfg in enumerate(CONFIGS):
                    r = T.block_tie(S["d"], use, a, b, S["x"], S["y"], sigma=sig, window_m=WINDOW_M, along=along,
                                    min_cells=60, **cfg)
                    for w in r["windows"]:
                        rows.append(dict(scene=name, orient=th, offset_km=c / 1e3, config=k, reach_km=cfg["reach_m"] / 1e3,
                                         deg=cfg["deg"], e=w["e"], se=w["se"], cells_a=w["cells_a"], cells_b=w["cells_b"],
                                         x=w["x"], y=w["y"]))
        df = pd.DataFrame([r for r in rows if r["scene"] == name])
        if len(df):
            for k, g in df.groupby("config"):
                print(f"{name} config {k} ({CONFIGS[k]}): {len(g)} windows, |e| p50 {g.e.abs().median():.2f} "
                      f"p90 {g.e.abs().quantile(0.9):.2f} p95 {g.e.abs().quantile(0.95):.2f} max {g.e.abs().max():.2f}; "
                      f"|e| > 0.5: {np.mean(g.e.abs() > 0.5):.1%}", flush=True)
    out = pd.DataFrame(rows)
    os.makedirs(os.path.join(HERE, "..", "results"), exist_ok=True)
    out.to_csv(os.path.join(HERE, "..", "results", "tie_null.csv"), index=False)
    print("->", os.path.join(HERE, "..", "results", "tie_null.csv"))


if __name__ == "__main__":
    main()
