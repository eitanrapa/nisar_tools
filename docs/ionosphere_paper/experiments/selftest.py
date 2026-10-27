"""Checks of `estimators.py`, noise-free, on 150 x 150 cells (run before the experiments).

1. A gentle deformation (+-5 rad) and ionosphere (+-25 rad), every cell used: every estimator's raw estimate equals
   the true dispersive phase, up to a constant, to round-off. The three forms are one estimator when the unwrapping
   is right; the double difference stays within +-pi, so even "wrapped" holds. ISCE3's passes find nothing more than
   4 km inside, but do move cycles at the grid's edge: their one-sided low-pass lags a 2.7 rad/km gradient there.
2. A smooth but steep deformation (a 6 km tanh of +-40 rad on a 0.002 rad/m ramp): the forms without corrections
   are still exact; ISCE3's passes move cycles (reported, not a failure: this is the mechanism), and "wrapped"
   fails once the double difference spans more than 2 pi (expected).
3. A cycle of the main band on half the scene moves the reordered and reformulated raw estimates by 2 pi c0 = 3.21 rad
   there, and the two-band one by 2 pi a = 76.2 rad (Table 1 of the paper).

usage: .../envs/remote_sensing/bin/python selftest.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import estimators as E  # noqa: E402

F_A, F_B, CELL = 1239e6, 1293.5e6, 240.0


def main():
    n = 150
    y, x = np.mgrid[0:n, 0:n] * CELL
    io = 25 * np.sin(x / 9e3) * np.cos(y / 13e3)
    coh = np.full((n, n), 0.9)
    use = np.ones((n, n), bool)
    sig = np.ones((n, n))
    xc, yc = x[0], y[:, 0]
    C = E.coefficients(F_A, F_B)
    edge_km = np.minimum.reduce([y, x, y.max() - y, x.max() - x]) / 1e3
    ok = True
    for case, psi, exact in (("1. gentle", 5 * np.tanh((y - 18e3) / 6e3), set(E.METHODS)),
                             ("2. steep", 40 * np.tanh((y - 18e3) / 6e3) + 0.002 * x,
                              {"ours", "diff_smooth", "diff_snaphu", "side_snaphu", "side_isce2"})):
        phi_a = psi + io
        phi_b = (F_B / F_A) * psi + (F_A / F_B) * io
        psi_a, psi_b = E.ss.wrap(phi_a), E.ss.wrap(phi_b)
        for m in E.METHODS:
            out = E.estimate(m, phi_a, psi_a, psi_b, coh, coh, use, sig, F_A, F_B, CELL, x=xc, y=yc)
            e = out["raw"] - io
            e = e - np.median(e)
            err = float(np.max(np.abs(e)))
            if m in exact and not m.endswith("isce3"):
                good = err < 1e-6
                ok &= good
                verdict = "ok" if good else "FAIL"
            elif m in exact:                 # ISCE3's one-sided low-pass lags the gradient at the grid's edge
                inner = float(np.max(np.abs(e[edge_km > 4.0])))
                good = inner < 1e-6
                ok &= good
                verdict = (f"{'ok' if good else 'FAIL'}: exact more than 4 km inside (max {inner:.1e}); cycles moved at "
                           f"the grid edge only, where the one-sided low-pass lags the gradient")
            else:
                verdict = f"(expected to move: cycles A {out['cycles_a']:.3f} B {out['cycles_b']:.3f})"
            print(f"{case} {m:13s} max |raw - truth - const| = {err:.1e} rad  {verdict}")
    psi = 5 * np.tanh((y - 18e3) / 6e3)
    phi_a = psi + io
    phi_b = (F_B / F_A) * psi + (F_A / F_B) * io
    psi_a, psi_b = E.ss.wrap(phi_a), E.ss.wrap(phi_b)
    half = y > 18e3
    moved = np.where(half, phi_a + 2 * np.pi, phi_a)
    for m, expect in (("ours", 2 * np.pi * C["c0"]), ("diff_snaphu", 2 * np.pi * C["c0"]),
                      ("side_snaphu", 2 * np.pi * C["a"])):
        o0 = E.estimate(m, phi_a, psi_a, psi_b, coh, coh, use, sig, F_A, F_B, CELL)
        o1 = E.estimate(m, moved, psi_a, psi_b, coh, coh, use, sig, F_A, F_B, CELL)
        dd = o1["raw"] - o0["raw"]
        got = float(np.median(dd[half]) - np.median(dd[~half]))
        good = abs(got - expect) < 1e-6
        ok &= good
        print(f"3. {m:13s} one main-band cycle on half the scene: {got:.4f} rad (expected {expect:.4f})  "
              f"{'ok' if good else 'FAIL'}")
    print("all ok" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
