"""Time Green's-function assembly: half-space against layered, adaptive against fixed.

The claim this exists to support is a single number: **with adaptive quadrature a
layered engine costs about the same as the closed-form homogeneous one**, and the
reference's fixed 91-point rule costs two orders of magnitude more than either.
That matters because a homogeneous half-space gives the whole crust one rigidity,
and the shallowest few kilometres are much softer than that -- so assuming
otherwise makes the same surface displacement require less shallow slip and more
deep slip, a systematic bias in exactly the quantity a coseismic inversion is for.
If the layered engine is free, there is no reason left to accept that bias.

Runs on a **synthetic** fault and homogeneous EDGRN tables by default, so it needs
no data and no Fortran and reproduces anywhere. ``--tables edgrn.inp`` swaps in
real layered tables; the timings barely move, since the cost is the number of
table lookups rather than what is in the table.

    python scripts/bench_greens.py
    python scripts/bench_greens.py --obs 2000 --elements 400 --csv out.csv
    python scripts/bench_greens.py --tables workdir/edgrn_venezuela/edgrn.inp

⚠️ **The reference rule is `tolerance=None` WITH `max_points=91`.** Passing
``tolerance=None`` alone gives the *cap* (900 points, order 30), which is nine
times more work than the rule it is supposed to reproduce. That mistake once cost
a whole launch, so this script never spells it any other way.
"""

import argparse
import time
import warnings
from pathlib import Path

import numpy as np

warnings.simplefilter("ignore")

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nisar_tools.slip import (  # noqa: E402
    EdgrnTables,
    FaultMesh,
    FaultTrace,
    HalfSpaceTDE,
    LayeredPointSource,
)
from nisar_tools.slip.layered import REFERENCE_QUADRATURE_POINTS  # noqa: E402

#: A gently curved trace, so the mesh is not a single flat plane.
_LON = np.array([-68.681, -68.300, -67.900, -67.500, -67.100, -66.700, -66.523])
_LAT = np.array([10.410, 10.490, 10.540, 10.565, 10.595, 10.620, 10.630])


def build(elements, observations, seed=0):
    """A mesh of about ``elements`` triangles and observations around it."""
    trace = FaultTrace(_LON, _LAT, name="bench")
    frame = trace.local_frame()
    # Element count scales as 1/edge^2, so solve for the edge that lands near the
    # requested count rather than making the caller guess.
    edge = float(np.sqrt(2.0 * trace.length() * 20e3 / max(elements, 1)))
    mesh = FaultMesh.curved(trace, frame, uniform_dip=70.0, max_depth=20e3,
                            edge_length=edge)

    rng = np.random.default_rng(seed)
    cx, cy = mesh.centroids[:, 0], mesh.centroids[:, 1]
    ox = rng.uniform(cx.min() - 40e3, cx.max() + 40e3, observations * 2)
    oy = rng.uniform(cy.min() - 50e3, cy.max() + 50e3, observations * 2)
    keep = trace.distance(ox, oy, frame) > 8e3
    return mesh, ox[keep][:observations], oy[keep][:observations]


def tables_for(path):
    if path:
        print(f"tables: {path}")
        return EdgrnTables.from_input_file(Path(path))
    print("tables: synthetic homogeneous (no Fortran needed)")
    return EdgrnTables.homogeneous(nu=0.25, r=np.linspace(0.0, 300e3, 301),
                                   z=np.linspace(250.0, 30e3, 120), n_azimuth=16)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--obs", type=int, default=400)
    ap.add_argument("--elements", type=int, default=240)
    ap.add_argument("--basis", default="element", choices=("element", "node"))
    ap.add_argument("--tables", default=None, help="path to an edgrn.inp")
    ap.add_argument("--tolerances", default="1e-2,3e-3,1e-3")
    ap.add_argument("--skip-reference", action="store_true",
                    help="the fixed 91-point rule is ~100x slower; skip it")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    mesh, ox, oy = build(args.elements, args.obs)
    oz = np.zeros_like(ox)
    from nisar_tools.slip.basis import make_basis
    basis = make_basis(mesh, args.basis)
    # ⚠️ r/L is the variable that decides everything here, so report it. The
    # adaptive rule asks for order ~20*L/r, so it is cheap exactly where
    # observations are far from an element in units of that element's own size.
    # A small benchmark mesh has large elements and therefore *small* r/L, which
    # understates the win: the real six-scene problem runs at median r/L 38.
    d = np.sqrt((ox[:, None] - mesh.centroids[None, :, 0]) ** 2
                + (oy[:, None] - mesh.centroids[None, :, 1]) ** 2
                + mesh.centroids[None, :, 2] ** 2)
    ratio = d / np.sqrt(2.0 * mesh.areas)[None, :]
    print(f"mesh {mesh.n_elements} elements / {mesh.n_nodes} nodes, "
          f"{ox.size} observations, {args.basis} basis "
          f"({basis.n_basis} basis functions)")
    print("r/L over all pairs: p05 %.1f  median %.1f  p95 %.1f   "
          "(real six-scene problem: median 38)\n"
          % (np.percentile(ratio, 5), np.median(ratio), np.percentile(ratio, 95)))

    tables = tables_for(args.tables)
    runs = [("HalfSpaceTDE", HalfSpaceTDE(0.25), None)]
    for tol in (float(t) for t in args.tolerances.split(",")):
        runs.append((f"layered adaptive {tol:g}",
                     LayeredPointSource(tables, tolerance=tol), tol))
    if not args.skip_reference:
        runs.append((f"layered reference ({REFERENCE_QUADRATURE_POINTS} pt)",
                     LayeredPointSource(tables, tolerance=None,
                                        max_points=REFERENCE_QUADRATURE_POINTS),
                     None))

    print("%-30s %10s %14s %9s %12s %9s"
          % ("assembly", "seconds", "s / 1000 obs", "vs h-space", "pts / pair",
             "capped %"))
    rows, baseline = [], None
    for label, eng, _ in runs:
        supports_basis = bool(getattr(eng, "supports_basis", False))
        t0 = time.time()
        if supports_basis:
            g = eng.los_matrix(mesh, ox, oy, np.full(ox.size, 0.6),
                               np.full(ox.size, 0.1), np.full(ox.size, 0.8),
                               z=oz, basis=basis)
        else:
            g = eng.los_matrix(mesh, ox, oy, np.full(ox.size, 0.6),
                               np.full(ox.size, 0.1), np.full(ox.size, 0.8), z=oz)
            if basis.name != "element":
                g = g @ np.eye(g.shape[1])          # shape only; timing is above
        dt = time.time() - t0
        baseline = dt if baseline is None else baseline
        report = getattr(eng, "quadrature_report", None) or {}
        rows.append(dict(assembly=label, seconds=round(dt, 3),
                         per_1000_obs=round(1000 * dt / ox.size, 3),
                         ratio_to_halfspace=round(dt / baseline, 2),
                         points_per_pair=round(report.get("points_per_pair", 0.0), 2),
                         capped_fraction=round(report.get("capped_fraction", 0.0), 5),
                         shape=str(g.shape)))
        r = rows[-1]
        print("%-30s %10.2f %14.2f %9.2fx %12.2f %8.2f%%"
              % (label, r["seconds"], r["per_1000_obs"], r["ratio_to_halfspace"],
                 r["points_per_pair"], 100 * r["capped_fraction"]), flush=True)

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\n-> {args.csv}")

    default = next((r for r in rows if "adaptive 0.003" in r["assembly"]), None)
    reference = next((r for r in rows if "reference" in r["assembly"]), None)
    halfspace = rows[0]
    if default is not None:
        print()
        if reference is not None:
            print("at tolerance 3e-3: %.0fx fewer quadrature points than the "
                  "reference rule (%.2f vs %.0f per pair), %.1fx faster"
                  % (reference["points_per_pair"] / max(default["points_per_pair"], 1e-9),
                     default["points_per_pair"], reference["points_per_pair"],
                     reference["seconds"] / default["seconds"]))
        verb = "cheaper than" if default["seconds"] < halfspace["seconds"] else "the cost of"
        print("layered assembly is %.2fx %s the closed-form half-space"
              % (halfspace["seconds"] / default["seconds"] if default["seconds"] else float("nan"),
                 verb))


if __name__ == "__main__":
    main()
