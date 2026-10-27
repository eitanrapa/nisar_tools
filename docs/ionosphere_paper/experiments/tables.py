"""The paper's tables, from ../results -> ../sections/tables/*.tex (and a plain-text echo).

usage: .../envs/remote_sensing/bin/python tables.py

tab_synthetic   per experiment and estimator: error rms 0-2 / 2-5 / 5-10 / >20 km (mm LOS), the leak and the 2-4 km step
                (mm), mean over seeds (and the worst seed for the error at 2-5 km); slip 4 m
tab_nisar       per NISAR scene and estimator: leak (slope, r), screen step (mm), difference from ours near / far (mm)
tab_tie         the Panama south blocks' ties: the offsets' against the side band's (two estimators) and against m read
                with the side band's estimator; the Venezuela controls and the fake boundaries
tab_nearfield   the near-trace recipe, one change at a time (synthetic, 4 m, with and without D141-like near-field
                roughness): error rms by distance and the 2-4 km step, mean over seeds
tab_fill        the fills: synthetic ionospheres (fault zones, a bay, an edge strip) and fake gaps in eight NISAR scenes
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from figures import LABEL, LOS_MM, ORDER  # noqa: E402

RES = os.path.join(HERE, "..", "results")
OUT = os.path.join(HERE, "..", "sections", "tables")
EXPS = [("finite", "finite rupture"), ("through", "through-going"), ("noexcl", "through-going, no exclusion")]
ROBUST = [("lowcoh", "coherence 0.35"), ("coast", "coast"), ("patchy", "patchy decorrelation")]
ROBUST_METHODS = ("ours", "diff_smooth", "diff_snaphu", "diff_isce3", "side_snaphu")
#: the NISAR table's rows (every arm is in results/nisar_<scene>.csv)
NISAR_METHODS = ("ours", "diff_smooth", "diff_snaphu", "diff_isce3_noexcl", "side_snaphu", "gunw")


def fmt(v, nd=1):
    if v is None or not np.isfinite(v):
        return "--"
    if abs(v) >= 1e4:
        return f"{v:.1e}".replace("e+0", "e").replace("e+", "e")
    return f"{v:.{nd}f}"


def tab_synthetic(exps=None, methods=None, name="tab_synthetic"):
    p = os.path.join(RES, "synthetic.csv")
    if not os.path.exists(p):
        return
    exps = EXPS if exps is None else exps
    df = pd.read_csv(p)
    df = df[df.slip == 4.0]
    lines = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r"case & estimator & 0--2\,km & 2--5\,km & 5--10\,km & $>$20\,km & leak & step \\",
             r" & & \multicolumn{4}{c}{error rms (mm LOS)} & & (mm) \\", r"\midrule"]
    text = []
    for exp, title in exps:
        sub = df[df.experiment == exp]
        if exp == "patchy":
            sub = sub[sub.length > 1e5]
        if sub.empty:
            continue
        first = True
        for m in [m for m in ORDER if m in set(sub.method) and (methods is None or m in methods)]:
            s = sub[sub.method == m]
            r = {c: s[c].mean() for c in ("rms_0_2", "rms_2_5", "rms_5_10", "rms_20_inf", "leak", "step")}
            worst = s["rms_2_5"].max()
            row = (f"{title if first else ''} & {LABEL[m]} & {fmt(LOS_MM * r['rms_0_2'])} & {fmt(LOS_MM * r['rms_2_5'])}"
                   f" & {fmt(LOS_MM * r['rms_5_10'])} & {fmt(LOS_MM * r['rms_20_inf'])} & {fmt(r['leak'], 3)}"
                   f" & {fmt(LOS_MM * r['step'])} \\\\")
            lines.append(row)
            text.append(f"{exp:8s} {m:18s} rms 0-2 {LOS_MM * r['rms_0_2']:9.1f} 2-5 {LOS_MM * r['rms_2_5']:9.1f} "
                        f"(worst seed {LOS_MM * worst:9.1f}) 5-10 {LOS_MM * r['rms_5_10']:8.1f} >20 "
                        f"{LOS_MM * r['rms_20_inf']:6.1f} leak {r['leak']:+.3f} step {LOS_MM * r['step']:+8.1f} n {len(s)}")
            first = False
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, f"{name}.tex"), "w").write("\n".join(lines) + "\n")
    print("\n".join(text))


def tab_nisar():
    scenes = [s for s in ("D141", "A061", "D126", "A162", "D054") if os.path.exists(os.path.join(RES, f"nisar_{s}.csv"))]
    if not scenes:
        return
    df = pd.concat([pd.read_csv(os.path.join(RES, f"nisar_{s}.csv")) for s in scenes])
    lines = [r"\begin{tabular}{ll" + "r" * 5 + "}", r"\toprule",
             r"scene & estimator & leak & $r$ & step & \multicolumn{2}{c}{difference from ours} \\", r" & & & & (mm) & $<$10\,km & $>$20\,km (mm) \\", r"\midrule"]
    text = []
    for s in scenes:
        first = True
        for m in [m for m in ORDER if m in set(df[df.scene == s].method) and m in NISAR_METHODS]:
            r = df[(df.scene == s) & (df.method == m)].iloc[0]
            lines.append(f"{s if first else ''} & {LABEL[m]} & {fmt(r['leak'], 3)} & {fmt(r['leak_r'], 2)} & "
                         f"{fmt(LOS_MM * r['step'])} & {fmt(LOS_MM * r['diff_near'])} & {fmt(LOS_MM * r['diff_far'])} \\\\")
            text.append(f"{s} {m:18s} leak {r['leak']:+.3f} (r {r['leak_r']:+.2f}) step {LOS_MM * r['step']:+8.1f} mm, "
                        f"diff from ours near {LOS_MM * r['diff_near']:8.1f} far {LOS_MM * r['diff_far']:8.1f} mm; "
                        f"cycles moved A {r.get('cycles_a', np.nan):.4f} B {r.get('cycles_b', np.nan):.4f}")
            first = False
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "tab_nisar.tex"), "w").write("\n".join(lines) + "\n")
    print("\n".join(text))


def tab_tie():
    path = os.path.join(RES, "tie_real.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    null = pd.read_csv(os.path.join(RES, "tie_null.csv")) if os.path.exists(os.path.join(RES, "tie_null.csv")) else None
    lines = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r" & \multicolumn{2}{c}{range offsets} & \multicolumn{3}{c}{side band, $e$} & \multicolumn{2}{c}{$m$, as the side band} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}\cmidrule(lr){7-8}",
             r"pair & tie & measured & polynomial & kriging & windows & 2--12\,km & 6--12\,km \\", r"\midrule"]
    text = []
    for scene, label in (("D141 v8", "D141"), ("D040 v8", "D040"), ("A076 v8", "A076"), ("A004 v6", "A004"),
                         ("D141 v6", "D141, Section 5's cut"), ("A162 v6 (control)", "A162 (no cut)"),
                         ("D054 v6 (control)", "D054 (no cut)")):
        r = df[df.scene == scene]
        if not len(r):
            continue
        r = r.iloc[0]
        k = "--" if pd.isna(r.get("offsets_k")) else f"{int(r['offsets_k'])}"
        meas = fmt(r.get("offsets_measured"), 2)
        lines.append(f"{label} & {k} & {meas} & {fmt(r.get('poly'), 2)} & {fmt(r.get('kriging'), 2)} & "
                     f"{'--' if pd.isna(r.get('poly_windows')) else int(r['poly_windows'])} & {fmt(r.get('m_poly'), 2)} & "
                     f"{fmt(r.get('m_poly_6_12'), 2)} \\\\")
        text.append(f"{label:24s} offsets {k:>4s} ({meas}) side band {fmt(r.get('poly'), 2)} / kriging "
                    f"{fmt(r.get('kriging'), 2)}; m {fmt(r.get('m_poly'), 2)} / 6-12 km {fmt(r.get('m_poly_6_12'), 2)}")
    lines += [r"\bottomrule", r"\end{tabular}"]
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "tab_tie.tex"), "w").write("\n".join(lines) + "\n")
    print("\n".join(text))
    if null is not None:
        g = null[null.config == 1]
        print(f"fake boundaries (reach 12 km, quadratic): {len(g)} windows, |e| median {g.e.abs().median():.3f}, "
              f"p95 {g.e.abs().quantile(0.95):.3f}, max {g.e.abs().max():.2f}, > 0.5: {np.mean(g.e.abs() > 0.5):.2%}")
        for sc, h in g.groupby("scene"):
            print(f"   {sc}: {len(h)} windows, p95 {h.e.abs().quantile(0.95):.3f}, > 0.5: {np.mean(h.e.abs() > 0.5):.2%}")


#: the near-trace recipe, one change at a time: (label, filter, arm, exclusion m)
RECIPE = [("production (v6)", "v6", "cell", 2000.0), ("+ fill matched to the gap", "matched", "cell", 2000.0),
          ("+ first-order filter", "matched1", "cell", 2000.0), ("+ main band out per pixel", "matched1", "pixel", 2000.0),
          ("+ no fault-zone exclusion", "matched1", "pixel", 0.0)]


def tab_nearfield():
    path = os.path.join(RES, "pixel.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    df = df[df.slip == 4.0]

    def rms(sel, col):
        return float(np.sqrt(np.mean(sel[col] ** 2))) * LOS_MM

    lines = [r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
             r" & \multicolumn{4}{c}{Okada field only} & \multicolumn{4}{c}{with D141-like near field} \\",
             r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
             r"recipe & 0--2 & 2--5 & 5--10\,km & step & 0--2 & 2--5 & 5--10\,km & step \\", r"\midrule"]
    text = []
    for label, filt, arm, excl in RECIPE:
        cells = []
        for nf in (0, 1):
            sel = df[(df.nearfield == nf) & (df["filter"] == filt) & (df.arm == arm) & (df.exclude_m == excl)]
            if not len(sel):
                cells += ["--"] * 4
                continue
            cells += [fmt(rms(sel, "rms_0_2")), fmt(rms(sel, "rms_2_5")), fmt(rms(sel, "rms_5_10")),
                      fmt(float(np.sqrt(np.mean(sel.step ** 2))) * LOS_MM)]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
        text.append(f"{label:28s} " + "  ".join(cells))
    far = df[(df["filter"] == "matched1")]
    lines += [r"\bottomrule", r"\end{tabular}"]
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "tab_nearfield.tex"), "w").write("\n".join(lines) + "\n")
    print("\n".join(text))
    print(f"(beyond 20 km, first-order filter: {rms(far, 'rms_20_inf'):.1f} mm)")


#: the fills compared: (name in the CSVs, label)
FILL_ROWS = [("plane 1.5", "plane, 1.5 km"), ("plane 2.6", "plane, 2.6 km"), ("plane 4", "plane, 4 km"),
             ("plane 10", "plane, 10 km (production)"), ("ladder", "ladder 1.5--3--6--12 km"),
             ("ladder 2.6", "ladder 2.6--5--10 km"), ("guide", "ISCE3's guide")]


def tab_fill():
    ps, pr = os.path.join(RES, "fill_synthetic.csv"), os.path.join(RES, "fill_null.csv")
    if not (os.path.exists(ps) and os.path.exists(pr)):
        return
    syn, real = pd.read_csv(ps), pd.read_csv(pr)

    def q(df, cols):
        v = np.concatenate([df[c].to_numpy() for c in cols])
        v = v[np.isfinite(v)]
        return float(np.sqrt(np.mean(v ** 2))) * LOS_MM if v.size else np.nan

    lines = [r"\begin{tabular}{lrrrrrrrrr}", r"\toprule",
             r" & \multicolumn{4}{c}{synthetic ionospheres} & \multicolumn{5}{c}{real ionosphere, fake gaps} \\",
             r"\cmidrule(lr){2-5}\cmidrule(lr){6-10}",
             r" & \multicolumn{3}{c}{2\,km zone} & edge & \multicolumn{3}{c}{zone half-width} & \multicolumn{2}{c}{edge} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-5}\cmidrule(lr){6-8}\cmidrule(lr){9-10}",
             r"fill & default & rough & waves & 4--8\,km & 1\,km & 2\,km & 4\,km & 0--4\,km & 4--8\,km \\", r"\midrule"]
    text = []
    for f, label in FILL_ROWS:
        cells = []
        for v in ("default", "rough", "tid"):
            g = syn[(syn["gap"] == "fault 2") & (syn["variant"] == v) & (syn["fill"] == f)]
            cells.append(fmt(q(g, ["in_0_1", "in_1_2"])) if len(g) else "--")
        g = syn[(syn["gap"] == "edge 12") & (syn["variant"] == "default") & (syn["fill"] == f)]
        cells.append(fmt(q(g, ["in_4_8"])) if len(g) else "--")
        for gname in ("fault 1", "fault 2", "fault 4"):
            g = real[(real["gap"] == gname) & (real["fill"] == f)]
            cells.append(fmt(q(g, ["in_all"])) if len(g) else "--")
        g = real[(real["gap"] == "edge") & (real["fill"] == f)]
        cells += [fmt(q(g, ["in_0_1", "in_1_2", "in_2_4"])) if len(g) else "--", fmt(q(g, ["in_4_8"])) if len(g) else "--"]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
        text.append(f"{label:30s} " + "  ".join(f"{c:>5s}" for c in cells))
    lines += [r"\bottomrule", r"\end{tabular}"]
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "tab_fill.tex"), "w").write("\n".join(lines) + "\n")
    print("\n".join(text))
    print(f"(real: {real.groupby(['scene', 'window']).ngroups} windows in {real.scene.nunique()} scenes)")


def tab_tie_readings():
    """The Panama ties read three ways from three fields (run_tie_panama_checks.py): cycles to add to the pipelines'
    offset ties."""
    path = os.path.join(RES, "tie_panama_checks.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    df = df[df.check == "GUNW offsets"].set_index("track")
    lines = [r"\begin{tabular}{lrrrrrrrrr}", r"\toprule",
             r" & \multicolumn{3}{c}{side band} & \multicolumn{3}{c}{offsets, smoothed} & \multicolumn{3}{c}{offsets, raw} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
             r"pair & 2\,km & 6\,km & planes & 2\,km & 6\,km & planes & 2\,km & 6\,km & planes \\",
             r"\midrule"]
    for track in ("D141", "D040", "A076"):
        r = df.loc[track]
        vals = []
        for field in ("side band", "our offsets", "our offsets, raw"):
            vals += [fmt(r[f"{field}: 2-12 km"], 2), fmt(r[f"{field}: 6-12 km"], 2), fmt(r[f"{field}: pipeline"], 2)]
        lines.append(f"{track} & " + " & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "tab_tie_readings.tex"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def tab_antarctica():
    """The Antarctic pairs (run_antarctica.py): the product's screen against ours, the components' split three ways, the
    trunk test and the residual metric."""
    import json
    path = os.path.join(RES, "antarctica.json")
    if not os.path.exists(path):
        return
    S = json.load(open(path))
    zones = pd.read_csv(os.path.join(RES, "antarctica_zones.csv"))
    a, b = S["1"], S["2"]
    zz = lambda p, zone, col: float(zones[(zones.pair == p) & (zones.sigma_km == 3) & (zones.zone == zone)][col].iloc[0])  # noqa: E731
    f = lambda v, nd=2: "--" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"  # noqa: E731
    km = lambda v: f"{v:,.0f}".replace(",", "\\,")  # noqa: E731
    rs = lambda s, name, sel: s["residual"][name][sel]  # noqa: E731
    lines = [r"\begin{tabular}{lrr}", r"\toprule", r" & pair 1 & pair 2 \\",
             r" & 19--31 Aug & 31 Aug--12 Sep \\", r"\midrule",
             r"\multicolumn{3}{l}{\emph{The product's screen minus ours}} \\",
             f"component 1, in main-band cycles & {f(a['component1_cycles'], 3)} & {f(b['component1_cycles'], 3)} \\\\",
             f"the block, in double-difference cycles & {f(a['block_cycles'], 4)} & {f(b['block_cycles'], 4)} \\\\",
             f"\\quad its area (km$^2$) & {km(a['block_km2'])} & {km(b['block_km2'])} \\\\",
             f"left after the integers: robust sd (rad) & {f(a['leftover_rsd'])} & {f(b['leftover_rsd'])} \\\\",
             f"\\quad area beyond 3\\,rad (km$^2$) & {km(a['leftover_km2_over3'])} & {km(b['leftover_km2_over3'])} \\\\",
             r"\addlinespace", r"\multicolumn{3}{l}{\emph{The split of component 1, in main-band cycles}} \\",
             f"side band (median of {a['n_windows']} windows) & {f(a['tie_side_band'])} & {f(b['tie_side_band'])} \\\\",
             f"range offsets, $\\phi_A - k\\rho_\\mathrm{{off}}$: the GUNW's & {f(a['tie_offsets'])} & {f(b['tie_offsets'])} \\\\",
             *([f"\\quad ours, a gross-outlier cull only & {f(a.get('tie_offsets_ours'))} & {f(b.get('tie_offsets_ours'))} \\\\"]
               if "tie_offsets_ours" in a or "tie_offsets_ours" in b else []),
             f"ITS\\_LIVE, slow ice within 20\\,km & {f(a.get('tie_velocity_near'))} & {f(b.get('tie_velocity_near'))} \\\\",
             r"GrIMP's fix \citep{joughin2026nisarerrors} & 7 & 7 \\",
             f"non-dispersive step across the gap (rad) & {abs(a['nd_step_max']):.0f}--{abs(a['nd_step_min']):.0f} & "
             f"{abs(b['nd_step_max']):.0f}--{abs(b['nd_step_min']):.0f} \\\\",
             r"\addlinespace", r"\multicolumn{3}{l}{\emph{Over the tributary's trunk, 5--20\,km outside the block}} \\",
             f"product $-$ ours $-$ integers (rad) & {a['plateau_product']:.2f} & {b['plateau_product']:.2f} \\\\",
             f"\\quad in main-band cycles & {a['plateau_product_cycles']:.2f} & {b['plateau_product_cycles']:.2f} \\\\",
             f"ISCE3's passes on our cells (rad) & {a['plateau_corrections']:.2f} & {b['plateau_corrections']:.2f} \\\\",
             f"leak away from the integer steps: product & {a['leak_away']['product, integers repaired']['slope']:.3f} "
             f"({a['leak_away']['product, integers repaired']['r']:.2f}) & "
             f"{b['leak_away']['product, integers repaired']['slope']:.3f} ({b['leak_away']['product, integers repaired']['r']:.2f}) \\\\",
             f"\\quad ours & {a['leak_away']['ours']['slope']:.3f} ({a['leak_away']['ours']['r']:.2f}) & "
             f"{b['leak_away']['ours']['slope']:.3f} ({b['leak_away']['ours']['r']:.2f}) \\\\",
             r"\addlinespace", r"\multicolumn{3}{l}{\emph{The offsets' ionosphere minus each screen (rad)}} \\",
             f"product above ours by $>$3\\,rad: minus ours / minus product & {zz(1, 'above', 'off_minus_ours'):+.2f} / "
             f"{zz(1, 'above', 'off_minus_product'):+.2f} & {zz(2, 'above', 'off_minus_ours'):+.2f} / {zz(2, 'above', 'off_minus_product'):+.2f} \\\\",
             f"product below ours by $>$3\\,rad: minus ours / minus product & {zz(1, 'below', 'off_minus_ours'):+.2f} / "
             f"{zz(1, 'below', 'off_minus_product'):+.2f} & {zz(2, 'below', 'off_minus_ours'):+.2f} / {zz(2, 'below', 'off_minus_product'):+.2f} \\\\",
             *([f"over the frame, rms (6\\,km): product / repaired / ours & "
                + " & ".join(" / ".join(f"{S['quality']['offsets_iono'][n][p]['rms']:.2f}"
                                        for n in ("GUNW", "GUNW, repaired", "ours")) for p in ("1", "2")) + " \\\\"]
               if "quality" in S else []),
             r"\addlinespace", r"\multicolumn{3}{l}{\emph{Phase minus an ITS\_LIVE synthetic, slow ice: $\sigma$ / robust $\sigma$ (rad)}} \\"]
    for name, label in (("uncorrected", "uncorrected"), ("product screen", "the product's screen"),
                        ("product, integers repaired", "the product's, integers repaired"), ("ours (tied)", "ours")):
        lines.append(f"{label} & {rs(a, name, 'slow')[0]:.2f} / {rs(a, name, 'slow')[1]:.2f} & "
                     f"{rs(b, name, 'slow')[0]:.2f} / {rs(b, name, 'slow')[1]:.2f} \\\\")
    if "quality" in S:
        q = S["quality"]["pair_difference"]
        lines += [r"\addlinespace",
                  r"\multicolumn{3}{l}{\emph{Pair 1 minus pair 2, a median and plane out: $\sigma$ / robust $\sigma$ (rad)}} \\"]
        for name, label in (("uncorrected, tied", "uncorrected"), ("GUNW", "the product's screen"),
                            ("GUNW, repaired", "the product's, integers repaired"), ("ours", "ours")):
            lines.append(f"{label} & \\multicolumn{{2}}{{c}}{{{q[name]['sd']:.2f} / {q[name]['robust']:.2f}}} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "tab_antarctica.tex"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    tab_synthetic()
    tab_synthetic(ROBUST, ROBUST_METHODS, "tab_synthetic_robust")
    tab_nisar()
    tab_tie()
    tab_nearfield()
    tab_fill()
    tab_antarctica()
    tab_tie_readings()
