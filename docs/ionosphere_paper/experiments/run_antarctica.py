"""The Antarctic pairs of nisarErrors (track 94 D, frame 129): the PR GUNW's screen against ours, why they differ, and
the GUNW recipe on our cells. Section 8 of the paper.

usage: .../envs/remote_sensing/bin/python -u run_antarctica.py [P ...]        (default: 1 2)
       .../envs/remote_sensing/bin/python -u run_antarctica.py post [P ...]   (the post step alone, from the saved maps)
       .../envs/remote_sensing/bin/python -u run_antarctica.py ours [P ...]   (the ties and the trunk test again, on our
                                                                               own offsets: ant_roff.py must have run)

The pairs, 19 -> 31 August and 31 August -> 12 September 2026 (NISAR 40 + 5 MHz, HH), are processed by the pipeline
in ~/Documents/data/antarctica_t094_f129 (scripts/ant_pipeline.py; its README.md is the reference). It must have run
up to `residual` for each pair. What this script reads from it:
  - 240 m cells of both bands, the GUNW's unwrapped phase taken out of every pixel before the sums (`phase_model`,
    Section 7.3) and its cell mean kept;
  - work/screens_P{P}.nc: our screens (the main band's integers as delivered, the main component only, and the cut-off
    component re-tied by the side band) and the product's screen binned onto the cells;
  - scripts/tie_check.py: the split of the components read three ways (side band, range offsets, ITS_LIVE), and the
    offsets' ionosphere where the product and ours still differ;
  - ant_pipeline.residuals: nisarErrors' residual metric against an ITS_LIVE synthetic.

Here, on the same cells: every estimator of estimators.py that the GUNW recipe needs ('ours', 'diff_snaphu',
'diff_isce3'), the main band tied by the side band, and 'diff_isce3' again on the main band as delivered (the
product's own integers). 'ours' must reproduce the pipeline's tied screen.

Writes ../results/antarctica.json (the scalars), antarctica_windows.csv (the tie per window), antarctica_zones.csv (the
trunk test), antarctica_arms.csv (each estimator against ours and the product) and antarctica_P{P}.npz (maps).

The post step (`post`, run after each pair, or alone from the saved maps) splits what is left of the product's screen
after its integers by where it lies: within 3 km of the block's edge it is the product's filter smearing the block's
73 rad step; 5-20 km outside the block, over the tributary's trunk, it is a plateau of about two main-band cycles, which
ISCE3's correction passes put there (diff_isce3 - diff_snaphu on our cells). It adds the plateau's numbers, the leak
statistic with the cells within 5 km of either integer step left out, and the profile across the block's edge
(antarctica_profile.csv) to the results.
"""

import functools
import json
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
print = functools.partial(print, flush=True)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import estimators as E  # noqa: E402

ANT = os.path.expanduser("~/Documents/data/antarctica_t094_f129")
sys.path.insert(0, os.path.join(ANT, "scripts"))
import ant_pipeline as ap  # noqa: E402
import tie_check as tc  # noqa: E402

RESULTS = os.path.join(HERE, "..", "results")
CELL = 240.0


def cells(P):
    """Both bands' cells on the screens' grid, as stage_screen intersected them."""
    ca = ap.load_cells(ap.D / "work" / f"cells_P{P}_A.npz")
    cb = ap.load_cells(ap.D / "work" / f"cells_P{P}_B.npz")
    xs = np.intersect1d(ca.x.values, cb.x.values)
    ys = np.intersect1d(ca.y.values, cb.y.values)[::-1]
    return ca.sel(x=xs, y=ys), cb.sel(x=xs, y=ys)


def rsd(v):
    v = v[np.isfinite(v)]
    return float(1.4826 * np.median(np.abs(v - np.median(v)))) if v.size else np.nan


def med(v):
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else np.nan


def run(P):
    from scipy.ndimage import binary_dilation, distance_transform_edt
    from nisar_tools import split_spectrum as ss
    t0 = time.time()
    L = tc.load(P)
    T = tc.tie_readings(L)
    R = tc.trunk_test(L, T["comb"], T["okr"])
    Q = ap.residuals(P, g=L["g"], syn=L["syn"])
    print(f"P{P}: inputs [{time.time() - t0:.0f}s]")
    ds, dec = L["ds"], R["dec"]
    ca, cb = cells(P)
    assert np.allclose(ca.x.values, ds.x.values) and np.allclose(ca.y.values, ds.y.values)
    f_a, f_b = L["f_a"], L["f_b"]
    comp = ds.component.values
    sign, k_ours = L["sign"], L["k_ours"]
    # the cells' absolute phases: the model's cell mean, scaled to each band, plus the residual angle
    m_a = (f_a / float(ca.attrs["model_frequency"])) * ca.model.values
    m_b = (f_b / float(cb.attrs["model_frequency"])) * cb.model.values
    psi_a = ss.wrap(m_a + np.angle(ca.ifg.values))
    psi_b = ss.wrap(m_b + np.angle(cb.ifg.values))
    ga, gb = ca.coherence.values, cb.coherence.values
    na, nb = ca.n.values / float(ca.attrs["pixels_per_cell"]), cb.n.values / float(cb.attrs["pixels_per_cell"])
    g = L["g"]
    phase_all, wsum = ss.cell_average(sign * g["unw"], g["x"], g["y"], ca)
    frac_px = (CELL / abs(float(g["x"][1] - g["x"][0]))) ** 2
    valid = np.isfinite(phase_all) & (wsum / frac_px > 0.5)
    sig = E.raw_sigma(ga, gb, na, nb, f_a, f_b, CELL)
    use = valid & np.isfinite(m_a) & (ga > 0.2) & (gb > 0.2) & (na >= 0.5) & (nb >= 0.5)
    phi_tied = m_a + 2 * np.pi * k_ours * (comp == 1)
    arms = {}
    for name, method, phi in (("ours", "ours", phi_tied), ("diff_snaphu", "diff_snaphu", phi_tied),
                              ("diff_isce3", "diff_isce3", phi_tied), ("diff_isce3_delivered", "diff_isce3", m_a)):
        t1 = time.time()
        out = E.estimate(method, phi, psi_a, psi_b, ga, gb, use, sig, f_a, f_b, CELL, x=ds.x.values, y=ds.y.values,
                         data=valid & np.isfinite(phi))
        arms[name] = out
        print(f"  {name:22s} [{time.time() - t1:4.0f}s] cycles moved A {out['cycles_a']:.4f} B {out['cycles_b']:.4f}")
    ours = ds.screen_tied.values
    dv = arms["ours"]["screen"] - ours
    print(f"  ours (estimators.py) vs the pipeline's tied screen: max |diff| {np.nanmax(np.abs(dv)):.2e} rad, "
          f"{np.mean(np.isfinite(arms['ours']['screen']) == np.isfinite(ours)):.4%} same coverage")
    # zones: the product's block core, component 1 far from the boundary, the trunk zones, the rest
    ok = dec["ok"]
    left = dec["leftover"]
    blockm = dec["m_dd"] != 0
    core = blockm & ~binary_dilation(~blockm, iterations=25)
    far1 = ok & (comp == 1) & (distance_transform_edt(comp != L["main"]) * CELL > 10e3)
    zones = {"component 1": far1, "block core": core & ok, "trunks, product above": ok & (left > 3),
             "trunks, product below": ok & (left < -3), "elsewhere": ok & (np.abs(left) < 1) & ~blockm & (comp == 2)}
    rep_minus_ours = ds.screen_product.values - dec["correction"] - ours
    rows = []
    for name, out in arms.items():
        d = out["screen"] - ours
        base = med(d[zones["elsewhere"]])
        row = dict(pair=int(P), arm=name, cycles_a=out["cycles_a"], cycles_b=out["cycles_b"], baseline=base)
        for zn, zm in zones.items():
            row[f"{zn}: arm - ours"] = med(d[zm] - base)
        lo = ok & np.isfinite(d) & np.isfinite(rep_minus_ours) & ~blockm & (comp == 2) & (np.abs(left) > 1)
        if lo.sum() > 100:
            x_, y_ = rep_minus_ours[lo], d[lo] - base
            row["trunk slope on the product's leftover"] = float((x_ @ y_) / (x_ @ x_))
            row["trunk corr with the product's leftover"] = float(np.corrcoef(x_, y_)[0, 1])
        rows.append(row)
        print(f"  {name:22s} " + ", ".join(f"{k.split(':')[0]} {v:+.2f}" for k, v in row.items() if k.endswith("- ours"))
              + (f"; trunks: slope {row.get('trunk slope on the product' + chr(39) + 's leftover', np.nan):+.2f}, corr "
                 f"{row.get('trunk corr with the product' + chr(39) + 's leftover', np.nan):+.2f}"))
    # the screens' leak on the ice: 15 km high-pass of each screen on that of the non-dispersive phase, all valid cells
    phi_nd = phi_tied + ss.wrap(psi_a - phi_tied) - ours
    hp = lambda a, mk: a - tc.nangauss(a, mk & np.isfinite(a), 15e3 / CELL / 2.355)  # noqa: E731  (15 km FWHM)
    leak = {}
    for name, scr in (("ours", ours), ("product, integers repaired", ds.screen_product.values - dec["correction"]),
                      ("diff_isce3", arms["diff_isce3"]["screen"]), ("diff_snaphu", arms["diff_snaphu"]["screen"])):
        mk = ok & np.isfinite(scr) & np.isfinite(phi_nd)
        a_, b_ = hp(scr, mk), hp(phi_nd, mk)
        sel = mk & np.isfinite(a_) & np.isfinite(b_)
        leak[name] = dict(slope=float((a_[sel] @ b_[sel]) / (b_[sel] @ b_[sel])), r=float(np.corrcoef(a_[sel], b_[sel])[0, 1]))
        print(f"  leak, {name:28s} slope {leak[name]['slope']:+.4f}, r {leak[name]['r']:+.2f}")
    # scalars
    st = np.array([w["nd_step"] for w in T["windows"]])
    near = [a for a in T["areas"] if a["vmax"] == 50 and a["within"] == 20]
    S = dict(pair=int(P), dates=f"{ap.PAIRS[P]['ref_date']}-{ap.PAIRS[P]['sec_date']}", side_band_tie=k_ours, f_a=f_a, f_b=f_b,
             baseline=dec["baseline"], per_main=dec["per_main"], per_dd=dec["per_dd"],
             component1_cycles=dec["components"][1]["offset"] / dec["per_main"], component1_rsd=dec["components"][1]["rsd"],
             block_km2=dec["blocks"][0]["km2"], block_cycles=dec["blocks"][0]["core"] / dec["per_dd"],
             block_core_rsd=dec["blocks"][0]["core_rsd"], leftover_rsd=dec["leftover_rsd"],
             leftover_km2_over3=float((np.abs(left[ok]) > 3).sum() * CELL * CELL / 1e6),
             tie_side_band=med(np.array([w["side_band"] for w in T["windows"]])),
             tie_offsets=med(np.array([w["offsets"] for w in T["windows"]])),
             tie_offsets_sd=float(np.nanstd([w["offsets"] for w in T["windows"]])),
             tie_side_band_sd=float(np.std([w["side_band"] for w in T["windows"]])),
             tie_velocity_near=near[0]["deg1"] if near else np.nan, n_windows=len(T["windows"]),
             nd_step_median=float(np.nanmedian(st)), nd_step_min=float(np.nanmin(st)), nd_step_max=float(np.nanmax(st)),
             ratio_slope=R["ratio"]["slope"], ratio_r2=R["ratio"]["r2"], alpha={f["sigma_km"]: f["alpha"] for f in R["fits"]},
             residual_slope=Q["slope"], residual={k: v for k, v in Q["scores"].items()}, leak=leak,
             ours_vs_pipeline_maxdiff=float(np.nanmax(np.abs(dv))))
    pd.DataFrame(T["windows"]).to_csv(os.path.join(RESULTS, f"antarctica_windows_P{P}.csv"), index=False)
    pd.DataFrame(R["zones"]).to_csv(os.path.join(RESULTS, f"antarctica_zones_P{P}.csv"), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, f"antarctica_arms_P{P}.csv"), index=False)
    np.savez_compressed(os.path.join(RESULTS, f"antarctica_P{P}.npz"), x=ds.x.values, y=ds.y.values,
                        component=comp.astype(np.int8), ok=ok, product=ds.screen_product.values.astype(np.float32),
                        ours=ours.astype(np.float32), diff=dec["diff"].astype(np.float32),
                        leftover=left.astype(np.float32), coherence_a=ga.astype(np.float32),
                        **{f"arm_{k}": v["screen"].astype(np.float32) for k, v in arms.items()},
                        offsets_iono=R["maps"][3]["offsets"].astype(np.float32),
                        ours_3km=R["maps"][3]["ours"].astype(np.float32),
                        product_3km=R["maps"][3]["product"].astype(np.float32))
    print(f"P{P} done [{time.time() - t0:.0f}s]")
    return S


def offsets_ours(P, S):
    """The tie windows and the trunk test again, on our own offsets (tie_check.load(offsets="ours")); adds
    tie_offsets_ours (+ sd, alpha, zones) to S and writes antarctica_windows_ours_P{P}.csv / antarctica_zones_ours_P{P}.csv."""
    L = tc.load(P, offsets="ours")
    T = tc.tie_readings(L)
    R = tc.trunk_test(L, T["comb"], T["okr"])
    v = np.array([w["offsets"] for w in T["windows"]])
    S["tie_offsets_ours"] = med(v)
    S["tie_offsets_ours_sd"] = float(np.nanstd(v))
    S["tie_offsets_ours_n"] = int(np.isfinite(v).sum())
    S["alpha_ours"] = {f["sigma_km"]: f["alpha"] for f in R["fits"]}
    S["offsets_ours_spread"] = T["spread"]
    for z in R["zones"]:
        if z["sigma_km"] == 3:
            S[f"zone_ours_{z['zone']}"] = dict(off_minus_ours=z["off_minus_ours"], off_minus_product=z["off_minus_product"],
                                               km2=z["km2"])
    pd.DataFrame(T["windows"]).to_csv(os.path.join(RESULTS, f"antarctica_windows_ours_P{P}.csv"), index=False)
    pd.DataFrame(R["zones"]).to_csv(os.path.join(RESULTS, f"antarctica_zones_ours_P{P}.csv"), index=False)
    print(f"P{P} (our offsets): tie {S['tie_offsets_ours']:+.2f} (sd {S['tie_offsets_ours_sd']:.2f}, "
          f"{S['tie_offsets_ours_n']} windows); alpha {S['alpha_ours']}; zones " + "; ".join(
              f"{z['zone']} {z['off_minus_ours']:+.2f} / {z['off_minus_product']:+.2f}" for z in R["zones"] if z["sigma_km"] == 3))
    return S


EDGE_BINS = [-12, -8, -5, -3, -2, -1, 0, 1, 2, 3, 5, 8, 12, 16, 20, 25, 30, 40]


def post(P, S):
    """The leftover split by where it lies (see the module docstring); returns S with the post step's keys added."""
    from scipy.ndimage import distance_transform_edt as edt
    from nisar_tools import split_spectrum as ss
    if "f_a" not in S:
        S["f_a"], S["f_b"] = ap.band_frequencies(P)
    z = np.load(os.path.join(RESULTS, f"antarctica_P{P}.npz"))
    ok, comp = z["ok"], z["component"]
    diff, left, ours = (z[k].astype(float) for k in ("diff", "leftover", "ours"))
    per_main, per_dd = S["per_main"], S["per_dd"]
    n1 = int(round(S["component1_cycles"]))
    block = np.where(ok, np.round((diff - per_main * n1 * (comp == 1)) / per_dd), 0) != 0
    c = CELL / 1e3
    d_out, d_in, d_c1, d_c2 = edt(~block) * c, edt(block) * c, edt(comp != 1) * c, edt(comp == 1) * c
    sd = np.where(block, -d_in, d_out)                       # signed distance from the block's edge, + outside
    plateau = ok & (comp == 2) & (sd > 5) & (sd < 20) & (d_c1 > 5)
    quiet = ok & (comp == 2) & ~block & (sd > 25) & (np.abs(left) < 1)
    corr = z["arm_diff_isce3"].astype(float) - z["arm_diff_snaphu"].astype(float)   # what the passes add on our cells
    corr = corr - np.nanmedian(corr[quiet])
    S["plateau_km2"] = float(plateau.sum() * c * c)
    S["plateau_product"] = float(np.median(left[plateau]))
    S["plateau_product_cycles"] = S["plateau_product"] / per_main
    S["plateau_share_over3"] = float(np.mean(left[plateau] > 3))
    S["plateau_corrections"] = float(np.nanmedian(corr[plateau]))
    S["plateau_corrections_p90"] = float(np.nanpercentile(corr[plateau], 90))
    S["corrections_cells_moved"] = None
    arms = pd.read_csv(os.path.join(RESULTS, f"antarctica_arms_P{P}.csv")).set_index("arm")
    S["corrections_cells_moved"] = float(arms.loc["diff_isce3", "cycles_a"])
    for name in ("diff_snaphu", "diff_isce3", "diff_isce3_delivered"):
        d = z[f"arm_{name}"].astype(float) - ours
        base = np.nanmedian(d[quiet])
        S[f"arm_{name}_plateau"] = float(np.nanmedian(d[plateau] - base))
        S[f"arm_{name}_component1"] = float(np.nanmedian(d[ok & (comp == 1) & (d_c2 > 10)] - base))
        S[f"arm_{name}_block"] = float(np.nanmedian(d[block & (d_in > 6)] - base))
    smear = ok & (comp == 2) & (sd > 0) & (sd <= 3)
    S["smear_0_3km"] = float(np.median(left[smear]))
    dv = z["arm_ours"].astype(float) - ours
    S["ours_vs_pipeline_rms"] = float(np.sqrt(np.nanmean(dv[ok] ** 2)))
    S["ours_vs_pipeline_p99"] = float(np.nanpercentile(np.abs(dv[ok]), 99))
    # the leak statistic: 15 km high-pass (FWHM) of the screen on that of the non-dispersive phase, the cells within
    # 5 km of either integer step left out
    ca, _ = cells(P)
    m_a = (float(ca.attrs["center_frequency"]) / float(ca.attrs["model_frequency"])) * ca.model.values
    phi_nd = m_a + ss.wrap(np.angle(ca.ifg.values)) + 2 * np.pi * S["side_band_tie"] * (comp == 1) - ours
    away = ok & np.isfinite(phi_nd) & (np.where(block, d_in, d_out) > 5) & (np.where(comp == 1, d_c2, d_c1) > 5)
    rep = z["product"].astype(float) - (diff - left) - S["baseline"]  # the product, its integers out
    hp = lambda a, mk: a - tc.nangauss(a, mk & np.isfinite(a), 15e3 / CELL / 2.355)  # noqa: E731
    for name, scr in (("ours", ours), ("product, integers repaired", rep)):
        mk = away & np.isfinite(scr)
        a_, b_ = hp(scr, mk), hp(phi_nd, mk)
        sel = mk & np.isfinite(a_) & np.isfinite(b_)
        S.setdefault("leak_away", {})[name] = dict(slope=float((a_[sel] @ b_[sel]) / (b_[sel] @ b_[sel])),
                                                   r=float(np.corrcoef(a_[sel], b_[sel])[0, 1]))
    # the non-dispersive step across each of the block's edges (3-6 km either side), and its share in the double
    # difference (Delta f / f_A of it, in cycles): where the product's double difference had to cross more than pi
    rest = ok & (comp == 2) & ~block
    d_rest, d_c1o = edt(~rest) * c, edt(comp != 1) * c
    for name, a_side, b_side in (("block_vs_rest", block & (d_rest > 3) & (d_rest < 6), rest & (d_out > 3) & (d_out < 6)),
                                 ("component1_vs_block", ok & (comp == 1) & (d_out > 3) & (d_out < 6) & (d_rest > 10),
                                  block & (d_c1o > 3) & (d_c1o < 6)),
                                 ("component1_vs_rest", ok & (comp == 1) & (d_rest > 3) & (d_rest < 6) & (d_out > 10),
                                  rest & (d_c1o > 3) & (d_c1o < 6) & (d_out > 10))):
        ga_, gb_ = a_side & np.isfinite(phi_nd), b_side & np.isfinite(phi_nd)
        step = float(np.median(phi_nd[ga_]) - np.median(phi_nd[gb_])) if ga_.sum() > 50 and gb_.sum() > 50 else np.nan
        S[f"nd_step_{name}"] = step
        S[f"nd_step_{name}_dd_cycles"] = step * (S["f_b"] - S["f_a"]) / S["f_a"] / (2 * np.pi) if np.isfinite(step) else np.nan
    prof = []
    off_ours = z["offsets_iono"].astype(float) - z["ours_3km"].astype(float)
    off_prod = z["offsets_iono"].astype(float) - z["product_3km"].astype(float)
    for lo, hi in zip(EDGE_BINS[:-1], EDGE_BINS[1:]):
        m = ok & (comp == 2) & (sd >= lo) & (sd < hi) & (d_c1 > 5)
        if m.sum() < 50:
            continue
        prof.append(dict(pair=int(P), lo_km=lo, hi_km=hi, n=int(m.sum()), product=float(np.median(left[m])),
                         corrections=float(np.nanmedian(corr[m])), offsets_minus_ours=float(np.nanmedian(off_ours[m])),
                         offsets_minus_product=float(np.nanmedian(off_prod[m]))))
    pd.DataFrame(prof).to_csv(os.path.join(RESULTS, f"antarctica_profile_P{P}.csv"), index=False)
    print(f"P{P} post: leftover 0-3 km outside the block {S['smear_0_3km']:+.2f} rad (the filter's smear); the plateau "
          f"5-20 km out ({S['plateau_km2']:.0f} km2) {S['plateau_product']:+.2f} rad = {S['plateau_product_cycles']:.2f} "
          f"main-band cycles, the passes on our cells {S['plateau_corrections']:+.2f} (p90 "
          f"{S['plateau_corrections_p90']:+.2f}); leak away from the steps: ours {S['leak_away']['ours']['slope']:+.4f} "
          f"(r {S['leak_away']['ours']['r']:+.2f}), product {S['leak_away']['product, integers repaired']['slope']:+.4f} "
          f"(r {S['leak_away']['product, integers repaired']['r']:+.2f}); ours vs the pipeline rms "
          f"{S['ours_vs_pipeline_rms']:.4f} rad, p99 {S['ours_vs_pipeline_p99']:.4f}")
    print(f"   non-dispersive steps (rad, double-difference cycles): " + ", ".join(
        f"{n} {S[f'nd_step_{n}']:+.0f} ({S[f'nd_step_{n}_dd_cycles']:+.2f})" for n in
        ("block_vs_rest", "component1_vs_block", "component1_vs_rest")))
    return S


def quality():
    """How good each correction is, two ways that need no velocity map: each screen minus the range offsets'
    ionosphere over the frame, and pair 1 - pair 2 (the Antarctic folder's scripts/corrected_maps.py, run here if its
    work/corrected_metrics.json is missing)."""
    path = os.path.join(ANT, "work", "corrected_metrics.json")
    if not os.path.exists(path):
        import corrected_maps
        corrected_maps.main()
    return json.load(open(path))


if __name__ == "__main__":
    args = sys.argv[1:]
    mode = args[0] if args and args[0] in ("post", "ours", "quality") else "run"
    pairs = (args[1:] if mode != "run" else args) or ["1", "2"]
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "antarctica.json")
    allS = json.load(open(path)) if os.path.exists(path) else {}
    if mode == "quality":
        allS["quality"] = quality()
        json.dump(allS, open(path, "w"), indent=1, default=float)
        print(json.dumps(allS["quality"], indent=1))
        sys.exit(0)
    for P in pairs:
        if mode == "ours":
            allS[P] = offsets_ours(P, allS[P])
        else:
            allS[P] = post(P, allS[P] if mode == "post" else run(P))
        json.dump(allS, open(path, "w"), indent=1, default=float)
    for name in ("windows", "zones", "arms", "profile", "windows_ours", "zones_ours"):
        parts = [pd.read_csv(os.path.join(RESULTS, f"antarctica_{name}_P{p}.csv")) for p in ("1", "2")
                 if os.path.exists(os.path.join(RESULTS, f"antarctica_{name}_P{p}.csv"))]
        pd.concat(parts).to_csv(os.path.join(RESULTS, f"antarctica_{name}.csv"), index=False)
