"""The paper's figures, from ../results -> ../figures (PDF + PNG).

usage: .../envs/remote_sensing/bin/python figures.py [fig1 fig2 ...]     (default: every figure whose inputs exist)

fig_profiles   the example through-going rupture: what each estimator unwraps, across the trace
fig_maps       the example scenes: each estimator's screen error (rad), finite and through-going
fig_sweep      error 2-5 km from the trace and leak against peak slip, per estimator and experiment
fig_offset     band co-location: error against the side band's range offset
fig_nisar      the NISAR scenes: leak per estimator, and the GUNW's
fig_d141       Panama D141: screens of ours, the GUNW recipe on our cells, and the GUNW, high-passed across the trace
fig_d126       the same for Venezuela D126
fig_tie        the side-band tie: cross-fault profiles on two Panama pairs, fake boundaries against the real ruptures,
               and what one cycle of tie does to each production screen near the trace
fig_nearfield  near the trace: the recipe on a synthetic, the fills on real ionosphere (fake gaps), D141's cells per cell
               and per pixel, and D141's screens with the main band as cut and repaired
fig_antarctica the Antarctic pairs of nisarErrors: the product's screen against ours, the two integers between them, what
               is left on the fast ice and what the range offsets say there, the GUNW recipe on our cells, and the
               components' split read by the side band and by the offsets
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import metrics as M  # noqa: E402

RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "figures")
LOS_MM = 299792458.0 / 1239e6 / (4 * np.pi) * 1e3      # mm of LOS per rad at f_A

LABEL = {"ours": "reordered (ours)", "diff_smooth": "reformulated, same unwrapper", "diff_snaphu": "reformulated (SNAPHU)",
         "diff_trivial": "reformulated, wrapped", "diff_isce3": "reformulated + ISCE3 corrections",
         "diff_isce3_noexcl": "ISCE3 recipe, no exclusion", "side_snaphu": "two-band (SNAPHU)",
         "side_isce2": "two-band + ISCE2 correction", "side_isce3": "two-band + ISCE3 corrections", "gunw": "PR GUNW"}
COLOR = {"ours": "k", "diff_smooth": "C0", "diff_snaphu": "C1", "diff_trivial": "C8", "diff_isce3": "C3",
         "diff_isce3_noexcl": "C6", "side_snaphu": "C2", "side_isce2": "C9", "side_isce3": "C4", "gunw": "C5"}
ORDER = ["ours", "diff_smooth", "diff_snaphu", "diff_trivial", "diff_isce3", "diff_isce3_noexcl",
         "side_snaphu", "side_isce2", "side_isce3", "gunw"]


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def _example(geom):
    p = os.path.join(RES, f"synthetic_example_{geom}.npz")
    return np.load(p, allow_pickle=True) if os.path.exists(p) else None


def fig_maps():
    exs = [(g, _example(g)) for g in ("finite", "through")]
    exs = [(g, z) for g, z in exs if z is not None]
    if not exs:
        return
    methods = [m for m in ("ours", "diff_smooth", "diff_snaphu", "diff_isce3", "side_snaphu")
               if f"screen_{m}" in exs[0][1].files]
    fig, axes = plt.subplots(len(exs), len(methods) + 1, figsize=(2.35 * (len(methods) + 1) + 1.2, 2.55 * len(exs)),
                             squeeze=False, constrained_layout=True)
    for i, (g, z) in enumerate(exs):
        x, y = z["truth_x"] / 1e3, z["truth_y"] / 1e3
        ext = [x[0], x[-1], y[-1], y[0]]
        land = z["truth_land"]
        io = np.where(land, z["truth_iono"], np.nan)
        ax = axes[i, 0]
        im0 = ax.imshow(io - np.nanmedian(io), extent=ext, cmap="RdBu_r")
        ax.plot(z["trace_x"] / 1e3, z["trace_y"] / 1e3, "k-", lw=0.8)
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        ax.set_title({"finite": "rupture inside the scene", "through": "rupture through the scene"}[g]
                     + "\ntrue screen (rad)", fontsize=8)
        ax.tick_params(labelsize=7)
        fig.colorbar(im0, ax=ax, shrink=0.75).ax.tick_params(labelsize=7)
        for j, m in enumerate(methods):
            e = np.where(land, z[f"screen_{m}"] - z["truth_iono"], np.nan)
            far = land & (np.abs(z["truth_dist"]) > 20e3)
            e = e - np.nanmedian(e[far])
            ax = axes[i, j + 1]
            im = ax.imshow(e, extent=ext, cmap="PuOr_r", vmin=-3, vmax=3)
            ax.plot(z["trace_x"] / 1e3, z["trace_y"] / 1e3, "k-", lw=0.5)
            ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
            if i == 0:
                ax.set_title(LABEL[m].replace(", ", ",\n").replace(" + ", "\n+ "), fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
        cb = fig.colorbar(im, ax=axes[i, 1:].tolist(), shrink=0.85, extend="both")
        cb.set_label("screen error (rad)", fontsize=8); cb.ax.tick_params(labelsize=7)
    save(fig, "fig_maps")


def fig_profiles():
    z = _example("through")
    if z is None:
        return
    d = z["truth_dist"] / 1e3
    land = z["truth_land"]
    bins = np.arange(-20, 20.5, 0.5)
    mid = 0.5 * (bins[1:] + bins[:-1])

    def prof(a):
        return np.array([np.nanmedian(a[land & (d >= b0) & (d < b1)]) if (land & (d >= b0) & (d < b1)).sum() > 10
                         else np.nan for b0, b1 in zip(bins[:-1], bins[1:])])

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.9), constrained_layout=True)
    psi = z["truth_psi"]
    ax = axes[0]
    ax.plot(mid, 1.044 * prof(psi), "C2-", label=r"in $\phi_B$: $1.044\,\phi_{nd}$ (left)")
    ax.set(xlabel="distance from the trace (km, + north)", ylabel=r"rad, $\phi_B$",
           title="(a) the deformation in each unwrapped field")
    ax2 = ax.twinx()
    ax2.plot(mid, prof(-0.044 * psi), "C1-", label=r"in $\Delta$: $-0.044\,\phi_{nd}$ (right)")
    ax2.plot(mid, 0 * mid, "k--", label=r"in $r$: none (right)")
    ax2.set_ylabel(r"rad, $\Delta$ and $r$")
    ax2.set_ylim(-3.5, 3.5); ax.set_ylim(-3.5 / 0.044 * 1.044, 3.5 / 0.044 * 1.044)
    ax.axvspan(-2, 2, color="0.92")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper right")
    errs = {}
    for m in [m for m in ORDER if f"screen_{m}" in z.files]:
        e = np.where(land, z[f"screen_{m}"] - z["truth_iono"], np.nan)
        errs[m] = LOS_MM * prof(e - np.nanmedian(e[land & (np.abs(z["truth_dist"]) > 20e3)]))
    for ax, sel, title in ((axes[1], list(errs), "(b) every estimator (symmetric log)"),
                           (axes[2], [m for m in errs if np.nanmax(np.abs(errs[m])) < 200],
                            "(c) the ones that stay within 200 mm")):
        for m in sel:
            ax.plot(mid, errs[m], color=COLOR[m], label=LABEL[m], lw=1.6 if m == "ours" else 1.0)
        ax.axvspan(-2, 2, color="0.92"); ax.axhline(0, color="k", lw=0.4)
        ax.set(xlabel="distance from the trace (km, + north)", ylabel="screen error (mm LOS)", title=title)
        ax.legend(fontsize=6)
    axes[1].set_yscale("symlog", linthresh=10)
    save(fig, "fig_profiles")


def fig_sweep():
    p = os.path.join(RES, "synthetic.csv")
    if not os.path.exists(p):
        return
    df = pd.read_csv(p)
    exps = [e for e in ("finite", "through", "noexcl") if e in set(df.experiment)]
    fig, axes = plt.subplots(2, len(exps), figsize=(4.2 * len(exps), 6.5), squeeze=False, constrained_layout=True)
    for j, e in enumerate(exps):
        sub = df[df.experiment == e]
        for m in [m for m in ORDER if m in set(sub.method)]:
            g = sub[sub.method == m].groupby("slip")
            for i, col in enumerate(("rms_2_5", "leak")):
                mean, sd = g[col].mean(), g[col].std()
                scale = LOS_MM if col.startswith("rms") else 1.0
                axes[i, j].errorbar(mean.index, scale * mean.values, yerr=scale * sd.values, color=COLOR[m],
                                    label=LABEL[m], marker="o", ms=3, capsize=2, lw=1.4 if m == "ours" else 0.9)
        axes[0, j].set(title={"finite": "finite rupture, fault zone excluded", "through": "through-going, excluded",
                              "noexcl": "through-going, no exclusion"}[e], ylabel="error rms 2-5 km (mm LOS)")
        axes[1, j].set(xlabel="peak slip (m)", ylabel="leak (slope on deformation)")
        axes[0, j].set_yscale("symlog", linthresh=5)
        axes[1, j].set_yscale("symlog", linthresh=0.01)
        axes[1, j].axhline(0, color="k", lw=0.4)
    axes[0, 0].legend(fontsize=6)
    save(fig, "fig_sweep")


def fig_offset():
    """The error's change with the side band's range offset, paired by noise seed (the seed spread is larger than
    the effect)."""
    p = os.path.join(RES, "synthetic.csv")
    if not os.path.exists(p):
        return
    df = pd.read_csv(p)
    sub = df[df.experiment == "offset"]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(4.6, 3.2), constrained_layout=True)
    for col, style, lab in (("rms_2_5", "-", "2-5 km"), ("rms_0_2", "--", "0-2 km (filled)")):
        piv = sub[sub.method == "ours"].pivot_table(index="seed", columns="offset_b_m", values=col)
        dd = LOS_MM * (piv.sub(piv[0.0], axis=0))
        ax.errorbar(dd.columns, dd.mean(), yerr=dd.std(), fmt="o" + style, color="k", ms=3, capsize=2,
                    label=f"error rms {lab}")
    ax.set(xlabel="side-band range offset (m)", ylabel="change from no offset (mm LOS)")
    ax.axhline(0, color="k", lw=0.4)
    ax.legend(fontsize=7)
    save(fig, "fig_offset")


def fig_nisar():
    """(a) Each estimator's leak on the five NISAR scenes. (b) The leak beyond ours on the same track (ours' value
    is the metric's floor there: every non-failing arm shares it) for the GUNW recipe on our cells, the same with the
    product's own main band and with the product's side-band footprint, and the product itself."""
    scenes = [s for s in ("D141", "A061", "D126", "A162", "D054") if os.path.exists(os.path.join(RES, f"nisar_{s}.csv"))]
    if not scenes:
        return
    df = pd.concat([pd.read_csv(os.path.join(RES, f"nisar_{s}.csv")) for s in scenes])
    LABEL.update({"diff_isce3_noexcl_gunwphase": "ISCE3 recipe, the GUNW's main band",
                  "diff_isce3_noexcl_footprint": "ISCE3 recipe, the GUNW's side-band footprint",
                  "ours_gunwphase": "ours, the GUNW's main band", "ours_footprint": "ours, the GUNW's footprint"})
    COLOR.update({"diff_isce3_noexcl_gunwphase": "C8", "diff_isce3_noexcl_footprint": "C9",
                  "ours_gunwphase": "0.3", "ours_footprint": "0.7"})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.0), constrained_layout=True)
    panels = ((axes[0], ("ours", "diff_smooth", "diff_snaphu", "diff_isce3", "diff_isce3_noexcl", "gunw"), False,
               "(a) leak (15 km high-pass slope within 10 km)"),
              (axes[1], ("ours_gunwphase", "ours_footprint", "diff_isce3", "diff_isce3_noexcl",
                         "diff_isce3_noexcl_gunwphase", "diff_isce3_noexcl_footprint", "gunw"), True,
               "(b) leak beyond ours on the same track"))
    for ax, want, rel, title in panels:
        methods = [m for m in want if m in set(df.method)]
        w = 0.8 / max(len(methods), 1)
        for k, m in enumerate(methods):
            v = []
            for s in scenes:
                a = df[(df.scene == s) & (df.method == m)]["leak"].values
                o = df[(df.scene == s) & (df.method == "ours")]["leak"].values
                v.append((a[0] - (o[0] if rel else 0.0)) if a.size else np.nan)
            ax.bar(np.arange(len(scenes)) + (k - len(methods) / 2 + 0.5) * w, v, w, color=COLOR[m], label=LABEL[m])
        ax.set_xticks(np.arange(len(scenes))); ax.set_xticklabels(scenes); ax.axhline(0, color="k", lw=0.4)
        ax.set_title(title, fontsize=9)
        ax.set_yscale("symlog", linthresh=0.05)
        ax.legend(fontsize=7, ncol=2)
    save(fig, "fig_nisar")


def fig_d141():
    fig_scene("D141")


def fig_d126():
    fig_scene("D126")


def fig_scene(scene):
    p = os.path.join(RES, f"nisar_{scene}.npz")
    if not os.path.exists(p):
        return
    z = np.load(p, allow_pickle=True)
    x, y = z["x"] / 1e3, z["y"] / 1e3
    ext = [x[0], x[-1], y[-1], y[0]]
    land = z["land"]
    X, Y = np.meshgrid(z["x"], z["y"])
    show = [m for m in ("ours", "diff_snaphu", "diff_isce3_noexcl", "gunw") if f"screen_{m}" in z.files]
    fig = plt.figure(figsize=(12, 7.6), constrained_layout=True)
    gs = fig.add_gridspec(2, len(show), height_ratios=[1.15, 1.0])
    d = z["dist"] / 1e3
    bins = np.arange(-20, 20.5, 1.0)
    mid = 0.5 * (bins[1:] + bins[:-1])
    axp = fig.add_subplot(gs[1, :])
    map_axes = []
    for j, m in enumerate(show):
        s_ = np.where(land, z[f"screen_{m}"], np.nan)
        h = M.highpass(s_, X, Y, 240.0)
        ax = fig.add_subplot(gs[0, j])
        map_axes.append(ax)
        im = ax.imshow(LOS_MM * h, extent=ext, cmap="RdBu_r", vmin=-30, vmax=30)
        ax.plot(z["trace_x"] / 1e3, z["trace_y"] / 1e3, "k-", lw=0.8)
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        ax.set_title(LABEL[m], fontsize=9)
        ax.tick_params(labelsize=7)
        if j:
            ax.set_yticklabels([])
        prof = [np.nanmedian(h[land & (d >= b0) & (d < b1)]) if (land & (d >= b0) & (d < b1)).sum() > 20 else np.nan
                for b0, b1 in zip(bins[:-1], bins[1:])]
        axp.plot(mid, LOS_MM * np.asarray(prof), color=COLOR[m], label=LABEL[m], lw=1.6 if m == "ours" else 1.1)
    cb = fig.colorbar(im, ax=map_axes, shrink=0.85, extend="both")
    cb.set_label("15 km high-pass of the screen (mm LOS)", fontsize=8)
    hn = M.highpass(np.where(land, z["phi_nd"], np.nan), X, Y, 240.0)
    prof = [np.nanmedian(hn[land & (d >= b0) & (d < b1)]) if (land & (d >= b0) & (d < b1)).sum() > 20 else np.nan
            for b0, b1 in zip(bins[:-1], bins[1:])]
    axp.plot(mid, 0.05 * LOS_MM * np.asarray(prof), "k:", label="data (non-dispersive) x 0.05")
    axp.axvspan(-2, 2, color="0.92"); axp.axhline(0, color="k", lw=0.4)
    axp.set(xlabel="distance from the trace (km, + north)", ylabel="median high-passed screen (mm LOS)")
    axp.legend(fontsize=8, ncol=3)
    save(fig, f"fig_{scene.lower()}")


def fig_tie():
    z = np.load(os.path.join(RES, "tie_profiles.npz"))
    null = pd.read_csv(os.path.join(RES, "tie_null.csv"))
    win = pd.read_csv(os.path.join(RES, "tie_real_windows.csv"))
    eff = pd.read_csv(os.path.join(RES, "tie_screen_effect.csv"))
    step = float(z["step"])
    c = 0.5 * (z["edges"][:-1] + z["edges"][1:])
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 8.2), constrained_layout=True)
    for a, track, seg, letter in ((ax[0, 0], "A076", 5, "a"), (ax[0, 1], "D141", 2, "b")):
        prof = z[f"{track}_d"]
        k, e = z[f"{track}_tie"]
        north, south = c > 2.5, c < -2.5        # the half-bin at the excluded zone's edge holds a few cells only
        p = prof[seg]
        a.plot(c[north], p[north], ".-", color="k", ms=3, lw=0.9, label="north block")
        for cyc, col in ((0, "0.6"), (1, "C3"), (2, "C0")):
            a.plot(c[south], p[south] + cyc * step, ".-", color=col, ms=3, lw=0.9,
                   label=f"south block{'' if cyc == 0 else f' + {cyc} cycle' + ('s' if cyc > 1 else '')}"
                         + (" (offsets' tie)" if cyc == 0 else ""))
        a.axvspan(-2, 2, color="0.92", zorder=0)
        a.axvline(0, color="k", lw=0.5)
        a.set(xlabel="distance from the trace (km, + north)", ylabel="raw dispersive estimate (rad at $f_A$)")
        a.text(0.02, 0.97, f"({letter}) {track}, {10 * seg:.0f}--{10 * seg + 10:.0f} km along the trace",
               transform=a.transAxes, va="top", fontsize=9)
        a.legend(fontsize=7, loc="upper right" if track == "D141" else "lower right", framealpha=0.9)
        a.grid(alpha=0.3)
    a = ax[1, 0]
    g = null[null.config == 1].e.to_numpy()
    bins = np.arange(-2.0, 2.01, 0.05)
    a.hist(g, bins=bins, color="0.7", label=f"fake boundaries: {len(g)} windows, 8 scenes")
    rows = [("D141 v8", "C0", "D141"), ("D040 v8", "C1", "D040"), ("A076 v8", "C3", "A076"),
            ("A162 v6 (control)", "C2", "A162, no cut"), ("D054 v6 (control)", "C4", "D054, no cut")]
    for i, (scene, col, lab) in enumerate(rows):
        w = win[(win.scene == scene) & (win.field == "dispersive") & (win.estimator == "polynomial")]
        a.plot(w.e, np.full(len(w), 6000.0 / 2.2 ** i), "v", color=col, ms=7, mec="k", mew=0.4,
               label=f"{lab}: the rupture's windows")
    a.set_yscale("log")
    a.set(xlabel="$e$, cycles to add (per 20 km window)", ylabel="windows", xlim=(-2, 2.6), ylim=(0.7, 2e4))
    a.text(0.02, 0.97, "(c)", transform=a.transAxes, va="top", fontsize=9)
    a.legend(fontsize=7, loc="upper left", bbox_to_anchor=(0.0, 0.92), framealpha=0.9)
    a.grid(alpha=0.3)
    a = ax[1, 1]
    for scene, col in (("D141 v8", "C0"), ("D040 v8", "C1"), ("A076 v8", "C3")):
        e = eff[eff.scene == scene]
        x = 0.5 * (e.lo_km + e.hi_km)
        a.plot(x, -e.mm, "o-", color=col, ms=3, label=scene.replace(" v8", ""))   # the CSV holds the phase's change
    a.axhline(0, color="k", lw=0.5)
    a.axvline(0, color="k", lw=0.5)
    a.set(xlabel="distance from the trace (km, + north)", ylabel="corrected LOS change, + toward the satellite (mm)")
    a.text(0.02, 0.97, "(d)", transform=a.transAxes, va="top", fontsize=9)
    a.legend(fontsize=8, loc="lower left")
    a.grid(alpha=0.3)
    save(fig, "fig_tie")


def _profile(field, dist, edges):
    c = 0.5 * (edges[:-1] + edges[1:])
    out = np.full(c.size, np.nan)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = np.isfinite(field) & (dist >= lo * 1e3) & (dist < hi * 1e3)
        if m.sum() > 20:
            out[i] = np.median(field[m])
    return c, out


def fig_nearfield():
    """Near the trace: the recipe on a synthetic, the fills on real ionosphere, and D141's cells and integers."""
    fig, ax = plt.subplots(2, 3, figsize=(15.5, 8.6), constrained_layout=True)
    # (a) the synthetic example: signed error across the trace, one change at a time
    a = ax[0, 0]
    z = np.load(os.path.join(RES, "pixel_example.npz"))
    d, io = z["dist"], z["truth_iono"]
    far = np.abs(d) > 20e3
    edges = np.arange(-12, 12.01, 0.5)
    for key, col, lab in (("cell_2000_v6", "0.4", "production (v6)"), ("cell_2000_matched", "C1", "+ fill matched to the gap"),
                          ("cell_2000_matched1", "C0", "+ first-order filter"),
                          ("pixel_0_matched1", "C3", "+ per pixel, no exclusion")):
        e = z[f"{key}_screen"] - io
        e = e - np.median(e[far])
        c, p = _profile(e * LOS_MM, d, edges)
        a.plot(c, p, ".-", color=col, ms=3, lw=1, label=lab)
    a.axvspan(-2, 2, color="0.92", zorder=0)
    a.axhline(0, color="k", lw=0.5)
    a.set(xlabel="distance from the trace (km, + north)", ylabel="screen error (mm of LOS)")
    a.text(0.02, 0.97, "(a) synthetic, 4 m, D141-like near field", transform=a.transAxes, va="top", fontsize=9)
    a.legend(fontsize=7, loc="lower right")
    a.grid(alpha=0.3)
    # (b, c) the fills on real ionosphere: fake gaps in eight scenes
    fn = pd.read_csv(os.path.join(RES, "fill_null.csv"))
    fills = (("plane 10", "0.4", "plane 10 km (production)"), ("plane 2.6", "C2", "plane 2.6 km"),
             ("ladder", "C1", "ladder 1.5-3-6-12 km"), ("ladder 2.6", "C0", "ladder 2.6-5-10 km"), ("guide", "C4", "ISCE3's guide"))
    a = ax[0, 1]
    for f, col, lab in fills:
        y = []
        for gname in ("fault 1", "fault 2", "fault 4"):
            g = fn[(fn["gap"] == gname) & (fn["fill"] == f)]
            y.append(float(np.sqrt(np.nanmean(g["in_all"] ** 2))) * LOS_MM if len(g) else np.nan)
        a.plot([1, 2, 4], y, "o-", color=col, ms=4, lw=1, label=lab)
    a.set(xlabel="fault zone half-width (km)", ylabel="rms error inside the zone (mm)", xticks=[1, 2, 4])
    a.text(0.02, 0.97, f"(b) real ionosphere: fake fault zones ({fn.groupby(['scene', 'window']).ngroups} windows)",
           transform=a.transAxes, va="top", fontsize=9)
    a.legend(fontsize=7, loc="lower right")
    a.grid(alpha=0.3)
    a = ax[0, 2]
    bins = ((0, 1), (1, 2), (2, 4), (4, 8), (8, 12))
    for f, col, lab in fills:
        g = fn[(fn["gap"] == "edge") & (fn["fill"] == f)]
        y = [float(np.sqrt(np.nanmean(g[f"in_{lo}_{hi}"] ** 2))) * LOS_MM for lo, hi in bins]
        a.plot([0.5 * (lo + hi) for lo, hi in bins], y, "o-", color=col, ms=4, lw=1, label=lab)
    a.set_yscale("log")
    a.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    a.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a.set_yticks([2, 3, 5, 10, 20, 30, 50])
    a.set(xlabel="distance into a one-sided strip (km)", ylabel="rms error (mm)")
    a.text(0.02, 0.97, "(c) real ionosphere: a one-sided edge strip", transform=a.transAxes, va="top", fontsize=9)
    a.legend(fontsize=7, loc="lower right")
    a.grid(alpha=0.3, which="both")
    # (d) D141: the cells kept, per cell and per pixel
    a = ax[1, 0]
    nd = pd.read_csv(os.path.join(RES, "nisar_pixel_D141.csv"))
    nd = nd[(nd.exclude_m == 0.0) & (nd["filter"] == "v6")]
    B = ((0, .5), (.5, 1), (1, 2), (2, 5), (5, 10), (10, 20))
    xs = np.arange(len(B))
    for arm, off, col in (("cell", -0.2, "0.5"), ("pixel", 0.2, "C0")):
        r = nd[nd.arm == arm].iloc[0]
        a.bar(xs + off - 0.09, [100 * r[f"used_N_{lo}_{hi}"] for lo, hi in B], 0.18, color=col, label=f"{arm}, north")
        a.bar(xs + off + 0.09, [100 * r[f"used_S_{lo}_{hi}"] for lo, hi in B], 0.18, color=col, alpha=0.45,
              label=f"{arm}, south")
    a.set_xticks(xs, [f"{lo:g}-{hi:g}" for lo, hi in B])
    a.set(xlabel="distance from the trace (km)", ylabel="cells with coherence > 0.2 in both bands (%)")
    a.text(0.02, 0.97, "(d) D141: the cells, per cell and per pixel", transform=a.transAxes, va="top", fontsize=9)
    a.legend(fontsize=7, loc="lower right", ncol=2)
    a.grid(alpha=0.3, axis="y")
    # (e, f) D141: the recipes against production, the main band as cut and repaired
    di = pd.read_csv(os.path.join(RES, "d141_integers.csv"))
    recipes = (("production, first order", "C5"), ("plane 2.6", "C2"), ("ladder 1.5", "C1"), ("ladder 2.6, first order", "C0"))
    for a, integ, letter in ((ax[1, 1], "as cut", "e"), (ax[1, 2], "repaired", "f")):
        for rname, col in (("production", "0.4"),) + recipes:
            g = di[(di.integers == integ) & (di.recipe == rname)]
            a.plot(0.5 * (g.lo_km + g.hi_km), g.mm, "o-", color=col, ms=3, lw=1, label=rname)
        a.axvspan(-2, 2, color="0.92", zorder=0)
        a.axhline(0, color="k", lw=0.5)
        a.set(xlabel="distance from the trace (km, + north)", ylabel="minus the production screen as cut (mm)",
              ylim=(-8, 18))
        a.text(0.02, 0.97, f"({letter}) D141, " + ("main band as cut" if integ == "as cut" else "local cycles repaired"),
               transform=a.transAxes, va="top", fontsize=9)
        if integ == "as cut":
            a.legend(fontsize=7, loc="upper right")
        a.grid(alpha=0.3)
    save(fig, "fig_nearfield")


def fig_antarctica():
    import json
    path = os.path.join(RES, "antarctica_P1.npz")
    if not os.path.exists(path):
        return
    z = np.load(path)
    S = json.load(open(os.path.join(RES, "antarctica.json")))
    win = pd.read_csv(os.path.join(RES, "antarctica_windows.csv"))
    prof = pd.read_csv(os.path.join(RES, "antarctica_profile.csv"))
    x, y = z["x"] / 1e3, z["y"] / 1e3
    ext = (x.min() - 0.12, x.max() + 0.12, y.min() - 0.12, y.max() + 0.12)
    ok, comp = z["ok"], z["component"]
    on = lambda a: np.where(ok, a, np.nan)  # noqa: E731
    base = S["1"]["baseline"]
    lo, hi = np.nanpercentile(np.concatenate([on(z["product"]).ravel(), on(z["ours"]).ravel()]), [1, 99])
    n1 = -S["1"]["side_band_tie"]
    passes = z["arm_diff_isce3"].astype(float) - z["arm_diff_snaphu"].astype(float)
    quiet = ok & (np.abs(z["leftover"]) < 1) & (comp == 2)
    panels = [("the PR GUNW's screen", on(z["product"] - base), "RdBu_r", (lo - base, hi - base)),
              ("ours (component 1 re-tied)", on(z["ours"]), "RdBu_r", (lo - base, hi - base)),
              (f"product $-$ ours: 0, {n1}$\\times$3.21, 72.96", on(z["diff"]), "turbo", (-10, 85)),
              ("product $-$ ours $-$ those integers", on(z["leftover"]), "RdBu_r", (-8, 8)),
              ("offsets' ionosphere $-$ product", on(z["offsets_iono"] - z["product_3km"]), "RdBu_r", (-8, 8)),
              ("ISCE3's passes on our cells", on(passes - np.nanmedian(passes[quiet])), "RdBu_r", (-8, 8))]
    fig = plt.figure(figsize=(13, 10.6), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.05])
    edge = comp == 1
    for i, (title, val, cmap, lim) in enumerate(panels):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        im = ax.imshow(val, extent=ext, origin="upper", cmap=cmap, vmin=lim[0], vmax=lim[1], interpolation="nearest")
        ax.contour(x, y, edge.astype(float), levels=[0.5], colors="k", linewidths=0.6)
        ax.set_aspect("equal")
        ax.set_xlim(x.min() - 3, x.max() + 3)
        ax.set_ylim(y.min() - 3, y.max() + 3)
        ax.tick_params(labelsize=10)
        ax.text(0.02, 0.98, f"({chr(97 + i)})", transform=ax.transAxes, va="top", fontsize=13, fontweight="bold")
        ax.set_title(title, fontsize=12)
        cb = fig.colorbar(im, ax=ax, shrink=0.72, pad=0.02)
        cb.ax.tick_params(labelsize=10)
        cb.set_label("rad", fontsize=10)
        if i % 3 == 0:
            ax.set_ylabel("northing (km, EPSG:3031)", fontsize=11)
        if i >= 3:
            ax.set_xlabel("easting (km)", fontsize=11)
    ax = fig.add_subplot(gs[2, :2])
    for P, ls in (("1", "-"), ("2", "--")):
        q = prof[prof.pair == int(P)]
        xc = 0.5 * (q.lo_km + q.hi_km)
        for col, key, lab in (("C3", "product", "product $-$ ours $-$ integers"),
                              ("C1", "corrections", "ISCE3's passes on our cells"),
                              ("k", "offsets_minus_ours", "offsets' ionosphere $-$ ours")):
            ax.plot(xc, q[key], ls, color=col, marker="o", ms=4, lw=1.4, label=lab if P == "1" else None)
    ax.axvline(0, color="0.5", lw=0.8)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.axhline(2 * S["1"]["per_main"], color="0.5", lw=0.8, ls=":")
    ax.set_xlabel("distance from the block's edge (km, + outside it)", fontsize=11)
    ax.set_ylabel("rad, median over the band", fontsize=11)
    ax.set(ylim=(-6, 32), xlim=(-12, 40))
    ax.text(0.01, 0.98, "(g)", transform=ax.transAxes, va="top", fontsize=13, fontweight="bold")
    ax.text(0.99, 0.03, "solid: pair 1; dashed: pair 2; dotted: two main-band cycles", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9.5)
    ax.legend(fontsize=10, loc="upper right", framealpha=0.9)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=10)
    ax = fig.add_subplot(gs[2, 2])
    path_ours = os.path.join(RES, "antarctica_windows_ours.csv")
    win_ours = pd.read_csv(path_ours) if os.path.exists(path_ours) else None
    for P, col in (("1", "C0"), ("2", "C3")):
        w = win[win.pair == int(P)].sort_values("x_km")
        ax.errorbar(w.x_km, w.side_band, yerr=w.side_band_se, fmt="o", color=col, ms=5, label=f"pair {P}: side band")
        ax.plot(w.x_km, w.offsets, "s", mfc="none", mec=col, ms=7, label=f"pair {P}: GUNW's offsets")
        if win_ours is not None:
            wo = win_ours[win_ours.pair == int(P)].sort_values("x_km")
            ax.plot(wo.x_km, wo.offsets, "^", mfc="none", mec=col, ms=6, label=f"pair {P}: our offsets")
    ax.axhline(7, color="k", ls="--", lw=1, label="GrIMP's fix (both)")
    v = S["1"].get("tie_velocity_near")
    if v is not None and np.isfinite(v):
        ax.axhline(v, color="C0", ls=":", lw=1.2, label="pair 1: ITS_LIVE")
    ax.set_xlabel("window easting (km)", fontsize=11)
    ax.set_ylabel("split of component 1 (main-band cycles)", fontsize=11)
    ax.set(ylim=(6.0, 9.8))
    ax.text(0.03, 0.98, "(h)", transform=ax.transAxes, va="top", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="lower center", framealpha=0.9, ncol=2)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=10)
    save(fig, "fig_antarctica")


FIGS = {f.__name__: f for f in (fig_profiles, fig_maps, fig_sweep, fig_offset, fig_nisar, fig_d141, fig_d126,
                                  fig_tie, fig_nearfield, fig_antarctica)}

if __name__ == "__main__":
    for name in (sys.argv[1:] or FIGS):
        FIGS[name]()
