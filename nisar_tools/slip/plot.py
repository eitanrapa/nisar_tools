"""Figures for a slip inversion: the model, the fit, the sampling, the L-curve.

Slip on a **vertical** fault is best read on an *unrolled* section -- along-strike
distance against depth -- rather than in 3-D. The unrolling is exact, since the
fault is parameterized by exactly those two coordinates, and it wastes no space
on a projection that would show a nearly edge-on plane. A dipping fault will want
a 3-D view; the ``(s, z)`` panel stays correct for it too, just less complete.

Every function returns ``(fig, ax)`` and draws nothing on import, matching
:mod:`nisar_tools.plot`.
"""

import numpy as np


def plot_slip(model, component="magnitude", ax=None, cmap="magma_r",
              vmin=None, vmax=None, title=None):
    """Slip on the unrolled fault: along-strike distance against depth.

    ``component`` is ``"magnitude"``, ``"strike"`` or ``"dip"``. Elements are
    drawn as true triangles, so the mesh itself is visible and a bad element
    cannot hide behind interpolation.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    mesh = model.mesh
    values = {
        "magnitude": model.slip_magnitude,
        "strike": model.strike_slip,
        "dip": model.dip_slip,
    }
    if component not in values:
        raise ValueError(f"component must be one of {sorted(values)}")
    value = values[component]

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 3.2))

    verts = [mesh.params[tri][:, :2] / 1e3 for tri in mesh.triangles]
    verts = [np.column_stack([v[:, 0], v[:, 1]]) for v in verts]
    coll = PolyCollection(verts, array=value, cmap=cmap, edgecolors="none")
    if component != "magnitude":
        # A signed field wants a symmetric diverging scale about zero, or the
        # colour of "no slip" drifts with the data range.
        limit = np.abs(value).max() or 1.0
        coll.set_cmap("RdBu_r")
        coll.set_clim(-limit if vmin is None else vmin, limit if vmax is None else vmax)
    else:
        coll.set_clim(vmin, vmax)
    ax.add_collection(coll)

    ax.set_xlim(mesh.params[:, 0].min() / 1e3, mesh.params[:, 0].max() / 1e3)
    ax.set_ylim(mesh.params[:, 1].min() / 1e3, 0.0)
    ax.set_xlabel("Distance along strike (km)")
    ax.set_ylabel("Depth (km)")
    ax.set_aspect("equal")
    if title is None:
        title = (f"{component} slip -- VR {model.variance_reduction:.1f}%, "
                 f"Mw {model.moment_magnitude:.2f}, max {model.max_slip:.2f} m")
    ax.set_title(title)
    plt.colorbar(coll, ax=ax, label="Slip (m)", pad=0.02)
    return ax.figure, ax


def plot_fit(model, track=None, cmap="RdBu_r", trace=None):
    """Data, model and residual for one track, as three map panels.

    Sharing one colour scale between data and model is the point -- an
    independently scaled model panel can make a poor fit look convincing. The
    residual gets its own, tighter, scale.
    """
    import matplotlib.pyplot as plt

    obs = model.obs
    track = track or obs.tracks[0]
    sel = obs.track_mask(track)
    x = obs.ds["x"].values[sel] / 1e3
    y = obs.ds["y"].values[sel] / 1e3
    panels = [
        ("Data", model.data[sel]),
        ("Model", model.prediction[sel]),
        ("Residual", model.residual[sel]),
    ]

    limit = np.abs(np.concatenate([panels[0][1], panels[1][1]])).max() or 1.0
    res_limit = np.abs(panels[2][1]).max() or 1.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharex=True, sharey=True)
    for ax, (name, value) in zip(axes, panels):
        span = res_limit if name == "Residual" else limit
        sc = ax.scatter(x, y, c=value, s=9, cmap=cmap, vmin=-span, vmax=span)
        if trace is not None:
            tx, ty = trace.to_local(obs.frame)
            ax.plot(tx / 1e3, ty / 1e3, "k-", lw=1.2)
        ax.set_aspect("equal")
        ax.set_xlabel("East (km)")
        ax.set_title(name)
        plt.colorbar(sc, ax=ax, label="LOS (m)", pad=0.02)
    axes[0].set_ylabel("North (km)")
    fig.suptitle(f"{track}: RMS residual {np.sqrt(np.mean(panels[2][1] ** 2)):.4f} m")
    fig.tight_layout()
    return fig, axes


def plot_samples(obs, trace=None, ax=None, cmap="RdBu_r"):
    """Quadtree samples coloured by displacement and sized by cell.

    Cell size is the diagnostic: it should shrink toward the fault. If it does
    not, the roughness threshold is above the signal and the sampler is behaving
    like a fixed stride.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6))
    x = obs.ds["x"].values / 1e3
    y = obs.ds["y"].values / 1e3
    value = obs.ds["los"].values
    limit = np.abs(value).max() or 1.0
    size = 4.0 + 40.0 * obs.ds["cell_size"].values / obs.ds["cell_size"].values.max()

    sc = ax.scatter(x, y, c=value, s=size, cmap=cmap, vmin=-limit, vmax=limit,
                    edgecolors="none")
    if trace is not None:
        tx, ty = trace.to_local(obs.frame)
        ax.plot(tx / 1e3, ty / 1e3, "k-", lw=1.5)
    ax.set_aspect("equal")
    ax.set_xlabel("East (km)")
    ax.set_ylabel("North (km)")
    ax.set_title(f"{obs.n} samples from {', '.join(obs.tracks)} (marker size = cell)")
    plt.colorbar(sc, ax=ax, label="LOS (m)", pad=0.02)
    return ax.figure, ax


def plot_l_curve(curve, ax=None):
    """Model roughness against misfit, one point per smoothing weight.

    The corner is the conventional choice: to its left the fit stops improving
    and the model only gets rougher. Deliberately left to the eye -- automatic
    corner detection on a short discrete sweep is unreliable.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 5))
    rough = curve["roughness"].values
    misfit = curve["rms_misfit"].values
    ax.plot(rough, misfit, "o-", color="0.3")
    for r, m, lam, ok in zip(rough, misfit, curve["smoothing"].values,
                             curve["converged"].values):
        ax.annotate(f"{lam:g}" + ("" if ok else " (!)"), (r, m),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("Model roughness")
    ax.set_ylabel("RMS misfit (m)")
    ax.set_title("L-curve (labels are smoothing weights; ! = unconverged)")
    ax.grid(alpha=0.3)
    return ax.figure, ax


def plot_coverage(report, ax=None, name=None):
    """Valid area on each side of the trace, against along-strike distance.

    A scalar coverage fraction cannot express the failure this is for. On the
    Venezuela scenes "19% of samples north of the trace" reads like thin
    two-sided coverage; the profile showed the north block was missing along the
    eastern *two-thirds* -- precisely where the largest signal was -- which is a
    resolution limit no sampling parameter can repair. Shaded spans mark the
    stretches with data on one side only.

    Takes the :class:`xarray.Dataset` from
    :func:`~nisar_tools.slip.diagnostics.scene_report`.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3.4))

    s = report["along"].values / 1e3
    left = report["valid_left"].values / 1e6
    right = report["valid_right"].values / 1e6

    ax.fill_between(s, 0, left, color="#3b6ea5", alpha=0.75, label="left of trace")
    ax.fill_between(s, 0, -right, color="#a5553b", alpha=0.75, label="right of trace")
    ax.axhline(0.0, color="0.2", lw=1)

    one_sided = ~report["two_sided"].values
    if one_sided.any() and s.size > 1:
        step = float(np.diff(s).mean())
        for centre in s[one_sided]:
            ax.axvspan(centre - step / 2, centre + step / 2,
                       color="0.6", alpha=0.25, lw=0)

    ax.set_xlabel("Along-strike distance (km)")
    ax.set_ylabel("Valid area (km²)")
    title = "Coverage either side of the trace (grey = one side only)"
    ax.set_title(title if name is None else f"{name}: {title}")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    return ax.figure, ax
