"""Figures for a slip inversion: the model, the fit, the sampling, the L-curve.

Slip on a **vertical** fault is best read on an *unrolled* section -- along-strike
distance against depth -- rather than in 3-D. The unrolling is exact, since the
fault is parameterized by exactly those two coordinates, and it wastes no space
on a projection that would show a nearly edge-on plane. That panel stays correct
for a dipping fault too, but it projects the dip away by construction, so
:func:`plot_slip_3d` is the view that shows where the fault actually goes.

Every function returns ``(fig, ax)`` and draws nothing on import, matching
:mod:`nisar_tools.plot`.
"""

import numpy as np


def _slip_component(model, component):
    """The per-element field named by ``component``.

    Shared by the 2-D and 3-D views so the two cannot drift apart -- they are
    meant to be compared by eye, which only works if "strike slip" means the
    same array and the same colour scale in both.
    """
    element = model.element_slip
    values = {
        "magnitude": np.hypot(*element.T),
        "strike": element[:, 0],
        "dip": element[:, 1],
    }
    if component not in values:
        raise ValueError(f"component must be one of {sorted(values)}")
    return values[component]


def _scale_slip(coll, value, component, vmin, vmax):
    """Symmetric diverging scale for a signed component, plain range otherwise.

    A signed field needs zero pinned to the middle of the colour map, or the
    colour of "no slip" drifts with the data range and two models stop being
    comparable.
    """
    if component != "magnitude":
        limit = np.abs(value).max() or 1.0
        coll.set_cmap("RdBu_r")
        coll.set_clim(-limit if vmin is None else vmin, limit if vmax is None else vmax)
    else:
        coll.set_clim(vmin, vmax)


def _slip_title(model, component):
    return (f"{component} slip -- VR {model.variance_reduction:.1f}%, "
            f"Mw {model.moment_magnitude:.2f}, max {model.max_slip:.2f} m")


def plot_slip(model, component="magnitude", ax=None, cmap="magma_r",
              vmin=None, vmax=None, title=None):
    """Slip on the unrolled fault: along-strike distance against depth.

    ``component`` is ``"magnitude"``, ``"strike"`` or ``"dip"``. Elements are
    drawn as true triangles, so the mesh itself is visible and a bad element
    cannot hide behind interpolation.

    Slip is drawn **per element** whatever the model's basis. For a nodal model
    that means each triangle is shaded by the mean of its three nodes, which
    understates the peak of a continuous field slightly -- but drawing the
    parameters themselves would put a nodal model on a different geometry from an
    element one and make the two impossible to compare by eye, which is the main
    thing this plot is for.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    mesh = model.mesh
    value = _slip_component(model, component)

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 3.2))

    coll = PolyCollection(mesh.param_vertices / 1e3, array=value, cmap=cmap,
                          edgecolors="none")
    _scale_slip(coll, value, component, vmin, vmax)
    ax.add_collection(coll)

    ax.set_xlim(mesh.params[:, 0].min() / 1e3, mesh.params[:, 0].max() / 1e3)
    ax.set_ylim(mesh.params[:, 1].min() / 1e3, 0.0)
    ax.set_xlabel("Distance along strike (km)")
    ax.set_ylabel("Depth (km)")
    ax.set_aspect("equal")
    ax.set_title(_slip_title(model, component) if title is None else title)
    plt.colorbar(coll, ax=ax, label="Slip (m)", pad=0.02)
    return ax.figure, ax


def plot_slip_3d(model, component="magnitude", ax=None, cmap="magma_r",
                 vmin=None, vmax=None, title=None, exaggeration=4.0,
                 view=(22.0, -70.0), trace=None, edgecolors="0.4"):
    """Slip on the fault surface in three dimensions, in local-frame kilometres.

    What :func:`plot_slip` cannot show: the unrolled section is parameterized by
    arc length and depth, so it projects the dip away by construction. On a
    curved or dipping mesh -- especially one built from a bottom trace, where the
    dip varies along strike and may reverse -- this is the view that shows where
    the fault actually goes.

    Slip is drawn **per element**, on the same colour scale
    :func:`plot_slip` uses, so the two figures can be read against each other.

    ``exaggeration`` stretches every axis except the longest one. A real fault is
    far longer along strike than it is deep -- the San Sebastian mesh is
    264 x 25 x 40 km -- so at true scale (``exaggeration=1``) it renders as an
    unreadable sliver. Because the *two* short axes are stretched together, the
    apparent dip is preserved whenever the fault strikes near a grid axis, which
    is the case this default is chosen for; on a fault striking diagonally the
    across-strike direction has components on both stretched and unstretched
    axes, and the dip is then distorted like any vertically exaggerated section.

    ``view`` is ``(elevation, azimuth)`` in degrees. The default looks along
    strike from slightly above; ``(60, -90)`` looks down on the surface, which is
    where a dip reversal is easiest to see.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    mesh = model.mesh
    value = _slip_component(model, component)

    if ax is None:
        ax = plt.figure(figsize=(13, 6)).add_subplot(projection="3d")
    elif not hasattr(ax, "add_collection3d"):
        raise TypeError(
            "plot_slip_3d needs a 3-D axes; make one with "
            "fig.add_subplot(projection='3d')"
        )

    coll = Poly3DCollection(mesh.vertices / 1e3, cmap=cmap,
                            edgecolors=edgecolors, linewidths=0.15)
    # Poly3DCollection ignores `array=` in the constructor, unlike PolyCollection.
    coll.set_array(value)
    _scale_slip(coll, value, component, vmin, vmax)
    ax.add_collection3d(coll)

    nodes = mesh.nodes / 1e3
    ax.set_xlim(nodes[:, 0].min(), nodes[:, 0].max())
    ax.set_ylim(nodes[:, 1].min(), nodes[:, 1].max())
    ax.set_zlim(nodes[:, 2].min(), nodes[:, 2].max())

    span = nodes.max(axis=0) - nodes.min(axis=0)
    stretch = np.full(3, float(exaggeration))
    stretch[int(np.argmax(span))] = 1.0
    # Guard a degenerate axis (a single depth level, say) so the box stays valid.
    ax.set_box_aspect(np.maximum(span / span.max() * stretch, 1e-3))

    if trace is not None and mesh.frame is not None:
        tx, ty = trace.to_local(mesh.frame)
        ax.plot(tx / 1e3, ty / 1e3, np.zeros_like(tx), "r-", lw=1.4, zorder=5)

    ax.view_init(elev=view[0], azim=view[1])
    ax.set_xlabel("East (km)", labelpad=18)
    ax.set_ylabel("North (km)", labelpad=8)
    ax.set_zlabel("Depth (km)", labelpad=2)
    ax.tick_params(labelsize=7, pad=1)
    ax.set_title(_slip_title(model, component) if title is None else title)
    # A roomier pad than the flat panels use: a 3-D axes reserves no space for
    # its z label, so a tight colorbar lands on top of "Depth (km)".
    ax.figure.colorbar(coll, ax=ax, label="Slip (m)", shrink=0.6, pad=0.10)
    return ax.figure, ax


def plot_mesh(mesh, trace=None, color="area", axes=None, cmap="viridis"):
    """The fault discretisation itself: map view beside the unrolled section.

    Takes a bare :class:`~nisar_tools.slip.mesh.FaultMesh`, no solved model
    needed -- this is the figure to look at *before* committing to an inversion,
    which is when the mesh can still be changed.

    ``color`` is ``"area"`` (the default), ``"depth"`` or ``"dip"``. Element area
    is the useful one: it is what a graded mesh (``bias_w``) is changing, and it
    should be read against the sample spacing from
    :func:`~nisar_tools.slip.plot.plot_samples` -- elements much smaller than the
    data that constrain them are what the smoothing weight then has to paper over.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    fields = {
        "area": (mesh.areas / 1e6, "Element area (km²)"),
        "depth": (-mesh.centroids[:, 2] / 1e3, "Centroid depth (km)"),
        "dip": (mesh.dip, "Dip (deg)"),
    }
    if color not in fields:
        raise ValueError(f"color must be one of {sorted(fields)}")
    value, label = fields[color]

    if axes is None:
        _, axes = plt.subplots(2, 1, figsize=(12, 7.5),
                               gridspec_kw={"height_ratios": [2, 1]})
    map_ax, sec_ax = axes

    map_ax.add_collection(PolyCollection(
        mesh.vertices[:, :, :2] / 1e3, array=value, cmap=cmap,
        edgecolors="0.35", linewidths=0.2,
    ))
    if trace is not None and mesh.frame is not None:
        tx, ty = trace.to_local(mesh.frame)
        map_ax.plot(tx / 1e3, ty / 1e3, "r-", lw=1.4, zorder=3)
    map_ax.autoscale_view()
    map_ax.set_aspect("equal")
    map_ax.set_xlabel("East (km)")
    map_ax.set_ylabel("North (km)")
    map_ax.set_title(
        f"{mesh.n_elements} elements, {mesh.n_nodes} nodes "
        f"({mesh.attrs.get('kind', '?')}, edge {mesh.attrs.get('edge_length', 0) / 1e3:g} km)"
    )

    sec = PolyCollection(mesh.param_vertices / 1e3, array=value, cmap=cmap,
                         edgecolors="0.35", linewidths=0.2)
    sec_ax.add_collection(sec)
    sec_ax.set_xlim(mesh.params[:, 0].min() / 1e3, mesh.params[:, 0].max() / 1e3)
    sec_ax.set_ylim(mesh.params[:, 1].min() / 1e3, 0.0)
    sec_ax.set_aspect("equal")
    sec_ax.set_xlabel("Distance along strike (km)")
    sec_ax.set_ylabel("Depth (km)")

    plt.colorbar(sec, ax=list(axes), label=label, pad=0.02)
    return sec_ax.figure, axes


def plot_fit(model, track=None, cmap="RdBu_r", trace=None):
    """Data, model and residual for one track, as three map panels.

    Sharing one colour scale between data and model is the point -- an
    independently scaled model panel can make a poor fit look convincing. The
    residual gets its own, tighter, scale.

    The residual is **observed minus modelled**, so it reads in the same sense as
    the data panel beside it: red where there is more displacement than the model
    accounts for, not less.
    """
    import matplotlib.pyplot as plt

    obs = model.obs
    track = track or obs.tracks[0]
    sel = obs.track_mask(track)
    if not sel.any():
        # Otherwise this surfaces as "zero-size array to reduction operation
        # maximum" from the colour-scale line below, which names nothing. The
        # usual cause is a track name read off a *different* Observations than
        # the one the model carries -- a stale notebook variable, or a config
        # whose SCENES keys were renamed since the model was saved.
        raise ValueError(
            f"No observations for track {track!r}. This model has {obs.tracks}. "
            "Track names come from the Observations the model was built on; "
            "use model.obs.tracks, or omit track= for the first one."
        )
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
