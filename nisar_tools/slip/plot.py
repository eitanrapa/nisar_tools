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
              vmin=None, vmax=None, title=None, kernels=None):
    """Slip on the unrolled fault: along-strike distance against depth.

    ``component`` is ``"magnitude"``, ``"strike"`` or ``"dip"``. Elements are
    drawn as true triangles, so the mesh itself is visible and a bad element
    cannot hide behind interpolation.

    ``kernels`` takes a :func:`~nisar_tools.slip.resolution.resolution_report`
    and draws the **resolving length** as a bar on the panel, with the median
    along-strike kernel width in the title. Worth doing whenever the figure will
    be read for structure: on the real six-scene data the elements are 2.4 km and
    the along-strike kernel is 26-41 km, so a slip map drawn at element scale
    shows roughly ten times more detail than the data supports, and nothing in
    the picture itself says so.

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

    heading = _slip_title(model, component) if title is None else title
    if kernels is not None:
        width = float(np.median(np.asarray(kernels["width_along"].values)))
        heading += f"\nresolving length {width / 1e3:.0f} km along strike"
        # Drawn at the top-left, one kernel width long: anything narrower than
        # the bar is the regularizer's shape, not the data's.
        x0 = mesh.params[:, 0].min() / 1e3 + 0.02 * np.ptp(mesh.params[:, 0]) / 1e3
        y0 = 0.12 * mesh.params[:, 1].min() / 1e3
        ax.plot([x0, x0 + width / 1e3], [y0, y0], "-", color="0.15", lw=3,
                solid_capstyle="butt", zorder=5)
        ax.annotate("resolving length", (x0, y0), textcoords="offset points",
                    xytext=(0, 5), fontsize=8, color="0.15", zorder=5)
    ax.set_title(heading)
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


def plot_mesh(mesh, trace=None, color="area", axes=None, cmap="viridis",
              label=None):
    """The fault discretisation itself: map view beside the unrolled section.

    Takes a bare :class:`~nisar_tools.slip.mesh.FaultMesh`, no solved model
    needed -- this is the figure to look at *before* committing to an inversion,
    which is when the mesh can still be changed.

    ``color`` is ``"area"`` (the default), ``"depth"``, ``"dip"``, or a
    per-element array with ``label`` naming it -- which is how model resolution
    is drawn, and resolution is the field this figure is most worth reading for
    on an irregular mesh. Element area is the other one: it is what grading
    changes, and it should be read against the sample spacing from
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
    if isinstance(color, str):
        if color not in fields:
            raise ValueError(
                f"color must be one of {sorted(fields)}, or a per-element array")
        value, label = fields[color]
    else:
        value = np.asarray(color, dtype=float)
        if value.shape != (mesh.n_elements,):
            raise ValueError(
                f"color has {value.shape} entries for {mesh.n_elements} elements")
        label = label or "value"

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
    # The element-size *range*, not `edge_length`: a graded or adapted mesh has no
    # single edge length, and the attribute reads 0 on a mesh that never had one.
    size = np.sqrt(2.0 * mesh.areas) / 1e3
    map_ax.set_title(
        f"{mesh.n_elements} elements, {mesh.n_nodes} nodes "
        f"({mesh.attrs.get('kind', '?')}/{mesh.attrs.get('layout', 'uniform')}, "
        f"{size.min():.1f}-{size.max():.1f} km)"
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


def plot_fit(model, track=None, cmap="RdBu_r", trace=None, ramp=True):
    """Data, model and residual for one track, as map panels.

    Sharing one colour scale between data and model is the point -- an
    independently scaled model panel can make a poor fit look convincing. The
    residual gets its own, tighter, scale.

    The residual is **observed minus modelled**, so it reads in the same sense as
    the data panel beside it: red where there is more displacement than the model
    accounts for, not less.

    **When the inversion solved a nuisance ramp, the figure gains a second row**
    (``ramp=False`` keeps the three-panel form): the ramp on its own, the data
    with that ramp taken out, and the fault's own prediction. The top row shows
    the fit; the bottom row shows *what was fitted with*. That separation is the
    only way to see the failure mode :func:`~nisar_tools.slip.regularize.ramp_columns`
    warns about -- unbounded ramp columns drifting into the coseismic signal --
    because it leaves variance reduction untouched while the slip model quietly
    goes wrong. The ramp panel deliberately shares the **data** panel's colour
    scale, so how much of the data the ramp is carrying can be read off directly;
    the deramped pair share their own, since between them that is the comparison
    that matters.
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

    data = model.data[sel]
    prediction = model.prediction[sel]
    residual = model.residual[sel]
    ramp_surface = model.ramp_prediction[sel]
    show_ramp = bool(ramp) and model.ramp.size > 0

    limit = np.abs(np.concatenate([data, prediction])).max() or 1.0
    res_limit = np.abs(residual).max() or 1.0
    panels = [
        ("Data", data, limit),
        ("Model", prediction, limit),
        ("Residual", residual, res_limit),
    ]
    if show_ramp:
        deramped_data = data - ramp_surface
        fault = prediction - ramp_surface
        # One scale for the two deramped panels: they are the comparison.
        flat = np.abs(np.concatenate([deramped_data, fault])).max() or 1.0
        panels += [
            ("Ramp", ramp_surface, limit),
            ("Data - ramp", deramped_data, flat),
            ("Model - ramp (slip only)", fault, flat),
        ]

    rows = 2 if show_ramp else 1
    fig, axes = plt.subplots(rows, 3, figsize=(15, 4.4 * rows),
                             sharex=True, sharey=True, squeeze=False)
    for ax, (name, value, span) in zip(axes.ravel(), panels):
        sc = ax.scatter(x, y, c=value, s=9, cmap=cmap, vmin=-span, vmax=span)
        if trace is not None:
            tx, ty = trace.to_local(obs.frame)
            ax.plot(tx / 1e3, ty / 1e3, "k-", lw=1.2)
        ax.set_aspect("equal")
        ax.set_xlabel("East (km)")
        ax.set_title(name)
        plt.colorbar(sc, ax=ax, label="LOS (m)", pad=0.02)
    for ax in axes[:, 0]:
        ax.set_ylabel("North (km)")

    title = f"{track}: RMS residual {np.sqrt(np.mean(residual ** 2)):.4f} m"
    if show_ramp:
        # How much of the data the nuisance columns are carrying, in the units
        # the panels are drawn in -- the number to weigh against peak slip when
        # deciding whether the ramp degree is too generous.
        title += (f" | ramp spans {np.ptp(ramp_surface):.3f} m of "
                  f"{np.ptp(data):.3f} m")
    fig.suptitle(title)
    fig.tight_layout()
    return fig, (axes if show_ramp else axes[0])


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

def plot_quadrature_order(engine, mesh, obs, ax=None, cmap="viridis",
                          stat="total", title=None):
    """Where the adaptive quadrature actually spends its points.

    Shades each triangle by how many point sources it is cut into, reduced over
    the observations with ``stat``:

    ``"total"``
        Sum over observations -- the *work* that triangle costs, which is what
        the runtime is made of.
    ``"mean"`` / ``"max"``
        Per-observation order, for reading the rule rather than the bill.

    The order is ``(L/r) * sqrt(C / tolerance)``, and looking at it is worth more
    than reasoning about it. On the real Venezuela mesh the cost is heaviest at
    **depth**, not at the surface -- the opposite of what "``r`` is at least the
    element's own depth" suggests on its own. The reason is ``bias_w``: a graded
    mesh runs 0.99 km levels at the top to 4.96 km at the base, so ``L`` varies
    **5x** with depth and beats the ``r`` variation outright. Deep elements are
    bigger, so they need more points despite being further away.

    Two consequences worth having in mind. Depth grading is a quadrature-cost
    knob as much as a resolution one, and the expensive half of a graded mesh is
    the half that resolves slip worst. The along-strike variation, by contrast,
    does track observation proximity, since the mesh is uniform along strike.

    A plot that comes out uniform means the tolerance is tight enough that
    everything saturated at the cap.

    ``engine`` must be a :class:`~nisar_tools.slip.layered.LayeredPointSource`;
    the half-space engine evaluates a closed form and has no quadrature at all.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    if not hasattr(engine, "_orders"):
        raise TypeError(
            f"{type(engine).__name__} has no quadrature -- only the layered "
            "point-source engine cuts elements into point sources; the "
            "half-space engine evaluates a closed form per element"
        )
    reduce = {"total": np.sum, "mean": np.mean, "max": np.max}
    if stat not in reduce:
        raise ValueError(f"stat must be one of {sorted(reduce)}")

    ds = obs.ds if hasattr(obs, "ds") else obs
    x = np.asarray(ds["x"].values, dtype=float)
    y = np.asarray(ds["y"].values, dtype=float)
    z = np.zeros_like(x)

    centroids = np.asarray(mesh.centroids, dtype=float)
    sizes = np.sqrt(2.0 * np.asarray(mesh.areas, dtype=float))
    value = np.empty(mesh.n_elements)
    for k in range(mesh.n_elements):
        points = engine._orders(centroids[k], sizes[k], x, y, z) ** 2
        value[k] = reduce[stat](points)

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 3.2))
    coll = PolyCollection(mesh.param_vertices / 1e3, array=value, cmap=cmap,
                          edgecolors="none")
    ax.add_collection(coll)
    ax.set_xlim(mesh.params[:, 0].min() / 1e3, mesh.params[:, 0].max() / 1e3)
    ax.set_ylim(mesh.params[:, 1].min() / 1e3, 0.0)
    ax.set_xlabel("Distance along strike (km)")
    ax.set_ylabel("Depth (km)")
    ax.set_aspect("equal")
    if title is None:
        cap = int(np.sqrt(engine.max_points)) ** 2
        title = (f"quadrature points per triangle ({stat}) -- tolerance "
                 f"{engine.tolerance!r}, cap {cap}")
    ax.set_title(title)
    label = {"total": "points summed over observations",
             "mean": "points per observation",
             "max": "points at the nearest observation"}[stat]
    plt.colorbar(coll, ax=ax, label=label, pad=0.02)
    return ax.figure, ax


def _nodal_component(model, component):
    """Per-*node* slip, for a model whose parameters actually live on nodes."""
    basis = getattr(model, "basis", None)
    if getattr(basis, "name", None) != "node":
        raise ValueError(
            "this view draws the nodal parameters themselves, so it needs a "
            f"model solved with basis='node' (this one is "
            f"{getattr(basis, 'name', 'unknown')!r}). Use plot_slip for an "
            "element-constant model."
        )
    strike = np.asarray(model.strike_slip, dtype=float)
    dip = np.asarray(model.dip_slip, dtype=float)
    values = {"magnitude": np.hypot(strike, dip), "strike": strike, "dip": dip}
    if component not in values:
        raise ValueError(f"component must be one of {sorted(values)}")
    return values[component]


def plot_slip_edges(model, component="magnitude", ax=None, cmap="magma_r",
                    vmin=None, vmax=None, segments=12, linewidth=2.0,
                    title=None):
    """Nodal slip drawn along the mesh edges, varying linearly between nodes.

    :func:`plot_slip` shades each triangle by the mean of its three nodes, which
    is what makes a nodal and an element model comparable by eye -- but it also
    throws away the thing that distinguishes them. A nodal basis represents slip
    as a continuous field, linear across every element, and this draws that
    directly: each edge is subdivided into ``segments`` pieces coloured by the
    interpolated value, so the gradient between two nodes is visible rather than
    averaged away.

    Reading it: a smooth model shows colour changing gradually along every edge.
    Adjacent edges disagreeing sharply at a shared node is the nodal signature of
    an under-smoothed solve -- the same thing ``roughness`` scores, but localised
    so you can see *where*.

    Needs ``basis="node"``; an element-constant model has no nodal values to
    interpolate and raises rather than inventing some.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    mesh = model.mesh
    value = _nodal_component(model, component)
    params = np.asarray(mesh.params, dtype=float)[:, :2] / 1e3
    if value.size != params.shape[0]:
        raise ValueError(
            f"{value.size} nodal values against {params.shape[0]} nodes -- the "
            "model and its mesh disagree"
        )

    triangles = np.asarray(mesh.triangles)
    edges = {(min(a, b), max(a, b))
             for tri in triangles
             for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0]))}
    edges = np.array(sorted(edges), dtype=int)

    # Subdivide every edge once, vectorised: node ends -> `segments` pieces, each
    # coloured by the value at its own midpoint.
    t = np.linspace(0.0, 1.0, int(segments) + 1)
    start, end = params[edges[:, 0]], params[edges[:, 1]]
    points = start[:, None, :] + (end - start)[:, None, :] * t[None, :, None]
    pieces = np.stack([points[:, :-1, :], points[:, 1:, :]], axis=2)
    pieces = pieces.reshape(-1, 2, 2)

    v0, v1 = value[edges[:, 0]], value[edges[:, 1]]
    mid = 0.5 * (t[:-1] + t[1:])
    shade = (v0[:, None] + (v1 - v0)[:, None] * mid[None, :]).ravel()

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 3.2))
    coll = LineCollection(pieces, array=shade, cmap=cmap, linewidths=linewidth)
    _scale_slip(coll, shade, component, vmin, vmax)
    ax.add_collection(coll)

    ax.set_xlim(params[:, 0].min(), params[:, 0].max())
    ax.set_ylim(params[:, 1].min(), 0.0)
    ax.set_xlabel("Distance along strike (km)")
    ax.set_ylabel("Depth (km)")
    ax.set_aspect("equal")
    ax.set_title(_slip_title(model, component) if title is None else title)
    plt.colorbar(coll, ax=ax, label="Slip (m)", pad=0.02)
    return ax.figure, ax
