"""What the data can actually resolve, and the mesh and sampling that follow.

Variance reduction measures *fit*, and a slip model can fit beautifully on a
discretisation the data never constrained -- the extra freedom is simply spent
on whatever the noise asks for. Model resolution measures the other thing: how
much of each parameter's estimate comes from the data rather than from the
regulariser. Two long-standing recipes fall straight out of it, and this module
is both, sharing one matrix:

* **Mesh from resolution** (Barnhart & Lohman 2010): make the elements large
  where resolution is poor and small where it is good, so no parameter is
  either starved or free. :func:`target_element_size` and :func:`adapt_layout`.
* **Sampling from resolution** (Lohman & Simons 2005): put the observations
  where they most improve resolution. :func:`resolution_gain` returns a field
  that :meth:`~nisar_tools.slip.sampling.Observations.from_los` already accepts
  as its ``field=`` argument, so no new sampler is needed -- only a different
  thing to quadtree.

Using one quantity for both is the point. Alternating a resolution-sized mesh
against a Wang & Fialko predicted-field sampling is two heuristics in different
currencies; driving both from ``R`` is one objective.

⚠️ **Two caveats, both real, both in the callers' way if ignored.**

1. :meth:`~nisar_tools.slip.inversion.SlipInversion.solve` is **bound
   constrained**, and the Venezuela run had 86% of elements sitting on a bound.
   ``R`` describes the *unbounded* damped problem, which is a different problem.
   Treat these numbers as a discretisation heuristic, never as a statement about
   uncertainty in the model that was actually solved.
2. ``R`` depends on the smoothing weight, so it must be evaluated at the weight
   the run uses; a resolution computed at the wrong lambda sizes the mesh for a
   different inversion.

⚠️ **There is deliberately no cheap column-norm stand-in for ``R``, and that is
measured.** ``||G[:, k]||`` -- how strongly the data sees element ``k`` -- is the
obvious way to avoid a factorisation, and it does not work: on the two-scene
synthetic fixture its correlation with depth is **-0.036** against the true
resolution's **-0.742**, and it stays flat at ~0.003 from the surface to 20 km
while resolution falls from 0.34 to 2e-6. A deeper element radiates a broader but
weaker pattern whose norm is roughly conserved; resolution decays because deep
elements *trade off with each other*, and a column norm structurally cannot see a
trade-off. Nothing is lost by removing it either -- the premise that round zero
has no smoothing weight to evaluate ``R`` at was wrong, since the loop's own
``solve_kwargs`` always carries one.

A third assumption is worth naming rather than leaving implicit: the data
covariance here is ``sigma**2 I`` modulated by the per-track weights
:class:`~nisar_tools.slip.sampling.Observations` already carries. Lohman & Simons'
other contribution is a full ``Cd`` estimated from the residual variogram, which
would enter as ``G' Cd^-1 G`` throughout. That is not built.
"""

import warnings

import numpy as np
import scipy.sparse as sp
import xarray as xr
from scipy.linalg import cho_factor, cho_solve

from .layout import MAX_COUNT_RATIO, MeshLayout, require_boundaries, triangulate

#: Ridge added to the normal equations before factorising, relative to the mean
#: diagonal. Only to keep the Cholesky well posed when a parameter is completely
#: unconstrained -- far below the smoothing term for anything the data sees.
JITTER = 1e-10

#: Grid the resolution-gain field is evaluated on, in metres. Deliberately much
#: coarser than :data:`~nisar_tools.slip.predict.DEFAULT_SPACING`: the gain is a
#: smooth sum of squared Green's functions, and it is only ever used to *rank*
#: locations, while the cost is a full design matrix at every probe point. On the
#: real Venezuela crop at 2 km that is a 30 580 x 3392 matrix -- 830 MB and two
#: minutes, per scene per round; at 5 km it is 134 MB and twenty seconds.
GAIN_SPACING = 5000.0


def _normal_equations(inversion, smoothing, ss_ratio=1.0, ds_ratio=3.0,
                      boundary_ratio=1.0, sides=("bottom", "left", "right"),
                      smooth_weighting="uniform"):
    """``(data, damping)`` normal-equation blocks for the solve's own operator.

    Assembled from exactly what :meth:`SlipInversion.solve` stacks -- weighted
    design, scaled smoothing, boundary rows -- so the resolution describes the
    inversion being run rather than a tidier relative of it.
    """
    from .regularize import laplace_beltrami, neighbor_smoothing

    if inversion.basis.name == "element":
        smooth = neighbor_smoothing(inversion.mesh.neighbors, ss_ratio, ds_ratio,
                                    weighting=smooth_weighting, mesh=inversion.mesh)
    else:
        smooth = laplace_beltrami(inversion.mesh, ss_ratio, ds_ratio)
    boundary = _boundary_rows(inversion.basis, sides, boundary_ratio)

    n_ramp = inversion.n_ramp
    if n_ramp:
        smooth = sp.hstack([smooth, sp.csr_matrix((smooth.shape[0], n_ramp))],
                           format="csr")
        boundary = sp.hstack([boundary, sp.csr_matrix((boundary.shape[0], n_ramp))],
                             format="csr")
    smooth = smooth * (float(smoothing) / max(1, smooth.shape[0]))

    design = np.hstack([inversion.g, inversion.ramp]) if n_ramp else inversion.g
    weighted = design * inversion.w[:, None]

    data = weighted.T @ weighted
    damping = (smooth.T @ smooth + boundary.T @ boundary).toarray()
    return data, damping


def _boundary_rows(basis, sides, ratio):
    from .inversion import _boundary_rows as rows
    return rows(basis, sides, ratio)


def resolution_diagonal(inversion, smoothing, ss_ratio=1.0, ds_ratio=3.0,
                        boundary_ratio=1.0, sides=("bottom", "left", "right"),
                        smooth_weighting="uniform"):
    """Per-parameter model resolution, the diagonal of ``R``.

    ``R = (A'A + S'S + B'B)^-1 A'A`` with ``A`` the weighted design matrix and
    ``S``/``B`` the smoothing and boundary blocks -- i.e. the resolution of the
    damped problem :meth:`SlipInversion.solve` sets up, minus its bounds (see
    this module's caveats). ``1`` means the estimate came entirely from the data,
    ``0`` entirely from the regulariser.

    Returns an :class:`xarray.Dataset` on the ``basis`` dimension with
    ``strike_slip``, ``dip_slip`` and their pointwise maximum ``resolution``,
    plus the ramp columns' diagonal in ``attrs`` where there are any.

    One dense Cholesky of a ``(2 n_basis + n_ramp)`` square matrix: seconds and a
    few hundred megabytes at the production 918-node mesh, which is why it is
    affordable once per round of an adaptive loop but not per candidate cell.
    """
    data, damping = _normal_equations(inversion, smoothing, ss_ratio, ds_ratio,
                                      boundary_ratio, sides, smooth_weighting)
    total = data + damping
    total.flat[:: total.shape[0] + 1] += JITTER * np.trace(total) / total.shape[0]

    resolved = cho_solve(cho_factor(total, lower=True), data)
    diagonal = np.diag(resolved).copy()

    n = inversion.basis.n_basis
    strike, dip = diagonal[:n], diagonal[n:2 * n]
    out = xr.Dataset(
        {
            "strike_slip": ("basis", strike),
            "dip_slip": ("basis", dip),
            "resolution": ("basis", np.maximum(strike, dip)),
        },
        coords={"basis": np.arange(n)},
    )
    out.attrs.update(
        smoothing=float(smoothing), basis=inversion.basis.name,
        n_ramp=int(inversion.n_ramp),
        # The effective number of parameters the data pays for. Falls with the
        # smoothing weight, which is the sanity check that this is a resolution
        # and not just a scaled column norm.
        effective_rank=float(np.sum(diagonal[:2 * n])),
        ramp_resolution=[float(v) for v in diagonal[2 * n:]],
    )
    return out


def element_resolution(inversion, resolution):
    """Resolution per element, whatever basis it was computed in."""
    values = np.asarray(resolution["resolution"].values, dtype=float)
    if inversion.basis.name == "element":
        return values
    return inversion.basis.element_values(values)


def element_size(mesh):
    """The equivalent square edge of each element, ``sqrt(2 * area)``.

    The same length ``layered._orders`` uses to price a triangle against its
    distance to an observation, so "element size" means one thing in the package.
    """
    return np.sqrt(2.0 * np.asarray(mesh.areas, dtype=float))


def smooth_over_mesh(mesh, values, rounds=3):
    """Average a per-element field with its neighbours, ``rounds`` times.

    A target size field has to be Lipschitz or it asks for a jump no
    triangulation can honour without slivers; a few Jacobi sweeps over the
    element adjacency is the cheapest way to get there and needs no geometry
    beyond ``mesh.neighbors``.
    """
    values = np.asarray(values, dtype=float).copy()
    neighbors = mesh.neighbors
    for _ in range(int(rounds)):
        total = values.copy()
        count = np.ones(values.size)
        for k in range(neighbors.shape[1]):
            other = neighbors[:, k]
            has = other >= 0
            total[has] += values[other[has]]
            count[has] += 1.0
        values = total / count
    return values


def target_element_size(mesh, resolution, target=None, gain=0.5, max_ratio=3.0,
                        size_min=None, size_max=None, rounds=3):
    """Element sizes that would even out ``resolution`` across the mesh.

    ``h_new = h * (target / r) ** gain``. An element resolved better than
    ``target`` can afford to be smaller; one resolved worse is carrying
    parameters the data does not pay for and is grown. ``target`` defaults to the
    median, so the node budget is roughly conserved and the *distribution* is
    what moves -- pass a higher value to buy a coarser mesh outright.

    ``gain`` below 1 under-relaxes deliberately. Resolution is itself a function
    of the discretisation, so a full step chases its own tail; half a step
    converges in a few rounds instead of oscillating.

    ⚠️ **``max_ratio`` is not a nicety.** Below a few kilometres of slip depth,
    resolution does not merely get small, it underflows to *exactly* zero -- on
    the 330-element synthetic fixture at lambda 0.3 the mean below 14 km is 0.0000
    against 0.3304 above 6 km. ``target / r`` is then whatever the floor makes it,
    ``gain`` cannot temper an infinity, and one step asks for elements four times
    the mesh's own extent: measured, an unbounded first pass collapsed a 20 km
    fault to **three** depth levels with a 7.5 degree minimum angle. Bounding the
    *ratio* per step, and iterating, is what keeps a converging loop from taking
    one meaningless leap. ``size_min``/``size_max`` remain as absolute clamps on
    top.

    ⚠️ **This sets the size field's *shape*, not the mesh's size.** Targeting the
    median looks self-normalising and is not: the resolution distribution is
    heavily skewed -- most elements near zero, a few near one -- so most of them
    hit the growth cap while few can shrink, and the mean element size climbs
    every round. Measured on the two-scene synthetic fixture, a loop with no
    element budget coarsened 330 elements to 186, then 46, then 44, with variance
    reduction collapsing 94.3% -> 91.6% -> **0.74%** and peak slip falling to a
    millimetre: it ran end to end, converged, and returned nothing. The count is
    held by :func:`adapt_layout`'s ``budget``, which is the right division of
    labour anyway -- the caller says how many parameters they can afford, the
    resolution says where to put them.
    """
    resolution = np.asarray(resolution, dtype=float)
    if resolution.shape != (mesh.n_elements,):
        raise ValueError(
            f"resolution has {resolution.shape} entries for {mesh.n_elements} elements; "
            "pass element_resolution(inversion, ...) rather than the basis diagonal"
        )
    current = element_size(mesh)
    target = float(np.median(resolution)) if target is None else float(target)
    if target <= 0:
        raise ValueError("target resolution must be positive")
    max_ratio = float(max_ratio)
    if max_ratio < 1.0:
        raise ValueError("max_ratio is a growth factor and must be at least 1")

    floor = np.maximum(resolution, 1e-12)
    ratio = np.clip((target / floor) ** float(gain), 1.0 / max_ratio, max_ratio)
    # The *correction* is smoothed, not the absolute size, so a resolution field
    # asking for no change gets none: smoothing `current * ratio` would blur the
    # mesh's own slightly uneven element sizes into each other and drift.
    size = current * smooth_over_mesh(mesh, ratio, rounds=rounds)

    lo = current.min() / max_ratio if size_min is None else float(size_min)
    hi = current.max() * max_ratio if size_max is None else float(size_max)
    if lo > hi:
        raise ValueError(f"size_min={lo} is above size_max={hi}")
    return np.clip(size, lo, hi)


def _stations_for_density(extent, density, samples=1024):
    """Positions across ``[0, extent]`` at a varying target spacing.

    Integrating ``1/h`` and then splitting the total into a whole number of
    equal steps is what lands on both ends *exactly*, which is what
    :func:`~nisar_tools.slip.layout.boundary_node_mask` needs; marching by the
    local spacing would leave the far end a ragged fraction short.
    """
    grid = np.linspace(0.0, float(extent), int(samples))
    inverse = 1.0 / np.maximum(density(grid), 1e-9)
    cumulative = np.concatenate([[0.0], np.cumsum(
        0.5 * (inverse[1:] + inverse[:-1]) * np.diff(grid))])
    n = max(1, int(round(cumulative[-1])))
    out = np.interp(np.linspace(0.0, cumulative[-1], n + 1), cumulative, grid)
    out[0], out[-1] = 0.0, float(extent)
    return out


def adapt_layout(mesh, size, budget="conserve", top_depth=None, max_depth=None,
                 length=None, balance=MAX_COUNT_RATIO, samples=1024, tolerance=0.05):
    """A new row-structured layout honouring a per-element target size.

    ``budget`` is the element count to land on: ``"conserve"`` keeps the mesh's
    own, an integer asks for that many, ``None`` takes whatever ``size`` implies.
    It is found by scaling the whole size field and rebuilding -- a handful of
    bisection steps, each two interpolations and one Delaunay -- rather than from
    a tiling formula, because this builder integrates ``1/h`` along each axis
    *separately* and a closed form for the count is then wrong by a factor that
    depends on how anisotropic the size field is.

    ⚠️ **Without a budget an adaptive loop does not converge, it drifts**; see
    :func:`target_element_size` for the measurement.

    Depth levels come from integrating ``1/h`` down dip, and each level's nodes
    from integrating ``1/h`` along strike *at that depth* -- so both the count
    **and** the positions vary per level, which is what "vary the number of nodes
    at the depth levels" costs once the size field is not separable.

    The size field is scattered from element centroids and read back by nearest
    neighbour in ``(s, z)``; it is a target for a mesh generator, not a function
    being integrated accurately, so a smooth interpolant would be false
    precision.
    """
    size = np.asarray(size, dtype=float)
    if size.shape != (mesh.n_elements,):
        raise ValueError(f"size has {size.shape} entries for {mesh.n_elements} elements")
    if budget is not None:
        wanted = mesh.n_elements if budget == "conserve" else int(budget)
        if wanted < 1:
            raise ValueError("budget must be at least one element")
        return _fit_budget(
            lambda scale: adapt_layout(mesh, size * scale, budget=None,
                                       top_depth=top_depth, max_depth=max_depth,
                                       length=length, balance=balance, samples=samples),
            wanted, float(tolerance))

    centre = mesh.element_params
    params = mesh.params
    length = float(params[:, 0].max()) if length is None else float(length)
    top_depth = float(-params[:, 1].max()) if top_depth is None else float(top_depth)
    max_depth = float(-params[:, 1].min()) if max_depth is None else float(max_depth)

    def at(s, depth):
        s = np.atleast_1d(np.asarray(s, dtype=float))
        depth = np.broadcast_to(np.asarray(depth, dtype=float), s.shape)
        # Distances in metres on both axes, so nearest is well posed without a
        # scaling choice; the parameter plane is already metric.
        d2 = ((s[:, None] - centre[None, :, 0]) ** 2
              + (-depth[:, None] - centre[None, :, 1]) ** 2)
        return size[np.argmin(d2, axis=1)]

    middle = 0.5 * length
    depths = top_depth + _stations_for_density(
        max_depth - top_depth,
        lambda d: at(np.full_like(d, middle), top_depth + d),
        samples=samples,
    )
    depths = _balance_levels(depths, balance)

    rows, counts = [], []
    for depth in depths:
        stations = _stations_for_density(
            length, lambda s, d=depth: at(s, np.full_like(s, d)), samples=samples)
        rows.append(stations)
        counts.append(stations.size)

    counts = _rebalance(np.asarray(counts, dtype=int), balance)
    params = np.vstack([
        np.column_stack([_resample_stations(row, int(n)),
                         np.full(int(n), -float(depth))])
        for row, n, depth in zip(rows, counts, depths)
    ])

    layout = MeshLayout(params, triangulate(params),
                        levels=depths, counts=counts,
                        attrs={"layout": "adapted"})
    require_boundaries(layout.params)
    return layout


def _fit_budget(build, wanted, tolerance):
    """Scale the size field until the built layout holds about ``wanted`` elements.

    Monotone in the scale -- bigger elements, fewer of them -- so a geometric
    bracket followed by bisection in log-scale converges in a handful of builds,
    and each build is cheap enough to do this properly rather than estimate it.
    """
    layout = build(1.0)
    if abs(layout.n_elements - wanted) <= tolerance * wanted:
        return layout

    lo = hi = 1.0
    if layout.n_elements > wanted:
        for _ in range(12):                     # too many elements -> coarsen
            lo, hi = hi, hi * 1.5
            layout = build(hi)
            if layout.n_elements <= wanted:
                break
    else:
        for _ in range(12):                     # too few -> refine
            lo, hi = lo / 1.5, lo
            layout = build(lo)
            if layout.n_elements >= wanted:
                break

    best = layout
    for _ in range(20):
        mid = float(np.sqrt(lo * hi))
        layout = build(mid)
        if abs(layout.n_elements - wanted) < abs(best.n_elements - wanted):
            best = layout
        if abs(layout.n_elements - wanted) <= tolerance * wanted:
            return layout
        if layout.n_elements > wanted:
            lo = mid
        else:
            hi = mid
    return best


def _balance_levels(depths, balance):
    """Limit how fast the level thickness may change from one level to the next.

    The counts are balanced across strike by
    :func:`~nisar_tools.slip.layout._balance_counts`; this is the same rule down
    dip, and it is needed for the same reason. A size field that saturates at
    depth otherwise puts one 17 km-tall element under a 2.5 km one, which is a
    7-to-1 aspect ratio however many nodes the row has.
    """
    thickness = np.diff(np.asarray(depths, dtype=float))
    if thickness.size < 2:
        return np.asarray(depths, dtype=float)
    for _ in range(thickness.size):
        before = thickness.copy()
        thickness[1:] = np.minimum(thickness[1:], thickness[:-1] * balance)
        thickness[:-1] = np.minimum(thickness[:-1], thickness[1:] * balance)
        if np.allclose(thickness, before):
            break
    # Renormalise so the levels still span exactly the same depth range; the
    # bottom must stay at max_depth or the mesh quietly changes extent.
    thickness *= (depths[-1] - depths[0]) / thickness.sum()
    return depths[0] + np.concatenate([[0.0], np.cumsum(thickness)])


def _resample_stations(stations, n):
    """Change a row's node count while keeping the shape of its spacing.

    Resampled uniformly *in index*, so a row that was dense at one end stays
    dense there, and both ends land on their exact original values -- which is
    what keeps the boundary test exact after ``_balance_counts`` has raised a
    count.
    """
    if stations.size == n:
        return stations
    return np.interp(np.linspace(0.0, 1.0, n),
                     np.linspace(0.0, 1.0, stations.size), stations)


def _rebalance(counts, balance):
    from .layout import _balance_counts

    # Quiet: these counts came from integrating a size field, not from a caller
    # asking for them, so refining one is the algorithm rather than an override
    # -- and the budget search rebuilds a dozen times.
    return _balance_counts(counts, balance, warn=False)


def resolution_gain(model, los_stack, resolution, frame=None, spacing=GAIN_SPACING,
                    exponent=1.0):
    """Where another observation would most improve model resolution.

    Lohman & Simons 2005's criterion, evaluated as a field rather than by
    recursing on candidate cells: a sample at ``(x, y)`` helps in proportion to
    how strongly it sees the parameters that are *currently unresolved*,

    ``w(x, y) = sum_k G(x, y; k)**2 * (1 - r_k)``.

    Returns an ``(ny, nx)`` array on ``los_stack``'s own grid, normalised to unit
    standard deviation over its finite samples, ready to hand to
    :meth:`~nisar_tools.slip.sampling.Observations.from_los` as ``field=``.

    Evaluated on a coarse grid (:data:`GAIN_SPACING`, coarser than the predicted
    field's) and interpolated up, because the cost is a full Green's-function
    assembly at every probe point. Recursing instead -- testing
    each candidate cell against ``R`` as the paper describes -- needs a
    Green's-function row per candidate, thousands of them, against a quadtree
    that currently costs 0.05 s on 0.32 M pixels.
    """
    from scipy.interpolate import RegularGridInterpolator

    from .sampling import _to_frame, stack_epsg

    spacing = GAIN_SPACING if spacing is None else float(spacing)
    frame = frame or model.mesh.frame
    if frame is None:
        raise ValueError("resolution_gain needs a LocalFrame; the mesh carries none")

    mesh = model.mesh
    unresolved = 1.0 - np.clip(
        np.asarray(resolution["resolution"].values, dtype=float), 0.0, 1.0)

    ds = los_stack.ds
    epsg = stack_epsg(ds)
    x_native = np.asarray(ds["x"].values, dtype=float)
    y_native = np.asarray(ds["y"].values, dtype=float)
    gx, gy = np.meshgrid(x_native, y_native)
    fx, fy = _to_frame(gx.ravel(), gy.ravel(), frame, epsg)

    cx = np.arange(fx.min(), fx.max() + spacing, spacing)
    cy = np.arange(fy.min(), fy.max() + spacing, spacing)
    px, py = np.meshgrid(cx, cy)

    # The scene's own mean look vector, renormalised. One direction for the whole
    # coarse grid is an approximation -- a wide swath runs 25 to 49 degrees of
    # incidence -- but the gain only has to *rank* candidate locations, and using
    # the mean beats a nominal vertical, which would misweight the east-west
    # sensitivity that a strike-slip fault's signal mostly lives in.
    look = np.array([np.nanmean(np.asarray(ds[f"los_{c}"].values, dtype=float))
                     for c in ("east", "north", "up")])
    norm = np.linalg.norm(look)
    look = look / norm if norm > 0 else np.array([0.0, 0.0, 1.0])
    ones = np.ones(px.size)
    engine = model.inversion.engine if model.inversion is not None else None
    if engine is None:
        raise ValueError(
            "resolution_gain needs the model's engine, which SlipModel.load "
            "deliberately does not restore; rebuild the SlipInversion first"
        )
    green = engine.los_matrix(mesh, px.ravel(), py.ravel(),
                              look[0] * ones, look[1] * ones, look[2] * ones)

    n = model.basis.n_basis
    if green.shape[1] != 2 * n:
        projection = model.basis.projection()
        green = np.hstack([green[:, :mesh.n_elements] @ projection,
                           green[:, mesh.n_elements:] @ projection])
    weight = np.concatenate([unresolved, unresolved]) ** float(exponent)
    gain = (green ** 2) @ weight

    interp = RegularGridInterpolator(
        (cy, cx), gain.reshape(px.shape), bounds_error=False, fill_value=np.nan)
    out = interp(np.column_stack([fy, fx])).reshape(gx.shape)

    finite = np.isfinite(out)
    if not finite.any():
        raise ValueError("the resolution-gain field is empty over this scene")
    scale = out[finite].std()
    if scale == 0:
        warnings.warn(
            "the resolution-gain field is constant, so it cannot steer a "
            "quadtree; every parameter is equally (un)resolved from this scene.",
            RuntimeWarning, stacklevel=2,
        )
        return out
    return (out - out[finite].mean()) / scale
