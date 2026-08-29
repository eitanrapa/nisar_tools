"""Measure a scene's sampling parameters instead of inheriting them.

Every knob :meth:`~nisar_tools.slip.sampling.Observations.from_los` takes is a
statement about the *data*, but nothing in the package measured any of them, so
in practice they got copied from the notebook example. Each wrong value then
produced a symptom that looked like something else entirely:

* ``rms_min`` below the scene's noise makes the quadtree unable to stop. The
  split test is on the *pixel* scatter inside a cell, and that scatter does not
  shrink as the cell does -- so beneath the noise floor every cell runs down to
  ``width_min`` and the sample count is set by a size limit rather than by
  information. Measured on the Venezuela scenes: a median within-cell scatter of
  **11.8 mm** against ``rms_min=6 mm``, which is the whole explanation for 18 631
  samples, most of them resolving atmosphere.
* ``width_min`` acts discontinuously. Cells are index rectangles halved at their
  midpoint, so the reachable sizes are a dyadic ladder -- per axis, and different
  for every scene. It only bites when it crosses a rung: 1000 m gave a 1075 m
  terminal cell and 57 246 samples, 1500 m gave 1750 m and 24 429.
* ``exclude_within`` is usually set far larger than it needs to be. The hard
  requirement is only that a sample not sit *on* the fault, where the
  displacement field is discontinuous and the dislocation solution returns
  non-finite values. The binding constraint is milder: a *cell* must not straddle
  the trace, so ``exclude_within >= width_min / 2``.

And none of those is the thing that actually limits the answer. Coverage is, and
it is invisible in every number the package otherwise reports: on the same
scenes, 19% of samples lay north of the trace, and north-side data existed only
along the western third while the largest signal sat in the east.

Everything here is computed **per scene**, because these are per-scene properties
that :meth:`Observations.concat` averages away.
"""

import numpy as np
import xarray as xr

from .sampling import _to_frame, stack_epsg

#: Fallback stand-off for :func:`noise_floor` when no mesh is available to size
#: it from. Coseismic displacement has not died away a few tens of kilometres
#: from a large rupture, so blocks nearer than this carry real signal gradient
#: and bias the noise estimate upward.
DEFAULT_MIN_DISTANCE = 50e3

#: ``rms_min`` is set this far above the measured scatter. At exactly the noise
#: floor roughly half the cells still test "rough" and run to ``width_min``; a
#: modest margin puts the stopping decision back on the data.
RMS_MIN_MARGIN = 1.5

#: Share of the better-covered side that the worse-covered side must reach for a
#: stretch of trace to count as two-sided. A bare ``> 0`` test is far too weak:
#: a handful of stray blocks makes a one-sided stretch look balanced, which is
#: exactly the reassurance this module exists to withhold.
TWO_SIDED_MIN_SHARE = 0.1


def _pixel_size(ds):
    dx = abs(float(np.diff(np.asarray(ds["x"].values, dtype=float))[0]))
    dy = abs(float(np.diff(np.asarray(ds["y"].values, dtype=float))[0]))
    return dx, dy


def _project(trace, x, y, frame):
    """Along-strike position, perpendicular distance and side, in one pass.

    :class:`~nisar_tools.slip.trace.FaultTrace` exposes ``distance`` and ``side``,
    but each re-runs the whole nearest-segment search and neither yields the
    along-strike coordinate that the coverage profile is binned on. This mirrors
    ``FaultTrace._nearest_segment`` and returns all three.
    """
    px, py = trace.to_local(frame)
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    x0, y0 = px[:-1], py[:-1]
    vx, vy = px[1:] - x0, py[1:] - y0
    vv = vx * vx + vy * vy
    vv = np.where(vv == 0, np.finfo(float).tiny, vv)

    wx = x[:, None] - x0[None, :]
    wy = y[:, None] - y0[None, :]
    t = np.clip((wx * vx + wy * vy) / vv, 0.0, 1.0)
    d2 = (wx - t * vx) ** 2 + (wy - t * vy) ** 2

    nearest = np.argmin(d2, axis=1)
    rows = np.arange(x.size)
    seg_len = np.sqrt(vv)
    start = np.concatenate([[0.0], np.cumsum(seg_len)])

    along = start[nearest] + t[rows, nearest] * seg_len[nearest]
    dist = np.sqrt(d2[rows, nearest])
    cross = vx[nearest] * wy[rows, nearest] - vy[nearest] * wx[rows, nearest]
    return along, dist, np.sign(cross).astype(np.int8)


def _blocks(los_stack, trace, frame, pair=0, block=2000.0):
    """Fixed-grid block statistics of one pair, with trace-relative positions.

    Deliberately a *fixed* grid rather than a quadtree: this is what configures
    the quadtree, so it must not depend on one.
    """
    ds = los_stack.ds
    da = ds["los"]
    if "pair" in da.dims:
        da = da.isel(pair=pair)

    dx, dy = _pixel_size(ds)
    # At least two pixels per axis, or a block holds one sample and its variance
    # is undefined -- every block filtered out and the caller told, unhelpfully,
    # that nothing was far enough from the trace. The realised size comes back as
    # ``block_size`` so a clamped request is visible rather than silent.
    nx = max(2, int(round(float(block) / dx)))
    ny = max(2, int(round(float(block) / dy)))

    # Accumulate count/sum/sum-of-squares rather than calling `.std()`, which
    # runs `nanstd` over blocks that may be entirely NaN outside the swath and
    # warns about it. This is also the estimator `_quadtree_cells` uses --
    # population variance from summed-area tables -- so the number here is
    # directly comparable to the one `rms_min` is tested against.
    coarse = {"y": ny, "x": nx}
    count = np.asarray(da.notnull().coarsen(**coarse, boundary="trim").sum().values,
                       dtype=float)
    filled = da.fillna(0.0)
    s1 = np.asarray(filled.coarsen(**coarse, boundary="trim").sum().values, dtype=float)
    s2 = np.asarray((filled ** 2).coarsen(**coarse, boundary="trim").sum().values,
                    dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, s1 / count, np.nan)
        var = np.where(count > 1, s2 / count - mean * mean, np.nan)
    std = np.sqrt(np.clip(var, 0.0, None))

    xc = ds["x"].coarsen(x=nx, boundary="trim").mean().values
    yc = ds["y"].coarsen(y=ny, boundary="trim").mean().values
    xx, yy = np.meshgrid(np.asarray(xc, dtype=float), np.asarray(yc, dtype=float))

    fx, fy = _to_frame(xx.ravel(), yy.ravel(), frame, stack_epsg(ds))
    along, dist, side = _project(trace, fx, fy, frame)

    return {
        "std": std.ravel(),
        # The block *value* and its position in the shared frame. `noise_floor`
        # wants only the scatter, but a variogram is built from the values
        # themselves and their separations, and both are already computed here --
        # see `nisar_tools.slip.covariance.empirical_variogram`, which is the
        # other caller.
        "mean": mean.ravel(),
        "x": fx,
        "y": fy,
        "count": count.ravel(),
        "along": along,
        "distance": dist,
        "side": side,
        "block_pixels": float(ny * nx),
        "pixel_area": float(dx * dy),
        "block_size": 0.5 * (ny * dy + nx * dx),
    }


def noise_floor(los_stack, trace, frame, pair=0, block=2000.0,
                min_distance=None, min_valid_fraction=0.5):
    """Median far-field within-block scatter, in metres -- the basis for ``rms_min``.

    Block-reduces the scene onto a fixed grid of ``block``-metre cells and takes
    the median scatter of those far enough from ``trace`` to be signal-free. That
    is the same quantity the quadtree thresholds, so the result is directly
    comparable to ``rms_min`` -- but measured without recursing, so it does not
    depend on the parameters it is about to set.

    ``min_distance`` defaults to :data:`DEFAULT_MIN_DISTANCE`; pass
    ``4 * max_depth`` of the mesh when there is one. Blocks closer than that
    carry real deformation gradient, which inflates the estimate.
    """
    b = _blocks(los_stack, trace, frame, pair=pair, block=block)
    if min_distance is None:
        min_distance = DEFAULT_MIN_DISTANCE

    keep = (
        np.isfinite(b["std"])
        & (b["count"] >= min_valid_fraction * b["block_pixels"])
        & (b["distance"] > float(min_distance))
    )
    if not keep.any():
        raise ValueError(
            f"No blocks survive: none are farther than {min_distance / 1e3:.0f} km "
            f"from the trace with at least {min_valid_fraction:.0%} valid pixels. "
            "Lower min_distance, lower min_valid_fraction, or widen the scene."
        )
    return float(np.median(b["std"][keep]))


def cell_size_ladder(los_stack, width_min=None, width_max=20000.0):
    """The cell sizes a quadtree of this scene can actually reach.

    ``_quadtree_cells`` halves an index rectangle at its midpoint and splits an
    axis only while that axis stays at or above its own minimum, so the reachable
    extents are a dyadic ladder from the raster's own shape -- *per axis*, and
    different for every scene. ``width_min`` therefore does nothing at all until
    it crosses a rung, which is why 1000 m and 1500 m gave 1075 m and 1750 m
    terminal cells while 1500 m and 2000 m would have given the same one.

    Returns the per-axis ladders in metres, and -- when ``width_min`` is given --
    the terminal cell that quadtree would bottom out at, reported the same way
    ``Observations`` reports ``cell_size``: the **mean of the two side lengths**,
    not the square root of the area.
    """
    ds = los_stack.ds
    dx, dy = _pixel_size(ds)
    ny = int(ds.sizes["y"])
    nx = int(ds.sizes["x"])

    def reachable(n, step, w_min, w_max):
        """Every extent the recursion can produce, and the terminal ones.

        Both halves have to be followed, not just ``n // 2``. An odd extent
        splits into ``n//2`` and ``n - n//2``, and those branches bottom out at
        different sizes -- 121 rows descends to 3 one way and all the way to 2
        the other, because only the even branch stays divisible. Chasing a single
        branch overestimated the terminal cell by 25% on a 121x187 raster.
        """
        n_min = max(1, int(round(float(w_min) / step)))
        n_max = max(1, int(round(float(w_max) / step)))
        seen, terminal, stack = set(), set(), [int(n)]
        while stack:
            m = stack.pop()
            if m in seen:
                continue
            seen.add(m)
            # Mirrors sampling._quadtree_cells: an oversize axis splits whatever
            # the floor says, otherwise only while a half stays above it.
            if m >= 2 and (m > n_max or m >= 2 * n_min):
                stack.extend((m // 2, m - m // 2))
            else:
                terminal.add(m)
        return np.array(sorted(seen), dtype=int), min(terminal)

    rows, _ = reachable(ny, dy, 1.0, width_max)
    cols, _ = reachable(nx, dx, 1.0, width_max)
    out = {
        "rows": rows,
        "cols": cols,
        "rows_m": rows.astype(float) * dy,
        "cols_m": cols.astype(float) * dx,
    }

    if width_min is not None:
        _, ty = reachable(ny, dy, width_min, width_max)
        _, tx = reachable(nx, dx, width_min, width_max)
        out["terminal_rows_m"] = ty * dy
        out["terminal_cols_m"] = tx * dx
        out["terminal_cell"] = 0.5 * (ty * dy + tx * dx)
    return out


def _snap_width_min(los_stack, target, width_max=20000.0):
    """The largest ``width_min`` whose terminal cell is nearest ``target``.

    Choosing ``width_min`` on a continuum is meaningless -- only the rung it
    lands on matters -- so pick the rung and report a ``width_min`` that reaches
    it, rather than handing back a number that silently rounds elsewhere.
    """
    ds = los_stack.ds
    dx, dy = _pixel_size(ds)
    candidates = np.unique(np.round(
        np.geomspace(max(dx, dy), max(target * 4.0, 4 * max(dx, dy)), 40)))

    best, best_err = None, np.inf
    for w in candidates:
        cell = cell_size_ladder(los_stack, width_min=float(w),
                                width_max=width_max)["terminal_cell"]
        err = abs(cell - target)
        # Ties go to the larger width_min: same cells, less recursion.
        if err <= best_err:
            best, best_err = float(w), err
    return best


def _geometry_consistent(ds):
    """Does ``los_east``'s sign match the pass and look direction?

    Left-looking on an ascending (roughly northward) pass illuminates *west* of
    the ground track, so the sensor is east of the target and the target->sensor
    east component is positive; descending reverses it. Real NISAR values are
    +0.68 ascending and -0.61 descending.

    Cheap, and it is the invariant that would have caught the inverted LOS sign
    fixed on 2026-07-28 had anything been checking it. Returns ``None`` when the
    stack does not carry both attributes.
    """
    direction = str(ds.attrs.get("direction") or "").lower()
    look = str(ds.attrs.get("look_direction") or "").lower()
    if not direction.startswith(("asc", "desc")) or not look.startswith(("left", "right")):
        return None, None

    ascending = direction.startswith("asc")
    left = look.startswith("left")
    expected = 1 if (ascending == left) else -1

    east = np.asarray(ds["los_east"].values, dtype=float)
    finite = east[np.isfinite(east)]
    if finite.size == 0:
        return None, None
    observed = int(np.sign(finite.mean()))
    return observed == expected, observed


def scene_report(los_stack, trace, frame, mesh=None, pair=0, block=2000.0,
                 along_bins=25, band=40e3, near=5e3, min_distance=None,
                 width_max=30000.0):
    """Everything worth knowing about one scene before sampling it.

    Returns an :class:`xarray.Dataset` whose data is the **along-strike coverage
    profile** -- valid area on each side of the trace, binned by arc length --
    and whose ``attrs`` carry the measured noise floor, the three sampling
    parameters derived from it, and a geometry sanity check.

    The profile is the point. Aggregate coverage numbers hid the real problem on
    the Venezuela scenes: "19% of samples north of the trace" sounds like thin
    two-sided coverage, when in fact the north block was absent along the eastern
    two-thirds -- exactly where the largest signal was. A scalar cannot say that
    and a profile can.

    ``band`` bounds the cross-strike distance counted as coverage, ``near`` the
    near-fault strip reported separately, and ``along_bins`` the profile's
    resolution.
    """
    ds = los_stack.ds
    b = _blocks(los_stack, trace, frame, pair=pair, block=block)

    if min_distance is None:
        min_distance = (4.0 * float(mesh.attrs["max_depth"])
                        if mesh is not None and "max_depth" in mesh.attrs
                        else DEFAULT_MIN_DISTANCE)

    floor = noise_floor(los_stack, trace, frame, pair=pair, block=block,
                        min_distance=min_distance)

    edge = float(mesh.attrs["edge_length"]) if mesh is not None and "edge_length" in mesh.attrs else 3e3
    w_min = _snap_width_min(los_stack, 0.5 * edge, width_max=width_max)
    ladder = cell_size_ladder(los_stack, width_min=w_min, width_max=width_max)

    # -- along-strike coverage profile -------------------------------------
    total = float(trace.length(frame))
    edges = np.linspace(0.0, total, int(along_bins) + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    area = b["count"] * b["pixel_area"]
    # A block whose centre is within half a block of the trace covers ground on
    # both sides, so attributing its area to the side its centre happens to fall
    # on leaks coverage across the fault. Measured: blanking one side entirely
    # still left 72% of bins looking two-sided until these were dropped.
    inside = (b["distance"] <= float(band)) & (b["distance"] > 0.5 * b["block_size"])

    def side_area(sign):
        w = np.where(inside & (b["side"] == sign), area, 0.0)
        return np.histogram(b["along"], bins=edges, weights=w)[0]

    left, right = side_area(1), side_area(-1)
    lesser = np.minimum(left, right)
    greater = np.maximum(left, right)
    with np.errstate(invalid="ignore", divide="ignore"):
        balance = np.where(greater > 0, lesser / greater, 0.0)
    both = balance >= TWO_SIDED_MIN_SHARE

    # Valid fraction of the blocks the raster actually spans near the trace, not
    # valid area over the band's geometric area: blocks are counted whole while
    # only their centres are tested, so the latter exceeds 1. Framing it per
    # block also isolates the question worth asking -- of the near-fault ground
    # this scene covers at all, how much survived masking and decorrelation.
    near_mask = b["distance"] <= float(near)
    near_blocks = float(near_mask.sum())
    near_valid = (float(b["count"][near_mask].sum()) / (near_blocks * b["block_pixels"])
                  if near_blocks else 0.0)

    consistent, east_sign = _geometry_consistent(ds)

    out = xr.Dataset(
        {
            "valid_left": ("along", left),
            "valid_right": ("along", right),
            "balance": ("along", balance),
            "two_sided": ("along", both),
        },
        coords={"along": centres},
    )
    out["along"].attrs["units"] = "m along strike from the trace's first vertex"
    out.attrs.update(
        track=str(ds.attrs.get("direction") or ""),
        noise_floor=floor,
        rms_min=RMS_MIN_MARGIN * floor,
        width_min=float(w_min),
        exclude_within=max(0.5 * ladder["terminal_cell"], 500.0),
        terminal_cell=float(ladder["terminal_cell"]),
        block=float(b["block_size"]),
        min_distance=float(min_distance),
        frac_left=float(left.sum() / (left.sum() + right.sum())) if (left.sum() + right.sum()) else 0.0,
        two_sided_fraction=float(both.mean()),
        near_fault_coverage=near_valid,
        los_east_sign=east_sign if east_sign is not None else 0,
        geometry_consistent=bool(consistent) if consistent is not None else None,
    )
    return out


def ramp_content(obs, covariance=None):
    """How much of the data each ramp degree explains, offset through cubic.

    The question this answers is whether ``ramp=`` is absorbing signal rather than
    nuisance. For a long, near east-west strike-slip fault it very well might: the
    far-field coseismic pattern is an arctangent step across the trace, which over
    a finite aperture looks a great deal like a gradient perpendicular to strike --
    and, at third order, a great deal like an odd cubic -- so the ramp columns and
    the slip genuinely compete.

    Needs no Green's matrix -- it reuses
    :func:`~nisar_tools.slip.regularize.ramp_columns` directly, so it costs
    milliseconds and can be run before deciding how to build the inversion.

    Because those columns are normalised by each track's own span, a returned
    gradient reads directly as **metres of line of sight across the scene**.
    Orbital ramps are centimetres at that scale; tens of centimetres is
    deformation being taken away from the slip model.

    ``gradient_only``, ``curvature_only`` and ``cubic_only`` are the marginal
    variance reduction of each step over the one before it, which is what says
    whether paying for the extra columns buys anything.

    ⚠️ **These numbers overstate the ramp, and by a lot.** Nothing here has a fault
    model, so the polynomial is free to claim the coseismic pattern -- on the real
    Venezuela set this reported an A061 curvature ten times larger, and opposite in
    sign, to what the joint inversion settles on once the fault competes for the
    same structure. So a large ``curvature_only`` or ``cubic_only`` is a prompt to
    solve at two adjacent degrees and compare peak slip and moment, **not** evidence
    that a ramp of that size is present. What it does argue for either way is
    solving the ramp jointly rather than pre-removing a higher degree.
    """
    from .regularize import ramp_columns

    ds = obs.ds
    d = np.asarray(ds["los"].values, dtype=float)
    denom = float(d @ d)

    # `covariance` is optional because this is a *pre-inversion* diagnostic --
    # it is normally run to choose `ramp=` before any noise model is fitted, and
    # then the diagonal weights are all there is. Pass one once it exists: a ramp
    # is a long-wavelength shape and correlated noise is the thing most likely to
    # be mistaken for it, so the two answers can differ materially.
    if covariance is None:
        w = np.asarray(ds["weight"].values, dtype=float)

        def whiten(m):
            return m * (w[:, None] if m.ndim == 2 else w)
    else:
        from .covariance import whitening_factors
        from scipy.linalg import solve_triangular

        track = np.asarray(ds["track"].values, dtype=object)
        factors = whitening_factors(obs, covariance)
        rows = {name: np.flatnonzero(track == name) for name in factors}

        def whiten(m):
            out_m = np.empty_like(np.asarray(m, dtype=float))
            for name, factor in factors.items():
                at = rows[name]
                out_m[at] = solve_triangular(factor, np.asarray(m, float)[at],
                                             lower=True)
            return out_m

    out = {}
    for kind in ("offset", "linear", "quadratic", "cubic"):
        columns, labels = ramp_columns(obs, kind)
        coef, *_ = np.linalg.lstsq(whiten(columns), whiten(d), rcond=None)
        residual = d - columns @ coef
        out[kind] = {
            "variance_reduction": 100.0 * (1.0 - float(residual @ residual) / denom) if denom else 0.0,
            "coefficients": dict(zip(labels, coef.tolist())),
        }
    out["gradient_only"] = out["linear"]["variance_reduction"] - out["offset"]["variance_reduction"]
    out["curvature_only"] = (out["quadratic"]["variance_reduction"]
                             - out["linear"]["variance_reduction"])
    out["cubic_only"] = (out["cubic"]["variance_reduction"]
                         - out["quadratic"]["variance_reduction"])
    return out


#: Fractional moment change below which a ramp degree is treated as harmless.
#: Calibrated against the real six-scene Venezuela sweep, where going from an
#: offset to a quadratic ramp moved peak slip by 4% and Mw from 7.49 to 7.55 --
#: a 23% moment *rise*, not a fall -- while the A061 residual dropped 43%. So a
#: fifth of the moment is comfortably outside what a benign ramp did there.
RAMP_MOMENT_TOLERANCE = 0.20


def ramp_leakage(inversion, slip=None, kinds=("offset", "linear", "quadratic", "cubic")):
    """How much of the fault's own signal each ramp degree *could* absorb.

    The companion to :func:`ramp_content`, and the half it cannot do. That
    function asks how much of the **data** a polynomial explains, which overstates
    the ramp badly because nothing in it has a fault model. This one asks the
    complementary question -- how much of the **earthquake** the polynomial can
    reproduce -- which is what actually decides whether the two compete:

    ``leakage = ||P_ramp (G s)|| / ||G s||``

    with ``P_ramp`` the least-squares projector onto the weighted ramp columns.
    Zero means the ramp is orthogonal to the coseismic field and cannot touch it
    whatever its coefficients do; one means the ramp reproduces it exactly, and
    since the ramp is unregularised while the slip is smoothed and bounded, the
    ramp then wins.

    ⚠️ **Read the marginal numbers, not the totals.** A per-track *offset* always
    absorbs that track's mean coseismic displacement -- an unwrapped
    interferogram has an arbitrary constant, so that leakage is unavoidable and
    not a defect. ``curvature_only`` is what the quadratic terms add over the
    linear ones, and that is the number the question "is the quadratic term
    eating the earthquake?" is really about.

    ``slip`` is a :class:`SlipModel`, a parameter vector, or ``None`` for uniform
    unit strike-slip -- a generic earthquake shape, so the answer is a property of
    the *geometry and aperture* rather than of one solved model. Costs one
    matrix-vector product plus a tiny least-squares per degree; no solve.

    ⚠️ Leakage says the ramp **can**, not that it **did**. Use
    :func:`ramp_sweep` for that -- high leakage with a stable moment means the
    data still pins the slip, which is what the real cubic run turned out to do.
    """
    from .regularize import ramp_columns

    n = inversion.basis.n_basis
    if slip is None:
        # Uniform unit strike-slip: a generic coseismic shape, so the number
        # describes the geometry rather than one particular answer.
        parameters = np.concatenate([np.ones(n), np.zeros(n)])
    elif hasattr(slip, "x"):
        parameters = np.asarray(slip.x[:2 * n], dtype=float)
    else:
        parameters = np.asarray(slip, dtype=float).ravel()[:2 * n]

    # Whitened, not merely weighted: leakage is "how much of this slip pattern
    # can the ramp columns absorb", and that projection has to be taken in the
    # metric the solver works in, or the answer describes a fit nobody performs.
    predicted = inversion.whiten(inversion.g[:, :2 * n] @ parameters)
    norm = float(np.linalg.norm(predicted))
    if norm == 0:
        raise ValueError("the slip pattern predicts no displacement to leak")

    track = np.asarray(inversion.obs.ds["track"].values)
    out = {}
    for kind in kinds:
        columns, _ = ramp_columns(inversion.obs, kind)
        design = inversion.whiten(columns)
        coefficients, *_ = np.linalg.lstsq(design, predicted, rcond=None)
        fitted = design @ coefficients
        per_track = {}
        for name in np.unique(track):
            at = track == name
            local = float(np.linalg.norm(predicted[at]))
            per_track[str(name)] = (float(np.linalg.norm(fitted[at]) / local)
                                    if local else float("nan"))
        out[kind] = {"leakage": float(np.linalg.norm(fitted) / norm),
                     "per_track": per_track}

    for name, (a, b) in (("gradient_only", ("offset", "linear")),
                         ("curvature_only", ("linear", "quadratic")),
                         ("cubic_only", ("quadratic", "cubic"))):
        if a in out and b in out:
            out[name] = out[b]["leakage"] - out[a]["leakage"]
    return out


def ramp_sweep(inversion, kinds=("offset", "linear", "quadratic"),
               reference="offset", tolerance=RAMP_MOMENT_TOLERANCE, **solve_kwargs):
    """Solve at each ramp degree and report what it cost the earthquake.

    The decisive test, and the one variance reduction cannot stand in for: extra
    free columns can only *improve* the fit, so a ramp that is quietly absorbing
    coseismic signal shows up as a **better fit with a smaller earthquake**. That
    conjunction is the signature, and it is what this checks -- a moment that
    falls while variance reduction rises. A moment that falls while the fit also
    worsens is simply a worse model, and one that moves either way while the fit
    barely changes is the columns doing their job.

    Affordable because :meth:`SlipInversion.with_ramp` shares the Green's matrix:
    the sweep costs one *solve* per degree, not one assembly.

    Returns an :class:`xarray.Dataset` on a ``ramp`` coordinate carrying
    ``n_ramp``, ``variance_reduction``, ``rms_misfit``, ``max_slip``,
    ``moment_magnitude``, ``converged``, and -- against ``reference`` --
    ``moment_ratio`` and ``vr_change``. A :class:`RuntimeWarning` names any degree
    whose moment falls by more than ``tolerance`` while its variance reduction
    rises.

    ⚠️ An iteration-capped solve makes every number here meaningless, so check
    ``converged`` before reading the rest; raise ``max_iter`` rather than
    interpreting a ``False``.
    """
    import warnings

    import xarray as xr

    kinds = list(kinds)
    if reference not in kinds:
        raise ValueError(f"reference={reference!r} must be one of the swept kinds {kinds}")

    rows, models = [], {}
    for kind in kinds:
        model = inversion.with_ramp(kind).solve(**solve_kwargs)
        models[kind] = model
        rows.append((model.inversion.n_ramp, model.variance_reduction, model.rms_misfit,
                     float(np.abs(model.element_slip).max()), model.moment_magnitude,
                     model.converged))

    rows = np.array(rows, dtype=float)
    base = kinds.index(reference)
    # Moment from magnitude rather than a second `moment()` call, so the ratio is
    # exactly consistent with the Mw column beside it whatever rigidity was used.
    moment_ratio = 10.0 ** (1.5 * (rows[:, 4] - rows[base, 4]))
    vr_change = rows[:, 1] - rows[base, 1]

    out = xr.Dataset(
        {"n_ramp": ("ramp", rows[:, 0].astype(int)),
         "variance_reduction": ("ramp", rows[:, 1]),
         "rms_misfit": ("ramp", rows[:, 2]),
         "max_slip": ("ramp", rows[:, 3]),
         "moment_magnitude": ("ramp", rows[:, 4]),
         "converged": ("ramp", rows[:, 5].astype(bool)),
         "moment_ratio": ("ramp", moment_ratio),
         "vr_change": ("ramp", vr_change)},
        coords={"ramp": kinds},
    )
    out.attrs.update(reference=reference, tolerance=float(tolerance))

    eating = [k for i, k in enumerate(kinds)
              if moment_ratio[i] < 1.0 - tolerance and vr_change[i] > 0.0]
    if eating:
        warnings.warn(
            f"ramp={eating} fit better than {reference!r} while recovering "
            f"{', '.join('%.0f%%' % (100 * moment_ratio[kinds.index(k)]) for k in eating)} "
            "of its moment -- a better fit with a smaller earthquake is what a ramp "
            "absorbing coseismic signal looks like, and variance reduction cannot "
            "tell you otherwise. Check ramp_leakage()'s marginal terms and prefer "
            "the lower degree unless something independent supports the ramp.",
            RuntimeWarning, stacklevel=2,
        )
    out.attrs["suspect"] = eating
    return out
