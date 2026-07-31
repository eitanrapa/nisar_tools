"""Adaptive downsampling of a :class:`~nisar_tools.los.LOSStack` to point observations.

An unwrapped NISAR scene has millions of pixels and a fault model has thousands
of parameters, so the inversion is fed a decimated set of samples -- dense where
the displacement field varies, sparse where it is smooth. That is a quadtree: a
cell splits while the displacement inside it is rougher than a threshold, and
what survives is one representative sample per cell.

The reference implementation recurses over unstructured ``(x, y, value)`` lists.
Here the input is a *regular raster*, which allows something both faster and
exactly reproducible: cumulative sums of the valid mask and of the values let any
candidate cell's count, mean and variance be read in constant time, and the
recursion walks integer index rectangles instead of copying coordinate arrays.
Because a cell *is* an index rectangle, re-reading the look vectors over exactly
the same pixels is trivially consistent -- in the reference that consistency has
to be arranged by passing the cell index lists back in.

Samples too close to the fault trace are dropped rather than kept: a dislocation
solution is singular on the fault surface, so a Green's function evaluated on the
trace is not finite (see :func:`nisar_tools.slip.greens._require_finite`), and
unwrapped InSAR is decorrelated there anyway.
"""

import numpy as np
import xarray as xr

from .._base import open_stage


class Observations:
    """Downsampled line-of-sight samples with their look vectors and weights.

    Wraps an :class:`xarray.Dataset` on a single ``obs`` dimension carrying
    ``x``, ``y`` (local frame metres), ``los`` (metres, positive toward the
    sensor), ``look_e``/``look_n``/``look_u`` (target->sensor unit vector),
    ``std`` (the within-cell scatter), ``weight`` and ``track``.
    """

    STAGE = "slip_observations"

    def __init__(self, ds):
        self.ds = ds

    # -- construction ------------------------------------------------------
    @classmethod
    def from_los(cls, los_stack, pair=0, name="track", frame=None, trace=None,
                 rms_min=0.005, width_min=1000.0, width_max=20000.0,
                 min_valid_fraction=0.5, exclude_within=0.0, stat="mean",
                 field=None, refine_within=0.0):
        """Quadtree-downsample one pair of a :class:`~nisar_tools.los.LOSStack`.

        ``rms_min`` is the within-cell standard deviation (metres) below which a
        cell stops splitting; ``width_min`` and ``width_max`` bound the cell size
        in metres. ``min_valid_fraction`` drops cells that are mostly outside the
        swath. ``exclude_within`` drops samples closer than that many metres to
        ``trace`` -- required, since the Green's functions are singular on the
        fault. ``frame`` defaults to one centred on the trace, or on the scene.

        The stack's ``los`` is taken as-is: it is already positive toward the
        sensor, because :func:`~nisar_tools.geometry.phase_to_los` applied
        ``sign`` when the stack was built. The attribute is validated and carried
        as provenance, **not** re-applied -- doing so squared it, which silently
        undid the very correction ``sign=-1`` exists to make.

        **Model-based sampling.** ``field`` is an optional ``(ny, nx)`` array that
        decides *where* to split, in place of the observed displacement. Passing a
        model's predicted LOS (see :func:`nisar_tools.slip.predict.predicted_los`)
        is the Wang & Fialko (2015) scheme: the observed field's within-cell
        scatter does not shrink as a cell shrinks, so once ``rms_min`` is below the
        noise the recursion cannot stop on information and simply runs down to
        ``width_min`` -- on real scenes that produced tens of thousands of samples
        of mostly atmosphere. A predicted field is noise-free by construction, so
        cells split on signal. What is *reduced* is always the observed data:
        ``field`` chooses the cells, and the cells are then filled from
        ``los``, exactly as the paper uses "the bounding coordinates of each
        resolution cell" to average the real interferogram.

        ``refine_within`` forces cells within that many metres of ``trace`` to keep
        splitting to ``width_min`` whatever ``field`` says there. It exists only for
        the model-based path and matters more than it looks: an initial model with
        little shallow slip predicts a smooth near-field, so the quadtree would
        coarsen precisely where shallow slip needs constraining, and the next
        iteration is then free to invent it. The paper keeps "a relatively dense
        sampling around the fault trace ... through all iterations" for this reason.

        ⚠️ ``rms_min`` means something different under ``field``: a threshold on
        model curvature, not a noise level, so
        :func:`~nisar_tools.slip.diagnostics.scene_report`'s recommendation does
        not apply -- use :func:`~nisar_tools.slip.diagnostics.model_rms_min`.
        """
        if stat not in ("mean", "median"):
            raise ValueError("stat must be 'mean' or 'median'")
        if exclude_within > 0 and trace is None:
            raise ValueError("exclude_within needs a trace to measure distance from")
        if refine_within > 0 and trace is None:
            raise ValueError("refine_within needs a trace to measure distance from")

        ds = los_stack.ds
        epsg = stack_epsg(ds)
        if frame is None:
            frame = trace.local_frame() if trace is not None else _frame_for(ds)

        los = np.asarray(ds["los"].isel(pair=pair).values, dtype=float)
        # Validated and recorded below as provenance only. `phase_to_los` already
        # applied it, so multiplying again would square it back to +1 and return
        # the uncorrected field.
        sign = int(ds.attrs.get("sign", 1))
        if sign not in (1, -1):
            raise ValueError(f"Unexpected LOSStack sign attribute {sign!r}")

        look = np.stack([
            np.asarray(ds["los_east"].values, dtype=float),
            np.asarray(ds["los_north"].values, dtype=float),
            np.asarray(ds["los_up"].values, dtype=float),
        ])

        valid = np.isfinite(los) & np.all(np.isfinite(look), axis=0)

        split_on = los
        if field is not None:
            split_on = np.asarray(
                field.values if hasattr(field, "values") else field, dtype=float
            )
            if split_on.shape != los.shape:
                raise ValueError(
                    f"field has shape {split_on.shape}, but the stack's grid is "
                    f"{los.shape}"
                )
            # A pixel with no prediction cannot be judged, so it cannot vote on
            # where to split -- and reducing it would mix modelled and observed
            # footprints inside one cell.
            valid = valid & np.isfinite(split_on)

        if not valid.any():
            raise ValueError("No pixel has both a finite displacement and a look vector")

        x_native = np.asarray(ds["x"].values, dtype=float)
        y_native = np.asarray(ds["y"].values, dtype=float)
        dx = abs(float(np.diff(x_native)[0]))
        dy = abs(float(np.diff(y_native)[0]))

        refine = None
        if refine_within > 0:
            gx, gy = np.meshgrid(x_native, y_native)
            fx_all, fy_all = _to_frame(gx.ravel(), gy.ravel(), frame, epsg)
            refine = (trace.distance(fx_all, fy_all, frame)
                      .reshape(los.shape) <= refine_within)

        cells = _quadtree_cells(
            split_on, valid,
            rms_min=rms_min,
            min_rows=max(1, int(round(width_min / dy))),
            min_cols=max(1, int(round(width_min / dx))),
            max_rows=max(1, int(round(width_max / dy))),
            max_cols=max(1, int(round(width_max / dx))),
            min_valid_fraction=min_valid_fraction,
            refine=refine,
        )
        if not cells:
            raise ValueError(
                "Quadtree produced no samples; loosen rms_min or min_valid_fraction"
            )

        rows = _reduce_cells(cells, los, look, valid, x_native, y_native, stat)

        # Native grid metres -> the shared local frame. Averaging in the native
        # projection and transforming once is exact to well under a metre over a
        # cell, and avoids transforming every pixel of the raster.
        fx, fy = _to_frame(rows["x"], rows["y"], frame, epsg)

        keep = np.ones(fx.size, dtype=bool)
        if exclude_within > 0:
            keep = trace.distance(fx, fy, frame) > exclude_within
            if not keep.any():
                raise ValueError(
                    f"exclude_within={exclude_within} m removed every sample"
                )

        out = xr.Dataset(
            {
                "x": ("obs", fx[keep]),
                "y": ("obs", fy[keep]),
                "los": ("obs", rows["los"][keep]),
                "look_e": ("obs", rows["look_e"][keep]),
                "look_n": ("obs", rows["look_n"][keep]),
                "look_u": ("obs", rows["look_u"][keep]),
                "std": ("obs", rows["std"][keep]),
                "cell_size": ("obs", rows["cell_size"][keep]),
                "weight": ("obs", np.ones(int(keep.sum()))),
                "track": ("obs", np.full(int(keep.sum()), name, dtype=object)),
            }
        )
        quadtree = {
            "rms_min": float(rms_min),
            "width_min": float(width_min),
            "width_max": float(width_max),
            "min_valid_fraction": float(min_valid_fraction),
            "exclude_within": float(exclude_within),
            "stat": stat,
        }
        if field is not None:
            # Folded in only when used, so a data-driven run keeps the params
            # hash it had before model-based sampling existed.
            quadtree["field"] = "model"
            quadtree["refine_within"] = float(refine_within)

        out.attrs.update(
            frame=frame.to_dict(),
            epsg=epsg,
            tracks=[name],
            source_epsg=epsg,
            pair=int(pair),
            sign=sign,
            direction=ds.attrs.get("direction"),
            wavelength=ds.attrs.get("wavelength"),
            quadtree=quadtree,
            n_raw_valid=int(valid.sum()),
        )
        return cls(out)

    @classmethod
    def concat(cls, items, normalize="sqrt_count", weights=None):
        """Combine several tracks into one observation set.

        ``normalize`` sets how a track's sample count feeds its influence:

        ``"sqrt_count"``
            each row scaled by ``1/sqrt(n)``, so a track's *total* contribution to
            the normal equations is independent of how densely it was sampled.
            The default, and almost always what is wanted.
        ``"count"``
            each row scaled by ``1/n``, which is what the reference
            implementation does -- and which means a track with twice as many
            samples gets *half* the influence, since a row's weight enters the
            normal equations squared.
        ``"none"``
            no count normalisation.

        ``weights`` optionally scales each named track on top of that.
        """
        if normalize not in ("sqrt_count", "count", "none"):
            raise ValueError("normalize must be 'sqrt_count', 'count' or 'none'")
        items = list(items)
        if not items:
            raise ValueError("Nothing to concatenate")

        frame = items[0].ds.attrs.get("frame")
        parts = []
        for item in items:
            if item.ds.attrs.get("frame") != frame:
                raise ValueError(
                    "All tracks must be sampled in the same LocalFrame; "
                    "pass frame= consistently to from_los."
                )
            ds = item.ds.copy(deep=True)
            n = ds.sizes["obs"]
            scale = {"sqrt_count": 1.0 / np.sqrt(n), "count": 1.0 / n, "none": 1.0}[normalize]

            # Build the scaled weights as a new array rather than editing in place.
            # ``Dataset.copy()`` is shallow, so an in-place edit would reach back
            # into the caller's Observations -- and passing the same object twice
            # would then apply the scaling to it twice.
            track = ds["track"].values
            factor = np.full(n, scale)
            if weights:
                for track_name in np.unique(track):
                    factor[track == track_name] *= float(weights.get(str(track_name), 1.0))
            ds["weight"] = ("obs", ds["weight"].values * factor)
            parts.append(ds)

        out = xr.concat(parts, dim="obs", combine_attrs="drop_conflicts")
        out.attrs["frame"] = frame
        out.attrs["tracks"] = [t for item in items for t in item.ds.attrs.get("tracks", [])]
        out.attrs["normalize"] = normalize
        if weights:
            out.attrs["track_weights"] = dict(weights)
        return cls(out)

    # -- accessors ---------------------------------------------------------
    @property
    def n(self):
        return int(self.ds.sizes["obs"])

    @property
    def frame(self):
        from .frame import LocalFrame

        return LocalFrame.from_dict(self.ds.attrs["frame"])

    @property
    def tracks(self):
        return [str(t) for t in np.unique(self.ds["track"].values)]

    def track_mask(self, name):
        return self.ds["track"].values == name

    # -- persistence -------------------------------------------------------
    def persist(self, workspace, name=None, overwrite=False, **params):
        """Write to a Zarr stage and return the reopened set.

        Worth caching: this is the one step that reads a multi-gigabyte
        ``LOSStack``, and every later experiment with smoothing or bounds reuses
        its output unchanged.
        """
        name = name or self.STAGE
        full = {
            "stage": name,
            "epsg": self.ds.attrs.get("epsg"),
            "frame": self.ds.attrs["frame"],
            "tracks": self.ds.attrs.get("tracks"),
            "quadtree": self.ds.attrs.get("quadtree"),
            "normalize": self.ds.attrs.get("normalize"),
            **params,
        }
        ds = self.ds.copy()
        # Zarr has no object dtype; track names ride as fixed-width text.
        ds["track"] = ds["track"].astype(str)
        return Observations(workspace.store(name, ds, full, overwrite=overwrite))

    @classmethod
    def from_zarr(cls, path):
        return cls(open_stage(path))

    def __repr__(self):
        return f"<Observations n={self.n} tracks={self.tracks}>"


def stack_epsg(ds):
    """The EPSG code to transform a stack's grid from, or ``None``.

    ``None`` means the stack was resampled onto a lattice in the local frame
    (:mod:`nisar_tools.slip.resample`) and its ``x``/``y`` are already local
    metres. Every helper that projects a stack's coordinates goes through this
    and :func:`_to_frame`, so a frame-gridded stack works everywhere a UTM one
    does rather than raising ``KeyError: 'epsg'`` somewhere downstream.
    """
    if ds.attrs.get("frame") is not None:
        return None
    return int(ds.attrs["epsg"])


def _to_frame(x, y, frame, epsg):
    """Bring native grid coordinates into ``frame``.

    ``epsg is None`` means the stack was already resampled onto a lattice in this
    frame (see :mod:`nisar_tools.slip.resample`), which grids in
    :attr:`~nisar_tools.slip.frame.LocalFrame.local_crs` -- so its ``x``/``y``
    *are* local metres and there is nothing to do. That is also the only available
    answer: the frame's transverse Mercator has no EPSG code to transform from.
    """
    if epsg is None:
        return np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return frame.from_epsg(x, y, epsg)


def _frame_for(ds):
    """A local frame centred on a scene, when no trace was supplied."""
    from pyproj import Transformer

    from .frame import LocalFrame

    if ds.attrs.get("frame") is not None:
        return LocalFrame.from_dict(ds.attrs["frame"])

    x = np.asarray(ds["x"].values, dtype=float)
    y = np.asarray(ds["y"].values, dtype=float)
    t = Transformer.from_crs(f"EPSG:{int(ds.attrs['epsg'])}", "EPSG:4326", always_xy=True)
    lon, lat = t.transform(x.mean(), y.mean())
    return LocalFrame(float(lon), float(lat))


def _quadtree_cells(values, valid, rms_min, min_rows, min_cols,
                    max_rows, max_cols, min_valid_fraction, refine=None):
    """Index rectangles ``(r0, r1, c0, c1)`` surviving the variance split.

    Constant-time per candidate: the count, sum and sum of squares over any
    rectangle come from three cumulative-sum tables, so the recursion never
    touches the pixels themselves.

    The split statistic is the within-cell **standard deviation**. The reference
    implementations describe a *curvature*-based rule (Simons et al. 2002; Fialko
    2004; Wang & Fialko 2015); on a smooth field the two behave alike, and the
    summed-area formulation is what makes this cost 0.05 s on 0.32 M pixels, so
    the difference is recorded rather than removed.

    ``refine`` is an optional boolean mask of pixels that must be resolved to
    ``min_rows``/``min_cols`` regardless of how flat ``values`` is over them. It
    rides on a fourth table, so the test stays O(1) per candidate.
    """
    v = np.where(valid, values, 0.0)
    m = valid.astype(np.float64)
    s0 = _sat(m)
    s1 = _sat(v)
    s2 = _sat(v * v)
    s3 = None if refine is None else _sat(np.asarray(refine, dtype=np.float64))

    def block(sat, r0, r1, c0, c1):
        return sat[r1, c1] - sat[r0, c1] - sat[r1, c0] + sat[r0, c0]

    ny, nx = values.shape
    out = []
    stack = [(0, ny, 0, nx)]
    while stack:
        r0, r1, c0, c1 = stack.pop()
        rows, cols = r1 - r0, c1 - c0
        count = block(s0, r0, r1, c0, c1)

        rough = False
        if count > 1:
            mean = block(s1, r0, r1, c0, c1) / count
            var = block(s2, r0, r1, c0, c1) / count - mean * mean
            rough = np.sqrt(max(var, 0.0)) > rms_min
        if s3 is not None and block(s3, r0, r1, c0, c1) > 0:
            rough = True

        # Each axis is halved only if the halves stay above that axis's minimum
        # width -- decided per axis, not once for the cell. Deciding jointly lets
        # a cell that is splittable along strike also be halved across it, taking
        # the short side below width_min.
        oversize_rows = rows > max_rows
        oversize_cols = cols > max_cols
        want = rough or oversize_rows or oversize_cols
        split_rows = oversize_rows or (want and rows >= 2 * min_rows)
        split_cols = oversize_cols or (want and cols >= 2 * min_cols)

        if (split_rows and rows >= 2) or (split_cols and cols >= 2):
            rm = r0 + rows // 2
            cm = c0 + cols // 2
            row_spans = ((r0, rm), (rm, r1)) if split_rows and rows >= 2 else ((r0, r1),)
            col_spans = ((c0, cm), (cm, c1)) if split_cols and cols >= 2 else ((c0, c1),)
            for rr in row_spans:
                for cc in col_spans:
                    stack.append((rr[0], rr[1], cc[0], cc[1]))
            continue

        if count > 0 and count >= min_valid_fraction * rows * cols:
            out.append((r0, r1, c0, c1))
    return out


def _reduce_cells(cells, los, look, valid, x_native, y_native, stat):
    """Collapse each surviving cell to one sample.

    Every quantity is averaged over the *same* valid pixels -- the cell is an
    index rectangle, so the data, the look vector and the coordinates cannot
    disagree about which pixels they came from.
    """
    n = len(cells)
    out = {k: np.empty(n) for k in
           ("x", "y", "los", "look_e", "look_n", "look_u", "std", "cell_size")}
    dx = abs(float(np.diff(x_native)[0]))
    dy = abs(float(np.diff(y_native)[0]))

    for i, (r0, r1, c0, c1) in enumerate(cells):
        m = valid[r0:r1, c0:c1]
        d = los[r0:r1, c0:c1][m]
        out["los"][i] = np.median(d) if stat == "median" else d.mean()
        out["std"][i] = d.std() if d.size > 1 else 0.0

        rr, cc = np.nonzero(m)
        out["x"][i] = x_native[c0:c1][cc].mean()
        out["y"][i] = y_native[r0:r1][rr].mean()
        for k, name in enumerate(("look_e", "look_n", "look_u")):
            out[name][i] = look[k, r0:r1, c0:c1][m].mean()
        out["cell_size"][i] = 0.5 * ((r1 - r0) * dy + (c1 - c0) * dx)

    # Averaging three components of a unit vector shortens it slightly; renormalise
    # so the projection stays a direction cosine and `los_up == cos(incidence)`
    # continues to hold.
    norm = np.sqrt(out["look_e"] ** 2 + out["look_n"] ** 2 + out["look_u"] ** 2)
    norm[norm == 0] = 1.0
    for name in ("look_e", "look_n", "look_u"):
        out[name] /= norm
    return out


def _sat(a):
    """Summed-area table with a zero first row and column."""
    out = np.zeros((a.shape[0] + 1, a.shape[1] + 1), dtype=np.float64)
    np.cumsum(np.cumsum(a, axis=0), axis=1, out=out[1:, 1:])
    return out
