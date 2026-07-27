"""The :class:`GSLCStack`: a set of co-registered GSLCs on a common grid.

Backed by a lazy ``xarray`` Dataset with a single ``slc`` variable of dims
``(time, y, x)``. Cropping and merging stay lazy; interferogram formation
produces an :class:`~nisar_tools.interferogram.InterferogramStack`.
"""

import numpy as np
import rioxarray  # noqa: F401
import xarray as xr

from . import geo
from ._base import SPATIAL_CHUNK, RasterStackMixin, open_stage, wrapped_phase
from .interferogram import InterferogramStack, make_pairs


class GSLCStack(RasterStackMixin):
    """A time stack of aligned GSLCs."""

    STAGE = "slc_stack"
    GRD_STACK_DIM = "time"
    # An SLC's absolute phase is not geophysically meaningful on its own, so
    # ``.grd`` export defaults to amplitude only; pass ``fields=["phase"]`` (or
    # ``["amplitude", "phase"]``) for the phase too.
    GRD_DEFAULT_FIELDS = ("amplitude",)

    def __init__(self, ds):
        self.ds = ds

    # -- construction ------------------------------------------------------
    @classmethod
    def from_gslcs(cls, gslcs, bbox=None):
        """Build a stack from open :class:`GSLC` objects, optionally cropped.

        ``bbox`` is ``(lon_min, lon_max, lat_min, lat_max)``. All granules
        must share one EPSG and pass direction and lie on the same grid (e.g.
        the same frame across dates).
        """
        if len(gslcs) == 0:
            raise ValueError("Need at least one GSLC")

        epsgs = {g.epsg for g in gslcs}
        if len(epsgs) > 1:
            raise ValueError(
                f"GSLCs span multiple EPSG codes: {sorted(epsgs)}. A stack "
                "shares one grid; build one stack per frame and stitch them "
                "with GSLCStack.merge, which warps across zones."
            )
        directions = {g.direction for g in gslcs}
        if len(directions) > 1:
            raise ValueError(f"GSLCs span multiple pass directions: {directions}")

        arrays = []
        times = []
        for i, g in enumerate(gslcs):
            da = g.crop(*bbox) if bbox is not None else g.data
            t = g.datetime if g.datetime is not None else np.datetime64(i, "s")
            arrays.append(da)
            times.append(t)

        order = np.argsort(np.asarray(times))
        arrays = [arrays[i] for i in order]
        times = [times[i] for i in order]

        slc = xr.concat(arrays, dim="time", join="exact")
        slc = slc.assign_coords(time=("time", np.asarray(times)))

        ds = slc.to_dataset(name="slc")
        ds = ds.rio.write_crs(f"EPSG:{gslcs[0].epsg}")
        ds.attrs.update(
            epsg=int(gslcs[0].epsg),
            direction=gslcs[0].direction,
            x_spacing=float(gslcs[0].x_spacing),
            y_spacing=float(gslcs[0].y_spacing),
        )
        return cls(ds)

    @classmethod
    def from_zarr(cls, path):
        return cls(open_stage(path))

    # -- lazy operations ---------------------------------------------------
    def merge(self, other, resampling="bilinear", time_tolerance=600.0):
        """Merge an adjacent-frame stack (same acquisitions) onto the union grid.

        Acquisitions are paired by nearest time within ``time_tolerance``
        seconds — frames along one pass share a track but not an exact
        ``zeroDopplerStartTime`` — and the merged stack keeps ``self``'s
        times. When ``other`` is gridded in a different CRS (a track crossing
        a UTM zone boundary), it is first warped onto ``self``'s grid, one
        date per dask task, resampling real and imaginary parts with
        ``resampling`` (``"nearest"`` preserves exact sample values).

        Both grids sit on the same lattice, so the union is reached by padding
        each side onto it rather than reindexing, and both are re-chunked onto
        the canonical grid before the ``fillna`` — otherwise the two crops,
        which start mid-chunk at their own phases, would be split into the union
        of their chunk boundaries. Self takes precedence where the two overlap.

        The union grid keeps NaN where neither stack has data — a cross-zone
        warp always leaves rotated nodata wedges — exactly as with the NaN fill
        of a single granule. :meth:`form_interferograms` is NaN-aware, so that
        fill neither spreads into nor erodes the valid data; :meth:`crop` is
        available at every stage if a ragged edge is worth trimming afterwards.
        """
        if (
            self.direction is not None
            and other.direction is not None
            and self.direction != other.direction
        ):
            raise ValueError("Cannot merge stacks with different pass directions")

        if min(self.sizes["y"], self.sizes["x"],
               other.sizes["y"], other.sizes["x"]) < 2:
            raise ValueError(
                "Merging needs stacks with at least 2 pixels along y and x"
            )

        times = np.asarray(self.ds["time"].values)
        mapped = _match_times(
            times, np.asarray(other.ds["time"].values), time_tolerance
        )
        other_slc = other.ds["slc"].assign_coords(time=("time", times[mapped]))

        if self.epsg != other.epsg:
            other_slc = self._warp_onto_grid(other_slc, other.epsg, resampling)
        else:
            _check_same_lattice(self.x, self.y, other_slc)

        chunks = {"time": 1, "y": SPATIAL_CHUNK, "x": SPATIAL_CHUNK}
        this_slc = self.ds["slc"]
        union_y = _union_lattice(self.y, other_slc["y"].values)
        union_x = _union_lattice(self.x, other_slc["x"].values)
        a = _pad_onto(this_slc, union_y, union_x)
        b = _pad_onto(other_slc, union_y, union_x)
        merged = a.chunk(chunks).fillna(b.chunk(chunks))
        ds = merged.to_dataset(name="slc")
        ds = ds.rio.write_crs(f"EPSG:{self.epsg}")
        ds.attrs.update(self.ds.attrs)
        return GSLCStack(ds)

    def _warp_onto_grid(self, slc, src_epsg, resampling):
        """Lazily warp a ``(time, y, x)`` DataArray onto this stack's lattice."""
        tx, ty = warp_target_lattice(
            self.x, self.y, slc["x"].values, slc["y"].values, src_epsg, self.epsg
        )
        return warp_onto_lattice(slc, tx, ty, src_epsg, self.epsg, resampling)

    def form_interferograms(
        self, pairs="sequential", looks=5, downsample=True,
        convolution="Uniform", nan_aware=True, min_valid_fraction=0.5,
    ):
        """Form an :class:`InterferogramStack` from pairs of acquisitions.

        ``nan_aware`` (default) multilooks over the valid samples only, so the
        NaN left by the swath edge or by :meth:`merge` does not spread across
        the filter footprint. ``min_valid_fraction`` is how much of an output
        pixel's filter weight must come from valid input for it to be kept.
        See :func:`nisar_tools._kernels.igram_coherence`.
        """
        return InterferogramStack.from_slc_stack(
            self,
            pairs=pairs,
            looks=looks,
            downsample=downsample,
            convolution=convolution,
            nan_aware=nan_aware,
            min_valid_fraction=min_valid_fraction,
        )

    # -- persistence -------------------------------------------------------
    def persist(self, workspace, name=None, overwrite=False, **params):
        """Write the stack to the workspace and return the reopened lazy stack."""
        name = name or self.STAGE
        ds = self.ds.chunk(self.disk_chunks("time"))
        full_params = {"stage": name, "epsg": self.epsg, **params}
        reopened = workspace.store(name, ds, full_params, overwrite=overwrite)
        return GSLCStack(reopened)

    def make_pairs(self, pairs="sequential"):
        """Resolve a pairs spec into an explicit list against this stack."""
        return make_pairs(pairs, self.sizes["time"])

    # -- export ------------------------------------------------------------
    def _grd_specs(self):
        """``amplitude`` and (wrapped) ``phase`` of the complex SLC, per time."""
        slc = self.ds["slc"]
        return [
            ("amplitude", np.abs(slc), True),
            ("phase", wrapped_phase(slc), True),
        ]

    def __repr__(self):
        s = self.sizes
        return (
            f"<GSLCStack EPSG:{self.epsg} "
            f"time={s.get('time')} y={s.get('y')} x={s.get('x')}>"
        )


def _union_lattice(ref, other):
    """Coordinates covering both axes, on ``ref``'s lattice and direction.

    The two grids are known to share a lattice, so their union needs no
    reindexing — it is ``ref`` extended by whole steps until it reaches
    ``other``. Keeping ``ref``'s step sign means the merged stack comes out
    oriented like this one (y running with the pass direction) for free.
    """
    spacing = float(ref[1] - ref[0])
    lo = min(float(np.min(ref)), float(np.min(other)))
    hi = max(float(np.max(ref)), float(np.max(other)))
    return _extend_lattice(ref, spacing, lo, hi)


def _pad_onto(arr, union_y, union_x, fill=np.nan):
    """Place a ``(stack, y, x)`` array on the union grid, ``fill`` elsewhere.

    Padding rather than ``xr.align(join="outer")``: both grids sit on the same
    lattice, so the union is a pure offset, and reindexing to reach it costs a
    per-element boolean mask (xarray builds one to mark the missing positions)
    whose broadcast both fragments the chunk grid and is what emitted dask's
    "Increasing number of chunks by factor of N" warning. Padding is a pure
    graph operation with no mask and no fancy indexing.

    The leading (stack) axis -- ``time`` or ``pair`` -- is not padded. ``fill``
    is the value for pixels outside ``arr``: ``NaN`` for phase / coherence, ``0``
    for an integer label layer (a connected-component map) that cannot hold NaN.
    """
    stack_dim = arr.dims[0]
    step_y = float(union_y[1] - union_y[0])
    step_x = float(union_x[1] - union_x[0])
    off_y = int(round((float(arr["y"].values[0]) - float(union_y[0])) / step_y))
    off_x = int(round((float(arr["x"].values[0]) - float(union_x[0])) / step_x))
    ny, nx = arr.sizes["y"], arr.sizes["x"]

    pad = (
        (0, 0),
        (off_y, len(union_y) - off_y - ny),
        (off_x, len(union_x) - off_x - nx),
    )
    if min(p for lohi in pad for p in lohi) < 0:  # pragma: no cover - guarded upstream
        raise ValueError("Cannot merge: grids do not share a lattice")

    if _is_dask(arr.data):
        import dask.array as dask_array

        data = dask_array.pad(arr.data, pad, mode="constant", constant_values=fill)
    else:
        data = np.pad(arr.data, pad, mode="constant", constant_values=fill)

    return xr.DataArray(
        data.astype(arr.dtype, copy=False),
        dims=arr.dims,
        coords={stack_dim: arr[stack_dim].values, "y": union_y, "x": union_x},
        name=arr.name,
    )


def _is_dask(arr):
    return type(arr).__module__.split(".")[0] == "dask"


def _match_times(ref_times, other_times, tolerance):
    """Pair each entry of ``other_times`` with its nearest ``ref_times`` entry.

    Adjacent frames on one pass are acquired seconds apart, so timestamps
    never match exactly; a one-to-one nearest match within ``tolerance``
    (seconds) is required instead. Returns indices into ``ref_times``, one
    per entry of ``other_times``.
    """
    if len(ref_times) != len(other_times):
        raise ValueError(
            "Cannot merge stacks with different numbers of acquisitions: "
            f"{len(ref_times)} vs {len(other_times)}"
        )
    if not isinstance(tolerance, np.timedelta64):
        tolerance = np.timedelta64(int(round(float(tolerance) * 1e9)), "ns")

    mapped = []
    for t in other_times:
        offsets = np.abs(ref_times - t)
        i = int(np.argmin(offsets))
        if offsets[i] > tolerance:
            raise ValueError(
                f"No acquisition within {tolerance} of {t}: the stacks do not "
                "cover the same dates (or increase time_tolerance)"
            )
        mapped.append(i)
    if len(set(mapped)) != len(mapped):
        raise ValueError(
            "Acquisitions in the two stacks do not pair one-to-one within "
            "the time tolerance"
        )
    return np.asarray(mapped)


def _check_same_lattice(x_ref, y_ref, slc):
    """Require a same-CRS grid to lie on the reference lattice.

    The outer join in :meth:`GSLCStack.merge` matches coordinates exactly, so
    a grid offset by a sub-pixel amount would silently interleave
    near-duplicate coordinates into a NaN checkerboard instead of merging.
    """
    for name, ref, vals in (("x", x_ref, slc["x"].values),
                            ("y", y_ref, slc["y"].values)):
        d_ref = float(ref[1] - ref[0])
        d = float(vals[1] - vals[0])
        if not np.isclose(d, d_ref, rtol=1e-6, atol=0.0):
            raise ValueError(
                f"Cannot merge: {name} spacings differ ({d} vs {d_ref})"
            )
        off = (float(vals[0]) - float(ref[0])) / d_ref
        if abs(off - round(off)) > 1e-3:
            raise ValueError(
                f"Cannot merge: {name} grids are offset by a sub-pixel amount "
                f"({abs(off - round(off)):.3g} px); the granules do not share "
                "a lattice"
            )


def _extend_lattice(ref, spacing, lo, hi):
    """Integer-step extension of a reference lattice to cover ``[lo, hi]``.

    Steps of ``spacing`` (its sign sets the direction) anchored at
    ``ref[0]``. Entries that land inside ``ref``'s own index range reuse
    ``ref``'s exact values, so a later outer join aligns them instead of
    interleaving floating-point near-duplicates.
    """
    ref = np.asarray(ref, dtype=np.float64)
    k1 = (lo - ref[0]) / spacing
    k2 = (hi - ref[0]) / spacing
    k = np.arange(int(np.floor(min(k1, k2))), int(np.ceil(max(k1, k2))) + 1)
    coords = ref[0] + k * spacing
    inside = (k >= 0) & (k < len(ref))
    coords[inside] = ref[k[inside]]
    return coords


def _warp_block(block, src_transform=None, src_epsg=None, dst_transform=None,
                dst_epsg=None, dst_shape=None, resampling=None):
    """``map_blocks`` kernel: warp one ``(1, ny, nx)`` slice onto the target."""
    out = geo.warp_to_grid(
        block[0], src_transform, src_epsg, dst_transform, dst_epsg,
        dst_shape, resampling,
    )
    return out[None].astype(block.dtype, copy=False)


def warp_target_lattice(x_ref, y_ref, sx, sy, src_epsg, dst_epsg):
    """Lattice on ``x_ref``/``y_ref``'s grid, extended to cover a source footprint.

    Keeping the reference spacing *and grid phase* is what lets the union in a
    merge be reached by padding: warping onto an arbitrary grid would interleave
    floating-point near-duplicates along the seam instead of lining the overlap up
    exactly.
    """
    sdx = float(sx[1] - sx[0])
    sdy = float(sy[1] - sy[0])
    x_min, x_max, y_min, y_max = geo.transform_native_bbox(
        min(sx[0], sx[-1]) - abs(sdx) / 2,
        max(sx[0], sx[-1]) + abs(sdx) / 2,
        min(sy[0], sy[-1]) - abs(sdy) / 2,
        max(sy[0], sy[-1]) + abs(sdy) / 2,
        src_epsg,
        dst_epsg,
    )
    tx = _extend_lattice(x_ref, float(x_ref[1] - x_ref[0]), x_min, x_max)
    ty = _extend_lattice(y_ref, float(y_ref[1] - y_ref[0]), y_min, y_max)
    return tx, ty


def warp_onto_lattice(field, tx, ty, src_epsg, dst_epsg, resampling="bilinear"):
    """Lazily warp a ``(stack, y, x)`` DataArray onto the ``(ty, tx)`` lattice.

    One slice per dask task warping a whole frame, so peak memory per task is one
    source frame plus one target frame. The stack dimension is taken from
    ``field.dims[0]``, so this serves a ``time`` axis and a ``pair`` axis alike.

    **Integer layers are warped as float32 with nearest resampling and refilled
    with 0.** A connected-component or subswath label must never be blended -- an
    interpolated label is not a label -- and the reprojection's out-of-coverage
    fill is NaN, which no integer dtype can hold. float32 represents integers
    exactly to 2**24, far above any label these products carry. Same reasoning as
    :func:`nisar_tools.geo.write_grd`.
    """
    geo.resampling_from_name(resampling)  # fail now, not at compute time

    stack_dim = field.dims[0]
    sx, sy = field["x"].values, field["y"].values
    integer = np.issubdtype(field.dtype, np.integer)
    work = field.astype(np.float32) if integer else field
    if integer:
        resampling = "nearest"

    src = work.chunk({stack_dim: 1, "y": -1, "x": -1}).data
    warped = src.map_blocks(
        _warp_block,
        dtype=src.dtype,
        chunks=(1, len(ty), len(tx)),
        src_transform=geo.grid_transform(sx, sy),
        src_epsg=src_epsg,
        dst_transform=geo.grid_transform(tx, ty),
        dst_epsg=dst_epsg,
        dst_shape=(len(ty), len(tx)),
        resampling=resampling,
    )
    out = xr.DataArray(
        warped,
        dims=(stack_dim, "y", "x"),
        coords={stack_dim: field[stack_dim].values, "y": ty, "x": tx},
        name=field.name,
    )
    if integer:
        out = out.fillna(0).astype(field.dtype)
    return out.rio.write_crs(f"EPSG:{int(dst_epsg)}")
