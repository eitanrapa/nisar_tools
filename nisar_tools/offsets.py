"""The :class:`PixelOffsetStack`: sub-pixel offsets by amplitude cross-correlation.

Backed by a lazy ``xarray`` Dataset with ``x_offset``, ``y_offset`` and
``correlation`` (all float32) variables of dims ``(pair, y, x)``, on a *coarse*
lattice of correlation locations rather than the full-resolution grid. Each pair
carries ``ref_time`` and ``sec_time`` auxiliary coordinates (plain coords rather
than a MultiIndex, which would not serialise to Zarr), and ``x_pixel``/``y_pixel``
give the 0-based index into the *source* grid that each axis's locations sit at.

This is a port of GMTSAR's ``xcorr``; :func:`nisar_tools._kernels.pixel_offsets`
holds the numerics and records where it departs from that reference.

**What the offsets mean here.** ``xcorr`` runs on radar-geometry SLCs, where the
two images are unaligned and the offsets are a coregistration solution in range
and azimuth. A NISAR GSLC is *geocoded*: the pair already shares one map grid, so
there is nothing to coregister and what the correlation measures is the shift of
the backscatter pattern in map x/y between the two dates -- ground displacement,
which is the measurement that survives where the interferogram decorrelates. The
fields are named for the map axes for that reason, and
:attr:`~PixelOffsetStack.east_offset` / :attr:`~PixelOffsetStack.north_offset`
convert them to metres.
"""

import warnings
from pathlib import Path

import numpy as np
import rioxarray  # noqa: F401
import xarray as xr

from . import _kernels
from ._base import RasterStackMixin, open_stage
from .interferogram import make_pairs

#: Default gap between correlation locations, in pixels. Half the default
#: correlation window, so adjacent estimates share half their data -- the usual
#: posting for an offset-tracking map.
DEFAULT_STEP = 64

#: ``fitoffset.csh`` refuses to fit fewer than this many points.
MIN_USEFUL_ROWS = 8


class PixelOffsetStack(RasterStackMixin):
    """A stack of sub-pixel offset fields with their correlation."""

    STAGE = "offsets"
    #: The three measured fields; ``east_offset``/``north_offset`` (the same
    #: numbers in metres) are available from :meth:`to_grd` on request.
    GRD_DEFAULT_FIELDS = ("x_offset", "y_offset", "correlation")

    def __init__(self, ds):
        self.ds = ds

    # -- construction ------------------------------------------------------
    @classmethod
    def from_slc_stack(
        cls, stack, pairs="sequential", step=None, nx=None, ny=None,
        nx_corr=128, ny_corr=128, xsearch=64, ysearch=64, x_shift=0, y_shift=0,
        interp_factor=16, subpixel_window=16, subpixel=True, oversample=1,
        min_valid_fraction=0.5, locations_per_tile=16,
    ):
        """Cross-correlate pairs of acquisitions on a coarse lattice.

        The correlation locations are laid out either by ``step`` (the gap
        between them in pixels, the default at :data:`DEFAULT_STEP`) or by
        ``nx``/``ny`` (how many of them, GMTSAR's ``-nx``/``-ny``); the two are
        mutually exclusive. Either way the lattice is exactly uniform and inset by
        half a correlation window, so every patch lies wholly inside the raster
        and the result is a georeferenced raster in its own right --
        :meth:`~nisar_tools._base.RasterStackMixin.crop`,
        :meth:`~nisar_tools._base.RasterStackMixin.to_grd` and
        :meth:`persist` all work on it unchanged.

        ``nx_corr``/``ny_corr`` size the correlation template and
        ``xsearch``/``ysearch`` the search half-width, both in pixels; the patch
        that gets transformed is ``ny_corr + 2*ysearch`` by ``nx_corr + 2*xsearch``.
        The defaults are GMTSAR's, which at NISAR GSLC's ~10 m posting means a
        1.3 km template searched over +/-640 m. ``x_shift``/``y_shift`` centre the
        search on a known bulk displacement (GMTSAR's PRM ``rshift``/``ashift``);
        they are 0 here, because a geocoded pair is already aligned.

        See :func:`nisar_tools._kernels.pixel_offsets` for the sub-pixel
        refinement (``interp_factor``, ``subpixel_window``, ``subpixel``), the
        input oversampling (``oversample``) and the invalid-sample rule
        (``min_valid_fraction``). Note ``subpixel_window`` defaults to 16 rather
        than GMTSAR's 8, which is measured there to cut a systematic sub-pixel
        bias 3.8-fold at no cost in scatter. ``locations_per_tile`` is a dask
        granularity dial only and does not change the result.
        """
        if step is not None and (nx is not None or ny is not None):
            raise ValueError(
                "give either step= (the gap between locations, in pixels) or "
                "nx=/ny= (how many locations), not both"
            )
        if (nx is None) != (ny is None):
            raise ValueError(
                f"give both nx= and ny= or neither, got nx={nx!r}, ny={ny!r}"
            )
        if step is None and nx is None:
            step = DEFAULT_STEP

        pair_list = make_pairs(pairs, stack.sizes["time"])
        if len(pair_list) == 0:
            raise ValueError("No pairs to correlate (need >= 2 acquisitions)")

        slc = stack.ds["slc"]
        ref = slc.isel(time=[i for i, _ in pair_list])
        sec = slc.isel(time=[j for _, j in pair_list])
        ref_times = np.asarray(ref["time"].values)
        sec_times = np.asarray(sec["time"].values)

        npy = int(ny_corr) + 2 * int(ysearch)
        npx = int(nx_corr) + 2 * int(xsearch)
        y_origins = _kernels.offset_locations(
            stack.sizes["y"], npy, count=ny, step=step, shift=y_shift
        )
        x_origins = _kernels.offset_locations(
            stack.sizes["x"], npx, count=nx, step=step, shift=x_shift
        )
        # The location is attributed to the pixel at its window's centre, so the
        # coarse axis is that pixel's own map coordinate and the ASCII export can
        # name a real pixel index.
        y_pixel = _kernels.offset_centre_pixels(y_origins, npy)
        x_pixel = _kernels.offset_centre_pixels(x_origins, npx)

        x_off, y_off, corr = _kernels.pixel_offsets_dask(
            ref.data, sec.data, y_origins, x_origins,
            nx_corr=int(nx_corr), ny_corr=int(ny_corr),
            xsearch=int(xsearch), ysearch=int(ysearch),
            x_shift=int(x_shift), y_shift=int(y_shift),
            interp_factor=int(interp_factor),
            subpixel_window=int(subpixel_window), subpixel=bool(subpixel),
            oversample=int(oversample),
            min_valid_fraction=float(min_valid_fraction),
            locations_per_tile=int(locations_per_tile),
        )

        ds = xr.Dataset(
            {
                "x_offset": (("pair", "y", "x"), x_off),
                "y_offset": (("pair", "y", "x"), y_off),
                "correlation": (("pair", "y", "x"), corr),
                # Deliberately data variables, not coordinates. As coordinates
                # they ride along on every field cut from this stack, and then
                # collide with the reprojected axis of anything that warps one --
                # `.grd` export and plotting both raised "conflicting sizes for
                # dimension 'y'". Nothing needs them attached to a field; to_text
                # reads them off the Dataset.
                "y_pixel": ("y", y_pixel),
                "x_pixel": ("x", x_pixel),
            },
            coords={
                "pair": np.arange(len(pair_list)),
                "y": stack.y[y_pixel],
                "x": stack.x[x_pixel],
                "ref_time": ("pair", ref_times),
                "sec_time": ("pair", sec_times),
            },
        )
        ds = ds.rio.write_crs(f"EPSG:{stack.epsg}")
        ds.attrs.update(
            epsg=stack.epsg,
            direction=stack.direction,
            # Resolved, not as passed, so the recorded value (which feeds the
            # stage hash) is the layout actually used.
            step=None if step is None else int(step),
            nx=None if nx is None else int(nx),
            ny=None if ny is None else int(ny),
            nx_corr=int(nx_corr),
            ny_corr=int(ny_corr),
            xsearch=int(xsearch),
            ysearch=int(ysearch),
            x_shift=int(x_shift),
            y_shift=int(y_shift),
            interp_factor=int(interp_factor),
            subpixel_window=int(subpixel_window),
            subpixel=bool(subpixel),
            oversample=int(oversample),
            min_valid_fraction=float(min_valid_fraction),
            x_spacing=float(stack.ds.attrs.get("x_spacing", np.nan)),
            y_spacing=float(stack.ds.attrs.get("y_spacing", np.nan)),
            pairs=[list(p) for p in pair_list],
        )
        return cls(ds)

    @classmethod
    def from_zarr(cls, path):
        return cls(open_stage(path))

    # -- derived fields ----------------------------------------------------
    @property
    def east_offset(self):
        """``x_offset`` in metres of eastward ground displacement."""
        out = self.ds["x_offset"] * float(self.ds.attrs["x_spacing"])
        return out.rename("east_offset")

    @property
    def north_offset(self):
        """``y_offset`` in metres of **northward** ground displacement.

        The sign flip is free: ``y_spacing`` is stored signed and is negative on a
        north-up grid, so a positive ``y_offset`` -- a feature moving to a larger
        row index, i.e. southward -- multiplies out to a negative northing.
        """
        out = self.ds["y_offset"] * float(self.ds.attrs["y_spacing"])
        return out.rename("north_offset")

    # -- persistence -------------------------------------------------------
    def persist(self, workspace, name=None, overwrite=False, **params):
        """Write the stack to the workspace and return the reopened lazy stack."""
        name = name or self.STAGE
        ds = self.ds.chunk(self.disk_chunks("pair"))
        full = {
            "stage": name,
            "epsg": self.epsg,
            "pairs": self.ds.attrs.get("pairs"),
            **{
                key: self.ds.attrs.get(key)
                for key in (
                    "step", "nx", "ny", "nx_corr", "ny_corr", "xsearch",
                    "ysearch", "x_shift", "y_shift", "interp_factor",
                    "subpixel_window", "subpixel", "oversample",
                    "min_valid_fraction",
                )
            },
            **params,
        }
        reopened = workspace.store(name, ds, full, overwrite=overwrite)
        return PixelOffsetStack(reopened)

    # -- export ------------------------------------------------------------
    def _grd_specs(self):
        """The three measured fields, plus the same offsets in metres."""
        return [
            ("x_offset", self.ds["x_offset"], True),
            ("y_offset", self.ds["y_offset"], True),
            ("correlation", self.ds["correlation"], True),
            ("east_offset", self.east_offset, True),
            ("north_offset", self.north_offset, True),
        ]

    def to_text(self, outdir, indices=None, min_correlation=0.0,
                stem="freq_xcorr", comment=None):
        """Write the offsets as GMTSAR ``xcorr`` ASCII, one file per pair.

        Each line is ``x_pixel x_offset y_pixel y_offset correlation`` under
        ``xcorr``'s own format string, leading and trailing space included, so the
        files feed ``fitoffset.csh`` -- or any ``awk '{if ($5 > 20) ...}'`` on top
        of it -- unchanged. Pixel indices are 0-based, as the reference's C array
        indices are, and index the **source** GSLC grid, not this coarse one.

        Files are named ``{stem}_pair{i}.dat``, mirroring
        :meth:`~nisar_tools._base.RasterStackMixin.to_grd`; ``indices`` selects
        which pairs to write (default: all). Returns the written paths.

        Locations whose offset is NaN are dropped -- a ``nan`` in the offset
        column would poison a ``trend2d`` fit rather than simply score badly --
        and ``min_correlation`` drops more. The default of 0 keeps everything
        measurable, which is the reference's behaviour: it writes every location
        and leaves the cut to ``fitoffset.csh``.

        ``comment`` writes ``#``-prefixed provenance lines first. Off by default,
        because ``xcorr`` writes no header and an unprefixed one would be compared
        numerically by ``awk``.
        """
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        x_pixel = np.asarray(self.ds["x_pixel"].values)
        y_pixel = np.asarray(self.ds["y_pixel"].values)
        # (y, x) grids of the pixel indices, so a location's row survives the
        # ravel that pairs it with its offsets.
        yy, xx = np.meshgrid(y_pixel, x_pixel, indexing="ij")

        paths = []
        for i in range(self.ds.sizes["pair"]) if indices is None else indices:
            sel = self.ds.isel(pair=i)
            x_off = np.asarray(sel["x_offset"].values, dtype=float)
            y_off = np.asarray(sel["y_offset"].values, dtype=float)
            corr = np.asarray(sel["correlation"].values, dtype=float)

            keep = (
                np.isfinite(x_off) & np.isfinite(y_off)
                & (corr >= float(min_correlation))
            )
            table = np.column_stack([
                xx[keep], x_off[keep], yy[keep], y_off[keep], corr[keep],
            ])
            if len(table) < MIN_USEFUL_ROWS:
                warnings.warn(
                    f"pair {i}: only {len(table)} of {keep.size} locations "
                    f"survive min_correlation={min_correlation}; fitoffset.csh "
                    f"needs at least {MIN_USEFUL_ROWS} points",
                    RuntimeWarning,
                    stacklevel=2,
                )

            path = outdir / f"{stem}_pair{i}.dat"
            with open(path, "w") as handle:
                if comment:
                    for line in str(comment).splitlines():
                        handle.write(f"# {line}\n")
                # xcorr's print_results.c format string, exactly.
                np.savetxt(handle, table, fmt=" %d %6.3f %d %6.3f %6.2f ")
            paths.append(path)
        return paths

    # -- plotting ----------------------------------------------------------
    def plot_offsets(self, pair=0, units="pixels", min_correlation=0.0,
                     quiver=False):
        """Plot one pair's offset field. See :func:`nisar_tools.plot.plot_offsets`."""
        from .plot import plot_offsets

        if units == "metres":
            x_field, y_field = self.east_offset, self.north_offset
        elif units == "pixels":
            x_field, y_field = self.ds["x_offset"], self.ds["y_offset"]
        else:
            raise ValueError(f"units must be 'pixels' or 'metres', got {units!r}")

        return plot_offsets(
            x_field.isel(pair=pair),
            y_field.isel(pair=pair),
            correlation=self.ds["correlation"].isel(pair=pair),
            epsg_code=self.ds.attrs.get("epsg"),
            units=units,
            min_correlation=min_correlation,
            quiver=quiver,
        )

    def __repr__(self):
        s = self.sizes
        return (
            f"<PixelOffsetStack EPSG:{self.epsg} "
            f"pair={s.get('pair')} y={s.get('y')} x={s.get('x')}>"
        )
