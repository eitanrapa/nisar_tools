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

from . import _kernels, geo
from ._base import RasterStackMixin, open_stage
from .interferogram import make_pairs

#: Default gap between correlation locations, in pixels -- 80 m at NISAR's 5 m
#: posting. This deliberately **oversamples**: the resolution of the measurement
#: is the 64 px correlation template, so adjacent estimates share 83% of their
#: data and are not independent. Sampling below the resolution is what makes a
#: readable map of a displacement field and lets a discontinuity be localised to
#: better than a template width; it does not add information. Raise it to
#: ``nx_corr // 2`` for independent estimates at a quarter of the cost.
DEFAULT_STEP = 16

#: ``fitoffset.csh`` refuses to fit fewer than this many points.
MIN_USEFUL_ROWS = 8

#: Default correlation cut for :meth:`PixelOffsetStack.robust_offsets`, matching
#: ``fitoffset.csh``'s own threshold on ``xcorr``'s correlation column.
DEFAULT_MIN_CORRELATION = 20.0

#: Default robust-sigma cut for :meth:`PixelOffsetStack.robust_offsets`. Wide
#: enough to keep real deformation (which is what the field is measured for) and
#: still catch the search-edge false peaks, which sit many sigma out.
DEFAULT_N_SIGMA = 6.0


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
        nx_corr=64, ny_corr=64, xsearch=16, ysearch=16, x_shift=0, y_shift=0,
        interp_factor=16, subpixel_window=8, subpixel=True, oversample=1,
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

        ``nx_corr``/``ny_corr`` size the correlation template -- which is the
        **resolution** of the result, as distinct from ``step``, which is only how
        densely it is sampled -- and ``xsearch``/``ysearch`` the search half-width,
        both in pixels. The patch that gets transformed is ``ny_corr + 2*ysearch``
        by ``nx_corr + 2*xsearch``. The defaults are a 64 px template searched over
        +/-16 px, i.e. 320 m and +/-80 m at NISAR's 5 m posting; they are **not**
        GMTSAR's 128/64, which are sized for a radar-geometry pair that still needs
        coregistering. See :func:`nisar_tools._kernels.pixel_offsets` for why a
        narrower search is better here.

        ``x_shift``/``y_shift`` centre the search on a known bulk displacement
        (GMTSAR's PRM ``rshift``/``ashift``) and are added back into the answer.
        They default to 0 because a geocoded pair is already aligned to the whole
        pixel; set them from the median of a first pass when it is not.

        See :func:`nisar_tools._kernels.pixel_offsets` for the sub-pixel
        refinement (``interp_factor``, ``subpixel_window``, ``subpixel``), the
        input oversampling (``oversample``) and the invalid-sample rule
        (``min_valid_fraction``). ``locations_per_tile`` is a dask granularity dial
        only and does not change the result.

        The raw fields this returns are not yet readable as ground motion: they
        carry the pair's bulk geolocation offset and a few percent of false peaks
        that the correlation does not flag. :meth:`robust_offsets` removes both.
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

    def robust_offsets(self, field="x_offset",
                       min_correlation=DEFAULT_MIN_CORRELATION,
                       n_sigma=DEFAULT_N_SIGMA, centre=True):
        """One offset component, gated on correlation, MAD-clipped and centred.

        This is how an offset field is read as ground motion. Three steps, each
        removing something the raw field carries that is not displacement:

        1. **Drop locations below** ``min_correlation``. The default of 20 is
           ``fitoffset.csh``'s own threshold on the same number.
        2. **Drop locations more than** ``n_sigma`` **robust standard deviations
           from the median**, the scale being the MAD,
           ``1.4826 * median(|x - median|)``. This is what removes the false
           peaks, and nothing else does: a window that fails to match still
           reports an argmax, which pins at the edge of the search and comes back
           as +/-(search - 0.5) px *with a high correlation*. Measured on a real
           A061 crop, 3-4% of locations, none of which a correlation cut finds.
           The MAD is used rather than a standard deviation precisely because
           those outliers would inflate the scale that is meant to catch them.
           A MAD of zero says over half the kept locations hold *exactly* the
           median, which real data does -- the two D_071 granules are
           co-registered to the whole pixel and their offset is uniform to an IQR
           of 0.000 -- so the scale is then taken from the locations that do
           differ, rather than every one of them being called an outlier.
        3. **Subtract the median** (``centre=False`` keeps it). That median is the
           pair's bulk geolocation offset -- measured at -1.06 px, i.e. -5.3 m,
           between two real D_071 granules, agreeing between two crops 20 km apart
           to 0.06 px. It is several times the deformation signal in the same
           scene and constant across it, so plotting the raw field shows the
           offset between two products and hides the ground motion.

        The statistics are computed **per pair**, since each pair has its own bulk
        shift, and per component, since the two scatter differently (0.093 px in
        ``y`` against 0.278 px in ``x`` on that same crop, the difference being the
        signal). The removed median and the robust sigma are recorded in the
        result's ``attrs`` as ``robust_median``/``robust_sigma``, one per pair --
        the median is itself a measurement, of how far apart the two products are
        geolocated.

        ``field`` is any of ``x_offset``, ``y_offset`` (pixels) or ``east_offset``,
        ``north_offset`` (metres). Returns a lazy-free :class:`xarray.DataArray`
        with this stack's dims and coords, so it plots and exports like any other
        field.
        """
        fields = {
            "x_offset": lambda: self.ds["x_offset"],
            "y_offset": lambda: self.ds["y_offset"],
            "east_offset": lambda: self.east_offset,
            "north_offset": lambda: self.north_offset,
        }
        if field not in fields:
            raise KeyError(
                f"unknown field {field!r}; choose from {sorted(fields)}"
            )
        if n_sigma <= 0:
            raise ValueError(f"n_sigma must be > 0, got {n_sigma}")

        values = np.asarray(fields[field]().values, dtype=float)
        corr = np.asarray(self.ds["correlation"].values, dtype=float)
        # A 2-D stack (one pair already selected) still has to be reduced per
        # plane, so give it a leading axis and take it away again at the end.
        flat = values.reshape((-1,) + values.shape[-2:])
        flat_corr = corr.reshape(flat.shape)

        out = np.full_like(flat, np.nan)
        medians, sigmas = [], []
        for k in range(flat.shape[0]):
            plane, good = flat[k], flat_corr[k] >= float(min_correlation)
            sample = plane[good & np.isfinite(plane)]
            if sample.size == 0:
                warnings.warn(
                    f"pair {k}: no location has correlation >= "
                    f"{min_correlation} and a finite {field}; the whole plane "
                    "is blank",
                    RuntimeWarning,
                    stacklevel=2,
                )
                medians.append(np.nan)
                sigmas.append(np.nan)
                continue
            median = float(np.median(sample))
            deviation = np.abs(sample - median)
            sigma = 1.4826 * float(np.median(deviation))
            if sigma == 0.0:
                # Over half the kept locations hold *exactly* the median. Real
                # data does this: the two D_071 granules are co-registered to the
                # whole pixel and their offset came out uniform to an IQR of
                # 0.000. Taking the scale from the locations that do differ keeps
                # those few instead of calling every one of them an outlier.
                spread = deviation[deviation > 0]
                sigma = 1.4826 * float(np.median(spread)) if spread.size else 0.0
            medians.append(median)
            sigmas.append(sigma)
            # Only a field that is *entirely* one value still has no scale here --
            # a granule correlated against itself, which returns 0.000 at every
            # location. Keep exactly the median then; testing `< n_sigma * 0`
            # would blank the plane instead.
            keep = good & (
                np.abs(plane - median) < float(n_sigma) * sigma
                if sigma > 0 else plane == median
            )
            out[k] = np.where(keep, plane - median if centre else plane, np.nan)

        template = fields[field]()
        result = xr.DataArray(
            out.reshape(values.shape).astype(np.float32),
            dims=template.dims,
            coords=template.coords,
            name=f"{field}_robust",
        )
        result.attrs.update(
            source_field=field,
            min_correlation=float(min_correlation),
            n_sigma=float(n_sigma),
            centred=bool(centre),
            robust_median=medians,
            robust_sigma=sigmas,
        )
        return result

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
                stem="freq_xcorr", comment=None, lonlat=False):
        """Write the offsets as GMTSAR ``xcorr`` ASCII, one file per pair.

        Each line is ``x_pixel x_offset y_pixel y_offset correlation`` under
        ``xcorr``'s own format string, leading and trailing space included, so the
        files feed ``fitoffset.csh`` -- or any ``awk '{if ($5 > 20) ...}'`` on top
        of it -- unchanged. Pixel indices are 0-based, as the reference's C array
        indices are, and index the **source** GSLC grid, not this coarse one.

        ``lonlat=True`` appends two more columns, ``lon`` and ``lat`` in decimal
        degrees, giving the location's own ground position -- what a GMT ``psxy``
        or ``blockmean`` wants, and the only thing in the file that means anything
        outside this granule's grid. They are **appended**, in that order (GMT's
        x-then-y, not the spoken "lat/lon"), so the five columns the reference
        writes keep their positions and every ``awk`` on top of them still reads
        the same fields. Off by default: the plain file is byte-for-byte what
        ``xcorr`` produces, and that is worth keeping as the default.

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

        # xcorr's print_results.c format string, exactly.
        fmt = " %d %6.3f %d %6.3f %6.2f "
        extra = ()
        if lonlat:
            # Transformed as a full (y, x) grid rather than per axis: lon and lat
            # each vary along *both* map axes, so the two 1-D coordinate vectors
            # do not separate. It is the coarse lattice, so this is a few thousand
            # points, not the source grid's millions.
            map_yy, map_xx = np.meshgrid(
                np.asarray(self.ds["y"].values, dtype=float),
                np.asarray(self.ds["x"].values, dtype=float),
                indexing="ij",
            )
            extra = geo.native_to_lonlat(map_xx, map_yy, self.crs)
            fmt += "%11.6f %10.6f "

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
                *(column[keep] for column in extra),
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
                np.savetxt(handle, table, fmt=fmt)
            paths.append(path)
        return paths

    # -- plotting ----------------------------------------------------------
    def plot_offsets(self, pair=0, units="pixels", min_correlation=0.0,
                     quiver=False, robust=False,
                     n_sigma=DEFAULT_N_SIGMA, vmax=None, **kwargs):
        """Plot one pair's offset field. See :func:`nisar_tools.plot.plot_offsets`.

        ``robust=True`` plots :meth:`robust_offsets` of both components instead of
        the raw fields -- correlation-gated, MAD-clipped and centred on their own
        median -- which is almost always what you want to look at: the raw field is
        dominated by the pair's bulk geolocation offset and speckled with
        search-edge false peaks. It then takes ``min_correlation`` (defaulting to
        :data:`DEFAULT_MIN_CORRELATION` rather than 0, since a robust statistic
        needs a gated sample) and ``n_sigma``. ``vmax`` sets the symmetric colour
        limit explicitly; with ``robust=True`` a value near the expected
        displacement, e.g. ``vmax=0.5`` px, is usually more legible than the
        automatic 98th-percentile scale. ``cmap`` and ``nan_color`` pass through to
        :func:`nisar_tools.plot.plot_offsets`.
        """
        from .plot import plot_offsets

        if units == "metres":
            names = ("east_offset", "north_offset")
        elif units == "pixels":
            names = ("x_offset", "y_offset")
        else:
            raise ValueError(f"units must be 'pixels' or 'metres', got {units!r}")

        if robust:
            if not min_correlation:
                min_correlation = DEFAULT_MIN_CORRELATION
            # Select the pair *before* reducing: robust_offsets computes, and on a
            # lazy stack that would correlate every pair to plot one. The result is
            # identical either way, the statistics being per-pair already.
            one = PixelOffsetStack(self.ds.isel(pair=[pair]))
            x_field, y_field = (
                one.robust_offsets(
                    name, min_correlation=min_correlation, n_sigma=n_sigma
                )
                for name in names
            )
            pair = 0
            # robust_offsets has already applied the correlation cut, and passing
            # it on would re-align a (pair, y, x) correlation against a selected
            # 2-D plane.
            correlation, cut = None, 0.0
        else:
            x_field, y_field = (
                self.ds[name] if name in self.ds else getattr(self, name)
                for name in names
            )
            correlation, cut = self.ds["correlation"], min_correlation

        return plot_offsets(
            x_field.isel(pair=pair),
            y_field.isel(pair=pair),
            correlation=(
                None if correlation is None else correlation.isel(pair=pair)
            ),
            epsg_code=self.ds.attrs.get("epsg"),
            units=units,
            min_correlation=cut,
            quiver=quiver,
            vmax=vmax,
            centred=robust,
            **kwargs,
        )

    def __repr__(self):
        s = self.sizes
        return (
            f"<PixelOffsetStack EPSG:{self.epsg} "
            f"pair={s.get('pair')} y={s.get('y')} x={s.get('x')}>"
        )
