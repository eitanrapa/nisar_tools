"""The :class:`UnwrappedStack`: unwrapped phase + connected components.

One class covers both provenances of an unwrapped stack:

- **SNAPHU** (:meth:`~UnwrappedStack.from_interferograms`) -- we unwrap an
  :class:`~nisar_tools.interferogram.InterferogramStack` ourselves. Unwrapping is
  the one non-lazy stage: SNAPHU is a global optimiser that needs a whole raster,
  so the *pair* is the unit of work. The store is created metadata-only up front,
  each pair is unwrapped and written into its own region and flagged done; peak
  memory is one pair and an interrupted run resumes at the first unfinished pair.
- **NASA GUNW** (:meth:`~UnwrappedStack.from_gunw_files`) -- an already-unwrapped
  geocoded product we simply read. A GUNW additionally ships an
  ``ionospherePhaseScreen``, carried here as the optional ``phase_screen`` layer
  and subtracted on request by :meth:`~UnwrappedStack.remove_phase_screen`.

After ingestion the two are the *same class*: ``unw`` / ``conncomp`` on
``(pair, y, x)`` (plus ``coherence`` / ``phase_screen`` for a GUNW), and the same
downstream operations -- water/edge masking, spline outlier rejection, deramping,
2*pi cycle shifts, LOS conversion, persistence. ``attrs["source"]`` (``"snaphu"``
or ``"gunw"``) records which path built it and drives :meth:`to_los`'s geometry.
"""

import itertools
import os
import warnings
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import dask.array as da
import h5py
import numpy as np
import rioxarray  # noqa: F401
import snaphu
import xarray as xr

from . import _kernels, geometry
from ._base import (
    SPATIAL_CHUNK,
    RasterStackMixin,
    compute_chunks,
    open_stage,
)

# -- NASA GUNW HDF5 layout (verified from real granules) ----------------------
# The unwrappedInterferogram group holds the grid coordinates/projection and the
# per-polarisation data subgroups; only centerFrequency sits one level up.
_GUNW_GRID = "science/LSAR/GUNW/grids/frequency{f}"
_GUNW_UNW = _GUNW_GRID + "/unwrappedInterferogram"
_GUNW_IDENT = "science/LSAR/identification"
_CC_FILL = 65535  # connected-components fill; normalised to the unassigned label 0
_MASK_FILL = 255  # subswath/water mask fill value (out of swath)


def _decode(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def _as_list(paths):
    """A single path is a scalar; anything else is an iterable of paths.

    ``str``/``os.PathLike`` are iterable, so a bare ``list(path)`` would split a
    filename into characters -- guard for that explicitly.
    """
    if isinstance(paths, (str, os.PathLike)):
        return [paths]
    return list(paths)


def _match_pairs(self_ds, other_ds, tolerance_s):
    """Order ``other``'s pairs to line up with ``self``'s by ref+secondary time.

    Frames of the same acquisitions define the same pairs, but each frame's
    ``zeroDopplerStartTime`` can differ by seconds, so pairs are matched by
    nearest ``(ref_time, sec_time)`` within ``tolerance_s`` rather than by exact
    time. Returns an index array ``order`` with ``other.isel(pair=order)`` in
    ``self``'s pair order. Raises if the pair sets do not correspond one-to-one.

    ``tolerance_s=None`` skips the matching entirely and pairs by position --
    for two stacks built from *different* acquisition dates, which define
    genuinely different interferograms and so have no correspondence to find.
    """
    sr = np.asarray(self_ds["ref_time"].values)
    ss = np.asarray(self_ds["sec_time"].values)
    orr = np.asarray(other_ds["ref_time"].values)
    ors = np.asarray(other_ds["sec_time"].values)
    if len(orr) != len(sr):
        raise ValueError(
            f"Cannot merge: stacks have different pair counts ({len(sr)} vs "
            f"{len(orr)})"
        )
    if tolerance_s is None:
        return np.arange(len(sr))

    tol = np.timedelta64(int(round(tolerance_s * 1e9)), "ns")
    order = np.empty(len(sr), dtype=int)
    for i in range(len(sr)):
        dr, dsec = np.abs(orr - sr[i]), np.abs(ors - ss[i])
        j = int(np.argmin(dr + dsec))
        if dr[j] > tol or dsec[j] > tol:
            raise ValueError(
                f"Cannot merge: no pair in the other stack matches self pair {i} "
                f"({sr[i]} / {ss[i]}) within {tolerance_s}s. If the two stacks "
                "really are different acquisitions -- independent "
                "interferograms covering neighbouring ground -- pass "
                "time_tolerance=None to pair them by position instead."
            )
        order[i] = j
    if len(set(order.tolist())) != len(order):
        raise ValueError("Cannot merge: pairs do not correspond one-to-one")
    return order


class UnwrappedStack(RasterStackMixin):
    """A stack of unwrapped phases with connected-component labels."""

    STAGE = "unwrapped"

    def __init__(self, ds):
        self.ds = ds

    @classmethod
    def from_zarr(cls, path):
        return cls(open_stage(path))

    # -- construction: SNAPHU on our own interferograms --------------------
    @classmethod
    def from_interferograms(
        cls, igrams, workspace, name="unwrapped", nproc=1, res_az=8, res_rg=3,
        overwrite=False, ntiles=None, tile_overlap=None,
        max_tile_pixels=_kernels.DEFAULT_MAX_TILE_PIXELS, min_region_size=100,
        tile_cost_thresh=500, single_tile_reoptimize=True, pairs_in_flight=1,
    ):
        """Unwrap every pair with SNAPHU into a region-written, resumable store.

        **Tiling is a function of the raster, not of ``nproc``.** SNAPHU's tile
        grid decides where the 2*pi seams fall, so it is part of the answer, not a
        performance dial: ``max_tile_pixels`` (and an explicit ``ntiles``) set the
        geometry, and ``nproc`` only says how many of those tiles may be solved at
        once. Consequently ``nproc`` is *not* in the params hash -- varying it must
        not invalidate a finished store -- while the tiling parameters are.

        Lower ``max_tile_pixels`` for more, smaller tiles (more concurrency inside
        a pair, but more seams); a raster under the budget is unwrapped as a single
        tile, which is the best-quality option and skips the tile-assembly and
        ``single_tile_reoptimize`` passes entirely. ``single_tile_reoptimize=False``
        drops a second, fully serial whole-raster SNAPHU pass -- the biggest single
        speed knob for a tiled unwrap.

        Pairs are independent, so ``pairs_in_flight`` unwraps them concurrently.
        Even at the default 1 the *next* pair's inputs are read while SNAPHU works
        on the current one, which hides the whole HDF5 -> multilook -> filter chain
        behind the unwrap. Raise it above 1 when SNAPHU cannot use the cores itself
        -- a single-tile raster, or ``nproc`` above the tile count -- since the
        process budget is then split ``nproc // pairs_in_flight`` ways. Peak memory
        is proportional to ``pairs_in_flight``, not to the stack length.
        """
        ds = igrams.ds
        npair = ds.sizes["pair"]
        ny, nx = ds.sizes["y"], ds.sizes["x"]
        looks = int(ds.attrs.get("looks", 1))
        # On a geocoded near-polar-orbit grid, azimuth (along-track) runs
        # closest to y and range to x.
        spacing_az = float(ds.attrs.get("y_spacing", 1.0))
        spacing_rg = float(ds.attrs.get("x_spacing", 1.0))

        nlooks = _kernels.snaphu_nlooks(
            looks, looks, spacing_az, spacing_rg, res_az, res_rg
        )
        ntiles, overlap = _kernels.snaphu_params(
            (ny, nx), nproc, ntiles=ntiles, tile_overlap=tile_overlap,
            max_tile_pixels=max_tile_pixels, min_region_size=min_region_size,
        )
        # Fail here, naming the numbers, rather than as a bare RuntimeError
        # carrying only the subprocess's stderr.
        _kernels.snaphu_params_check(
            (ny, nx), ntiles, overlap, min_region_size=min_region_size
        )

        snaphu_kwargs = dict(
            nlooks=nlooks, ntiles=ntiles, tile_overlap=overlap, nproc=nproc,
            min_region_size=min_region_size, tile_cost_thresh=tile_cost_thresh,
            single_tile_reoptimize=single_tile_reoptimize,
        )

        params = {
            "stage": name,
            "epsg": int(ds.attrs["epsg"]),
            "source": "snaphu",
            "looks": looks,
            "nlooks": nlooks,
            "res_az": res_az,
            "res_rg": res_rg,
            "pairs": ds.attrs.get("pairs"),
            # Tiling changes the unwrapped phase, so it belongs in the hash.
            # ``nproc`` deliberately does not: it no longer changes the result.
            "ntiles": list(ntiles),
            "tile_overlap": overlap,
            "min_region_size": int(min_region_size),
            "tile_cost_thresh": int(tile_cost_thresh),
            "single_tile_reoptimize": bool(single_tile_reoptimize),
        }

        # Metadata-only store so each pair can be written by region.
        template = _template(ds, npair, ny, nx)
        workspace.init_store(
            name, template, params, overwrite=overwrite, source=ds
        )

        done = workspace.pairs_done(name)
        todo = [i for i in range(npair) if i not in done]
        inflight = max(1, int(pairs_in_flight))
        if inflight > 1:
            # SNAPHU forks NPROC children per pair, so the process budget has to
            # be shared or the box is oversubscribed by `pairs_in_flight` times.
            snaphu_kwargs["nproc"] = max(1, nproc // inflight)

        def _load(i):
            """Materialise one pair. This is the whole upstream graph."""
            return (
                np.asarray(ds["igram"].isel(pair=i).values),
                np.asarray(ds["coherence"].isel(pair=i).values),
            )

        def _unwrap_and_write(i, igram, corr):
            unw, conncomp = _unwrap_pair(igram, corr, pair=i, **snaphu_kwargs)
            pair_ds = xr.Dataset(
                {
                    "unw": (("pair", "y", "x"), unw[None]),
                    "conncomp": (("pair", "y", "x"), conncomp[None]),
                    # Carry the SNAPHU input coherence straight through to the
                    # output so it is available to mask_edges(min_coherence).
                    "coherence": (("pair", "y", "x"), corr.astype(np.float32)[None]),
                }
            )
            # Disk chunks are 1 deep along `pair`, so concurrent region writes
            # never touch the same Zarr chunk and need no synchronizer.
            workspace.write_region(name, pair_ds, region={"pair": slice(i, i + 1)})
            workspace.mark_pair_done(name, i)

        with ThreadPoolExecutor(inflight, thread_name_prefix="unwrap") as pool:
            running = deque()
            for i, (igram, corr) in _prefetch(_load, todo, lookahead=inflight):
                # Bound the unwraps in flight as well as the loads, so peak memory
                # stays proportional to `pairs_in_flight` and not to npair.
                while len(running) >= inflight:
                    running.popleft().result()
                running.append(pool.submit(_unwrap_and_write, i, igram, corr))
            for future in running:
                future.result()

        workspace.consolidate(name)
        return cls.from_zarr(workspace.path(name))

    # -- construction: read a NASA GUNW ------------------------------------
    @classmethod
    def from_gunw_file(cls, path, frequency="A", polarization="HH"):
        """Read a single NASA GUNW ``.h5`` as a one-pair unwrapped stack."""
        return cls.from_gunw_files([path], frequency=frequency,
                                   polarization=polarization)

    @classmethod
    def from_gunw_files(cls, paths, frequency="A", polarization="HH"):
        """Read one or more NASA GUNW granules into a ``(pair, y, x)`` stack.

        A GUNW arrives already unwrapped, so this reads rather than computes.
        Layers are mapped onto the same names the SNAPHU path uses --
        ``unwrappedPhase`` -> ``unw``, ``coherenceMagnitude`` -> ``coherence``,
        ``connectedComponents`` -> ``conncomp`` (uint32; the 65535 fill and every
        out-of-swath pixel -> label 0). The GUNW's ``ionospherePhaseScreen`` and
        its ``mask`` (water + subswath-validity flag) are carried as the optional
        ``phase_screen`` and ``subswath_mask`` layers, feeding
        :meth:`remove_phase_screen` and ``mask_edges(use_builtin_mask=True)``.

        Every granule must share the same geocoded grid (a single frame's time
        series); a granule on a different grid is rejected. Rasters are modest
        (tens of MB) so they are read eagerly; :meth:`persist` chunks them.
        """
        paths = [str(p) for p in _as_list(paths)]
        if not paths:
            raise ValueError("Need at least one GUNW granule")

        unw_list, coh_list, cc_list, iono_list, mask_list = [], [], [], [], []
        ref_times, sec_times = [], []
        x0 = y0 = epsg0 = None
        direction = look_direction = wavelength = None
        any_iono = any_mask = False

        for path in paths:
            with h5py.File(path, "r") as f:
                center_freq = float(f[_GUNW_GRID.format(f=frequency)]["centerFrequency"][()])

                unw_group = f[_GUNW_UNW.format(f=frequency)]
                x = unw_group["xCoordinates"][()].astype(float)
                y = unw_group["yCoordinates"][()].astype(float)
                epsg = int(unw_group["projection"].attrs["epsg_code"])

                ig = unw_group[polarization]
                unw = ig["unwrappedPhase"][()].astype(np.float32)
                coh = ig["coherenceMagnitude"][()].astype(np.float32)
                cc = ig["connectedComponents"][()]
                iono = (ig["ionospherePhaseScreen"][()].astype(np.float32)
                        if "ionospherePhaseScreen" in ig else None)
                # The subswath/water validity mask sits on the group above HH.
                swmask = (unw_group["mask"][()].astype(np.uint8)
                          if "mask" in unw_group else None)

                ident = f[_GUNW_IDENT]
                dirn = _decode(ident["orbitPassDirection"][()])
                look = (_decode(ident["lookDirection"][()])
                        if "lookDirection" in ident else None)
                ref_t = _decode(ident["referenceZeroDopplerStartTime"][()])
                sec_t = _decode(ident["secondaryZeroDopplerStartTime"][()])

            if x0 is None:
                x0, y0, epsg0 = x, y, epsg
                direction, look_direction = dirn, look
                wavelength = geometry.SPEED_OF_LIGHT / center_freq
            elif not (epsg == epsg0 and np.array_equal(x, x0)
                      and np.array_equal(y, y0)):
                raise ValueError(
                    f"GUNW granule {path!r} is on a different grid than the "
                    "first; from_gunw_files stacks a single frame's time series "
                    "(merging separate frames is not supported here)."
                )

            invalid = ~np.isfinite(unw)
            cc = cc.astype(np.uint32)
            cc[invalid | (cc == _CC_FILL)] = 0
            if iono is None:
                iono = np.full_like(unw, np.nan)
            else:
                any_iono = True
            if swmask is None:
                swmask = np.full(unw.shape, _MASK_FILL, np.uint8)
            else:
                any_mask = True

            unw_list.append(unw)
            coh_list.append(coh)
            cc_list.append(cc)
            iono_list.append(iono)
            mask_list.append(swmask)
            ref_times.append(np.datetime64(ref_t))
            sec_times.append(np.datetime64(sec_t))

        npair = len(paths)
        data = {
            "unw": (("pair", "y", "x"), np.stack(unw_list)),
            "coherence": (("pair", "y", "x"), np.stack(coh_list)),
            "conncomp": (("pair", "y", "x"), np.stack(cc_list)),
        }
        if any_iono:
            data["phase_screen"] = (("pair", "y", "x"), np.stack(iono_list))
        if any_mask:
            data["subswath_mask"] = (("pair", "y", "x"), np.stack(mask_list))

        ds = xr.Dataset(
            data,
            coords={
                "pair": np.arange(npair),
                "y": y0,
                "x": x0,
                "ref_time": ("pair", np.asarray(ref_times)),
                "sec_time": ("pair", np.asarray(sec_times)),
            },
        )
        ds = ds.rio.write_crs(f"EPSG:{int(epsg0)}")
        ds.attrs.update(
            epsg=int(epsg0),
            source="gunw",
            direction=direction,
            look_direction=look_direction,
            wavelength=float(wavelength),
            frequency=frequency,
            polarization=polarization,
            # Kept so to_los can find the embedded geometry cube after a reload.
            source_files=paths,
            pairs=[[str(r), str(s)] for r, s in zip(ref_times, sec_times)],
        )
        return cls(ds)

    # -- operations --------------------------------------------------------
    def mask_water(self, mask_cache=None, resolution="f", spacing=None,
                   mask_name=None):
        """Lazily mask water on the unwrapped phase. Returns a new stack.

        Lazy: the masked values are **not** written anywhere. Call
        :meth:`persist` (under a new stage name) if you want them on disk.

        ``mask_cache`` is a :class:`~nisar_tools.workspace.Workspace` used to
        cache the *coastline mask itself*, keyed on the grid, so GMT is not
        re-run for the same crop. It is not where the masked data goes.

        ``resolution`` is the GMT coastline resolution; use a coarser value
        (e.g. ``"i"``) if the full-resolution GSHHG dataset is unavailable.
        ``spacing`` defaults to tracking this stack's own pixel size.
        ``mask_name`` overrides the cache store's name.
        """
        from .mask import grid_spacing_arg, water_mask_for_grid

        # Resolve here so the recorded value (which feeds the stage hash) is
        # the increment actually used, not a placeholder None.
        if spacing is None:
            spacing = grid_spacing_arg(self.x, self.y, self.epsg)

        mask = water_mask_for_grid(
            self.x, self.y, self.epsg, workspace=mask_cache, name=mask_name,
            resolution=resolution, spacing=spacing,
        )
        ds = self.ds.copy()
        # The mask is land=1 / water=NaN; ``where`` needs a boolean condition
        # (NaN is truthy, so passing the raw mask would keep water pixels).
        ds["unw"] = self.ds["unw"].where(mask.notnull())
        ds.attrs.update(self.ds.attrs)
        ds.attrs["water_mask"] = {"resolution": resolution, "spacing": spacing}
        return UnwrappedStack(ds)

    def mask_edges(self, edge_pixels=8, min_coherence=None, use_builtin_mask=False,
                   edges="along_track"):
        """Mask swath-edge effects. Returns a new stack (lazy).

        Trims ``edge_pixels`` pixels off each pair's swath boundary, removing the
        decorrelated fringe there. Works for both SNAPHU and GUNW stacks.
        ``min_coherence`` additionally nulls pixels whose coherence is below that
        threshold; both provenances carry a ``coherence`` layer (a GUNW's
        ``coherenceMagnitude``, or the interferogram's carried through the SNAPHU
        unwrap).

        ``edges="along_track"`` (default) trims only the near/far-range
        boundaries -- the swath edges that run along-track, which are the ones
        carrying the fringe. ``edges="all"`` instead erodes the whole finite
        footprint, which also eats a collar around coastlines, water-masked lakes
        and the frame's azimuth ends: on a coastal descending frame that was
        nearly half of everything it removed, right where the near-fault signal
        lives. See :func:`~nisar_tools._kernels.along_track_edge_mask` for the
        geometry this assumes.

        ``use_builtin_mask`` (GUNW only) first nulls the pixels the product's own
        ``subswath_mask`` flags as invalid samples -- an *exact* edge/gap mask
        rather than a blunt erosion. The 3-digit flag's two low digits are the
        pixel's subswath number in the reference and secondary RSLC; a 0 in
        either marks an out-of-subswath sample. It composes with ``edge_pixels``:
        pass ``edge_pixels=0`` for the mask alone, or a few pixels of extra
        erosion for a safety margin inside the exact boundary. (The mask's water
        digit is left to :meth:`mask_water`.)
        """
        unw = self.ds["unw"]

        if use_builtin_mask:
            if "subswath_mask" not in self.ds:
                raise ValueError(
                    "use_builtin_mask needs the GUNW subswath-validity layer, "
                    "which only GUNW-derived stacks carry (and only when the "
                    "product ships a `mask` layer)."
                )
            m = self.ds["subswath_mask"]
            ss_ref = (m // 10) % 10  # subswath number in the reference RSLC
            ss_sec = m % 10          # subswath number in the secondary RSLC
            valid = (m != _MASK_FILL) & (ss_ref > 0) & (ss_sec > 0)
            unw = unw.where(valid)

        if edges == "along_track":
            # Support is the whole row (which sample is the row's first?), so
            # there is no halo that would do -- x is gathered and the split runs
            # along y, where rows are independent.
            masked = _plane_kernel_rows(
                _kernels.mask_edges, unw,
                edge_pixels=int(edge_pixels), edges="along_track",
            )
        elif edges == "all":
            # Erosion by ``edge_pixels`` cross iterations reaches exactly that
            # far, so a matching halo decomposes it spatially and the result is
            # identical to the whole-plane fit -- unlike the old
            # chunk({"y": -1, "x": -1}), which capped concurrency at the pair count.
            masked = _plane_kernel(
                _kernels.mask_edges, unw, depth=max(1, int(edge_pixels)),
                edge_pixels=int(edge_pixels), edges="all",
            )
        else:
            raise ValueError(
                f"edges must be 'along_track' or 'all', got {edges!r}"
            )
        if min_coherence is not None:
            if "coherence" not in self.ds:
                raise ValueError(
                    "min_coherence needs a coherence layer, which this stack "
                    "lacks. Both SNAPHU and GUNW stacks normally carry one; an "
                    "unwrapped store written before coherence was carried through "
                    "would not -- rebuild it (overwrite=True or ws.clear)."
                )
            masked = masked.where(self.ds["coherence"] >= min_coherence)

        ds = self.ds.copy()
        ds["unw"] = masked
        ds.attrs.update(self.ds.attrs)
        ds.attrs["edges_masked"] = {
            "edge_pixels": int(edge_pixels), "min_coherence": min_coherence,
            "use_builtin_mask": bool(use_builtin_mask), "edges": edges,
        }
        return UnwrappedStack(ds)

    def remove_outliers(self, scale=16.0, threshold=1.0, iterations=2):
        """Reject residual outliers against a smooth spline. Returns a new stack.

        Fits a NaN-aware smooth surface (Gaussian sigma ``scale`` px) to each
        pair's unwrapped phase, nulls pixels where ``|phase - surface|`` exceeds
        ``threshold`` radians, and refits ``iterations`` times. This is the
        tension-spline + residual-mask step of ``filt_gunw.csh``, in scipy.
        """
        # scipy's Gaussian truncates at 4 sigma, so the smooth surface has finite
        # support and a 4*scale halo per iteration reproduces the whole-plane
        # result exactly while letting the chunks run independently.
        cleaned = _plane_kernel(
            _kernels.remove_outliers, self.ds["unw"],
            depth=_kernels.remove_outliers_depth(scale, iterations),
            scale=float(scale), threshold=float(threshold),
            iterations=int(iterations),
        )
        ds = self.ds.copy()
        ds["unw"] = cleaned
        ds.attrs.update(self.ds.attrs)
        ds.attrs["outliers_removed"] = {
            "scale": float(scale), "threshold": float(threshold),
            "iterations": int(iterations),
        }
        return UnwrappedStack(ds)

    def deramp(self, degree=1, method="poly", scale=None, mask=None):
        """Remove a long-wavelength ramp (e.g. ionosphere). Returns a new stack.

        ``method="poly"`` subtracts a total-degree-``degree`` 2D polynomial (the
        classic InSAR deramp; ``1`` = plane), flattening the far field toward
        zero. ``method="spline"`` subtracts a NaN-aware smooth surface at Gaussian
        sigma ``scale`` px (a high-pass for gently curved ionosphere ramps). Best
        run after :meth:`mask_edges` / :meth:`remove_outliers` so the fit is not
        pulled by edges or spikes.

        ``mask`` optionally excludes a **signal region** from the ramp fit, so
        real deformation does not bias the trend: the surface is fit only to the
        finite pixels *outside* the mask, then subtracted **everywhere** (the
        signal keeps its data, minus the ramp). Pass either a boolean ``(y, x)``
        array / DataArray that is ``True`` over the signal, or a
        ``(lon_min, lon_max, lat_min, lat_max)`` lon/lat bbox for a rectangular
        region. With ``method="spline"`` keep ``scale`` comfortably larger than
        the masked gap, or the region's interior fills with NaN.
        """
        exclude, mask_prov = self._deramp_mask(mask)
        unw = self.ds["unw"]
        if method == "poly":
            # A least-squares fit is a sum over pixels, so the normal equations
            # accumulate per chunk and the (tiny) system is solved once. Same
            # answer as the whole-plane fit without ever building an
            # (npix, ncoef) design matrix -- 768 MB for a degree-2 fit on a
            # 4000x4000 plane.
            deramped = _plane_kernel_poly(unw, int(degree), exclude)
        else:
            # A spline deramp's sigma defaults to a quarter of the raster, so its
            # support is global by design; there is nothing to decompose.
            deramped = _plane_kernel(
                _kernels.deramp, unw, depth=max(unw.sizes["y"], unw.sizes["x"]),
                degree=int(degree), method=method, scale=scale, exclude=exclude,
            )
        ds = self.ds.copy()
        ds["unw"] = deramped
        ds.attrs.update(self.ds.attrs)
        prov = {"degree": int(degree), "method": method, "scale": scale}
        if mask_prov is not None:
            prov["mask"] = mask_prov
        ds.attrs["deramp"] = prov
        return UnwrappedStack(ds)

    def _deramp_mask(self, mask):
        """Resolve a deramp signal ``mask`` to a boolean ``(y, x)`` exclude array
        plus a JSON-able provenance tag.

        ``mask`` is ``None`` (no exclusion), a boolean grid (``True`` = signal to
        leave out of the fit), or a ``(lon_min, lon_max, lat_min, lat_max)`` bbox.
        """
        if mask is None:
            return None, None
        # A lon/lat bbox is a length-4 sequence of scalars.
        if (not isinstance(mask, (xr.DataArray, np.ndarray))
                and len(mask) == 4 and all(np.isscalar(v) for v in mask)):
            from . import geo

            x_min, x_max, y_min, y_max = geo.bbox_to_native(*mask, self.epsg)
            x, y = self.x, self.y
            in_x = (x >= min(x_min, x_max)) & (x <= max(x_min, x_max))
            in_y = (y >= min(y_min, y_max)) & (y <= max(y_min, y_max))
            return (in_y[:, None] & in_x[None, :]), list(mask)

        arr = mask.values if isinstance(mask, xr.DataArray) else np.asarray(mask)
        exclude = np.asarray(arr, dtype=bool)
        expected = (self.sizes["y"], self.sizes["x"])
        if exclude.shape != expected:
            raise ValueError(
                f"deramp mask shape {exclude.shape} does not match the (y, x) "
                f"grid {expected}"
            )
        return exclude, "custom"

    def remove_phase_screen(self):
        """Subtract the GUNW's NASA ionosphere phase screen (lazy). Returns a new stack.

        A NASA GUNW ships an ``ionospherePhaseScreen`` (loaded as
        ``phase_screen``); this subtracts it from the unwrapped phase
        (``unw - phase_screen``, as in the reference ``h52grd.py``). The screen
        layer is kept for inspection.

        Raises if the stack carries no screen -- a SNAPHU stack has none, and
        instead flattens long-wavelength trends with the deramp pipeline
        (:meth:`mask_edges` -> :meth:`remove_outliers` -> :meth:`deramp`) -- or
        if the screen was already removed.
        """
        if "phase_screen" not in self.ds:
            raise ValueError(
                "This stack carries no 'phase_screen' to remove. Only a NASA "
                "GUNW ships one; a SNAPHU stack instead flattens long-wavelength "
                "trends with deramp (mask_edges -> remove_outliers -> deramp)."
            )
        if self.ds.attrs.get("phase_screen_removed"):
            raise ValueError(
                "The phase screen has already been removed from this stack"
            )

        ds = self.ds.copy()
        unw = self.ds["unw"]
        ds["unw"] = (unw - self.ds["phase_screen"]).astype(unw.dtype)
        ds.attrs.update(self.ds.attrs)
        ds.attrs["phase_screen_removed"] = True
        return UnwrappedStack(ds)

    def add_cycles(self, cycles, pair=None, conncomp=None):
        """Shift the unwrapped phase by an integer number of 2*pi cycles.

        Unwrapping recovers phase only up to a global multiple of 2*pi, and the
        unwrapper resolves each *connected component* independently, so distinct
        components can sit whole cycles apart with no way to tell from the data
        alone. This applies the offset once you know it -- from a GPS station, a
        known-stable area, or a neighbouring component that should be continuous.

        ``cycles`` is added (negative removes) and must be whole: any other shift
        changes the wrapped phase. ``pair`` selects pair indices (default all) and
        ``conncomp`` selects component labels (default the whole raster; label 0
        is "unassigned", not a real region).

        Lazy, like :meth:`mask_water`. The shift carries into :meth:`to_los`, so
        apply it before converting.
        """
        if int(cycles) != cycles:
            raise ValueError(
                f"cycles must be a whole number of 2*pi, got {cycles!r}; "
                "a fractional shift would change the wrapped phase"
            )
        cycles = int(cycles)

        unw = self.ds["unw"]
        # Accumulate in float64 and round once on the way out: phase is stored
        # as float32, and rounding 2*pi to float32 first would leave a residue
        # that compounds over repeated shifts.
        shift = xr.zeros_like(unw, dtype=np.float64) + (cycles * 2.0 * np.pi)

        if pair is not None:
            wanted = np.atleast_1d(np.asarray(pair))
            shift = shift.where(unw["pair"].isin(wanted), 0.0)
        if conncomp is not None:
            wanted = np.atleast_1d(np.asarray(conncomp))
            shift = shift.where(self.ds["conncomp"].isin(wanted), 0.0)

        ds = self.ds.copy()
        # NaN + shift stays NaN, so the invalid footprint is preserved.
        ds["unw"] = (unw + shift).astype(unw.dtype)
        ds.attrs.update(self.ds.attrs)
        applied = list(self.ds.attrs.get("cycle_shifts", []))
        applied.append(
            {"cycles": cycles,
             "pair": None if pair is None else np.atleast_1d(pair).tolist(),
             "conncomp": None if conncomp is None
             else np.atleast_1d(conncomp).tolist()}
        )
        ds.attrs["cycle_shifts"] = applied
        return UnwrappedStack(ds)

    def merge(self, other, resampling="bilinear", time_tolerance=600.0,
              tie="auto"):
        """Stitch an adjacent same-track :class:`UnwrappedStack` onto the union grid.

        The same lattice-padding machinery as :meth:`~nisar_tools.stack.GSLCStack.merge`
        (union grid, no reindex, ``self`` wins in the overlap), plus the one thing
        unwrapped phase needs: each frame was unwrapped independently, so across the
        seam the two frames' ``unw`` share the *identical wrapped phase* but sit a
        whole number of 2*pi cycles apart. Per pair, that integer is read from the
        overlap -- the rounded median of ``(unw_self - unw_other) / 2*pi`` -- and
        added to ``other``, so the phase is **continuous across the join with no
        2*pi step**.

        When ``other`` is gridded in a **different UTM zone** -- a track crossing a
        zone boundary -- every one of its layers is first warped onto ``self``'s
        grid, one pair per dask task, keeping ``self``'s spacing and grid phase so
        the overlap still lines up exactly. ``resampling`` applies to the continuous
        layers (``"nearest"`` preserves exact sample values); the label layers
        (``conncomp``, ``subswath_mask``) are always nearest, since an interpolated
        label is not a label. The 2*pi offset is read *after* the warp, on the shared
        grid, so resampling cannot smear the cycle estimate -- the median over the
        whole overlap is far more robust than the sub-radian resampling error. A
        cross-zone warp leaves rotated nodata wedges, exactly as
        :meth:`~nisar_tools.stack.GSLCStack.merge` does; :meth:`crop` trims a ragged
        edge if it matters.

        The same warp also rescues two stacks in **one** CRS whose grids are offset
        by a sub-pixel amount -- which is what an interferogram multilooked before
        ``align_looks`` existed looks like, since it anchored its grid phase to its
        own crop origin rather than to the native lattice. That case is a
        ``RuntimeWarning``, not an error: resampling smooth unwrapped phase is
        harmless, but re-forming both frames' interferograms with
        ``align_looks=True`` avoids it entirely and is exact. Differing pixel
        *spacing* (two stages multilooked with different ``looks``) still raises --
        no regridding here would make them one product.

        By default the two stacks must cover the **same acquisitions** (pairs are
        matched by reference/secondary time within ``time_tolerance`` seconds, so
        per-frame ``zeroDopplerStartTime`` jitter is tolerated). ``coherence`` --
        and, for a GUNW, ``phase_screen`` / ``subswath_mask`` -- are carried through
        with the same ``self`` precedence; ``other``'s ``conncomp`` labels are
        shifted clear of ``self``'s so the two frames' components stay distinct.

        **Different dates.** ``time_tolerance=None`` pairs by position instead of
        by time, which is how two *independent* interferograms -- neighbouring
        ground, but different acquisition dates -- are mosaicked into one field.
        That changes what the overlap means, and so what ``tie`` may be:

        ``"cycles"``
            the offset is a whole number of 2*pi, read as
            ``round(median((self - other) / 2*pi))``. Correct **only** for one
            interferogram unwrapped separately per frame, where the two frames
            carry the identical wrapped phase in the overlap and can differ by
            nothing but an integer ambiguity.
        ``"offset"``
            the offset is the real-valued ``median(self - other)``. Two different
            date pairs measure different deformation, so their difference in the
            overlap is *not* an integer number of cycles -- rounding it to one
            would leave a residual step up to pi. This ties the two to a common
            datum; the leftover disagreement is real signal plus atmosphere.
        ``"none"``
            leave ``other`` as it is.
        ``"auto"`` (default)
            ``"cycles"`` when the pairs matched by time, ``"offset"`` when they
            did not -- so the existing same-acquisition behaviour is unchanged.

        A different-dates mosaic is one raster with **one** arbitrary constant and
        one ramp, so for a slip inversion prefer keeping the scenes apart and
        combining them at :meth:`nisar_tools.slip.Observations.concat`, which gives
        each its own nuisance columns via ``SlipInversion(..., ramp=...)``. Merge
        them here for a continuous map, an export, or when each frame covers a
        different piece of the fault and there is only one interferogram per frame.

        The offset is one value per pair over the *whole* overlap. If SNAPHU split
        the overlap into components sitting at different cycle offsets, the median
        aligns the dominant one; correct any stragglers afterwards with
        :meth:`add_cycles` (``conncomp=``). Lazy, like the other operations.
        """
        from .stack import (
            LATTICE_TOL_PX,
            _pad_onto,
            _union_lattice,
            lattice_offset,
            warp_onto_lattice,
            warp_target_lattice,
        )

        if tie not in ("auto", "cycles", "offset", "none"):
            raise ValueError("tie must be 'auto', 'cycles', 'offset' or 'none'")
        if (self.direction is not None and other.direction is not None
                and self.direction != other.direction):
            raise ValueError("Cannot merge stacks with different pass directions")
        if min(self.sizes["y"], self.sizes["x"],
               other.sizes["y"], other.sizes["x"]) < 2:
            raise ValueError("Merging needs stacks with at least 2 pixels along y and x")

        # Line other's pairs up with self's (same acquisitions, maybe reordered),
        # then relabel them with self's pair identity for the shared union grid.
        order = _match_pairs(self.ds, other.ds, time_tolerance)
        if tie == "auto":
            # Only the same acquisitions can differ by a pure integer ambiguity.
            tie = "cycles" if time_tolerance is not None else "offset"
        if time_tolerance is None:
            warnings.warn(
                "Merging by pair position without matching acquisition times: "
                f"self pair 0 is {self.ds['ref_time'].values[0]} -> "
                f"{self.ds['sec_time'].values[0]} and other pair 0 is "
                f"{other.ds['ref_time'].values[0]} -> "
                f"{other.ds['sec_time'].values[0]}. These measure different "
                f"intervals, so the merged field is a mosaic tied by {tie!r}, "
                "not one interferogram; the merged stack keeps self's times. "
                "For a slip inversion prefer sampling each scene separately "
                "and combining with Observations.concat, which gives each its "
                "own offset/ramp columns.",
                RuntimeWarning,
                stacklevel=2,
            )
        o = other.ds.isel(pair=order).assign_coords(
            pair=self.ds["pair"].values,
            ref_time=("pair", np.asarray(self.ds["ref_time"].values)),
            sec_time=("pair", np.asarray(self.ds["sec_time"].values)),
        )

        cross_zone = int(self.epsg) != int(other.epsg)
        if cross_zone:
            offsets = ()
        else:
            # Raises if the spacings differ -- two stages multilooked with
            # different ``looks`` are not the same product and no regridding
            # here would make them one.
            offsets = lattice_offset(self.x, self.y, o)
        offset_lattice = any(off > LATTICE_TOL_PX for off in offsets)
        if offset_lattice:
            warnings.warn(
                "Merging stacks whose grids are offset by a sub-pixel amount "
                f"(x {offsets[0]:.3g} px, y {offsets[1]:.3g} px): the second "
                f"stack is being resampled onto the first's grid with "
                f"{resampling!r}. Frames of one track share a lattice by "
                "construction when the interferograms are formed with "
                "align_looks=True (the default since 2026-07-27); a stack "
                "multilooked before that anchored its grid to its own crop "
                "origin. Re-forming both frames' interferograms avoids the "
                "resampling entirely.",
                RuntimeWarning,
                stacklevel=2,
            )

        if cross_zone or offset_lattice:
            # One target lattice for every layer, or the padded grids would not
            # line up with each other.
            tx, ty = warp_target_lattice(
                self.x, self.y, o["x"].values, o["y"].values,
                other.epsg, self.epsg,
            )
            o = xr.Dataset(
                {
                    name: warp_onto_lattice(
                        o[name], tx, ty, other.epsg, self.epsg, resampling
                    )
                    for name in o.data_vars
                    if name != "spatial_ref" and "y" in o[name].dims
                },
                coords={
                    "pair": o["pair"].values, "y": ty, "x": tx,
                    "ref_time": ("pair", np.asarray(o["ref_time"].values)),
                    "sec_time": ("pair", np.asarray(o["sec_time"].values)),
                },
            )

        union_y = _union_lattice(self.y, o["y"].values)
        union_x = _union_lattice(self.x, o["x"].values)

        # Pad both onto the union grid (identical coords), read the per-pair
        # offset from the overlap, and shift other to close the step.
        two_pi = 2.0 * np.pi
        a_unw = _pad_onto(self.ds["unw"], union_y, union_x)
        b_unw = _pad_onto(o["unw"], union_y, union_x)
        if tie != "none":
            diff = (a_unw - b_unw).chunk({"y": -1, "x": -1})
            shift = diff.median(dim=("y", "x"), skipna=True)
            if tie == "cycles":
                # One interferogram unwrapped per frame: the two can differ by
                # nothing but an integer ambiguity, so rounding is a correction,
                # not an approximation.
                shift = np.round(shift / two_pi) * two_pi
            # No overlap (or an all-NaN one) leaves the median undefined; a zero
            # shift is the only defensible answer and keeps the merge lazy.
            b_unw = b_unw + shift.fillna(0.0)
        self_has = a_unw.notnull()  # every layer follows the phase footprint

        merged = {"unw": xr.where(self_has, a_unw, b_unw).astype(np.float32)}

        base = self.ds["conncomp"].max() if "conncomp" in self.ds else 0
        for name in self.ds.data_vars:
            if name in ("unw", "spatial_ref") or name not in o.data_vars:
                continue
            integer = np.issubdtype(self.ds[name].dtype, np.integer)
            b_var = o[name]
            if name == "conncomp":  # keep the two frames' labels distinct
                b_var = xr.where(b_var > 0, b_var + base, 0).astype(self.ds[name].dtype)
            fill = 0 if integer else np.nan
            a_pad = _pad_onto(self.ds[name], union_y, union_x, fill=fill)
            b_pad = _pad_onto(b_var, union_y, union_x, fill=fill)
            merged[name] = xr.where(self_has, a_pad, b_pad).astype(self.ds[name].dtype)

        ds = xr.Dataset(
            merged,
            coords={
                "pair": self.ds["pair"].values,
                "y": union_y, "x": union_x,
                "ref_time": ("pair", np.asarray(self.ds["ref_time"].values)),
                "sec_time": ("pair", np.asarray(self.ds["sec_time"].values)),
            },
        )
        ds = ds.rio.write_crs(f"EPSG:{self.epsg}")
        ds.attrs.update(self.ds.attrs)
        # A merged stack's geometry is no longer shared across the footprint, so a
        # GUNW's to_los must sample each frame's own cube: keep both frames' files.
        if self.ds.attrs.get("source_files") and other.ds.attrs.get("source_files"):
            ds.attrs["source_files"] = (
                list(self.ds.attrs["source_files"])
                + list(other.ds.attrs["source_files"])
            )
        merges = list(self.ds.attrs.get("merged", []))
        record = {
            "other_epsg": int(other.epsg),
            "other_npair": int(other.sizes["pair"]),
            "other_y": [float(np.min(o["y"].values)), float(np.max(o["y"].values))],
            "other_x": [float(np.min(o["x"].values)), float(np.max(o["x"].values))],
        }
        if cross_zone or offset_lattice:
            # Only recorded when a warp actually ran, so a merge of two stacks
            # that already shared a lattice keeps the hash it had before
            # cross-zone (and now off-lattice) support existed.
            record["resampling"] = str(resampling)
        if tie != "cycles":
            # Same reasoning: the default tie stays out of the record, so a
            # same-acquisition merge keeps its pre-existing hash.
            record["tie"] = tie
        if time_tolerance is None:
            # The merged stack carries self's times, so the other frame's dates
            # would otherwise be lost -- and they are what makes this a mosaic
            # of two interferograms rather than one.
            record["other_ref_time"] = [
                str(t) for t in np.asarray(other.ds["ref_time"].values)
            ]
            record["other_sec_time"] = [
                str(t) for t in np.asarray(other.ds["sec_time"].values)
            ]
        merges.append(record)
        ds.attrs["merged"] = merges
        return UnwrappedStack(ds)

    # -- persistence -------------------------------------------------------
    def persist(self, workspace, name=None, overwrite=False, **params):
        """Write the stack to the workspace and return the reopened lazy stack.

        :meth:`from_interferograms` already writes its own store, so this is for a
        *derived* stack -- masked, cleaned, derampled, cycle-shifted, or a
        just-read GUNW. Persist under a new stage name; writing back over the
        store it reads from is refused.
        """
        name = name or self.STAGE
        ds = self.ds.chunk(self.disk_chunks("pair"))
        full = {
            "stage": name,
            "epsg": self.epsg,
            "source": self.ds.attrs.get("source"),
            "looks": self.ds.attrs.get("looks"),
            "pairs": self.ds.attrs.get("pairs"),
            **params,
        }
        # Provenance of each lazy transform, folded into the hash only once
        # applied, so an untouched stage keeps its own identity.
        for key in ("water_mask", "cycle_shifts", "phase_screen_removed",
                    "edges_masked", "outliers_removed", "deramp", "merged"):
            value = self.ds.attrs.get(key)
            if value:
                full[key] = value
        reopened = workspace.store(name, ds, full, overwrite=overwrite)
        return UnwrappedStack(reopened)

    # -- LOS conversion ----------------------------------------------------
    def to_los(self, gslc=None, dem=None, frequency="A", wavelength=None, sign=1,
               mask_geometry=True):
        """Convert to LOS displacement + per-pixel look geometry.

        The geometry source depends on provenance (``attrs["source"]``):

        - **GUNW** stacks are self-contained -- the geometry cube and wavelength
          come from the product's own ``metadata/radarGrid``, so no ``gslc`` is
          needed (``gslc`` overrides which GUNW file supplies the cube).
        - **SNAPHU** stacks need a ``gslc`` granule (one per frame for a merged
          stack) for the geometry cube.

        ``dem``, ``sign`` and ``mask_geometry`` behave as in
        :meth:`LOSStack.from_unwrapped <nisar_tools.los.LOSStack.from_unwrapped>`.
        """
        from .los import LOSStack

        source = self.ds.attrs.get("source")
        if source is None:
            source = "gunw" if self.ds.attrs.get("source_files") else "snaphu"

        if source == "gunw":
            if gslc is None:
                files = self.ds.attrs.get("source_files")
                if not files:
                    raise ValueError(
                        "GUNW-derived stack has no recorded source files; pass "
                        "gslc= pointing at the GUNW .h5 file for the geometry cube."
                    )
                # One cube per frame: a single-frame GUNW has one file, a merged
                # one has both, each covering its own part of the footprint.
                gslc = list(files)
            if wavelength is None:
                wavelength = self.ds.attrs.get("wavelength")
            product = "GUNW"
        else:
            if gslc is None:
                raise ValueError(
                    "This unwrapped stack was built from GSLCs with SNAPHU; "
                    "to_los needs a gslc= granule (one per frame) for the "
                    "geometry cube."
                )
            product = "GSLC"

        return LOSStack.from_unwrapped(
            self, gslc, dem=dem, frequency=frequency, wavelength=wavelength,
            sign=sign, mask_geometry=mask_geometry, product=product,
        )

    # -- export -----------------------------------------------------------
    def _grd_specs(self):
        """Every layer present, per pair: ``unw`` always, then whichever of
        ``coherence`` / ``phase_screen`` / ``conncomp`` / ``subswath_mask`` the
        stack carries (a SNAPHU result has only ``unw`` + ``conncomp``; a GUNW
        adds coherence and, when present, the ionosphere screen and mask)."""
        specs = [("unw", self.ds["unw"], True)]
        for v in ("coherence", "phase_screen", "conncomp", "subswath_mask"):
            if v in self.ds.data_vars:
                specs.append((v, self.ds[v], True))
        return specs

    # -- reprojection / plotting ------------------------------------------
    def to_latlon(self, pair=0):
        """Reproject a single pair's unwrapped phase to lon/lat (eager)."""
        from . import geo

        return geo.project_to_latlon(self.ds["unw"].isel(pair=pair))

    def plot(self, pair=0):
        from .plot import plot_unwrapped_phase

        return plot_unwrapped_phase(self.ds["unw"].isel(pair=pair), epsg_code=self.epsg)

    def __repr__(self):
        s = self.sizes
        src = self.ds.attrs.get("source", "?")
        return (
            f"<UnwrappedStack source={src} EPSG:{self.epsg} "
            f"pair={s.get('pair')} y={s.get('y')} x={s.get('x')}>"
        )


# SNAPHU writes warnings and errors to the same stream (``sp0`` is stderr), and
# snaphu-py re-raises the whole captured buffer as the exception message. This one
# is emitted on *every* tiled run -- snaphu warns below TILEOVRLPWARNTHRESH = 400,
# above any overlap we would sensibly ask for -- so it is always the first line of
# a failure and always a red herring. Strip it so the real fault is what surfaces.
_SNAPHU_NOISE = (
    "WARNING: Tile overlap is small (may give bad results)",
    "only one tile--disregarding tile overlap values",
    "only one tile--disregarding multiprocessor option",
)


def _plane_kernel(func, field, depth, target_blocks=None, **kwargs):
    """Run a 2-D plane kernel over a ``(pair, y, x)`` DataArray, chunk by chunk.

    Wraps :func:`_kernels.halo_planes`, which overlaps the spatial axes by
    ``depth`` so the kernel decomposes across chunks instead of forcing one whole
    plane per task. Dims, coords and attrs are preserved.

    A persisted stack arrives on the 2048-px *disk* chunk, which for a multilooked
    raster is usually the whole plane -- so it is rechunked to a working size first
    (see :func:`_base.compute_chunks`), or there would be nothing to decompose.
    """
    if _kernels._is_dask(field.data):
        working = compute_chunks(
            field.sizes["y"], field.sizes["x"], depth, target_blocks
        )
        if working is not None:
            field = field.chunk({"pair": 1, "y": working[0], "x": working[1]})
    data = _kernels.halo_planes(func, field.data, depth, **kwargs)
    return xr.DataArray(
        data, dims=field.dims, coords=field.coords, attrs=field.attrs,
        name=field.name,
    )


def _plane_kernel_rows(func, field, target_blocks=None, **kwargs):
    """Run a row-wise 2-D plane kernel over a ``(pair, y, x)`` DataArray.

    Wraps :func:`_kernels.row_planes` for a kernel whose support is a whole
    raster row, so it cannot be given a halo. ``x`` is gathered into one chunk
    and the work is split along ``y`` instead; that is exact, not
    exact-to-a-halo. Dims, coords and attrs are preserved.
    """
    if _kernels._is_dask(field.data):
        # Only y is free to split, so ask compute_chunks for the row band and
        # ignore its x suggestion. halo=1: there is no overlap to amortise.
        working = compute_chunks(
            field.sizes["y"], field.sizes["x"], halo=1, target_blocks=target_blocks
        )
        if working is not None:
            field = field.chunk({"pair": 1, "y": working[0], "x": -1})
    data = _kernels.row_planes(func, field.data, **kwargs)
    return xr.DataArray(
        data, dims=field.dims, coords=field.coords, attrs=field.attrs,
        name=field.name,
    )


def _plane_kernel_poly(field, degree, exclude, target_blocks=None):
    """Polynomial deramp of a ``(pair, y, x)`` DataArray as a chunked reduction."""
    if _kernels._is_dask(field.data):
        # No halo here (the fit is a global reduction), so the only floor on the
        # working chunk is not fragmenting the graph: one design-matrix block per
        # chunk, so a handful of coefficients' worth of scratch each.
        working = compute_chunks(
            field.sizes["y"], field.sizes["x"], halo=1,
            target_blocks=target_blocks,
        )
        if working is not None:
            field = field.chunk({"pair": 1, "y": working[0], "x": working[1]})
        data = _kernels.deramp_poly_dask(field.data, degree, exclude=exclude)
    else:
        data = _kernels.deramp_planes(
            field.data, degree=degree, method="poly", scale=None, exclude=exclude
        )
    return xr.DataArray(
        data, dims=field.dims, coords=field.coords, attrs=field.attrs,
        name=field.name,
    )


def _prefetch(load, items, lookahead=1):
    """Yield ``(item, load(item))`` with ``lookahead`` loads running ahead.

    A bounded sliding window, so peak memory tracks the window rather than
    ``len(items)`` -- the point of the whole out-of-core design. The loads run on
    their own threads, which is what lets a pair's read/multilook/filter chain
    overlap the SNAPHU call on the pair before it.
    """
    lookahead = max(1, int(lookahead))
    pool = ThreadPoolExecutor(lookahead, thread_name_prefix="unwrap-load")
    try:
        pending = iter(items)
        window = deque(
            (item, pool.submit(load, item))
            for item in itertools.islice(pending, lookahead + 1)
        )
        while window:
            item, future = window.popleft()
            for nxt in itertools.islice(pending, 1):
                window.append((nxt, pool.submit(load, nxt)))
            yield item, future.result()
    finally:
        # A consumer that raises must not leave loader threads running.
        pool.shutdown(wait=False, cancel_futures=True)


def _snaphu_error_detail(message):
    """The lines of a SNAPHU stderr buffer that actually describe the failure."""
    lines = [
        line for line in str(message).splitlines()
        if line.strip() and line.strip() not in _SNAPHU_NOISE
    ]
    return "\n".join(lines) or str(message).strip() or "(no message)"


def _unwrap_pair(igram, corr, *, nlooks, ntiles, tile_overlap, nproc, pair=None,
                 **snaphu_kwargs):
    """Unwrap one pair, keeping its invalid pixels out of the solution.

    SNAPHU silently substitutes zeros for NaN and returns a finite value
    everywhere, so without a mask the area outside the swath comes back as
    plausible-looking phase that is indistinguishable downstream. Its ``mask``
    argument excludes those pixels properly; we then restore NaN so the invalid
    footprint survives into :class:`~nisar_tools.los.LOSStack`.

    The mask is passed only when something is actually invalid, so a fully
    valid pair takes exactly the call it did before.
    """
    valid = np.isfinite(igram.real) & np.isfinite(igram.imag)

    if not valid.any():
        # SNAPHU has nothing to solve; skip it rather than let it fail.
        return (
            np.full(igram.shape, np.nan, dtype=np.float32),
            np.zeros(igram.shape, dtype=np.uint32),
        )

    kwargs = {} if valid.all() else {"mask": valid}
    try:
        unw, conncomp = snaphu.unwrap(
            igram,
            corr,
            nlooks=nlooks,
            ntiles=ntiles,
            tile_overlap=tile_overlap,
            nproc=nproc,
            **snaphu_kwargs,
            **kwargs,
        )
    except RuntimeError as exc:
        (ni, nj), _ = _kernels.snaphu_tile_shape(igram.shape, ntiles, tile_overlap)
        where = "" if pair is None else f"pair {pair}: "
        raise RuntimeError(
            f"{where}SNAPHU failed on a {igram.shape[0]}x{igram.shape[1]} raster "
            f"with ntiles={tuple(ntiles)}, tile_overlap={tile_overlap} "
            f"(tile {ni}x{nj} = {ni * nj} px), nlooks={nlooks}, nproc={nproc}:\n"
            f"{_snaphu_error_detail(exc)}"
        ) from exc

    unw = unw.astype(np.float32)
    conncomp = conncomp.astype(np.uint32)
    if not valid.all():
        unw[~valid] = np.nan
        conncomp[~valid] = 0
    return unw, conncomp


def _template(igram_ds, npair, ny, nx):
    """Lazy zero-filled template carrying the final shape, dtypes and coords."""
    chunks = (1, min(SPATIAL_CHUNK, ny), min(SPATIAL_CHUNK, nx))
    unw = da.zeros((npair, ny, nx), chunks=chunks, dtype=np.float32)
    conncomp = da.zeros((npair, ny, nx), chunks=chunks, dtype=np.uint32)
    coherence = da.zeros((npair, ny, nx), chunks=chunks, dtype=np.float32)

    template = xr.Dataset(
        {
            "unw": (("pair", "y", "x"), unw),
            "conncomp": (("pair", "y", "x"), conncomp),
            # The interferogram's coherence, carried through unchanged so a
            # SNAPHU stack -- like a GUNW -- can drive mask_edges(min_coherence).
            "coherence": (("pair", "y", "x"), coherence),
        },
        coords={
            "pair": igram_ds["pair"].values,
            "y": igram_ds["y"].values,
            "x": igram_ds["x"].values,
            "ref_time": ("pair", np.asarray(igram_ds["ref_time"].values)),
            "sec_time": ("pair", np.asarray(igram_ds["sec_time"].values)),
        },
    )
    template = template.rio.write_crs(f"EPSG:{int(igram_ds.attrs['epsg'])}")
    template.attrs.update(
        epsg=int(igram_ds.attrs["epsg"]),
        source="snaphu",
        direction=igram_ds.attrs.get("direction"),
        looks=igram_ds.attrs.get("looks"),
    )
    return template
