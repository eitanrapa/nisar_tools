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

import os

import dask.array as da
import h5py
import numpy as np
import rioxarray  # noqa: F401
import snaphu
import xarray as xr

from . import _kernels, geometry
from ._base import SPATIAL_CHUNK, RasterStackMixin, open_stage

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
        overwrite=False,
    ):
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
        ntiles, overlap = _kernels.snaphu_params((ny, nx), nproc)

        params = {
            "stage": name,
            "epsg": int(ds.attrs["epsg"]),
            "source": "snaphu",
            "looks": looks,
            "nlooks": nlooks,
            "res_az": res_az,
            "res_rg": res_rg,
            "pairs": ds.attrs.get("pairs"),
        }

        # Metadata-only store so each pair can be written by region.
        template = _template(ds, npair, ny, nx)
        workspace.init_store(
            name, template, params, overwrite=overwrite, source=ds
        )

        done = workspace.pairs_done(name)
        for i in range(npair):
            if i in done:
                continue
            igram = np.asarray(ds["igram"].isel(pair=i).values)
            corr = np.asarray(ds["coherence"].isel(pair=i).values)
            unw, conncomp = _unwrap_pair(
                igram, corr, nlooks=nlooks, ntiles=ntiles,
                tile_overlap=overlap, nproc=nproc,
            )

            pair_ds = xr.Dataset(
                {
                    "unw": (("pair", "y", "x"), unw[None]),
                    "conncomp": (("pair", "y", "x"), conncomp[None]),
                }
            )
            workspace.write_region(name, pair_ds, region={"pair": slice(i, i + 1)})
            workspace.mark_pair_done(name, i)

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
        # Label the NASA screen's provenance so it is distinguishable from a
        # spline screen fitted by estimate_phase_screen (dispersive-only vs a
        # broadband deramp proxy).
        if any_iono:
            ds.attrs["phase_screen_method"] = {
                "name": "phase_screen", "method": "nasa_split_spectrum",
            }
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

    def mask_edges(self, edge_pixels=8, min_coherence=None, use_builtin_mask=False):
        """Mask swath-edge effects. Returns a new stack (lazy).

        Erodes each pair's finite footprint by ``edge_pixels`` pixels
        (:func:`scipy.ndimage.binary_erosion`, also trimming the raster edge),
        removing the decorrelated fringe along the swath boundary. Works for both
        SNAPHU and GUNW stacks. ``min_coherence`` additionally nulls pixels below
        that coherence (GUNW stacks only, which carry a ``coherence`` layer).

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

        masked = xr.apply_ufunc(
            _kernels.mask_edges_planes, unw.chunk({"pair": 1, "y": -1, "x": -1}),
            kwargs={"edge_pixels": int(edge_pixels)},
            input_core_dims=[["y", "x"]], output_core_dims=[["y", "x"]],
            dask="parallelized", output_dtypes=[unw.dtype],
        )
        if min_coherence is not None:
            if "coherence" not in self.ds:
                raise ValueError(
                    "min_coherence needs a coherence layer; this stack has none "
                    "(only GUNW-derived stacks carry coherence)."
                )
            masked = masked.where(self.ds["coherence"] >= min_coherence)

        ds = self.ds.copy()
        ds["unw"] = masked
        ds.attrs.update(self.ds.attrs)
        ds.attrs["edges_masked"] = {
            "edge_pixels": int(edge_pixels), "min_coherence": min_coherence,
            "use_builtin_mask": bool(use_builtin_mask),
        }
        return UnwrappedStack(ds)

    def remove_outliers(self, scale=16.0, threshold=1.0, iterations=2):
        """Reject residual outliers against a smooth spline. Returns a new stack.

        Fits a NaN-aware smooth surface (Gaussian sigma ``scale`` px) to each
        pair's unwrapped phase, nulls pixels where ``|phase - surface|`` exceeds
        ``threshold`` radians, and refits ``iterations`` times. This is the
        tension-spline + residual-mask step of ``filt_gunw.csh``, in scipy.
        """
        unw = self.ds["unw"].chunk({"pair": 1, "y": -1, "x": -1})
        cleaned = xr.apply_ufunc(
            _kernels.remove_outliers_planes, unw,
            kwargs={"scale": float(scale), "threshold": float(threshold),
                    "iterations": int(iterations)},
            input_core_dims=[["y", "x"]], output_core_dims=[["y", "x"]],
            dask="parallelized", output_dtypes=[unw.dtype],
        )
        ds = self.ds.copy()
        ds["unw"] = cleaned
        ds.attrs.update(self.ds.attrs)
        ds.attrs["outliers_removed"] = {
            "scale": float(scale), "threshold": float(threshold),
            "iterations": int(iterations),
        }
        return UnwrappedStack(ds)

    def deramp(self, degree=1, method="poly", scale=None):
        """Remove a long-wavelength ramp (e.g. ionosphere). Returns a new stack.

        ``method="poly"`` subtracts a total-degree-``degree`` 2D polynomial (the
        classic InSAR deramp; ``1`` = plane), flattening the far field toward
        zero. ``method="spline"`` subtracts a NaN-aware smooth surface at Gaussian
        sigma ``scale`` px (a high-pass for gently curved ionosphere ramps). Best
        run after :meth:`mask_edges` / :meth:`remove_outliers` so the fit is not
        pulled by edges or spikes.
        """
        unw = self.ds["unw"].chunk({"pair": 1, "y": -1, "x": -1})
        deramped = xr.apply_ufunc(
            _kernels.deramp_planes, unw,
            kwargs={"degree": int(degree), "method": method, "scale": scale},
            input_core_dims=[["y", "x"]], output_core_dims=[["y", "x"]],
            dask="parallelized", output_dtypes=[unw.dtype],
        )
        ds = self.ds.copy()
        ds["unw"] = deramped
        ds.attrs.update(self.ds.attrs)
        ds.attrs["deramp"] = {"degree": int(degree), "method": method,
                              "scale": scale}
        return UnwrappedStack(ds)

    def estimate_phase_screen(self, method="spline", scale=None, degree=1,
                              weighted=True, name="phase_screen", overwrite=False):
        """Fit a phase screen to the unwrapped phase and carry it. Returns a new stack.

        For a GSLC-derived (SNAPHU) stack, which ships no NASA screen, this is how
        a ``phase_screen`` is produced: fit the same smooth surface
        :meth:`deramp` subtracts (:func:`nisar_tools._kernels.fit_surface`) and
        *keep* it. ``deramp(method=..., scale=..., degree=...)`` is then exactly
        ``estimate_phase_screen(...).remove_phase_screen()``.

        ``method`` / ``scale`` / ``degree`` are as in :meth:`deramp` (``"spline"``
        = NaN-aware Gaussian at sigma ``scale`` px, the default; ``"poly"`` = a
        degree-``degree`` surface). ``weighted`` uses this stack's ``coherence``
        (when present) to down-weight noisy pixels, echoing how NASA's split-
        spectrum ionosphere estimate is coherence-driven.

        The result is stored under ``name`` (``float32``, radians, ``(pair, y, x)``,
        re-masked to the ``unw`` footprint), so it is format-identical to a GUNW's
        ``ionospherePhaseScreen`` and flows through :meth:`remove_phase_screen`
        the same way. **Semantics differ**: a spline captures the *total* smooth
        trend (ionosphere plus orbital/tropo ramps and any long-wavelength
        deformation), whereas NASA's screen is the dispersive part only -- so this
        is a broadband deramp used as an ionosphere proxy, not a true dispersive
        estimate. Recorded in ``attrs["phase_screen_method"]``.

        By default it refuses to overwrite an existing ``name`` -- so a GUNW's
        NASA screen is not silently clobbered; pass a distinct ``name`` (e.g.
        ``"phase_screen_spline"``) to keep both for comparison, or
        ``overwrite=True``.
        """
        if name in self.ds and not overwrite:
            raise ValueError(
                f"{name!r} already exists on this stack (a GUNW carries NASA's "
                f"ionospherePhaseScreen there). Pass a different name= to keep "
                f"both for comparison, or overwrite=True to replace it."
            )
        chunks = {"pair": 1, "y": -1, "x": -1}
        unw = self.ds["unw"].chunk(chunks)
        kwargs = {"method": method, "scale": scale, "degree": int(degree)}

        use_weight = weighted and "coherence" in self.ds
        if use_weight:
            coherence = self.ds["coherence"].chunk(chunks)
            screen = xr.apply_ufunc(
                _kernels.fit_surface_planes, unw, coherence, kwargs=kwargs,
                input_core_dims=[["y", "x"], ["y", "x"]],
                output_core_dims=[["y", "x"]],
                dask="parallelized", output_dtypes=[unw.dtype],
            )
        else:
            screen = xr.apply_ufunc(
                _kernels.fit_surface_planes, unw, kwargs=kwargs,
                input_core_dims=[["y", "x"]], output_core_dims=[["y", "x"]],
                dask="parallelized", output_dtypes=[unw.dtype],
            )
        # Confine the screen to the data footprint (the smooth surface bleeds a
        # little past the swath edge), so unw - screen adds no new valid pixels
        # and the footprint matches a GUNW's screen.
        screen = screen.where(unw.notnull())

        ds = self.ds.copy()
        ds[name] = screen
        ds.attrs.update(self.ds.attrs)
        ds.attrs["phase_screen_method"] = {
            "name": name, "method": method, "scale": scale,
            "degree": int(degree), "weighted": bool(use_weight),
        }
        return UnwrappedStack(ds)

    def remove_phase_screen(self, name="phase_screen"):
        """Subtract a carried phase screen from the unwrapped phase (lazy).

        The screen ``name`` is either a GUNW's NASA ``ionospherePhaseScreen``
        (loaded as ``phase_screen``) or one produced by
        :meth:`estimate_phase_screen`; both are subtracted the same way
        (``unw - screen``, as in the reference ``h52grd.py``). Raises if the named
        screen is absent or was already removed. The screen layer is kept for
        inspection.
        """
        if name not in self.ds:
            raise ValueError(
                f"This stack carries no {name!r} to remove. A GUNW loads NASA's "
                "screen as 'phase_screen'; otherwise call estimate_phase_screen() "
                "first to fit one."
            )
        removed = list(self.ds.attrs.get("phase_screen_removed", []))
        if name in removed:
            raise ValueError(f"{name!r} has already been removed from this stack")

        ds = self.ds.copy()
        unw = self.ds["unw"]
        ds["unw"] = (unw - self.ds[name]).astype(unw.dtype)
        ds.attrs.update(self.ds.attrs)
        ds.attrs["phase_screen_removed"] = removed + [name]
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
                    "phase_screen_method", "edges_masked", "outliers_removed",
                    "deramp"):
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
                gslc = [files[0]]  # shared geometry -> one cube suffices
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


def _unwrap_pair(igram, corr, *, nlooks, ntiles, tile_overlap, nproc):
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
    unw, conncomp = snaphu.unwrap(
        igram,
        corr,
        nlooks=nlooks,
        ntiles=ntiles,
        tile_overlap=tile_overlap,
        nproc=nproc,
        **kwargs,
    )

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

    template = xr.Dataset(
        {
            "unw": (("pair", "y", "x"), unw),
            "conncomp": (("pair", "y", "x"), conncomp),
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
