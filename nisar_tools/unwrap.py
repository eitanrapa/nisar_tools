"""The :class:`UnwrappedStack` and the per-pair SNAPHU driver.

Unwrapping is the one stage that cannot be lazy: SNAPHU is a global optimiser
that needs a whole raster. So the *pair* is the unit of work. The output store
is created metadata-only up front, then each pair is unwrapped, written into
its own region, and flagged done. Peak memory is therefore one multilooked
pair, regardless of how many acquisitions the stack contains, and an
interrupted run resumes at the first unfinished pair.
"""

import dask.array as da
import numpy as np
import rioxarray  # noqa: F401
import snaphu
import xarray as xr

from . import _kernels
from ._base import SPATIAL_CHUNK, RasterStackMixin, open_stage


class UnwrappedStack(RasterStackMixin):
    """A stack of unwrapped phases with connected-component labels."""

    STAGE = "unwrapped"

    def __init__(self, ds):
        self.ds = ds

    @classmethod
    def from_zarr(cls, path):
        return cls(open_stage(path))

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

    # -- operations --------------------------------------------------------
    def mask_water(self, mask_cache=None, resolution="f", spacing=None,
                   mask_name=None):
        """Lazily mask water on the unwrapped phase. Returns a new stack.

        Lazy: the masked values are **not** written anywhere. Call
        :meth:`persist` (under a new stage name) if you want them on disk;
        otherwise reloading this stage gives the unmasked phase back.

        ``mask_cache`` is a :class:`~nisar_tools.workspace.Workspace` used to
        cache the *coastline mask itself*, keyed on the grid, so GMT is not
        re-run for the same crop. It is not where the masked data goes.

        ``resolution`` is the GMT coastline resolution; use a coarser value
        (e.g. ``"i"``) if the full-resolution GSHHG dataset is unavailable.
        ``spacing`` defaults to tracking this stack's own pixel size, so a
        multilooked stack builds a correspondingly coarse coastline.
        ``mask_name`` overrides the cache store's name, which otherwise is
        derived from the grid so masks for different grids coexist.
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

    def add_cycles(self, cycles, pair=None, conncomp=None):
        """Shift the unwrapped phase by an integer number of 2*pi cycles.

        Unwrapping recovers phase only up to a global multiple of 2*pi, and
        SNAPHU resolves each *connected component* independently, so distinct
        components can sit whole cycles apart from one another with no way to
        tell from the data alone. This is how you apply the offset once you
        know it -- from a GPS station, a known-stable area, or a neighbouring
        component that should be continuous with this one.

        ``cycles`` is added, so pass a negative value to remove cycles. It must
        be a whole number: any other shift changes the wrapped phase and no
        longer describes the same interferogram.

        ``pair`` selects pair indices (default all) and ``conncomp`` selects
        connected-component labels (default the whole raster). Component 0 is
        SNAPHU's "not assigned to any component" label, so shifting it is
        usually a mistake -- it is allowed, but it is not a real region.

        Lazy, like :meth:`mask_water`: it returns a new stack and writes
        nothing. The shift carries into :meth:`to_los` (one cycle is half a
        wavelength of range change), so apply it before converting.
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

        :meth:`from_interferograms` already writes its own store, so this is
        for a *derived* stack -- most often a water-masked one, which is lazy.
        Persist it under a new stage name: writing back over the store it reads
        from is refused.
        """
        name = name or self.STAGE
        ds = self.ds.chunk(self.disk_chunks("pair"))
        full = {
            "stage": name,
            "epsg": self.epsg,
            "looks": self.ds.attrs.get("looks"),
            "pairs": self.ds.attrs.get("pairs"),
            **params,
        }
        # Only recorded once applied, so an untouched stage keeps its own hash.
        if self.ds.attrs.get("water_mask") is not None:
            full["water_mask"] = self.ds.attrs["water_mask"]
        if self.ds.attrs.get("cycle_shifts"):
            full["cycle_shifts"] = self.ds.attrs["cycle_shifts"]
        reopened = workspace.store(name, ds, full, overwrite=overwrite)
        return UnwrappedStack(reopened)

    def to_los(self, gslc, dem=None, frequency="A", wavelength=None, sign=1,
               mask_geometry=True):
        """Convert to LOS displacement + per-pixel look geometry.

        Scales the unwrapped phase to metres and attaches the incidence angle,
        look angle and ENU line-of-sight unit vector interpolated from
        ``gslc``'s built-in geometry cube at the ``dem`` height.

        ``gslc`` is one granule path, or **one per frame** for a merged stack:
        each cube spans only its own frame, so a single granule leaves the rest
        of a merged stack without geometry. ``mask_geometry`` blanks the
        geometry outside the data footprint; see
        :meth:`LOSStack.from_unwrapped <nisar_tools.los.LOSStack.from_unwrapped>`.
        """
        from .los import LOSStack

        return LOSStack.from_unwrapped(
            self, gslc, dem=dem, frequency=frequency,
            wavelength=wavelength, sign=sign, mask_geometry=mask_geometry,
        )

    def to_latlon(self, pair=0):
        """Reproject a single pair's unwrapped phase to lon/lat (eager)."""
        from . import geo

        return geo.project_to_latlon(self.ds["unw"].isel(pair=pair))

    def plot(self, pair=0):
        from .plot import plot_unwrapped_phase

        return plot_unwrapped_phase(self.ds["unw"].isel(pair=pair), epsg_code=self.epsg)

    def __repr__(self):
        s = self.sizes
        return (
            f"<UnwrappedStack EPSG:{self.epsg} "
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
        direction=igram_ds.attrs.get("direction"),
        looks=igram_ds.attrs.get("looks"),
    )
    return template
