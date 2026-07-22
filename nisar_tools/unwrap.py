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
from ._base import SPATIAL_CHUNK, RasterStackMixin


class UnwrappedStack(RasterStackMixin):
    """A stack of unwrapped phases with connected-component labels."""

    STAGE = "unwrapped"

    def __init__(self, ds):
        self.ds = ds

    @classmethod
    def from_zarr(cls, path):
        return cls(xr.open_zarr(path))

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
    def mask_water(self, mask_cache=None, resolution="f", spacing="5e"):
        """Lazily mask water on the unwrapped phase. Returns a new stack.

        Lazy: the masked values are **not** written anywhere. Call
        :meth:`persist` (under a new stage name) if you want them on disk;
        otherwise reloading this stage gives the unmasked phase back.

        ``mask_cache`` is a :class:`~nisar_tools.workspace.Workspace` used to
        cache the *coastline mask itself*, keyed on the grid, so GMT is not
        re-run for the same crop. It is not where the masked data goes.

        ``resolution`` is the GMT coastline resolution; use a coarser value
        (e.g. ``"i"``) if the full-resolution GSHHG dataset is unavailable.
        """
        from .mask import water_mask_for_grid

        mask = water_mask_for_grid(
            self.x, self.y, self.epsg, workspace=mask_cache,
            resolution=resolution, spacing=spacing,
        )
        ds = self.ds.copy()
        # The mask is land=1 / water=NaN; ``where`` needs a boolean condition
        # (NaN is truthy, so passing the raw mask would keep water pixels).
        ds["unw"] = self.ds["unw"].where(mask.notnull())
        ds.attrs.update(self.ds.attrs)
        ds.attrs["water_mask"] = {"resolution": resolution, "spacing": spacing}
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
        # Only recorded once masked, so an unmasked stage keeps its own hash.
        if self.ds.attrs.get("water_mask") is not None:
            full["water_mask"] = self.ds.attrs["water_mask"]
        reopened = workspace.store(name, ds, full, overwrite=overwrite)
        return UnwrappedStack(reopened)

    def to_los(self, gslc, dem=None, frequency="A", wavelength=None, sign=1):
        """Convert to LOS displacement + per-pixel look geometry.

        Scales the unwrapped phase to metres and attaches the incidence angle
        and ENU line-of-sight unit vector interpolated from ``gslc``'s built-in
        geometry cube at the ``dem`` height. See :class:`LOSStack`.
        """
        from .los import LOSStack

        return LOSStack.from_unwrapped(
            self, gslc, dem=dem, frequency=frequency,
            wavelength=wavelength, sign=sign,
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
