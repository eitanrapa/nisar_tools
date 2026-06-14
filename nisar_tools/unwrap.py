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
        spacing_az = float(ds.attrs.get("x_spacing", 1.0))
        spacing_rg = float(ds.attrs.get("y_spacing", 1.0))

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
        workspace.init_store(name, template, params, overwrite=overwrite)

        done = workspace.pairs_done(name)
        for i in range(npair):
            if i in done:
                continue
            igram = np.asarray(ds["igram"].isel(pair=i).values)
            corr = np.asarray(ds["coherence"].isel(pair=i).values)

            unw, conncomp = snaphu.unwrap(
                igram,
                corr,
                nlooks=nlooks,
                ntiles=ntiles,
                tile_overlap=overlap,
                nproc=nproc,
            )

            pair_ds = xr.Dataset(
                {
                    "unw": (("pair", "y", "x"), unw[None].astype(np.float32)),
                    "conncomp": (("pair", "y", "x"), conncomp[None].astype(np.uint32)),
                }
            )
            workspace.write_region(name, pair_ds, region={"pair": slice(i, i + 1)})
            workspace.mark_pair_done(name, i)

        workspace.consolidate(name)
        return cls.from_zarr(workspace.path(name))

    # -- operations --------------------------------------------------------
    def mask_water(self, workspace=None):
        from .mask import water_mask_for_grid

        mask = water_mask_for_grid(self.x, self.y, self.epsg, workspace=workspace)
        ds = self.ds.copy()
        ds["unw"] = self.ds["unw"].where(mask)
        ds.attrs.update(self.ds.attrs)
        return UnwrappedStack(ds)

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
