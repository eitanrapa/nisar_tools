"""The :class:`LOSStack`: line-of-sight displacement + per-pixel look geometry.

Produced from an :class:`~nisar_tools.unwrap.UnwrappedStack` by scaling the
unwrapped phase to metres and attaching the incidence angle and ENU
line-of-sight unit vector sampled from a GSLC's built-in geometry cube at the
DEM height (see :mod:`nisar_tools.geometry`).

The ``los`` displacement is per-pair ``(pair, y, x)`` and stays lazy; the
geometry (``incidence_angle``, ``look_angle``, ``los_east``/``los_north``/
``los_up``, ``height``) is one field per grid ``(y, x)`` -- the viewing geometry
is shared across the repeat-pass stack -- and is computed eagerly, once.

``incidence_angle`` and ``look_angle`` are not the same thing: incidence is
measured at the target, between the line of sight and the local vertical, while
the look (off-nadir) angle is measured at the spacecraft, against the ellipsoid
normal there. Earth curvature makes the look angle the smaller of the two.
"""

import numpy as np
import rioxarray  # noqa: F401
import xarray as xr

from ._base import RasterStackMixin

_GEOM_2D = ("incidence_angle", "look_angle", "los_east", "los_north",
            "los_up", "height")


class LOSStack(RasterStackMixin):
    """LOS displacement (per pair) plus shared per-pixel look geometry."""

    STAGE = "los"

    def __init__(self, ds):
        self.ds = ds

    @classmethod
    def from_unwrapped(cls, unwrapped, gslc, dem=None, frequency="A",
                       wavelength=None, sign=1):
        """Build a :class:`LOSStack` from an unwrapped stack.

        ``gslc`` is a granule path supplying the geometry cube and (unless
        ``wavelength`` is given) the radar wavelength. ``dem`` is a GeoTIFF path
        or DataArray of ellipsoidal heights (``None`` -> sea-level geometry).
        ``sign`` flips the LOS displacement convention (see
        :func:`nisar_tools.geometry.phase_to_los`).
        """
        from . import geometry

        du = unwrapped.ds
        x, y, epsg = unwrapped.x, unwrapped.y, unwrapped.epsg
        if wavelength is None:
            wavelength = geometry.radar_wavelength(gslc, frequency)

        cube = geometry.read_geometry_cube(gslc, frequency)
        heights = geometry.dem_heights_on_grid(dem, x, y, epsg)
        geom = geometry.sample_look_geometry(cube, x, y, epsg, heights)

        los = geometry.phase_to_los(du["unw"], wavelength, sign=sign)
        los = los.astype(np.float32).rename("los")

        # Drop any CRS coord before combining so the two sources' spatial_ref
        # don't collide on merge; write the CRS once on the result.
        los = los.drop_vars("spatial_ref", errors="ignore")
        geom = geom.drop_vars("spatial_ref", errors="ignore")

        ds = xr.Dataset({"los": los, **{v: geom[v] for v in _GEOM_2D}})
        ds = ds.rio.write_crs(f"EPSG:{int(epsg)}")
        ds.attrs.update(
            epsg=int(epsg),
            direction=du.attrs.get("direction"),
            wavelength=float(wavelength),
            frequency=frequency,
            sign=int(sign),
            look_direction=cube.attrs.get("look_direction"),
            pairs=du.attrs.get("pairs"),
        )
        return cls(ds)

    @classmethod
    def from_zarr(cls, path):
        return cls(xr.open_zarr(path))

    # -- persistence -------------------------------------------------------
    def persist(self, workspace, name=None, overwrite=False, **params):
        name = name or self.STAGE
        ds = self.ds.chunk(self.disk_chunks("pair"))
        full = {
            "stage": name,
            "epsg": self.epsg,
            "wavelength": self.ds.attrs.get("wavelength"),
            "frequency": self.ds.attrs.get("frequency"),
            "sign": self.ds.attrs.get("sign"),
            "pairs": self.ds.attrs.get("pairs"),
            **params,
        }
        reopened = workspace.store(name, ds, full, overwrite=overwrite)
        return LOSStack(reopened)

    # -- reprojection / plotting ------------------------------------------
    def to_latlon(self, pair=0):
        """Reproject one pair's LOS displacement to lon/lat (eager)."""
        from . import geo

        return geo.project_to_latlon(self.ds["los"].isel(pair=pair))

    def plot(self, pair=0):
        from .plot import plot_los_displacement

        return plot_los_displacement(self.ds["los"].isel(pair=pair), epsg_code=self.epsg)

    def plot_incidence(self):
        """Incidence angle: at the target, from the local vertical."""
        from .plot import plot_angle

        return plot_angle(
            self.ds["incidence_angle"], epsg_code=self.epsg,
            title="Incidence angle", label="Incidence (deg)",
        )

    def plot_look_angle(self):
        """Look (off-nadir) angle: at the spacecraft, from the ellipsoid normal.

        Smaller than the incidence angle by the Earth-curvature term.
        """
        from .plot import plot_angle

        return plot_angle(
            self.ds["look_angle"], epsg_code=self.epsg,
            title="Look angle", label="Look angle (deg)",
        )

    def __repr__(self):
        s = self.sizes
        wl = self.ds.attrs.get("wavelength")
        wl = f"{wl:.4f}m" if wl is not None else "?"
        return (
            f"<LOSStack EPSG:{self.epsg} lambda={wl} "
            f"pair={s.get('pair')} y={s.get('y')} x={s.get('x')}>"
        )
