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

import os

import numpy as np
import rioxarray  # noqa: F401
import xarray as xr

from ._base import RasterStackMixin, open_stage

_GEOM_2D = ("incidence_angle", "look_angle", "los_east", "los_north",
            "los_up", "height")


def _as_granule_list(gslc):
    """Normalise a granule argument to a list of paths.

    A single path is by far the common case, but a merged stack needs one per
    frame. Strings and ``os.PathLike`` are scalars here -- a string is iterable,
    so testing for iterability alone would silently split a path into
    characters.
    """
    if isinstance(gslc, (str, os.PathLike)):
        return [gslc]
    granules = list(gslc)
    if not granules:
        raise ValueError("Need at least one GSLC granule for the geometry cube")
    return granules


class LOSStack(RasterStackMixin):
    """LOS displacement (per pair) plus shared per-pixel look geometry."""

    STAGE = "los"

    def __init__(self, ds):
        self.ds = ds

    @classmethod
    def from_unwrapped(cls, unwrapped, gslc, dem=None, frequency="A",
                       wavelength=None, sign=1, mask_geometry=True,
                       product="GSLC"):
        """Build a :class:`LOSStack` from an unwrapped stack.

        ``gslc`` is a granule path supplying the geometry cube and (unless
        ``wavelength`` is given) the radar wavelength. For a stack built by
        :meth:`~nisar_tools.stack.GSLCStack.merge`, pass **one granule per
        frame**: each cube only spans its own frame, so a single granule leaves
        the rest of a merged stack without geometry. The cubes are sampled in
        order and combined, earlier granules taking precedence where they
        overlap, matching ``merge``'s own rule.

        ``product`` names the granule's product group (``"GSLC"``, or ``"GUNW"``
        when the geometry comes from a NASA GUNW's own embedded cube). ``dem`` is
        a GeoTIFF path or DataArray of ellipsoidal heights (``None`` -> sea-level
        geometry). ``sign`` flips the LOS displacement convention (see
        :func:`nisar_tools.geometry.phase_to_los`).

        ``mask_geometry`` (default) blanks the geometry outside the data. The
        cube spans the frame's whole bounding rectangle and knows nothing about
        where the radar actually had returns, so interpolating it fills every
        pixel -- which plots as a solid rectangle bearing no resemblance to the
        swath, and quietly reports an incidence angle for ground the pass never
        illuminated. Pass ``False`` to keep the full rectangle.
        """
        from . import geometry

        du = unwrapped.ds
        x, y, epsg = unwrapped.x, unwrapped.y, unwrapped.epsg
        granules = _as_granule_list(gslc)
        if wavelength is None:
            wavelength = geometry.radar_wavelength(granules[0], frequency, product)

        heights = geometry.dem_heights_on_grid(dem, x, y, epsg)
        # One cube per frame, sampled onto the target grid and stacked. Each
        # cube is NaN outside its own frame, so this fills a merged stack;
        # holding one at a time keeps peak memory at a single cube.
        geom = None
        look_direction = None
        for path in granules:
            cube = geometry.read_geometry_cube(path, frequency, product)
            sampled = geometry.sample_look_geometry(cube, x, y, epsg, heights)
            geom = sampled if geom is None else geom.fillna(sampled)
            if look_direction is None:
                look_direction = cube.attrs.get("look_direction")
            del cube, sampled

        los = geometry.phase_to_los(du["unw"], wavelength, sign=sign)
        los = los.astype(np.float32).rename("los")

        # Drop any CRS coord before combining so the two sources' spatial_ref
        # don't collide on merge; write the CRS once on the result.
        los = los.drop_vars("spatial_ref", errors="ignore")
        geom = geom.drop_vars("spatial_ref", errors="ignore")

        if mask_geometry:
            # Geometry is one field shared by every pair, so keep it wherever
            # *any* pair has data rather than only where all of them do.
            footprint = los.notnull().any("pair")
            geom = geom.where(footprint)

        ds = xr.Dataset({"los": los, **{v: geom[v] for v in _GEOM_2D}})
        ds = ds.rio.write_crs(f"EPSG:{int(epsg)}")
        ds.attrs.update(
            epsg=int(epsg),
            direction=du.attrs.get("direction"),
            wavelength=float(wavelength),
            frequency=frequency,
            sign=int(sign),
            look_direction=look_direction,
            granules=[str(p) for p in granules],
            pairs=du.attrs.get("pairs"),
        )
        return cls(ds)

    @classmethod
    def from_zarr(cls, path):
        return cls(open_stage(path))

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
