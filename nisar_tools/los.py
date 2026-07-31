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

#: Multiplier taking a displacement grid to metres, which is what everything
#: downstream -- ``rms_min``, ``exclude_within``, the Green's functions -- assumes.
_LOS_UNITS = {"m": 1.0, "cm": 0.01, "mm": 0.001}


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


def _match_grid(da, ref, resampling):
    """Put ``da`` on ``ref``'s lattice, resampling only if it is not already."""
    from . import geo

    same = (
        da.dims == ref.dims
        and all(np.array_equal(da[d].values, ref[d].values) for d in ("y", "x"))
        and da.rio.crs == ref.rio.crs
    )
    if same:
        return da
    return da.rio.reproject_match(ref, resampling=geo.resampling_from_name(resampling))


def _finalise_look(ds):
    """Validate, renormalise and complete the look geometry.

    Reprojecting three components independently shortens the vector slightly, and
    ``los_up == cos(incidence)`` only holds for a unit vector -- so it is
    renormalised, and the incidence angle derived from it. A norm far from 1 to
    begin with means the grids are not a unit vector at all, which is raised on
    rather than quietly rescaled.
    """
    e, n, u = (np.asarray(ds[f"los_{c}"].values, float) for c in ("east", "north", "up"))
    norm = np.sqrt(e * e + n * n + u * u)
    finite = np.isfinite(norm) & (norm > 0)
    if not finite.any():
        raise ValueError("The look-vector grids have no pixel in common with data")

    median = float(np.median(norm[finite]))
    if not 0.9 <= median <= 1.1:
        raise ValueError(
            f"look_e/n/u are not a unit vector (median norm {median:.3g}); "
            "check the three grids are the components of the LOS direction."
        )

    mean_up = float(np.mean(u[finite]))
    if mean_up <= 0:
        raise ValueError(
            f"Mean look_u is {mean_up:+.3f}, but a target->sensor unit vector "
            "points up (cos of the incidence angle). Pass "
            "look_convention='sensor_to_target' if the grids point sensor->target."
        )

    scale = np.where(finite, norm, np.nan)
    for c, v in (("east", e), ("north", n), ("up", u)):
        ds[f"los_{c}"] = (("y", "x"), (v / scale).astype(np.float32))
    ds["incidence_angle"] = (
        ("y", "x"),
        np.degrees(np.arccos(np.clip(u / scale, -1.0, 1.0))).astype(np.float32),
    )
    ds["los"] = ds["los"].astype(np.float32)
    return ds


class LOSStack(RasterStackMixin):
    """LOS displacement (per pair) plus shared per-pixel look geometry."""

    STAGE = "los"
    # The per-pair displacement plus the ENU LOS **unit vectors** and the two
    # angles. Keeping the unit vector -- not just the displacement magnitude --
    # is what lets the scalar LOS be decomposed into ground components later
    # (e.g. inverting ascending + descending for vertical and east-west). The
    # sampled DEM ``height`` is available via ``fields=["height", ...]``.
    GRD_DEFAULT_FIELDS = ("los", "los_east", "los_north", "los_up",
                          "incidence_angle", "look_angle")

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
        geometry). ``sign=-1`` corrects data whose fringe sense is inverted
        relative to this package's ``ref * conj(sec)`` convention; the stored
        ``los`` is positive toward the sensor either way, and nothing downstream
        re-applies it (see :func:`nisar_tools.geometry.phase_to_los`).

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
    def from_grd(cls, los, look_e, look_n, look_u, units="m", sign=1,
                 look_convention="target_to_sensor", epsg=None,
                 resolution=None, resampling="bilinear", direction=None,
                 look_direction=None, wavelength=None):
        """Build a :class:`LOSStack` from GMT ``.grd`` grids of displacement and
        look vector -- the route in for LOS products this package did not make
        (GMTSAR, ISCE, ALOS, a colleague's ``.grd``).

        ``los`` is one displacement grid or a sequence of them (one per pair);
        ``look_e``/``look_n``/``look_u`` are the three components of the
        line-of-sight unit vector, shared by every pair. Look grids on a
        different lattice than ``los`` are resampled onto it.

        Three things have to be declared because no ``.grd`` records them, and
        each is silently wrong-looking rather than loud if mis-set:

        ``units``
            ``"m"`` (default), ``"cm"`` or ``"mm"`` -- a file called
            ``los_cm.grd`` needs ``units="cm"``. Everything downstream works in
            metres, so a factor of 100 here becomes a factor of 100 in the
            recovered slip.
        ``sign``
            ``+1`` if the grid is already **positive toward the sensor** (this
            package's convention, and the sense of a *decreasing* range);
            ``-1`` if it is positive away, i.e. positive range change, which is
            the other common convention. Applied on load, exactly as
            :func:`~nisar_tools.geometry.phase_to_los` applies it, so what is
            stored is canonical and nothing downstream re-applies it.
        ``look_convention``
            ``"target_to_sensor"`` (default, this package's) or
            ``"sensor_to_target"``, which is negated on load. Checked against
            the data: the vertical component of a target->sensor vector is
            ``cos(incidence)`` and must be **positive**, so a mis-declaration
            raises rather than inverting the geometry.

        The grid is reprojected to a metric CRS if it is geographic -- the
        quadtree in :meth:`~nisar_tools.slip.Observations.from_los` measures cell
        widths against ``width_min`` in metres, so a lon/lat lattice would put
        ``width_min=1000`` at a million columns. ``epsg`` picks the target
        (default: the UTM zone of the scene centre) and ``resolution`` its pixel
        size in metres (default: rasterio's estimate from the source grid).

        ``direction`` (``"ascending"``/``"descending"``) and ``look_direction``
        (``"left"``/``"right"``; ALOS is **right**-looking, NISAR is left) are
        carried as provenance and are what lets
        :func:`nisar_tools.slip.diagnostics.scene_report` check ``los_east``'s
        sign against the pass geometry -- the invariant that catches an inverted
        look vector. ``wavelength`` is provenance only; the grid is already
        displacement, so nothing rescales by it.
        """
        from . import geo

        try:
            scale = _LOS_UNITS[units]
        except KeyError:
            raise ValueError(
                f"units must be one of {sorted(_LOS_UNITS)}, not {units!r}"
            ) from None
        if sign not in (1, -1):
            raise ValueError(f"sign must be +1 or -1, not {sign!r}")
        if look_convention not in ("target_to_sensor", "sensor_to_target"):
            raise ValueError(
                "look_convention must be 'target_to_sensor' or 'sensor_to_target'"
            )

        paths = _as_granule_list(los)
        grids = [geo.read_grd(p) for p in paths]
        ref = grids[0]
        for i, g in enumerate(grids[1:], 1):
            grids[i] = _match_grid(g, ref, resampling)

        look = {
            f"los_{c}": _match_grid(geo.read_grd(p), ref, resampling)
            for c, p in (("east", look_e), ("north", look_n), ("up", look_u))
        }
        if look_convention == "sensor_to_target":
            look = {k: -v for k, v in look.items()}

        stacked = xr.concat(
            [g * (scale * sign) for g in grids], dim="pair", coords="minimal",
        )
        ds = xr.Dataset({"los": stacked, **look})

        if ds.rio.crs is None:
            raise ValueError(
                f"{paths[0]} carries no CRS and its coordinates are not lon/lat; "
                "pass grids with a CRS, or reproject them first."
            )
        if ds.rio.crs.is_geographic:
            if epsg is None:
                lon, lat = ds["x"].values.mean(), ds["y"].values.mean()
                epsg = geo.utm_epsg(lon, lat)
            ds = ds.rio.reproject(
                f"EPSG:{int(epsg)}", resolution=resolution,
                resampling=geo.resampling_from_name(resampling), nodata=np.nan,
            )
        epsg = int(ds.rio.crs.to_epsg())

        ds = _finalise_look(ds)
        ds.attrs.update(
            epsg=epsg,
            direction=direction,
            wavelength=None if wavelength is None else float(wavelength),
            frequency=None,
            # Applied above, so the stored field is positive toward the sensor
            # whatever was passed -- the same contract `from_unwrapped` has.
            sign=int(sign),
            look_direction=look_direction,
            granules=[str(p) for p in paths],
            pairs=None,
            source="grd",
            units=units,
        )
        return cls(ds)

    @classmethod
    def from_zarr(cls, path):
        return cls(open_stage(path))

    # -- persistence -------------------------------------------------------
    def persist(self, workspace, name=None, overwrite=False, **params):
        from . import geometry

        name = name or self.STAGE
        ds = self.ds.chunk(self.disk_chunks("pair"))
        full = {
            "stage": name,
            # `.get`, not `self.epsg`: a stack resampled into a LocalFrame has no
            # EPSG code at all, and the property raises. The frame goes in beside
            # it, and only when present, so a UTM stack keeps the hash it had.
            "epsg": self.ds.attrs.get("epsg"),
            **({"frame": self.ds.attrs["frame"]}
               if self.ds.attrs.get("frame") is not None else {}),
            **({"resampled": self.ds.attrs["resampled"]}
               if self.ds.attrs.get("resampled") is not None else {}),
            "wavelength": self.ds.attrs.get("wavelength"),
            "frequency": self.ds.attrs.get("frequency"),
            "sign": self.ds.attrs.get("sign"),
            # `sign` alone cannot invalidate a store when the *formula* changes
            # underneath it -- its value stayed 1 across the 2026-07-28 sign fix
            # while every pixel flipped. Pinning the convention itself makes a
            # stale `los_*.zarr` raise instead of silently reloading mirrored
            # displacement.
            "phase_convention": f"ref*conj(sec),d={geometry.PHASE_RANGE_SIGN:+d}(lambda/4pi)phi",
            "pairs": self.ds.attrs.get("pairs"),
            **params,
        }
        reopened = workspace.store(name, ds, full, overwrite=overwrite)
        return LOSStack(reopened)

    # -- export -----------------------------------------------------------
    def _grd_specs(self):
        """The per-pair ``los`` displacement (written ``los_pair{i}.grd``) and
        the shared 2-D look geometry: the ENU LOS unit vector, the two angles,
        and the sampled DEM ``height`` (one file each)."""
        specs = [("los", self.ds["los"], True)]
        for v in _GEOM_2D:
            if v in self.ds.data_vars:
                specs.append((v, self.ds[v], False))
        return specs

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
