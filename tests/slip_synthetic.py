"""Synthetic ``LOSStack`` builders shared by the slip-inversion tests.

Lives beside the tests as an importable module, the same arrangement
``legacy_reference`` uses: pytest puts this directory on ``sys.path``, so both
``test_slip_sampling`` and ``test_slip_inversion`` can build the same scenes
without duplicating the plumbing or growing the package's own conftest.

Two fields are offered. :func:`analytic_los_stack` is an arctangent ramp across
the fault -- cheap, so it can cover a whole raster, and adequate for testing the
sampler. :func:`forward_los_stack` is the real elastic forward model of a given
slip distribution, which is what a recovery test needs; it is evaluated only at
the points asked for, because a full raster would be millions of dislocation
evaluations.
"""

import numpy as np
import rioxarray  # noqa: F401  (registers .rio, needed for write_crs)
import xarray as xr
from pyproj import Transformer

from nisar_tools.los import LOSStack

# Two look geometries roughly like NISAR ascending and descending passes at
# mid-swath: right-looking, so the LOS unit vector's east component reverses
# between them. That reversal is what makes strike-slip and dip-slip separable.
GEOMETRIES = {
    "asc": {"incidence": 35.0, "heading": -10.0},
    "desc": {"incidence": 39.0, "heading": 190.0},
}


def look_vector(incidence_deg, heading_deg):
    """Target->sensor ENU unit vector for a right-looking pass.

    Matches :mod:`nisar_tools.geometry`'s convention: ``los_up`` is positive and
    equals ``cos(incidence)``.
    """
    inc = np.deg2rad(incidence_deg)
    # Look direction is 90 degrees clockwise of the heading for a right-looking
    # radar; the horizontal part of the target->sensor vector points back at it.
    az = np.deg2rad(heading_deg + 90.0)
    return (-np.sin(inc) * np.sin(az), -np.sin(inc) * np.cos(az), np.cos(inc))


def _grid(frame, epsg, half_x, half_y, spacing):
    """A north-up raster in ``epsg`` covering a box of the local frame."""
    t = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    lon, lat = frame.to_lonlat(np.array([-half_x, half_x]), np.array([-half_y, half_y]))
    ux, uy = t.transform(lon, lat)
    x = np.arange(min(ux), max(ux), spacing)
    y = np.arange(max(uy), min(uy), -spacing)
    return x, y


def _stack(los, x, y, look, epsg, name, sign=1, incidence=35.0):
    e, n, u = look
    shape = los.shape
    ds = xr.Dataset(
        {
            "los": (("pair", "y", "x"), (sign * los)[None].astype(np.float32)),
            "los_east": (("y", "x"), np.full(shape, e, np.float32)),
            "los_north": (("y", "x"), np.full(shape, n, np.float32)),
            "los_up": (("y", "x"), np.full(shape, u, np.float32)),
            "incidence_angle": (("y", "x"), np.full(shape, incidence, np.float32)),
        },
        coords={"pair": [0], "y": y, "x": x},
    ).rio.write_crs(f"EPSG:{epsg}")
    ds.attrs.update(epsg=int(epsg), sign=int(sign), direction=name,
                    wavelength=0.242, frequency="A")
    return LOSStack(ds)


def analytic_los_stack(trace, frame, geometry="asc", epsg=32619, spacing=800.0,
                       half_x=140e3, half_y=90e3, amplitude=0.2, width=10e3,
                       noise=0.0, nan_rows=0, sign=1, seed=0):
    """An arctangent step across the fault -- the classic screw-dislocation shape.

    Not a physical model of any particular slip distribution, but it has the
    right character for exercising the sampler: a sharp gradient at the fault
    that a quadtree should resolve finely, and flat far field it should leave
    coarse.
    """
    x, y = _grid(frame, epsg, half_x, half_y, spacing)
    xx, yy = np.meshgrid(x, y)
    fx, fy = frame.from_epsg(xx.ravel(), yy.ravel(), epsg)

    dist = trace.distance(fx, fy, frame).reshape(xx.shape)
    side = trace.side(fx, fy, frame).reshape(xx.shape).astype(float)
    los = side * (2.0 / np.pi) * np.arctan(dist / width) * amplitude
    if noise:
        los = los + np.random.default_rng(seed).normal(0.0, noise, los.shape)
    if nan_rows:
        los[:nan_rows, :] = np.nan

    return _stack(los, x, y, look_vector(**_geom(geometry)), epsg, geometry, sign=sign,
                  incidence=_geom(geometry)["incidence_deg"])


def forward_los_stack(mesh, slip, trace, frame, geometry="asc", epsg=32619,
                      spacing=2000.0, half_x=140e3, half_y=90e3,
                      noise=0.0, nan_rows=0, seed=0):
    """The elastic response of ``slip``, sampled onto a raster.

    Deliberately coarse by default: every pixel costs a full dislocation
    evaluation against every element, so a 500 m grid over this fault would be
    tens of millions of them. A recovery test wants a coarse *raster* and then a
    quadtree of it, not a fine one.
    """
    from nisar_tools.slip.greens import HalfSpaceTDE

    x, y = _grid(frame, epsg, half_x, half_y, spacing)
    xx, yy = np.meshgrid(x, y)
    fx, fy = frame.from_epsg(xx.ravel(), yy.ravel(), epsg)

    geom = _geom(geometry)
    e, n, u = look_vector(**geom)
    ones = np.ones(fx.size)
    los = HalfSpaceTDE(0.25).forward(
        mesh, slip, fx, fy, look=(e * ones, n * ones, u * ones)
    ).reshape(xx.shape)

    if noise:
        los = los + np.random.default_rng(seed).normal(0.0, noise, los.shape)
    if nan_rows:
        los[:nan_rows, :] = np.nan
    return _stack(los, x, y, (e, n, u), epsg, geometry,
                  incidence=geom["incidence_deg"])


def tapered_slip(mesh, peak=-2.0, along_centre=0.55, along_width=40e3,
                 locking_depth=12e3):
    """A single right-lateral patch: Gaussian along strike, tapering with depth.

    Negative strike-slip is right-lateral (see :mod:`nisar_tools.slip`), which is
    the sense of the San Sebastian and Sagaing systems.
    """
    s, z = mesh.element_params[:, 0], mesh.element_params[:, 1]
    length = s.max() - s.min()
    profile = np.exp(-((s - (s.min() + along_centre * length)) / along_width) ** 2)
    taper = np.clip(1.0 + z / locking_depth, 0.0, 1.0)
    slip = np.zeros(2 * mesh.n_elements)
    slip[: mesh.n_elements] = peak * profile * taper
    return slip


def _geom(name):
    if name not in GEOMETRIES:
        raise ValueError(f"Unknown geometry {name!r}; expected one of {list(GEOMETRIES)}")
    g = GEOMETRIES[name]
    return {"incidence_deg": g["incidence"], "heading_deg": g["heading"]}
