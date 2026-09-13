"""A synthetic NISAR-like acquisition for the RSLC -> GSLC tests.

A circular orbit in a fixed Earth-centred frame (the Earth does not rotate
under it, which changes nothing the geometry code cares about), sampled every
10 s like a NISAR RSLC's embedded orbit, plus a left-looking radar grid whose
swath sits near 10 deg N. Position and velocity are analytic, so interpolators
can be checked against the truth rather than against themselves.
"""

import numpy as np

from nisar_tools import radar

EPOCH = np.datetime64("2026-06-22T00:00:00", "ns")
GM = 3.986004418e14                          # m^3 / s^2
ORBIT_RADIUS = radar.WGS84_A + 747_000.0     # m
OMEGA = np.sqrt(GM / ORBIT_RADIUS ** 3)      # rad / s
T0 = 82_400.0                                # s since EPOCH, first state vector
INCLINATION = np.deg2rad(98.4)
NODE_LON = np.deg2rad(-75.0)
PHASE0 = np.deg2rad(8.0)                     # argument of latitude at T0


def _basis():
    e1 = np.array([np.cos(NODE_LON), np.sin(NODE_LON), 0.0])
    e2 = (np.cos(INCLINATION) * np.array([-np.sin(NODE_LON), np.cos(NODE_LON), 0.0])
          + np.sin(INCLINATION) * np.array([0.0, 0.0, 1.0]))
    return e1, e2


def truth(t):
    """Analytic ``(position, velocity)`` at times ``t`` (s since EPOCH)."""
    t = np.atleast_1d(np.asarray(t, float))
    u = PHASE0 + OMEGA * (t - T0)
    e1, e2 = _basis()
    pos = ORBIT_RADIUS * (np.cos(u)[:, None] * e1 + np.sin(u)[:, None] * e2)
    vel = ORBIT_RADIUS * OMEGA * (-np.sin(u)[:, None] * e1 + np.cos(u)[:, None] * e2)
    return pos, vel


def orbit(n=22, spacing=10.0):
    t = T0 + spacing * np.arange(n)
    pos, vel = truth(t)
    return radar.Orbit(t, pos, vel, EPOCH)


def radar_grid(length=400, width=300, *, prf=1520.0, starting_range=880_000.0,
               spacing=3.75, wavelength=0.242, sensing_start=None, look_side="left"):
    return radar.RadarGrid(
        sensing_start=T0 + 100.0 if sensing_start is None else sensing_start,
        wavelength=wavelength, prf=prf, starting_range=starting_range,
        range_pixel_spacing=spacing, look_side=look_side, length=length,
        width=width, ref_epoch=EPOCH)


def constant_doppler(grid, hz):
    """A native Doppler centroid LUT that is ``hz`` everywhere on (and around) ``grid``."""
    t = np.linspace(grid.sensing_start - 5.0, grid.sensing_stop + 5.0, 5)
    r = np.linspace(grid.starting_range - 500.0, grid.end_range + 500.0, 4)
    return radar.LUT2d.from_coords(r, t, np.full((t.size, r.size), float(hz)))
