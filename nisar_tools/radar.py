"""Radar-geometry primitives, ported from ISCE3 v0.25.16 and vectorized.

Everything :mod:`nisar_tools.rslc` needs to turn range-Doppler coordinates into
map coordinates and back: the WGS84 ellipsoid, orbit interpolation, ``geo2rdr``
/ ``rdr2geo``, ISCE3's 2-D look-up tables and its "biquintic" spline. The NISAR
GSLC products ASF distributes are made by ISCE3 0.25.16 (it is recorded in every
product's ``processingInformation/algorithms/softwareVersion``), so each routine
here is a line-for-line port of the one named in its docstring, operating on
arrays instead of one pixel at a time. The C++ is the reference; where the port
could have been "cleaner" but would then have disagreed with it, it follows the
C++.

Verified against ISCE3's own numbers, not against itself: the reference GSLC's
``metadata/radarGrid`` cube stores float64 slant range and zero-Doppler time per
``(x, y, height)`` node, computed by ISCE3 from the product's embedded orbit
(see ``tests/test_radar.py`` and the gated real-data test).

Conventions carried over from ISCE3, all of which matter:

- Angles are **radians**, heights metres above the WGS84 ellipsoid, times
  seconds since the orbit's reference epoch.
- An :class:`Orbit`'s state vectors must be uniformly spaced; interpolation
  outside their span returns NaN (``OrbitInterpBorderMode::FillNaN``).
- ``geo2rdr`` returns the ``(t, r)`` of the iteration whose Newton step fell
  below ``threshold`` -- the step is *not* applied afterwards -- so a returned
  pair is always self-consistent.
"""

from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

SPEED_OF_LIGHT = 299_792_458.0

#: WGS84 semi-major axis and first eccentricity squared, exactly as ISCE3's
#: ``Constants.h`` has them (e2 derived from 1/f = 298.257223563).
WGS84_A = 6378137.0
WGS84_E2 = 0.006694379990141317


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------
def parse_epoch(units):
    """``"seconds since 2026-06-22T00:00:00"`` -> ``numpy.datetime64[ns]``."""
    if isinstance(units, bytes):
        units = units.decode()
    text = str(units).strip()
    prefix = "seconds since"
    if not text.startswith(prefix):
        raise ValueError(f"not a 'seconds since <epoch>' unit: {units!r}")
    return np.datetime64(text[len(prefix):].strip().replace(" ", "T"), "ns")


def epoch_units(epoch):
    """The inverse of :func:`parse_epoch`, at whole-second precision.

    NISAR product specs require an integer-second epoch; ISCE3 writes it without
    a fractional part, which is what this reproduces.
    """
    epoch = np.datetime64(epoch, "s")
    return f"seconds since {epoch.astype(str)}"


def seconds_since(epoch, when):
    """Seconds from ``epoch`` to ``when`` (both datetime64-compatible)."""
    delta = np.datetime64(when, "ns") - np.datetime64(epoch, "ns")
    return delta.astype("timedelta64[ns]").astype(np.int64) / 1e9


# ---------------------------------------------------------------------------
# Ellipsoid (isce3::core::Ellipsoid)
# ---------------------------------------------------------------------------
def lonlat_to_xyz(lon, lat, h):
    """Geodetic (rad, rad, m) -> ECEF metres. Returns an ``(..., 3)`` array."""
    lon, lat, h = np.broadcast_arrays(np.asarray(lon, float),
                                      np.asarray(lat, float),
                                      np.asarray(h, float))
    re = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
    out = np.empty(lon.shape + (3,))
    out[..., 0] = (re + h) * np.cos(lat) * np.cos(lon)
    out[..., 1] = (re + h) * np.cos(lat) * np.sin(lon)
    out[..., 2] = (re * (1.0 - WGS84_E2) + h) * np.sin(lat)
    return out


def xyz_to_lonlat(xyz):
    """ECEF metres -> geodetic ``(lon, lat, h)`` in (rad, rad, m).

    Vermeille (2002), in ISCE3's exact operation order.
    """
    xyz = np.asarray(xyz, float)
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    e4 = WGS84_E2 * WGS84_E2
    a2 = WGS84_A * WGS84_A
    p = (x ** 2 + y ** 2) / a2
    q = ((1.0 - WGS84_E2) * z ** 2) / a2
    r = (p + q - e4) / 6.0
    s = (e4 * p * q) / (4.0 * r ** 3)
    t = (1.0 + s + np.sqrt(s * (2.0 + s))) ** (1.0 / 3.0)
    u = r * (1.0 + t + 1.0 / t)
    rv = np.sqrt(u ** 2 + e4 * q)
    w = (WGS84_E2 * (u + rv - q)) / (2.0 * rv)
    k = np.sqrt(u + rv + w ** 2) - w
    d = (k * np.sqrt(x ** 2 + y ** 2)) / (k + WGS84_E2)
    lat = np.arctan2(z, d)
    lon = np.arctan2(y, x)
    h = ((k + WGS84_E2 - 1.0) * np.sqrt(d ** 2 + z ** 2)) / k
    return np.stack([lon, lat, h], axis=-1)


def xyz_to_enu_matrix(lat, lon):
    """Rotation ECEF -> local ENU at geodetic ``(lat, lon)``; ``(..., 3, 3)``.

    ``Mat3::xyzToEnu``: rows are the east, north and up unit vectors.
    """
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    sl, cl = np.sin(lat), np.cos(lat)
    so, co = np.sin(lon), np.cos(lon)
    zero = np.zeros_like(lat + lon)
    return np.stack([
        np.stack([-so + zero, co + zero, zero], axis=-1),
        np.stack([-sl * co, -sl * so, cl + zero], axis=-1),
        np.stack([cl * co, cl * so, sl + zero], axis=-1),
    ], axis=-2)


def _dot(a, b):
    return np.einsum("...i,...i->...", a, b)


def _norm(a):
    return np.sqrt(_dot(a, a))


def _normalized(a):
    return a / _norm(a)[..., None]


# ---------------------------------------------------------------------------
# Orbit (isce3::core::Orbit, detail/InterpolateOrbit.icc)
# ---------------------------------------------------------------------------
class Orbit:
    """Uniformly sampled ECEF state vectors with ISCE3's interpolators.

    ``time`` is seconds since ``reference_epoch``. ISCE3 stores it as a
    ``Linspace`` built from the first sample and the first spacing, and every
    lookup uses *that* lattice, not the stored array -- so does this class.
    """

    def __init__(self, time, position, velocity, reference_epoch,
                 interp_method="Hermite", orbit_type="Undefined"):
        time = np.asarray(time, float)
        position = np.asarray(position, float)
        velocity = np.asarray(velocity, float)
        if time.ndim != 1 or time.size < 2:
            raise ValueError("an orbit needs at least two state vectors")
        if position.shape != (time.size, 3) or velocity.shape != (time.size, 3):
            raise ValueError("position/velocity must be (n, 3) matching time")
        spacing = time[1] - time[0]
        # getOrbitTime: DateTime::isClose, i.e. within a nanosecond or so.
        if not np.allclose(np.diff(time), spacing, rtol=0.0, atol=1e-6):
            raise ValueError("orbit state vectors are not uniformly sampled")
        if interp_method not in ("Hermite", "Legendre"):
            raise ValueError(f"unknown orbit interpolation {interp_method!r}")
        self.first = float(time[0])
        self.spacing = float(spacing)
        self.size = int(time.size)
        self.position = position
        self.velocity = velocity
        self.reference_epoch = np.datetime64(reference_epoch, "ns")
        self.interp_method = interp_method
        self.orbit_type = orbit_type

    # -- lattice ------------------------------------------------------------
    @property
    def time(self):
        return self.first + self.spacing * np.arange(self.size)

    @property
    def start_time(self):
        return self.first

    @property
    def end_time(self):
        return self.first + (self.size - 1) * self.spacing

    @property
    def mid_time(self):
        return 0.5 * (self.start_time + self.end_time)

    def _search(self, t):
        """``Linspace::search``: index of the first sample strictly after ``t``."""
        idx = np.where(np.isfinite(t), (t - self.first) / self.spacing + 1, 0.0)
        idx = np.trunc(idx).astype(np.int64)
        idx = np.where(t < self.first, 0, idx)
        return np.where(t > self.end_time, self.size, idx)

    # -- interpolation ------------------------------------------------------
    def interpolate(self, t):
        """Position and velocity at ``t`` (any shape) -> two ``(..., 3)`` arrays.

        Outside the orbit's span both come back NaN, as ISCE3's ``FillNaN``
        border mode does.
        """
        t = np.asarray(t, float)
        shape = t.shape
        tf = t.ravel()
        if self.interp_method == "Hermite":
            pos, vel = self._hermite(tf)
        else:
            pos, vel = self._legendre(tf)
        if not (tf.size == 0 or (tf.min() >= self.start_time and tf.max() <= self.end_time)):
            outside = ~((tf >= self.start_time) & (tf <= self.end_time))
            pos[outside] = np.nan
            vel[outside] = np.nan
        return pos.reshape(shape + (3,)), vel.reshape(shape + (3,))

    def _hermite(self, t):
        """``interpolateOrbitHermite`` on the uniform lattice, as polynomials.

        ISCE3 builds the 4-point Lagrange basis ``h_i`` and its derivative from
        nested products of ``(t - T_j) / (T_i - T_j)``. On a uniform lattice
        every weight multiplying a state vector is a fixed polynomial of degree
        <= 7 in the window-centred time ``v``, so over one window the position
        and velocity are themselves degree-7 polynomials with vector
        coefficients (:meth:`_hermite_table`). Evaluating those takes one
        Vandermonde matrix and two small products instead of ~100 array
        passes; it agrees with the product form to ~1e-8 m.
        """
        if self.size < 4:
            raise ValueError("Hermite interpolation needs >= 4 state vectors")
        n = t.size
        pos = np.empty((n, 3))
        vel = np.empty((n, 3))
        if n == 0:
            return pos, vel
        table_pos, table_vel = self._hermite_table()
        lo, hi = np.min(t), np.max(t)
        if np.isfinite(lo) and np.isfinite(hi) and \
                self._hermite_window(lo) == self._hermite_window(hi):
            # The common case -- one block of lines spans ~1 s of a 10 s lattice.
            w = self._hermite_window(lo)
            V = np.vander((t - self._hermite_centre(w)) / self.spacing, 8, increasing=True)
            np.matmul(V, table_pos[w], out=pos)
            np.matmul(V, table_vel[w], out=vel)
            return pos, vel
        w = self._hermite_window(t)
        V = np.vander((t - self._hermite_centre(w)) / self.spacing, 8, increasing=True)
        for wi in np.flatnonzero(np.bincount(w, minlength=self.size - 3)):
            sel = w == wi
            Vs = V[sel]
            pos[sel] = Vs @ table_pos[wi]
            vel[sel] = Vs @ table_vel[wi]
        return pos, vel

    def _hermite_window(self, t):
        """First state vector of the 4-point window for ``t`` (scalar or array).

        ``clip(search(t) - 2, 0, size - 4)`` spelt with ``fmax``/``fmin`` so a
        NaN lands in window 0 instead of becoming an undefined integer.
        """
        w = np.fmin(np.fmax(np.trunc((t - self.first) / self.spacing + 1.0) - 2.0, 0.0),
                    self.size - 4.0)
        return w.astype(np.int64) if isinstance(w, np.ndarray) else int(w)

    def _hermite_centre(self, window):
        return self.first + self.spacing * (window + 1.5)

    def _hermite_table(self):
        """Per-window ``(windows, 8, 3)`` power-basis coefficients, built once."""
        table = self.__dict__.get("_hermite_tables")
        if table is None:
            C = _hermite_coefficients(self.spacing)
            nw = self.size - 3
            tp = np.empty((nw, 8, 3))
            tv = np.empty((nw, 8, 3))
            for w in range(nw):
                P4 = self.position[w:w + 4]
                V4 = self.velocity[w:w + 4]
                tp[w] = C[:, 0:4] @ P4 + C[:, 4:8] @ V4
                tv[w] = C[:, 8:12] @ P4 + C[:, 12:16] @ V4
            table = self._hermite_tables = (tp, tv)
        return table

    def _legendre(self, t):
        if self.size < 9:
            raise ValueError("Legendre interpolation needs >= 9 state vectors")
        idx = np.clip(self._search(t) - 5, 0, self.size - 9)
        t0 = self.first + self.spacing * idx
        t8 = self.first + self.spacing * (idx + 8)
        trel = 8.0 * (t - t0) / (t8 - t0)
        teller = np.ones_like(t)
        for i in range(9):
            teller = teller * (trel - i)
        noemer = np.array([40320.0, -5040.0, 1440.0, -720.0, 576.0,
                           -720.0, 1440.0, -5040.0, 40320.0])
        pos = np.zeros((t.size, 3))
        vel = np.zeros((t.size, 3))
        exact = teller == 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            for i in range(9):
                coeff = (teller / noemer[i]) / (trel - i)
                pos += coeff[:, None] * self.position[idx + i]
                vel += coeff[:, None] * self.velocity[idx + i]
        if exact.any():
            j = idx[exact] + trel[exact].astype(np.int64)
            pos[exact] = self.position[j]
            vel[exact] = self.velocity[j]
        return pos, vel

    # -- construction / serialization --------------------------------------
    @classmethod
    def from_h5(cls, group):
        """Read a NISAR ``metadata/orbit`` group."""
        time_ds = group["time"]
        epoch = parse_epoch(time_ds.attrs["units"])
        method = _text(group["interpMethod"][()]) if "interpMethod" in group else "Hermite"
        kind = _text(group["orbitType"][()]) if "orbitType" in group else "Undefined"
        return cls(time_ds[()], group["position"][()], group["velocity"][()],
                   epoch, interp_method=method, orbit_type=kind)

    @classmethod
    def from_xml(cls, path, reference_epoch=None):
        """Read a NISAR orbit-ephemeris XML (FOE/NOE/MOE/POE).

        Port of ``nisar.products.readers.orbit.load_orbit_from_xml``; the
        reference epoch defaults to the first state vector's time.
        """
        import xml.etree.ElementTree as ET

        root = ET.parse(str(path)).getroot()
        svl = root.find("orbitStateVectorList")
        if svl is None:
            raise IOError(f"{path}: no orbitStateVectorList")
        times, pos, vel = [], [], []
        for node in svl.findall("orbitStateVector"):
            times.append(np.datetime64(node.find("utc").text.strip(), "ns"))
            pos.append([float(node.find(n).text) for n in "xyz"])
            vel.append([float(node.find(n).text) for n in ("vx", "vy", "vz")])
        count = int(svl.attrib["count"])
        if len(times) != count:
            raise IOError(f"{path}: expected {count} state vectors, got {len(times)}")
        kind = root.find("productInformation/productType")
        if kind is None:
            kind = root.find("productInformation/fileType")
        if reference_epoch is None:
            reference_epoch = times[0]
        t = np.array([seconds_since(reference_epoch, ti) for ti in times])
        return cls(t, pos, vel, reference_epoch,
                   orbit_type=kind.text.strip() if kind is not None else "Undefined")

    def crop(self, start, end, npad=7):
        """``Orbit::crop``: state vectors spanning ``[start, end]`` plus ``npad``.

        ``start``/``end`` are seconds since this orbit's reference epoch.
        """
        for bound in (start, end):
            if not self.start_time <= bound <= self.end_time:
                raise ValueError(f"time {bound} outside orbit "
                                 f"[{self.start_time}, {self.end_time}]")
        i0 = max(int(self._search(np.array([start]))[0]) - 1 - npad, 0)
        i1 = min(int(self._search(np.array([end]))[0]) + 1 + npad, self.size)
        return Orbit(self.time[i0:i1], self.position[i0:i1],
                     self.velocity[i0:i1], self.reference_epoch,
                     self.interp_method, self.orbit_type)

    def with_reference_epoch(self, epoch):
        """The same state vectors re-expressed against another epoch."""
        epoch = np.datetime64(epoch, "ns")
        shift = seconds_since(epoch, self.reference_epoch)
        return Orbit(self.time + shift, self.position, self.velocity, epoch,
                     self.interp_method, self.orbit_type)


@lru_cache(maxsize=8)
def _hermite_coefficients(spacing):
    """``(8, 16)`` power-basis coefficients of the Hermite weights.

    The variable is ``v = (t - T_c) / spacing`` with ``T_c`` the centre of the
    4-point window, so the nodes sit at -1.5, -0.5, 0.5, 1.5 and ``|v|`` stays
    small enough for the degree-7 powers to be well conditioned. Columns 0-3
    weight the positions and 4-7 the velocities for the interpolated position;
    8-11 and 12-15 likewise for the velocity. Built from ISCE3's definitions:
    ``h_i`` the cubic Lagrange basis, ``f0_i = 1 - 2 s_i (v - v_i)``,
    ``hdot_i = h_i' / spacing``, ``g1_i = h_i + 2 hdot_i (t - T_i)``,
    ``g0_i = 2 (f0_i hdot_i - s_i h_i / spacing)``, with
    ``s_i = sum_{j != i} 1 / (v_i - v_j)``.
    """
    from numpy.polynomial import polynomial as poly

    nodes = [-1.5, -0.5, 0.5, 1.5]
    out = np.zeros((8, 16))
    for i in range(4):
        h = np.array([1.0])
        for j in range(4):
            if j != i:
                h = poly.polymul(h, np.array([-nodes[j], 1.0]) / (nodes[i] - nodes[j]))
        dh = poly.polyder(h)                                  # d h_i / du
        s = sum(1.0 / (nodes[i] - nodes[j]) for j in range(4) if j != i)
        u_minus_i = np.array([-nodes[i], 1.0])
        f0 = poly.polysub(np.array([1.0]), 2.0 * s * u_minus_i)
        d = spacing * u_minus_i                               # t - T_i
        hdot = dh / spacing
        g1 = poly.polyadd(h, 2.0 * poly.polymul(hdot, d))
        g0 = 2.0 * poly.polysub(poly.polymul(f0, hdot), (s / spacing) * h)
        hh = poly.polymul(h, h)
        for col, c in ((i, poly.polymul(hh, f0)), (4 + i, poly.polymul(hh, d)),
                       (8 + i, poly.polymul(h, g0)), (12 + i, poly.polymul(h, g1))):
            out[:len(c), col] = c
    return out


def crop_external_orbit(external, internal, npad=7):
    """``isce3.core.crop_external_orbit``: crop ``external`` to ``internal``'s span.

    Both orbits must share a reference epoch (ISCE3 re-references the external
    file to the radar grid's epoch before calling this).
    """
    start = max(internal.start_time, external.start_time)
    end = min(internal.end_time, external.end_time)
    return external.crop(start, end, npad=npad)


def _text(value):
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.ndarray) and value.shape == ():
        return _text(value[()])
    return str(value)


# ---------------------------------------------------------------------------
# 2-D interpolators (isce3::core::{Bilinear,NearestNeighbor,Spline2d}Interpolator)
# ---------------------------------------------------------------------------
def bilinear_interp(z, x, y):
    """``BilinearInterpolator``: ``x`` is the column, ``y`` the row coordinate.

    ISCE3 takes floor and ceil of each and special-cases integral coordinates
    (where floor == ceil) rather than using the usual ``x - x1`` weight form,
    which would divide by zero there.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    x1 = np.floor(x).astype(np.int64)
    x2 = np.ceil(x).astype(np.int64)
    y1 = np.floor(y).astype(np.int64)
    y2 = np.ceil(y).astype(np.int64)
    q11, q12 = z[y1, x1], z[y2, x1]
    q21, q22 = z[y1, x2], z[y2, x2]
    wx1, wx2 = x2 - x, x - x1
    wy1, wy2 = y2 - y, y - y1
    # Off the lattice lines x2 - x1 == y2 - y1 == 1, and dividing by 1 is exact.
    out = (wx1 * wy1) * q11 + (wx2 * wy1) * q21 + (wx1 * wy2) * q12 + (wx2 * wy2) * q22
    on_line = (x1 == x2) | (y1 == y2)
    if on_line.any():
        k = np.flatnonzero(on_line.ravel())
        out = np.array(out, copy=True).reshape(-1) if np.ndim(out) else np.array([out])
        dx = (x2 - x1).ravel()[k]
        dy = (y2 - y1).ravel()[k]
        a11, a12, a21 = (np.ravel(q)[k] for q in (q11, q12, q21))
        with np.errstate(divide="ignore", invalid="ignore"):
            only_x = (wx1.ravel()[k] / dx) * a11 + (wx2.ravel()[k] / dx) * a21
            only_y = (wy1.ravel()[k] / dy) * a11 + (wy2.ravel()[k] / dy) * a12
        out[k] = np.where((dx == 0) & (dy == 0), a11, np.where(dy == 0, only_x, only_y))
        out = out.reshape(np.shape(x1))
    return np.asarray(out).astype(np.result_type(z.dtype, np.float32), copy=False)


def nearest_interp(z, x, y):
    """``NearestNeighborInterpolator``: ``std::round`` (half away from zero)."""
    row = _round_half_away(np.asarray(y, float))
    col = _round_half_away(np.asarray(x, float))
    return z[row, col]


def _round_half_away(v):
    return (np.sign(v) * np.floor(np.abs(v) + 0.5)).astype(np.int64)


def _spline1d_weights(dtype, n):
    """The data-independent ``Q`` and ``p`` of ``_initSpline``, in ``dtype``."""
    one, half, two = dtype.type(1.0), dtype.type(0.5), dtype.type(2.0)
    Q = [dtype.type(0.0)]
    p = [None]
    for i in range(1, n - 1):
        pi = one / (half * Q[i - 1] + two)
        p.append(pi)
        Q.append(dtype.type(-0.5) * pi)
    return Q, p


def _spline1d(x, Y, dtype):
    """``_initSpline`` + ``_spline`` over the last axis of ``Y`` (length n).

    Natural cubic spline through ``n`` equally spaced samples, evaluated at the
    1-based coordinate ``x``, in ISCE3's operation order and float type.
    """
    n = Y.shape[-1]
    U = dtype.type
    Q, p = _spline1d_weights(dtype, n)
    R = np.zeros_like(Y)
    for i in range(1, n - 1):
        R[..., i] = (U(3.0) * (Y[..., i + 1] - U(2.0) * Y[..., i] + Y[..., i - 1])
                     - U(0.5) * R[..., i - 1]) * p[i]
    for i in range(n - 2, 0, -1):
        R[..., i] = Q[i] * R[..., i + 1] + R[..., i]

    x = np.asarray(x, float)
    j = np.clip(np.floor(x).astype(np.int64), 1, n - 1)
    take = lambda arr, jj: np.take_along_axis(arr, jj[..., None], axis=-1)[..., 0]
    xx = (x - j).astype(dtype)
    Yj, Yj1 = take(Y, j), take(Y, j - 1)
    Rj, Rj1 = take(R, j), take(R, j - 1)
    t0 = Yj - Yj1 - Rj1 / U(3.0) - Rj / U(6.0)
    t1 = xx * (Rj1 / U(2.0) + xx * ((Rj - Rj1) / U(6.0)))
    mid = Yj1 + xx * (t0 + t1)
    low = Y[..., 0] + (x - 1.0).astype(dtype) * (Y[..., 1] - Y[..., 0] - R[..., 1] / U(6.0))
    high = Y[..., n - 1] + (x - n).astype(dtype) * (
        Y[..., n - 1] - Y[..., n - 2] + R[..., n - 2] / U(6.0))
    return np.where(x < 1.0, low, np.where(x > n, high, mid))


def spline2d_interp(z, x, y, order=6):
    """``Spline2dInterpolator(order)`` -- what ISCE3 calls "biquintic".

    It is a natural *cubic* spline through ``order`` samples per axis (6 for
    "biquintic"), fitted along each row of the window and then down the column
    of row results. The window is not centred: for even orders it spans
    ``floor(v) - 1 ... floor(v) + 4``. Complex input is interpolated per part,
    which is exact because every coefficient is real.
    """
    z = np.asarray(z)
    if np.iscomplexobj(z):
        return (spline2d_interp(z.real, x, y, order)
                + 1j * spline2d_interp(z.imag, x, y, order)).astype(z.dtype)
    dtype = z.dtype if z.dtype in (np.float32, np.float64) else np.dtype(np.float64)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if order % 2 == 0 and x.size and x.shape == y.shape:
        lo = min(np.min(x), np.min(y))
        hi = max(np.max(x), np.max(y))
        if lo >= 0 and np.isfinite(hi) and hi < 2.0 ** 52:
            span = ((np.trunc(np.max(y)) - np.trunc(np.min(y)) + order)
                    * (np.trunc(np.max(x)) - np.trunc(np.min(x)) + 1))
            if span <= 16 * x.size + 4096:
                return _spline2d_even_table(z.astype(dtype, copy=False), x.ravel(),
                                            y.ravel(), order, dtype).reshape(x.shape)
    return _spline2d_pointwise(z, x, y, order, dtype)


def _spline2d_pointwise(z, x, y, order, dtype):
    """:func:`spline2d_interp`'s general path: a 6x6 window gathered per point."""
    ny, nx = z.shape
    if order % 2:
        i0 = np.trunc(y - 0.5).astype(np.int64)
        j0 = np.trunc(x - 0.5).astype(np.int64)
    else:
        i0 = np.trunc(y).astype(np.int64)
        j0 = np.trunc(x).astype(np.int64)
    i0 = i0 - order // 2 + 1
    j0 = j0 - order // 2 + 1
    k = np.arange(order)
    rows = np.clip(i0[..., None] + k, 0, ny - 2) + 1
    cols = np.clip(j0[..., None] + k, 0, nx - 2) + 1
    A = z.astype(dtype, copy=False)[rows[..., :, None], cols[..., None, :]]
    HC = _spline1d((x - j0)[..., None], A, dtype)
    return _spline1d(y - i0, HC, dtype)


def _spline_second_derivatives(Y, dtype):
    """``_initSpline``'s ``R`` over the last axis, in ISCE3's float order."""
    n = Y.shape[-1]
    U = dtype.type
    Q, p = _spline1d_weights(dtype, n)
    R = np.zeros_like(Y)
    for i in range(1, n - 1):
        R[..., i] = (U(3.0) * (Y[..., i + 1] - U(2.0) * Y[..., i] + Y[..., i - 1])
                     - U(0.5) * R[..., i - 1]) * p[i]
    for i in range(n - 2, 0, -1):
        R[..., i] = Q[i] * R[..., i + 1] + R[..., i]
    return R


def _spline2d_even_table(z, x, y, order, dtype):
    """:func:`spline2d_interp` for even orders and non-negative coordinates.

    There every row spline is evaluated on the same interval, ``j = order/2 -
    1``, so the terms of ``_spline`` that depend only on the data -- ``Y[j-1]``,
    ``t0`` and the two parenthesised factors of ``t1`` -- are the same for every
    point sharing a window. They are computed once per window over the patch
    the points touch; per point only the powers of the fractional coordinate
    remain. Every float operation is the general path's, so the results are
    bit-identical to it.
    """
    U = dtype.type
    n = order
    m = n // 2 - 1
    ny, nx = z.shape
    j0 = np.trunc(x).astype(np.int64) - m
    i0 = np.trunc(y).astype(np.int64) - m
    jlo, jhi = int(j0.min()), int(j0.max())
    ilo, ihi = int(i0.min()), int(i0.max())
    rows = np.clip(np.arange(ilo, ihi + n), 0, ny - 2) + 1
    cols = np.clip(np.arange(jlo, jhi + n), 0, nx - 2) + 1
    A = np.lib.stride_tricks.sliding_window_view(z[np.ix_(rows, cols)], n, axis=1)
    R = _spline_second_derivatives(A, dtype)
    table = np.stack([
        A[..., m - 1],
        A[..., m] - A[..., m - 1] - R[..., m - 1] / U(3.0) - R[..., m] / U(6.0),
        R[..., m - 1] / U(2.0),
        (R[..., m] - R[..., m - 1]) / U(6.0),
    ], axis=-1)                                           # (rows, windows, 4)
    k = np.arange(n)
    terms = table[(i0 - ilo)[:, None] + k, (j0 - jlo)[:, None]]    # (P, n, 4)
    xx = ((x - j0) - m).astype(dtype)[:, None]
    HC = terms[..., 0] + xx * (terms[..., 1] + xx * (terms[..., 2] + xx * terms[..., 3]))
    R = _spline_second_derivatives(HC, dtype)
    yy = ((y - i0) - m).astype(dtype)
    t0 = HC[:, m] - HC[:, m - 1] - R[:, m - 1] / U(3.0) - R[:, m] / U(6.0)
    t1 = yy * (R[:, m - 1] / U(2.0) + yy * ((R[:, m] - R[:, m - 1]) / U(6.0)))
    return HC[:, m - 1] + yy * (t0 + t1)


_INTERPOLATORS = {
    "bilinear": bilinear_interp,
    "nearest": nearest_interp,
    "biquintic": spline2d_interp,
}


# ---------------------------------------------------------------------------
# LUT2d (isce3::core::LUT2d)
# ---------------------------------------------------------------------------
class LUT2d:
    """A regularly gridded function of ``(y=azimuth time, x=slant range)``.

    With no data it evaluates to ``ref_value`` everywhere and contains every
    point -- ISCE3's default-constructed LUT, which is how "no correction" and
    "zero-Doppler grid" are spelled.
    """

    def __init__(self, x_start=0.0, y_start=0.0, dx=1.0, dy=1.0, data=None,
                 method="bilinear", bounds_error=True, ref_value=0.0):
        if method not in _INTERPOLATORS:
            raise ValueError(f"unknown LUT interpolation {method!r}")
        self.method = method
        self.bounds_error = bounds_error
        if data is None:
            self.have_data = False
            self.ref_value = ref_value
            self.data = None
            self.x_start = self.y_start = 0.0
            self.dx = self.dy = 1.0
            return
        data = np.asarray(data)
        if data.ndim != 2:
            raise ValueError("LUT2d data must be 2-D")
        self.have_data = True
        self.data = data
        self.ref_value = data[0, 0]
        self.x_start, self.y_start = float(x_start), float(y_start)
        self.dx, self.dy = float(dx), float(dy)

    @classmethod
    def from_coords(cls, xcoord, ycoord, data, method="bilinear",
                    bounds_error=True):
        """``LUT2d(xcoord, ycoord, data)``: spacing from the first difference."""
        xcoord = np.asarray(xcoord, float)
        ycoord = np.asarray(ycoord, float)
        data = np.asarray(data)
        if xcoord.size == 0 and ycoord.size == 0:
            return cls(method=method, bounds_error=bounds_error)
        if xcoord.size == 1 and ycoord.size == 1:
            return cls(method=method, bounds_error=bounds_error,
                       ref_value=data.ravel()[0])
        if data.shape != (ycoord.size, xcoord.size):
            raise ValueError("LUT2d data shape does not match its coordinates")
        return cls(xcoord[0], ycoord[0], xcoord[1] - xcoord[0],
                   ycoord[1] - ycoord[0], data, method, bounds_error)

    @property
    def length(self):
        return self.data.shape[0] if self.have_data else 0

    @property
    def width(self):
        return self.data.shape[1] if self.have_data else 0

    @property
    def x_end(self):
        return self.x_start + (self.width - 1) * self.dx

    @property
    def y_end(self):
        return self.y_start + (self.length - 1) * self.dy

    def contains(self, y, x):
        y = np.asarray(y, float)
        x = np.asarray(x, float)
        if not self.have_data:
            return np.ones(np.broadcast(y, x).shape, bool)
        return (x >= self.x_start) & (x <= self.x_end) \
            & (y >= self.y_start) & (y <= self.y_end)

    def eval(self, y, x):
        """Evaluate at azimuth ``y`` and range ``x``, clamping to the grid.

        ISCE3 logs (it does not throw) on an out-of-bounds lookup with
        ``bounds_error`` set, then clamps anyway; callers that must not
        extrapolate check :meth:`contains` first, as ``geocodeSlc`` does.
        """
        y = np.asarray(y, float)
        x = np.asarray(x, float)
        shape = np.broadcast(y, x).shape
        if not self.have_data:
            return np.full(shape, self.ref_value, dtype=np.result_type(self.ref_value, float))
        xi = np.clip((x - self.x_start) / self.dx, 0.0, self.width - 1.0)
        yi = np.clip((y - self.y_start) / self.dy, 0.0, self.length - 1.0)
        xi, yi = np.broadcast_arrays(xi, yi)
        return _INTERPOLATORS[self.method](self.data, xi, yi)


# ---------------------------------------------------------------------------
# Radar grid (isce3::product::RadarGridParameters)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RadarGrid:
    """A zero-Doppler range-Doppler raster's time/range lattice."""

    sensing_start: float
    wavelength: float
    prf: float
    starting_range: float
    range_pixel_spacing: float
    look_side: str          # "left" or "right"
    length: int
    width: int
    ref_epoch: np.datetime64

    @property
    def azimuth_time_interval(self):
        return 1.0 / self.prf

    @property
    def sensing_stop(self):
        return self.sensing_start + (self.length - 1.0) / self.prf

    @property
    def sensing_mid(self):
        return 0.5 * (self.sensing_start + self.sensing_stop)

    @property
    def end_range(self):
        return self.starting_range + (self.width - 1.0) * self.range_pixel_spacing

    @property
    def mid_range(self):
        return 0.5 * (self.starting_range + self.end_range)

    def sensing_time(self, line):
        return self.sensing_start + np.asarray(line, float) / self.prf

    def slant_range(self, sample):
        return self.starting_range + np.asarray(sample, float) * self.range_pixel_spacing

    def contains(self, t, r):
        """``RadarGridParameters::contains``: half a pixel beyond each edge."""
        t = np.asarray(t, float)
        r = np.asarray(r, float)
        half_t = self.azimuth_time_interval / 2
        half_r = self.range_pixel_spacing / 2
        return (t >= self.sensing_start - half_t) & (r >= self.starting_range - half_r) \
            & (t <= self.sensing_stop + half_t) & (r <= self.end_range + half_r)

    def resize_keep_start_stop(self, ysize, xsize):
        """``resizeKeepStartStop``: a coarser grid with the same corners."""
        if ysize <= 1 or xsize <= 1:
            raise ValueError("resized radar grid needs > 1 sample per axis")
        prf = self.prf
        if self.length > 1:
            prf = (ysize - 1.0) / (self.length - 1.0) * prf
        dr = (self.width - 1.0) / (xsize - 1.0) * self.range_pixel_spacing
        return replace(self, prf=prf, range_pixel_spacing=dr,
                       length=int(ysize), width=int(xsize))


# ---------------------------------------------------------------------------
# geo2rdr / rdr2geo (isce3::geometry::detail)
# ---------------------------------------------------------------------------
def _look_side_ok(rvec, vel, pos, look_side):
    """ISCE3's first-iteration check, ``(side == Right) xor (r x v . p > 0)``."""
    positive = _dot(np.cross(rvec, vel), pos) > 0.0
    right = str(look_side).lower() == "right"
    return ~(right ^ positive)


def _coarse_aztime(xyz, orbit, look_side, rmin=np.nan, rmax=np.nan):
    """``updateAztime``: best of 15 evenly spaced orbit times (vectorized)."""
    n = 15
    times = orbit.start_time + np.arange(n) * (orbit.end_time - orbit.start_time) / (n - 1)
    best_r = np.full(xyz.shape[0], 1e16)
    best_t = np.full(xyz.shape[0], -1000.0)
    ok = np.ones(xyz.shape[0], bool)
    for kk, tt in enumerate(times):
        pos, vel = orbit.interpolate(np.array(tt))
        rvec = xyz - pos
        if kk == 0:
            ok &= _look_side_ok(rvec, vel[None], pos[None], look_side)
        r = _norm(rvec)
        valid = np.ones_like(ok)
        if np.isfinite(rmin):
            valid &= r >= rmin
        if np.isfinite(rmax):
            valid &= r <= rmax
        better = valid & (r < best_r)
        best_r = np.where(better, r, best_r)
        best_t = np.where(better, tt, best_t)
    return np.where(best_t < 0.0, orbit.mid_time, best_t), ok


def geo2rdr(llh, orbit, doppler=None, wavelength=1.0, look_side="left",
            t0=None, threshold=1e-8, maxiter=25, delta_range=1e-8):
    """``detail::geo2rdr``: target ``(lon, lat, h)`` -> ``(t, r, converged)``.

    ``llh`` is an ``(n, 3)`` array in (rad, rad, m); ``doppler`` a
    :class:`LUT2d` of the grid's Doppler in Hz (``None`` = zero-Doppler, ISCE3's
    empty LUT). ``t0`` is the initial azimuth time (scalar or per point); points
    whose guess lies outside the orbit get ISCE3's coarse search instead.

    Each point stops iterating once its Newton step is below ``threshold`` and
    keeps that iteration's ``(t, r)``. Non-converged points (including every
    wrong-look-side and out-of-orbit point) come back with ``converged=False``;
    their ``(t, r)`` are meaningless.
    """
    llh = np.atleast_2d(np.asarray(llh, float))
    xyz = lonlat_to_xyz(llh[:, 0], llh[:, 1], llh[:, 2])
    return geo2rdr_xyz(xyz, orbit, doppler, wavelength, look_side, t0,
                       threshold, maxiter, delta_range)


def geo2rdr_xyz(xyz, orbit, doppler=None, wavelength=1.0, look_side="left",
                t0=None, threshold=1e-8, maxiter=25, delta_range=1e-8):
    """:func:`geo2rdr` for targets already in ECEF metres.

    Vectorised over points without changing any point's iterates: a point
    that converges (or is rejected) is frozen -- its time stops moving and its
    range is no longer recorded -- and the working set is compacted only once
    at least half of it is frozen, since NISAR pixels seeded at mid-swath all
    converge within an iteration or two of each other.
    """
    xyz = np.atleast_2d(np.asarray(xyz, float))
    npts = xyz.shape[0]
    if doppler is None:
        doppler = LUT2d()
    t = np.full(npts, orbit.mid_time) if t0 is None else \
        np.broadcast_to(np.asarray(t0, float), (npts,)).copy()
    alive = np.ones(npts, bool)
    in_orbit = (t >= orbit.start_time) & (t <= orbit.end_time)
    if not in_orbit.all():
        guess, ok = _coarse_aztime(xyz[~in_orbit], orbit, look_side)
        t[~in_orbit] = guess
        alive[np.flatnonzero(~in_orbit)[~ok]] = False
    # Every point starts from the same time: one orbit lookup serves them all.
    shared_seed = npts > 0 and bool(in_orbit.all()) and (t0 is None or np.ndim(t0) == 0)

    r = np.full(npts, np.nan)
    converged = np.zeros(npts, bool)
    # The working set: `index` into the full arrays (None = all of them), and
    # `live` within it (None = every point still iterating).
    index = None if alive.all() else np.flatnonzero(alive)
    xw = xyz if index is None else xyz[index]
    tw = t.copy() if index is None else t[index]
    rw = np.full(tw.size, np.nan)
    cw = np.zeros(tw.size, bool)
    dtw = None
    live = None
    for it in range(maxiter):
        if tw.size == 0:
            break
        if dtw is not None:
            tw -= dtw
        pos, vel = orbit.interpolate(tw[:1] if it == 0 and shared_seed else tw)
        rvec = xw - pos
        ra = _norm(rvec)
        if live is None:
            rw = ra
            keep = np.ones(tw.size, bool)
        else:
            np.copyto(rw, ra, where=live)
            keep = live
        if it == 0:
            keep = keep & _look_side_ok(rvec, vel, pos, look_side)
        dopfact = _dot(rvec, vel)
        c1 = -_dot(vel, vel)
        if doppler.have_data:
            if doppler.bounds_error:
                keep = keep & doppler.contains(tw, ra) & doppler.contains(tw, ra + delta_range)
            # Frozen and non-finite points are evaluated somewhere harmless.
            ev = keep & np.isfinite(tw) & np.isfinite(ra)
            ts = np.where(ev, tw, doppler.y_start)
            rs = np.where(ev, ra, doppler.x_start)
            fdop = 0.5 * wavelength * doppler.eval(ts, rs)
            fdopder = (0.5 * wavelength * doppler.eval(ts, rs + delta_range) - fdop) / delta_range
            fn = dopfact - fdop * ra
            fnprime = c1 + (fdop / ra + fdopder) * dopfact
        else:
            # A zero-Doppler grid: every Doppler term is an exact zero.
            fn, fnprime = dopfact, c1
        with np.errstate(invalid="ignore", divide="ignore"):
            step = fn / fnprime
        small = np.abs(step) < threshold
        cw |= keep & small
        # A NaN step (the time left the orbit) can never converge: freeze it.
        live = keep & ~small & np.isfinite(step)
        dtw = np.where(live, step, 0.0)
        nlive = int(np.count_nonzero(live))
        if nlive == 0:
            break
        if 2 * nlive <= tw.size:
            if index is None:
                index = np.arange(npts)
            t[index], r[index], converged[index] = tw, rw, cw
            sel = np.flatnonzero(live)
            index, xw, tw, dtw = index[sel], xw[sel], tw[sel], dtw[sel]
            rw, cw, live = rw[sel], cw[sel], None
    if index is None:
        return tw, rw, cw
    t[index], r[index], converged[index] = tw, rw, cw
    return t, r, converged


def rdr2geo(t, r, orbit, height=0.0, doppler_hz=0.0, wavelength=1.0,
            look_side="left", threshold=0.05, maxiter=25, extraiter=10,
            h0=None, dem=None):
    """``detail::rdr2geo``: radar ``(t, r)`` -> target ``(lon, lat, h)``.

    The target height comes from ``dem(lon_rad, lat_rad)`` when given, else the
    constant ``height`` (ISCE3's raster-less ``DEMInterpolator``). ``h0`` is the
    starting height guess (NaN/None = the satellite's nadir height, as ISCE3).
    Returns ``(llh, converged)`` with ``llh`` an ``(n, 3)`` array.
    """
    t = np.atleast_1d(np.asarray(t, float))
    r = np.broadcast_to(np.asarray(r, float), t.shape).astype(float)
    fd = np.broadcast_to(np.asarray(doppler_hz, float), t.shape)
    pos, vel = orbit.interpolate(t)

    nhat = -_normalized(pos)
    chat = _normalized(np.cross(nhat, vel))
    that = _normalized(np.cross(chat, nhat))
    vmag = _norm(vel)
    dopfact = 0.5 * wavelength * fd * r / vmag
    vhat = vel / vmag[:, None]
    ndotv = _dot(nhat, vhat)
    vdott = _dot(vhat, that)

    major = WGS84_A
    minor = major * np.sqrt(1.0 - WGS84_E2)
    sat_dist = _norm(pos)
    eta = 1.0 / np.sqrt((pos[:, 0] / major) ** 2 + (pos[:, 1] / major) ** 2
                        + (pos[:, 2] / minor) ** 2)
    radius = eta * sat_dist
    nadir_height = (1.0 - eta) * sat_dist
    right = str(look_side).lower() == "right"

    def height_at(llh):
        if dem is None:
            return np.broadcast_to(np.asarray(height, float), llh.shape[:-1])
        return dem(llh[..., 0], llh[..., 1])

    def update_llh(h, sel):
        a = sat_dist[sel]
        b = radius[sel] + h
        rr = r[sel]
        cos_theta = 0.5 * (a / rr + rr / a - (b / a) * (b / rr))
        sin_theta = np.sqrt(1.0 - cos_theta * cos_theta)
        gamma = rr * cos_theta
        alpha = (dopfact[sel] - gamma * ndotv[sel]) / vdott[sel]
        beta = np.sqrt((rr * sin_theta) ** 2 - alpha * alpha)
        beta = beta if right else -beta
        delta = (alpha[:, None] * that[sel] + beta[:, None] * chat[sel]
                 + gamma[:, None] * nhat[sel])
        return xyz_to_lonlat(pos[sel] + delta)

    n = t.size
    if h0 is None:
        h = nadir_height.copy()
    else:
        h = np.broadcast_to(np.asarray(h0, float), (n,)).astype(float).copy()
        h = np.where(np.isnan(h), nadir_height, h)
    converged = np.zeros(n, bool)
    active = np.ones(n, bool)
    llh_old = np.full((n, 3), np.nan)
    for i in range(maxiter + extraiter):
        active &= ~(nadir_height - h >= r)
        sel = np.flatnonzero(active)
        if sel.size == 0:
            break
        llh_new = update_llh(h[sel], sel)
        llh_new[:, 2] = height_at(llh_new)
        xyz_new = lonlat_to_xyz(llh_new[:, 0], llh_new[:, 1], llh_new[:, 2])
        h[sel] = _norm(xyz_new) - radius[sel]
        dr = np.abs(r[sel] - _norm(pos[sel] - xyz_new))
        done = dr < threshold
        converged[sel[done]] = True
        active[sel[done]] = False
        if i > maxiter:
            still = ~done
            xyz_old = lonlat_to_xyz(llh_old[sel, 0], llh_old[sel, 1], llh_old[sel, 2])
            xyz_avg = 0.5 * (xyz_old + xyz_new)
            llh_avg = xyz_to_lonlat(xyz_avg)
            llh_new = np.where(still[:, None], llh_avg, llh_new)
            h[sel] = np.where(still, _norm(xyz_avg) - radius[sel], h[sel])
        llh_old[sel] = llh_new
    llh = update_llh(h, np.arange(n))
    return llh, converged


def geo_perimeter(radar_grid, orbit, height=0.0, points_per_edge=11):
    """``getGeoPerimeter`` at a constant height: ``(n, 3)`` lon/lat/h (rad, rad, m).

    Counter-clockwise from ``(t0, r0)`` (the vertex order depends on the look
    side), ``4 * (points_per_edge - 1)`` points, zero-Doppler grid.
    """
    t0, t1 = radar_grid.sensing_start, radar_grid.sensing_stop
    r0, r1 = radar_grid.starting_range, radar_grid.end_range
    if str(radar_grid.look_side).lower() == "left":
        vertices = [(t0, r0), (t1, r0), (t1, r1), (t0, r1), (t0, r0)]
    else:
        vertices = [(t0, r0), (t0, r1), (t1, r1), (t1, r0), (t0, r0)]
    frac = np.arange(points_per_edge - 1) / (points_per_edge - 1)
    ts, rs = [], []
    for (ta, ra), (tb, rb) in zip(vertices[:-1], vertices[1:]):
        ts.append(ta + frac * (tb - ta))
        rs.append(ra + frac * (rb - ra))
    llh, ok = rdr2geo(np.concatenate(ts), np.concatenate(rs), orbit, height=height,
                      wavelength=radar_grid.wavelength, look_side=radar_grid.look_side,
                      threshold=1e-5, maxiter=50, extraiter=10, h0=height)
    if not ok.all():
        raise RuntimeError(f"perimeter rdr2geo failed at height {height}")
    return llh


def geo_bounding_box(radar_grid, orbit, epsg, heights=(-500.0, 9000.0),
                     margin=0.0, points_per_edge=11):
    """``getGeoBoundingBox``: ``(x_min, x_max, y_min, y_max)`` in ``epsg``.

    ``margin`` is in degrees and, as in ISCE3, converted to metres
    (``deg * pi / 180 * 6.37e6``) for projected systems.
    """
    from pyproj import Transformer

    xs, ys = [], []
    for h in heights:
        llh = geo_perimeter(radar_grid, orbit, h, points_per_edge)
        lon, lat = np.degrees(llh[:, 0]), np.degrees(llh[:, 1])
        if int(epsg) == 4326:
            x, y = lon, lat
        else:
            x, y = Transformer.from_crs("EPSG:4326", f"EPSG:{int(epsg)}",
                                        always_xy=True).transform(lon, lat)
        xs.append(np.asarray(x))
        ys.append(np.asarray(y))
    xs, ys = np.concatenate(xs), np.concatenate(ys)
    delta = margin if int(epsg) == 4326 else margin * (np.pi / 180.0) * 6.37e6
    return (float(xs.min() - delta), float(xs.max() + delta),
            float(ys.min() - delta), float(ys.max() + delta))


# ---------------------------------------------------------------------------
# Metadata cubes (isce3::geometry::makeRadarGridCubes)
# ---------------------------------------------------------------------------
#: The ``metadata/radarGrid`` layers, with the dtype ISCE3 writes each in.
CUBE_LAYERS = {
    "slantRange": np.float64,
    "zeroDopplerAzimuthTime": np.float64,
    "incidenceAngle": np.float32,
    "losUnitVectorX": np.float32,
    "losUnitVectorY": np.float32,
    "alongTrackUnitVectorX": np.float32,
    "alongTrackUnitVectorY": np.float32,
    "elevationAngle": np.float32,
    "groundTrackVelocity": np.float64,
}


def radar_grid_cube_layer(lon, lat, height, radar_grid, orbit, native_doppler,
                          threshold=1e-8, maxiter=25, delta_range=1e-8):
    """One height of the ``metadata/radarGrid`` cube at nodes ``(lon, lat)``.

    ``lon``/``lat`` are radians of any shape; returns a dict of arrays of that
    shape keyed like :data:`CUBE_LAYERS`. Port of ``makeRadarGridCubes`` with
    ``flag_ground_velocity_from_rdr2geo=True`` (the binding's default):

    - range and time come from a **zero-Doppler** ``geo2rdr``;
    - every vector layer comes from the satellite state at the **native-Doppler**
      time (a second ``geo2rdr`` against the RSLC's Doppler centroid, bounds
      checking off) -- so the LOS is the beam-centre look, not the zero-Doppler
      one;
    - ground-track velocity is a finite difference of two constant-height
      ``rdr2geo`` solutions half a PRI either side of the zero-Doppler time.

    ISCE3 seeds each node's ``geo2rdr`` with its neighbour's solution; this
    seeds every node at mid-swath. The answers differ by less than the
    convergence threshold (measured: 6.4 nm in range and 1.4e-8 s in time
    against a real cube).
    """
    lon = np.asarray(lon, float)
    lat = np.asarray(lat, float)
    shape = lon.shape
    llh = np.stack([lon.ravel(), lat.ravel(),
                    np.full(lon.size, float(height))], axis=1)
    wvl, side = radar_grid.wavelength, radar_grid.look_side
    out = {name: np.full(lon.size, np.nan, dtype=dt) for name, dt in CUBE_LAYERS.items()}

    t, r, ok = geo2rdr(llh, orbit, None, wvl, side, t0=radar_grid.sensing_mid,
                       threshold=threshold, maxiter=maxiter, delta_range=delta_range)
    out["slantRange"][ok] = r[ok]
    out["zeroDopplerAzimuthTime"][ok] = t[ok]

    sel = np.flatnonzero(ok)
    doppler = native_doppler
    if doppler is not None and doppler.have_data:
        doppler = LUT2d(doppler.x_start, doppler.y_start, doppler.dx, doppler.dy,
                        doppler.data, doppler.method, bounds_error=False)
    tn, _, okn = geo2rdr(llh[sel], orbit, doppler, wvl, side,
                         t0=radar_grid.sensing_mid, threshold=threshold,
                         maxiter=maxiter, delta_range=delta_range)
    sel, tn = sel[okn], tn[okn]
    if sel.size:
        target_xyz = lonlat_to_xyz(llh[sel, 0], llh[sel, 1], llh[sel, 2])

        # Ground-track velocity by rdr2geo finite differences.
        dt = 1.0 / radar_grid.prf
        tz, rz = t[sel], r[sel]
        kw = dict(height=float(height), doppler_hz=0.0, wavelength=wvl,
                  look_side=side, threshold=threshold, maxiter=maxiter,
                  extraiter=0, h0=float(height))
        llh_b, conv_b = rdr2geo(tz - dt / 2.0, rz, orbit, **kw)
        llh_a, conv_a = rdr2geo(tz + dt / 2.0, rz, orbit, **kw)
        step = np.full(sel.size, dt)
        llh_b = np.where((~conv_b)[:, None], llh[sel], llh_b)
        llh_a = np.where((conv_b & ~conv_a)[:, None], llh[sel], llh_a)
        step = np.where(~conv_b | ~conv_a, dt / 2.0, step)
        xyz_b = lonlat_to_xyz(llh_b[:, 0], llh_b[:, 1], llh_b[:, 2])
        xyz_a = lonlat_to_xyz(llh_a[:, 0], llh_a[:, 1], llh_a[:, 2])
        gv = _norm((xyz_a - xyz_b) / step[:, None])
        out["groundTrackVelocity"][sel] = np.where(conv_b | conv_a, gv, np.nan)

        sat, vel = orbit.interpolate(tn)
        look = _normalized(sat - target_xyz)
        sat_llh = xyz_to_lonlat(sat)
        look_sat = _normalized(np.einsum(
            "nij,nj->ni", xyz_to_enu_matrix(sat_llh[:, 1], sat_llh[:, 0]), look))
        out["elevationAngle"][sel] = np.degrees(np.arccos(look_sat[:, 2]))

        enu = xyz_to_enu_matrix(llh[sel, 1], llh[sel, 0])
        look_enu = _normalized(np.einsum("nij,nj->ni", enu, look))
        out["incidenceAngle"][sel] = np.degrees(np.arccos(np.abs(look_enu[:, 2])))
        out["losUnitVectorX"][sel] = look_enu[:, 0]
        out["losUnitVectorY"][sel] = look_enu[:, 1]

        along = np.einsum("nij,nj->ni", enu, vel)
        horiz = np.sqrt(along[:, 0] ** 2 + along[:, 1] ** 2)
        out["alongTrackUnitVectorX"][sel] = along[:, 0] / horiz
        out["alongTrackUnitVectorY"][sel] = along[:, 1] / horiz

    return {name: arr.reshape(shape) for name, arr in out.items()}
