"""The fault's surface trace: the one piece of geometry the user supplies.

Everything about the fault -- where the mesh goes, which way is along-strike,
which side of it a quadtree cell sits on -- is derived from a polyline of
lon/lat vertices. Traces are usually digitised in Google Earth and arrive as
KML, so both KML and the plain two-column ASCII that ``gmt kml2gmt`` produces are
read here.

KML is XML, so it is parsed with the standard library rather than by shelling out
to GMT: ``pygmt`` is an optional extra in this package (see
:mod:`nisar_tools.mask`) and requiring it merely to read a list of coordinates
would be a poor trade.
"""

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .frame import LocalFrame

_KML_NS = {"kml": "http://www.opengis.net/kml/2.2"}


def _read_kml(path):
    """Every ``<coordinates>`` block of a KML, concatenated, as lon/lat.

    The namespace is declared on the root of every KML Google Earth writes, but
    hand-edited files sometimes drop it, so the search falls back to matching on
    the local tag name.
    """
    root = ET.parse(path).getroot()
    blocks = root.findall(".//kml:coordinates", _KML_NS)
    if not blocks:
        blocks = [el for el in root.iter() if el.tag.rsplit("}", 1)[-1] == "coordinates"]
    if not blocks:
        raise ValueError(f"No <coordinates> element in {path}")

    lon, lat = [], []
    for block in blocks:
        for token in (block.text or "").split():
            # "lon,lat" or "lon,lat,alt" -- the altitude is always 0 for a trace
            # digitised on the ground and is discarded either way.
            parts = token.split(",")
            if len(parts) < 2:
                continue
            lon.append(float(parts[0]))
            lat.append(float(parts[1]))
    return np.asarray(lon), np.asarray(lat)


def _read_text(path):
    """Whitespace-separated lon/lat, skipping comments and GMT segment headers.

    ``gmt kml2gmt`` emits one ``> -L"name"`` multisegment header before the
    coordinates (which is what the usual ``| tail -n +2`` strips), and GMT tables
    use ``#`` for comments. Both are skipped here so the raw ``kml2gmt`` output
    works whether or not it was trimmed.
    """
    lon, lat = [], []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith(">"):
            continue
        fields = re.split(r"[\s,]+", line)
        if len(fields) < 2:
            continue
        lon.append(float(fields[0]))
        lat.append(float(fields[1]))
    return np.asarray(lon), np.asarray(lat)


class FaultTrace:
    """A fault's surface trace as an ordered lon/lat polyline.

    Vertex *order* is meaningful: it fixes the along-strike direction, and hence
    the sign of strike-slip and which side of the fault is "left". Reversing the
    file reverses both consistently.
    """

    def __init__(self, lon, lat, name=None):
        lon = np.asarray(lon, dtype=float).ravel()
        lat = np.asarray(lat, dtype=float).ravel()
        if lon.size != lat.size:
            raise ValueError("lon and lat must have the same length")
        if lon.size < 2:
            raise ValueError("A fault trace needs at least two vertices")
        self.lon = lon
        self.lat = lat
        self.name = name

    # -- construction ------------------------------------------------------
    @classmethod
    def from_file(cls, path, name=None):
        """Read a trace from ``.kml`` or from two-column ASCII."""
        path = Path(path).expanduser()
        lon, lat = _read_kml(path) if path.suffix.lower() == ".kml" else _read_text(path)
        return cls(lon, lat, name=name or path.stem)

    @classmethod
    def from_lonlat(cls, lon, lat, name=None):
        return cls(lon, lat, name=name)

    # -- frame -------------------------------------------------------------
    def local_frame(self, **kwargs):
        """A :class:`~nisar_tools.slip.frame.LocalFrame` centred on this trace.

        The origin is the trace's midpoint in arc length, not the mean vertex --
        vertices are unevenly spaced (4.2-13.7 km on the Venezuela trace), so the
        mean would be pulled toward whichever end was digitised more finely.
        """
        mid_lon, mid_lat = self.midpoint()
        return LocalFrame(mid_lon, mid_lat, **kwargs)

    def midpoint(self):
        """The lon/lat at half the trace's length."""
        # A provisional frame centred on the mean vertex is accurate enough to
        # measure arc length with; the answer only picks the final origin.
        frame = LocalFrame(float(self.lon.mean()), float(self.lat.mean()))
        x, y = self.to_local(frame)
        s = _arc_length(x, y)
        half = s[-1] / 2.0
        xm = np.interp(half, s, x)
        ym = np.interp(half, s, y)
        return frame.to_lonlat(xm, ym)

    def to_local(self, frame):
        """Project the vertices into ``frame``; returns ``(x, y)`` in metres."""
        return frame.to_local(self.lon, self.lat)

    # -- geometry ----------------------------------------------------------
    def length(self, frame=None):
        """Total trace length in metres."""
        frame = frame or self.local_frame()
        x, y = self.to_local(frame)
        return float(_arc_length(x, y)[-1])

    def resample(self, spacing, frame=None):
        """A new trace with vertices equally spaced by ``spacing`` metres.

        The mesh needs a regular along-strike lattice; a digitised trace does
        not have one. Resampling is done in the local frame (metres) and the
        result converted back to lon/lat, so the class stays lon/lat-native.
        The spacing is adjusted slightly so a whole number of intervals covers
        the trace exactly -- otherwise the last patch would be a sliver.
        """
        frame = frame or self.local_frame()
        x, y = self.to_local(frame)
        s = _arc_length(x, y)
        n = max(1, int(round(s[-1] / float(spacing))))
        s_new = np.linspace(0.0, s[-1], n + 1)
        lon, lat = frame.to_lonlat(np.interp(s_new, s, x), np.interp(s_new, s, y))
        return FaultTrace(lon, lat, name=self.name)

    def tangents(self, frame=None):
        """Unit along-strike vectors at each vertex, in the local frame.

        Averaged over the two adjacent segments at interior vertices (one-sided
        at the ends), so the direction is continuous along a curving trace.
        """
        frame = frame or self.local_frame()
        x, y = self.to_local(frame)
        dx = np.gradient(x)
        dy = np.gradient(y)
        norm = np.hypot(dx, dy)
        norm[norm == 0] = 1.0
        return dx / norm, dy / norm

    def normals(self, frame=None):
        """Unit **left-hand** normals, i.e. ``(-t_y, t_x)`` per vertex.

        This is the reference direction that fixes triangle winding, and hence
        the sign of slip, across the whole mesh. Deriving it from the trace
        rather than from a fixed axis is deliberate: SlipSolve's
        ``orientTriFromLeft.m`` tests the normal against ``[1 0 0]``, which is
        degenerate for a fault striking near east-west (the Venezuela trace
        strikes 84 degrees, so its normal is very nearly the y axis and the test
        collapses).
        """
        tx, ty = self.tangents(frame)
        return -ty, tx

    def side(self, x, y, frame, tol=0.0):
        """Which side of the trace each local-frame point lies on.

        Returns ``+1`` on the left-hand side (the side :meth:`normals` points
        to), ``-1`` on the right, and ``0`` within ``tol`` metres of the trace.
        The sign is taken from the *nearest* segment, so it stays meaningful for
        a curving, multi-segment trace.
        """
        d2, cross, _, _ = self._nearest_segment(x, y, frame)
        s = np.sign(cross).astype(np.int8)
        if tol > 0:
            s[d2 <= tol * tol] = 0
        return s

    def distance(self, x, y, frame):
        """Perpendicular distance in metres from each point to the trace."""
        d2, _, _, _ = self._nearest_segment(x, y, frame)
        return np.sqrt(d2)

    def to_curvilinear(self, x, y, frame):
        """Map local-frame points to ``(arc length, signed cross-strike offset)``.

        The curvilinear coordinate system the fault surface is fitted in: ``s`` is
        the arc length of the closest point on the trace, and ``cross`` is the
        signed perpendicular distance, positive on the side
        :meth:`normals` points to. Together with :meth:`from_curvilinear` this is
        the change of variables that lets a dipping fault be described as
        ``cross = f(s, depth)`` -- one scalar field over a rectangle -- rather than
        as a surface in three dimensions.

        The reference implementation instead picks whichever of the map axes the
        trace spans further and treats that as "along" (``build_fault_geometry.m``'s
        ``fitCrossX = yRange >= xRange``). That is degenerate for a fault striking
        near 45 degrees and mis-parameterises any trace that curves through it;
        arc length is well defined at every strike.
        """
        d2, cross, nearest, t = self._nearest_segment(x, y, frame)
        px, py = self.to_local(frame)
        s_vertex = _arc_length(px, py)
        seg_len = np.hypot(np.diff(px), np.diff(py))
        s = s_vertex[nearest] + t * seg_len[nearest]
        return s, np.sign(cross) * np.sqrt(d2)

    def from_curvilinear(self, s, cross, frame):
        """The inverse of :meth:`to_curvilinear`; returns local-frame ``(x, y)``.

        Positions on the trace are interpolated in arc length, and the offset is
        applied along the interpolated left-hand normal -- the same normal
        :class:`~nisar_tools.slip.mesh.FaultMesh` winds its elements against, so a
        positive ``cross`` always puts the fault surface on the same side.
        """
        px, py = self.to_local(frame)
        s_vertex = _arc_length(px, py)
        nx, ny = self.normals(frame)

        s = np.atleast_1d(np.asarray(s, dtype=float))
        cross = np.broadcast_to(np.asarray(cross, dtype=float), s.shape)
        x = np.interp(s, s_vertex, px)
        y = np.interp(s, s_vertex, py)
        # Re-normalise: interpolating two unit normals across a bend gives a
        # shorter vector, which would quietly shrink the offset at every corner.
        ex = np.interp(s, s_vertex, nx)
        ey = np.interp(s, s_vertex, ny)
        norm = np.hypot(ex, ey)
        norm[norm == 0] = 1.0
        return x + cross * ex / norm, y + cross * ey / norm

    def min_curvature_radius(self, frame=None):
        """Smallest radius of curvature along the trace, in metres.

        The limit on how far the trace can be projected down dip: an offset that
        exceeds this folds the fault surface back through itself, and the
        resulting mesh has inverted elements. Measured on the San Sebastian trace
        resampled to 3 km this is 74.4 km, so a 20 km fault at 45 degrees (a
        20 km offset) is safe -- but a tighter bend would not be.
        """
        frame = frame or self.local_frame()
        x, y = self.to_local(frame)
        s = _arc_length(x, y)
        if s[-1] == 0:
            return np.inf
        d1x, d1y = np.gradient(x, s), np.gradient(y, s)
        d2x, d2y = np.gradient(d1x, s), np.gradient(d1y, s)
        kappa = np.abs(d1x * d2y - d1y * d2x)
        if not np.any(kappa > 0):
            return np.inf
        return float(1.0 / kappa.max())

    def _nearest_segment(self, x, y, frame):
        """Closest segment for each point: squared distance, side, index, position.

        The side comes from the cross product against the *clamped* closest-point
        offset rather than the raw vertex offset, so points near a concave corner
        are classified by the geometry that actually is nearest to them instead of
        by whichever segment's infinite extension happens to win.
        """
        px, py = self.to_local(frame)
        x = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
        y = np.atleast_1d(np.asarray(y, dtype=float)).ravel()

        x0, y0 = px[:-1], py[:-1]
        vx, vy = px[1:] - x0, py[1:] - y0
        vv = vx * vx + vy * vy
        vv = np.where(vv == 0, np.finfo(float).tiny, vv)

        wx = x[:, None] - x0[None, :]
        wy = y[:, None] - y0[None, :]
        t = np.clip((wx * vx + wy * vy) / vv, 0.0, 1.0)
        dx = wx - t * vx
        dy = wy - t * vy
        d2 = dx * dx + dy * dy

        nearest = np.argmin(d2, axis=1)
        rows = np.arange(x.size)
        cross = vx[nearest] * wy[rows, nearest] - vy[nearest] * wx[rows, nearest]
        return d2[rows, nearest], cross, nearest, t[rows, nearest]

    def __len__(self):
        return self.lon.size

    def __repr__(self):
        return (f"<FaultTrace {self.name!r} n={len(self)} "
                f"length={self.length() / 1e3:.1f}km>")


class FaultSegment:
    """A straight deep-control segment: where one stretch of fault goes at depth.

    The reference implementation's unit of dip. A fault is divided into a handful
    of segments -- each a straight line in the local frame -- and each is given
    one dip; the segment's *bottom* line is that line pushed down-dip, and the
    fitted surface is what smoothly connects those bottom lines to the surface
    trace. So a segment is not a piece of the trace, it is the piece of the
    *plan view* whose dip is being asserted.

    Coordinates are local-frame metres, not lon/lat: the reference reads them
    straight out of a four-number file (``build_fault_geometry.m`` divides by
    1e3 to reach its kilometre units).
    """

    def __init__(self, x_begin, y_begin, x_end, y_end, name=None):
        self.x_begin = float(x_begin)
        self.y_begin = float(y_begin)
        self.x_end = float(x_end)
        self.y_end = float(y_end)
        self.name = name
        if self.length == 0.0:
            raise ValueError("A fault segment must have non-zero length")

    @classmethod
    def from_file(cls, path):
        """Read ``x_begin y_begin x_end y_end`` (metres) from whitespace ASCII."""
        path = Path(path).expanduser()
        fields = [float(v) for v in re.split(r"[\s,]+", path.read_text().strip()) if v]
        if len(fields) < 4:
            raise ValueError(
                f"{path} has {len(fields)} numbers; a segment file needs four "
                "(x_begin y_begin x_end y_end, in metres)"
            )
        return cls(*fields[:4], name=path.stem)

    @classmethod
    def from_files(cls, paths):
        return [cls.from_file(p) for p in paths]

    @classmethod
    def from_trace(cls, trace, frame, count=1):
        """Split ``trace`` into ``count`` straight segments of equal arc length.

        The convenience path for "I have a trace and one dip per stretch of it"
        without hand-writing segment files. Endpoints are taken on the trace, so
        a single segment is the chord from end to end.
        """
        x, y = trace.to_local(frame)
        s = _arc_length(x, y)
        edges = np.linspace(0.0, s[-1], int(count) + 1)
        px, py = np.interp(edges, s, x), np.interp(edges, s, y)
        return [cls(px[i], py[i], px[i + 1], py[i + 1], name=f"{trace.name}_{i + 1:03d}")
                for i in range(int(count))]

    @property
    def length(self):
        return float(np.hypot(self.x_end - self.x_begin, self.y_end - self.y_begin))

    def strike_vector(self):
        """Unit vector from begin to end."""
        return np.array([self.x_end - self.x_begin,
                         self.y_end - self.y_begin]) / self.length

    def project(self, depth, dip_deg):
        """The segment pushed ``depth`` metres down at ``dip_deg``; two endpoints.

        A port of ``project_segment_3d``: the horizontal offset is
        ``depth / tan(dip)`` along the segment's **left-hand** normal
        ``(-s_y, s_x)``. Returns ``(x, y)`` arrays of length two, both at
        ``z = -depth``.

        A dip greater than 90 degrees is legal and meaningful -- ``tan`` goes
        negative and the fault leans the other way. The reference's Myanmar
        configuration uses ``[75 75 70 80 85 90 100]``, so this is exercised, not
        hypothetical; do not clamp it into ``[0, 90]``.

        At exactly 90 the offset is written as a literal zero rather than
        ``depth / tan(pi/2)``, which in float64 is 1.2e-12 m rather than 0. That
        residue is harmless in itself, but a literal zero is what makes a
        vertical mesh built through here bit-identical to
        :meth:`~nisar_tools.slip.mesh.FaultMesh.vertical`.
        """
        offset = dip_offset(depth, dip_deg)
        sx, sy = self.strike_vector()
        nx, ny = -sy, sx
        return (np.array([self.x_begin, self.x_end]) + offset * nx,
                np.array([self.y_begin, self.y_end]) + offset * ny)

    def __repr__(self):
        return (f"<FaultSegment {self.name!r} "
                f"({self.x_begin / 1e3:.1f},{self.y_begin / 1e3:.1f})->"
                f"({self.x_end / 1e3:.1f},{self.y_end / 1e3:.1f}) km "
                f"length={self.length / 1e3:.1f}km>")


#: A dip this close to vertical is treated as exactly vertical.
#:
#: Both the triangular-dislocation solution and Okada's have a removable
#: singularity at 90 degrees which each handles cleanly *at* 90 and loses
#: catastrophically beside. Measured here (a rectangle as two triangular elements
#: against :mod:`nisar_tools.slip._okada`, max-abs relative error on the
#: strike-slip component)::
#:
#:     dip      85      89    89.9   89.99  89.999  89.9999   90.000
#:          4.9e-12 1.4e-10 2.4e-07 1.2e-04 2.6e-02  1.9e+02  2.2e-14
#:
#: It is a *resonance*, not a gradual decay: the error is negligible at exactly
#: 90, grows to 190x the signal a ten-thousandth of a degree away, and falls back
#: to nothing by 85. ``cutde`` -- an independent port of the same algorithm --
#: tracks our answer far more closely than either tracks Okada's inside the band,
#: which identifies it as a cancellation shared by the formulation rather than an
#: artefact of the oracle. A self-comparison sweep puts the band's onset near
#: 1e-6 degrees at element sizes of 300 m, 3 km and 30 km alike, so it is an angle
#: and not a length.
#:
#: 1e-2 degrees clears the whole band with two orders of margin, leaves a worst
#: unsnapped error of 1.2e-4 relative, and costs 3.5 m of geometry at 20 km depth
#: -- nothing beside a 3 km element. (This constant used to live in ``mesh.py``
#: at 1e-6, which would let 89.9999 through, and was referenced nowhere.)
VERTICAL_TOL_DEG = 1e-2


def dip_offset(depth, dip_deg):
    """Horizontal down-dip offset ``depth / tan(dip)``, snapped near vertical.

    Shared by :meth:`FaultSegment.project` and the mesh builders so that
    "vertical" means the same thing everywhere. See :data:`VERTICAL_TOL_DEG`.
    """
    dip_deg = np.asarray(dip_deg, dtype=float)
    vertical = np.abs(dip_deg - 90.0) < VERTICAL_TOL_DEG
    safe = np.where(vertical, 45.0, dip_deg)      # any non-degenerate placeholder
    offset = np.asarray(depth, dtype=float) / np.tan(np.radians(safe))
    return np.where(vertical, 0.0, offset)[()]


def _arc_length(x, y):
    """Cumulative distance along a polyline, starting at 0."""
    step = np.hypot(np.diff(x), np.diff(y))
    return np.concatenate([[0.0], np.cumsum(step)])
