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
        d2, cross = self._nearest_segment(x, y, frame)
        s = np.sign(cross).astype(np.int8)
        if tol > 0:
            s[d2 <= tol * tol] = 0
        return s

    def distance(self, x, y, frame):
        """Perpendicular distance in metres from each point to the trace."""
        d2, _ = self._nearest_segment(x, y, frame)
        return np.sqrt(d2)

    def _nearest_segment(self, x, y, frame):
        """Squared distance to, and side of, the closest segment for each point.

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
        return d2[rows, nearest], cross

    def __len__(self):
        return self.lon.size

    def __repr__(self):
        return (f"<FaultTrace {self.name!r} n={len(self)} "
                f"length={self.length() / 1e3:.1f}km>")


def _arc_length(x, y):
    """Cumulative distance along a polyline, starting at 0."""
    step = np.hypot(np.diff(x), np.diff(y))
    return np.concatenate([[0.0], np.cumsum(step)])
