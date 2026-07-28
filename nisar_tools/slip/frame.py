"""The local Cartesian frame every slip-inversion object is expressed in.

An elastic dislocation model lives in a flat, right-handed Cartesian space:
east, north and up in metres, with the free surface at ``z = 0`` and the fault
below it at ``z < 0``. The observations, the fault trace and the mesh must all
be in the *same* such space, or the Green's functions relate one geometry to
another.

The obvious candidate -- the UTM grid a :class:`~nisar_tools.los.LOSStack`
already sits on -- is not good enough: two tracks over one fault can fall in
different zones, and their x/y would then be metres from two different central
meridians. So :class:`LocalFrame` is a transverse Mercator with the central
meridian placed *on the study area*, with the projected origin subtracted so the
frame's ``(0, 0)`` is a chosen point. That is precisely what SlipSolve's
``ll2xy.m`` does with its ``lon_c`` argument, and why it takes one.

Distortion over a few hundred kilometres of an on-meridian transverse Mercator is
a few parts in 10\\ :sup:`4`, well below the resolution any slip model claims.
"""

from pyproj import CRS, Transformer

# UTM's own scale factor and false easting. Keeping them means a frame whose
# ``ref_lon`` happens to be a UTM central meridian reproduces that zone's grid
# exactly (up to the origin shift), which makes cross-checks against a UTM-gridded
# product easy to reason about.
_K0 = 0.9996
_FALSE_EASTING = 500000.0


class LocalFrame:
    """Local east/north metres about an origin, shared by every slip object.

    ``ref_lon`` is the projection's central meridian and defaults to
    ``origin_lon``; pass it separately only to reproduce an existing model built
    on a different meridian. ``ellps`` defaults to WGS84 -- set it to
    ``"clrk66"`` to reproduce SlipSolve's ``ll2xy.m`` numerically.
    """

    def __init__(self, origin_lon, origin_lat, ref_lon=None, ellps="WGS84"):
        self.origin_lon = float(origin_lon)
        self.origin_lat = float(origin_lat)
        self.ref_lon = float(origin_lon if ref_lon is None else ref_lon)
        self.ellps = str(ellps)

        self.crs = CRS.from_proj4(
            f"+proj=tmerc +lat_0=0 +lon_0={self.ref_lon} +k={_K0} "
            f"+x_0={_FALSE_EASTING} +y_0=0 +ellps={self.ellps} +units=m +no_defs"
        )
        self._fwd = Transformer.from_crs("EPSG:4326", self.crs, always_xy=True)
        self._inv = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)
        self._x0, self._y0 = self._fwd.transform(self.origin_lon, self.origin_lat)

    # -- conversions -------------------------------------------------------
    def to_local(self, lon, lat):
        """Project lon/lat to local ``(x, y)`` metres east/north of the origin."""
        x, y = self._fwd.transform(lon, lat)
        return x - self._x0, y - self._y0

    def to_lonlat(self, x, y):
        """Inverse of :meth:`to_local`."""
        return self._inv.transform(x + self._x0, y + self._y0)

    def from_epsg(self, x, y, epsg):
        """Bring coordinates already in a projected CRS into this frame.

        Used for a :class:`~nisar_tools.los.LOSStack`'s native UTM x/y, which
        would otherwise be metres from a different central meridian.
        """
        transformer = Transformer.from_crs(f"EPSG:{int(epsg)}", self.crs, always_xy=True)
        px, py = transformer.transform(x, y)
        return px - self._x0, py - self._y0

    # -- persistence -------------------------------------------------------
    def to_dict(self):
        """JSON-able description, stored in each object's ``attrs``.

        Every slip object records the frame it was built in so that combining
        two of them can *check* rather than assume they agree -- mixing frames
        is a silent metres-scale error, not a crash.
        """
        return {
            "origin_lon": self.origin_lon,
            "origin_lat": self.origin_lat,
            "ref_lon": self.ref_lon,
            "ellps": self.ellps,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(d["origin_lon"], d["origin_lat"],
                   ref_lon=d.get("ref_lon"), ellps=d.get("ellps", "WGS84"))

    def matches(self, other):
        """True if ``other`` (a frame or its dict form) is the same frame."""
        if isinstance(other, LocalFrame):
            other = other.to_dict()
        return other == self.to_dict()

    def require_match(self, other, what):
        """Raise unless ``other`` describes this same frame."""
        if not self.matches(other):
            raise ValueError(
                f"{what} was built in a different LocalFrame "
                f"({other!r} vs {self.to_dict()!r}); rebuild it in this frame."
            )

    def __repr__(self):
        return (f"<LocalFrame origin=({self.origin_lon:.4f}, {self.origin_lat:.4f}) "
                f"lon_0={self.ref_lon:.4f} {self.ellps}>")
