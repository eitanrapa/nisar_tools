"""The triangulated fault surface the inversion solves slip on.

The fault is parameterized by **arc length along the trace and depth**, an
``(s, z)`` lattice, and the mesh is that lattice's quads cut in two. Two
consequences make this worth doing rather than triangulating scattered points:

* **Winding is consistent by construction.** Flipping a triangle's vertex order
  flips its normal, which negates its dip-slip Green's-function column while
  leaving strike-slip untouched -- so on a strike-slip fault an inconsistent
  winding gives a plausible model with a randomly-signed dip-slip field. The
  reference implementation guards this by testing each normal against a fixed
  axis (``orientTriFromLeft.m``), which has almost no margin on a fault striking
  near east-west. Generating triangles from the lattice removes the problem
  instead of policing it.
* **Boundary nodes are exact.** "The along-strike ends" is ``s == s.min()`` and
  ``s == s.max()``, not an ``argmin`` over a map coordinate.

Phase one extrudes the trace straight down (a vertical fault). The
parameterization is already the one a dipping or listric fault needs -- only the
map position of each ``(s, z)`` node changes -- so nothing downstream has to know
which it is looking at.
"""

import hashlib

import numpy as np
import xarray as xr

# A dip this close to vertical is treated as exactly vertical. Both the
# triangular-dislocation solution and Okada's have a removable singularity at 90
# degrees which each handles cleanly *at* 90 and loses precision beside: a
# 1e-6-degree tilt over 20 km of depth is a 0.35 mm offset, physically nothing,
# but it lands in the band where digits are lost (see
# tests/test_slip_tde.py::test_near_vertical_precision_band).
VERTICAL_TOL_DEG = 1e-6


class FaultMesh:
    """A triangulated fault surface in local ENU metres, ``z <= 0``.

    Built through :meth:`vertical` (or, later, a dipping equivalent) rather than
    directly; the constructor exists to reopen a persisted mesh.
    """

    def __init__(self, nodes, triangles, params, frame=None, epsg=None, attrs=None):
        nodes = np.asarray(nodes, dtype=float)
        triangles = np.asarray(triangles, dtype=int)
        params = np.asarray(params, dtype=float)

        if nodes.ndim != 2 or nodes.shape[1] != 3:
            raise ValueError("nodes must be (n, 3)")
        if triangles.ndim != 2 or triangles.shape[1] != 3:
            raise ValueError("triangles must be (m, 3)")
        # Metres, negative-up. The reference mixes kilometres and metres between
        # the mesh and the layered Green's functions, which is a silent
        # thousand-fold error; refusing anything else here is the cheap guard.
        if np.any(nodes[:, 2] > 0):
            raise ValueError("Fault nodes must be at or below the free surface (z <= 0)")
        if np.abs(nodes).max() > 1e7:
            raise ValueError(
                "Fault node coordinates look too large for metres; are they in kilometres?"
            )

        self.nodes = nodes
        self.triangles = triangles
        self.params = params
        self.frame = frame
        self.attrs = dict(attrs or {})
        if epsg is not None:
            self.attrs.setdefault("epsg", int(epsg))

    # -- construction ------------------------------------------------------
    @classmethod
    def vertical(cls, trace, frame=None, max_depth=20e3, edge_length=3e3, top_depth=0.0):
        """Extrude ``trace`` straight down into a vertical fault.

        ``max_depth`` and ``edge_length`` are metres; ``top_depth`` is the depth
        of the shallowest node (``0`` = surface-breaking). The along-strike
        spacing is set by resampling the trace, so it is uniform in arc length
        even though the digitised vertices are not.

        The extrusion is *exact*: every depth level shares one map position, so
        the element edges are exactly horizontal and vertical and the dip is
        exactly 90 degrees -- rather than ``max_depth / tan(dip)`` with ``dip``
        set to 90, which would leave a sub-millimetre tilt in the numerically
        awkward band beside vertical.
        """
        frame = frame or trace.local_frame()
        if max_depth <= top_depth:
            raise ValueError("max_depth must be deeper than top_depth")

        resampled = trace.resample(edge_length, frame)
        tx, ty = resampled.to_local(frame)
        s = _arc_length(tx, ty)

        n_down = max(1, int(round((max_depth - top_depth) / float(edge_length))))
        depths = np.linspace(top_depth, max_depth, n_down + 1)

        ns = tx.size            # nodes along strike
        nz = depths.size        # nodes down dip
        nodes = np.empty((ns * nz, 3))
        params = np.empty((ns * nz, 2))
        for j, depth in enumerate(depths):
            sl = slice(j * ns, (j + 1) * ns)
            nodes[sl, 0] = tx
            nodes[sl, 1] = ty
            nodes[sl, 2] = -depth
            params[sl, 0] = s
            params[sl, 1] = -depth

        triangles = _lattice_triangles(ns, nz)

        attrs = {
            "kind": "vertical",
            "dip_deg": 90.0,
            "max_depth": float(max_depth),
            "top_depth": float(top_depth),
            "edge_length": float(edge_length),
            "trace_length": float(s[-1]),
            "trace_name": trace.name,
            "n_along": int(ns),
            "n_down": int(nz),
        }
        mesh = cls(nodes, triangles, params, frame=frame, attrs=attrs)
        mesh._check_winding(trace)
        return mesh

    # -- derived geometry --------------------------------------------------
    @property
    def n_nodes(self):
        return self.nodes.shape[0]

    @property
    def n_elements(self):
        return self.triangles.shape[0]

    @property
    def vertices(self):
        """The three vertex positions per element, shape ``(m, 3, 3)``."""
        return self.nodes[self.triangles]

    @property
    def normals(self):
        """Unit normals, ``normalize(cross(P2 - P1, P3 - P1))``.

        The same expression :mod:`nisar_tools.slip._tde` uses, so these *are* the
        elements' own ``Vnorm`` and fix the sign of both slip components.
        """
        v = self.vertices
        n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
        return n / np.linalg.norm(n, axis=1, keepdims=True)

    @property
    def areas(self):
        v = self.vertices
        return 0.5 * np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1)

    @property
    def centroids(self):
        return self.vertices.mean(axis=1)

    @property
    def element_params(self):
        """Each element's ``(s, z)`` centroid -- the coordinates to plot slip in."""
        return self.params[self.triangles].mean(axis=1)

    @property
    def dip(self):
        """Dip in degrees: 0 for a horizontal element, 90 for a vertical one."""
        return np.degrees(np.arccos(np.clip(np.abs(self.normals[:, 2]), 0.0, 1.0)))

    @property
    def strike(self):
        """Azimuth in degrees of the element's own ``cross(eZ, Vnorm)``.

        This is the strike direction :mod:`nisar_tools.slip._tde` measures
        strike-slip along, *not* necessarily the geological strike of the trace --
        for an east-striking fault the two are 180 degrees apart. Stored so the
        layered Green's functions, which need a strike angle, take it from the
        same place the displacements do.
        """
        n = self.normals
        return np.degrees(np.arctan2(-n[:, 1], n[:, 0])) % 360.0

    @property
    def neighbors(self):
        """Element adjacency, ``(m, 3)``, ``-1`` where an edge is on a boundary.

        Built from a shared-edge map rather than from a triangulation object, so
        it works for any mesh this class can hold.
        """
        edges = {}
        for t, tri in enumerate(self.triangles):
            for a, b in ((0, 1), (1, 2), (2, 0)):
                edges.setdefault(frozenset((tri[a], tri[b])), []).append(t)

        out = np.full((self.n_elements, 3), -1, dtype=int)
        for t, tri in enumerate(self.triangles):
            for k, (a, b) in enumerate(((0, 1), (1, 2), (2, 0))):
                shared = edges[frozenset((tri[a], tri[b]))]
                for other in shared:
                    if other != t:
                        out[t, k] = other
                        break
        return out

    # -- boundaries --------------------------------------------------------
    def boundary_elements(self, side):
        """Elements touching one edge of the fault, as a boolean mask.

        ``side`` is ``"bottom"``, ``"top"``, ``"left"`` or ``"right"``. Resolved
        in ``(s, z)`` parameter space, so "left" and "right" mean the along-strike
        ends whatever direction the fault happens to run -- the reference picks
        them by ``argmin``/``argmax`` of the north coordinate, which is nearly
        degenerate for an east-west fault and would anchor an interior node at
        any restraining bend that reaches further north than an endpoint.
        """
        s, z = self.params[:, 0], self.params[:, 1]
        if side == "bottom":
            on = z == z.min()
        elif side == "top":
            on = z == z.max()
        elif side == "left":
            on = s == s.min()
        elif side == "right":
            on = s == s.max()
        else:
            raise ValueError(
                f"Unknown boundary {side!r}; expected bottom, top, left or right"
            )
        return on[self.triangles].any(axis=1)

    # -- identity / persistence -------------------------------------------
    def digest(self):
        """A short hash of the geometry, for a stage's params dict.

        ``Workspace.hash_params`` JSON-serialises what it is given, so a mesh
        cannot go in directly -- and leaving it out would let a re-run at a
        different ``edge_length`` silently reuse a stale store.
        """
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(self.nodes, dtype=np.float64).tobytes())
        h.update(np.ascontiguousarray(self.triangles, dtype=np.int64).tobytes())
        return h.hexdigest()[:16]

    def to_dataset(self):
        """The mesh as an :class:`xarray.Dataset` on ``node`` and ``element``."""
        ds = xr.Dataset(
            {
                "node_x": ("node", self.nodes[:, 0]),
                "node_y": ("node", self.nodes[:, 1]),
                "node_z": ("node", self.nodes[:, 2]),
                "node_s": ("node", self.params[:, 0]),
                "triangles": (("element", "vertex"), self.triangles),
                "area": ("element", self.areas),
                "strike": ("element", self.strike),
                "dip": ("element", self.dip),
            }
        )
        ds.attrs.update(self.attrs)
        ds.attrs["digest"] = self.digest()
        if self.frame is not None:
            ds.attrs["frame"] = self.frame.to_dict()
        return ds

    @classmethod
    def from_dataset(cls, ds):
        from .frame import LocalFrame

        nodes = np.column_stack([ds["node_x"].values, ds["node_y"].values, ds["node_z"].values])
        params = np.column_stack([ds["node_s"].values, ds["node_z"].values])
        frame = ds.attrs.get("frame")
        return cls(
            nodes, ds["triangles"].values, params,
            frame=LocalFrame.from_dict(frame) if frame else None,
            attrs={k: v for k, v in ds.attrs.items() if k not in ("frame", "digest")},
        )

    # -- checks ------------------------------------------------------------
    def _check_winding(self, trace):
        """Every normal must lie on the trace's left-hand side.

        The invariant that keeps the dip-slip column's sign meaningful. Compared
        against the trace's own perpendicular rather than a fixed axis, so it
        holds at any strike.
        """
        nx, ny = trace.normals(self.frame)
        # Compare each element against the trace normal nearest its centroid in
        # arc length, which is the right reference for a curving fault.
        tx, ty = trace.to_local(self.frame)
        s_trace = _arc_length(tx, ty)
        s_elem = self.element_params[:, 0]
        ref_x = np.interp(s_elem, s_trace, nx)
        ref_y = np.interp(s_elem, s_trace, ny)

        n = self.normals
        dot = n[:, 0] * ref_x + n[:, 1] * ref_y
        if np.any(dot <= 0):
            bad = int(np.sum(dot <= 0))
            raise ValueError(
                f"{bad} of {self.n_elements} elements are wound against the trace's "
                "left-hand normal; their dip-slip Green's functions would carry the "
                "opposite sign to the rest of the mesh."
            )

    def __repr__(self):
        return (f"<FaultMesh {self.attrs.get('kind', '?')} "
                f"nodes={self.n_nodes} elements={self.n_elements} "
                f"depth={self.attrs.get('max_depth', 0) / 1e3:.0f}km>")


def _lattice_triangles(ns, nz):
    """Split an ``ns x nz`` node lattice into triangles with one fixed winding.

    Node ``(i, j)`` -- ``i`` along strike, ``j`` down dip -- is index
    ``j * ns + i``. Each quad becomes ``(a, b, c)`` and ``(a, c, d)`` with
    ``a = (i, j)``, ``b = (i + 1, j)``, ``c = (i + 1, j + 1)``, ``d = (i, j + 1)``.
    For a fault extruded downward along an increasing-``s`` trace, both halves'
    normals come out on the trace's left-hand side.
    """
    tris = []
    for j in range(nz - 1):
        for i in range(ns - 1):
            a = j * ns + i
            b = j * ns + i + 1
            c = (j + 1) * ns + i + 1
            d = (j + 1) * ns + i
            tris.append((a, b, c))
            tris.append((a, c, d))
    return np.array(tris, dtype=int)


def _arc_length(x, y):
    step = np.hypot(np.diff(x), np.diff(y))
    return np.concatenate([[0.0], np.cumsum(step)])
