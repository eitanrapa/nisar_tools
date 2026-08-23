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

:meth:`FaultMesh.vertical` extrudes the trace straight down;
:meth:`FaultMesh.curved` moves each node cross-strike by a fitted
``cross(s, z)`` field (see :mod:`nisar_tools.slip.surface`). Only the map
position of a node changes between them -- its ``(s, z)`` parameters, and
therefore every boundary set, adjacency and persisted array, are identical in
form -- so nothing downstream has to know which it is looking at.
"""

import hashlib

import numpy as np
import xarray as xr

from .layout import (  # noqa: F401  (_lattice_triangles is re-exported)
    MAX_COUNT_RATIO,
    MeshLayout,
    _lattice_triangles,
    boundary_node_mask,
    graded_layout,
    require_boundaries,
)
from .trace import VERTICAL_TOL_DEG, FaultSegment, dip_offset  # noqa: F401

# There is deliberately NO near-vertical element guard here, and that is a
# measured decision rather than an oversight. An irregular triangulation puts
# most of a vertical mesh's elements a hair off 90 degrees -- 107 of 146 within
# 1e-2 degrees, median 1.8e-12 -- because `arccos(|n_z|)` cannot return exactly
# 90 when the normal's vertical component is at machine epsilon. That looks like
# the catastrophic band this package's notes record beside vertical, and it is
# not: measured against `cutde`, an independent implementation of the same
# triangular dislocation, that mesh's whole LOS design matrix agrees to
# **2.4e-14**, the same as a uniform mesh with no such elements at all. A single
# triangle agrees to 1e-14 at every tilt from 0 to 5 degrees, 1e-14 degrees
# included. The recorded band is a rectangle-as-two-TDEs disagreement with the
# *Okada* oracle, which is why the note says cutde tracks `_tde` far more
# closely than either tracks Okada inside it. A warning here would fire on two
# thirds of every irregular vertical mesh and teach the reader to ignore it.


class FaultMesh:
    """A triangulated fault surface in local ENU metres, ``z <= 0``.

    Built through :meth:`vertical` (or, later, a dipping equivalent) rather than
    directly; the constructor exists to reopen a persisted mesh.
    """

    def __init__(self, nodes, triangles, params, frame=None, epsg=None, attrs=None,
                 centres=None, centre_params=None):
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
        # Cell centres of the (s, z) lattice. Not used by the homogeneous
        # half-space engine, which needs only the element vertices, but they are
        # where the layered Green's functions place their point sources -- and
        # the reference fits them as their own surface rather than interpolating
        # the node grid, so they cannot be reconstructed here.
        self.centres = None if centres is None else np.asarray(centres, dtype=float)
        self.centre_params = (None if centre_params is None
                              else np.asarray(centre_params, dtype=float))
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

    @classmethod
    def curved(cls, trace, frame=None, *, segments=None, dips=None, uniform_dip=None,
               surface=None, bottom_trace=None, bottom_depth=None,
               max_depth=20e3, edge_length=3e3, top_depth=0.0,
               down_dip_levels=None, bias_w=1.0, bias_l=1.0, smoothness=None,
               depth_control=None, layout=None, depths=None, nodes_per_level=None,
               along_spacing=None, balance=MAX_COUNT_RATIO, **fit_kwargs):
        """A fault that dips, built from one dip per deep segment.

        The reference implementation's normal workflow: a surface trace, a set of
        straight *deep segments* in plan view, and one dip for each. Each
        segment is pushed down-dip to give a bottom line at ``max_depth``; a
        smooth surface is fitted through those lines and the trace
        (:class:`~nisar_tools.slip.surface.FaultSurface`); and the mesh nodes are
        that surface sampled on the ``(s, z)`` lattice.

        Four ways to say what the geometry is, in decreasing generality:

        * ``surface=`` -- a :class:`~nisar_tools.slip.surface.FaultSurface` you
          fitted yourself.
        * ``bottom_trace=`` -- a digitised bottom edge, at ``bottom_depth``
          (default ``max_depth``). The dip is whatever connects the two traces,
          so it varies continuously along strike and may reverse direction where
          the bottom trace crosses the surface trace -- which a list of dips can
          only express with hand-tuned values straddling 90 degrees.
        * ``segments=`` and ``dips=`` -- the reference's
          ``cfg.geometry.segmentFiles`` / ``segmentDipDegrees``. ``segments`` may
          be an integer, in which case the trace is chopped into that many equal
          chords and ``dips`` gives one dip each.
        * ``uniform_dip=`` -- one dip for the whole fault.

        ``uniform_dip`` is computed **analytically**, not fitted: every node
        simply steps ``depth / tan(dip)`` along its own local normal. The
        reference runs its gridder even in this case, but there is nothing for a
        gridder to decide when the answer is known in closed form, and going
        through it would leave ``curved(uniform_dip=90)`` agreeing with
        :meth:`vertical` only to the fit's tolerance instead of exactly.

        ``bias_w`` thickens the depth levels geometrically downward (the
        reference's ``cfg.mesh.biasW = 1.15``), putting the fine resolution where
        the data resolves slip. ``bias_w=1`` gives even levels. Measured on the
        real trace at ``down_dip_levels=8``: ``bias_w=1.15`` runs 1.8 km levels at
        the surface to 4.2 km at 20 km depth, and ``1.3`` runs 1.1 km to 5.5 km.

        Three knobs place the nodes freely, in increasing generality; all three
        keep the geometry arguments above untouched, because how a node is
        carried onto the fitted surface does not depend on where it sits.

        * ``depths=`` -- the depth levels outright, any spacing, in place of the
          ``down_dip_levels``/``bias_w`` geometric progression.
        * ``nodes_per_level=`` (a scalar, one value per level, or a callable of
          depth) or ``along_spacing=`` (the same thing in metres) -- **a
          different number of nodes at each depth level**. Rows of unequal count
          have no quads to split, so those meshes are triangulated by Delaunay
          in ``(s, z)`` and wound by
          :func:`~nisar_tools.slip.layout.orient_triangles`; a uniform count
          still takes the classic lattice split, bit for bit.
        * ``layout=`` -- a :class:`~nisar_tools.slip.layout.MeshLayout` you built
          yourself, including a fully unstructured one from
          :func:`~nisar_tools.slip.layout.lloyd_layout`. The depth extent then
          comes from the layout rather than from ``max_depth``.

        ``balance`` caps how fast the node count may change between adjacent
        levels (2:1 by default). It is a shape guard, not a preference: a fine
        row against a coarse one is what makes Delaunay reach across for a long
        thin triangle.

        ``bias_l`` is accepted only as ``1``. It is the reference's along-strike
        counterpart -- a single geometric ratio applied along strike -- which is
        not what the knobs above give you, and it has no meaning once the levels
        may hold different node counts anyway.
        """
        frame = frame or trace.local_frame()
        if layout is None and max_depth <= top_depth:
            raise ValueError("max_depth must be deeper than top_depth")
        if bias_l != 1.0:
            raise ValueError(
                f"bias_l={bias_l} is not supported. It is the reference's single "
                "geometric ratio applied along strike; to vary resolution along "
                "strike or per depth level, use nodes_per_level= / along_spacing= "
                "(which take a sequence or a callable of depth), depths= for the "
                "levels themselves, or layout= for an arbitrary node set. The "
                "reference's own demo configuration sets biasL = 1."
            )
        if bottom_depth is not None and bottom_trace is None:
            raise ValueError(
                "bottom_depth= only means anything with bottom_trace=; it is the "
                "depth the digitised bottom edge sits at. Use max_depth to set how "
                "deep the mesh goes."
            )
        unknown = set(fit_kwargs) - _FIT_KWARGS
        if unknown:
            raise TypeError(
                f"Unexpected keyword argument(s) {sorted(unknown)} for "
                f"FaultMesh.curved; surface-fit options are {sorted(_FIT_KWARGS)}"
            )

        resampled = trace.resample(edge_length, frame)
        tx, ty = resampled.to_local(frame)
        s = _arc_length(tx, ty)

        layout, depths = _resolve_layout(
            layout, depths, nodes_per_level, along_spacing, s,
            top_depth, max_depth, down_dip_levels, bias_w, edge_length, balance)
        top_depth = float(-layout.params[:, 1].max())
        max_depth = float(-layout.params[:, 1].min())

        # Everything curvilinear is measured against the *resampled* trace, not
        # the digitised one: `s` came from its vertices, so it is the polyline
        # that defines what arc length means here. Using the original would
        # reintroduce the lon/lat round trip `resample` goes through, and a
        # nominally vertical fault would land tens of microns off its own trace.
        surface, kind = _resolve_surface(
            resampled, frame, s, depths, surface, segments, dips, uniform_dip,
            bottom_trace, bottom_depth, smoothness, depth_control, fit_kwargs,
        )

        # `cross_nodes` is indexed by ascending z (deepest row first); the mesh
        # wants shallowest first, so the depth axis is reversed once, here.
        cross = _snap_vertical_columns(surface.cross_nodes[::-1], depths)
        nodes = _place_layout(layout, resampled, frame, s, depths, cross, tx, ty)
        params = layout.params

        centres = centre_params = None
        # The fitted cell centres are the *fit lattice's*, so they are the mesh's
        # own cells only while the mesh is that lattice. Nothing in the package
        # reads them, so an irregular layout simply leaves them out rather than
        # shipping a grid that does not describe it.
        if (surface.cross_centres is not None
                and layout.uniform_counts == (s.size, depths.size)):
            cx, cy, cz, cs = surface.centre_points(resampled, frame)
            centres = np.column_stack([cx, cy, cz])
            centre_params = np.column_stack([cs, cz])

        attrs = {
            "kind": kind,
            "max_depth": float(max_depth),
            "top_depth": float(top_depth),
            "edge_length": float(edge_length),
            "trace_length": float(s[-1]),
            "trace_name": trace.name,
            "bias_w": float(bias_w),
        }
        attrs.update(_layout_attrs(layout))
        attrs.update({k: v for k, v in surface.attrs.items() if k != "kind"})
        if uniform_dip is not None:
            attrs["dip_deg"] = float(uniform_dip)

        mesh = cls(nodes, triangles=layout.triangles, params=params,
                   frame=frame, attrs=attrs, centres=centres,
                   centre_params=centre_params)
        mesh._check_winding(trace)
        require_boundaries(params)
        return mesh

    @classmethod
    def from_layout(cls, layout, trace, frame=None, **kwargs):
        """Build a mesh on a node layout chosen elsewhere.

        The seam an optimizer works through: :mod:`nisar_tools.slip.layout`
        decides where the nodes go in ``(s, z)`` and this carries them onto the
        very surface :meth:`curved` fits, so geometry and discretisation can be
        varied independently. Every geometry keyword of :meth:`curved` applies;
        the depth extent comes from the layout.
        """
        return cls.curved(trace, frame, layout=layout, **kwargs)

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
    def param_vertices(self):
        """The three vertex ``(s, z)`` per element, shape ``(m, 3, 2)``.

        The unrolled counterpart of :attr:`vertices`: arc length along strike and
        depth, which is exactly what the mesh is parameterized by, so the section
        it draws is an exact picture of the fault rather than a projection of one.
        """
        return self.params[self.triangles]

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
        """Dip in degrees: 0 for a horizontal element, 90 for a vertical one.

        Folded into ``[0, 90]`` by the ``abs``, which is the convention the
        reference's ten-column model table uses -- so an element dipping 60
        degrees one way and one dipping 60 the other report the same number. Read
        :attr:`dip_direction` alongside it to tell them apart.
        """
        return np.degrees(np.arccos(np.clip(np.abs(self.normals[:, 2]), 0.0, 1.0)))

    @property
    def dip_direction(self):
        """Azimuth in degrees of the down-dip direction, clockwise from north.

        What :attr:`dip` throws away. On a vertical fault the down-dip vector is
        straight down and this is degenerate, so it falls back to the normal's
        own azimuth -- which for a vertical element is still meaningful (it is
        the side the fault faces) and is never used to place anything.
        """
        n = self.normals
        # Down-dip = the in-plane direction of steepest descent, i.e. the
        # component of -eZ left after removing the normal.
        d = np.column_stack([-n[:, 0] * -n[:, 2], -n[:, 1] * -n[:, 2],
                             -1.0 - n[:, 2] * -n[:, 2]])
        d = np.where(np.hypot(d[:, 0], d[:, 1])[:, None] > 1e-12, d,
                     np.column_stack([n[:, 0], n[:, 1], np.zeros(len(n))]))
        return np.degrees(np.arctan2(d[:, 0], d[:, 1])) % 360.0

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
        return self.boundary_nodes(side)[self.triangles].any(axis=1)

    def boundary_nodes(self, side):
        """Nodes on one edge of the fault, as a boolean mask over ``nodes``.

        The single implementation of "which nodes are on an edge"; the element
        mask above, ``NodeBasis.boundary`` and ``SlipModel.surface_displacement``
        all read it, so an irregular mesh cannot have three answers.
        """
        return boundary_node_mask(self.params, side)

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

    def has_centres(self):
        """Whether the cell-centre grid is present (needed by the layered engine)."""
        return self.centres is not None

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
                "dip_direction": ("element", self.dip_direction),
            }
        )
        if self.centres is not None:
            ds["centre_x"] = ("centre", self.centres[:, 0])
            ds["centre_y"] = ("centre", self.centres[:, 1])
            ds["centre_z"] = ("centre", self.centres[:, 2])
            ds["centre_s"] = ("centre", self.centre_params[:, 0])
        ds.attrs.update(self.attrs)
        ds.attrs["digest"] = self.digest()
        if self.frame is not None:
            ds.attrs["frame"] = self.frame.to_dict()
        return ds

    @classmethod
    def from_dataset(cls, ds):
        from .frame import LocalFrame

        nodes = np.column_stack([ds["node_x"].values, ds["node_y"].values, ds["node_z"].values])
        # params[:, 1] is rebuilt from node_z rather than persisted, which is
        # exact for every mesh this class builds: the fitted surface moves a node
        # in map view only, so its parameter depth *is* its Cartesian z.
        params = np.column_stack([ds["node_s"].values, ds["node_z"].values])
        frame = ds.attrs.get("frame")
        centres = centre_params = None
        if "centre_x" in ds:
            centres = np.column_stack([ds["centre_x"].values, ds["centre_y"].values,
                                       ds["centre_z"].values])
            centre_params = np.column_stack([ds["centre_s"].values, ds["centre_z"].values])
        return cls(
            nodes, ds["triangles"].values, params,
            frame=LocalFrame.from_dict(frame) if frame else None,
            attrs={k: v for k, v in ds.attrs.items() if k not in ("frame", "digest")},
            centres=centres, centre_params=centre_params,
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
        # The test only sees the horizontal projection, whose length is sin(dip):
        # 1.0 for a vertical element, 0.5 at 30 degrees, and exactly 0 for a
        # horizontal one -- where the sign of `dot` is decided by rounding noise
        # rather than by geometry. Refuse that case explicitly instead of
        # answering it at random.
        horizontal = np.hypot(n[:, 0], n[:, 1])
        flat = horizontal < 1e-6
        if np.any(flat):
            raise ValueError(
                f"{int(flat.sum())} of {self.n_elements} elements are within "
                "1e-4 degrees of horizontal, where winding cannot be checked "
                "against the trace's normal (the normal is vertical, so it has no "
                "side). A fault this flat needs a different reference direction; "
                f"offending elements: {np.nonzero(flat)[0][:10].tolist()}"
            )

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


#: Surface-fit options ``FaultMesh.curved`` forwards, so an unknown one is caught
#: where the caller typed it rather than deep inside the gridder -- or, on the
#: closed-form ``uniform_dip`` path, not at all.
_FIT_KWARGS = frozenset({
    "surface_weight_ratio", "samples_per_cell",
    "interp", "regularizer", "autoscale", "solver", "weights",
})


def _depth_levels(top_depth, max_depth, n_levels, bias_w=1.0):
    """Depth of each node level, thickening downward by ``bias_w`` per level.

    ``bias_w = 1`` gives even levels. Above 1 the shallow levels are thin, which
    is where slip is resolved: a surface observation sees a patch at 2 km far more
    sharply than one at 18 km, so uniform levels spend parameters where the data
    cannot constrain them. The reference builds the same geometric progression
    (``widthFactors = biasW .^ (0:layerCount-1)``) but solves for the level
    *count* from a target top thickness; taking the count as given is the same
    family of grids with the knob the caller actually wants to turn.
    """
    n_levels = max(2, int(n_levels))
    if bias_w <= 0:
        raise ValueError("bias_w must be positive")
    if bias_w == 1.0:
        return np.linspace(top_depth, max_depth, n_levels)
    thickness = bias_w ** np.arange(n_levels - 1, dtype=float)
    thickness *= (max_depth - top_depth) / thickness.sum()
    return top_depth + np.concatenate([[0.0], np.cumsum(thickness)])


def _snap_vertical_columns(cross, depths):
    """Pull near-vertical steps of the fitted surface onto exactly vertical.

    :data:`~nisar_tools.slip.trace.VERTICAL_TOL_DEG` guards the *input* dip, but a
    fitted surface that leans one way at one end and the other way at the other
    has to pass through vertical somewhere, and the elements it passes through
    are not asked for by anybody -- they are wherever the fit happens to cross.
    Those elements land in the triangular-dislocation solution's catastrophic
    near-vertical band, where the error reaches 190x the signal.

    A step from depth level ``j`` to ``j+1`` is in the band when the cross-strike
    change over it is under ``dz * tan(tol)`` -- half a metre over a 3 km level.
    Snapping copies the shallower value down, which makes that step exactly
    vertical and therefore exactly in the safe regime, at a cost bounded by the
    same half metre.
    """
    cross = np.array(cross, dtype=float)
    for j in range(cross.shape[0] - 1):
        dz = abs(depths[j + 1] - depths[j])
        near = np.abs(cross[j + 1] - cross[j]) < dz * np.tan(np.radians(VERTICAL_TOL_DEG))
        cross[j + 1] = np.where(near, cross[j], cross[j + 1])
    return cross


def _resolve_layout(layout, depths, nodes_per_level, along_spacing, stations,
                    top_depth, max_depth, down_dip_levels, bias_w, edge_length,
                    balance):
    """The ``(s, z)`` node layout, and the depth levels to fit the surface on.

    The two are the same list for a row-structured layout -- which is what keeps
    a node's parameter depth equal to its Cartesian ``z``, the invariant
    ``from_dataset`` and every boundary test rest on. A freeform cloud has no
    rows to fit, so the surface gets its own even lattice over the same depth
    range and the nodes are interpolated off it.
    """
    if layout is not None:
        clash = [name for name, value in (("depths", depths),
                                          ("nodes_per_level", nodes_per_level),
                                          ("along_spacing", along_spacing))
                 if value is not None]
        if clash:
            raise ValueError(
                f"layout= already places every node, so {clash} would be a "
                "second and contradicting answer"
            )
        if layout.levels is not None:
            return layout, np.asarray(layout.levels, dtype=float)
        depth = -layout.params[:, 1]
        n = (int(down_dip_levels) if down_dip_levels
             else max(2, int(round((depth.max() - depth.min()) / edge_length)) + 1))
        return layout, _depth_levels(depth.min(), depth.max(), n, 1.0)

    if depths is None:
        if down_dip_levels is None:
            down_dip_levels = max(1, int(round((max_depth - top_depth) / edge_length))) + 1
        depths = _depth_levels(top_depth, max_depth, int(down_dip_levels), bias_w)
    else:
        depths = np.asarray(depths, dtype=float).ravel()
        if bias_w != 1.0:
            raise ValueError(
                "depths= lists the levels outright, so bias_w -- which exists only "
                "to generate a geometric progression of them -- would be ignored. "
                "Drop one of the two."
            )
        if down_dip_levels is not None:
            raise ValueError(
                "depths= already fixes the level count; drop down_dip_levels="
            )

    if nodes_per_level is None and along_spacing is None:
        # The classic lattice: one node per resampled trace vertex, every level.
        nodes_per_level = stations.size

    return graded_layout(stations[-1], depths, counts=nodes_per_level,
                         along_spacing=along_spacing, stations=stations,
                         balance=balance), depths


def _layout_attrs(layout):
    """Provenance for a layout, keeping the lattice's own keys where they apply.

    ``n_along`` means nothing once levels differ, so an irregular layout records
    ``nodes_per_level`` instead; ``layout`` is folded in only when it is not the
    classic uniform one, so an existing mesh's attrs are unchanged.
    """
    kind = layout.attrs.get("layout", "freeform")
    out = {} if kind == "uniform" else {"layout": kind}
    if layout.counts is None:
        out["n_nodes"] = int(layout.n_nodes)
        return out
    out["n_down"] = int(layout.counts.size)
    uniform = layout.uniform_counts
    if uniform is not None:
        out["n_along"] = int(uniform[0])
    else:
        out["nodes_per_level"] = [int(v) for v in layout.counts]
    return out


def _sample_cross(cross, s_fit, depths, params):
    """Cross-strike offset at arbitrary ``(s, z)``, bilinear on the fit lattice.

    Exact on the lattice's own knots: ``searchsorted`` puts a node sitting on a
    knot at weight exactly 1, so a row-structured layout would get the same
    numbers as the per-level path in :func:`_place_layout`.
    """
    s = params[:, 0]
    depth = -params[:, 1]

    i = np.clip(np.searchsorted(s_fit, s) - 1, 0, s_fit.size - 2)
    ws = (s - s_fit[i]) / (s_fit[i + 1] - s_fit[i])
    j = np.clip(np.searchsorted(depths, depth) - 1, 0, depths.size - 2)
    wz = (depth - depths[j]) / (depths[j + 1] - depths[j])

    lower = cross[j, i] * (1.0 - ws) + cross[j, i + 1] * ws
    upper = cross[j + 1, i] * (1.0 - ws) + cross[j + 1, i + 1] * ws
    return lower * (1.0 - wz) + upper * wz


def _place_layout(layout, resampled, frame, s_fit, depths, cross, tx, ty):
    """Carry an ``(s, z)`` layout into 3-D through the fitted cross-strike field.

    A node's ``z`` is copied across untouched -- the surface moves nodes in map
    view only -- which is what keeps ``params[:, 1] == nodes[:, 2]`` exact, and
    with it every boundary test and ``from_dataset``'s reconstruction.
    """
    params = layout.params
    nodes = np.empty((layout.n_nodes, 3))
    nodes[:, 2] = params[:, 1]

    if layout.counts is not None:
        for j, sl in enumerate(layout.level_slices()):
            s_j = params[sl, 0]
            if np.all(cross[j] == 0.0):
                # Exactly on the trace. `np.interp` returns its knots bit for
                # bit, so a uniform layout writes the same coordinates
                # `vertical()` does rather than agreeing to round-off.
                nodes[sl, 0] = np.interp(s_j, s_fit, tx)
                nodes[sl, 1] = np.interp(s_j, s_fit, ty)
            else:
                nodes[sl, 0], nodes[sl, 1] = resampled.from_curvilinear(
                    s_j, np.interp(s_j, s_fit, cross[j]), frame)
        return nodes

    offset = _sample_cross(cross, s_fit, depths, params)
    flat = offset == 0.0
    if flat.any():
        nodes[flat, 0] = np.interp(params[flat, 0], s_fit, tx)
        nodes[flat, 1] = np.interp(params[flat, 0], s_fit, ty)
    if not flat.all():
        nodes[~flat, 0], nodes[~flat, 1] = resampled.from_curvilinear(
            params[~flat, 0], offset[~flat], frame)
    return nodes


def _resolve_surface(trace, frame, s, depths, surface, segments, dips, uniform_dip,
                     bottom_trace, bottom_depth, smoothness, depth_control, fit_kwargs):
    """Turn whichever geometry description the caller gave into a FaultSurface."""
    from .surface import DEFAULT_SMOOTHNESS, FaultSurface, centres as _centres

    z_nodes = -depths[::-1]
    given = [name for name, value in
             (("surface", surface), ("bottom_trace", bottom_trace),
              ("segments", segments), ("uniform_dip", uniform_dip))
             if value is not None]
    if len(given) != 1:
        raise ValueError(
            "FaultMesh.curved needs exactly one of surface=, bottom_trace=, "
            f"segments= or uniform_dip=; got {given or ['none']}"
        )

    if surface is not None:
        if not np.allclose(surface.s_nodes, s) or not np.allclose(surface.z_nodes, z_nodes):
            raise ValueError(
                "The supplied surface is on a different lattice than this mesh "
                f"({surface.cross_nodes.shape} vs {(depths.size, s.size)}); build it "
                "with the same edge_length, max_depth and down_dip_levels."
            )
        return surface, "curved"

    if bottom_trace is not None:
        built = FaultSurface.from_bottom_trace(
            trace, frame, bottom_trace, s_nodes=s, z_nodes=z_nodes,
            bottom_depth=bottom_depth, depth_control=depth_control,
            smoothness=DEFAULT_SMOOTHNESS if smoothness is None else smoothness,
            **fit_kwargs,
        )
        return built, "curved"

    if uniform_dip is not None:
        # Closed form: at depth d every node steps d/tan(dip) along its own
        # local normal, so the cross field is constant along strike.
        offsets = dip_offset(depths, float(uniform_dip))
        cross = np.broadcast_to(offsets[::-1, None], (depths.size, s.size))
        s_c, z_c = _centres(s), _centres(z_nodes)
        centre_cross = np.broadcast_to(
            dip_offset(-z_c, float(uniform_dip))[:, None], (z_c.size, s_c.size))
        built = FaultSurface(s, z_nodes, np.array(cross), s_c, z_c,
                             np.array(centre_cross),
                             attrs={"dips": [float(uniform_dip)]})
        return built, ("vertical" if np.all(offsets == 0.0) else "curved")

    if isinstance(segments, (int, np.integer)):
        segments = FaultSegment.from_trace(trace, frame, int(segments))
    if dips is None:
        raise ValueError("segments= also needs dips=, one dip in degrees per segment")
    built = FaultSurface.from_segments(
        trace, frame, segments, dips, s_nodes=s, z_nodes=z_nodes,
        depth_control=depth_control,
        smoothness=DEFAULT_SMOOTHNESS if smoothness is None else smoothness,
        **fit_kwargs,
    )
    return built, "curved"


def _arc_length(x, y):
    step = np.hypot(np.diff(x), np.diff(y))
    return np.concatenate([[0.0], np.cumsum(step)])
