"""Green's functions for a layered crust, by superposing tabulated point sources.

The half-space engine in :mod:`nisar_tools.slip.greens` evaluates a closed form
for each triangular element. A layered medium has no closed form, so this engine
does what EDCMP does instead: cut each element into point sources, look each one
up in the precomputed :class:`~nisar_tools.slip.edgrn.EdgrnTables`, put the
azimuthal dependence back, and add them up.

Why bother: a homogeneous half-space gives the whole crust one rigidity, and the
shallowest few kilometres are much softer than that. Assuming they are not makes
the same surface displacement require less shallow slip and more deep slip, which
is a systematic bias in exactly the quantity a coseismic inversion is for.

Two departures from the reference, both deliberate:

* **Evaluate at the observations, not on a grid.** SlipSolve precomputes onto a
  ~190 000-point adaptive grid -- 4.5 GB across its three component matrices --
  and interpolates from there onto its few thousand observations. Going straight
  to the observations is less work and has no interpolation error at all.
* **Choose the quadrature order per element and observation.** The reference uses
  a fixed 91 points per triangle everywhere. A point source is a good
  approximation to a patch once you are a few patch-widths away, and almost every
  observation is far from almost every element, so a fixed order spends nearly
  all of its effort where one point would have done. Measured on the real problem
  (1148 elements, 8000 observations, r/L median 38): 89x fewer source-receiver
  evaluations at 1e-2 accuracy, 52x at 3e-3, 10x at 1e-3. Set
  ``tolerance=None`` to force the reference's fixed order.

The moment tensor of an element is built from the element's **own** orthonormal
basis rather than from a strike/dip/rake triple. ``edcdisc.m`` goes through
angles because its input is a rectangular patch described that way; ours is a
triangle that already carries ``Vnorm``, ``Vstrike`` and ``Vdip``, and the
package's ``strike`` is deliberately the azimuth of ``cross(eZ, Vnorm)`` rather
than the geological strike -- for an east-striking fault the two differ by 180
degrees. Converting to angles and back would put that ambiguity in the middle of
the one calculation whose sign nothing downstream can check.
"""

import numpy as np

from .greens import _as_points, _require_finite

#: Relative accuracy the adaptive quadrature aims for, per element.
#:
#: The single-point (centroid) error is about ``0.06 * L / r`` for an element of
#: size ``L`` seen from distance ``r``, and an order-``k`` rule divides that by
#: roughly ``k``. 3e-3 is comfortably below the noise floor of a real
#: interferogram -- measured at 11.8 mm within-cell scatter on the Venezuela
#: scenes, against slip of order a metre -- while costing under two points per
#: source-receiver pair.
DEFAULT_TOLERANCE = 3e-3

#: Quadrature order cap. 91 points is the reference's fixed choice
#: (``interpolate_triangle2`` with ``n = 15``), kept as the ceiling so the
#: adaptive rule can never do more work than the thing it replaces.
MAX_QUADRATURE_POINTS = 91

#: Empirical constant in the error model above, measured against exact
#: triangular-dislocation solutions over ``r/L`` from 0.5 to 20.
_QUADRATURE_CONSTANT = 0.06


class LayeredPointSource:
    """Point-source superposition through EDGRN tables; a layered engine.

    Satisfies the same protocol as
    :class:`~nisar_tools.slip.greens.HalfSpaceTDE` -- ``los_matrix``,
    ``displacement``, ``forward``, ``name``, ``nu`` -- so
    :class:`~nisar_tools.slip.inversion.SlipInversion` takes it with
    ``engine=`` and nothing else changes.
    """

    name = "layered_point_source"

    def __init__(self, tables, nu=0.25, tolerance=DEFAULT_TOLERANCE,
                 max_points=MAX_QUADRATURE_POINTS):
        """``tables`` carries the elastic structure; ``nu`` does **not**.

        Poisson's ratio is recorded for provenance and to satisfy the engine
        protocol, and is deliberately not used in any calculation here: EDGRN
        already integrated through the layer stack, so every elastic property the
        displacements depend on -- including how Poisson's ratio varies with depth
        -- is baked into the tables. Passing a different ``nu`` changes nothing.
        That is the opposite of :class:`~nisar_tools.slip.greens.HalfSpaceTDE`,
        where ``nu`` is the only material parameter there is.
        """
        self.tables = tables
        self.nu = float(nu)
        self.tolerance = None if tolerance is None else float(tolerance)
        self.max_points = int(max_points)

    # -- the engine protocol ----------------------------------------------
    #: This engine can integrate a basis function directly, so
    #: :class:`~nisar_tools.slip.inversion.SlipInversion` hands it the basis
    #: instead of projecting an element-wise matrix afterwards.
    supports_basis = True

    def displacement(self, mesh, x, y, z=None, basis=None):
        """ENU displacement per unit slip, shape ``(npts, 3, 2 * n_basis)``."""
        x, y, z = _as_points(x, y, z)
        n = _n_basis(mesh, basis)
        out = np.zeros((x.size, 3, 2 * n))
        for k, comp, enu in self._columns(mesh, x, y, z, basis):
            out[:, :, k + comp * n] = enu
        _require_finite(out, mesh, x, y, z)
        return out

    def los_matrix(self, mesh, x, y, look_e, look_n, look_u, z=None, basis=None):
        """Line-of-sight design matrix, shape ``(npts, 2 * n_basis)``."""
        x, y, z = _as_points(x, y, z)
        look = np.stack([np.asarray(look_e, dtype=float).ravel(),
                         np.asarray(look_n, dtype=float).ravel(),
                         np.asarray(look_u, dtype=float).ravel()], axis=1)
        if look.shape[0] != x.size:
            raise ValueError("Look-vector components must match the observation count")

        n = _n_basis(mesh, basis)
        g = np.zeros((x.size, 2 * n))
        for k, comp, enu in self._columns(mesh, x, y, z, basis):
            g[:, k + comp * n] = (enu * look).sum(axis=1)
        _require_finite(g, mesh, x, y, z)
        return g

    def forward(self, mesh, slip, x, y, look=None, z=None, basis=None):
        slip = np.asarray(slip, dtype=float)
        if slip.ndim == 2:
            slip = np.concatenate([slip[:, 0], slip[:, 1]])
        expected = 2 * _n_basis(mesh, basis)
        if slip.size != expected:
            raise ValueError(f"slip has {slip.size} values; expected {expected}")
        if look is None:
            return self.displacement(mesh, x, y, z, basis=basis) @ slip
        return self.los_matrix(mesh, x, y, *look, z=z, basis=basis) @ slip

    # -- assembly ----------------------------------------------------------
    def _columns(self, mesh, x, y, z, basis=None):
        """Yield ``(index, component, (npts, 3) ENU)`` for every column.

        With no basis, or an element basis, one column per element with slip
        constant over it. With a **nodal** basis, one column per node with slip
        varying *linearly* from 1 at that node to 0 at every neighbour: each
        quadrature point in each incident triangle is weighted by the tent
        function there, so the integral is of the basis function itself rather
        than of its average.

        That is the difference between this and the projection
        :meth:`~nisar_tools.slip.basis.NodeBasis.projection` applies to an
        element-wise matrix. Both preserve the moment; only this one gets the
        near-field shape right, because a tent is not constant across a triangle.
        It costs about **three times** an element assembly -- each triangle is
        visited once per vertex -- not the thousandfold a fixed 91-point rule per
        node would.
        """
        if basis is None or basis.name == "element":
            supports = [(k, [k]) for k in range(mesh.n_elements)]
            tented = False
        elif basis.name == "node":
            supports = [(i, basis._incident[i]) for i in range(mesh.n_nodes)]
            tented = True
        else:
            raise ValueError(f"This engine cannot assemble the {basis.name!r} basis")

        vertices, areas = mesh.vertices, mesh.areas
        centroids, normals = mesh.centroids, mesh.normals

        for index, elements in supports:
            accumulated = [np.zeros((x.size, 3)), np.zeros((x.size, 3))]
            for element in elements:
                p1, p2, p3 = vertices[element]
                normal = normals[element]
                size = np.sqrt(2.0 * areas[element])
                orders = self._orders(centroids[element], size, x, y, z)

                for comp, direction in enumerate(_slip_directions(normal)):
                    moment = _moment_components(normal, direction)
                    for order in np.unique(orders):
                        at = orders == order
                        points, share = _triangle_quadrature(p1, p2, p3, int(order))
                        weights = share * areas[element]
                        if tented:
                            weights = weights * _tent(
                                vertices[element], mesh.triangles[element], index, points)
                        accumulated[comp][at] += self._sum_sources(
                            points, weights, moment, x[at], y[at])
            for comp in (0, 1):
                yield index, comp, accumulated[comp]

    def _orders(self, centroid, size, x, y, z):
        """Quadrature order for each observation, from how far away it is.

        Solving the measured error model ``0.06 * L / (k * r) = tolerance`` for
        ``k``, then rounding up and clamping. Note the order is per *observation*,
        not per element: the point of the exercise is that one element is near a
        handful of observations and far from thousands.
        """
        if self.tolerance is None:
            return np.full(x.size, int(np.sqrt(self.max_points)))
        r = np.sqrt((x - centroid[0]) ** 2 + (y - centroid[1]) ** 2
                    + (z - centroid[2]) ** 2)
        r = np.maximum(r, size / 100.0)
        wanted = np.ceil(_QUADRATURE_CONSTANT * size / (self.tolerance * r))
        return np.clip(wanted, 1, np.sqrt(self.max_points)).astype(int)

    def _sum_sources(self, points, weights, moment, x, y):
        """Superpose every point source of one element at every observation.

        ``points`` is ``(nq, 3)`` in ENU, ``weights`` the area each carries in
        square metres, ``moment`` the element's five packed tensor components.
        Broadcasting is ``(nobs, nq)`` throughout and collapsed at the end, so the
        cost is one table lookup per source-receiver pair -- which is what the
        adaptive order exists to keep small.
        """
        m1, m2, m3, m4, m5 = moment
        # EDGRN works in north/east/down. The observations are on the surface.
        src_n, src_e, src_d = points[:, 1], points[:, 0], -points[:, 2]
        dn = y[:, None] - src_n[None, :]
        de = x[:, None] - src_e[None, :]

        distance = np.hypot(dn, de)
        azimuth = np.arctan2(de, dn)
        depth = np.broadcast_to(src_d[None, :], distance.shape)

        cos1, sin1 = np.cos(azimuth), np.sin(azimuth)
        cos2, sin2 = np.cos(2.0 * azimuth), np.sin(2.0 * azimuth)

        # The azimuthal factors, exactly as edcgrn.f pairs them: order 2 for the
        # ss table, order 1 for ds, order 0 for the CLVD; and the transverse
        # component takes the quadrature partner of the radial one.
        ps_ss = m1 * sin2 + m4 * cos2
        sh_ss = m1 * cos2 - m4 * sin2
        ps_ds = m2 * cos1 + m5 * sin1
        sh_ds = m2 * sin1 - m5 * cos1

        look = self.tables.interpolate
        uz = (ps_ss * look("ss", "uz", distance, depth)
              + ps_ds * look("ds", "uz", distance, depth)
              + m3 * look("cl", "uz", distance, depth))
        ur = (ps_ss * look("ss", "ur", distance, depth)
              + ps_ds * look("ds", "ur", distance, depth)
              + m3 * look("cl", "ur", distance, depth))
        ut = (sh_ss * look("ss", "ut", distance, depth)
              + sh_ds * look("ds", "ut", distance, depth))

        # Cylindrical to north/east/down, then to the package's east/north/up.
        north = ur * cos1 - ut * sin1
        east = ur * sin1 + ut * cos1
        return np.stack([(east * weights).sum(axis=1),
                         (north * weights).sum(axis=1),
                         (-uz * weights).sum(axis=1)], axis=1)


def _n_basis(mesh, basis):
    return mesh.n_elements if basis is None else basis.n_basis


def _tent(vertices, triangle, node, points):
    """The nodal tent function of ``node``, sampled inside its own triangle."""
    from .basis import _barycentric_weight

    return _barycentric_weight(vertices, triangle, node, points)


def _slip_directions(normal):
    """The element's strike and dip unit vectors, in the order the columns use.

    The same ``Vstrike = cross(eZ, Vnorm)``, ``Vdip = cross(Vnorm, Vstrike)`` the
    triangular-dislocation solution uses, so column ``k`` of a layered design
    matrix means precisely what column ``k`` of a half-space one means and the two
    engines are interchangeable.
    """
    # The same expression `_tde._element_basis` uses, reproduced from the normal
    # alone because there are no vertices to hand here.
    e_z = np.array([0.0, 0.0, 1.0])
    vstrike = np.cross(e_z, normal)
    if np.linalg.norm(vstrike) == 0:
        vstrike = np.array([0.0, 1.0, 0.0]) * normal[2]
    vstrike = vstrike / np.linalg.norm(vstrike)
    return vstrike, np.cross(normal, vstrike)


def _moment_components(normal, slip):
    """Pack a unit dislocation's moment tensor into EDCMP's five components.

    A dislocation of area ``A`` and slip ``s`` on a plane with unit normal ``n``
    in the unit direction ``d`` has moment tensor ``mu*A*s*(n(x)d + d(x)n)``. The
    tables already carry the ``mu``, so what is needed here is the dimensionless
    tensor, converted to north/east/down and projected onto the basis EDCMP's
    three source types span::

        m1 = M_ne      m2 = M_nd      m3 = M_dd
        m4 = (M_nn - M_ee) / 2        m5 = M_ed

    Five components, not six, because a shear dislocation's tensor is traceless
    -- ``n`` and ``d`` are perpendicular -- so ``M_nn`` and ``M_ee`` are recovered
    from ``m3`` and ``m4`` together.
    """
    n = np.asarray(normal, dtype=float)
    d = np.asarray(slip, dtype=float)
    tensor = np.outer(n, d) + np.outer(d, n)

    # east/north/up -> north/east/down.
    swap = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    m = swap @ tensor @ swap.T
    return (m[0, 1], m[0, 2], m[2, 2], 0.5 * (m[0, 0] - m[1, 1]), m[1, 2])


def _triangle_quadrature(p1, p2, p3, order):
    """Equal-weight point sources on a triangle: ``order**2`` sub-triangle centroids.

    Subdividing each edge into ``order`` gives ``order**2`` congruent
    sub-triangles -- ``order*(order+1)/2`` pointing the same way as the parent and
    ``order*(order-1)/2`` inverted -- and their centroids with equal weights is a
    convergent rule. Taking only the upright ones, which a barycentric lattice of
    the reference's shape does, is **not**: it is a biased sample of the triangle
    and its error stops falling however many points are added. Measured, the
    upright-only rule plateaus around 5e-3 relative while this one keeps
    converging.
    """
    p1, p2, p3 = (np.asarray(p, dtype=float) for p in (p1, p2, p3))
    order = max(1, int(order))
    if order == 1:
        return ((p1 + p2 + p3) / 3.0)[None, :], np.array([1.0])

    points = []
    for a in range(order):
        for b in range(order - a):
            c = order - 1 - a - b
            points.append(((a + 1 / 3) * p1 + (b + 1 / 3) * p2 + (c + 1 / 3) * p3) / order)
            if c >= 1:
                points.append(((a + 2 / 3) * p1 + (b + 2 / 3) * p2
                               + (c - 1 / 3) * p3) / order)
    points = np.array(points)
    return points, np.full(points.shape[0], 1.0 / points.shape[0])
