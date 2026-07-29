"""How slip is parameterized over the mesh: one value per element, or per node.

An inversion does not solve for "the slip field"; it solves for the coefficients
of a slip field expanded in some basis. Two are available here and the choice
changes what a parameter *means*, so it propagates to the smoothing operator, the
bounds, the moment and every plot:

* :class:`ElementBasis` -- slip is constant on each triangle. Phase one's
  parameterization, and still the default, because the triangular-dislocation
  solution assumes exactly this: a patch of constant Burgers vector. One
  parameter per element per component, and a Green's-function column is one call.
* :class:`NodeBasis` -- slip is the piecewise-linear "tent" that is 1 at one node
  and 0 at every other, so the field is continuous across every edge. This is
  what the reference implementation's layered path uses, and it is the natural
  partner for a point-source engine, which is already integrating over each
  triangle and can weight the samples by the tent for free.

The tent basis is the better *model* -- a real slip distribution is continuous,
and a piecewise-constant one has a spurious stress singularity on every edge --
but it is not free: there are roughly half as many nodes as triangles, so it also
halves the parameter count, and it needs a smoothing operator defined on nodes
(:func:`nisar_tools.slip.regularize.laplace_beltrami`) rather than on element
adjacency.

Both remain reachable on purpose. The layered engine has to be checked against
the homogeneous one, and that comparison is only meaningful if both can be
assembled in the same basis; and every phase-one result stays comparable only
while the element basis keeps meaning what it meant.
"""

import numpy as np
import scipy.sparse as sp

#: Sub-triangle subdivisions per node basis function.
#:
#: The reference uses ``interpolate_triangle2(..., 15)``, a strictly interior
#: barycentric lattice of ``(15-1)(15-2)/2 = 91`` points, and
#: ``build_full_xyz.m`` hard-errors if asked for any other number. We subdivide
#: into ``13**2 = 169`` sub-triangles instead, which is the same order of work and
#: a *convergent* rule: an equal-weight sample of only the upright sub-triangles,
#: which is what a plain barycentric lattice gives, is a biased estimate of the
#: integral and its error stops falling however many points are added.
DEFAULT_SUBDIVISION = 13


class ElementBasis:
    """Slip constant on each triangle: one parameter per element per component."""

    name = "element"

    def __init__(self, mesh):
        self.mesh = mesh

    @property
    def n_basis(self):
        return self.mesh.n_elements

    @property
    def n_param(self):
        return 2 * self.n_basis

    def projection(self):
        """Identity: a coefficient already *is* its element's slip."""
        return sp.identity(self.n_basis, format="csr")

    def element_values(self, coefficients):
        """The field sampled per element -- the identity, for this basis."""
        return np.asarray(coefficients, dtype=float)

    def lumped_areas(self):
        """Area each parameter carries, for a moment sum."""
        return self.mesh.areas

    def boundary(self, sides):
        mask = np.zeros(self.n_basis, dtype=bool)
        for side in sides:
            mask |= self.mesh.boundary_elements(side)
        return mask


class NodeBasis:
    """Slip as a continuous piecewise-linear field: one parameter per node.

    Basis function ``i`` is the barycentric weight of node ``i``, which is 1 at
    that node, falls linearly to 0 along every edge of every incident triangle,
    and is 0 outside them. Its integral over one incident triangle is a third of
    that triangle's area, so the total area a node carries -- its lumped mass --
    is a third of its 1-ring.
    """

    name = "node"

    def __init__(self, mesh, subdivision=DEFAULT_SUBDIVISION):
        self.mesh = mesh
        self.subdivision = int(subdivision)
        self._incident = _incident_elements(mesh.triangles, mesh.n_nodes)

    @property
    def n_basis(self):
        return self.mesh.n_nodes

    @property
    def n_param(self):
        return 2 * self.n_basis

    def projection(self):
        """Each node's mean contribution to each element it touches: one third.

        This is how a nodal design matrix is assembled -- build the element one
        and multiply -- rather than by evaluating a tent-weighted point cloud per
        node. The two differ, and the difference is worth being explicit about.

        A tent function is linear across a triangle, and this replaces it with its
        *mean* over that triangle. The replacement is exact in every respect that
        the far field can see, because it preserves the integral: the tent's
        integral over one incident triangle is exactly ``A/3``, which is what a
        third of a constant slip gives. What it loses is the variation *within* a
        triangle, so a receiver closer to the fault than one element sees a
        slightly different field.

        The reason to take that trade is that the alternative costs a full
        dislocation evaluation per sub-triangle per node -- of order a thousand
        times the work -- for a difference confined to distances where
        ``exclude_within`` has already removed the observations. It also keeps the
        homogeneous and layered engines assembling identically, which is what lets
        one be checked against the other in either basis.
        """
        rows = np.repeat(np.arange(self.mesh.n_elements), 3)
        cols = self.mesh.triangles.ravel()
        return sp.csr_matrix(
            (np.full(rows.size, 1.0 / 3.0), (rows, cols)),
            shape=(self.mesh.n_elements, self.n_basis),
        )

    def tent_weights(self, index):
        """The exact tent sample cloud for one node: points, areas, elements.

        Not used to assemble a design matrix (see :meth:`projection` for why), but
        it is the definition the projection approximates, and the check that the
        weights sum to the node's lumped area is what pins that.
        """
        from .layered import _triangle_quadrature

        points, weights, owners = [], [], []
        for element in self._incident[index]:
            vertices = self.mesh.vertices[element]
            sub, share = _triangle_quadrature(*vertices, self.subdivision)
            tent = _barycentric_weight(vertices, self.mesh.triangles[element], index, sub)
            points.append(sub)
            weights.append(share * tent * self.mesh.areas[element])
            owners.append(np.full(sub.shape[0], element))
        return (np.concatenate(points), np.concatenate(weights),
                np.concatenate(owners))

    def element_values(self, coefficients):
        """The field averaged over each element -- the mean of its three nodes."""
        return np.asarray(coefficients, dtype=float)[self.mesh.triangles].mean(axis=1)

    def lumped_areas(self):
        """A third of each node's 1-ring: the P1 lumped-mass share."""
        areas = np.zeros(self.n_basis)
        np.add.at(areas, self.mesh.triangles.ravel(),
                  np.repeat(self.mesh.areas, 3) / 3.0)
        return areas

    def boundary(self, sides):
        """Nodes on the flagged edges, in parameter space."""
        s, z = self.mesh.params[:, 0], self.mesh.params[:, 1]
        mask = np.zeros(self.n_basis, dtype=bool)
        for side in sides:
            if side == "bottom":
                mask |= z == z.min()
            elif side == "top":
                mask |= z == z.max()
            elif side == "left":
                mask |= s == s.min()
            elif side == "right":
                mask |= s == s.max()
            else:
                raise ValueError(
                    f"Unknown boundary {side!r}; expected bottom, top, left or right")
        return mask


def make_basis(mesh, kind="element", **kwargs):
    """``"element"`` or ``"node"`` by name; an already-built basis passes through."""
    if hasattr(kind, "n_basis"):
        return kind
    if kind == "element":
        return ElementBasis(mesh)
    if kind == "node":
        return NodeBasis(mesh, **kwargs)
    raise ValueError(f"Unknown slip basis {kind!r}; expected 'element' or 'node'")


def _incident_elements(triangles, n_nodes):
    """For each node, the elements that contain it."""
    incident = [[] for _ in range(n_nodes)]
    for element, tri in enumerate(triangles):
        for node in tri:
            incident[node].append(element)
    return [np.array(v, dtype=int) for v in incident]


def _barycentric_weight(vertices, triangle, node, points):
    """The tent function of ``node`` sampled at ``points`` inside its triangle.

    A port of ``taperTriangle.m``: solve the barycentric coordinates in the
    triangle's own plane and return the one belonging to ``node``, clipped to
    ``[0, 1]`` so a sample a hair outside the triangle cannot contribute a
    negative area.
    """
    anchor = int(np.nonzero(triangle == node)[0][0])
    a = vertices[anchor]
    b, c = vertices[(anchor + 1) % 3], vertices[(anchor + 2) % 3]

    v0, v1 = b - a, c - a
    d00, d01, d11 = v0 @ v0, v0 @ v1, v1 @ v1
    denominator = d00 * d11 - d01 * d01
    v2 = points - a
    d20, d21 = v2 @ v0, v2 @ v1
    u = (d11 * d20 - d01 * d21) / denominator
    v = (d00 * d21 - d01 * d20) / denominator
    return np.clip(1.0 - u - v, 0.0, 1.0)
