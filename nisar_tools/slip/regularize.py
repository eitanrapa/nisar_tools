"""Regularization: what makes an under-determined slip inversion have an answer.

Surface displacement constrains slip on a fault about as well as a shadow
constrains a shape. Deep elements barely register at the surface, adjacent ones
produce nearly the same signal, and there are typically more parameters than
independent observations -- so the least-squares problem is rank-deficient and
its unregularized solution is a wild oscillation that fits the noise.

Three additions make it well posed, and each is appended as extra rows to the
system rather than solved separately:

* **Smoothing** penalises the difference in slip between neighbouring elements.
* **Zero-slip boundaries** stop slip running off the edges of a mesh that is only
  a piece of a longer fault.
* **Bounds** cap the magnitude and, usefully, fix the *sense* of slip -- pinning
  a known right-lateral fault to non-positive strike-slip removes a large part of
  the null space for free.

Parameters are ordered as :mod:`nisar_tools.slip.greens` builds them: strike-slip
for every element, then dip-slip for every element.
"""

import numpy as np
import scipy.sparse as sp


def neighbor_smoothing(neighbors, ss_ratio=1.0, ds_ratio=1.0):
    """Rows penalising the slip difference across each shared element edge.

    ``neighbors`` is the ``(m, 3)`` adjacency from
    :attr:`~nisar_tools.slip.mesh.FaultMesh.neighbors`, with ``-1`` for a
    boundary edge. Returns a sparse ``(2 * n_edges, 2 * m)`` matrix: one row per
    undirected edge in each of the strike and dip blocks.

    ``ss_ratio`` and ``ds_ratio`` scale the two blocks independently. A larger
    ``ds_ratio`` smooths dip-slip harder than strike-slip, which is the usual
    choice on a strike-slip fault where dip-slip is small and poorly resolved --
    the reference implementation defaults it to 3.

    **One row per edge, not two.** The reference emits a row for each ordered
    ``(element, neighbour)`` pair, so every interior edge appears twice; that is
    harmless algebraically but doubles the row count and rescales the effective
    smoothing weight by ``sqrt(2)``. Worth knowing when comparing a weight
    against a run of the MATLAB code.
    """
    neighbors = np.asarray(neighbors)
    n_elements = neighbors.shape[0]

    edges = set()
    for i in range(n_elements):
        for j in neighbors[i]:
            if j >= 0:
                edges.add((min(i, int(j)), max(i, int(j))))
    edges = sorted(edges)
    if not edges:
        return sp.csr_matrix((0, 2 * n_elements))

    rows, cols, vals = [], [], []
    for r, (i, j) in enumerate(edges):
        for block, ratio in ((0, ss_ratio), (1, ds_ratio)):
            row = r + block * len(edges)
            offset = block * n_elements
            rows += [row, row]
            cols += [i + offset, j + offset]
            vals += [ratio, -ratio]

    return sp.csr_matrix(
        (vals, (rows, cols)), shape=(2 * len(edges), 2 * n_elements)
    )


def laplace_beltrami(mesh, ss_ratio=1.0, ds_ratio=1.0, form="curvature",
                     mass="voronoi"):
    """Smoothing for a **nodal** slip field: a Laplacian on the fault surface.

    :func:`neighbor_smoothing` differences neighbouring *elements*, which is the
    right operator when slip is constant on each. A tent-function field lives on
    the nodes, and the natural penalty there is the surface Laplacian -- the
    discrete Laplace-Beltrami operator, which is what
    ``smoothingMatrix_laplace2.m`` builds.

    The edge weights are **hybrid**, and deliberately so. The cotangent weight is
    the standard finite-element one and is exact for a well-shaped triangle, but
    it goes *negative* on an obtuse one, which turns the smoother into an
    anti-smoother on part of the mesh. So a cotangent weight is used only when
    both triangles across an edge exist and both opposite angles are acute;
    otherwise the symmetric mean-value (Floater) weight, which is positive by
    construction, takes over.

    ``form="curvature"`` gives ``ell0**2 * M**(-1/2) * (D - W)`` with ``ell0`` the
    median edge length -- penalising curvature, and scaled to be dimensionless so
    a smoothing weight means the same thing on a coarse mesh as a fine one.
    ``form="dirichlet"`` penalises neighbour differences instead. The reference's
    spectral normalisation is left out, as it is there too (commented out).
    """
    nodes = mesh.nodes
    n = mesh.n_nodes
    rings = _one_rings(mesh.triangles, n)

    rows, cols, vals = [], [], []
    for i, neighbours in enumerate(rings):
        for j, opposite in neighbours.items():
            if j < i:
                continue
            weight = _edge_weight(nodes, i, j, opposite, rings)
            for a, b in ((i, j), (j, i)):
                rows.append(a)
                cols.append(b)
                vals.append(weight)
    weights = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))

    degree = np.asarray(weights.sum(axis=1)).ravel()
    laplacian = sp.diags(degree) - weights

    if form == "curvature":
        scale = sp.diags(1.0 / np.sqrt(np.maximum(_mixed_mass(mesh, mass), 1e-15)))
        power = 2
    elif form == "dirichlet":
        scale = sp.diags(1.0 / np.sqrt(np.maximum(degree, 1e-15)))
        power = 1
    else:
        raise ValueError(f"Unknown form {form!r}; expected curvature or dirichlet")

    operator = scale @ laplacian

    # Normalise to the same scale as `neighbor_smoothing`, whose rows are a plain
    # `[+1, -1]` difference and so have an absolute row sum of 2.
    #
    # The reference instead multiplies by the median edge length to the power one
    # or two, aiming at the same thing -- a weight that means the same on a coarse
    # mesh as a fine one. That prefactor does not get there: with metre units,
    # ``ell0**2 * M**(-1/2)`` is of order the edge length, so on a 6 km mesh the
    # operator comes out about **1e4** times stronger than the element smoother
    # and a smoothing weight of 0.3 drives every parameter to zero (measured:
    # variance reduction 0.2%, moment 0.2% of the truth). Normalising by the
    # operator's own norm reaches the intended invariance directly, and keeps one
    # smoothing weight comparable across both parameterizations.
    row_sum = np.abs(operator).sum(axis=1).max()
    if row_sum > 0:
        operator = operator * (2.0 / float(row_sum))
    _ = power  # the reference's edge-length prefactor, cancelled by the above

    zero = sp.csr_matrix((n, n))
    return sp.bmat([[ss_ratio * operator, zero], [zero, ds_ratio * operator]],
                   format="csr")


def _one_rings(triangles, n_nodes):
    """For each node, its neighbours mapped to the third vertices across each edge."""
    rings = [{} for _ in range(n_nodes)]
    for tri in triangles:
        for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            i, j, k = int(tri[a]), int(tri[b]), int(tri[c])
            rings[i].setdefault(j, []).append(k)
            rings[j].setdefault(i, []).append(k)
    return rings


def _edge_weight(nodes, i, j, opposite, rings):
    """Cotangent where it is safe, mean-value where it is not."""
    third = sorted(set(opposite))
    if len(third) == 2:
        angles = [_angle_at(nodes, i, j, k) for k in third]
        if all(0.0 < a < np.pi / 2 for a in angles):
            weight = 0.5 * sum(1.0 / np.tan(a) for a in angles)
            if np.isfinite(weight) and weight > 0:
                return weight
    return max(0.0, 0.5 * (_mean_value_weight(nodes, i, j, rings)
                           + _mean_value_weight(nodes, j, i, rings)))


def _angle_at(nodes, i, j, k):
    """The angle at ``k`` in the triangle ``(i, j, k)``."""
    u, v = nodes[i] - nodes[k], nodes[j] - nodes[k]
    norms = np.linalg.norm(u) * np.linalg.norm(v)
    if norms == 0:
        return np.nan
    return float(np.arccos(np.clip(u @ v / norms, -1.0, 1.0)))


def _mean_value_weight(nodes, i, j, rings):
    """Floater's one-sided weight: ``(tan(a/2) + tan(b/2)) / |ij|``.

    Positive for any triangle, which is exactly why it is the fallback -- the
    cotangent weight is negative on an obtuse pair and would smooth the wrong way.
    """
    neighbours = rings[i].get(j, [])
    if not neighbours:
        return 0.0
    distance = max(float(np.linalg.norm(nodes[j] - nodes[i])), 1e-15)
    half = sum(np.tan(min(_angle_at(nodes, j, k, i), np.pi - 1e-9) / 2.0)
               for k in sorted(set(neighbours)))
    return float(half / distance) if np.isfinite(half) else 0.0


def _mixed_mass(mesh, kind="voronoi"):
    """Area attributed to each node: mixed Voronoi, or barycentric.

    Meyer et al.'s mixed area -- the Voronoi cell where the triangle is acute, and
    a half or a quarter of it where it is obtuse, because a Voronoi cell of an
    obtuse triangle falls outside the triangle.
    """
    mass = np.zeros(mesh.n_nodes)
    nodes = mesh.nodes
    for tri, area in zip(mesh.triangles, mesh.areas):
        if kind == "barycentric":
            np.add.at(mass, tri, area / 3.0)
            continue
        angles = [_angle_at(nodes, tri[(a + 1) % 3], tri[(a + 2) % 3], tri[a])
                  for a in range(3)]
        obtuse = max(angles) > np.pi / 2 + 1e-14
        for a in range(3):
            i = tri[a]
            if obtuse:
                mass[i] += 0.5 * area if angles[a] > np.pi / 2 else 0.25 * area
            else:
                b = np.linalg.norm(nodes[tri[(a + 2) % 3]] - nodes[i])
                c = np.linalg.norm(nodes[tri[(a + 1) % 3]] - nodes[i])
                mass[i] += 0.125 * (b ** 2 / np.tan(angles[(a + 1) % 3])
                                    + c ** 2 / np.tan(angles[(a + 2) % 3]))
    return np.maximum(mass, 1e-15)


def zero_slip_boundary(mesh, sides=("bottom", "left", "right"), ratio=1.0):
    """Rows pulling slip toward zero on the named edges of the mesh.

    A mesh is a finite piece of a fault that continues past it, so without this
    the inversion happily puts large slip on the bottom and end elements, where
    the data constrain it least. ``sides`` names any of ``"bottom"``, ``"top"``,
    ``"left"``, ``"right"``; the free surface (``"top"``) is normally left
    unconstrained, since surface rupture is a real and interesting outcome.

    Returns a sparse ``(k, 2 * m)`` selection matrix with ``ratio`` on the
    flagged elements' strike and dip parameters -- only the non-zero rows. The
    reference builds a dense ``2m x 2m`` diagonal per side and stacks three of
    them, which appends ``6m`` rows of which almost all are zero.
    """
    flagged = np.zeros(mesh.n_elements, dtype=bool)
    for side in sides:
        flagged |= mesh.boundary_elements(side)

    idx = np.nonzero(flagged)[0]
    if idx.size == 0:
        return sp.csr_matrix((0, 2 * mesh.n_elements))

    cols = np.concatenate([idx, idx + mesh.n_elements])
    rows = np.arange(cols.size)
    return sp.csr_matrix(
        (np.full(cols.size, float(ratio)), (rows, cols)),
        shape=(cols.size, 2 * mesh.n_elements),
    )


def slip_bounds(n_elements, strike=(-10.0, 10.0), dip=(-10.0, 10.0), polarity=None):
    """Lower and upper bounds on the parameter vector, in metres.

    ``polarity`` optionally forces the *sense* of slip as
    ``(strike, dip)`` flags: ``+1`` restricts that component to be
    non-negative, ``-1`` to be non-positive, ``0`` or ``None`` leaves it free.
    A third element is accepted and ignored, so the reference's
    ``(strike, dip, tensile)`` triples can be passed unchanged.

    Because positive strike-slip is left-lateral (see :mod:`nisar_tools.slip`), a
    known **right-lateral** fault wants ``polarity=(-1, 0)``. This is the
    cheapest real constraint available: it halves the search space and removes
    the physically absurd alternating-sense solutions that smoothing alone still
    permits.
    """
    lo = np.concatenate([np.full(n_elements, float(strike[0])),
                        np.full(n_elements, float(dip[0]))])
    hi = np.concatenate([np.full(n_elements, float(strike[1])),
                        np.full(n_elements, float(dip[1]))])

    if polarity is not None:
        for block, flag in enumerate(list(polarity)[:2]):
            if not flag:
                continue
            sl = slice(block * n_elements, (block + 1) * n_elements)
            if flag > 0:
                lo[sl] = np.maximum(lo[sl], 0.0)
            else:
                hi[sl] = np.minimum(hi[sl], 0.0)

    if np.any(lo > hi):
        raise ValueError(
            "Bounds are empty: a polarity flag contradicts the strike/dip range."
        )
    return lo, hi


def ramp_columns(obs, kind="linear"):
    """Extra unbounded columns absorbing a per-track offset or ramp.

    An unwrapped interferogram has an arbitrary additive constant, and usually a
    long-wavelength orbital or ionospheric ramp on top. Without somewhere to put
    them, the inversion absorbs them into slip -- typically as broad, deep,
    entirely fictitious slip, because that is what produces a smooth far-field
    signal.

    ``kind`` is ``"offset"`` (one column per track), ``"linear"`` (offset plus
    ``x`` and ``y`` gradients: three columns) or ``"none"``. Columns are non-zero
    only on their own track's rows. Returns ``(matrix, labels)``; the matrix has
    ``obs.n`` rows.

    The reference implementation reserves space for this and never uses it -- its
    ``bounds_new`` takes an ``add_col`` argument that every caller passes as zero.
    """
    if kind == "none":
        return np.zeros((obs.n, 0)), []
    if kind not in ("offset", "linear"):
        raise ValueError("kind must be 'offset', 'linear' or 'none'")

    ds = obs.ds
    track = ds["track"].values
    x = np.asarray(ds["x"].values, dtype=float)
    y = np.asarray(ds["y"].values, dtype=float)

    columns, labels = [], []
    for name in [str(t) for t in np.unique(track)]:
        sel = track.astype(str) == name
        # Centre and scale each track's coordinates so the ramp columns are O(1)
        # and do not dominate the conditioning of the joint system.
        cx = np.zeros(obs.n)
        cy = np.zeros(obs.n)
        span = max(np.ptp(x[sel]), np.ptp(y[sel]), 1.0)
        cx[sel] = (x[sel] - x[sel].mean()) / span
        cy[sel] = (y[sel] - y[sel].mean()) / span

        columns.append(sel.astype(float))
        labels.append(f"{name}:offset")
        if kind == "linear":
            columns += [cx, cy]
            labels += [f"{name}:dx", f"{name}:dy"]

    return np.column_stack(columns), labels
