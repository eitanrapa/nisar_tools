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
