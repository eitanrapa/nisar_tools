"""Green's functions: the linear operator from slip on the fault to what we see.

Column ``k`` of the design matrix is the displacement every observation would
record for one metre of slip on element ``k`` and none anywhere else. Because
elastic displacement is linear in the Burgers vector, a slip model is just a
weighted sum of those columns -- which is the whole reason the inversion is a
least-squares problem and not an optimisation over forward models.

Parameters are ordered ``[strike-slip on every element, dip-slip on every
element]``, so a design matrix has ``2 * n_elements`` columns; that block layout
is what the smoothing operator and the bound vectors in
:mod:`nisar_tools.slip.regularize` assume.

The engine is deliberately split from the projection. ``displacement`` returns
east/north/up, and only :func:`project_los` knows about a satellite -- so the
same engine serves a layered-rigidity replacement, or GPS, without the inversion
learning anything new.

**Assembly is serial, deliberately.** The obvious move is a thread pool over
elements, on the usual reasoning that numpy releases the GIL. Measured on the
real problem (238 km fault, 1106 elements, 8000 observations, 10 cores), it makes
things uniformly *worse* -- 1 worker 24.6 s, 2 workers 28.5 s, 4 workers 42.5 s,
10 workers 62-79 s -- and no block size rescues it. Each element's evaluation is
~50 numpy calls on 8000-element arrays, i.e. 64 KB a piece: too small for the
GIL-free stretch to outweigh the dispatch and hand-off overhead, so extra threads
only add contention. Twenty-five seconds for a full design matrix is acceptable;
if it ever needs to be faster, the fix is to vectorize
:mod:`nisar_tools.slip._tde` over elements so each numpy call is large, not to
add threads back.
"""

import numpy as np

from ._tde import tde_disp_hs


class HalfSpaceTDE:
    """Triangular dislocations in a homogeneous elastic half-space.

    ``nu`` is Poisson's ratio; 0.25 is the usual crustal value and the reference
    implementation's default. A homogeneous half-space has no rigidity structure,
    so ``nu`` is the only material parameter the displacements depend on -- the
    shear modulus cancels, and enters only when slip is converted to moment.
    """

    name = "half_space_tde"

    def __init__(self, nu=0.25):
        self.nu = float(nu)

    def displacement(self, mesh, x, y, z=None):
        """ENU displacement per unit slip, shape ``(npts, 3, 2 * n_elements)``.

        Memory-hungry by construction -- three components times every parameter.
        Use :meth:`los_matrix` for an inversion, which never materialises the
        third axis.
        """
        x, y, z = _as_points(x, y, z)
        vertices = mesh.vertices
        out = np.zeros((x.size, 3, 2 * mesh.n_elements))
        for k in range(mesh.n_elements):
            p1, p2, p3 = vertices[k]
            for comp, slip in ((0, (1.0, 0.0, 0.0)), (1, (0.0, 1.0, 0.0))):
                ue, un, uv = tde_disp_hs(x, y, z, p1, p2, p3, *slip, self.nu)
                col = k + comp * mesh.n_elements
                out[:, 0, col] = ue
                out[:, 1, col] = un
                out[:, 2, col] = uv
        _require_finite(out, mesh, x, y, z)
        return out

    def los_matrix(self, mesh, x, y, look_e, look_n, look_u, z=None):
        """Line-of-sight design matrix, shape ``(npts, 2 * n_elements)``.

        Each element's ENU response is projected onto that observation's own look
        vector as it is computed, so the ``(npts, 3)`` intermediate is transient
        and peak memory is the matrix itself.
        """
        x, y, z = _as_points(x, y, z)
        look_e = np.asarray(look_e, dtype=float).ravel()
        look_n = np.asarray(look_n, dtype=float).ravel()
        look_u = np.asarray(look_u, dtype=float).ravel()
        if not (look_e.size == look_n.size == look_u.size == x.size):
            raise ValueError("Look-vector components must match the observation count")

        vertices = mesh.vertices
        g = np.zeros((x.size, 2 * mesh.n_elements))
        for k in range(mesh.n_elements):
            p1, p2, p3 = vertices[k]
            for comp, slip in ((0, (1.0, 0.0, 0.0)), (1, (0.0, 1.0, 0.0))):
                ue, un, uv = tde_disp_hs(x, y, z, p1, p2, p3, *slip, self.nu)
                g[:, k + comp * mesh.n_elements] = (
                    ue * look_e + un * look_n + uv * look_u
                )
        _require_finite(g, mesh, x, y, z)
        return g

    def forward(self, mesh, slip, x, y, look=None, z=None):
        """Displacement produced by a given slip model.

        ``slip`` is either the ``2 * n_elements`` parameter vector or an
        ``(n_elements, 2)`` array of ``(strike_slip, dip_slip)``. With ``look``
        as a 3-tuple of arrays the result is scalar line-of-sight; without it,
        ENU of shape ``(npts, 3)``.
        """
        slip = np.asarray(slip, dtype=float)
        if slip.ndim == 2:
            slip = np.concatenate([slip[:, 0], slip[:, 1]])
        if slip.size != 2 * mesh.n_elements:
            raise ValueError(
                f"slip has {slip.size} values; expected {2 * mesh.n_elements}"
            )
        if look is None:
            return self.displacement(mesh, x, y, z) @ slip
        return self.los_matrix(mesh, x, y, *look, z=z) @ slip


def project_los(disp_enu, look_e, look_n, look_u):
    """Project ENU displacement onto the line of sight.

    The look vector points **from the target to the sensor** and the result is
    positive **toward the sensor** -- both conventions inherited unchanged from
    :mod:`nisar_tools.geometry`, which is why this is a plain dot product with no
    sign to remember. ``disp_enu`` is ``(npts, 3)`` or ``(npts, 3, nparam)``.
    """
    look = np.stack([np.asarray(look_e, dtype=float).ravel(),
                     np.asarray(look_n, dtype=float).ravel(),
                     np.asarray(look_u, dtype=float).ravel()], axis=1)
    disp_enu = np.asarray(disp_enu, dtype=float)
    if disp_enu.ndim == 2:
        return (disp_enu * look).sum(axis=1)
    return np.einsum("pcn,pc->pn", disp_enu, look)


def tde_greens(mesh, obs, nu=0.25):
    """The line-of-sight design matrix for an :class:`~nisar_tools.slip.sampling.Observations`.

    A thin convenience over :meth:`HalfSpaceTDE.los_matrix` that also checks the
    observations were built in the mesh's own frame -- mixing frames is a
    kilometre-scale error that produces a perfectly smooth, entirely wrong matrix.
    """
    if mesh.frame is not None and obs.ds.attrs.get("frame") is not None:
        mesh.frame.require_match(obs.ds.attrs["frame"], "Observations")
    ds = obs.ds
    return HalfSpaceTDE(nu).los_matrix(
        mesh, ds["x"].values, ds["y"].values,
        ds["look_e"].values, ds["look_n"].values, ds["look_u"].values,
    )


def _as_points(x, y, z):
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    z = np.zeros(x.size) if z is None else np.asarray(z, dtype=float).ravel()
    if y.size != x.size or z.size != x.size:
        raise ValueError("x, y and z must have the same length")
    return x, y, z


def _require_finite(g, mesh, x, y, z=None):
    """Refuse a design matrix with non-finite entries, and say which observation.

    A dislocation solution is singular on the fault surface itself, so an
    observation sitting on the surface trace produces NaN. Silently zeroing it
    would leave a row of the inversion quietly meaningless; the caller wants to
    know to widen the exclusion buffer around the trace.

    The distance quoted is a true three-dimensional one. Measuring it in map
    view, as this used to, is the same number on a vertical fault and an
    understatement on a dipping one -- the deep nodes of a 45-degree fault sit
    kilometres to one side, so the nearest node in plan view can be one the
    observation is nowhere near.
    """
    bad = ~np.isfinite(g)
    if not bad.any():
        return
    rows = np.unique(np.nonzero(bad.any(axis=tuple(range(1, g.ndim))))[0])
    z = np.zeros(x.size) if z is None else np.asarray(z, dtype=float).ravel()
    nearest = np.sqrt(np.min(
        (x[rows, None] - mesh.nodes[None, :, 0]) ** 2
        + (y[rows, None] - mesh.nodes[None, :, 1]) ** 2
        + (z[rows, None] - mesh.nodes[None, :, 2]) ** 2,
        axis=1,
    ))
    raise ValueError(
        f"{rows.size} observation(s) gave a non-finite Green's function; the "
        f"closest sits {nearest.min():.0f} m from the fault. Dislocation "
        "solutions are singular on the fault surface -- exclude observations "
        "near the trace (Observations.from_los(exclude_within=...)). "
        f"Offending observation indices: {rows[:10].tolist()}"
    )
