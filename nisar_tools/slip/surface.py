"""The curved fault surface: where the fault goes once it stops being vertical.

A dipping fault is described here as a single scalar field ``cross = f(s, z)``
over the ``(arc length, depth)`` rectangle -- how far, and which way, the fault
has stepped off its surface trace by the time it reaches depth ``z``. That is
enough to place every mesh node, and it keeps the whole of
:mod:`nisar_tools.slip.mesh` working in the parameterization it already uses.

The field is *fitted*, not evaluated, because the constraints are sparse: the
surface trace pins ``cross = 0`` at ``z = 0``, and each deep segment pins a
bottom line at ``z = -max_depth``, and between them there is nothing. What fills
that gap is the regularizer, so the regularizer is not a detail -- it **is** the
dip profile. This module therefore ports the specific gridder the reference
implementation uses (John D'Errico's ``gridfit``, vendored in SlipSolve-Curve)
rather than substituting a generic smoother:

* ``RBFInterpolator`` and ``SmoothBivariateSpline`` are not linear in the node
  values in the same way and do not reproduce ``gridfit``'s ``smoothness``.
* :func:`nisar_tools._kernels.smooth_surface` is a filled-grid smoother -- it
  needs a value at (almost) every cell and returns NaN interiors when handed a
  cloud with a hole through the middle, which is exactly this cloud's shape.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr

from .trace import VERTICAL_TOL_DEG, dip_offset  # noqa: F401  (re-exported)

#: Default smoothing weight, matching the reference's Myanmar configuration
#: (``cfg.geometry.surfaceFitSmoothness``).
DEFAULT_SMOOTHNESS = 0.008

#: How much more the surface trace counts than the deep bottom lines.
#:
#: The reference does not weight rows; it *oversamples*, interpolating the
#: surface trace to ``1e5`` times its vertex count and the bottom lines to only
#: ``1e2`` times theirs (``surfaceInterpolationFactor`` /
#: ``bottomInterpolationFactor``). In a least-squares fit those are the same
#: thing -- a thousand collinear copies of a constraint is a constraint with a
#: thousand times the weight -- except that the literal reading also builds a
#: twelve-million-row design matrix for a 240 km trace. Weighting the rows
#: reproduces the effect at a few thousand rows. Sampling density is set
#: separately, by :data:`SAMPLES_PER_CELL`, and only has to be enough that every
#: grid cell is covered.
SURFACE_WEIGHT_RATIO = 1e3

#: Control points per grid cell along each constraint line.
SAMPLES_PER_CELL = 8


def gridfit(x, y, v, x_nodes, y_nodes, smoothness=1.0, weights=None,
            interp="triangle", regularizer="gradient", autoscale=True,
            solver="lsqr"):
    """Fit a smooth surface on a lattice through scattered ``(x, y, v)`` samples.

    A port of ``gridfit.m`` restricted to the options the reference actually
    passes -- it supplies only ``smoothness``, so everything else here defaults
    to ``gridfit``'s own default. Returns the fitted values on the lattice with
    shape ``(len(y_nodes), len(x_nodes))``.

    Unlike an interpolant this is an *approximation*: the surface is not required
    to pass through the data, and it is defined everywhere on the lattice
    including where there is no data at all. Both properties are wanted here --
    the control points are contradictory where segments meet, and most of the
    ``(s, z)`` rectangle has no constraint whatsoever.

    ``regularizer="gradient"`` is the reference's choice and is **not** the
    Laplacian. Both build the same second-difference stencils in each direction;
    ``"diffusion"`` *adds* them, penalising only the sum, while ``"gradient"``
    *stacks* them, penalising each direction separately. Penalising the sum lets
    curvature in one direction pay for curvature in the other, which on this
    problem lets the surface bow along strike to buy a straighter dip profile.

    ``autoscale`` divides each direction's spacing by its mean before the
    stencils are built. It is not cosmetic: the along-strike node spacing is
    kilometres and the depth spacing is hundreds of metres, so without it the
    smoothing is anisotropic by that ratio squared.

    ``weights`` scales the data rows only, before the regularizer is normalised
    against them -- so a row's weight changes what the fit is, not what
    ``smoothness`` means relative to it.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    w = (np.ones(x.size) if weights is None
         else np.broadcast_to(np.asarray(weights, dtype=float).ravel(), x.shape).copy())

    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(v) & np.isfinite(w)
    x, y, v, w = x[good], y[good], v[good], w[good]
    if x.size < 3:
        raise ValueError("gridfit needs at least three finite control points")

    x_nodes = np.asarray(x_nodes, dtype=float).ravel()
    y_nodes = np.asarray(y_nodes, dtype=float).ravel()
    if np.any(np.diff(x_nodes) <= 0) or np.any(np.diff(y_nodes) <= 0):
        raise ValueError("x_nodes and y_nodes must be strictly increasing")

    dx, dy = np.diff(x_nodes), np.diff(y_nodes)
    nx, ny = x_nodes.size, y_nodes.size
    ngrid = nx * ny
    x_scale = dx.mean() if autoscale else 1.0
    y_scale = dy.mean() if autoscale else 1.0

    # Node (ix, iy) is column iy + ny*ix -- MATLAB's column-major order, kept so
    # the stencil offsets below read the same as the reference's.
    ix = np.clip(np.searchsorted(x_nodes, x, side="right") - 1, 0, nx - 2)
    iy = np.clip(np.searchsorted(y_nodes, y, side="right") - 1, 0, ny - 2)
    tx = np.clip((x - x_nodes[ix]) / dx[ix], 0.0, 1.0)
    ty = np.clip((y - y_nodes[iy]) / dy[iy], 0.0, 1.0)
    ind = iy + ny * ix

    rows = np.repeat(np.arange(x.size), 3 if interp == "triangle" else 4)
    if interp == "triangle":
        # Each cell is cut by its main diagonal; a point uses the corner it is
        # on the same side of, plus the two shared corners.
        third = np.where(tx > ty, ny, 1)
        t1, t2 = np.minimum(tx, ty), np.maximum(tx, ty)
        cols = np.column_stack([ind, ind + ny + 1, ind + third]).ravel()
        vals = np.column_stack([1.0 - t2, t1, t2 - t1]).ravel()
    elif interp == "bilinear":
        cols = np.column_stack([ind, ind + 1, ind + ny, ind + ny + 1]).ravel()
        vals = np.column_stack([(1 - tx) * (1 - ty), (1 - tx) * ty,
                                tx * (1 - ty), tx * ty]).ravel()
    else:
        raise ValueError(f"Unknown interp {interp!r}; expected triangle or bilinear")

    design = sp.csr_matrix((vals, (rows, cols)), shape=(x.size, ngrid))
    design = sp.diags(w) @ design
    rhs = v * w

    reg = _regularizer(nx, ny, dx, dy, x_scale, y_scale, regularizer)

    # Scale the regularizer against the data block so that `smoothness` means the
    # same thing whatever the units are. The 1-norm, as the reference uses.
    n_data = _norm1(design)
    n_reg = _norm1(reg)
    if n_reg == 0:
        raise ValueError("Lattice is too small to regularize; need >= 3 nodes on an axis")
    stacked = sp.vstack([design, reg * (float(smoothness) * n_data / n_reg)], format="csr")
    full_rhs = np.concatenate([rhs, np.zeros(reg.shape[0])])

    if solver == "lsqr":
        # The regularizer leaves the stacked system well conditioned, but it is
        # rank-deficient without it, so an iterative least-squares solve is the
        # honest analogue of MATLAB's sparse QR backslash.
        z = lsqr(stacked, full_rhs, atol=1e-12, btol=1e-12,
                 iter_lim=max(10000, 10 * ngrid))[0]
    elif solver == "normal":
        z = sp.linalg.spsolve((stacked.T @ stacked).tocsc(), stacked.T @ full_rhs)
    else:
        raise ValueError(f"Unknown solver {solver!r}; expected lsqr or normal")

    return z.reshape((ny, nx), order="F")


def _regularizer(nx, ny, dx, dy, x_scale, y_scale, kind):
    """Second-difference stencils in each direction, stacked or summed."""
    blocks = []
    for axis in (0, 1):
        if axis == 0:                      # along y, for interior rows
            if ny < 3:
                continue
            i, j = np.meshgrid(np.arange(nx), np.arange(1, ny - 1), indexing="xy")
            ind = j.ravel() + ny * i.ravel()
            h1, h2 = dy[j.ravel() - 1] / y_scale, dy[j.ravel()] / y_scale
            offsets = (-1, 0, 1)
        else:                              # along x, for interior columns
            if nx < 3:
                continue
            i, j = np.meshgrid(np.arange(1, nx - 1), np.arange(ny), indexing="xy")
            ind = j.ravel() + ny * i.ravel()
            h1, h2 = dx[i.ravel() - 1] / x_scale, dx[i.ravel()] / x_scale
            offsets = (-ny, 0, ny)

        vals = np.column_stack([-2.0 / (h1 * (h1 + h2)),
                                2.0 / (h1 * h2),
                                -2.0 / (h2 * (h1 + h2))]).ravel()
        cols = np.column_stack([ind + o for o in offsets]).ravel()
        rows = np.repeat(ind, 3)
        blocks.append(sp.coo_matrix((vals, (rows, cols)),
                                    shape=(nx * ny, nx * ny)).tocsr())

    if not blocks:
        return sp.csr_matrix((0, nx * ny))
    if kind == "gradient":
        return sp.vstack(blocks, format="csr")
    if kind in ("diffusion", "laplacian"):
        return sum(blocks[1:], blocks[0]).tocsr()
    raise ValueError(f"Unknown regularizer {kind!r}; expected gradient or diffusion")


def _norm1(matrix):
    """Maximum absolute column sum -- MATLAB's ``norm(A, 1)``."""
    if matrix.shape[0] == 0:
        return 0.0
    return float(abs(matrix).sum(axis=0).max())


def centres(values):
    """Midpoints of consecutive entries -- the reference's ``movmean(x, 2)``."""
    values = np.asarray(values, dtype=float)
    return 0.5 * (values[:-1] + values[1:])


class FaultSurface:
    """A fitted ``cross(s, z)`` field, on the mesh lattice and at cell centres.

    Two grids, because they serve different consumers: the **node** grid places
    the mesh's vertices, and the **centre** grid is where the layered Green's
    functions put their point sources. The reference computes the centre grid by
    running the fit a second time on the centre lattice rather than interpolating
    the node grid (``gridFitInterpolate.m``), and that is reproduced here -- the
    two differ wherever the regularizer is doing the work, which for this cloud
    is almost everywhere.
    """

    def __init__(self, s_nodes, z_nodes, cross_nodes,
                 s_centres=None, z_centres=None, cross_centres=None, attrs=None):
        self.s_nodes = np.asarray(s_nodes, dtype=float)
        self.z_nodes = np.asarray(z_nodes, dtype=float)
        self.cross_nodes = np.asarray(cross_nodes, dtype=float)
        if self.cross_nodes.shape != (self.z_nodes.size, self.s_nodes.size):
            raise ValueError(
                f"cross_nodes has shape {self.cross_nodes.shape}; expected "
                f"{(self.z_nodes.size, self.s_nodes.size)} = (depth, along)"
            )
        self.s_centres = None if s_centres is None else np.asarray(s_centres, float)
        self.z_centres = None if z_centres is None else np.asarray(z_centres, float)
        self.cross_centres = (None if cross_centres is None
                              else np.asarray(cross_centres, float))
        self.attrs = dict(attrs or {})

    # -- construction ------------------------------------------------------
    @classmethod
    def from_control_points(cls, s, z, cross, s_nodes, z_nodes,
                            smoothness=DEFAULT_SMOOTHNESS, weights=None, **kwargs):
        """Fit both grids from a curvilinear control cloud."""
        node_grid = gridfit(s, z, cross, s_nodes, z_nodes,
                            smoothness=smoothness, weights=weights, **kwargs)
        s_c, z_c = centres(s_nodes), centres(z_nodes)
        centre_grid = (gridfit(s, z, cross, s_c, z_c, smoothness=smoothness,
                               weights=weights, **kwargs)
                       if s_c.size and z_c.size else None)
        return cls(s_nodes, z_nodes, node_grid, s_c, z_c, centre_grid,
                   attrs={"smoothness": float(smoothness)})

    @classmethod
    def from_segments(cls, trace, frame, segments, dips, s_nodes, z_nodes,
                      depth_control=None, smoothness=DEFAULT_SMOOTHNESS,
                      surface_weight_ratio=SURFACE_WEIGHT_RATIO,
                      samples_per_cell=SAMPLES_PER_CELL, **kwargs):
        """Build the control cloud from a trace plus one dip per deep segment.

        ``segments`` is a sequence of :class:`~nisar_tools.slip.trace.FaultSegment`
        and ``dips`` one dip in degrees for each. The cloud has exactly two
        constrained depths -- ``cross = 0`` all along ``z = 0``, and each
        segment's down-dip projection at ``z = z_nodes.min()`` -- which is why
        the reference's "variable dip" is really *dip varying along strike, and
        planar with depth*. ``depth_control`` supplies intermediate
        ``(x, y, depth)`` points (from relocated seismicity, say) and is the only
        way to bend the profile with depth.
        """
        segments = list(segments)
        dips = np.broadcast_to(np.asarray(dips, dtype=float).ravel(), (len(segments),))
        if not segments:
            raise ValueError("from_segments needs at least one segment")

        max_depth = float(-np.min(z_nodes))
        radius = trace.min_curvature_radius(frame)
        worst = float(np.max(np.abs(dip_offset(max_depth, dips))))
        if worst >= radius:
            raise ValueError(
                f"A down-dip offset of {worst / 1e3:.1f} km at {max_depth / 1e3:.1f} km "
                f"depth exceeds the trace's smallest radius of curvature "
                f"({radius / 1e3:.1f} km), so the projected surface folds through "
                "itself and the mesh would contain inverted elements. Use a "
                "steeper dip, a shallower fault, or a smoother trace."
            )

        n_along = max(2, int(round(samples_per_cell * (s_nodes.size - 1))))
        s_dense = np.linspace(s_nodes.min(), s_nodes.max(), n_along)

        # The surface trace: cross is identically zero, by definition of the frame.
        s_all = [s_dense]
        z_all = [np.zeros_like(s_dense)]
        c_all = [np.zeros_like(s_dense)]
        w_all = [np.full(s_dense.size, float(surface_weight_ratio))]

        for segment, dip in zip(segments, dips):
            bx, by = segment.project(max_depth, dip)
            t = np.linspace(0.0, 1.0, max(2, int(round(samples_per_cell * s_nodes.size))))
            px = bx[0] + t * (bx[1] - bx[0])
            py = by[0] + t * (by[1] - by[0])
            s_seg, c_seg = trace.to_curvilinear(px, py, frame)
            s_all.append(s_seg)
            z_all.append(np.full(s_seg.size, -max_depth))
            c_all.append(c_seg)
            w_all.append(np.ones(s_seg.size))

        if depth_control is not None:
            cx, cy, cz = (np.asarray(a, dtype=float).ravel() for a in depth_control)
            s_c, c_c = trace.to_curvilinear(cx, cy, frame)
            s_all.append(s_c)
            z_all.append(-np.abs(cz))
            c_all.append(c_c)
            w_all.append(np.ones(s_c.size))

        surface = cls.from_control_points(
            np.concatenate(s_all), np.concatenate(z_all), np.concatenate(c_all),
            s_nodes, z_nodes, smoothness=smoothness, weights=np.concatenate(w_all),
            **kwargs,
        )
        surface.attrs.update(
            dips=[float(d) for d in dips],
            segments=[[s.x_begin, s.y_begin, s.x_end, s.y_end] for s in segments],
            max_depth=max_depth,
            surface_weight_ratio=float(surface_weight_ratio),
            min_curvature_radius=float(radius),
        )
        return surface

    @classmethod
    def vertical(cls, s_nodes, z_nodes):
        """A surface with ``cross`` identically zero -- the phase-one fault."""
        shape = (np.asarray(z_nodes).size, np.asarray(s_nodes).size)
        s_c, z_c = centres(s_nodes), centres(z_nodes)
        return cls(s_nodes, z_nodes, np.zeros(shape), s_c, z_c,
                   np.zeros((z_c.size, s_c.size)), attrs={"kind": "vertical"})

    # -- use ---------------------------------------------------------------
    def nodes(self, trace, frame):
        """Map positions of every lattice node; ``(x, y, z, s)``, each ``(nz*ns,)``.

        Flattened with the along-strike index fastest, which is the ordering
        :func:`nisar_tools.slip.mesh._lattice_triangles` assumes.
        """
        return self._place(trace, frame, self.s_nodes, self.z_nodes, self.cross_nodes)

    def centre_points(self, trace, frame):
        """Map positions of every cell centre; the layered point-source locations."""
        if self.cross_centres is None:
            return None
        return self._place(trace, frame, self.s_centres, self.z_centres,
                           self.cross_centres)

    @staticmethod
    def _place(trace, frame, s_axis, z_axis, cross):
        ss, zz = np.meshgrid(s_axis, z_axis, indexing="xy")
        x, y = trace.from_curvilinear(ss.ravel(), cross.ravel(), frame)
        return x, y, zz.ravel(), ss.ravel()

    def __repr__(self):
        return (f"<FaultSurface {self.s_nodes.size}x{self.z_nodes.size} "
                f"cross={self.cross_nodes.min() / 1e3:.1f}.."
                f"{self.cross_nodes.max() / 1e3:.1f}km>")
