"""The inversion itself: a bounded, regularized linear least-squares solve.

The system is the reference implementation's, stacked:

.. code-block:: text

    [ G * w        ]        [ d * w ]
    [ lam * L / nL ] * m =  [   0   ]
    [ W_boundary   ]        [   0   ]

with ``lb <= m <= ub``. ``G`` is the Green's matrix, ``w`` the per-observation
weights, ``L`` the smoothing operator normalised by its own row count (so
``lam`` means the same thing as the mesh is refined), and ``W_boundary`` the
zero-slip edge rows.

It is *not* assembled as one dense array. ``G`` is dense and ``L``/``W`` are
sparse, so the stack is wrapped in a :class:`~scipy.sparse.linalg.LinearOperator`
that applies each in its natural form: the smoothing block then costs a sparse
matrix-vector product rather than a dense one, and the 0.5-1 GB materialised copy
never exists. ``scipy.optimize.lsq_linear`` accepts a ``LinearOperator`` with
``lsq_solver="lsmr"``, which is the combination used here.

Solver honesty matters more than usual in this problem. Runtime depends far more
on conditioning than on size, and an iteration-capped solve returns a
plausible-looking model with a meaningless variance reduction -- so
:class:`SlipModel` carries the solver's status and refuses to pretend.
"""

from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp
from scipy.optimize import lsq_linear
from scipy.sparse.linalg import LinearOperator

from .basis import make_basis
from .greens import HalfSpaceTDE
from .regularize import (
    laplace_beltrami,
    neighbor_smoothing,
    ramp_columns,
    slip_bounds,
    zero_slip_boundary,  # noqa: F401  (public API, and used by callers)
)


def _blocked(projection):
    """A basis projection applied to both the strike and dip halves."""
    zero = sp.csr_matrix(projection.shape)
    return sp.bmat([[projection, zero], [zero, projection]], format="csr")


def _far_enough(x, y, trace_nodes, distance, block=20000):
    """True where a point is further than ``distance`` from any trace node.

    Blocked over the points, since the full pairwise matrix against a long fault's
    node row is large and only its row minimum is wanted.
    """
    keep = np.empty(x.size, dtype=bool)
    for start in range(0, x.size, block):
        sl = slice(start, start + block)
        d2 = ((x[sl, None] - trace_nodes[None, :, 0]) ** 2
              + (y[sl, None] - trace_nodes[None, :, 1]) ** 2)
        keep[sl] = d2.min(axis=1) > distance ** 2
    return keep


class _MissingEngine:
    """Placeholder for an engine a loaded model cannot rebuild.

    A layered engine is defined by its EDGRN tables, which are megabytes and are
    deliberately not saved with a model -- as the Green's matrix is not. Silently
    substituting the homogeneous half-space would let ``forward`` return plausible
    numbers computed with the wrong physics, so it raises instead.
    """

    def __init__(self, name):
        self.name = name
        self.nu = 0.25

    def _refuse(self, *_, **__):
        raise RuntimeError(
            f"This model was solved with the {self.name!r} engine, whose Green's "
            "functions come from EDGRN tables that are not stored in a saved "
            "model. Rebuild the engine (LayeredPointSource with the same tables) "
            "and construct a fresh SlipInversion to forward-model with it."
        )

    forward = displacement = los_matrix = _refuse


def _boundary_rows(basis, sides, ratio):
    """Selection rows pulling the flagged boundary parameters toward zero."""
    idx = np.nonzero(basis.boundary(sides))[0]
    n = basis.n_basis
    if idx.size == 0:
        return sp.csr_matrix((0, 2 * n))
    cols = np.concatenate([idx, idx + n])
    return sp.csr_matrix(
        (np.full(cols.size, float(ratio)), (np.arange(cols.size), cols)),
        shape=(cols.size, 2 * n),
    )

#: Crustal shear modulus used for moment when no velocity model is supplied.
DEFAULT_SHEAR_MODULUS = 30e9

#: Default cap on trust-region iterations.
#:
#: Generous on purpose. Iteration count tracks *conditioning*, not problem size,
#: and not monotonically in the smoothing weight either: on the reference
#: Venezuela problem, weights of 2.0, 0.3 and 0.01 each converge in about 30
#: iterations while 0.05 needs 236. A tight cap therefore silently truncates
#: isolated points of an L-curve sweep rather than the obvious rough end. Since a
#: well-conditioned solve stops long before the cap, raising it costs nothing
#: where it is not needed.
DEFAULT_MAX_ITER = 400

#: Inner-solver tolerance handed to ``lsq_linear``'s ``lsmr`` sub-problem.
#:
#: ``"auto"`` lets scipy tighten the inner tolerance as the outer trust-region
#: iteration converges, instead of solving every sub-problem to ``atol=btol=0``
#: -- which is what scipy's ``None`` default does, and which makes ``lsmr`` run
#: to its own iteration cap on every single outer step.
#:
#: Measured on the reference problem (590 elements, 860 observations, 1186
#: parameters): ``None`` takes 162 outer iterations and **174 809** matrix-vector
#: products in 22.3 s; ``"auto"`` takes 28 outer iterations and **26 359** in
#: 3.9 s -- a **5.8x** speed-up for the same variance reduction (90.06%), the two
#: slip models agreeing to 6.6 mm. Since roughly 89% of a solve is spent inside
#: those products, cutting their number is by far the largest available win.
#:
#: ⚠️ **That win is not uniform in the smoothing weight.** Re-measured 2026-07-28
#: on the test fixture (240 elements, 761 observations, 480 parameters), counting
#: matrix-vector products, ``None`` / ``"auto"``::
#:
#:     lam    2.0    1.0    0.5    0.3    0.1   0.05
#:          0.70x  2.18x  1.93x  1.91x  1.19x  0.78x
#:
#: so the adaptive tolerance *loses* at both ends of the sweep and wins by ~2x in
#: the middle, netting **1.17x** over the whole sweep -- which is how
#: :meth:`SlipInversion.l_curve` uses it, and why it stays the default. Judge it
#: over a sweep, never at one weight. Note also that outer iteration count is the
#: wrong meter: ``"auto"`` solves each sub-problem loosely, so it routinely takes
#: *more* outer steps while doing less total work.
#:
#: Do **not** reach for ``lsmr_maxiter`` instead. A hard cap is not monotone:
#: caps of 100 and 200 both drove the outer loop into ``DEFAULT_MAX_ITER`` and
#: returned materially different, *worse* models (4.15 m and 2.47 m from the
#: converged answer) while a cap of 25 was fine. The adaptive tolerance has no
#: such cliff.
DEFAULT_LSMR_TOL = "auto"


class SlipInversion:
    """A mesh, a set of observations, and the Green's matrix relating them.

    The Green's matrix is built once, on construction, and reused across every
    smoothing weight -- which is what makes an L-curve sweep cheap.
    """

    def __init__(self, mesh, obs, engine=None, ramp="none", basis="element",
                 velocity_model=None):
        """``engine`` and ``basis`` choose the physics and the parameterization.

        ``velocity_model`` is **not** used to compute Green's functions -- for a
        layered medium that information is already inside the engine's EDGRN
        tables, and for a homogeneous half-space the displacements do not depend
        on rigidity at all (it cancels; only Poisson's ratio survives). What it is
        for is **moment**: ``sum(mu * area * slip)`` needs a rigidity, and without
        one every reported ``Mw`` silently assumes
        :data:`DEFAULT_SHEAR_MODULUS` = 30 GPa. Pass the same
        :class:`~nisar_tools.slip.edgrn.VelocityModel` you built the tables from
        and every statistic -- ``moment``, ``moment_magnitude``, ``to_text``,
        ``repr`` -- uses the rigidity at each parameter's own depth instead.

        It matters more than it sounds. A model whose shallowest layer is 9 GPa,
        as the Venezuela CRUST2.0 profile's is, has a third the rigidity of the
        default over the top 2 km -- which is exactly where a vertical
        strike-slip fault's shallow slip sits.
        """
        self.mesh = mesh
        self.obs = obs
        self.engine = engine or HalfSpaceTDE()
        self.basis = make_basis(mesh, basis)
        self.velocity_model = velocity_model

        if mesh.frame is not None and obs.ds.attrs.get("frame") is not None:
            mesh.frame.require_match(obs.ds.attrs["frame"], "Observations")

        ds = obs.ds
        self.d = np.asarray(ds["los"].values, dtype=float)
        self.w = np.asarray(ds["weight"].values, dtype=float)
        # An engine that can integrate a basis function directly is given the
        # basis, so a nodal column is the integral of the tent itself. One that
        # cannot -- the closed-form triangular dislocation, which is defined on a
        # patch of *constant* slip -- assembles per element and has the basis
        # projected on afterwards. Both preserve the moment exactly; only the
        # first gets the near-field shape of a linearly varying slip patch right.
        points = (ds["x"].values, ds["y"].values,
                  ds["look_e"].values, ds["look_n"].values, ds["look_u"].values)
        self.exact_basis = bool(getattr(self.engine, "supports_basis", False))
        if self.exact_basis:
            self.g = self.engine.los_matrix(mesh, *points, basis=self.basis)
        else:
            element_g = self.engine.los_matrix(mesh, *points)
            self.g = (element_g if self.basis.name == "element"
                      else element_g @ _blocked(self.basis.projection()))

        self.ramp, self.ramp_labels = ramp_columns(obs, ramp)
        self.n_slip = self.basis.n_param
        self.n_ramp = self.ramp.shape[1]

    @property
    def n_param(self):
        return self.n_slip + self.n_ramp

    # -- solving -----------------------------------------------------------
    def solve(self, smoothing=0.3, ss_ratio=1.0, ds_ratio=3.0,
              boundary_ratio=1.0, sides=("bottom", "left", "right"),
              polarity=None, strike=(-10.0, 10.0), dip=(-10.0, 10.0),
              max_iter=DEFAULT_MAX_ITER, tol=1e-10,
              lsmr_tol=DEFAULT_LSMR_TOL, verbose=0):
        """Solve for slip at one smoothing weight; returns a :class:`SlipModel`.

        ``lsmr_tol`` is the inner least-squares tolerance; see
        :data:`DEFAULT_LSMR_TOL` for why the default is ``"auto"`` and not
        scipy's ``None``. Pass ``lsmr_tol=None`` to reproduce a pre-2026-07-27
        run exactly.
        """
        if self.basis.name == "element":
            smooth = neighbor_smoothing(self.mesh.neighbors, ss_ratio, ds_ratio)
        else:
            # A continuous field lives on the nodes, so neighbouring *elements* is
            # the wrong thing to difference; the surface Laplacian is the right one.
            smooth = laplace_beltrami(self.mesh, ss_ratio, ds_ratio)
        boundary = _boundary_rows(self.basis, sides, boundary_ratio)
        lo, hi = slip_bounds(self.basis.n_basis, strike, dip, polarity)

        if self.n_ramp:
            # Nuisance columns are unbounded and unregularized: their whole job is
            # to soak up whatever the slip model should not be explaining.
            pad = sp.csr_matrix((smooth.shape[0], self.n_ramp))
            smooth = sp.hstack([smooth, pad], format="csr")
            boundary = sp.hstack(
                [boundary, sp.csr_matrix((boundary.shape[0], self.n_ramp))], format="csr"
            )
            lo = np.concatenate([lo, np.full(self.n_ramp, -np.inf)])
            hi = np.concatenate([hi, np.full(self.n_ramp, np.inf)])

        n_smooth = max(1, smooth.shape[0])
        scaled_smooth = smooth * (float(smoothing) / n_smooth)

        design = np.hstack([self.g, self.ramp]) if self.n_ramp else self.g
        weighted = design * self.w[:, None]
        rhs = np.concatenate([
            self.d * self.w,
            np.zeros(smooth.shape[0]),
            np.zeros(boundary.shape[0]),
        ])

        operator = _StackedOperator(weighted, scaled_smooth, boundary)
        result = lsq_linear(
            operator, rhs, bounds=(lo, hi),
            lsq_solver="lsmr", max_iter=max_iter, tol=tol,
            lsmr_tol=lsmr_tol, verbose=verbose,
        )

        return SlipModel(
            self, result.x, result,
            smoothing=float(smoothing),
            smooth_matrix=smooth,
            options={
                "ss_ratio": float(ss_ratio), "ds_ratio": float(ds_ratio),
                "boundary_ratio": float(boundary_ratio), "sides": list(sides),
                "polarity": None if polarity is None else list(polarity),
                "strike": list(strike), "dip": list(dip), "max_iter": int(max_iter),
                "lsmr_tol": lsmr_tol,
                # Carried so a saved model keeps its own rigidity; without it a
                # reloaded layered model would report a 30 GPa moment.
                "velocity_model": (None if self.velocity_model is None
                                   else self.velocity_model.to_dict()),
            },
        )

    def l_curve(self, smoothing_values, **kwargs):
        """Solve at each smoothing weight and tabulate misfit against roughness.

        The corner of the resulting curve is the conventional choice of weight --
        past it the model stops improving the fit and only gets rougher. Returned
        as an :class:`xarray.Dataset` so it can be persisted and plotted; the
        models themselves come back in ``models``.

        Swept from **large to small** smoothing deliberately: a smoother problem
        is better conditioned and converges in fewer iterations, and the sweep's
        total cost is dominated by its roughest end.

        **The sweep is serial, deliberately.** Every weight is an independent
        solve over the same Green's matrix, which makes a worker pool the obvious
        move; measured on an 8-weight sweep (240 elements, 578 observations, 10
        cores), threads gave **1.02x** at 2 workers and got *worse* above that
        (0.96x at 4, 0.90x at 8), and a process pool -- pickling ``G`` once per
        worker -- managed only **1.10x**. Two independent reasons:

        * **Load imbalance dominates.** The sweep is not eight equal solves. On
          that run the weights cost 0.78/0.75/0.54/0.84/0.61/0.91/**12.15**/1.59
          seconds: ``lam=0.02`` ran to :data:`DEFAULT_MAX_ITER` and was **67% of
          the entire sweep** on its own, so no scheduler beats **1.50x**. The
          expensive weight is precisely the one whose result is meaningless
          (``converged`` is False) -- it should be dropped, not parallelised.
        * **Threads cannot overlap the solver.** scipy's ``lsmr`` is pure Python,
          so its iteration loop holds the GIL between matrix-vector products.
          Capping ``OMP_NUM_THREADS=1`` to rule out BLAS oversubscription changed
          nothing (1.03x at 2 workers), which is what identifies the GIL rather
          than core contention as the limit. (The dense matrix-vector product on
          its own already runs at ~34 GFLOP/s, i.e. multi-threaded BLAS.)

        A ``workers=`` argument was written, verified to give identical results,
        and then removed: it is API surface that can only make a sweep slower.
        What does make one faster is :data:`DEFAULT_LSMR_TOL` (a measured
        **5.8x** on a single solve) and dropping the weights that never converge.

        Green's assembly is not parallelisable either -- see
        :mod:`nisar_tools.slip.greens`, where threading over elements measured
        uniformly worse for a different reason again (numpy dispatch overhead).
        """
        import xarray as xr

        values = np.sort(np.asarray(smoothing_values, dtype=float))[::-1]
        models = [self.solve(smoothing=v, **kwargs) for v in values]

        rows = []
        for model in models:
            rows.append((model.rms_misfit, model.roughness, model.variance_reduction,
                         model.max_slip, model.moment_magnitude,
                         int(model.result.nit), int(model.result.status),
                         bool(model.converged)))

        rows = np.array(rows, dtype=float)
        ds = xr.Dataset(
            {
                "rms_misfit": ("smoothing", rows[:, 0]),
                "roughness": ("smoothing", rows[:, 1]),
                "variance_reduction": ("smoothing", rows[:, 2]),
                "max_slip": ("smoothing", rows[:, 3]),
                "moment_magnitude": ("smoothing", rows[:, 4]),
                "iterations": ("smoothing", rows[:, 5].astype(int)),
                "status": ("smoothing", rows[:, 6].astype(int)),
                "converged": ("smoothing", rows[:, 7].astype(bool)),
            },
            coords={"smoothing": values},
        )
        ds.attrs["n_observations"] = self.obs.n
        ds.attrs["n_parameters"] = self.n_param
        return ds, models

    def __repr__(self):
        return (f"<SlipInversion obs={self.obs.n} elements={self.mesh.n_elements} "
                f"params={self.n_param}>")


class _StackedOperator(LinearOperator):
    """``[dense; sparse; sparse]`` applied without materialising the stack.

    Keeping the smoothing and boundary blocks sparse is not a micro-optimisation:
    at realistic sizes the smoothing block has more rows than the data does, and
    densifying it would both dominate memory and make every matrix-vector product
    several times more expensive than the physics it is regularising.
    """

    def __init__(self, dense, *sparse_blocks):
        self.dense = dense
        self.blocks = [b for b in sparse_blocks if b.shape[0] > 0]
        rows = dense.shape[0] + sum(b.shape[0] for b in self.blocks)
        super().__init__(dtype=np.float64, shape=(rows, dense.shape[1]))

    def _matvec(self, x):
        x = np.asarray(x, dtype=float).ravel()
        return np.concatenate([self.dense @ x] + [b @ x for b in self.blocks])

    def _rmatvec(self, y):
        y = np.asarray(y, dtype=float).ravel()
        n = self.dense.shape[0]
        out = self.dense.T @ y[:n]
        for block in self.blocks:
            out = out + block.T @ y[n:n + block.shape[0]]
            n += block.shape[0]
        return out


class SlipModel:
    """A solved slip distribution, with its fit statistics and provenance."""

    STAGE = "slip_model"

    def __init__(self, inversion, x, result, smoothing, smooth_matrix, options):
        self.mesh = inversion.mesh
        self.obs = inversion.obs
        self.x = np.asarray(x, dtype=float)
        self.result = result
        self.smoothing = smoothing
        self.options = options
        self._smooth = smooth_matrix
        self._inversion = inversion

        self.basis = getattr(inversion, "basis", None) or make_basis(self.mesh)
        n = self.basis.n_basis
        self.strike_slip = self.x[:n]
        self.dip_slip = self.x[n:2 * n]
        self.ramp = self.x[2 * n:]
        self.ramp_labels = inversion.ramp_labels

        design = (np.hstack([inversion.g, inversion.ramp])
                  if inversion.n_ramp else inversion.g)
        self.prediction = design @ self.x
        self.data = inversion.d
        #: Observed minus modelled, the geodetic convention -- so a positive
        #: residual is displacement the model failed to account for, and the
        #: residual map reads in the same sense as the data map beside it.
        self.residual = self.data - self.prediction

    # -- fit ---------------------------------------------------------------
    @property
    def converged(self):
        """True if the solver stopped on a tolerance rather than the iteration cap.

        Every statistic below is meaningless otherwise, which is why it is a
        property of the model and not something buried in a log line.
        """
        return bool(self.result.status > 0 and self.result.nit < self.options["max_iter"])

    @property
    def variance_reduction(self):
        """Percentage of the data's variance explained by the model."""
        total = float(self.data @ self.data)
        if total == 0:
            return 0.0
        return 100.0 * (total - float(self.residual @ self.residual)) / total

    @property
    def rms_misfit(self):
        return float(np.sqrt(np.mean(self.residual ** 2)))

    @property
    def roughness(self):
        """RMS of the smoothing operator applied to the model."""
        v = self._smooth @ self.x
        return float(np.sqrt(np.mean(v ** 2))) if v.size else 0.0

    @property
    def slip_magnitude(self):
        return np.hypot(self.strike_slip, self.dip_slip)

    @property
    def max_slip(self):
        return float(self.slip_magnitude.max())

    def moment(self, shear_modulus=None):
        """Scalar seismic moment, ``sum(mu * area * slip)``, in newton-metres.

        ``shear_modulus`` may be a scalar, one value per parameter, or a
        :class:`~nisar_tools.slip.edgrn.VelocityModel`, in which case the rigidity
        is sampled at each parameter's own depth. That is the point of a layered
        inversion reaching this far: a uniform 30 GPa overstates the rigidity of
        the shallow crust, so it overstates the moment of shallow slip.

        The area each parameter carries comes from the basis -- a triangle's own
        area for element-constant slip, a third of the node's 1-ring for a tent --
        so the moment means the same thing in either parameterization.
        """
        areas = self.basis.lumped_areas()
        if shear_modulus is None:
            shear_modulus = self._default_shear_modulus()
        elif hasattr(shear_modulus, "at"):
            shear_modulus = shear_modulus.at(self._parameter_depths(), "mu")
        mu = np.broadcast_to(np.asarray(shear_modulus, dtype=float), areas.shape)
        return float(np.sum(mu * areas * self.slip_magnitude))

    def _default_shear_modulus(self):
        """The inversion's own velocity model if it had one, else 30 GPa."""
        model = getattr(self._inversion, "velocity_model", None)
        if model is None:
            stored = self.options.get("velocity_model")
            if stored is None:
                return DEFAULT_SHEAR_MODULUS
            from .edgrn import VelocityModel

            model = VelocityModel.from_dict(stored)
        return model.at(self._parameter_depths(), "mu")

    @property
    def shear_modulus(self):
        """The rigidity this model's statistics actually use, per parameter."""
        return np.broadcast_to(np.asarray(self._default_shear_modulus(), dtype=float),
                               self.basis.lumped_areas().shape)

    def _parameter_depths(self):
        """Depth of each parameter: an element centroid, or a node."""
        if self.basis.name == "element":
            return self.mesh.centroids[:, 2]
        return self.mesh.nodes[:, 2]

    @property
    def moment_magnitude(self):
        m0 = self.moment()
        return float((2.0 / 3.0) * (np.log10(m0) - 9.1)) if m0 > 0 else float("nan")

    def track_residual(self, name):
        """One track's residual, observed minus modelled."""
        return self.residual[self.obs.track_mask(name)]

    # -- forward -----------------------------------------------------------
    @property
    def element_slip(self):
        """``(n_elements, 2)`` slip per element, whatever the basis.

        The engines are defined on elements, so this is the form anything that
        evaluates the model needs. For element-constant slip it is the parameters
        themselves; for a tent field it is each element's mean of its three nodes.
        """
        return np.column_stack([self.basis.element_values(self.strike_slip),
                                self.basis.element_values(self.dip_slip)])

    def forward(self, x, y, look=None):
        """Predict displacement at arbitrary points from this slip model."""
        engine = self._inversion.engine
        if getattr(self._inversion, "exact_basis", False):
            # The engine assembled in the basis, so hand it the coefficients.
            return engine.forward(self.mesh, self.x[:2 * self.basis.n_basis],
                                  x, y, look=look, basis=self.basis)
        return engine.forward(self.mesh, self.element_slip, x, y, look=look)

    def surface_displacement(self, spacing=1000.0, pad=50e3, bounds=None,
                             exclude_within=None, block=4096):
        """The full three-component surface displacement field this model predicts.

        Returns an :class:`xarray.Dataset` on the mesh's own local frame with
        ``ux``, ``uy`` and ``uz`` -- east, north and up in metres, positive in
        those directions. This is the model's *prediction of the ground*, not of
        any satellite: line of sight collapses three components into one, and what
        an inversion recovers is the full vector, so it is worth being able to
        look at it. Ascending and descending line-of-sight fields can be checked
        against it, and the horizontal field is what a GPS network would measure.

        ``bounds`` is ``(x_min, x_max, y_min, y_max)`` in local-frame metres;
        without it the mesh's footprint plus ``pad`` is used.

        Evaluated in blocks of ``block`` points, because the engines build a
        ``(points, 3, 2 * n_elements)`` array and a 1 km grid over a real
        footprint would ask for gigabytes of it -- 60 000 points against 1148
        elements is 3.3 GB. The block never materialises more than a few tens of
        megabytes.

        Grid points on the fault trace are a genuine singularity, so points within
        ``exclude_within`` of the surface trace come back NaN rather than
        non-finite or, worse, enormous. It defaults to the mesh's own element
        size, which is the scale below which a discretised fault does not mean
        anything anyway.
        """
        import xarray as xr

        if self.mesh.frame is None:
            raise ValueError(
                "surface_displacement needs the mesh's LocalFrame to georeference "
                "the grid; this mesh has none."
            )
        nodes = self.mesh.nodes
        if bounds is None:
            bounds = (nodes[:, 0].min() - pad, nodes[:, 0].max() + pad,
                      nodes[:, 1].min() - pad, nodes[:, 1].max() + pad)
        x_min, x_max, y_min, y_max = (float(v) for v in bounds)
        x = np.arange(x_min, x_max + spacing, spacing)
        # Descending, so the raster is north-up like every other grid here.
        y = np.arange(y_max, y_min - spacing, -spacing)
        xx, yy = np.meshgrid(x, y)

        if exclude_within is None:
            exclude_within = float(np.sqrt(2.0 * self.mesh.areas.mean()))
        # The fault outcrops along its shallowest node row; that is where a
        # dislocation solution is singular at the free surface.
        top = nodes[self.mesh.params[:, 1] == self.mesh.params[:, 1].max()]
        keep = _far_enough(xx.ravel(), yy.ravel(), top, exclude_within)

        flat = np.full((xx.size, 3), np.nan)
        index = np.nonzero(keep)[0]
        for start in range(0, index.size, int(block)):
            at = index[start:start + int(block)]
            flat[at] = self._inversion.engine.forward(
                self.mesh, self.element_slip, xx.ravel()[at], yy.ravel()[at])

        ds = xr.Dataset(
            {name: (("y", "x"), flat[:, i].reshape(xx.shape).astype(np.float32))
             for i, name in enumerate(("ux", "uy", "uz"))},
            coords={"y": y, "x": x},
        )
        for name, direction in (("ux", "east"), ("uy", "north"), ("uz", "up")):
            ds[name].attrs.update(units="m", long_name=f"surface displacement, {direction}")
        ds.attrs.update(engine=self._inversion.engine.name, basis=self.basis.name,
                        smoothing=self.smoothing, spacing=float(spacing),
                        exclude_within=float(exclude_within))
        # `local_crs`, not `crs`: this grid's x/y are local metres, and the frame's
        # bare projection would place them 500 km east and 1167 km south of where
        # they are -- which `to_grd` would then bake into a lon/lat file.
        return ds.rio.write_crs(self.mesh.frame.local_crs)

    def to_grd(self, outdir, fields=("ux", "uy", "uz"), **kwargs):
        """Write the surface displacement field as GMT ``.grd`` files.

        One single-variable grid per component, reprojected to lon/lat by the
        same :func:`nisar_tools.geo.write_grd` every other stage exports through,
        so the files drop straight into an existing GMT workflow. Extra keyword
        arguments go to :meth:`surface_displacement`.
        """
        from pathlib import Path

        from ..geo import write_grd

        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        grid = self.surface_displacement(**kwargs)

        unknown = set(fields) - set(grid.data_vars)
        if unknown:
            raise KeyError(
                f"Unknown field(s) {sorted(unknown)}; available: {sorted(grid.data_vars)}")
        return [write_grd(grid[name].rio.write_crs(self.mesh.frame.crs),
                          outdir / f"{name}.grd") for name in fields]

    # -- output ------------------------------------------------------------
    def to_dataset(self):
        """The model as an :class:`xarray.Dataset` on ``element`` and ``obs``.

        Slip is reported **per element** in either basis, so the file format does
        not change with the parameterization; the parameters themselves are kept
        alongside under ``coefficient``.
        """
        import xarray as xr

        lon, lat = (self.mesh.frame.to_lonlat(*self.mesh.centroids[:, :2].T)
                    if self.mesh.frame is not None else (None, None))
        element = self.element_slip
        data = {
            "strike_slip": ("element", element[:, 0]),
            "dip_slip": ("element", element[:, 1]),
            "slip": ("element", np.hypot(*element.T)),
            "coefficient_strike_slip": ("basis", self.strike_slip),
            "coefficient_dip_slip": ("basis", self.dip_slip),
            "area": ("element", self.mesh.areas),
            "depth": ("element", self.mesh.centroids[:, 2]),
            "along_strike": ("element", self.mesh.element_params[:, 0]),
            "strike": ("element", self.mesh.strike),
            "dip": ("element", self.mesh.dip),
            "data": ("obs", self.data),
            "prediction": ("obs", self.prediction),
            "residual": ("obs", self.residual),
        }
        if lon is not None:
            data["lon"] = ("element", np.asarray(lon))
            data["lat"] = ("element", np.asarray(lat))

        ds = xr.Dataset(data)
        ds.attrs.update(
            smoothing=self.smoothing,
            variance_reduction=self.variance_reduction,
            rms_misfit=self.rms_misfit,
            roughness=self.roughness,
            max_slip=self.max_slip,
            moment=self.moment(),
            moment_magnitude=self.moment_magnitude,
            converged=int(self.converged),
            solver_status=int(self.result.status),
            solver_iterations=int(self.result.nit),
            mesh_digest=self.mesh.digest(),
            engine=self._inversion.engine.name,
            basis=self.basis.name,
            options=self.options,
        )
        if self.ramp_labels:
            ds.attrs["ramp"] = dict(zip(self.ramp_labels, self.ramp.tolist()))
        return ds

    def to_text(self, path, shear_modulus=None):
        """Write the reference implementation's ten-column element table.

        ``element_id lon lat depth_m strike_deg dip_deg strike_slip_m dip_slip_m
        area_m2 shear_modulus_pa`` -- the format of SlipSolve's
        ``simple_triangular_model.txt``, so downstream GMT scripts work unchanged.

        Column 10 is the rigidity the inversion's own velocity model gives at each
        element's depth, or 30 GPa if it had none -- so the file records what the
        moment in it was actually computed with rather than a constant that may not
        be the one used.
        """
        if not self.converged:
            raise ValueError(
                "Refusing to write an unconverged model "
                f"(solver status {self.result.status}, {self.result.nit} iterations). "
                "Raise max_iter or increase the smoothing weight."
            )
        lon, lat = self.mesh.frame.to_lonlat(*self.mesh.centroids[:, :2].T)
        if shear_modulus is None:
            model = getattr(self._inversion, "velocity_model", None) or \
                self.options.get("velocity_model")
            if model is None:
                shear_modulus = DEFAULT_SHEAR_MODULUS
            else:
                from .edgrn import VelocityModel

                shear_modulus = (model if hasattr(model, "at")
                                 else VelocityModel.from_dict(model))
        if hasattr(shear_modulus, "at"):
            shear_modulus = shear_modulus.at(self.mesh.centroids[:, 2], "mu")
        mu = np.broadcast_to(np.asarray(shear_modulus, dtype=float),
                             (self.mesh.n_elements,))
        element = self.element_slip
        table = np.column_stack([
            np.arange(1, self.mesh.n_elements + 1), lon, lat,
            self.mesh.centroids[:, 2], self.mesh.strike, self.mesh.dip,
            element[:, 0], element[:, 1], self.mesh.areas, mu,
        ])
        header = ("element_id\tlongitude_deg\tlatitude_deg\tdepth_m\tstrike_deg\t"
                  "dip_deg\tstrike_slip_m\tdip_slip_m\tarea_m2\tshear_modulus_pa")
        np.savetxt(path, table, delimiter="\t", header=header, comments="")
        return path

    def persist(self, workspace, name=None, overwrite=False, **params):
        name = name or self.STAGE
        full = {
            "stage": name,
            "mesh": self.mesh.digest(),
            "smoothing": self.smoothing,
            "options": self.options,
            "engine": self._inversion.engine.name,
            **params,
        }
        return workspace.store(name, self.to_dataset(), full, overwrite=overwrite)

    # -- save / load -------------------------------------------------------
    def save(self, path):
        """Write the whole model to one self-contained netCDF file.

        Everything needed to work with the result later travels with it -- the
        slip vector, the fit, the mesh, and the observations -- in three netCDF
        groups (``model``, ``mesh``, ``observations``). :meth:`load` gives back a
        :class:`SlipModel` that reports every statistic, re-exports with
        :meth:`to_text`, plots, and can :meth:`forward`-model new points.

        This is the counterpart to a long run: solve under ``nohup``, save, and
        come back to it in a notebook. Unlike :meth:`persist` it needs no
        :class:`~nisar_tools.workspace.Workspace` and produces a single file that
        can just be copied off the machine that ran it.

        The container is a **zipped Zarr store** rather than netCDF: netCDF
        groups need a backend the environment does not have (only scipy's
        group-less one is installed), while ``zarr`` is already a hard dependency
        -- and its attributes are JSON, so the provenance dicts (``options``,
        ``frame``, ``quadtree``) round-trip natively instead of needing an
        encoding layer.

        The **Green's matrix is deliberately not saved** -- it is the largest
        object in the problem (hundreds of megabytes) and nothing downstream of a
        solved model needs it. A loaded model therefore cannot be re-solved at a
        new smoothing weight; rebuild the :class:`SlipInversion` for that.
        """
        import zarr

        path = str(path)
        model = self.to_dataset()
        # to_dataset() splits the parameter vector into its physical parts; keep
        # the raw vector too so a load is exact rather than reassembled.
        model["x"] = ("param", self.x)
        model.attrs["ramp_labels"] = list(self.ramp_labels)
        model.attrs["engine_nu"] = float(getattr(self._inversion.engine, "nu", 0.25))

        obs = self.obs.ds.copy()
        # Zarr has no object dtype; track names ride as fixed-width text.
        obs["track"] = obs["track"].astype(str)

        store = zarr.ZipStore(path, mode="w")
        try:
            model.to_zarr(store, group="model", mode="w", consolidated=False)
            self.mesh.to_dataset().to_zarr(
                store, group="mesh", mode="a", consolidated=False)
            obs.to_zarr(store, group="observations", mode="a", consolidated=False)
        finally:
            store.close()
        return path

    @classmethod
    def load(cls, path):
        """Read back a model written by :meth:`save`."""
        import xarray as xr
        import zarr

        from .mesh import FaultMesh
        from .sampling import Observations

        path = str(path)
        store = zarr.ZipStore(path, mode="r")
        try:
            model = xr.open_zarr(store, group="model", consolidated=False).load()
            mesh = FaultMesh.from_dataset(
                xr.open_zarr(store, group="mesh", consolidated=False).load())
            obs = Observations(
                xr.open_zarr(store, group="observations", consolidated=False).load())
        finally:
            store.close()

        options = dict(model.attrs["options"])
        self = object.__new__(cls)
        self.mesh = mesh
        self.obs = obs
        self.x = np.asarray(model["x"].values, dtype=float)
        self.result = SimpleNamespace(
            status=int(model.attrs["solver_status"]),
            nit=int(model.attrs["solver_iterations"]),
            x=self.x,
        )
        self.smoothing = float(model.attrs["smoothing"])
        self.options = options
        self.ramp_labels = [str(s) for s in model.attrs.get("ramp_labels", [])]

        self.basis = make_basis(mesh, str(model.attrs.get("basis", "element")))
        n = self.basis.n_basis
        self.strike_slip = self.x[:n]
        self.dip_slip = self.x[n:2 * n]
        self.ramp = self.x[2 * n:]
        self.data = np.asarray(model["data"].values, dtype=float)
        self.prediction = np.asarray(model["prediction"].values, dtype=float)
        # Recomputed rather than read back. It is a pure function of the two
        # fields above, so deriving it costs nothing and means a file written
        # before the residual convention changed to observed-minus-modelled loads
        # with the current sign instead of a silently mirrored one.
        self.residual = self.data - self.prediction

        # The smoothing operator is a pure function of the mesh and the options,
        # so `roughness` is recoverable exactly without storing a sparse matrix.
        smooth = (neighbor_smoothing(mesh.neighbors, options["ss_ratio"],
                                     options["ds_ratio"])
                  if self.basis.name == "element"
                  else laplace_beltrami(mesh, options["ss_ratio"], options["ds_ratio"]))
        n_ramp = self.ramp.size
        if n_ramp:
            smooth = sp.hstack([smooth, sp.csr_matrix((smooth.shape[0], n_ramp))],
                               format="csr")
        self._smooth = smooth
        # A stand-in for the inversion: `forward` needs only the engine, and the
        # Green's matrix is deliberately absent (see `save`).
        #
        # A layered engine is *not* reconstructed: its EDGRN tables are megabytes
        # and are not saved, so `forward` on a loaded layered model would silently
        # use the homogeneous half-space instead of saying so. Refuse, and name
        # what to do about it.
        engine_name = str(model.attrs.get("engine", HalfSpaceTDE.name))
        engine = (HalfSpaceTDE(float(model.attrs.get("engine_nu", 0.25)))
                  if engine_name == HalfSpaceTDE.name
                  else _MissingEngine(engine_name))
        self._inversion = SimpleNamespace(
            engine=engine, basis=self.basis,
            g=None, ramp=None, n_ramp=n_ramp, ramp_labels=self.ramp_labels,
        )
        return self

    def __repr__(self):
        flag = "" if self.converged else " UNCONVERGED"
        return (f"<SlipModel VR={self.variance_reduction:.1f}% "
                f"max_slip={self.max_slip:.2f}m Mw={self.moment_magnitude:.2f} "
                f"roughness={self.roughness:.3g}{flag}>")
