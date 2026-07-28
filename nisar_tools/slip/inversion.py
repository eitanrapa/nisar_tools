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

import numpy as np
import scipy.sparse as sp
from scipy.optimize import lsq_linear
from scipy.sparse.linalg import LinearOperator

from .greens import HalfSpaceTDE
from .regularize import (
    neighbor_smoothing,
    ramp_columns,
    slip_bounds,
    zero_slip_boundary,
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


class SlipInversion:
    """A mesh, a set of observations, and the Green's matrix relating them.

    The Green's matrix is built once, on construction, and reused across every
    smoothing weight -- which is what makes an L-curve sweep cheap.
    """

    def __init__(self, mesh, obs, engine=None, ramp="none"):
        self.mesh = mesh
        self.obs = obs
        self.engine = engine or HalfSpaceTDE()

        if mesh.frame is not None and obs.ds.attrs.get("frame") is not None:
            mesh.frame.require_match(obs.ds.attrs["frame"], "Observations")

        ds = obs.ds
        self.d = np.asarray(ds["los"].values, dtype=float)
        self.w = np.asarray(ds["weight"].values, dtype=float)
        self.g = self.engine.los_matrix(
            mesh,
            ds["x"].values, ds["y"].values,
            ds["look_e"].values, ds["look_n"].values, ds["look_u"].values,
        )

        self.ramp, self.ramp_labels = ramp_columns(obs, ramp)
        self.n_slip = 2 * mesh.n_elements
        self.n_ramp = self.ramp.shape[1]

    @property
    def n_param(self):
        return self.n_slip + self.n_ramp

    # -- solving -----------------------------------------------------------
    def solve(self, smoothing=0.3, ss_ratio=1.0, ds_ratio=3.0,
              boundary_ratio=1.0, sides=("bottom", "left", "right"),
              polarity=None, strike=(-10.0, 10.0), dip=(-10.0, 10.0),
              max_iter=DEFAULT_MAX_ITER, tol=1e-10, verbose=0):
        """Solve for slip at one smoothing weight; returns a :class:`SlipModel`."""
        smooth = neighbor_smoothing(self.mesh.neighbors, ss_ratio, ds_ratio)
        boundary = zero_slip_boundary(self.mesh, sides, boundary_ratio)
        lo, hi = slip_bounds(self.mesh.n_elements, strike, dip, polarity)

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
            lsq_solver="lsmr", max_iter=max_iter, tol=tol, verbose=verbose,
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
        """
        import xarray as xr

        values = np.sort(np.asarray(smoothing_values, dtype=float))[::-1]
        models, rows = [], []
        for value in values:
            model = self.solve(smoothing=value, **kwargs)
            models.append(model)
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

        n = self.mesh.n_elements
        self.strike_slip = self.x[:n]
        self.dip_slip = self.x[n:2 * n]
        self.ramp = self.x[2 * n:]
        self.ramp_labels = inversion.ramp_labels

        design = (np.hstack([inversion.g, inversion.ramp])
                  if inversion.n_ramp else inversion.g)
        self.prediction = design @ self.x
        self.data = inversion.d
        self.residual = self.prediction - self.data

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

    def moment(self, shear_modulus=DEFAULT_SHEAR_MODULUS):
        """Scalar seismic moment, ``sum(mu * area * slip)``, in newton-metres."""
        mu = np.broadcast_to(np.asarray(shear_modulus, dtype=float), (self.mesh.n_elements,))
        return float(np.sum(mu * self.mesh.areas * self.slip_magnitude))

    @property
    def moment_magnitude(self):
        m0 = self.moment()
        return float((2.0 / 3.0) * (np.log10(m0) - 9.1)) if m0 > 0 else float("nan")

    def track_residual(self, name):
        return self.residual[self.obs.track_mask(name)]

    # -- forward -----------------------------------------------------------
    def forward(self, x, y, look=None):
        """Predict displacement at arbitrary points from this slip model."""
        return self._inversion.engine.forward(self.mesh, self.x[:2 * self.mesh.n_elements],
                                              x, y, look=look)

    # -- output ------------------------------------------------------------
    def to_dataset(self):
        """The model as an :class:`xarray.Dataset` on ``element`` and ``obs``."""
        import xarray as xr

        lon, lat = (self.mesh.frame.to_lonlat(*self.mesh.centroids[:, :2].T)
                    if self.mesh.frame is not None else (None, None))
        data = {
            "strike_slip": ("element", self.strike_slip),
            "dip_slip": ("element", self.dip_slip),
            "slip": ("element", self.slip_magnitude),
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
            options=self.options,
        )
        if self.ramp_labels:
            ds.attrs["ramp"] = dict(zip(self.ramp_labels, self.ramp.tolist()))
        return ds

    def to_text(self, path, shear_modulus=DEFAULT_SHEAR_MODULUS):
        """Write the reference implementation's ten-column element table.

        ``element_id lon lat depth_m strike_deg dip_deg strike_slip_m dip_slip_m
        area_m2 shear_modulus_pa`` -- the format of SlipSolve's
        ``simple_triangular_model.txt``, so downstream GMT scripts work unchanged.
        """
        if not self.converged:
            raise ValueError(
                "Refusing to write an unconverged model "
                f"(solver status {self.result.status}, {self.result.nit} iterations). "
                "Raise max_iter or increase the smoothing weight."
            )
        lon, lat = self.mesh.frame.to_lonlat(*self.mesh.centroids[:, :2].T)
        mu = np.broadcast_to(np.asarray(shear_modulus, dtype=float),
                             (self.mesh.n_elements,))
        table = np.column_stack([
            np.arange(1, self.mesh.n_elements + 1), lon, lat,
            self.mesh.centroids[:, 2], self.mesh.strike, self.mesh.dip,
            self.strike_slip, self.dip_slip, self.mesh.areas, mu,
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

    def __repr__(self):
        flag = "" if self.converged else " UNCONVERGED"
        return (f"<SlipModel VR={self.variance_reduction:.1f}% "
                f"max_slip={self.max_slip:.2f}m Mw={self.moment_magnitude:.2f} "
                f"roughness={self.roughness:.3g}{flag}>")
