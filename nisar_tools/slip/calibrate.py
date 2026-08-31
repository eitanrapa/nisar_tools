"""Standardised synthetic recovery: what can THIS data resolve on THIS fault?

The question this answers is the one every slip inversion starts with and almost
none answers: given the scenes you actually have, their coverage, their measured
noise, and the fault geometry, **what element size, what mesh layout and what
regularisation weight should you use, and what wavelength of slip can you claim
to have resolved?**

The method is a checkerboard recovery test, run to a fixed protocol. Truth is
planted on the mesh, forward-modelled through the real look vectors and the real
coverage masks, given noise drawn from the covariance measured on the real
residuals, and inverted with the same estimator that will be used in anger. Error
is then measured against something known instead of against another model.

Seven rules make the difference between this and a test that looks fine and means
nothing. Every one of them was learned by getting it wrong first.

1. **The target must be reachable under the bounds.** A textbook checkerboard
   alternates sign; a real inversion usually constrains slip to the fault's known
   sense. Planting one inside the other measures the *bound*, not the resolution
   -- and every summary statistic stays plausible while it does. So the target
   here alternates in **magnitude** about a base, with its sign taken from
   ``bounds["polarity"]``, and :func:`plant_checkerboard` refuses to return a
   target that violates it.
2. **Score the oscillating part.** A uniform offset is recovered trivially and
   would flatter every wavelength equally, which is the one thing a resolution
   ladder must not do. Each field has its own mean removed before scoring.
3. **Noise is mandatory, and drawn from the measured covariance.** A noiseless
   test measures only the *bias* half of the error, and therefore gets the
   optimal damping badly wrong: measured on real six-scene geometry, the L-curve
   corner sat at effective rank 21 while total error against truth was still
   falling at rank 79. Several realisations, not one.
4. **Compare at matched effective rank, never at matched lambda.** Two meshes at
   one lambda are damped by different amounts, which is the confound that makes
   mesh comparisons look meaningful when they are not.
5. **Check the discretisation ceiling before believing a failure.** If the mesh
   cannot carry the target, the test says nothing about resolution.
   :func:`discretisation_ceiling` is pure geometry and costs milliseconds; run it
   first and drop any (budget, wavelength) pair that fails it.
6. **The coverage mask reaches the forward model through the LOOK VECTORS**, not
   through the LOS field. Blanking only ``los`` leaves a "full coverage" world
   byte-identical to the masked one.
7. **Solve the ramp jointly, as production does.** A per-track polynomial absorbs
   long-wavelength correlated noise, which is most of what atmospheric noise is,
   and leaving it out changes the optimal rank.

Cost is managed by **staging rather than a factorial**. The rungs of a rank
ladder are nearly free -- they re-solve on a factorisation that already exists --
while every wavelength, budget, layout and seed needs its own sampling-and-mesh
loop. Since the dominant result across four independent studies is that lambda
determines the model and the mesh does not, the search is ordered to exploit
that: find the wavelength and rank first on one cheap configuration, confirm with
seeds, and only then check that budget and layout really do not matter.

    stage 0   discretisation ceiling        pure geometry, seconds
    stage 1   wavelengths x rank ladder     one budget, one layout, one seed
    stage 2   seeds at the best wavelength  spread on the optimum
    stage 3   budgets x layouts             confirm they are second-order

Thirteen loops instead of the ninety a full factorial would need.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "CheckerboardTarget",
    "DEFAULT_CELLS",
    "DEFAULT_RANKS",
    "bound_sign",
    "discretisation_ceiling",
    "plant_checkerboard",
    "recovery_score",
    "resolving_wavelength",
]

#: Checkerboard cell sizes to plant, metres. A ladder rather than one size: a
#: single wavelength cannot distinguish "the data cannot see this" from "the
#: regulariser threw it away", because both look like a low score. Bracketing the
#: expected resolving length is what separates them -- one rung should fail at
#: every damping, one should recover, and the crossing is the answer.
DEFAULT_CELLS = (20e3, 40e3, 60e3)

#: Effective ranks to solve at. Spans roughly an L-curve corner (the low end) to
#: well past it, because the corner and the minimum-error point are **not** the
#: same rank once truth is known.
DEFAULT_RANKS = (10.0, 21.0, 40.0, 80.0, 130.0, 200.0)

#: Amplitude a mesh must retain for a recovery failure at that wavelength to be
#: attributable to resolution rather than to discretisation.
CEILING_THRESHOLD = 0.95

#: Correlation on the oscillating part above which a wavelength counts as
#: resolved. 0.5 is a convention, not a measurement -- it is roughly where the
#: recovered pattern stops being readable by eye -- so it is exposed as an
#: argument everywhere it is used.
RESOLVED_CORRELATION = 0.5


def bound_sign(bounds, component=0):
    """The sign the estimator forces on a slip component, or 0 if it is free.

    Reads ``bounds["polarity"]``, the same structure
    :meth:`~nisar_tools.slip.inversion.SlipInversion.solve` takes. This is the
    single fact a synthetic target must respect: plant the other sign and the
    solver cannot reach it, so the recovery score measures the constraint.
    """
    polarity = (bounds or {}).get("polarity")
    if polarity is None:
        return 0
    return int(np.sign(polarity[component]))


@dataclass
class CheckerboardTarget:
    """A planted slip field plus everything needed to interpret its recovery."""

    slip: np.ndarray                  #: ``(2 * n_basis,)`` at the mesh nodes
    cell: float                       #: cell size, metres
    sign: int                         #: the polarity it was built to respect
    base: float                       #: uniform component, metres
    swing: float                      #: oscillating amplitude, metres
    attrs: dict = field(default_factory=dict)

    @property
    def strike(self):
        return self.slip[: self.slip.size // 2]

    def unreachable_fraction(self, sign=None):
        """Fraction of the target the polarity bound forbids. Must be 0."""
        sign = self.sign if sign is None else sign
        if not sign:
            return 0.0
        return float(np.mean(sign * self.strike < 0.0))


def plant_checkerboard(mesh, cell, bounds=None, base=1.25, swing=1.0,
                       component=0):
    """A magnitude checkerboard the bounded estimator is allowed to recover.

    ``strike = sign * (base + swing * sin(pi s / cell) * sin(pi z / cell))``

    so the field alternates between ``base - swing`` and ``base + swing`` in
    magnitude while never changing sign. With the defaults that is a factor of
    nine between weak and strong cells -- a strong probe -- entirely inside a
    polarity constraint.

    ⚠️ **Do not "fix" this into a sign-alternating checkerboard.** That is the
    textbook form and it is wrong here whenever ``bounds`` carries a polarity:
    43.8% of such a target fell outside the bound on the run that motivated this
    module, every recovered model came back single-signed at all 18 000 probe
    points, and the scores looked entirely reasonable. If the estimator really is
    sign-free (``polarity`` absent), ``sign`` is 0 and the classic form is
    recovered by passing ``base=0``.

    Slip is planted at the mesh **nodes** and the dip-slip block is zero, so the
    vector is ``(2 * n_nodes,)``. A node basis is assumed; pass an element-basis
    mesh and the length will not match what the forward model wants.
    """
    if swing < 0:
        raise ValueError("swing is an amplitude and must be non-negative")
    sign = bound_sign(bounds, component)
    if sign and base < swing:
        raise ValueError(
            f"base={base} below swing={swing} would cross zero, which the "
            f"polarity bound {sign:+d} forbids; raise base or lower swing"
        )

    s, z = mesh.params[:, 0], mesh.params[:, 1]
    wave = np.sin(np.pi * s / cell) * np.sin(np.pi * z / cell)
    strike = (sign or 1) * (base + swing * wave) if sign else swing * wave
    target = CheckerboardTarget(
        slip=np.concatenate([strike, np.zeros(mesh.n_nodes)]),
        cell=float(cell), sign=sign, base=float(base), swing=float(swing),
        attrs={"n_nodes": int(mesh.n_nodes)})

    bad = target.unreachable_fraction()
    if bad:
        raise ValueError(
            f"{100 * bad:.1f}% of this target has the sign the polarity bound "
            f"forbids; its recovery would measure the bound, not the resolution"
        )
    return target


def discretisation_ceiling(mesh, cells=DEFAULT_CELLS, probe=None,
                           threshold=CEILING_THRESHOLD):
    """How much of each wavelength the mesh can carry at all. Stage 0.

    Returns one row per cell size with the amplitude and pattern correlation that
    survive being sampled at the mesh's nodes -- the **ceiling** any inversion on
    that mesh is working under. A wavelength that fails here cannot be used to
    say anything about resolution, because the parameterisation lost it before
    the regulariser got a chance to.

    Pure geometry: no data, no Green's functions, no solve. Run it before
    committing hours to a recovery study.
    """
    from scipy.spatial import cKDTree

    from .predict import _probe_lattice

    probe = _probe_lattice(mesh, n_along=300, n_down=60) if probe is None else probe
    nearest = cKDTree(mesh.params).query(probe)[1]
    ps, pz = probe[:, 0], probe[:, 1]
    s, z = mesh.params[:, 0], mesh.params[:, 1]

    # Node spacing along strike is the tight axis on a fault that is far longer
    # than it is deep, so it is what decides whether a wavelength is sampled.
    columns = np.unique(np.round(s, -2)).size
    spacing = float(s.max() / max(columns - 1, 1))

    rows = []
    for cell in cells:
        exact = np.sin(np.pi * ps / cell) * np.sin(np.pi * pz / cell)
        held = (np.sin(np.pi * s / cell) * np.sin(np.pi * z / cell))[nearest]
        rows.append({
            "cell": float(cell),
            "samples_per_cell": float(cell / spacing),
            "amplitude_retained": float(np.std(held) / np.std(exact)),
            "pattern_correlation": float(np.corrcoef(held, exact)[0, 1]),
            "usable": bool(np.std(held) / np.std(exact) >= threshold
                           and cell / spacing >= 2.0),
        })
    return rows


def recovery_score(got, want):
    """Compare a recovered field to truth, on the part that carries information.

    ⚠️ **Both fields have their own mean removed first.** The uniform component
    of a magnitude checkerboard is recovered by any inversion at any damping, so
    scoring the raw field reports mostly that -- measured, a target whose pattern
    correlation was 0.03 still returned an amplitude ratio of 0.88. Amplitude
    ratio on the raw field is not a recovery statistic.

    ``rel_error`` is the honest headline: rms error over rms truth, so 1.0 means
    the estimate is no better than returning the mean.
    """
    got = np.asarray(got, dtype=float)
    want = np.asarray(want, dtype=float)
    go, wo = got - got.mean(), want - want.mean()
    err = got - want
    scale = max(float(np.sqrt(np.mean(want ** 2))), 1e-12)
    return {
        "correlation": float(np.corrcoef(go, wo)[0, 1]),
        "amplitude_ratio": float(np.std(go) / max(np.std(wo), 1e-12)),
        "rms_error": float(np.sqrt(np.mean(err ** 2))),
        "rel_error": float(np.sqrt(np.mean(err ** 2)) / scale),
        "raw_min": float(got.min()),
        "raw_max": float(got.max()),
    }


def resolving_wavelength(table, rank=None, threshold=RESOLVED_CORRELATION):
    """Smallest planted cell that clears ``threshold``, interpolated in log cell.

    ``table`` is a sequence of mappings with ``cell``, ``correlation`` and
    ``rank``. When ``rank`` is given, only that rung is read; otherwise the best
    rung per cell is used, which reports what the geometry can do rather than
    what one damping choice achieved.

    Returns ``None`` when no wavelength clears the threshold -- an honest answer,
    and the one that should stop a paper claiming a resolved rupture length.
    """
    rows = [r for r in table if rank is None or abs(r["rank"] - rank) < 1e-6]
    if not rows:
        return None
    best = {}
    for r in rows:
        cell = float(r["cell"])
        best[cell] = max(best.get(cell, -np.inf), float(r["correlation"]))
    cells = np.array(sorted(best))
    corr = np.array([best[c] for c in cells])
    if not (corr >= threshold).any():
        return None
    if corr[0] >= threshold:
        return float(cells[0])
    i = int(np.argmax(corr >= threshold))
    # Linear in log cell size between the straddling rungs.
    lo, hi = cells[i - 1], cells[i]
    c0, c1 = corr[i - 1], corr[i]
    frac = (threshold - c0) / max(c1 - c0, 1e-12)
    return float(np.exp(np.log(lo) + frac * (np.log(hi) - np.log(lo))))


#: Default search space. Budgets bracket the point where the discretisation
#: ceiling is reached; layouts are the two :func:`~.resolution.adapt_layout`
#: structures; seeds are noise realisations of the same target.
DEFAULT_BUDGETS = (330, 660, 1320)
DEFAULT_LAYOUTS = ("rows", "unstructured")
DEFAULT_SEEDS = 5


@dataclass
class CalibrationCase:
    """One sampling-and-mesh loop, plus the rungs to solve on it."""

    stage: int
    cell: float
    budget: int
    layout: str
    seed: int
    ranks: tuple
    why: str

    @property
    def name(self):
        return ("s%d_%.0fkm_b%d_%s_seed%d"
                % (self.stage, self.cell / 1e3, self.budget, self.layout,
                   self.seed))


def calibration_plan(mesh, cells=DEFAULT_CELLS, budgets=DEFAULT_BUDGETS,
                     layouts=DEFAULT_LAYOUTS, ranks=DEFAULT_RANKS,
                     seeds=DEFAULT_SEEDS, threshold=CEILING_THRESHOLD):
    """The staged search, as an explicit list of loops to run.

    Returned rather than executed so the cost is visible before it is spent, and
    so a caller can drop stages, reorder them, or run them on different machines.
    Every entry is one ``iterate_sampling`` loop; the ``ranks`` on it are almost
    free by comparison, since they re-solve on a factorisation that already
    exists.

    **Why staged and not factorial.** A full grid is
    ``cells x budgets x layouts x seeds`` loops -- 90 at the defaults, roughly
    a day. The staging exploits the study's own dominant finding, that lambda
    determines the model and the mesh does not, by settling wavelength and rank
    first on one cheap configuration and only then testing whether budget and
    layout move the answer. If stage 3 ever shows they do, the ordering is
    wrong for that dataset and the plan should be widened -- which is itself a
    reportable result.

    Stage 0 runs here, in this function: wavelengths the mesh cannot carry are
    dropped before any of them costs a loop, and the reason is recorded.
    """
    ceiling = {r["cell"]: r for r in
               discretisation_ceiling(mesh, cells, threshold=threshold)}
    usable = [c for c in cells if ceiling[float(c)]["usable"]]
    dropped = [c for c in cells if not ceiling[float(c)]["usable"]]

    base_budget = budgets[len(budgets) // 2]
    base_layout = layouts[-1]
    plan = []

    # Stage 1 -- wavelength and rank, one configuration, one realisation.
    for cell in usable:
        plan.append(CalibrationCase(
            1, float(cell), int(base_budget), base_layout, 1, tuple(ranks),
            "wavelength ladder at the reference configuration"))

    # Stage 2 -- is the stage-1 optimum a realisation or a property? The cell is
    # resolved at run time from stage 1's result, so it is left as None here and
    # filled by the driver; the count is what the plan needs to price.
    for k in range(2, int(seeds) + 1):
        plan.append(CalibrationCase(
            2, float("nan"), int(base_budget), base_layout, k, tuple(ranks),
            "noise realisation, at stage 1's best-resolved wavelength"))

    # Stage 3 -- confirm budget and layout are second-order, at that wavelength.
    for budget in budgets:
        for layout in layouts:
            if budget == base_budget and layout == base_layout:
                continue          # stage 1 already ran this one
            plan.append(CalibrationCase(
                3, float("nan"), int(budget), layout, 1, tuple(ranks),
                "budget x layout, expected second-order"))

    return {
        "cases": plan,
        "ceiling": [ceiling[float(c)] for c in cells],
        "dropped_cells": [float(c) for c in dropped],
        "n_loops": len(plan),
        "n_solves": sum(len(c.ranks) for c in plan),
        "factorial_loops": len(usable) * len(budgets) * len(layouts) * int(seeds),
    }


class _CarrierResult:
    """Stand-in for a solver result on a model that was planted, not solved."""

    status, nit, success = 1, 0, True


def carrier_model(inversion, slip):
    """Wrap a planted slip vector as a :class:`SlipModel` for forward use.

    :func:`~nisar_tools.slip.predict.predicted_los` takes a model, not a vector,
    so a planted target has to be dressed as one. Only the mesh, the parameter
    vector and the engine are read on that path; the solver fields are
    placeholders and **no fit statistic on this object means anything**.
    """
    from .inversion import SlipModel

    slip = np.asarray(slip, dtype=float)
    x = np.concatenate([slip, np.zeros(inversion.n_ramp)])
    return SlipModel(inversion, x, _CarrierResult(), 0.0, None, {"max_iter": 1})


def unmask_geometry(stack):
    """The same scene with full coverage, by filling its LOOK VECTORS.

    ⚠️ **Not by filling ``los``.** The forward model projects slip onto
    ``los_east/north/up``, so a scene's coverage reaches it through the geometry;
    blanking or filling only ``los`` leaves a "full coverage" world identical to
    the masked one. Measured when this was first written: both came out at 24%
    coverage and the controlled pair was silently a no-op.
    """
    from .predict import _fill_nearest

    ds = stack.ds.copy()
    for component in ("los_east", "los_north", "los_up"):
        ds[component] = (ds[component].dims,
                         _fill_nearest(np.asarray(ds[component].values, float)))
    return type(stack)(ds)


def synthesize(target, scenes, inversion, frame, spacing=2000.0,
               full_coverage=False):
    """Replace every scene's ``los`` with ``target``'s forward model.

    Keeps each scene's real look vectors, real grid and (unless ``full_coverage``)
    real coverage mask, so the synthetic differs from the observation only in what
    produced the displacement. Returns ``{name: stack}`` of the same type it was
    given -- the stack is duck-typed on ``.ds`` and a one-argument constructor,
    so this does not require a NISAR ``LOSStack`` specifically.
    """
    from .predict import predicted_los

    model = carrier_model(inversion, target.slip)
    out = {}
    for name, stack in scenes.items():
        source = unmask_geometry(stack) if full_coverage else stack
        values = predicted_los(model, source, frame, spacing=spacing)
        ds = source.ds.copy()
        ds["los"] = (("pair", "y", "x"), values[None].astype(np.float32))
        out[name] = type(stack)(ds)
    return out


def add_noise(scenes, covariances, seed=0):
    """Add a correlated realisation to each scene, from its measured covariance.

    ⚠️ Noise is added **only where the scene already has data**. Adding it to the
    masked region would create observations the real product does not have,
    quietly undoing the coverage the synthetic exists to reproduce.

    ``covariances`` is ``{name: CovarianceModel}``, normally the output of
    :func:`~nisar_tools.slip.covariance.observation_covariance`'s per-scene fit on
    the *real* residuals -- which is the point: the noise is measured, not assumed.
    """
    from .covariance import correlated_field

    rng = np.random.default_rng(seed)
    out = {}
    for name, stack in scenes.items():
        cov = covariances[name]
        ds = stack.ds.copy()
        los = np.asarray(ds["los"].values[0], dtype=float)
        spacing = float(abs(np.diff(np.asarray(ds["x"].values))[0]))
        field = correlated_field(los.shape, spacing, cov,
                                 seed=int(rng.integers(1 << 31)))
        finite = np.isfinite(los)
        noisy = los.copy()
        noisy[finite] += field[finite]
        ds["los"] = (("pair", "y", "x"), noisy[None].astype(np.float32))
        out[name] = type(stack)(ds)
    return out


def truth_on_probe(mesh, target, probe):
    """The planted field at the probe points, by the same rule the estimate uses.

    :func:`~nisar_tools.slip.predict._probe_slip` reads the recovered field by
    nearest node, so truth must be read the same way or the comparison charges
    the inversion for an interpolation difference.
    """
    from scipy.spatial import cKDTree

    strike = target.strike if isinstance(target, CheckerboardTarget) else target
    return np.asarray(strike, float)[cKDTree(mesh.params).query(probe)[1]]


def run_case(case, scenes, mesh0, trace, frame, sample_kwargs, inversion_kwargs,
             solve_kwargs, geometry, covariances=None, spacing=2000.0,
             full_coverage=False, probe=None, max_rounds=3, verbose=True):
    """One loop of the plan: plant, forward-model, invert, score every rung.

    Returns one row per rank rung. The expensive part is the single
    :func:`~nisar_tools.slip.predict.iterate_sampling` loop; the rungs after it
    re-solve on a factorisation that already exists, which is what makes a ladder
    affordable and a factorial not.
    """
    import time

    from .inversion import SlipInversion
    from .predict import _probe_lattice, _probe_slip, iterate_sampling
    from . import resolution as R

    started = time.time()
    bounds = {k: v for k, v in solve_kwargs.items()
              if k in ("polarity", "strike", "dip")}
    target = plant_checkerboard(mesh0, case.cell, bounds)
    probe = _probe_lattice(mesh0, 300, 60) if probe is None else probe

    reference = SlipInversion(mesh0, _any_observations(
        scenes, trace, frame, mesh0, sample_kwargs), **inversion_kwargs)
    world = synthesize(target, scenes, reference, frame, spacing=spacing,
                       full_coverage=full_coverage)
    if covariances is not None:
        world = add_noise(world, covariances, seed=case.seed)

    adapt = {"geometry": geometry, "budget": int(case.budget),
             "rounds": max(1, max_rounds - 1)}
    if case.layout != "rows":
        adapt["structure"] = case.layout
    obs, model, _ = iterate_sampling(
        world, mesh0, trace, frame, sample_kwargs,
        inversion_kwargs=inversion_kwargs, solve_kwargs=solve_kwargs,
        max_rounds=max_rounds, verbose=False, mesh_adapt=adapt,
        sampling_field="both")

    inversion = SlipInversion(model.mesh, obs, **inversion_kwargs)
    grid = np.logspace(-1.5, 4, 28)
    achieved = [R.resolution_diagonal(inversion, float(w)).attrs["effective_rank"]
                for w in grid]
    want = truth_on_probe(mesh0, target, probe)

    rows = []
    for rung in case.ranks:
        if not (min(achieved) <= rung <= max(achieved)):
            continue
        lam = float(np.interp(rung, achieved[::-1], grid[::-1]))
        # More parameters come off their bounds at low lambda, so the trust
        # region needs a longer leash; a fixed budget returns partial answers
        # up here that still score plausibly.
        options = dict(solve_kwargs)
        options["max_iter"] = max(options.get("max_iter", 300),
                                  900 if rung > 100 else 300)
        solved = inversion.solve(smoothing=lam, **options)
        got = _probe_slip(model.mesh, solved, probe)[:, 0]
        row = dict(stage=case.stage, cell=case.cell, budget=int(case.budget),
                   layout=case.layout, seed=case.seed, rank=float(
                       R.resolution_diagonal(inversion, lam).attrs["effective_rank"]),
                   smoothing=lam, n_obs=int(obs.n),
                   n_elements=int(model.mesh.n_elements),
                   converged=bool(solved.converged),
                   max_ramp_m=float(np.abs(solved.ramp).max())
                   if solved.ramp.size else 0.0,
                   seconds=round(time.time() - started, 1),
                   **recovery_score(got, want))
        rows.append(row)
        if verbose:
            print("  %-28s rank %6.1f  corr %.4f  rel err %.4f%s"
                  % (case.name, row["rank"], row["correlation"],
                     row["rel_error"], "" if row["converged"] else "  UNCONVERGED"),
                  flush=True)
    return rows


def _any_observations(scenes, trace, frame, mesh, sample_kwargs):
    """One scene sampled, purely to give the forward model a design to build on.

    The truth is planted on the mesh, not fitted to these points, so which scene
    it is does not matter -- only that the inversion has somewhere to hang its
    engine and basis.
    """
    from .sampling import Observations

    name, stack = next(iter(scenes.items()))
    return Observations.from_los(stack, name=name, frame=frame, trace=trace,
                                 **sample_kwargs[name])
