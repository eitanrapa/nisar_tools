"""What the data can actually resolve, and the mesh and sampling that follow.

Variance reduction measures *fit*, and a slip model can fit beautifully on a
discretisation the data never constrained -- the extra freedom is simply spent
on whatever the noise asks for. Model resolution measures the other thing: how
much of each parameter's estimate comes from the data rather than from the
regulariser. Two long-standing recipes fall straight out of it, and this module
is both, sharing one matrix:

* **Mesh from resolution** (Barnhart & Lohman 2010): make the elements large
  where resolution is poor and small where it is good, so no parameter is
  either starved or free. :func:`target_element_size` and :func:`adapt_layout`.
* **Sampling from resolution** (Lohman & Simons 2005): put the observations
  where they most improve resolution. :func:`resolution_gain` returns a field
  that :meth:`~nisar_tools.slip.sampling.Observations.from_los` already accepts
  as its ``field=`` argument, so no new sampler is needed -- only a different
  thing to quadtree.

Using one quantity for both is the point. Alternating a resolution-sized mesh
against a Wang & Fialko predicted-field sampling is two heuristics in different
currencies; driving both from ``R`` is one objective.

⚠️ **Two caveats, both real, both in the callers' way if ignored.**

1. :meth:`~nisar_tools.slip.inversion.SlipInversion.solve` is **bound
   constrained**, and the Venezuela run had 86% of elements sitting on a bound.
   ``R`` describes the *unbounded* damped problem, which is a different problem.
   Treat these numbers as a discretisation heuristic, never as a statement about
   uncertainty in the model that was actually solved.
2. ``R`` depends on the smoothing weight, so it must be evaluated at the weight
   the run uses; a resolution computed at the wrong lambda sizes the mesh for a
   different inversion.

⚠️ **There is deliberately no cheap column-norm stand-in for ``R``, and that is
measured.** ``||G[:, k]||`` -- how strongly the data sees element ``k`` -- is the
obvious way to avoid a factorisation, and it does not work: on the two-scene
synthetic fixture its correlation with depth is **-0.036** against the true
resolution's **-0.742**, and it stays flat at ~0.003 from the surface to 20 km
while resolution falls from 0.34 to 2e-6. A deeper element radiates a broader but
weaker pattern whose norm is roughly conserved; resolution decays because deep
elements *trade off with each other*, and a column norm structurally cannot see a
trade-off. Nothing is lost by removing it either -- the premise that round zero
has no smoothing weight to evaluate ``R`` at was wrong, since the loop's own
``solve_kwargs`` always carries one.

A third assumption is worth naming rather than leaving implicit: the data
covariance here is ``sigma**2 I`` modulated by the per-track weights
:class:`~nisar_tools.slip.sampling.Observations` already carries. Lohman & Simons'
other contribution is a full ``Cd`` estimated from the residual variogram, which
would enter as ``G' Cd^-1 G`` throughout. That is not built.
"""

import warnings

import numpy as np
import scipy.sparse as sp
import xarray as xr
from scipy.linalg import cho_factor, cho_solve

from .layout import (
    MAX_COUNT_RATIO,
    MeshLayout,
    lloyd_layout,
    require_boundaries,
    triangulate,
)

#: Ridge added to the normal equations before factorising, relative to the mean
#: diagonal. Only to keep the Cholesky well posed when a parameter is completely
#: unconstrained -- far below the smoothing term for anything the data sees.
JITTER = 1e-10

#: Grid the resolution-gain field is evaluated on, in metres. Deliberately much
#: coarser than :data:`~nisar_tools.slip.predict.DEFAULT_SPACING`: the gain is a
#: smooth sum of squared Green's functions, and it is only ever used to *rank*
#: locations, while the cost is a full design matrix at every probe point. On the
#: real Venezuela crop at 2 km that is a 30 580 x 3392 matrix -- 830 MB and two
#: minutes, per scene per round; at 5 km it is 134 MB and twenty seconds.
GAIN_SPACING = 5000.0


def weighted_design(inversion):
    """``Cd^-1/2 [G | ramp]`` -- the design matrix the solver actually sees.

    Extracted because three functions here need it verbatim
    (:func:`_normal_equations`, :func:`data_importance`,
    :func:`resolution_report`) and because ``trace(R) == trace(N)`` holds **only
    if all three whiten the same ``A``**. When the whitening changed and one site
    was missed, that identity is what caught it -- so it is worth having exactly
    one place the change can be made.
    """
    design = (np.hstack([inversion.g, inversion.ramp]) if inversion.n_ramp
              else inversion.g)
    return inversion.whiten(design)


def _normal_equations(inversion, smoothing, ss_ratio=1.0, ds_ratio=3.0,
                      boundary_ratio=1.0, sides=("bottom", "left", "right"),
                      smooth_weighting="uniform"):
    """``(data, damping)`` normal-equation blocks for the solve's own operator.

    Assembled from exactly what :meth:`SlipInversion.solve` stacks -- weighted
    design, scaled smoothing, boundary rows -- so the resolution describes the
    inversion being run rather than a tidier relative of it.
    """
    from .regularize import laplace_beltrami, neighbor_smoothing

    if inversion.basis.name == "element":
        smooth = neighbor_smoothing(inversion.mesh.neighbors, ss_ratio, ds_ratio,
                                    weighting=smooth_weighting, mesh=inversion.mesh)
    else:
        smooth = laplace_beltrami(inversion.mesh, ss_ratio, ds_ratio)
    boundary = _boundary_rows(inversion.basis, sides, boundary_ratio)

    n_ramp = inversion.n_ramp
    if n_ramp:
        smooth = sp.hstack([smooth, sp.csr_matrix((smooth.shape[0], n_ramp))],
                           format="csr")
        boundary = sp.hstack([boundary, sp.csr_matrix((boundary.shape[0], n_ramp))],
                             format="csr")
    smooth = smooth * (float(smoothing) / max(1, smooth.shape[0]))

    weighted = weighted_design(inversion)

    data = weighted.T @ weighted
    damping = (smooth.T @ smooth + boundary.T @ boundary).toarray()
    return data, damping


def _boundary_rows(basis, sides, ratio):
    from .inversion import _boundary_rows as rows
    return rows(basis, sides, ratio)


def _factorized(inversion, smoothing, ss_ratio, ds_ratio, boundary_ratio, sides,
                smooth_weighting):
    """``(data, cholesky)`` of the damped normal equations, ready to solve against.

    Factored out because the model resolution matrix, its diagonal and the data
    importance are three readings of the *same* factorization, and at production
    size that factorization is the whole cost.
    """
    data, damping = _normal_equations(inversion, smoothing, ss_ratio, ds_ratio,
                                      boundary_ratio, sides, smooth_weighting)
    total = data + damping
    total.flat[:: total.shape[0] + 1] += JITTER * np.trace(total) / total.shape[0]
    return data, cho_factor(total, lower=True)


def resolution_matrix(inversion, smoothing, ss_ratio=1.0, ds_ratio=3.0,
                      boundary_ratio=1.0, sides=("bottom", "left", "right"),
                      smooth_weighting="uniform"):
    """The **full** model resolution matrix ``R``, not just its diagonal.

    Row ``k`` is the averaging kernel of parameter ``k``: the weights with which
    the *true* model contributes to its estimate. The diagonal says how much of
    the estimate is the parameter's own value;
    :func:`resolution_kernels` reads the off-diagonal structure, which is where
    the actual spatial resolution lives -- a parameter can have a respectable
    diagonal and still be an average over 40 km of fault.

    ``(n_param, n_param)`` float64: 35 MB at the production 1035-node mesh, and
    :func:`resolution_diagonal` was already forming it internally and discarding
    everything off the diagonal, so this costs nothing extra.
    """
    data, chol = _factorized(inversion, smoothing, ss_ratio, ds_ratio,
                             boundary_ratio, sides, smooth_weighting)
    return cho_solve(chol, data)


def resolution_diagonal(inversion, smoothing, ss_ratio=1.0, ds_ratio=3.0,
                        boundary_ratio=1.0, sides=("bottom", "left", "right"),
                        smooth_weighting="uniform"):
    """Per-parameter model resolution, the diagonal of ``R``.

    ``R = (A'A + S'S + B'B)^-1 A'A`` with ``A`` the weighted design matrix and
    ``S``/``B`` the smoothing and boundary blocks -- i.e. the resolution of the
    damped problem :meth:`SlipInversion.solve` sets up, minus its bounds (see
    this module's caveats). ``1`` means the estimate came entirely from the data,
    ``0`` entirely from the regulariser.

    Returns an :class:`xarray.Dataset` on the ``basis`` dimension with
    ``strike_slip``, ``dip_slip`` and their pointwise maximum ``resolution``,
    plus the ramp columns' diagonal in ``attrs`` where there are any.

    One dense Cholesky of a ``(2 n_basis + n_ramp)`` square matrix: seconds and a
    few hundred megabytes at the production 918-node mesh, which is why it is
    affordable once per round of an adaptive loop but not per candidate cell.
    """
    resolved = resolution_matrix(inversion, smoothing, ss_ratio, ds_ratio,
                                 boundary_ratio, sides, smooth_weighting)
    diagonal = np.diag(resolved).copy()

    n = inversion.basis.n_basis
    strike, dip = diagonal[:n], diagonal[n:2 * n]
    out = xr.Dataset(
        {
            "strike_slip": ("basis", strike),
            "dip_slip": ("basis", dip),
            "resolution": ("basis", np.maximum(strike, dip)),
        },
        coords={"basis": np.arange(n)},
    )
    out.attrs.update(
        smoothing=float(smoothing), basis=inversion.basis.name,
        n_ramp=int(inversion.n_ramp),
        # The effective number of parameters the data pays for. Falls with the
        # smoothing weight, which is the sanity check that this is a resolution
        # and not just a scaled column norm.
        effective_rank=float(np.sum(diagonal[:2 * n])),
        ramp_resolution=[float(v) for v in diagonal[2 * n:]],
    )
    return out


def parameter_params(inversion):
    """Each parameter's ``(s, z)`` position on the fault, in metres.

    The parameter plane, not 3-D space, because that is where "along strike" and
    "down dip" mean something -- and they are the two directions whose resolution
    differs most.
    """
    mesh = inversion.mesh
    return (mesh.params if inversion.basis.name != "element"
            else mesh.element_params)


def resolution_kernels(inversion, matrix):
    """Averaging-kernel width and offset per parameter, from the full ``R``.

    **This is the number the diagonal cannot give you.** Row ``k`` of ``R`` is the
    weighted average of the true model that the estimate of parameter ``k``
    actually represents; its *width* is the shortest wavelength that estimate can
    distinguish. A parameter can sit at resolution 0.05 and still be an average
    over tens of kilometres, in which case the elements around it are several
    names for one number and the mesh is finer than the data can support.

    Widths are second moments of ``|R[k, :]|`` about parameter ``k``'s own
    position, taken **within each slip component** (the strike-strike and dip-dip
    diagonal blocks): a cross-component weight is not a spatial spread. Returned
    per component and combined as the pointwise maximum, matching how
    :func:`resolution_diagonal` reports its diagonal.

    ``offset_along`` / ``offset_down`` are the signed centroids of the same
    weights. A large offset says the estimate is not even centred on its own
    parameter -- classically the case for deep parameters, whose kernels are
    pulled toward the surface where the data is.

    ``self_weight`` is ``|R[k, k]|`` over the row's total absolute weight: the
    fraction of the kernel that is the parameter itself.
    """
    matrix = np.asarray(matrix, dtype=float)
    n = inversion.basis.n_basis
    expected = 2 * n + inversion.n_ramp
    if matrix.shape != (expected, expected):
        raise ValueError(
            f"the matrix is {matrix.shape} but this inversion has {expected} "
            f"parameters ({n} basis functions x 2 components + "
            f"{inversion.n_ramp} ramp). Kernel positions come from the basis, so "
            "the matrix has to be the one this inversion produced."
        )
    params = np.asarray(parameter_params(inversion), dtype=float)
    if params.shape[0] != n:
        raise ValueError(
            f"the basis has {n} parameters but the mesh offers {params.shape[0]} "
            "positions; resolution_kernels needs the basis the matrix was built in"
        )
    s, z = params[:, 0], params[:, 1]

    blocks = {}
    for name, start in (("strike_slip", 0), ("dip_slip", n)):
        block = np.abs(matrix[start:start + n, start:start + n])
        mass = block.sum(axis=1)
        safe = np.maximum(mass, 1e-300)
        ds = s[None, :] - s[:, None]
        dz = z[None, :] - z[:, None]
        blocks[name] = dict(
            width_along=np.sqrt((block * ds ** 2).sum(axis=1) / safe),
            width_down=np.sqrt((block * dz ** 2).sum(axis=1) / safe),
            offset_along=(block * ds).sum(axis=1) / safe,
            offset_down=(block * dz).sum(axis=1) / safe,
            self_weight=np.abs(np.diag(block)) / safe,
        )

    data = {}
    for field in ("width_along", "width_down", "offset_along", "offset_down",
                  "self_weight"):
        for name in blocks:
            data[f"{name}_{field}"] = ("basis", blocks[name][field])
    # The worse-resolved component is the one that bounds what the mesh may
    # claim, so the summary fields take the wider kernel and the smaller share.
    data["width_along"] = ("basis", np.maximum(
        blocks["strike_slip"]["width_along"], blocks["dip_slip"]["width_along"]))
    data["width_down"] = ("basis", np.maximum(
        blocks["strike_slip"]["width_down"], blocks["dip_slip"]["width_down"]))
    data["self_weight"] = ("basis", np.minimum(
        blocks["strike_slip"]["self_weight"], blocks["dip_slip"]["self_weight"]))
    # Offsets are signed, so "worse" is the larger magnitude, not the larger value.
    for field in ("offset_along", "offset_down"):
        a = blocks["strike_slip"][field]
        b = blocks["dip_slip"][field]
        data[field] = ("basis", np.where(np.abs(a) >= np.abs(b), a, b))
    data["s"] = ("basis", s)
    data["depth"] = ("basis", -z)

    out = xr.Dataset(data, coords={"basis": np.arange(n)})
    out.attrs.update(basis=inversion.basis.name, n_ramp=int(inversion.n_ramp))
    return out


def data_importance(inversion, smoothing, ss_ratio=1.0, ds_ratio=3.0,
                    boundary_ratio=1.0, sides=("bottom", "left", "right"),
                    smooth_weighting="uniform"):
    """The data resolution matrix's diagonal, per observation.

    ``N = A (A'A + S'S + B'B)^-1 A'`` -- how much each observation's own value
    determines its own prediction. Near 1 means the datum is doing work no other
    datum can do; near 0 means the model would predict it just as well if it were
    deleted, because its neighbours already say the same thing. The other half of
    the pair :func:`resolution_matrix` gives: **model** resolution asks what the
    parameters mean, **data** resolution asks which observations paid for them.

    ``trace(N) == trace(R)``, both being the effective number of parameters the
    data supports, which is the arithmetic check that the two are consistent --
    :func:`resolution_report` asserts it.

    Only the diagonal: the full ``N`` is ``n_obs`` square (780 MB at 9900
    observations) and nothing here needs its off-diagonal structure.
    """
    from scipy.linalg import solve_triangular

    _, chol = _factorized(inversion, smoothing, ss_ratio, ds_ratio,
                          boundary_ratio, sides, smooth_weighting)
    weighted = weighted_design(inversion)
    # diag(N)_i = ||L^-1 a_i||^2, so one triangular solve against A' rather than
    # forming N. Costs one (n_param, n_obs) array -- 165 MB at production size.
    factor, lower = chol
    projected = solve_triangular(factor, weighted.T, lower=lower)
    return np.einsum("ij,ij->j", projected, projected)


def resolution_report(inversion, smoothing, ss_ratio=1.0, ds_ratio=3.0,
                      boundary_ratio=1.0, sides=("bottom", "left", "right"),
                      smooth_weighting="uniform"):
    """Model and data resolution together, with the discretisation verdict.

    One factorization, three readings: the diagonal of ``R`` (how much of each
    estimate is data), the averaging kernels (what spatial scale each estimate
    actually represents) and ``diag(N)`` (which observations paid for it).

    ``over_discretisation`` is the median kernel width divided by the median
    element size.

    ⚠️ **Do NOT read it as "the mesh is this many times too fine". That reading
    was tried on the real data and it is wrong.** The kernel width is not a
    property of the data: it falls with the effective rank on *every* mesh
    (47.6 -> 26.1 km measured across four mesh resolutions on one fixed
    dataset), so the ratio moves when you change lambda without changing the
    mesh at all. Measured properly -- four meshes from 2.5 to 20 km compared at
    **matched effective rank** -- rms agreed to 0.3% and Mw to 0.005, and where
    they differed the *coarser* mesh was marginally better. The mesh sets only
    the ceiling on reachable rank (140 parameters cannot reach rank 100); within
    that ceiling it does not change the answer.

    So use this number as a description, not a verdict: a large value says the
    estimates overlap heavily, which is expected and not by itself a fault.

    ⚠️ **The diagonal and the kernel width answer different questions, and only
    the kernel width is a length.** Measured on a two-track synthetic across a
    100x sweep of lambda, the median diagonal fell 0.327 -> 0.034 (10x) while the
    along-strike kernel width moved 24.6 -> 21.7 km (12%) -- the diagonal is
    largely a statement about how hard the problem is damped, the kernel width
    about what the data geometry can separate. Note this does *not* mean a
    diagonal-driven size field drifts with lambda:
    :func:`target_element_size` normalises by the median resolution, which
    divides most of that out. Its actual failure is saturation -- see
    :func:`kernel_element_size`.

    Returns an :class:`xarray.Dataset` merging the diagonal and kernel fields on
    ``basis``, with ``importance`` on ``obs``.
    """
    data, chol = _factorized(inversion, smoothing, ss_ratio, ds_ratio,
                             boundary_ratio, sides, smooth_weighting)
    matrix = cho_solve(chol, data)
    diagonal = np.diag(matrix).copy()
    kernels = resolution_kernels(inversion, matrix)

    from scipy.linalg import solve_triangular
    weighted = weighted_design(inversion)
    factor, lower = chol
    projected = solve_triangular(factor, weighted.T, lower=lower)
    importance = np.einsum("ij,ij->j", projected, projected)

    n = inversion.basis.n_basis
    out = kernels.copy()
    out["strike_slip"] = ("basis", diagonal[:n])
    out["dip_slip"] = ("basis", diagonal[n:2 * n])
    out["resolution"] = ("basis", np.maximum(diagonal[:n], diagonal[n:2 * n]))
    out["importance"] = ("obs", importance)

    element = element_size(inversion.mesh)
    width = np.maximum(kernels["width_along"].values, kernels["width_down"].values)
    trace_model = float(np.trace(matrix))
    trace_data = float(importance.sum())
    out.attrs.update(
        smoothing=float(smoothing), basis=inversion.basis.name,
        n_ramp=int(inversion.n_ramp),
        effective_rank=float(np.sum(diagonal[:2 * n])),
        trace_model_resolution=trace_model,
        trace_data_resolution=trace_data,
        # trace(A M^-1 A') == trace(M^-1 A'A) identically, so a mismatch means
        # the two halves were not built from the same factorization.
        trace_identity_error=float(abs(trace_model - trace_data)),
        median_kernel_width_m=float(np.median(width)),
        median_element_size_m=float(np.median(element)),
        over_discretisation=float(np.median(width) / max(np.median(element), 1e-9)),
        median_self_weight=float(np.median(kernels["self_weight"].values)),
        median_importance=float(np.median(importance)),
    )
    return out


#: Fraction of parameters allowed on a bound before
#: :func:`posterior_covariance` refuses. The linear covariance describes the
#: *unbounded* damped problem; a solution pressed against its bounds is a
#: different problem and the number would be a plausible-looking lie.
#:
#: ⚠️ This is a property of the **model**, not of the code, and it has moved a
#: long way. The 2026-07 Venezuela run had **86%** of elements on a bound, which
#: is why this module's own docstring says the resolution matrix is never a
#: statement about uncertainty. The 2026-08-30 production model has **1.6%**
#: (6 of 374, all on the polarity constraint). Re-check it per run.
MAX_BOUND_FRACTION = 0.10

#: Which covariance :func:`posterior_covariance` returns. Named apart from
#: :data:`nisar_tools.slip.covariance.COVARIANCE_KINDS`, which is the *noise*
#: model's shape and a different thing entirely.
POSTERIOR_KINDS = ("propagated", "posterior")


def bound_fraction(model, tolerance=1e-3):
    """Fraction of slip parameters sitting on one of their bounds.

    The gate on every linear uncertainty statement here -- see
    :data:`MAX_BOUND_FRACTION`.
    """
    options = getattr(model, "options", None) or {}
    strike = options.get("strike", (-10.0, 10.0))
    dip = options.get("dip", (-10.0, 10.0))
    polarity = options.get("polarity")
    if polarity:
        lo, hi = strike
        if polarity[0] < 0:
            hi = min(hi, 0.0)
        elif polarity[0] > 0:
            lo = max(lo, 0.0)
        strike = (lo, hi)

    on = 0
    for values, (lo, hi) in ((model.strike_slip, strike), (model.dip_slip, dip)):
        v = np.asarray(values, dtype=float)
        for edge in (lo, hi):
            if np.isfinite(edge):
                on += int(np.isclose(v, edge, atol=tolerance).sum())
    total = np.size(model.strike_slip) + np.size(model.dip_slip)
    return float(on) / max(total, 1)


def posterior_covariance(inversion, smoothing, kind="propagated", full=False,
                         model=None, ss_ratio=1.0, ds_ratio=3.0,
                         boundary_ratio=1.0, sides=("bottom", "left", "right"),
                         smooth_weighting="uniform"):
    """Parameter covariance of the damped solve. Returns sigma, or the full matrix.

    Two readings, and **they differ by an order of magnitude here**, so the choice
    is not a detail:

    * ``"propagated"`` (default) -- ``M^-1 (A' Cd^-1 A) M^-1``: the estimator's
      variance under data noise. **This is the one that matches reality.**
      Validated against 60 repeated bounded solves on noise drawn from the same
      ``Cd``: empirical spread / analytic sigma = **1.01**, per-parameter
      correlation **0.950**.
    * ``"posterior"`` -- ``M^-1`` with ``M = A' Cd^-1 A + lam^2 L'L + W'W``. The
      Bayesian posterior *if the smoothing is taken as a prior distribution you
      believe*. On the same test it is **11x too wide** (ratio 0.09).

    They differ by exactly ``M^-1 (lam^2 L'L + W'W) M^-1``, the prior's own
    contribution, so ``posterior = propagated + prior term`` identically. The
    factor of eleven is not an error in either: repeated observation varies only
    the data, which is what ``"propagated"`` predicts, while ``M^-1`` also carries
    a spread that no amount of re-observing samples. Ask for ``"posterior"`` only
    if you mean the Bayesian question.

    ⚠️ **Summarise with the mean or the max, never the median.** At light damping
    the per-parameter sigma spans four orders of magnitude -- the unresolved
    parameters carry enormous variance while the bulk stays small -- so the median
    describes the bulk and hides the tail. Measured over lambda 0.01 to 1e4 on the
    synthetic: mean sigma falls monotonically **128 -> 0.03 m** as it should,
    while the median *rises* 0.059 -> 0.207 before falling, purely because the
    damping pulls the tail in faster than it lifts the bulk. Reading the median as
    "the uncertainty" would say more smoothing makes the model less certain.

    ⚠️ **That the prior term dominates by 11x is itself the result.** It is the
    same fact as an effective rank of 21 out of 748 parameters -- a typical
    parameter is **97% regulariser** -- seen from the covariance side. Quote either
    number alongside :func:`resolution_diagonal`, which says how much of each
    estimate the data actually paid for.

    ⚠️ **Requires a covariance model.** Without one,
    :meth:`~nisar_tools.slip.inversion.SlipInversion.whiten` multiplies by
    ``1/sqrt(n)`` per track, which is not ``Cd^-1/2`` and carries no units -- the
    resulting "covariance" would be a number in arbitrary units that looks like
    a variance. Same rule as
    :attr:`~nisar_tools.slip.inversion.SlipModel.chi_square`.

    ⚠️ **Describes the UNBOUNDED problem.** Pass ``model=`` and it checks the
    fraction of parameters on a bound against :data:`MAX_BOUND_FRACTION` and
    raises rather than returning a number for a problem that was not solved.
    """
    if kind not in POSTERIOR_KINDS:
        raise ValueError(f"kind must be one of {POSTERIOR_KINDS}, got {kind!r}")
    if getattr(inversion, "covariance", None) is None:
        raise ValueError(
            "posterior_covariance needs a covariance model: without one the "
            "weights are 1/sqrt(n) per track, which is not Cd^-1/2, and the "
            "result would be in arbitrary units rather than metres squared. "
            "Build the inversion with SlipInversion(..., covariance=...)."
        )
    if model is not None:
        fraction = bound_fraction(model)
        if fraction > MAX_BOUND_FRACTION:
            raise ValueError(
                f"{fraction:.1%} of parameters sit on a bound, above "
                f"MAX_BOUND_FRACTION={MAX_BOUND_FRACTION:.0%}. This covariance "
                "describes the unbounded damped problem, so on a solution pressed "
                "against its bounds it would be a plausible-looking lie. Loosen "
                "the bounds, or use a Monte Carlo through the bounded solve."
            )

    data, chol = _factorized(inversion, smoothing, ss_ratio, ds_ratio,
                             boundary_ratio, sides, smooth_weighting)
    inverse = cho_solve(chol, np.eye(data.shape[0]))
    covariance = inverse if kind == "posterior" else inverse @ data @ inverse
    if full:
        return covariance
    return np.sqrt(np.maximum(np.diag(covariance), 0.0))


def resolution_bias(matrix, model):
    """``(R - I) @ m`` -- the smoothing's pull on *this* model, in metres.

    The other half of the error. A regularised estimate carries variance from the
    data noise (:func:`posterior_covariance`) **and** bias from the regulariser,
    and they trade off along the L-curve: more damping buys variance down and
    bias up. This is the bias term evaluated at the recovered model, which is the
    best stand-in available for the true one.

    Returns the pull per parameter **and** ``relative``, its norm over the model's
    own norm.

    ⚠️ **Across a lambda sweep, pass the SAME model every time.** Two traps sit
    here and both were hit before this warning existed:

    1. The **absolute** pull is taken at the recovered model, which shrinks as the
       damping rises -- so it shrinks too, and appears to say that more smoothing
       biases the answer *less*. Measured: mean pull 0.317 -> 0.098 m over lambda
       1 -> 100, exactly backwards. Use ``relative``.
    2. ``relative`` is not monotone either when the model moves with lambda,
       because ``m`` **rotates** toward the smooth directions ``R`` preserves
       best: measured 0.203 -> 0.431 -> 0.290 over the same sweep. The recovered
       model is already the smoothed one, so ``(R - I) m`` understates the pull
       away from the *truth*.

    So a lambda sweep must hold the reference model fixed -- the model at the
    corner, or the truth on a synthetic -- and vary only ``matrix``. Then
    ``relative`` rises monotonically toward 1, which is the statement that at
    heavy damping the estimate is entirely the regulariser's.

    ⚠️ **Do not add this in quadrature with the posterior sigma and call the
    result an error bar.** Both move when ``lambda`` moves, so they are not
    independent contributions to one total; the honest presentation is the
    trade-off across a ``lambda`` ladder, not a single number.
    """
    matrix = np.asarray(matrix, dtype=float)
    n = np.size(model.strike_slip)
    slip = np.concatenate([np.asarray(model.strike_slip, dtype=float),
                           np.asarray(model.dip_slip, dtype=float)])
    if matrix.shape[0] < 2 * n:
        raise ValueError(
            f"the matrix is {matrix.shape} but the model has {2 * n} slip "
            "parameters; pass the resolution matrix this inversion produced")
    block = matrix[:2 * n, :2 * n]
    pulled = block @ slip - slip
    norm = float(np.linalg.norm(slip))
    return {
        "strike_slip": pulled[:n],
        "dip_slip": pulled[n:2 * n],
        "relative": float(np.linalg.norm(pulled) / norm) if norm > 0 else 0.0,
    }


def element_resolution(inversion, resolution):
    """Resolution per element, whatever basis it was computed in."""
    values = np.asarray(resolution["resolution"].values, dtype=float)
    if inversion.basis.name == "element":
        return values
    return inversion.basis.element_values(values)


def element_size(mesh):
    """The equivalent square edge of each element, ``sqrt(2 * area)``.

    The same length ``layered._orders`` uses to price a triangle against its
    distance to an observation, so "element size" means one thing in the package.
    """
    return np.sqrt(2.0 * np.asarray(mesh.areas, dtype=float))


def smooth_over_mesh(mesh, values, rounds=3):
    """Average a per-element field with its neighbours, ``rounds`` times.

    A target size field has to be Lipschitz or it asks for a jump no
    triangulation can honour without slivers; a few Jacobi sweeps over the
    element adjacency is the cheapest way to get there and needs no geometry
    beyond ``mesh.neighbors``.
    """
    values = np.asarray(values, dtype=float).copy()
    neighbors = mesh.neighbors
    for _ in range(int(rounds)):
        total = values.copy()
        count = np.ones(values.size)
        for k in range(neighbors.shape[1]):
            other = neighbors[:, k]
            has = other >= 0
            total[has] += values[other[has]]
            count[has] += 1.0
        values = total / count
    return values


def target_element_size(mesh, resolution, target=None, gain=0.5, max_ratio=3.0,
                        size_min=None, size_max=None, rounds=3):
    """Element sizes that would even out ``resolution`` across the mesh.

    ``h_new = h * (target / r) ** gain``. An element resolved better than
    ``target`` can afford to be smaller; one resolved worse is carrying
    parameters the data does not pay for and is grown. ``target`` defaults to the
    median, so the node budget is roughly conserved and the *distribution* is
    what moves -- pass a higher value to buy a coarser mesh outright.

    ``gain`` below 1 under-relaxes deliberately. Resolution is itself a function
    of the discretisation, so a full step chases its own tail; half a step
    converges in a few rounds instead of oscillating.

    ⚠️ **``max_ratio`` is not a nicety.** Below a few kilometres of slip depth,
    resolution does not merely get small, it underflows to *exactly* zero -- on
    the 330-element synthetic fixture at lambda 0.3 the mean below 14 km is 0.0000
    against 0.3304 above 6 km. ``target / r`` is then whatever the floor makes it,
    ``gain`` cannot temper an infinity, and one step asks for elements four times
    the mesh's own extent: measured, an unbounded first pass collapsed a 20 km
    fault to **three** depth levels with a 7.5 degree minimum angle. Bounding the
    *ratio* per step, and iterating, is what keeps a converging loop from taking
    one meaningless leap. ``size_min``/``size_max`` remain as absolute clamps on
    top.

    ⚠️ **This sets the size field's *shape*, not the mesh's size.** Targeting the
    median looks self-normalising and is not: the resolution distribution is
    heavily skewed -- most elements near zero, a few near one -- so most of them
    hit the growth cap while few can shrink, and the mean element size climbs
    every round. Measured on the two-scene synthetic fixture, a loop with no
    element budget coarsened 330 elements to 186, then 46, then 44, with variance
    reduction collapsing 94.3% -> 91.6% -> **0.74%** and peak slip falling to a
    millimetre: it ran end to end, converged, and returned nothing. The count is
    held by :func:`adapt_layout`'s ``budget``, which is the right division of
    labour anyway -- the caller says how many parameters they can afford, the
    resolution says where to put them.
    """
    resolution = np.asarray(resolution, dtype=float)
    if resolution.shape != (mesh.n_elements,):
        raise ValueError(
            f"resolution has {resolution.shape} entries for {mesh.n_elements} elements; "
            "pass element_resolution(inversion, ...) rather than the basis diagonal"
        )
    current = element_size(mesh)
    target = float(np.median(resolution)) if target is None else float(target)
    if target <= 0:
        raise ValueError("target resolution must be positive")
    max_ratio = float(max_ratio)
    if max_ratio < 1.0:
        raise ValueError("max_ratio is a growth factor and must be at least 1")

    floor = np.maximum(resolution, 1e-12)
    ratio = np.clip((target / floor) ** float(gain), 1.0 / max_ratio, max_ratio)
    # The *correction* is smoothed, not the absolute size, so a resolution field
    # asking for no change gets none: smoothing `current * ratio` would blur the
    # mesh's own slightly uneven element sizes into each other and drift.
    size = current * smooth_over_mesh(mesh, ratio, rounds=rounds)

    lo = current.min() / max_ratio if size_min is None else float(size_min)
    hi = current.max() * max_ratio if size_max is None else float(size_max)
    if lo > hi:
        raise ValueError(f"size_min={lo} is above size_max={hi}")
    return np.clip(size, lo, hi)


def size_field(mesh, size):
    """A per-element target size, read back as a continuous ``h(s, depth)``.

    The seam between "what resolution says each element should be" and "what a
    mesh generator needs", which is a function of position rather than a value
    per element. Scattered from element centroids in ``(s, z)`` and read back by
    **nearest neighbour**: this is a target for a mesh generator, not a function
    being integrated accurately, so a smooth interpolant would be false
    precision. Distances are in metres on both axes, so nearest is well posed
    without a scaling choice -- the parameter plane is already metric.

    Both :func:`adapt_layout` structures consume this: the row builder samples it
    along one vertical and then along each level, the unstructured builder hands
    it straight to :func:`~nisar_tools.slip.layout.lloyd_layout` as its density.
    """
    size = np.asarray(size, dtype=float)
    if size.shape != (mesh.n_elements,):
        raise ValueError(f"size has {size.shape} entries for {mesh.n_elements} elements")
    centre = mesh.element_params

    def at(s, depth):
        # Shape-agnostic: the row builder asks along a line, the unstructured one
        # asks on a 2-D probe grid, and both get their own shape back.
        s = np.atleast_1d(np.asarray(s, dtype=float))
        depth = np.broadcast_to(np.asarray(depth, dtype=float), s.shape)
        flat_s, flat_d = s.ravel(), depth.ravel()
        d2 = ((flat_s[:, None] - centre[None, :, 0]) ** 2
              + (-flat_d[:, None] - centre[None, :, 1]) ** 2)
        return size[np.argmin(d2, axis=1)].reshape(s.shape)

    return at


#: How element size is read off an anisotropic averaging kernel. The kernels here
#: are far wider along strike than down dip (26-41 km against 10-16 km on the
#: Venezuela data), and an element carries **one** size, so the two directions
#: have to be reconciled. ``"min"`` keeps the better-resolved direction, which
#: costs redundancy along strike but never discards structure the data can see;
#: ``"max"`` does the opposite and is the one that loses information.
KERNEL_REDUCTIONS = ("min", "geometric", "max")


def kernel_element_size(inversion, smoothing, fraction=0.5, reduce="min",
                        size_min=None, size_max=None, rounds=3, report=None,
                        **kwargs):
    """Target element size from the averaging-kernel **width**, not the diagonal.

    ``h = fraction * kernel_width``. Two elements per resolvable feature at the
    default ``fraction=0.5``, which is the sampling analogue of Nyquist: finer
    than that and neighbouring elements are several names for one number.

    **Why this rather than** :func:`target_element_size`. That one forms
    ``(target / r) ** gain`` on the *diagonal* of ``R``, and on real data ``r``
    has underflowed over most of the fault -- so it is a ratio of two nearly-zero
    numbers and simply runs to its ``max_ratio`` bound. Measured on a
    132-element synthetic across a 1000x sweep of lambda, **42-75% of elements sit
    pinned at the clamp** (median ``r`` 0.0000, min 2e-8): more than half the size
    field is the clamp rather than the data. The kernel width over the same sweep
    stays a real length everywhere, 4-11 km, varying smoothly.

    ⚠️ It is **not** that the diagonal criterion chases lambda -- an earlier note
    here said so and it was wrong. ``target`` defaults to the *median* resolution,
    which divides out most of the lambda dependence: over that same sweep the
    diagonal-driven size field varied 13% and the kernel-driven one 18%, so if
    anything the diagonal is the steadier of the two. The defect is degeneracy,
    not drift.

    ``reduce`` picks how the anisotropic kernel becomes one number; see
    :data:`KERNEL_REDUCTIONS`. ``report`` reuses a
    :func:`resolution_report` you already have, which is the expensive part.

    ⚠️ **This is NOT Barnhart & Lohman's metric, despite being the kernel-width
    option.** They fit a *Gaussian* to row ``k`` of ``R`` against 3-D distance and
    take its 1-sigma; :func:`resolution_kernels` takes *second moments*, which
    weight by distance squared. The averaging kernel here is strongly
    non-Gaussian -- measured on the real six-scene model, **19.4% of a row's
    ``|R|`` mass sits beyond 3 sigma of the Gaussian fit against the 0.27% a
    Gaussian would put there**, a 70x excess -- so the second moment reads the
    tails and comes out **3.09x** their sigma along strike (39.3 km against 11.5).
    Trimming the weakest 10% of each row brings the ratio to 1.05, which
    identifies the tails as the whole of the difference. Down dip the medians
    nearly agree (1.21x) but the per-parameter correlation is only **0.50**.

    ⚠️ And ``reduce="min"`` selects the down-dip width on **100%** of parameters
    on that model, so the along-strike number never reaches the size field at all.
    The anisotropy is real (2.67x median) but it is not being *reconciled* -- one
    direction is simply always chosen.

    Consequence, measured: a four-round loop sized from this lost to one sized
    from the resolution diagonal at every effective rank, and to a *fixed* mesh at
    low rank. That is evidence against **this estimator**, not against Barnhart &
    Lohman's method, which has never been run here. Fitting a Gaussian instead is
    the untested alternative.
    """
    if reduce not in KERNEL_REDUCTIONS:
        raise ValueError(
            f"reduce must be one of {KERNEL_REDUCTIONS}, got {reduce!r}")
    if report is None:
        report = resolution_report(inversion, smoothing, **kwargs)

    along = np.asarray(report["width_along"].values, dtype=float)
    down = np.asarray(report["width_down"].values, dtype=float)
    if reduce == "min":
        width = np.minimum(along, down)
    elif reduce == "max":
        width = np.maximum(along, down)
    else:
        width = np.sqrt(along * down)

    if inversion.basis.name != "element":
        width = inversion.basis.element_values(width)
    size = float(fraction) * width
    size = smooth_over_mesh(inversion.mesh, size, rounds=rounds)

    current = element_size(inversion.mesh)
    lo = current.min() / 3.0 if size_min is None else float(size_min)
    hi = current.max() * 3.0 if size_max is None else float(size_max)
    if lo > hi:
        raise ValueError(f"size_min={lo} is above size_max={hi}")
    return np.clip(size, lo, hi)


def _stations_for_density(extent, density, samples=1024):
    """Positions across ``[0, extent]`` at a varying target spacing.

    Integrating ``1/h`` and then splitting the total into a whole number of
    equal steps is what lands on both ends *exactly*, which is what
    :func:`~nisar_tools.slip.layout.boundary_node_mask` needs; marching by the
    local spacing would leave the far end a ragged fraction short.
    """
    grid = np.linspace(0.0, float(extent), int(samples))
    inverse = 1.0 / np.maximum(density(grid), 1e-9)
    cumulative = np.concatenate([[0.0], np.cumsum(
        0.5 * (inverse[1:] + inverse[:-1]) * np.diff(grid))])
    n = max(1, int(round(cumulative[-1])))
    out = np.interp(np.linspace(0.0, cumulative[-1], n + 1), cumulative, grid)
    out[0], out[-1] = 0.0, float(extent)
    return out


def adapt_layout(mesh, size, budget="conserve", top_depth=None, max_depth=None,
                 length=None, balance=MAX_COUNT_RATIO, samples=1024, tolerance=0.05,
                 structure="rows", iterations=25, seed=0):
    """A new layout honouring a per-element target size.

    ``structure`` decides how much freedom the node set has:

    * ``"rows"`` (default) -- depth levels, each with its own node count and its
      own along-strike positions. What every run before 2026-08-27 produced.
    * ``"unstructured"`` -- no levels at all: the size field is handed to
      :func:`~nisar_tools.slip.layout.lloyd_layout` as a density and the nodes
      relax to it in two dimensions.

    ⚠️ **The row builder reads the depth levels off ONE vertical, at the fault's
    midpoint** (``s = length/2``), because a level is a single depth for the
    whole fault and so cannot be in two places at once. Along-strike node
    *density* does vary per level, but the layering does not: a resolution field
    wanting fine levels at one end and coarse ones at the other has no way to say
    so. ``"unstructured"`` is that constraint removed, and it is the reason it
    exists. The *outline* is a rectangle in ``(s, z)`` either way -- freeform
    means the discretisation, not the fault's extent.

    ``iterations`` and ``seed`` reach the relaxation and matter only for
    ``"unstructured"``. Both are fixed by default, so a layout is reproducible
    and ``FaultMesh.digest()`` is stable across runs.

    ``budget`` is the element count to land on: ``"conserve"`` keeps the mesh's
    own, an integer asks for that many, ``None`` takes whatever ``size`` implies.
    It is found by scaling the whole size field and rebuilding -- a handful of
    bisection steps -- rather than from a tiling formula, because neither
    builder's count follows one: the row builder integrates ``1/h`` along each
    axis *separately*, so a closed form is wrong by a factor that depends on how
    anisotropic the size field is, and the unstructured builder's count comes out
    of a relaxation. Each rebuild is cheap either way (0.02 s at 240 elements).

    ⚠️ **Without a budget an adaptive loop does not converge, it drifts**; see
    :func:`target_element_size` for the measurement.

    Under ``"rows"``, depth levels come from integrating ``1/h`` down dip and
    each level's nodes from integrating ``1/h`` along strike *at that depth* --
    so both the count **and** the positions vary per level, which is what
    "vary the number of nodes at the depth levels" costs once the size field is
    not separable.

    The size field is read back through :func:`size_field`. Note ``h`` is
    honoured as a node *spacing* while :func:`element_size` measures
    ``sqrt(2 * area)``, about 7% smaller for a well-shaped triangle; the budget
    absorbs that constant, and it is the same on both paths, so nothing between
    them is compared unfairly.
    """
    size = np.asarray(size, dtype=float)
    if size.shape != (mesh.n_elements,):
        raise ValueError(f"size has {size.shape} entries for {mesh.n_elements} elements")
    if structure not in ("rows", "unstructured"):
        raise ValueError(
            f"structure must be 'rows' or 'unstructured', got {structure!r}"
        )
    if budget is not None:
        wanted = mesh.n_elements if budget == "conserve" else int(budget)
        if wanted < 1:
            raise ValueError("budget must be at least one element")
        return _fit_budget(
            lambda scale: adapt_layout(mesh, size * scale, budget=None,
                                       top_depth=top_depth, max_depth=max_depth,
                                       length=length, balance=balance, samples=samples,
                                       structure=structure, iterations=iterations,
                                       seed=seed),
            wanted, float(tolerance))

    params = mesh.params
    length = float(params[:, 0].max()) if length is None else float(length)
    top_depth = float(-params[:, 1].max()) if top_depth is None else float(top_depth)
    max_depth = float(-params[:, 1].min()) if max_depth is None else float(max_depth)
    at = size_field(mesh, size)

    if structure == "unstructured":
        layout = lloyd_layout(length, max_depth, at, top_depth=top_depth,
                              iterations=iterations, seed=seed)
        layout.attrs["layout"] = "adapted_unstructured"
        layout.attrs["quality"] = layout.quality()
        _warn_if_slivered(layout)
        return layout

    middle = 0.5 * length
    depths = top_depth + _stations_for_density(
        max_depth - top_depth,
        lambda d: at(np.full_like(d, middle), top_depth + d),
        samples=samples,
    )
    depths = _balance_levels(depths, balance)

    rows, counts = [], []
    for depth in depths:
        stations = _stations_for_density(
            length, lambda s, d=depth: at(s, np.full_like(s, d)), samples=samples)
        rows.append(stations)
        counts.append(stations.size)

    counts = _rebalance(np.asarray(counts, dtype=int), balance)
    params = np.vstack([
        np.column_stack([_resample_stations(row, int(n)),
                         np.full(int(n), -float(depth))])
        for row, n, depth in zip(rows, counts, depths)
    ])

    layout = MeshLayout(params, triangulate(params),
                        levels=depths, counts=counts,
                        attrs={"layout": "adapted"})
    require_boundaries(layout.params)
    return layout


#: Minimum triangle angle, in degrees, below which an unstructured layout is
#: flagged. Chosen from measurement, not taste: the density-weighted relaxation
#: produces 21-27 degrees on every size field tried, and the *uniform* production
#: mesh already sits at 11.3 degrees (5 km along strike against ~1 km top
#: levels), so this is below both and fires only on something genuinely
#: degenerate. Relaxation is what buys the quality -- the unrelaxed cloud
#: measures 0.4 degrees.
MIN_ANGLE_WARN_DEG = 10.0


def _warn_if_slivered(layout):
    """Flag an unstructured layout whose triangulation has gone degenerate.

    Worth a warning rather than silence because the adaptive loop runs
    unattended for several rounds: a sliver degrades the TDE Green's functions
    and ``basis._barycentric_weight`` divides by a quantity that vanishes on one.
    A warning rather than a raise because the mesh is still usable and the caller
    may have asked for something extreme on purpose.
    """
    angle = float(layout.attrs["quality"]["min_angle_deg"])
    if angle < MIN_ANGLE_WARN_DEG:
        warnings.warn(
            f"The adapted unstructured layout has a minimum triangle angle of "
            f"{angle:.1f} degrees, below {MIN_ANGLE_WARN_DEG:.0f}. Raise "
            "iterations=, or bound the size field harder with max_ratio= / "
            "size_min= / size_max= on target_element_size.",
            RuntimeWarning,
            stacklevel=3,
        )


def _fit_budget(build, wanted, tolerance):
    """Scale the size field until the built layout holds about ``wanted`` elements.

    Monotone in the scale -- bigger elements, fewer of them -- so a geometric
    bracket followed by bisection in log-scale converges in a handful of builds,
    and each build is cheap enough to do this properly rather than estimate it.
    """
    layout = build(1.0)
    if abs(layout.n_elements - wanted) <= tolerance * wanted:
        return layout

    lo = hi = 1.0
    if layout.n_elements > wanted:
        for _ in range(12):                     # too many elements -> coarsen
            lo, hi = hi, hi * 1.5
            layout = build(hi)
            if layout.n_elements <= wanted:
                break
    else:
        for _ in range(12):                     # too few -> refine
            lo, hi = lo / 1.5, lo
            layout = build(lo)
            if layout.n_elements >= wanted:
                break

    best = layout
    for _ in range(20):
        mid = float(np.sqrt(lo * hi))
        layout = build(mid)
        if abs(layout.n_elements - wanted) < abs(best.n_elements - wanted):
            best = layout
        if abs(layout.n_elements - wanted) <= tolerance * wanted:
            return layout
        if layout.n_elements > wanted:
            lo = mid
        else:
            hi = mid
    return best


def _balance_levels(depths, balance):
    """Limit how fast the level thickness may change from one level to the next.

    The counts are balanced across strike by
    :func:`~nisar_tools.slip.layout._balance_counts`; this is the same rule down
    dip, and it is needed for the same reason. A size field that saturates at
    depth otherwise puts one 17 km-tall element under a 2.5 km one, which is a
    7-to-1 aspect ratio however many nodes the row has.
    """
    thickness = np.diff(np.asarray(depths, dtype=float))
    if thickness.size < 2:
        return np.asarray(depths, dtype=float)
    for _ in range(thickness.size):
        before = thickness.copy()
        thickness[1:] = np.minimum(thickness[1:], thickness[:-1] * balance)
        thickness[:-1] = np.minimum(thickness[:-1], thickness[1:] * balance)
        if np.allclose(thickness, before):
            break
    # Renormalise so the levels still span exactly the same depth range; the
    # bottom must stay at max_depth or the mesh quietly changes extent.
    thickness *= (depths[-1] - depths[0]) / thickness.sum()
    return depths[0] + np.concatenate([[0.0], np.cumsum(thickness)])


def _resample_stations(stations, n):
    """Change a row's node count while keeping the shape of its spacing.

    Resampled uniformly *in index*, so a row that was dense at one end stays
    dense there, and both ends land on their exact original values -- which is
    what keeps the boundary test exact after ``_balance_counts`` has raised a
    count.
    """
    if stations.size == n:
        return stations
    return np.interp(np.linspace(0.0, 1.0, n),
                     np.linspace(0.0, 1.0, stations.size), stations)


def _rebalance(counts, balance):
    from .layout import _balance_counts

    # Quiet: these counts came from integrating a size field, not from a caller
    # asking for them, so refining one is the algorithm rather than an override
    # -- and the budget search rebuilds a dozen times.
    return _balance_counts(counts, balance, warn=False)


def resolution_gain(model, los_stack, resolution, frame=None, spacing=GAIN_SPACING,
                    exponent=1.0):
    """Where another observation would most improve model resolution.

    Lohman & Simons 2005's criterion, evaluated as a field rather than by
    recursing on candidate cells: a sample at ``(x, y)`` helps in proportion to
    how strongly it sees the parameters that are *currently unresolved*,

    ``w(x, y) = sum_k G(x, y; k)**2 * (1 - r_k)``.

    Returns an ``(ny, nx)`` array on ``los_stack``'s own grid, normalised to unit
    standard deviation over its finite samples, ready to hand to
    :meth:`~nisar_tools.slip.sampling.Observations.from_los` as ``field=``.

    Evaluated on a coarse grid (:data:`GAIN_SPACING`, coarser than the predicted
    field's) and interpolated up, because the cost is a full Green's-function
    assembly at every probe point. Recursing instead -- testing
    each candidate cell against ``R`` as the paper describes -- needs a
    Green's-function row per candidate, thousands of them, against a quadtree
    that currently costs 0.05 s on 0.32 M pixels.
    """
    from scipy.interpolate import RegularGridInterpolator

    from .sampling import _to_frame, stack_epsg

    spacing = GAIN_SPACING if spacing is None else float(spacing)
    frame = frame or model.mesh.frame
    if frame is None:
        raise ValueError("resolution_gain needs a LocalFrame; the mesh carries none")

    mesh = model.mesh
    unresolved = 1.0 - np.clip(
        np.asarray(resolution["resolution"].values, dtype=float), 0.0, 1.0)

    ds = los_stack.ds
    epsg = stack_epsg(ds)
    x_native = np.asarray(ds["x"].values, dtype=float)
    y_native = np.asarray(ds["y"].values, dtype=float)
    gx, gy = np.meshgrid(x_native, y_native)
    fx, fy = _to_frame(gx.ravel(), gy.ravel(), frame, epsg)

    cx = np.arange(fx.min(), fx.max() + spacing, spacing)
    cy = np.arange(fy.min(), fy.max() + spacing, spacing)
    px, py = np.meshgrid(cx, cy)

    # The scene's own mean look vector, renormalised. One direction for the whole
    # coarse grid is an approximation -- a wide swath runs 25 to 49 degrees of
    # incidence -- but the gain only has to *rank* candidate locations, and using
    # the mean beats a nominal vertical, which would misweight the east-west
    # sensitivity that a strike-slip fault's signal mostly lives in.
    look = np.array([np.nanmean(np.asarray(ds[f"los_{c}"].values, dtype=float))
                     for c in ("east", "north", "up")])
    norm = np.linalg.norm(look)
    look = look / norm if norm > 0 else np.array([0.0, 0.0, 1.0])
    ones = np.ones(px.size)
    engine = model.inversion.engine if model.inversion is not None else None
    if engine is None:
        raise ValueError(
            "resolution_gain needs the model's engine, which SlipModel.load "
            "deliberately does not restore; rebuild the SlipInversion first"
        )
    green = engine.los_matrix(mesh, px.ravel(), py.ravel(),
                              look[0] * ones, look[1] * ones, look[2] * ones)

    n = model.basis.n_basis
    if green.shape[1] != 2 * n:
        projection = model.basis.projection()
        green = np.hstack([green[:, :mesh.n_elements] @ projection,
                           green[:, mesh.n_elements:] @ projection])
    weight = np.concatenate([unresolved, unresolved]) ** float(exponent)
    gain = (green ** 2) @ weight

    interp = RegularGridInterpolator(
        (cy, cx), gain.reshape(px.shape), bounds_error=False, fill_value=np.nan)
    out = interp(np.column_stack([fy, fx])).reshape(gx.shape)

    finite = np.isfinite(out)
    if not finite.any():
        raise ValueError("the resolution-gain field is empty over this scene")
    scale = out[finite].std()
    if scale == 0:
        warnings.warn(
            "the resolution-gain field is constant, so it cannot steer a "
            "quadtree; every parameter is equally (un)resolved from this scene.",
            RuntimeWarning, stacklevel=2,
        )
        return out
    return (out - out[finite].mean()) / scale
