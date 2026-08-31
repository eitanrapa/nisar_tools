"""Spatially correlated InSAR noise: measuring it, and whitening it out.

Until this module existed an observation's weight was ``1.0``, scaled per track
by ``1/sqrt(n)`` in :meth:`~nisar_tools.slip.sampling.Observations.concat`. So
every sample of a track counted the same, and a track's influence was set by how
many samples it happened to contribute -- not by how good they are. The
within-cell scatter that :class:`~nisar_tools.slip.sampling.Observations` has
always stored as ``std`` was read by nothing at all.

That is exactly the assumption Lohman & Simons (2005) exists to refute. The
dominant noise in a coseismic interferogram is atmospheric delay, and it is
**spatially correlated** over kilometres to tens of kilometres -- the same scale
as the deformation being fitted. A dense cluster of samples inside one
atmospheric cell is one measurement seen many times, but an unweighted normal
equation counts it as many independent ones, which is why sample *count* is a bad
proxy for information: 9898 samples yield ~68 effective data, and per-scene
efficiency varies 2x.

So: model the noise as a stationary random field, build its covariance ``Cd``
between the observations, and solve the whitened system ``L^-1 G m = L^-1 d``
where ``Cd = L L'``. Every downstream quantity -- the resolution matrix, the data
importance, the ramp diagnostics -- then has to use the same whitening, or they
describe a different problem from the one solved.

⚠️ **What this does NOT buy, measured rather than assumed: a better slip
estimate.** The natural argument for a covariance model is that a smooth
correlated blob is the shape a shallow slip patch makes, so an inversion that
does not know the noise is correlated will fit it with slip. On this problem that
does not happen. Swept over 24 smoothing weights and compared at each arm's own
optimum, the whitened solve recovers a planted patch *worse* than the plain one
-- 0.147 m against 0.069 m rms, on 5 of 5 seeds -- and it stays worse at SNR 2.0,
0.4 and 0.1, at matched effective rank up to 20, and at quadtree cells both 0.83x
and 0.31x the correlation length.

The reason is that generalised least squares is minimum variance only among
**unbiased** estimators, and this one is emphatically biased: bound constrained,
and damped hard enough to resolve tens of parameters out of hundreds, so the
regulariser rather than the data weighting sets the error. An exponential ``Cd``
has least variance at short wavelengths, so whitening up-weights precisely the
components the smoothing exists to suppress; the two pull against each other.
It is the same lesson the mesh work reached from the other side -- the
regularisation dominates.

**The case for ``Cd`` is therefore about honesty, not accuracy.** It makes
:attr:`~nisar_tools.slip.inversion.SlipModel.chi_square` mean something, it makes
the resolution and data-importance numbers describe the system that was actually
solved, and it states the noise instead of assuming it away. Reach for it when
the question is *how well is this constrained*, not *what is the best slip model*.

**What it does change, measured on the real six-scene problem: which scenes the
evidence comes from.** At matched total effective rank (~30 of 1836 parameters),
the per-scene share of ``diag(N)`` shifts, and it shifts **monotonically with each
scene's own sill**:

.. code-block:: text

    scene   sill      share vs white weighting
    D135    13.4 mm   1.35x
    D134    16.3 mm   1.31x
    A061    18.1 mm   0.82x
    A162    20.7 mm   0.82x
    D126    27.3 mm   0.74x
    D054    40.9 mm   0.73x

Quiet scenes gain, noisy scenes lose, in exact rank order, and the spread across
scenes widens from 1.43x to 2.02x. Under the old weighting a track's influence
came from its sample count alone, so a noisy scene and a quiet one counted the
same per sample; this is that error being corrected. Note the split falls cleanly
along the two processing chains -- the two ALOS scenes are the quiet ones.

⚠️ **Do not compare *total* effective data between the two weightings as if it
measured something.** ``trace(R) == trace(N)`` is an identity this module
enforces, so ``effective data == effective rank + n_ramp`` exactly, and matching
on rank forces the totals to agree at 1.000 whatever ``Cd`` does. Only the
*distribution* is free to move. (Recorded because the first attempt at this
measurement was that tautology.)

⚠️ **Existing smoothing weights do not transfer.** The old diagonal weights were
``1/sqrt(n)``, order 0.02; whitening divides by sigma instead, so the data block
grows by ~30x relative to the smoothing block and the same nominal ``lambda``
leaves the problem far less damped. Re-run the L-curve.

**What is approximated here, deliberately.** An observation is a quadtree cell
*average*, not a point. Averaging ``n`` pixels divides the uncorrelated part of
the variance by ``n`` and leaves the correlated part essentially untouched, so
this module uses

.. code-block:: text

    C[i, i] = sill + nugget / n_i
    C[i, j] = sill * exp(-d_ij / range)        (i != j)

which is first order in ``cell_size / range``. The exact treatment integrates the
covariance over both cells (what InSamp does); at the postings here the cells are
well inside the correlation length, and :func:`cell_ratio_warning` says so when
they are not.

**Tracks are whitened independently.** Two tracks are different dates and
different atmospheres, so their noise is uncorrelated by construction and ``Cd``
is block diagonal. That is not only physically right, it is what makes this
affordable: six blocks of ~1600 are a few MB each, against 900 MB for one dense
matrix over 10 000 observations.
"""

import warnings
from dataclasses import dataclass, replace

import numpy as np
import scipy.linalg as sla
from scipy.optimize import least_squares

from .diagnostics import DEFAULT_MIN_DISTANCE, _blocks

#: Covariance shapes available to :func:`fit_covariance`. ``"exponential"`` is
#: Lohman & Simons' and InSamp's, and is the default for that reason; the other
#: two are here because an atmospheric field is not always exponential and a
#: shape that fits visibly worse should be visible, not assumed away.
COVARIANCE_KINDS = ("exponential", "gaussian", "matern32")

#: Block size for the variogram's own reduction, metres. Independent of the
#: quadtree: this measures the field the quadtree will be told to sample, so it
#: must not be built on the quadtree's own cells.
DEFAULT_BLOCK = 2000.0

#: Number of lag bins.
DEFAULT_N_BINS = 25

#: Fit only out to this fraction of the largest separation. Beyond about a third
#: of the scene the estimator is built from few, highly dependent pairs and its
#: scatter swamps the shape it is meant to constrain -- the classic reason a
#: variogram fit runs away to an enormous range.
DEFAULT_MAX_LAG_FRACTION = 1.0 / 3.0

#: Polynomial degree removed before the variogram is measured, and it **must
#: match the nuisance ramp the inversion solves** -- ``ramp="quadratic"`` here,
#: hence 2.
#:
#: ⚠️ This is not a tuning knob, it is the difference between a covariance and
#: nonsense. Measured on the four real NISAR Venezuela scenes, which are stored
#: un-deramped: at degree 1 the fit **collapses** -- sill 0.0 mm, nugget
#: 174-501 mm, range 363-816 km against a 108 km maximum lag, i.e. the variogram
#: never plateaus and the fit puts everything in an unbounded trend. At degree 2
#: the same scenes give sill 18-41 mm, nugget 1.2-3.3 mm, range 22-29 km, which
#: is what tropospheric delay actually looks like. Degree 3 starts eating the
#: atmosphere itself (range falls to 7-25 km).
#:
#: The two ALOS scenes are the control: already deramped upstream, they barely
#: move between degree 1 and 2 (16.5 -> 16.3 mm sill, 36 -> 33 km range). So the
#: degree costs almost nothing on a clean scene and rescues a dirty one.
DEFAULT_DETREND_DEGREE = 2

#: A fitted range beyond this multiple of the largest lag the variogram covers
#: means the curve never plateaued, so the "sill" and "range" are an
#: extrapolation. :func:`fit_covariance` warns; see
#: :data:`DEFAULT_DETREND_DEGREE` for the usual cause.
RANGE_EXTRAPOLATION_LIMIT = 1.0

#: Smallest nugget kept, as a fraction of the sill. A fitted nugget of exactly
#: zero makes ``Cd`` singular the moment two cells coincide, and no real
#: interferogram has zero uncorrelated noise anyway.
MIN_NUGGET_FRACTION = 1e-3

#: Above this ratio of cell size to correlation length the point-separation
#: approximation above stops being first order and
#: :func:`cell_ratio_warning` complains.
CELL_RATIO_LIMIT = 0.5


@dataclass(frozen=True)
class CovarianceModel:
    """A stationary isotropic noise model, in metres of line-of-sight.

    ``sill`` is the variance of the *correlated* part, ``nugget`` the variance of
    the uncorrelated part, both in m^2; ``range_m`` is the correlation length in
    metres. Total point variance is ``sill + nugget``.

    Note the split matters more than the total: the nugget averages down inside a
    quadtree cell and the sill does not, so two models with the same total
    variance imply very different weights on a coarsely sampled scene.
    """

    sill: float
    range_m: float
    nugget: float
    kind: str = "exponential"

    def __post_init__(self):
        if self.kind not in COVARIANCE_KINDS:
            raise ValueError(
                f"kind must be one of {COVARIANCE_KINDS}, got {self.kind!r}")
        for name in ("sill", "range_m", "nugget"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative, got {value}")
        if self.range_m == 0:
            raise ValueError("range_m must be positive; a zero range is pure nugget, "
                             "which is what sill=0 says")

    @property
    def total_variance(self):
        """Variance of a single pixel, m^2."""
        return float(self.sill + self.nugget)

    def correlation(self, distance):
        """Correlated part's shape, 1 at zero separation and falling to 0."""
        h = np.asarray(distance, dtype=float) / float(self.range_m)
        if self.kind == "exponential":
            return np.exp(-h)
        if self.kind == "gaussian":
            return np.exp(-(h ** 2))
        root3 = np.sqrt(3.0) * h
        return (1.0 + root3) * np.exp(-root3)

    def covariance(self, distance):
        """``C(h)``, **excluding** the nugget, which lives only at zero lag."""
        return float(self.sill) * self.correlation(distance)

    def semivariance(self, distance):
        """``gamma(h)`` -- what :func:`empirical_variogram` estimates."""
        h = np.asarray(distance, dtype=float)
        gamma = float(self.nugget) + float(self.sill) * (1.0 - self.correlation(h))
        return np.where(h == 0, 0.0, gamma)

    def to_dict(self):
        return {"sill": float(self.sill), "range_m": float(self.range_m),
                "nugget": float(self.nugget), "kind": str(self.kind)}

    @classmethod
    def from_dict(cls, data):
        return cls(sill=float(data["sill"]), range_m=float(data["range_m"]),
                   nugget=float(data["nugget"]),
                   kind=str(data.get("kind", "exponential")))

    def __str__(self):
        return (f"{self.kind}: sill {1e3 * np.sqrt(self.sill):.1f} mm, "
                f"nugget {1e3 * np.sqrt(self.nugget):.1f} mm, "
                f"range {self.range_m / 1e3:.1f} km")


def _detrend(x, y, values, degree=1):
    """Remove a plane. A long-wavelength ramp is not what this is measuring.

    An orbital or ionospheric ramp is coherent across the whole scene, so it adds
    a term to the empirical variogram that grows without bound with lag -- which a
    fit reads as an enormous correlation length and a sill it never reaches. Every
    published recipe removes a trend first; this is that step, and it is the
    single most important line in the estimator.
    """
    sx = np.ptp(x) or 1.0
    sy = np.ptp(y) or 1.0
    u, v = (x - x.mean()) / sx, (y - y.mean()) / sy
    columns = [u ** (k - j) * v ** j
               for k in range(int(degree) + 1) for j in range(k + 1)]
    design = np.column_stack(columns)
    coef, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ coef


def empirical_variogram(los_stack, trace, frame, pair=0, block=DEFAULT_BLOCK,
                        min_distance=None, min_valid_fraction=0.5,
                        n_bins=DEFAULT_N_BINS, max_lag=None,
                        max_lag_fraction=DEFAULT_MAX_LAG_FRACTION,
                        detrend=DEFAULT_DETREND_DEGREE, max_blocks=4000,
                        seed=0):
    """Binned semivariance of one scene's far field.

    Reduces the scene onto the same fixed ``block``-metre grid
    :func:`~nisar_tools.slip.diagnostics.noise_floor` uses -- deliberately not the
    quadtree, since this measures the field the quadtree is about to be
    configured from -- keeps blocks farther than ``min_distance`` from ``trace``,
    removes a plane, and bins ``0.5 * (z_i - z_j)^2`` by separation.

    Returns an :class:`xarray.Dataset` on a ``lag`` dimension carrying
    ``semivariance`` and the pair ``count`` behind each bin, with the block
    variance and the retained block count in ``attrs``.

    ``max_blocks`` caps the O(n^2) pair count by random subsampling; 4000 blocks
    is 8 M pairs, which is a fraction of a second, and more blocks buy nothing
    because the pairs are not independent.
    """
    import xarray as xr

    if min_distance is None:
        min_distance = DEFAULT_MIN_DISTANCE

    b = _blocks(los_stack, trace, frame, pair=pair, block=block)
    keep = (
        np.isfinite(b["mean"])
        & (b["count"] >= min_valid_fraction * b["block_pixels"])
        & (b["distance"] > float(min_distance))
    )
    if keep.sum() < 3:
        raise ValueError(
            f"Only {int(keep.sum())} blocks survive: none are farther than "
            f"{min_distance / 1e3:.0f} km from the trace with at least "
            f"{min_valid_fraction:.0%} valid pixels. Lower min_distance, lower "
            "min_valid_fraction, or widen the scene."
        )

    x, y, z = b["x"][keep], b["y"][keep], b["mean"][keep]
    if x.size > int(max_blocks):
        pick = np.random.default_rng(seed).choice(x.size, int(max_blocks),
                                                  replace=False)
        x, y, z = x[pick], y[pick], z[pick]
    degree = int(detrend) if detrend is not True else 1
    if degree > 0:
        z = _detrend(x, y, z, degree=degree)

    iu, ju = np.triu_indices(x.size, k=1)
    separation = np.hypot(x[iu] - x[ju], y[iu] - y[ju])
    squared = 0.5 * (z[iu] - z[ju]) ** 2

    if max_lag is None:
        max_lag = float(max_lag_fraction) * float(separation.max())
    inside = separation <= float(max_lag)
    if inside.sum() < int(n_bins):
        raise ValueError(
            f"Only {int(inside.sum())} block pairs fall inside max_lag="
            f"{max_lag:.0f} m, fewer than the {int(n_bins)} bins asked for"
        )

    edges = np.linspace(0.0, float(max_lag), int(n_bins) + 1)
    which = np.clip(np.digitize(separation[inside], edges) - 1, 0, int(n_bins) - 1)
    count = np.bincount(which, minlength=int(n_bins))
    total = np.bincount(which, weights=squared[inside], minlength=int(n_bins))
    lag = np.bincount(which, weights=separation[inside], minlength=int(n_bins))

    with np.errstate(invalid="ignore", divide="ignore"):
        gamma = np.where(count > 0, total / np.maximum(count, 1), np.nan)
        centre = np.where(count > 0, lag / np.maximum(count, 1),
                          0.5 * (edges[:-1] + edges[1:]))

    return xr.Dataset(
        {"semivariance": ("lag", gamma), "count": ("lag", count.astype(int))},
        coords={"lag": centre},
        attrs={
            "block_size": float(b["block_size"]),
            "n_blocks": int(x.size),
            "block_variance": float(np.var(z)),
            "min_distance": float(min_distance),
            "detrend_degree": int(degree),
            "max_lag": float(max_lag),
            # Pixels averaged into each block, and the median *within*-block
            # variance. The second is how the nugget is actually measured -- see
            # `nugget_from_blocks`, and the warning in `fit_covariance` about why
            # the variogram's own intercept is not usable for it.
            "block_count": float(np.median(b["count"][keep])),
            "within_block_variance": float(
                np.median(b["std"][keep & np.isfinite(b["std"])] ** 2)),
        },
    )


def _mean_correlation_in_block(model, block_size, count):
    """Mean correlation over the ``count`` pixels of one block, ordered pairs.

    On the block's own pixel lattice, not a continuum: at the default 2 km block
    over an 80 m posting that distinction is immaterial, but on the coarse grids
    a synthetic uses -- four pixels in a block -- a continuum average is visibly
    wrong, and the four points are what the variance was actually computed from.
    """
    side = max(int(round(np.sqrt(max(float(count), 1.0)))), 1)
    if side == 1:
        return 1.0
    step = float(block_size) / side
    t = (np.arange(side) + 0.5) * step
    xx, yy = np.meshgrid(t, t)
    px, py = xx.ravel(), yy.ravel()
    h = np.hypot(px[:, None] - px[None, :], py[:, None] - py[None, :])
    return float(np.mean(model.correlation(h)))


def nugget_from_blocks(variogram, model):
    """Per-pixel nugget from the within-block scatter, given the correlated part.

    ⚠️ **This is measured here rather than read off the variogram's intercept,
    and that is not a stylistic choice.** A block variogram carries no lag shorter
    than one block, so on a 2 km block against a 12 km correlation length the
    first bin already sits a sixth of the way up the curve. There is no short-lag
    information left to separate a nugget from a slightly shorter range, and a
    free three-parameter fit resolves the ambiguity by driving the nugget to
    zero -- measured on synthetic fields with a known 10 mm nugget, the free fit
    returns 0.6 mm. The within-block scatter still has that information, because
    it is computed from the pixels themselves.

    The algebra is exact rather than approximate. For ``n`` samples with
    covariance ``C``, the *population* variance that
    :func:`~nisar_tools.slip.diagnostics._blocks` computes has expectation

    .. code-block:: text

        E[s^2] = C(0) - mean(C)  =  sill * (1 - rho_bar)  +  nugget * (1 - 1/n)

    so both corrections matter and neither is optional: the ``1 - 1/n`` is the
    familiar population-versus-sample bias (a full 25% at the four pixels a
    2 km block holds on a 1 km grid), and ``sill * (1 - rho_bar)`` is the part of
    the scatter that is correlated field rather than noise.

    ⚠️ The nugget stays the least-determined of the three parameters -- it is
    recovered to roughly 30% on synthetics where the sill and range come back to
    10% -- because it is a small difference between two larger measured numbers.
    It sets only the diagonal of ``Cd``, and the diagonal is the part cell
    averaging shrinks anyway, so the sensitivity of anything downstream to it is
    correspondingly mild.
    """
    within = float(variogram.attrs.get("within_block_variance", 0.0))
    block = float(variogram.attrs.get("block_size", 0.0))
    count = float(variogram.attrs.get("block_count", 1.0))
    floor = MIN_NUGGET_FRACTION * max(model.sill, 1e-12)
    if block <= 0 or count <= 1:
        return max(within, floor)

    # ⚠️ The stored number is a *median* of block variances -- `noise_floor` takes
    # one for robustness against blocks that still hold signal, and this reuses
    # its reduction. A sample variance is chi-square distributed and its median
    # sits below its mean, by 21% at the three degrees of freedom a four-pixel
    # block has. Left uncorrected that is a systematic 30% under-estimate of the
    # nugget, which is precisely the size of the bias measured before this line
    # existed. On a real scene (625 pixels to a 2 km block at 80 m posting) the
    # factor is 0.9993 and the correction does nothing, which is the point: it
    # costs nothing where it is unnecessary.
    from scipy.stats import chi2

    dof = max(count - 1.0, 1.0)
    within = within / (float(chi2.median(dof)) / dof)

    rho = _mean_correlation_in_block(model, block, count)
    correlated = float(model.sill) * (1.0 - rho)
    return max((within - correlated) / (1.0 - 1.0 / count), floor)


def fit_covariance(variogram, kind="exponential", weighted=True, nugget=None):
    """Least-squares fit of a :class:`CovarianceModel` to a binned variogram.

    Bins are weighted by their pair count by default -- the short-lag bins hold
    far fewer pairs than the long-lag ones on a gridded field, and they are the
    bins that determine the nugget.

    ``nugget`` fixes the uncorrelated variance and fits only ``sill`` and
    ``range_m``. **This is the mode you want**, and :func:`scene_covariance` uses
    it: a block variogram carries no lag shorter than one block, so it cannot
    separate a nugget from a slightly shorter range and a free three-parameter fit
    collapses the nugget to zero. :func:`nugget_from_blocks` measures it from the
    within-block scatter instead, which does have that information. ``None``
    leaves all three free, which is right only when the variogram came from point
    data rather than block means.

    The sill and range are well determined either way: averaging a correlated
    field over a block small against its correlation length changes the sill only
    at second order, and :func:`empirical_variogram`'s 2 km default is well
    inside any atmospheric range worth modelling.

    The nugget is floored at :data:`MIN_NUGGET_FRACTION` of the sill, because a
    zero nugget makes ``Cd`` singular as soon as two cells land on the same point.
    """
    if kind not in COVARIANCE_KINDS:
        raise ValueError(f"kind must be one of {COVARIANCE_KINDS}, got {kind!r}")

    lag = np.asarray(variogram["lag"].values, dtype=float)
    gamma = np.asarray(variogram["semivariance"].values, dtype=float)
    count = np.asarray(variogram["count"].values, dtype=float)
    good = np.isfinite(gamma) & (count > 0) & (lag > 0)
    if good.sum() < 3:
        raise ValueError(f"Only {int(good.sum())} usable variogram bins; need 3")
    lag, gamma, count = lag[good], gamma[good], count[good]
    weight = np.sqrt(count) if weighted else np.ones_like(count)

    # Start from the shape of the curve itself rather than from constants: the
    # plateau is roughly the largest bin, the nugget roughly the smallest, and the
    # range roughly where the curve first reaches 63% of the way up.
    sill0 = max(float(np.nanmax(gamma)), 1e-12)
    nugget0 = max(float(np.nanmin(gamma)), MIN_NUGGET_FRACTION * sill0)
    reached = lag[gamma >= nugget0 + 0.63 * (sill0 - nugget0)]
    range0 = float(reached[0]) if reached.size else float(np.median(lag))

    if nugget is None:
        def residual(p):
            model = CovarianceModel(sill=abs(p[0]), range_m=abs(p[1]) + 1e-6,
                                    nugget=abs(p[2]), kind=kind)
            return weight * (model.semivariance(lag) - gamma)

        fit = least_squares(
            residual,
            x0=[max(sill0 - nugget0, 1e-12), max(range0, 1e-6), nugget0],
            bounds=([0.0, 1e-6, 0.0], [np.inf, np.inf, np.inf]),
        )
        sill, range_m, fitted = (abs(float(v)) for v in fit.x)
    else:
        fitted = abs(float(nugget))

        def residual(p):
            model = CovarianceModel(sill=abs(p[0]), range_m=abs(p[1]) + 1e-6,
                                    nugget=fitted, kind=kind)
            return weight * (model.semivariance(lag) - gamma)

        fit = least_squares(
            residual,
            x0=[max(sill0 - nugget0, 1e-12), max(range0, 1e-6)],
            bounds=([0.0, 1e-6], [np.inf, np.inf]),
        )
        sill, range_m = (abs(float(v)) for v in fit.x)

    model = CovarianceModel(
        sill=sill, range_m=range_m,
        nugget=max(fitted, MIN_NUGGET_FRACTION * max(sill, 1e-12)),
        kind=kind,
    )
    _warn_if_extrapolated(model, variogram)
    return model


def _warn_if_extrapolated(model, variogram):
    """Say so when the variogram never plateaued inside the lags it covers.

    A range past the largest measured lag means the fit is extrapolating a curve
    it never saw level off, and the sill it reports is whatever that extrapolation
    needed rather than a variance. It is a loud, checkable signature of a trend
    left in the data -- see :data:`DEFAULT_DETREND_DEGREE` -- and it is worth
    catching here because the resulting model is not obviously wrong downstream:
    it whitens, it factorises, and it quietly tells the inversion the noise is
    coherent across the whole scene.
    """
    max_lag = float(variogram.attrs.get("max_lag", 0.0))
    if max_lag <= 0:
        return
    if model.range_m <= RANGE_EXTRAPOLATION_LIMIT * max_lag:
        return
    degree = variogram.attrs.get("detrend_degree", "?")
    warnings.warn(
        f"fitted range {model.range_m / 1e3:.0f} km exceeds the largest lag "
        f"measured ({max_lag / 1e3:.0f} km), so the variogram never plateaued and "
        f"this sill/range pair is an extrapolation. The usual cause is a trend "
        f"the detrend did not remove -- this variogram removed degree "
        f"{degree}. Match it to the ramp the inversion solves (see "
        f"DEFAULT_DETREND_DEGREE), or widen the scene.",
        RuntimeWarning, stacklevel=3,
    )


def scene_covariance(los_stack, trace, frame, kind="exponential", **kwargs):
    """:func:`empirical_variogram` then :func:`fit_covariance`, in one call.

    Returns ``(model, variogram)`` -- the variogram as well, because a fit whose
    curve does not follow its bins is the failure mode here and it is only
    visible by looking.
    """
    weighted = kwargs.pop("weighted", True)
    variogram = empirical_variogram(los_stack, trace, frame, **kwargs)

    # Two passes. The first fits the shape with the nugget free, only so that
    # `nugget_from_blocks` has a sill and a range to subtract the correlated
    # part's within-block variance with; the second refits sill and range with
    # the measured nugget held. One iteration is enough because the sill and
    # range barely move -- they are constrained by lags the nugget does not
    # reach.
    rough = fit_covariance(variogram, kind=kind, weighted=weighted)
    nugget = nugget_from_blocks(variogram, rough)
    return (fit_covariance(variogram, kind=kind, weighted=weighted, nugget=nugget),
            variogram)


def cell_ratio_warning(obs, model, track=None):
    """Message naming cells too large for the point approximation, or ``None``.

    ``C[i, j] = C(d_ij)`` treats a cell as a point. That is first order in
    ``cell_size / range``; past :data:`CELL_RATIO_LIMIT` the neglected term is
    not small and the covariance between adjacent cells is overstated.
    """
    ds = obs.ds if track is None else obs.ds.isel(
        obs=np.asarray(obs.ds["track"].values, dtype=object) == track)
    size = np.asarray(ds["cell_size"].values, dtype=float)
    if size.size == 0:
        return None
    ratio = float(np.max(size)) / float(model.range_m)
    if ratio <= CELL_RATIO_LIMIT:
        return None
    return (
        f"largest cell is {float(np.max(size)) / 1e3:.1f} km against a "
        f"{model.range_m / 1e3:.1f} km correlation length (ratio {ratio:.2f} > "
        f"{CELL_RATIO_LIMIT}); the cell-as-a-point covariance overstates the "
        "correlation between adjacent samples. Lower width_max, or integrate the "
        "covariance over the cells."
    )


def _pixel_counts(ds):
    """Pixels averaged into each cell, for the nugget's ``1/n`` reduction.

    ``count`` is written by newer :meth:`Observations.from_los` calls. An
    observation set persisted before that returns ``None``, and the caller then
    keeps the full nugget on the diagonal -- which over-states the variance of a
    large cell rather than under-stating it, so an old store stays usable and is
    merely conservative.
    """
    if "count" not in ds:
        return None
    n = np.asarray(ds["count"].values, dtype=float)
    return np.maximum(n, 1.0)


def observation_covariance(obs, model, track=None):
    """Dense ``Cd`` block for one track, m^2.

    ``C[i, j] = sill * corr(d_ij)`` off the diagonal and ``sill + nugget / n_i``
    on it -- see the module docstring for why the nugget is the part that
    averages down and the sill is not.
    """
    ds = obs.ds
    if track is not None:
        ds = ds.isel(obs=np.asarray(ds["track"].values, dtype=object) == track)
    x = np.asarray(ds["x"].values, dtype=float)
    y = np.asarray(ds["y"].values, dtype=float)
    if x.size == 0:
        raise ValueError(f"No observations on track {track!r}")

    separation = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    cd = model.covariance(separation)

    counts = _pixel_counts(ds)
    nugget = float(model.nugget) / (counts if counts is not None else 1.0)
    cd[np.diag_indices_from(cd)] = float(model.sill) + nugget
    return cd


def whitening_factors(obs, models, jitter=1e-10):
    """Lower-triangular Cholesky factor of ``Cd`` per track.

    ``models`` is a ``{track: CovarianceModel}`` mapping, or one model applied to
    every track. Returns ``{track: L}`` with ``Cd = L L'``, which is what
    :meth:`~nisar_tools.slip.inversion.SlipInversion.whiten` solves against.

    ``jitter`` is added to the diagonal as a fraction of its mean before
    factorising. The nugget floor already makes ``Cd`` positive definite in exact
    arithmetic; this covers the case where it is only *just* so and rounding
    takes it over.
    """
    tracks = [str(t) for t in dict.fromkeys(
        np.asarray(obs.ds["track"].values, dtype=object))]
    if isinstance(models, CovarianceModel):
        models = {t: models for t in tracks}
    missing = [t for t in tracks if t not in models]
    if missing:
        raise ValueError(
            f"No covariance model for track(s) {missing}. Tracks present: "
            f"{tracks}. Pass one CovarianceModel to use for all of them, or a "
            "mapping with an entry per track."
        )

    factors = {}
    for track in tracks:
        cd = observation_covariance(obs, models[track], track=track)
        cd[np.diag_indices_from(cd)] += float(jitter) * float(np.mean(np.diag(cd)))
        try:
            factors[track] = sla.cholesky(cd, lower=True)
        except sla.LinAlgError as exc:                        # pragma: no cover
            raise ValueError(
                f"Covariance for track {track!r} is not positive definite "
                f"({exc}). This usually means a near-zero nugget with coincident "
                "samples; raise the nugget or drop duplicate observations."
            ) from exc
    return factors


def scaled_to_unit_variance(model, reference=None):
    """The same shape, rescaled so the total variance is 1.

    Whitening by ``Cd^-1/2`` puts the residual in units of its own sigma, which
    is what makes chi-square per degree of freedom meaningful. If a caller would
    rather keep metres, this is the conversion, and ``reference`` lets several
    tracks share one scale so their *relative* weights survive it.
    """
    scale = float(reference if reference is not None else model.total_variance)
    if scale <= 0:
        raise ValueError("reference variance must be positive")
    return replace(model, sill=model.sill / scale, nugget=model.nugget / scale)


def correlated_field(shape, spacing, model, seed=0, pad=3):
    """Draw a Gaussian field with the covariance ``model`` describes.

    The forward direction of this module: everything above estimates a
    :class:`CovarianceModel` from a residual, and this synthesises a residual
    from a model. Circulant embedding -- build ``C`` on the wrapped lag grid,
    take its DFT, filter white noise by the square root, transform back.

    Used to give a synthetic recovery experiment (see
    :mod:`nisar_tools.slip.calibrate`) noise with the same structure as the real
    scenes, which is what makes such an experiment's answer about damping
    transferable to the real data.

    ⚠️ **This must never come to share code with the estimator above it.** The
    covariance tests verify the fit by round trip -- synthesise from known
    parameters, demand them back -- and that argument only holds while the two
    directions are independent implementations. ``tests/slip_synthetic.py``
    keeps its own copy as the oracle for exactly this reason, and a test pins
    the two against each other; if you refactor them onto a shared helper, that
    test starts passing for the wrong reason.

    ``pad`` embeds in a larger periodic grid so wrap-around does not contaminate
    the lags of interest; 3x is comfortable for an exponential, whose correlation
    is under 5% past three ranges. The nugget is added separately because it is
    white by definition, and folding it into ``C`` would put a discontinuity at
    zero lag that the FFT cannot represent.
    """
    ny, nx = int(shape[0]), int(shape[1])
    py, px = int(pad * ny), int(pad * nx)
    rng = np.random.default_rng(seed)

    dy = np.minimum(np.arange(py), py - np.arange(py)) * float(spacing)
    dx = np.minimum(np.arange(px), px - np.arange(px)) * float(spacing)
    h = np.hypot(dy[:, None], dx[None, :])

    if model.kind == "exponential":
        shape_fn = np.exp(-h / float(model.range_m))
    elif model.kind == "gaussian":
        shape_fn = np.exp(-((h / float(model.range_m)) ** 2))
    elif model.kind == "matern32":
        root3 = np.sqrt(3.0) * h / float(model.range_m)
        shape_fn = (1.0 + root3) * np.exp(-root3)
    else:
        raise ValueError(f"cannot synthesise kind {model.kind!r}")

    # C is real and symmetric so its DFT is real; clip the small negative
    # eigenvalues the embedding leaves rather than pretending they are absent.
    spectrum = np.maximum(np.fft.fft2(float(model.sill) * shape_fn).real, 0.0)
    white = rng.standard_normal((py, px))
    field = np.real(np.fft.ifft2(np.fft.fft2(white) * np.sqrt(spectrum)))
    field = field[:ny, :nx]

    if model.nugget:
        field = field + np.sqrt(float(model.nugget)) * rng.standard_normal((ny, nx))
    return field
