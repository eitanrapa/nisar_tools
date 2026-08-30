"""Pins the spatially correlated noise model and the whitening it drives.

Two things have to hold, and they pull in opposite directions:

* **The estimator must recover a covariance it did not generate.** So the fields
  here are synthesised *from* a covariance by circulant embedding
  (``slip_synthetic.correlated_noise``) and the fit is asked for the parameters
  back. Fitting the same expression the code composes would prove nothing.
* **Nothing may move when no covariance is supplied.** Every run before this
  weighted by ``1/sqrt(n)`` per track, and `covariance=None` has to stay
  bit-for-bit that -- which is what makes this safe to land ahead of the
  experiments that depend on it.

The one that caught a real design flaw is
``test_the_variogram_cannot_see_the_nugget_but_the_blocks_can``. A block
variogram carries no lag shorter than one block, so its shortest bin already sits
a sixth of the way up a 12 km exponential and a free three-parameter fit has
nothing left to separate the nugget from a slightly shorter range -- it drives the
nugget to zero. The first version of this module read the nugget straight off
that fit, which would have handed every quadtree cell a wildly over-confident
weight. It is measured from within-block scatter instead.
"""

import warnings

import numpy as np
import pytest

from nisar_tools.slip import (
    CovarianceModel,
    FaultMesh,
    FaultTrace,
    Observations,
    SlipInversion,
)
from nisar_tools.slip import covariance as C
from nisar_tools.slip import resolution as R

import slip_synthetic as ss

# The signal fixture deliberately samples at ~12 km cells against a 12 km
# correlation length, which is exactly the regime `cell_ratio_warning` exists to
# flag -- it is the coarse end of the sweep and it is meant to be there. The
# warning itself is pinned by `test_cells_too_large_for_the_point_approximation
# _are_reported`; silencing it here keeps the suite's warning output meaningful
# instead of burying real ones under twenty expected repeats.
pytestmark = pytest.mark.filterwarnings(
    "ignore:.*cell-as-a-point covariance.*:RuntimeWarning")

_LON = np.array([-69.0, -68.5, -68.0, -67.5, -67.0, -66.5, -66.0])
_LAT = np.array([10.55, 10.56, 10.57, 10.58, 10.585, 10.59, 10.60])

#: What the synthetic fields are built with, and what the fits must return.
SILL = 4.0e-4          # (20 mm)^2 of correlated atmosphere
RANGE = 12_000.0       # metres
NUGGET = 1.0e-4        # (10 mm)^2 of white pixel noise


@pytest.fixture(scope="module")
def geometry():
    trace = FaultTrace.from_lonlat(_LON, _LAT, name="t")
    frame = trace.local_frame()
    mesh = FaultMesh.vertical(trace, frame, max_depth=20e3, edge_length=10e3)
    return trace, frame, mesh


def _noise_stack(frame, sill=SILL, range_m=RANGE, nugget=NUGGET, spacing=1000.0,
                 seed=0, half_x=110e3, half_y=70e3, ramp=0.0):
    """A scene of pure noise with a known covariance -- no deformation at all."""
    epsg = 32619
    x, y = ss._grid(frame, epsg, half_x, half_y, spacing)
    field = ss.correlated_noise((y.size, x.size), spacing, sill, range_m,
                                nugget=nugget, seed=seed)
    if ramp:
        xx, _ = np.meshgrid(x - x.mean(), y - y.mean())
        field = field + ramp * xx
    return ss._stack(field, x, y, ss.look_vector(**ss._geom("asc")), epsg, "asc")


# -- the model itself ------------------------------------------------------
def test_the_model_is_the_covariance_it_claims_to_be():
    """gamma and C are the same statement; check both ends and the crossover."""
    m = CovarianceModel(sill=SILL, range_m=RANGE, nugget=NUGGET)

    assert m.semivariance(0.0) == 0.0, "no variance between a point and itself"
    assert m.semivariance(1e9) == pytest.approx(SILL + NUGGET)
    assert m.total_variance == pytest.approx(SILL + NUGGET)

    # gamma(h) + C(h) = sill + nugget away from zero: the defining relation.
    h = np.array([1e3, 5e3, RANGE, 4 * RANGE])
    assert m.semivariance(h) + m.covariance(h) == pytest.approx(SILL + NUGGET)
    # An exponential is down to 1/e at exactly one range.
    assert m.covariance(RANGE) / SILL == pytest.approx(np.exp(-1.0))


@pytest.mark.parametrize("kind", C.COVARIANCE_KINDS)
def test_every_shape_starts_at_the_sill_and_decays(kind):
    m = CovarianceModel(sill=SILL, range_m=RANGE, nugget=NUGGET, kind=kind)
    h = np.linspace(0.0, 6 * RANGE, 60)
    corr = m.correlation(h)
    assert corr[0] == pytest.approx(1.0)
    assert np.all(np.diff(corr) <= 1e-12), "correlation must be non-increasing"
    assert corr[-1] < 0.05
    assert CovarianceModel.from_dict(m.to_dict()) == m


@pytest.mark.parametrize("bad", [
    {"sill": -1.0}, {"nugget": -1.0}, {"range_m": 0.0}, {"kind": "banana"},
    {"sill": np.nan},
])
def test_a_nonsensical_model_is_refused(bad):
    args = {"sill": SILL, "range_m": RANGE, "nugget": NUGGET, **bad}
    with pytest.raises(ValueError):
        CovarianceModel(**args)


# -- the estimator, against a field it did not generate --------------------
def test_the_fit_recovers_the_covariance_that_made_the_field(geometry):
    """The forward-model check: choose a covariance, synthesise, demand it back.

    Over several draws, because one scene of this size holds only ~200
    independent samples of a 12 km field and a single realisation scatters far
    more than the estimator is biased. Testing one seed would pin luck.
    """
    trace, frame, _ = geometry
    fits = [C.scene_covariance(_noise_stack(frame, seed=s), trace, frame,
                               min_distance=0.0, block=2000.0, n_bins=20)[0]
            for s in range(11, 19)]

    # The nugget is the one this module estimates specially, and it is unbiased.
    nuggets = np.array([f.nugget for f in fits])
    assert np.median(nuggets) == pytest.approx(NUGGET, rel=0.25), nuggets

    # Sill and range trade off against each other in any variogram fit -- a
    # longer range with a smaller sill draws nearly the same curve -- so they are
    # judged by the covariance they jointly predict, not one at a time.
    for lag in (4e3, 12e3, 24e3):
        got = np.median([f.covariance(lag) for f in fits])
        want = SILL * np.exp(-lag / RANGE)
        assert got == pytest.approx(want, rel=0.4), (lag, got, want)


def test_the_variogram_cannot_see_the_nugget_but_the_blocks_can(geometry):
    """Why `nugget_from_blocks` exists at all.

    A block variogram's shortest lag is one block, so a free three-parameter fit
    has nothing to separate a nugget from a slightly shorter range and drives the
    nugget to its floor. The within-block scatter still carries that information.
    """
    trace, frame, _ = geometry
    free, measured = [], []
    for seed in range(11, 17):
        variogram = C.empirical_variogram(_noise_stack(frame, seed=seed), trace,
                                          frame, min_distance=0.0, block=2000.0,
                                          n_bins=20)
        rough = C.fit_covariance(variogram)
        free.append(rough.nugget)
        measured.append(C.nugget_from_blocks(variogram, rough))

    assert np.median(free) < NUGGET / 10.0, np.median(free)
    assert np.median(measured) == pytest.approx(NUGGET, rel=0.25), np.median(measured)


def test_the_within_block_corrections_are_both_needed(geometry):
    """Population bias and correlated-field leakage, each material at n=4.

    `E[s^2] = sill*(1 - rho_bar) + nugget*(1 - 1/n)`. Dropping the `1 - 1/n`
    under-states the nugget by 25% at four pixels; dropping the correlated term
    over-states it, because a correlated field varies across a block too.
    """
    trace, frame, _ = geometry
    variogram = C.empirical_variogram(_noise_stack(frame, seed=21), trace, frame,
                                      min_distance=0.0, block=2000.0, n_bins=20)
    model = C.fit_covariance(variogram)
    count = variogram.attrs["block_count"]
    assert count == pytest.approx(4.0), "a 2 km block on a 1 km grid holds four"

    rho = C._mean_correlation_in_block(model, variogram.attrs["block_size"], count)
    assert 0.0 < rho < 1.0
    # A 2 km block against a 12 km range keeps most, but not all, of the
    # correlation -- so the leakage term is real and smaller than the nugget.
    assert rho > 0.8
    assert 0.0 < model.sill * (1.0 - rho) < NUGGET


@pytest.mark.filterwarnings("ignore:fitted range.*:RuntimeWarning")
def test_a_ramp_must_be_removed_or_the_range_runs_away(geometry):
    """A trend is coherent at every lag, so it reads as an enormous range.

    The extrapolation warning fires here by design -- that is the failure being
    demonstrated; `test_an_unplateaued_variogram_says_so` is what pins it.
    """
    trace, frame, _ = geometry
    # 200 mm across the scene -- a small orbital ramp, and far more than enough.
    stack = _noise_stack(frame, seed=13, ramp=1e-6)

    clean, _ = C.scene_covariance(stack, trace, frame, min_distance=0.0,
                                  block=2000.0, n_bins=20, detrend=True)
    dirty, _ = C.scene_covariance(stack, trace, frame, min_distance=0.0,
                                  block=2000.0, n_bins=20, detrend=False)

    assert clean.range_m == pytest.approx(RANGE, rel=0.6)
    # The failure is total, not subtle: with the ramp left in, the variogram
    # never plateaus, so the fit pushes the range out to hundreds of kilometres
    # and gives up on the sill entirely, dumping the whole variance in a nugget.
    assert dirty.range_m > 10.0 * clean.range_m, (clean, dirty)
    assert dirty.total_variance > 10.0 * clean.total_variance, (clean, dirty)


def test_an_unplateaued_variogram_says_so(geometry):
    """A range past the largest measured lag means the fit is extrapolating.

    That is the loud, checkable signature of a trend the detrend did not remove,
    and it matters because the resulting model is *not* obviously broken
    downstream -- it whitens, it factorises, and it quietly tells the inversion
    the noise is coherent across the entire scene. Measured on the four real
    NISAR scenes, which are stored un-deramped: removing only a plane gives
    ranges of 363-816 km against a 108 km maximum lag.
    """
    trace, frame, _ = geometry
    stack = _noise_stack(frame, seed=13, ramp=1e-6)
    variogram = C.empirical_variogram(stack, trace, frame, min_distance=0.0,
                                      block=2000.0, n_bins=20, detrend=0)

    with pytest.warns(RuntimeWarning, match="never plateaued"):
        C.fit_covariance(variogram)

    # And the properly detrended one is silent.
    clean = C.empirical_variogram(stack, trace, frame, min_distance=0.0,
                                  block=2000.0, n_bins=20)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        C.fit_covariance(clean)


def test_the_detrend_degree_must_match_the_ramp_the_inversion_solves(geometry):
    """A plane cannot remove a quadratic, and the failure is total, not gradual.

    This is why `DEFAULT_DETREND_DEGREE` is 2 rather than 1: the scenes this
    package inverts carry an unremoved ramp and the inversion solves a quadratic
    one per track.
    """
    trace, frame, _ = geometry
    x, y = ss._grid(frame, 32619, 110e3, 70e3, 1000.0)
    field = ss.correlated_noise((y.size, x.size), 1000.0, SILL, RANGE,
                                nugget=NUGGET, seed=31)
    xx, yy = np.meshgrid(x - x.mean(), y - y.mean())
    # ~120 mm across the scene, against ~22 mm of noise -- dominant, as a real
    # un-removed orbital ramp is.
    curved = field + 1e-11 * xx ** 2          # a quadratic, not a plane
    stack = ss._stack(curved, x, y, ss.look_vector(**ss._geom("asc")), 32619, "asc")

    kwargs = dict(min_distance=0.0, block=2000.0, n_bins=20)
    plane = C.empirical_variogram(stack, trace, frame, detrend=1, **kwargs)
    quad = C.empirical_variogram(stack, trace, frame, detrend=2, **kwargs)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        by_plane = C.fit_covariance(plane)
    by_quad = C.fit_covariance(quad)

    assert by_plane.range_m > 3.0 * by_quad.range_m, (by_plane, by_quad)
    assert by_quad.range_m == pytest.approx(RANGE, rel=0.6), by_quad


# -- Cd structure ----------------------------------------------------------
@pytest.fixture(scope="module")
def two_track_obs(geometry):
    """Two tracks of pure noise, sampled -- no signal, so Cd is all there is."""
    trace, frame, mesh = geometry
    parts = []
    for i, name in enumerate(("asc", "desc")):
        stack = _noise_stack(frame, seed=20 + i)
        stack.ds.attrs["direction"] = name
        parts.append(Observations.from_los(
            stack, name=name, frame=frame, trace=trace, rms_min=0.006,
            width_min=10000.0, width_max=60000.0, exclude_within=8000.0))
    return Observations.concat(parts, normalize="none")


def test_the_covariance_is_block_diagonal_across_tracks(two_track_obs):
    """Two tracks are two dates and two atmospheres; they cannot correlate."""
    obs = two_track_obs
    model = CovarianceModel(sill=SILL, range_m=RANGE, nugget=NUGGET)
    factors = C.whitening_factors(obs, model)

    assert set(factors) == {"asc", "desc"}
    track = np.asarray(obs.ds["track"].values, dtype=object)
    for name, factor in factors.items():
        assert factor.shape == (int((track == name).sum()),) * 2
        # Lower triangular, and a Cholesky factor has a positive diagonal.
        assert np.allclose(np.triu(factor, k=1), 0.0)
        assert np.all(np.diag(factor) > 0)


def test_the_diagonal_carries_the_nugget_reduced_by_the_cell_count(two_track_obs):
    """sill + nugget/n on the diagonal -- averaging kills white noise, not colour."""
    obs = two_track_obs
    model = CovarianceModel(sill=SILL, range_m=RANGE, nugget=NUGGET)
    cd = C.observation_covariance(obs, model, track="asc")

    ds = obs.ds.isel(obs=np.asarray(obs.ds["track"].values, dtype=object) == "asc")
    n = np.asarray(ds["count"].values, dtype=float)
    assert n.max() > n.min(), "a quadtree with one cell size cannot test this"

    assert np.diag(cd) == pytest.approx(SILL + NUGGET / n)
    assert np.allclose(cd, cd.T)
    # Off the diagonal it is the correlated part alone, falling with separation.
    x, y = ds["x"].values, ds["y"].values
    sep = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    off = ~np.eye(len(x), dtype=bool)
    assert cd[off] == pytest.approx(SILL * np.exp(-sep[off] / RANGE))


def test_an_observation_set_without_counts_keeps_the_whole_nugget(two_track_obs):
    """An Observations persisted before `count` existed must still work.

    Conservative rather than wrong: it over-states a coarse cell's variance, so an
    old store loses information rather than gaining false confidence.
    """
    obs = two_track_obs
    stale = Observations(obs.ds.drop_vars("count"))
    model = CovarianceModel(sill=SILL, range_m=RANGE, nugget=NUGGET)

    cd = C.observation_covariance(stale, model, track="asc")
    assert np.diag(cd) == pytest.approx(SILL + NUGGET)
    # And it is genuinely more conservative than the version that has counts.
    full = C.observation_covariance(obs, model, track="asc")
    assert np.all(np.diag(cd) >= np.diag(full) - 1e-18)


def test_cells_too_large_for_the_point_approximation_are_reported(two_track_obs):
    obs = two_track_obs
    tight = CovarianceModel(sill=SILL, range_m=200.0, nugget=NUGGET)
    loose = CovarianceModel(sill=SILL, range_m=500e3, nugget=NUGGET)

    assert C.cell_ratio_warning(obs, loose, track="asc") is None
    message = C.cell_ratio_warning(obs, tight, track="asc")
    assert message is not None and "correlation length" in message


# -- the whitening seam ----------------------------------------------------
@pytest.fixture(scope="module")
def signal_setup(geometry):
    """A real recovery problem: tapered slip plus correlated noise, two looks."""
    trace, frame, mesh = geometry
    slip = ss.tapered_slip(mesh, peak=-2.0)
    stacks = {}
    for i, name in enumerate(("asc", "desc")):
        stack = ss.forward_los_stack(mesh, slip, trace, frame, geometry=name,
                                     spacing=4000.0, seed=30 + i)
        field = ss.correlated_noise(stack.ds["los"].shape[1:], 4000.0,
                                    SILL, RANGE, nugget=NUGGET, seed=40 + i)
        stack.ds["los"] = stack.ds["los"] + field[None].astype(np.float32)
        stacks[name] = stack
    parts = [
        Observations.from_los(stack, name=name, frame=frame, trace=trace,
                              rms_min=0.006, width_min=10000.0,
                              width_max=60000.0, exclude_within=8000.0)
        for name, stack in stacks.items()
    ]
    return trace, frame, mesh, slip, parts


def test_whitening_without_a_covariance_is_exactly_the_old_weighting(signal_setup):
    """The backward-compatibility pin. Bit-identical, not merely close."""
    _, _, mesh, _, parts = signal_setup
    obs = Observations.concat(parts)
    inv = SlipInversion(mesh, obs, ramp="offset")

    assert inv.covariance is None
    w = np.asarray(obs.ds["weight"].values, dtype=float)
    design = np.hstack([inv.g, inv.ramp])

    assert np.array_equal(inv.whiten(inv.d), inv.d * w)
    assert np.array_equal(inv.whiten(design), design * w[:, None])


def test_a_covariance_and_a_count_normalisation_are_the_same_job(signal_setup):
    """Both set relative track weight; taking both states it twice."""
    _, _, mesh, _, parts = signal_setup
    model = CovarianceModel(sill=SILL, range_m=RANGE, nugget=NUGGET)

    counted = Observations.concat(parts, normalize="sqrt_count")
    with pytest.raises(ValueError, match="normalize"):
        SlipInversion(mesh, counted, ramp="offset", covariance=model)

    plain = Observations.concat(parts, normalize="none")
    assert SlipInversion(mesh, plain, ramp="offset",
                         covariance=model).covariance is model


def test_whitening_makes_the_residual_unit_variance(signal_setup):
    """Cd^-1/2 puts the misfit in units of its own sigma -- so chi2/n lands at 1.

    This is the property that makes `reduced_chi_square` readable at all, and it
    is checked on noise drawn from the very model being whitened by.
    """
    _, _, mesh, _, parts = signal_setup
    obs = Observations.concat(parts, normalize="none")
    model = CovarianceModel(sill=SILL, range_m=RANGE, nugget=NUGGET)
    inv = SlipInversion(mesh, obs, ramp="offset", covariance=model)

    # Whiten a draw from Cd itself: the result must be standard normal.
    rng = np.random.default_rng(7)
    draw = np.empty(obs.n)
    track = np.asarray(obs.ds["track"].values, dtype=object)
    for name, factor in inv._chol.items():
        at = np.flatnonzero(track == name)
        draw[at] = factor @ rng.standard_normal(at.size)

    whitened = inv.whiten(draw)
    assert whitened.std() == pytest.approx(1.0, rel=0.15)
    assert float(whitened @ whitened) / obs.n == pytest.approx(1.0, rel=0.2)


def test_trace_identity_survives_the_whitening(signal_setup):
    """trace(R) == trace(N) holds only if every site whitens the same A.

    It is the cross-check that catches a missed site, so it has to be pinned
    *with* a covariance in play and not only without one.
    """
    _, _, mesh, _, parts = signal_setup
    obs = Observations.concat(parts, normalize="none")
    model = CovarianceModel(sill=SILL, range_m=RANGE, nugget=NUGGET)
    inv = SlipInversion(mesh, obs, ramp="offset", covariance=model)

    report = R.resolution_report(inv, 1.0)
    assert abs(report.attrs["trace_identity_error"]) < 1e-8, report.attrs
    assert report.attrs["effective_rank"] > 0


def test_the_whitened_design_is_what_the_solver_uses(signal_setup):
    """`weighted_design` exists so the three resolution sites cannot drift apart."""
    _, _, mesh, _, parts = signal_setup
    obs = Observations.concat(parts, normalize="none")
    model = CovarianceModel(sill=SILL, range_m=RANGE, nugget=NUGGET)
    inv = SlipInversion(mesh, obs, ramp="offset", covariance=model)

    design = np.hstack([inv.g, inv.ramp])
    assert np.array_equal(R.weighted_design(inv), inv.whiten(design))


def test_whitening_rescales_what_the_smoothing_weight_means(signal_setup):
    """⚠️ Existing lambda values do not transfer to a whitened run.

    The diagonal weights were ``1/sqrt(n)`` -- about 0.02 here. Whitening divides
    by sigma instead, about 0.022 m, so the data block grows by ~30x relative to
    the smoothing block and the same nominal lambda leaves the problem far less
    damped. Any L-curve has to be re-run; a weight carried over from an
    unwhitened run is effectively a thousand times too small.
    """
    _, _, mesh, _, parts = signal_setup
    obs = Observations.concat(parts, normalize="none")
    model = CovarianceModel(sill=SILL, range_m=RANGE, nugget=NUGGET)

    plain = SlipInversion(mesh, obs, ramp="offset")
    white = SlipInversion(mesh, obs, ramp="offset", covariance=model)

    ratio = (np.abs(white.whiten(white.d)).mean()
             / np.abs(plain.whiten(plain.d)).mean())
    assert ratio > 10.0, ratio

    # Same lambda, very different damping -- which is the whole warning.
    from nisar_tools.slip import resolution as R
    assert (R.resolution_diagonal(white, 30.0).attrs["effective_rank"]
            > 3.0 * R.resolution_diagonal(plain, 30.0).attrs["effective_rank"])


def test_whitening_does_not_by_itself_improve_the_slip_estimate(signal_setup):
    """⚠️ **A measured negative result, pinned so it is not "fixed" later.**

    The intuition for a covariance model is that correlated noise looks like a
    shallow slip patch and gets fitted as one unless the inversion is told
    otherwise. That intuition does not survive contact with this problem: swept
    over 24 smoothing weights and compared at each arm's own optimum, the
    whitened solve recovers the planted slip *worse* than the plain one -- 0.147
    against 0.069 m rms, on 5 of 5 seeds, and it stays worse at SNR 2.0 / 0.4 /
    0.1, at matched effective rank up to 20, and at cells both 0.83x and 0.31x
    the correlation length.

    The reason is that GLS is the minimum-variance estimator only among
    **unbiased** ones. This estimator is nothing of the sort: it is bound
    constrained and damped hard enough to resolve tens of parameters out of
    hundreds, so the regulariser, not the data weighting, sets the error. And an
    exponential ``Cd`` has least variance at short wavelengths, so whitening
    up-weights exactly the components the smoothing exists to suppress. The two
    pull against each other.

    So the case for ``Cd`` is **not** a better point estimate. It is that
    ``chi_square`` becomes meaningful, that the resolution and data-importance
    numbers describe the system actually solved, and that the noise is stated
    rather than assumed. Those are the tests above; this one records what a
    covariance does not buy.
    """
    _, _, mesh, truth, parts = signal_setup
    obs = Observations.concat(parts, normalize="none")
    model = CovarianceModel(sill=SILL, range_m=RANGE, nugget=NUGGET)
    bounds = {"strike": (-10.0, 0.0), "dip": (-1.0, 1.0)}
    truth_ss = np.asarray(truth, dtype=float)[:mesh.n_elements]

    best = {}
    for name, inversion in (
        ("plain", SlipInversion(mesh, obs, ramp="offset")),
        ("whitened", SlipInversion(mesh, obs, ramp="offset", covariance=model)),
    ):
        errors = []
        for lam in np.logspace(-1, 4.5, 14):
            m = inversion.solve(smoothing=float(lam), max_iter=400, **bounds)
            if m.converged:
                errors.append(float(np.sqrt(np.mean(
                    (m.strike_slip - truth_ss) ** 2))))
        assert errors, f"{name} never converged"
        best[name] = min(errors)

    # Both recover the patch; neither is broken. The point is only that whitening
    # is not the thing that improves it.
    assert best["plain"] < 0.5 and best["whitened"] < 0.5, best
    assert best["whitened"] > best["plain"], (
        "Whitening beat the plain solve here. That contradicts a result measured "
        "over 5 seeds, 3 SNRs, matched rank and two sampling densities -- so "
        f"check what changed rather than relaxing this: {best}"
    )


def test_chi_square_is_absent_without_a_noise_model(signal_setup):
    """A statistic only meaningful against a stated noise model must not appear
    when none was stated.

    Without a covariance the whitening is a multiply by `1/sqrt(n)`, so the sum of
    squares lands around 1e-4 times the residual norm -- it looks like a
    chi-square, it reads as a spectacular fit, and it means nothing. Caught on the
    real six-scene L-curve, where the plain arm printed `chi2/n 0.000` at every
    weight.
    """
    _, _, mesh, _, parts = signal_setup
    obs = Observations.concat(parts)
    plain = SlipInversion(mesh, obs, ramp="offset")
    solved = plain.solve(smoothing=0.3, max_iter=200,
                         strike=(-10.0, 0.0), dip=(-1.0, 1.0))

    assert solved.chi_square is None
    assert solved.reduced_chi_square is None
    # The metrics that do not depend on a noise model are still there.
    assert solved.rms_misfit > 0 and solved.variance_reduction > 0


def test_chi_square_reports_the_metric_that_was_minimised(signal_setup):
    _, _, mesh, _, parts = signal_setup
    obs = Observations.concat(parts, normalize="none")
    model = CovarianceModel(sill=SILL, range_m=RANGE, nugget=NUGGET)
    inv = SlipInversion(mesh, obs, ramp="offset", covariance=model)

    solved = inv.solve(smoothing=0.3, max_iter=200,
                       strike=(-10.0, 0.0), dip=(-1.0, 1.0))

    assert solved.chi_square > 0
    assert solved.reduced_chi_square == pytest.approx(
        solved.chi_square / obs.n, rel=1e-12)
    # It is a different number from the unweighted rms, which is the whole point
    # of having both.
    assert solved.rms_misfit > 0
    assert not np.isclose(solved.reduced_chi_square, solved.rms_misfit)


def test_scaling_to_unit_variance_keeps_the_shape(two_track_obs):
    m = CovarianceModel(sill=SILL, range_m=RANGE, nugget=NUGGET)
    unit = C.scaled_to_unit_variance(m)

    assert unit.total_variance == pytest.approx(1.0)
    assert unit.range_m == m.range_m
    assert unit.sill / unit.nugget == pytest.approx(m.sill / m.nugget)


def test_a_track_with_no_model_is_named(two_track_obs):
    with pytest.raises(ValueError, match="desc"):
        C.whitening_factors(two_track_obs, {"asc": CovarianceModel(
            sill=SILL, range_m=RANGE, nugget=NUGGET)})


# -- uncertainty, and how much of it is the prior ---------------------------

def test_the_propagated_sigma_matches_repeated_bounded_solves(signal_setup):
    """⚠️ The forward-model check, and it is what set the default.

    Draw noise from the SAME Cd the covariance is built from, re-solve through the
    real bound-constrained solver, and compare the scatter with the prediction.
    Measured over 60 draws on the synthetic: `"propagated"` gives ratio **1.01**
    and per-parameter correlation **0.950**, while `"posterior"` is **11x too
    wide**.

    Both are right for their own question -- repeated observation varies only the
    data, which is what `"propagated"` predicts, while `M^-1` also carries a prior
    spread that no re-observation samples. But "uncertainty on a slip model"
    conventionally means the former, so that is the default.
    """
    _, _, mesh, _, parts = signal_setup
    obs = Observations.concat(parts, normalize="none")
    model = CovarianceModel(sill=SILL, range_m=RANGE, nugget=NUGGET)
    inversion = SlipInversion(mesh, obs, ramp="offset", covariance=model)
    n = inversion.basis.n_basis
    lam = 3.0
    # Loose bounds: the linear covariance describes the *unbounded* problem, so
    # this tests it in the regime it claims, rather than dodging the question.
    bounds = {"strike": (-20.0, 20.0), "dip": (-20.0, 20.0)}

    prop = R.posterior_covariance(inversion, lam)
    post = R.posterior_covariance(inversion, lam, kind="posterior")
    assert np.median(post[:n]) > 3.0 * np.median(prop[:n]), (
        "the prior term should dominate on a problem this heavily damped")

    clean = np.asarray(obs.ds["los"].values, dtype=float)
    track = np.asarray(obs.ds["track"].values, dtype=object)
    rng = np.random.default_rng(0)
    draws = []
    for _ in range(24):
        noisy = clean.copy()
        for name, factor in inversion._chol.items():
            at = np.flatnonzero(track == name)
            noisy[at] += factor @ rng.standard_normal(at.size)
        shaken = Observations(obs.ds.assign(los=("obs", noisy)))
        solved = SlipInversion(mesh, shaken, ramp="offset",
                               covariance=model).solve(smoothing=lam, max_iter=400,
                                                       **bounds)
        draws.append(solved.strike_slip)

    empirical = np.asarray(draws).std(axis=0, ddof=1)
    ratio = float(np.median(empirical / np.maximum(prop[:n], 1e-12)))
    assert 0.6 < ratio < 1.6, f"propagated sigma is off by {ratio:.2f}x"
    assert np.corrcoef(empirical, prop[:n])[0, 1] > 0.7


def test_a_covariance_is_required_for_a_covariance(signal_setup):
    """Without Cd the weights are `1/sqrt(n)`, so the units are arbitrary.

    Same rule as `chi_square`: a statistic only meaningful against a stated noise
    model must be absent when none was stated, not silently wrong.
    """
    _, _, mesh, _, parts = signal_setup
    plain = SlipInversion(mesh, Observations.concat(parts), ramp="offset")
    with pytest.raises(ValueError, match="needs a covariance model"):
        R.posterior_covariance(plain, 1.0)


def test_a_bound_pressed_solution_is_refused(signal_setup):
    """The linear covariance describes the UNBOUNDED problem.

    `resolution.py` warns about exactly this, and the warning was written when the
    real run had 86% of elements on a bound. Tight bounds must raise rather than
    return a plausible-looking sigma for a problem nobody solved.
    """
    _, _, mesh, _, parts = signal_setup
    obs = Observations.concat(parts, normalize="none")
    model = CovarianceModel(sill=SILL, range_m=RANGE, nugget=NUGGET)
    inversion = SlipInversion(mesh, obs, ramp="offset", covariance=model)
    pinched = inversion.solve(smoothing=3.0, max_iter=200,
                              strike=(-0.02, 0.0), dip=(-0.02, 0.0))
    assert R.bound_fraction(pinched) > R.MAX_BOUND_FRACTION
    with pytest.raises(ValueError, match="sit on a bound"):
        R.posterior_covariance(inversion, 3.0, model=pinched)


def test_the_bias_grows_with_lambda_against_a_FIXED_reference(signal_setup):
    """`(R - I) m` -- the other half of the error. Two traps, both pinned.

    More damping buys variance down and bias up, and that trade is why a single
    error bar is the wrong presentation. But the pull must be measured against a
    **fixed** reference model, or it moves the wrong way:

    1. the *absolute* pull is taken at the recovered model, which shrinks with the
       damping, so it shrinks too (0.317 -> 0.098 m measured);
    2. even the *relative* pull is non-monotone when the model moves with lambda
       (0.203 -> 0.431 -> 0.290), because `m` rotates toward the smooth directions
       `R` preserves best -- the recovered model is already the smoothed one.
    """
    _, _, mesh, truth, parts = signal_setup
    obs = Observations.concat(parts, normalize="none")
    model = CovarianceModel(sill=SILL, range_m=RANGE, nugget=NUGGET)
    inversion = SlipInversion(mesh, obs, ramp="offset", covariance=model)
    n = inversion.basis.n_basis
    bounds = {"strike": (-20.0, 20.0), "dip": (-20.0, 20.0)}

    reference = inversion.solve(smoothing=1.0, max_iter=400, **bounds)
    fixed, moving, sigma = [], [], []
    for lam in (1.0, 10.0, 100.0):
        matrix = R.resolution_matrix(inversion, lam)
        fixed.append(R.resolution_bias(matrix, reference)["relative"])
        moving.append(R.resolution_bias(
            matrix, inversion.solve(smoothing=lam, max_iter=400, **bounds))["relative"])
        # ⚠️ Mean, not median. At light damping the sigma distribution spans four
        # orders of magnitude and the median describes only the bulk -- measured,
        # it *rises* 0.059 -> 0.207 over lambda 0.01 -> 100 while the mean falls
        # 128 -> 0.40, because damping pulls the tail in faster than it lifts the
        # bulk. Reading the median would say more smoothing means less certainty.
        sigma.append(float(np.mean(R.posterior_covariance(inversion, lam)[:n])))

    assert fixed == sorted(fixed), f"against a fixed model the bias must grow: {fixed}"
    assert fixed[-1] > 0.5, "at heavy damping the estimate is mostly the prior"
    assert sigma == sorted(sigma, reverse=True), f"variance must fall: {sigma}"
    assert sigma[0] > 3.0 * sigma[-1], "damping should cut the variance materially"

    # The trap itself, pinned: let the reference move and monotonicity is lost.
    assert moving != sorted(moving), (
        "this fixture no longer demonstrates the moving-reference trap; if the "
        "measure changed, re-check the warning in `resolution_bias`")
