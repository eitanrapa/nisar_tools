"""Dual-frequency (split-spectrum) ionospheric correction.

The ionosphere is **dispersive**: its contribution to the interferometric phase
scales as ``1/f`` while everything else -- topography, deformation, the orbit --
scales as ``f``. One acquisition at two carriers therefore separates the two,
and NISAR ships both in every granule: ``frequencyA`` (1.239 GHz, 40 MHz
bandwidth, 5 m posting on a real product) and ``frequencyB`` (1.2935 GHz, 5 MHz,
40 m posting). Writing the two sub-band interferograms as

    phi_low  = (f_low  / f0) phi_nondisp + (f0 / f_low ) phi_disp
    phi_high = (f_high / f0) phi_nondisp + (f0 / f_high) phi_disp

and inverting the 2x2 system gives :func:`split_coefficients`; the dispersive
half is the ionospheric screen and the non-dispersive half is what the
inversion actually wants.

**The inputs must be UNWRAPPED, and that is not a design choice.** At NISAR's
carriers the dispersive combination is ``12.12 * phi_A - 11.61 * phi_B``: both
coefficients are large, so an unknown 2*pi in either input moves the screen by
~76 radians. There is no wrapped-domain shortcut, and the complex product of the
two bands does not supply one either -- the exact identity is

    phi_disp = 0.51 * phi_A + 11.61 * (phi_A - phi_B)

so the estimate needs each band's *absolute* phase, not only their difference.
What *is* free to choose is where the estimate is applied:

- :meth:`~nisar_tools.unwrap.UnwrappedStack.correct_ionosphere` subtracts it
  from the unwrapped phase, and
- :meth:`~nisar_tools.interferogram.InterferogramStack.correct_ionosphere`
  rotates the *wrapped* interferogram by it, before unwrapping -- which is the
  point of doing it there: a large ionospheric ramp is fringes the unwrapper has
  to get right, and it is cheaper to remove them than to unwrap them. The screen
  may be estimated on a coarsely multilooked pair (easier to unwrap, and the
  ionosphere is long-wavelength anyway) and applied to a finer one; it is
  interpolated onto whatever grid it is handed.

⚠️ **Smoothing is not optional.** With independent phase noise ``sigma`` in each
band the raw screen carries ``hypot(12.12, 11.61) = 16.8 * sigma`` -- so an
unsmoothed "correction" makes the data strictly worse. Hence ``scale``, which
defaults to a Gaussian sigma of :data:`DEFAULT_SMOOTH_METRES` expressed in the
grid's own pixels, and the resolved value is recorded in the provenance.

This is a *measurement* of the ionosphere from two carriers, unlike the
``estimate_phase_screen`` that was removed in 2026-07-23: that one fitted a
smooth surface to a single band and was therefore only a slower
:meth:`~nisar_tools.unwrap.UnwrappedStack.deramp`. Nothing here can be
reproduced by a deramp -- and conversely, this cannot flatten an orbital ramp,
which is non-dispersive and survives the correction untouched.
"""

import math
import warnings

import numpy as np
import xarray as xr

from . import _kernels
from ._base import match_pairs, plane_kernel
from .geometry import SPEED_OF_LIGHT

#: Default Gaussian sigma of the screen's low-pass, in metres. Resolved to
#: pixels against the grid it is estimated on, so the same default means the
#: same ground distance whatever the stack was multilooked by. Ionospheric
#: structure is kilometre-scale and the raw estimate amplifies phase noise ~17x,
#: so the smoothing is what makes the correction worth applying at all.
DEFAULT_SMOOTH_METRES = 2000.0

#: Floor for the resolved smoothing sigma, in pixels. Below a couple of pixels a
#: Gaussian is barely a filter, and the noise gain would come straight through.
MIN_SMOOTH_PIXELS = 2.0


def split_coefficients(f_a, f_b, f_reference=None):
    """Split-spectrum coefficients on ``(phase_a, phase_b)``.

    Returns ``((d_a, d_b), (n_a, n_b))`` with

        phase_dispersive     = d_a * phase_a + d_b * phase_b
        phase_nondispersive  = n_a * phase_a + n_b * phase_b

    both referred to ``f_reference``, which defaults to ``f_a``. The
    coefficients are in the order the frequencies were given -- which of the two
    is the lower carrier is worked out here, so no caller has to sort them.

    At the default reference the two rows sum to ``(1, 0)``: the dispersive and
    non-dispersive parts add back up to ``phase_a``, which is what makes
    subtracting the screen from band ``a`` the same thing as keeping band ``a``'s
    non-dispersive phase.

    Raises if the two carriers are equal: one frequency cannot separate two
    unknowns, and the 2x2 system is singular rather than merely ill-conditioned.
    """
    f_a = float(f_a)
    f_b = float(f_b)
    if f_a == f_b:
        raise ValueError(
            f"The two sub-bands have the same centre frequency ({f_a:g} Hz), "
            "so there is nothing to separate. Frequency A and B of one NISAR "
            "granule differ (1.239 vs 1.2935 GHz); check the two stacks were "
            "built with different frequency= arguments."
        )
    f0 = f_a if f_reference is None else float(f_reference)
    f_low, f_high = (f_a, f_b) if f_a < f_b else (f_b, f_a)

    denom = f_high * f_high - f_low * f_low
    dispersive = (f_low * f_high * f_high / (f0 * denom),
                  -f_low * f_low * f_high / (f0 * denom))
    nondispersive = (-f0 * f_low / denom, f0 * f_high / denom)
    if f_a > f_b:  # coefficients came out low-first; hand them back a-first
        dispersive = dispersive[::-1]
        nondispersive = nondispersive[::-1]
    return dispersive, nondispersive


def noise_gain(f_a, f_b, f_reference=None):
    """Factor by which the dispersive estimate amplifies independent phase noise.

    ``hypot`` of the two dispersive coefficients: with the same ``sigma`` in each
    band, the raw screen's noise is this times ``sigma``. **16.8 for NISAR's A/B
    pair**, which is the whole reason the estimate is low-passed before use.
    """
    (d_a, d_b), _ = split_coefficients(f_a, f_b, f_reference)
    return float(math.hypot(d_a, d_b))


def dispersive_phase(phase_a, phase_b, f_a, f_b, f_reference=None):
    """Ionospheric (dispersive) part of two sub-band phases, at ``f_reference``.

    Pointwise and dtype-agnostic: numpy arrays, dask arrays and ``DataArray``s
    all go straight through, so this stays lazy on a lazy stack.
    """
    (d_a, d_b), _ = split_coefficients(f_a, f_b, f_reference)
    return d_a * phase_a + d_b * phase_b


def nondispersive_phase(phase_a, phase_b, f_a, f_b, f_reference=None):
    """Non-dispersive (geometric + deformation) part, at ``f_reference``.

    Equal to ``phase_a - dispersive_phase(...)`` at the default reference, which
    is what makes subtracting the screen the same thing as keeping this.
    """
    _, (n_a, n_b) = split_coefficients(f_a, f_b, f_reference)
    return n_a * phase_a + n_b * phase_b


def _carrier(stack, override, role):
    """Centre frequency of a stack's sub-band, in Hz."""
    if override is not None:
        return float(override)
    attrs = stack.ds.attrs
    if attrs.get("center_frequency") is not None:
        return float(attrs["center_frequency"])
    if attrs.get("wavelength"):
        return SPEED_OF_LIGHT / float(attrs["wavelength"])
    raise ValueError(
        f"The {role} stack does not record its centre frequency, so the two "
        "carriers cannot be read off the data. Stacks built since the "
        "dual-frequency work carry it from the granule's own centerFrequency; "
        "an older store predates that. Pass frequency= / other_frequency= in Hz "
        "(NISAR: 1.239e9 for band A, 1.2935e9 for band B)."
    )


def _pixel_sizes(field):
    """``(dy, dx)`` grid spacing in metres, from the coordinates not the attrs.

    Deliberately not ``attrs["x_spacing"]``/``["y_spacing"]``: those are the
    *native* GSLC posting, which a multilooked stack no longer has. Kept as two
    numbers because a NISAR sub-band grid is genuinely anisotropic -- band B is
    posted at 40 m in x and 5 m in y on a real product.
    """
    x = np.asarray(field["x"].values, dtype=float)
    y = np.asarray(field["y"].values, dtype=float)
    dx = abs(x[1] - x[0]) if x.size > 1 else np.nan
    dy = abs(y[1] - y[0]) if y.size > 1 else np.nan
    return float(dy), float(dx)


def _pixel_size(field):
    """Mean of the two grid spacings, in metres -- the basis a pixel ``scale``
    is read against."""
    return float(np.nanmean(_pixel_sizes(field)))


def resolve_ground_scale(field, scale):
    """Resolve a smoothing ``scale`` to a Gaussian sigma in **metres**.

    ``None`` means :data:`DEFAULT_SMOOTH_METRES`, floored at
    :data:`MIN_SMOOTH_PIXELS` of ``field``'s own pixel so a coarse grid never
    gets a filter narrower than its own samples; a number is taken as pixels of
    ``field``, as everywhere else in this package, and converted; ``0`` disables
    the low-pass.

    Metres, not pixels, because the same screen has to be estimated on two grids
    with different postings -- see :func:`axis_sigmas`.
    """
    pixel = _pixel_size(field)
    if not np.isfinite(pixel) or pixel <= 0:
        pixel = 1.0
    if scale is None:
        return max(DEFAULT_SMOOTH_METRES, MIN_SMOOTH_PIXELS * pixel)
    return float(scale) * pixel


def axis_sigmas(field, ground):
    """``(sigma_y, sigma_x)`` in pixels for a ``ground``-metre Gaussian sigma.

    Per axis, because ``scipy``'s ``gaussian_filter`` takes one sigma per axis
    and an anisotropic grid needs two: a single pixel count on band B's 40 m x /
    5 m y lattice would low-pass the two axes at ground scales 8x apart, and the
    difference between that and band A's isotropic smoothing would come through
    the ~17x amplification as signal.
    """
    dy, dx = _pixel_sizes(field)
    sy = ground / dy if np.isfinite(dy) and dy > 0 else 0.0
    sx = ground / dx if np.isfinite(dx) and dx > 0 else 0.0
    return float(sy), float(sx)


def interp_to_grid(field, y, x):
    """``field`` sampled on the ``(y, x)`` lattice, or itself if already on it.

    NISAR's two sub-bands are **not** on a common grid -- band B is posted at
    40 m in x against band A's 5 m on a real product, and the origins differ too
    -- so this is on the normal path, not an escape hatch. Linear interpolation
    is the right resampling for a field that has just been low-passed to
    kilometres, and it propagates NaN so a gap does not silently fill.

    Interpolation gathers the spatial axes, so the array is rechunked to whole
    planes first; the caller keeps one pair in memory either way.
    """
    if field.sizes.get("y") != len(y) or field.sizes.get("x") != len(x):
        on_grid = False
    elif "y" not in field.coords or "x" not in field.coords:
        # A bare array of the right shape is taken to be on the grid already.
        on_grid = True
    else:
        on_grid = (np.allclose(field["y"].values, y)
                   and np.allclose(field["x"].values, x))
    if on_grid:
        # Reassign rather than trust: xarray aligns arithmetic on coordinate
        # *values*, so two lattices that differ in the last bit would inner-join
        # to nothing instead of lining up. Interpolation lands on ``y``/``x``
        # exactly, and this makes the pass-through do the same.
        return field.assign_coords(y=y, x=x)
    if _kernels._is_dask(field.data):
        field = field.chunk({field.dims[0]: 1, "y": -1, "x": -1})
    return field.interp(y=y, x=x, method="linear",
                        kwargs={"fill_value": np.nan})


def _lowpass(field, sigma):
    """NaN-aware Gaussian low-pass of a ``(pair, y, x)`` field.

    ``sigma`` is ``(sigma_y, sigma_x)`` in pixels, handed straight to
    ``scipy``'s ``gaussian_filter`` through :func:`_kernels.smooth_surface`.
    That truncates at 4 sigma, so a halo of the wider of the two reproduces the
    whole-plane smooth exactly -- the same rule ``remove_outliers`` uses.
    """
    if max(sigma) <= 0:
        return field.astype(np.float32)
    return plane_kernel(
        _kernels.smooth_surface, field.astype(np.float32),
        depth=int(math.ceil(4.0 * max(sigma))), scale=tuple(float(s) for s in sigma),
    )


def _gated(stack_ds, min_coherence, role):
    """A stack's unwrapped phase, NaN below ``min_coherence``."""
    phase = stack_ds["unw"]
    if min_coherence is None:
        return phase
    if "coherence" not in stack_ds:
        raise ValueError(
            f"min_coherence needs a coherence layer, and the {role} stack has "
            "none. SNAPHU stacks carry the interferogram's coherence through "
            "and GUNWs ship one, but a store written before 2026-07-24 does not."
        )
    return phase.where(stack_ds["coherence"] >= float(min_coherence))


def dual_frequency_screen(
    stack, other, *, frequency=None, other_frequency=None,
    reference_frequency=None, scale=None, min_coherence=None,
    remove_offset=True, time_tolerance=600.0,
):
    """Ionospheric phase screen from two unwrapped sub-band stacks.

    ``stack`` is the band being corrected and ``other`` is the second sub-band;
    the screen comes back on ``stack``'s grid, in ``stack``'s pair order, as the
    dispersive phase **at ``stack``'s own carrier** -- so subtracting it leaves a
    phase that still means metres at the wavelength ``to_los`` will use. Which of
    the two is the lower carrier is worked out from the frequencies, so the
    argument order carries no hidden convention.

    Both stacks must be unwrapped (see the module docstring for why). Pairs are
    matched by acquisition time within ``time_tolerance`` seconds, as
    :meth:`~nisar_tools.unwrap.UnwrappedStack.merge` does.

    ⚠️ **Each band is low-passed on its own grid, and only then combined.** The
    combination is linear, so on a shared grid that is exactly the same answer as
    smoothing the screen afterwards -- but the two sub-bands are *not* on a
    shared grid (band B is posted at 40 m in x against band A's 5 m on a real
    product), and resampling a raw phase before combining puts every
    interpolation error through the ~17x amplification. Measured on a synthetic
    pair with band B decimated 4x, interpolating first left a **1.27 rad** false
    screen straddling a sharp deformation step; smoothing first removes it,
    because what gets resampled is then already smooth at its own posting.

    ``scale`` is the Gaussian sigma of that low-pass, **in pixels of** ``stack``
    -- the other band gets whatever pixel count matches the same ground
    distance, per axis. See :func:`resolve_ground_scale` for the default and
    :func:`noise_gain` for why the low-pass is not optional. ``scale=0``
    disables it and returns the raw estimate, which is a diagnostic, not a
    correction.

    ``min_coherence`` drops pixels below it in each band before the low-pass, so
    a decorrelated patch is filled in from its neighbours rather than smeared
    over them.

    ``remove_offset`` subtracts each pair's median. Every unwrapped phase carries
    an arbitrary constant, and the screen inherits ~12x band A's and ~12x band
    B's, so its absolute level is meaningless -- while the constant it would
    otherwise inject into the corrected phase is real enough to alarm anyone
    reading the numbers. The *shape*, which is all a differential interferogram
    can measure, is untouched.

    Returns a lazy ``(pair, y, x)`` DataArray named ``phase_screen`` whose attrs
    record the provenance the two ``correct_ionosphere`` methods fold into their
    persist hashes.
    """
    f_self = _carrier(stack, frequency, "first")
    f_other = _carrier(other, other_frequency, "second")
    f_ref = f_self if reference_frequency is None else float(reference_frequency)
    (a_self, a_other), _ = split_coefficients(f_self, f_other, f_ref)

    if int(stack.epsg) != int(other.epsg):
        raise ValueError(
            f"The two sub-band stacks are in different CRSs (EPSG:{stack.epsg} "
            f"vs EPSG:{other.epsg}). Frequency A and B of one granule share a "
            "projection; reproject before combining them."
        )
    order = match_pairs(stack.ds, other.ds, time_tolerance,
                        action="combine the two sub-bands")

    y = np.asarray(stack.ds["y"].values)
    x = np.asarray(stack.ds["x"].values)
    other_ds = other.ds.isel(pair=order)

    # One sigma in metres, four pixel counts: the bands are posted differently
    # and each grid is anisotropic, so a shared pixel count would low-pass the
    # four axes at four ground scales and the differences would land in the
    # screen as signal.
    ground = resolve_ground_scale(stack.ds["unw"], scale)
    sigma_self = axis_sigmas(stack.ds["unw"], ground)
    sigma_other = axis_sigmas(other_ds["unw"], ground)
    _warn_if_undersmoothed(ground, stack.ds["unw"], other_ds["unw"])

    smooth_self = _lowpass(_gated(stack.ds, min_coherence, "first"), sigma_self)
    smooth_other = _lowpass(_gated(other_ds, min_coherence, "second"), sigma_other)
    smooth_other = interp_to_grid(
        smooth_other.drop_vars(["ref_time", "sec_time"], errors="ignore"), y, x
    ).assign_coords(pair=stack.ds["pair"].values)

    screen = a_self * smooth_self + a_other * smooth_other

    if remove_offset:
        screen = screen - _plane_median(screen)

    screen = screen.astype(np.float32).rename("phase_screen")
    # A fresh dict, not a mutation: plane_kernel passes its input's attrs object
    # straight through, so clearing in place could reach back into the caller's.
    screen.attrs = _provenance(
        f_self, f_other, f_ref, a_self, a_other, ground, sigma_self, sigma_other,
        min_coherence, remove_offset,
    )
    return screen


def _plane_median(field):
    """Per-plane median of a ``(pair, y, x)`` field, NaNs skipped.

    Not ``field.median(dim=("y", "x"))``: dask has no chunked exact median at
    all, and ``da.nanmedian`` additionally refuses *any* two-axis reduction, so
    the spatial axes are gathered into one chunk and reduced inside the task --
    one pair at a time, the same thing ``merge`` does for its 2*pi offset. The
    explicit finite-mask loop is there because ``np.nanmedian`` warns on an
    all-NaN plane, which a fully masked pair legitimately is.
    """
    def _median(block):
        flat = np.asarray(block).reshape(block.shape[:-2] + (-1,))
        out = np.full(flat.shape[:-1], np.nan, dtype=np.float64)
        for index in np.ndindex(out.shape):
            values = flat[index]
            values = values[np.isfinite(values)]
            if values.size:
                out[index] = np.median(values)
        return out.astype(block.dtype)

    if _kernels._is_dask(field.data):
        field = field.chunk({field.dims[0]: 1, "y": -1, "x": -1})
    return xr.apply_ufunc(
        _median, field,
        input_core_dims=[["y", "x"]],
        dask="parallelized",
        output_dtypes=[field.dtype],
    )


def _warn_if_undersmoothed(ground, self_field, other_field):
    """Warn when the low-pass is too narrow for the coarser band's posting.

    The screen is combined after each band is smoothed and the coarser one is
    interpolated up, so what the resampling has to reproduce is a Gaussian of
    sigma ``ground``. Linear interpolation of that costs about
    ``(spacing**2 / 8) * |f''|``, and the ~17x amplification is applied to it --
    which is why this is worth a warning rather than a footnote.

    Measured on a synthetic pair with band B posted 8x coarser than band A and a
    sharp deformation step: the screen's rms error against the truth is
    **0.284 rad** at ``ground`` equal to the coarse posting and **0.159 rad** at
    twice it, after which it flattens (0.16-0.19 rad out to 12x) -- the floor
    there is the coarse band's own sampling of the step, not the filter. On a
    shared grid the error is exactly zero at every scale. Hence the factor of
    two: below it the resampling dominates, above it nothing more is bought.
    """
    if ground <= 0:
        return  # the low-pass was disabled on purpose
    spacing = max(max(_pixel_sizes(self_field)), max(_pixel_sizes(other_field)))
    if np.isfinite(spacing) and ground < 2.0 * spacing:
        warnings.warn(
            f"The ionosphere low-pass (sigma {ground:.0f} m) is narrower than "
            f"twice the coarser sub-band's posting ({spacing:.0f} m), so "
            "interpolating that band onto the other's grid dominates the "
            "screen -- and its error is amplified ~17x along with everything "
            f"else. Raise scale= (it is in pixels of the corrected stack, "
            f"{_pixel_size(self_field):.0f} m each) or multilook the two bands "
            "to closer postings.",
            RuntimeWarning,
            stacklevel=3,
        )


def _provenance(f_self, f_other, f_ref, a_self, a_other, ground, sigma_self,
                sigma_other, min_coherence, remove_offset):
    """JSON-able record of how a screen was made, for the persist hash."""
    return {
        "method": "dual_frequency",
        "center_frequency": float(f_self),
        "other_center_frequency": float(f_other),
        "reference_frequency": float(f_ref),
        "coefficients": [float(a_self), float(a_other)],
        "noise_gain": float(math.hypot(a_self, a_other)),
        "scale_metres": float(ground),
        "scale": [float(s) for s in sigma_self],
        "other_scale": [float(s) for s in sigma_other],
        "min_coherence": None if min_coherence is None else float(min_coherence),
        "remove_offset": bool(remove_offset),
    }


def as_screen(screen, y, x, npair):
    """Coerce a screen argument onto a ``(pair, y, x)`` grid.

    Accepts the DataArray :func:`dual_frequency_screen` returns, an
    :class:`~nisar_tools.unwrap.UnwrappedStack` carrying one as its
    ``phase_screen`` layer, or a plain array. Returns ``(field, provenance)``.
    """
    ds = getattr(screen, "ds", None)
    if ds is not None:
        if "phase_screen" not in ds:
            raise ValueError(
                "That stack carries no 'phase_screen' to apply. Estimate one "
                "with ionosphere.dual_frequency_screen(low, high), or pass the "
                "screen itself."
            )
        provenance = dict(ds.attrs.get("ionosphere") or {})
        screen = ds["phase_screen"]
    else:
        provenance = dict(getattr(screen, "attrs", {}) or {})

    if not isinstance(screen, xr.DataArray):
        screen = xr.DataArray(np.asarray(screen), dims=("pair", "y", "x"))
    if screen.sizes.get("pair") != npair:
        raise ValueError(
            f"The screen has {screen.sizes.get('pair')} pairs and the stack has "
            f"{npair}. Estimate the screen from stacks with the same pairs."
        )
    field = interp_to_grid(
        screen.drop_vars(["ref_time", "sec_time"], errors="ignore"), y, x
    )
    return field, provenance
