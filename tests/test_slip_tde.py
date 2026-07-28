"""Pins the triangular-dislocation port against an independent oracle.

:mod:`nisar_tools.slip._tde` is a port of ~540 lines of angular-dislocation
algebra, and essentially every way it can be wrong is silent: a dropped sign or a
swapped basis vector still returns smooth, plausible-looking displacement. So the
gate is agreement with :mod:`nisar_tools.slip._okada`, which computes the same
physics by a completely different route -- a closed form integrated over a
rectangle rather than a superposition of angular dislocations.

The vertical case gets its own attention throughout: a strike-slip fault mesh is
built entirely from vertical elements with exactly horizontal edges, which drives
``beta`` to exactly ``pi/2`` in the free-surface correction and puts every
``1/cos(beta)`` term on a removable singularity.
"""

import numpy as np
import pytest

from nisar_tools.slip._okada import okada_disp_surface, rect_corners
from nisar_tools.slip._tde import tde_disp_hs

NU = 0.25
LENGTH, WIDTH, DEPTH = 10e3, 5e3, 8e3


def _surface_grid(n=9, half=20e3, offset=5e3):
    """Surface observation points straddling the fault, never exactly on it."""
    gx = np.linspace(-half + offset, half + offset, n)
    gy = np.linspace(-half, half, n)
    x, y = np.meshgrid(gx, gy)
    return x.ravel(), y.ravel(), np.zeros(x.size)


def _tde_rectangle(dip_deg, slip, x, y, z, length=LENGTH, width=WIDTH, depth=DEPTH):
    """The same rectangle as two triangles sharing a winding."""
    c = rect_corners(length, width, depth, dip_deg)
    total = np.zeros((3, x.size))
    for tri in ((c[0], c[1], c[2]), (c[0], c[2], c[3])):
        total += np.array(tde_disp_hs(x, y, z, tri[0], tri[1], tri[2], *slip, NU))
    return total


# -- the gate: a rectangle as two triangles must reproduce Okada --------------

@pytest.mark.parametrize("dip", [90.0, 85.0, 80.0, 70.0, 45.0, 30.0, 110.0, 135.0])
@pytest.mark.parametrize(
    "name,slip",
    [("strike", (1.0, 0.0, 0.0)), ("dip", (0.0, 1.0, 0.0)), ("tensile", (0.0, 0.0, 1.0))],
)
def test_tde_rectangle_matches_okada(dip, name, slip):
    """Two TDEs tiling a rectangle equal Okada's closed form, component for component.

    The mapping is the identity -- Okada's ``U1``/``U2``/``U3`` are
    Nikkhoo & Walter's strike/dip/tensile with the same sign -- once the
    rectangle's down-dip offset is built the way :func:`rect_corners` does it.

    Dips *just* off vertical are excluded here and covered by
    :func:`test_near_vertical_precision_band`, which explains why.
    """
    x, y, z = _surface_grid()
    okada = np.array(okada_disp_surface(x, y, LENGTH, WIDTH, DEPTH, dip, *slip, NU))
    tde = _tde_rectangle(dip, slip, x, y, z)

    scale = np.abs(okada).max()
    assert scale > 0
    assert np.all(np.isfinite(tde))
    # Worst measured over this dip set is 1.8e-12 relative (strike-slip at 85 deg).
    np.testing.assert_allclose(tde, okada, atol=1e-11 * scale, rtol=0)


def test_near_vertical_precision_band():
    """Both solutions lose precision as the dip approaches -- but does not reach -- 90.

    Not a porting error: the two implementations degrade *together*, and each has
    its own removable singularity there. Okada's ``I1..I5`` divide by
    ``cos(dip)``; the triangular solution's free-surface correction divides by
    ``cos(beta)`` for an edge that is nearly, but not exactly, horizontal. At
    exactly 90 degrees both take a clean branch and agree to ~1e-14; a hair away,
    the cancellation is only approximate and digits are lost in proportion to
    ``1 / cos(dip)``.

    Measured relative disagreement on this geometry, worst of the three slip
    components (strike-slip is the worst, dip-slip the best): 2e-14 at exactly
    90, 1.8e-9 at 89.0, 1.7e-10 at 88.0, 1.8e-12 at 85.0 -- and, going the other
    way, 1.4e-5 at 89.99 and 1.2e-2 at 89.999.

    The invariant worth pinning is the *trend* -- error shrinking as the dip moves
    away from vertical -- not a tolerance at any one dip. The practical
    consequence lives in :class:`~nisar_tools.slip.mesh.FaultMesh`, which snaps a
    near-vertical dip to exactly 90 rather than sitting in this band.
    """
    x, y, z = _surface_grid()
    dips = (89.99, 89.9, 89.5, 89.0, 88.0, 85.0)

    for slip in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)):
        errors = []
        for dip in dips:
            okada = np.array(okada_disp_surface(x, y, LENGTH, WIDTH, DEPTH, dip, *slip, NU))
            tde = _tde_rectangle(dip, slip, x, y, z)
            assert np.all(np.isfinite(tde))
            errors.append(np.abs(tde - okada).max() / np.abs(okada).max())

        # Monotone improvement away from vertical, negligible by 85 degrees.
        assert all(a > b for a, b in zip(errors, errors[1:])), (slip, errors)
        assert errors[-1] < 1e-11, (slip, errors)

        # Exactly vertical takes the clean branch in both codes.
        okada = np.array(okada_disp_surface(x, y, LENGTH, WIDTH, DEPTH, 90.0, *slip, NU))
        tde = _tde_rectangle(90.0, slip, x, y, z)
        assert np.abs(tde - okada).max() / np.abs(okada).max() < 1e-12


def test_vertical_fault_is_not_a_special_case():
    """A vertical element is well conditioned even though Okada needs a branch.

    ``cross(eZ, Vnorm)`` degenerates for a *horizontal* element, not a vertical
    one -- so the mesh of a strike-slip fault sits on the safe side of the only
    degeneracy in the element basis. Meanwhile the every-edge-horizontal geometry
    drives ``beta`` to exactly ``pi/2``; this asserts nothing blows up there.
    """
    x, y, z = _surface_grid()
    for slip in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)):
        tde = _tde_rectangle(90.0, slip, x, y, z)
        assert np.all(np.isfinite(tde))
        assert np.abs(tde).max() > 0


def test_displacement_is_continuous_through_vertical():
    """No discontinuity at exactly 90 degrees.

    A broken ``beta`` branch in the free-surface correction shows up as a jump
    between 89.9 and 90.1 degrees while each side looks individually sane.
    """
    x, y, z = _surface_grid()
    below = _tde_rectangle(89.5, (1.0, 0.0, 0.0), x, y, z)
    at = _tde_rectangle(90.0, (1.0, 0.0, 0.0), x, y, z)
    above = _tde_rectangle(90.5, (1.0, 0.0, 0.0), x, y, z)

    scale = np.abs(at).max()
    # The midpoint of the two neighbours brackets the vertical case closely; a
    # branch error would put `at` far outside that bracket.
    np.testing.assert_allclose(at, 0.5 * (below + above), atol=0.02 * scale, rtol=0)


def test_linearity_in_slip():
    """Displacement is linear in the Burgers vector.

    Green's-function assembly relies on this: each column is the response to unit
    slip, and a model is their weighted sum.
    """
    x, y, z = _surface_grid()
    ss = _tde_rectangle(80.0, (1.0, 0.0, 0.0), x, y, z)
    ds = _tde_rectangle(80.0, (0.0, 1.0, 0.0), x, y, z)
    both = _tde_rectangle(80.0, (2.0, -3.0, 0.0), x, y, z)
    np.testing.assert_allclose(both, 2.0 * ss - 3.0 * ds, rtol=1e-9, atol=0)


def test_zero_slip_gives_exactly_zero():
    x, y, z = _surface_grid()
    tde = _tde_rectangle(70.0, (0.0, 0.0, 0.0), x, y, z)
    assert np.all(tde == 0.0)


def test_displacement_decays_in_the_far_field():
    """Far from a finite source the field must die off, and fast."""
    c = rect_corners(LENGTH, WIDTH, DEPTH, 75.0)
    near = np.array([[30e3], [0.0], [0.0]])
    far = np.array([[3000e3], [0.0], [0.0]])
    amps = []
    for pt in (near, far):
        total = np.zeros(3)
        for tri in ((c[0], c[1], c[2]), (c[0], c[2], c[3])):
            r = tde_disp_hs(pt[0], pt[1], pt[2], tri[0], tri[1], tri[2], 1.0, 0.0, 0.0, NU)
            total += np.array(r).ravel()
        amps.append(np.abs(total).max())
    assert amps[1] < amps[0] * 1e-3


def test_element_in_the_free_surface_radiates_nothing():
    """All three vertices at z = 0 is the one configuration the reference zeroes."""
    x, y, z = _surface_grid()
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([1e4, 0.0, 0.0])
    p3 = np.array([0.0, 1e4, 0.0])
    ue, un, uv = tde_disp_hs(x, y, z, p1, p2, p3, 1.0, 2.0, 3.0, NU)
    assert np.all(ue == 0) and np.all(un == 0) and np.all(uv == 0)


def test_rejects_observations_above_the_free_surface():
    p1, p2, p3 = rect_corners(LENGTH, WIDTH, DEPTH, 90.0)[:3]
    with pytest.raises(ValueError, match="must be <= 0"):
        tde_disp_hs([0.0], [0.0], [10.0], p1, p2, p3, 1.0, 0.0, 0.0, NU)


def test_inputs_are_not_mutated():
    """The reference flips the vertices' sign in place to build the image source.

    That is harmless under MATLAB's copy-on-write and a corruption bug here --
    the caller's mesh would come back mirrored through the free surface.
    """
    x, y, z = _surface_grid(n=3)
    p1, p2, p3 = rect_corners(LENGTH, WIDTH, DEPTH, 80.0)[:3]
    before = [p.copy() for p in (p1, p2, p3)]
    tde_disp_hs(x, y, z, p1, p2, p3, 1.0, 0.0, 0.0, NU)
    for got, want in zip((p1, p2, p3), before):
        np.testing.assert_array_equal(got, want)


# -- optional third opinion --------------------------------------------------

def test_matches_cutde():
    """Cross-check against ``cutde``, the established Python implementation.

    A third independent opinion on the same algorithm, and the only one that is
    itself a port of the same reference -- so it catches transcription slips that
    Okada, being a different derivation, would only reveal as a small
    discrepancy. Agreement is to ~5e-14 with the *identity* slip mapping, which
    also confirms that ``cutde`` orders its slip vector
    ``(strike, dip, tensile)`` the same way.

    Dev-only: ``cutde`` is in the ``dev`` extra and is never imported at runtime.
    """
    hs = pytest.importorskip("cutde.halfspace")

    rng = np.random.default_rng(0)
    x, y, z = _surface_grid(n=5)
    pts = np.column_stack([x, y, z])
    for _ in range(5):
        # Jitter the vertices horizontally only, so every triangle keeps a
        # sensible non-degenerate dip and stays below the free surface.
        tri = np.array([[0.0, 0.0, -5e3], [8e3, 1e3, -5e3], [4e3, -2e3, -12e3]])
        tri = tri + rng.normal(0.0, 1e3, size=(3, 3)) * np.array([1.0, 1.0, 0.0])
        slips = rng.normal(size=3)

        # disp_matrix is the Green's-function form: (nobs, 3, ntri, 3).
        matrix = hs.disp_matrix(pts, tri[None, :, :], NU)
        theirs = (matrix.reshape((pts.shape[0] * 3, -1)) @ slips).reshape(-1, 3).T
        mine = np.array(tde_disp_hs(x, y, z, tri[0], tri[1], tri[2], *slips, NU))

        scale = np.abs(theirs).max()
        np.testing.assert_allclose(mine, theirs, atol=1e-12 * scale, rtol=0)
