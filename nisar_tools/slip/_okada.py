"""Okada (1985) rectangular dislocation: the oracle for the TDE port.

Okada, Y. (1985), "Surface deformation due to shear and tensile faults in a
half-space", *Bulletin of the Seismological Society of America* **75**,
1135-1154 -- surface displacements, equations 25-30 with Chinnery's notation.

This exists to check :mod:`nisar_tools.slip._tde`, not to be used by the
inversion: a rectangle tiles only a plane, which is the whole reason the
inversion uses triangles. It is a genuinely *independent* derivation of the same
physics -- a closed form integrated over a rectangle, versus a superposition of
angular dislocations -- so a rectangle cut into two triangles agreeing with it to
round-off is strong evidence that the port is right.

It is also a complementary special case: a vertical fault needs a separate
``cos(dip) == 0`` branch here, while the triangular solution handles it without
one.

Geometry follows the paper. In the fault's own frame the rectangle spans
``0 <= xi <= length`` along strike (the ``+x`` axis) and ``0 <= w <= width``
down dip, with the **lower** edge at ``depth`` and the plane dipping so that

    P(xi, w) = (xi, -w cos(dip), -(depth - w sin(dip)))

Slip components are the paper's: ``U1`` left-lateral strike-slip, ``U2``
dip-slip (thrust positive), ``U3`` tensile opening.
"""

import numpy as np


def rect_corners(length, width, depth, dip_deg):
    """The rectangle's four corners in the fault frame, counter-clockwise.

    Returned in the order lower-left, lower-right, upper-right, upper-left,
    so ``[c[0], c[1], c[2]]`` and ``[c[0], c[2], c[3]]`` tile it with a shared
    winding -- which is what :mod:`nisar_tools.slip._tde` needs to see the same
    normal on both halves.
    """
    dip = np.deg2rad(dip_deg)
    cos_d, sin_d = np.cos(dip), np.sin(dip)

    def point(xi, w):
        return np.array([xi, w * cos_d, -(depth - w * sin_d)])

    return [point(0.0, 0.0), point(length, 0.0),
            point(length, width), point(0.0, width)]


def okada_disp_surface(x, y, length, width, depth, dip_deg, u1, u2, u3, nu=0.25):
    """Surface displacement of a rectangular dislocation.

    ``x``, ``y`` are surface observation coordinates in the fault frame (``z``
    is implicitly 0). Returns ``(ue, un, uv)`` as 1-D arrays.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    dip = np.deg2rad(dip_deg)
    sin_d, cos_d = np.sin(dip), np.cos(dip)
    # A dip of exactly 90 degrees leaves cos_d at ~6e-17 rather than 0, which is
    # the difference between the finite and the singular branch below.
    if abs(cos_d) < 1e-12:
        cos_d = 0.0
        sin_d = np.sign(sin_d) if sin_d != 0 else 1.0

    p = y * cos_d + depth * sin_d
    q = y * sin_d - depth * cos_d

    # Chinnery's operator: f(xi, eta) summed over the four corners.
    ue = np.zeros(x.size)
    un = np.zeros(x.size)
    uv = np.zeros(x.size)
    for xi, eta, sign in ((x, p, 1.0), (x, p - width, -1.0),
                          (x - length, p, -1.0), (x - length, p - width, 1.0)):
        e, n, v = _terms(xi, eta, q, sin_d, cos_d, u1, u2, u3, nu)
        ue += sign * e
        un += sign * n
        uv += sign * v
    return ue, un, uv


def _terms(xi, eta, q, sin_d, cos_d, u1, u2, u3, nu):
    """One corner's contribution to the Chinnery sum."""
    r = np.sqrt(xi * xi + eta * eta + q * q)
    y_t = eta * cos_d + q * sin_d
    d_t = eta * sin_d - q * cos_d
    r_eta = r + eta
    r_xi = r + xi
    r_dt = r + d_t

    # theta is arctan(xi*eta / (q*R)), zero where the fault plane is touched.
    with np.errstate(divide="ignore", invalid="ignore"):
        theta = np.arctan(xi * eta / (q * r))
    theta = np.where(np.abs(q) < 1e-12, 0.0, theta)

    i1, i2, i3, i4, i5 = _i_terms(xi, eta, q, r, y_t, d_t, r_eta, r_dt, sin_d, cos_d, nu)

    ue = un = uv = 0.0
    if u1:
        c = -u1 / (2.0 * np.pi)
        ue = ue + c * (xi * q / (r * r_eta) + theta + i1 * sin_d)
        un = un + c * (y_t * q / (r * r_eta) + q * cos_d / r_eta + i2 * sin_d)
        uv = uv + c * (d_t * q / (r * r_eta) + q * sin_d / r_eta + i4 * sin_d)
    if u2:
        c = -u2 / (2.0 * np.pi)
        ue = ue + c * (q / r - i3 * sin_d * cos_d)
        un = un + c * (y_t * q / (r * r_xi) + cos_d * theta - i1 * sin_d * cos_d)
        uv = uv + c * (d_t * q / (r * r_xi) + sin_d * theta - i5 * sin_d * cos_d)
    if u3:
        c = u3 / (2.0 * np.pi)
        ue = ue + c * (q * q / (r * r_eta) - i3 * sin_d * sin_d)
        un = un + c * (-d_t * q / (r * r_xi)
                       - sin_d * (xi * q / (r * r_eta) - theta) - i1 * sin_d * sin_d)
        uv = uv + c * (y_t * q / (r * r_xi)
                       + cos_d * (xi * q / (r * r_eta) - theta) - i5 * sin_d * sin_d)
    return ue, un, uv


def _i_terms(xi, eta, q, r, y_t, d_t, r_eta, r_dt, sin_d, cos_d, nu):
    """Okada's I1..I5. ``mu / (lambda + mu)`` reduces to ``1 - 2 nu``."""
    f = 1.0 - 2.0 * nu

    if cos_d == 0.0:
        # Vertical fault: the general forms all divide by cos(dip).
        i5 = -f * xi * sin_d / r_dt
        i4 = -f * q / r_dt
        i3 = f / 2.0 * (eta / r_dt + y_t * q / (r_dt * r_dt) - np.log(r_eta))
        i1 = -f / 2.0 * xi * q / (r_dt * r_dt)
    else:
        x = np.sqrt(xi * xi + q * q)
        with np.errstate(divide="ignore", invalid="ignore"):
            i5 = f * 2.0 / cos_d * np.arctan(
                (eta * (x + q * cos_d) + x * (r + x) * sin_d) / (xi * (r + x) * cos_d))
        i5 = np.where(np.abs(xi) < 1e-12, 0.0, i5)
        i4 = f / cos_d * (np.log(r_dt) - sin_d * np.log(r_eta))
        i3 = f * (y_t / (cos_d * r_dt) - np.log(r_eta)) + sin_d / cos_d * i4
        i1 = f * (-xi / (cos_d * r_dt)) - sin_d / cos_d * i5
    i2 = f * (-np.log(r_eta)) - i3
    return i1, i2, i3, i4, i5
