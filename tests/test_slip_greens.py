"""Pins the Green's-function assembly, and above all the sign of slip.

Nothing in the code says whether ``strike_slip = +1`` is left- or right-lateral;
it falls out of ``Vstrike = cross(eZ, Vnorm)`` composed with the mesh's winding
rule, and it has to be *measured*. Getting it backwards is the worst failure mode
in the package: a polarity bound would then enforce the wrong sense of motion, and
the inversion would return near-zero slip with a poor fit that looks exactly like
bad data.
"""

import numpy as np
import pytest

from nisar_tools.slip import FaultTrace
from nisar_tools.slip.greens import HalfSpaceTDE, project_los
from nisar_tools.slip.mesh import FaultMesh

NU = 0.25


def _mesh(lon, lat, **kwargs):
    trace = FaultTrace(lon, lat)
    frame = trace.local_frame()
    kwargs.setdefault("max_depth", 15e3)
    kwargs.setdefault("edge_length", 5e3)
    return trace, frame, FaultMesh.vertical(trace, frame, **kwargs)


def _north_striking(**kwargs):
    lat = np.linspace(10.0, 11.0, 9)
    return _mesh(np.full_like(lat, -67.0), lat, **kwargs)


def _east_striking(**kwargs):
    lon = np.linspace(-68.0, -67.0, 9)
    return _mesh(lon, np.full_like(lon, 10.5), **kwargs)


# -- the sign of slip --------------------------------------------------------

def test_strike_direction_opposes_the_trace_direction():
    """``cross(eZ, Vnorm)`` points backwards along the trace.

    With ``Vnorm`` wound to the trace's left-hand normal ``(-t_y, t_x)``,
    ``cross(eZ, Vnorm) = (-t_x, -t_y, 0)``. That identity is what makes the
    lateral sense of positive slip the same at every strike.

    It is exact against each *element's own* along-strike edge, and only
    approximate against the trace's vertex tangents, which average the two
    adjacent segments -- hence the comparison here is a dot product close to -1
    rather than componentwise equality.
    """
    for build in (_north_striking, _east_striking):
        trace, frame, mesh = build()
        tx, ty = trace.tangents(frame)
        sx, sy = trace.to_local(frame)
        s_trace = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(sx), np.diff(sy)))])
        s_elem = mesh.element_params[:, 0]

        v_strike = np.cross(np.array([0.0, 0.0, 1.0]), mesh.normals)
        np.testing.assert_allclose(np.linalg.norm(v_strike, axis=1), 1.0, atol=1e-12)
        np.testing.assert_allclose(v_strike[:, 2], 0.0, atol=1e-12)
        # Perpendicular to the normal, by construction of the cross product.
        np.testing.assert_allclose((v_strike * mesh.normals).sum(axis=1), 0.0, atol=1e-12)

        alignment = (v_strike[:, 0] * np.interp(s_elem, s_trace, tx)
                     + v_strike[:, 1] * np.interp(s_elem, s_trace, ty))
        np.testing.assert_allclose(alignment, -1.0, atol=1e-3)


@pytest.mark.parametrize("build", [_north_striking, _east_striking])
def test_positive_strike_slip_is_left_lateral(build):
    """Positive strike-slip moves the trace's left-hand block *backwards* along it.

    Checked at two strikes 90 degrees apart, because the mapping from the sign of
    ``strike_slip`` to a lateral sense is not obviously rotation-invariant --
    ``Vstrike`` is defined by a cross product with the vertical, not by the
    trace's direction.

    Viewed from the right-hand block, the left-hand block moving backwards along
    the trace is motion to the *left*: sinistral. So a right-lateral fault needs
    ``strike_slip <= 0``.
    """
    trace, frame, mesh = build()
    tx, ty = trace.tangents(frame)
    nx, ny = trace.normals(frame)
    x, y = trace.to_local(frame)
    i = len(x) // 2                       # mid-trace, away from the ends

    offset = 20e3
    px = np.array([x[i] + offset * nx[i], x[i] - offset * nx[i]])
    py = np.array([y[i] + offset * ny[i], y[i] - offset * ny[i]])

    slip = np.zeros(2 * mesh.n_elements)
    slip[: mesh.n_elements] = 1.0         # unit strike-slip everywhere
    disp = HalfSpaceTDE(NU).forward(mesh, slip, px, py)

    # Displacement resolved along the trace's own direction.
    along = disp[:, 0] * tx[i] + disp[:, 1] * ty[i]
    left, right = along[0], along[1]

    assert left < 0, "the left-hand block must move backwards along the trace"
    assert right > 0, "the right-hand block must move forwards along the trace"
    # Antisymmetric about the fault, as a pure strike-slip source must be.
    assert abs(left + right) < 0.05 * abs(left)
    # Vertical motion is negligible for pure strike-slip on a vertical fault.
    assert np.abs(disp[:, 2]).max() < 0.02 * abs(left)


def test_dip_slip_on_a_vertical_fault_moves_one_side_up_and_the_other_down():
    """``Vdip = cross(Vnorm, Vstrike)`` is vertical when ``Vnorm`` is horizontal.

    So dip-slip on a vertical fault is pure vertical offset across it: the
    **vertical** component is antisymmetric, while the fault-normal horizontal
    component is symmetric (both sides pull the same way as the two blocks shear
    past each other vertically). Along-strike motion is nil.
    """
    trace, frame, mesh = _east_striking()
    slip = np.zeros(2 * mesh.n_elements)
    slip[mesh.n_elements:] = 1.0
    disp = HalfSpaceTDE(NU).forward(mesh, slip, np.array([0.0, 0.0]), np.array([20e3, -20e3]))

    up = disp[:, 2]
    assert np.sign(up[0]) == -np.sign(up[1]), "vertical motion must reverse across the fault"
    assert abs(up[0] + up[1]) < 0.05 * abs(up[0])

    # This trace runs east, so along-strike (east) motion should be negligible.
    assert np.abs(disp[:, 0]).max() < 1e-6 * np.abs(disp[:, 1:]).max()
    # The fault-normal (north) component keeps one sign on both sides.
    assert np.sign(disp[0, 1]) == np.sign(disp[1, 1])


# -- assembly ----------------------------------------------------------------

def test_los_matrix_equals_projected_displacement():
    """The memory-lean path and the explicit one must agree exactly."""
    _, _, mesh = _east_striking(edge_length=8e3)
    rng = np.random.default_rng(0)
    x = rng.uniform(-60e3, 60e3, 40)
    y = rng.uniform(-60e3, 60e3, 40)
    le = rng.uniform(-0.6, -0.4, 40)
    ln = rng.uniform(-0.1, 0.1, 40)
    lu = np.sqrt(1.0 - le**2 - ln**2)

    engine = HalfSpaceTDE(NU)
    direct = engine.los_matrix(mesh, x, y, le, ln, lu)
    viaenu = project_los(engine.displacement(mesh, x, y), le, ln, lu)
    np.testing.assert_allclose(direct, viaenu, rtol=1e-12, atol=0)


def test_assembly_is_deterministic():
    """Repeat assembly is bit-identical: no accumulation order to drift.

    Worth pinning because the obvious optimisation here is a thread pool over
    elements, and that would introduce one. (It also measured *slower* -- see the
    module docstring of :mod:`nisar_tools.slip.greens`.)
    """
    _, _, mesh = _east_striking(edge_length=8e3)
    rng = np.random.default_rng(1)
    x, y = rng.uniform(-50e3, 50e3, 30), rng.uniform(-50e3, 50e3, 30)
    le, ln = np.full(30, -0.5), np.zeros(30)
    lu = np.full(30, np.sqrt(0.75))

    engine = HalfSpaceTDE(NU)
    np.testing.assert_array_equal(
        engine.los_matrix(mesh, x, y, le, ln, lu),
        engine.los_matrix(mesh, x, y, le, ln, lu),
    )


def test_forward_accepts_both_slip_layouts():
    _, _, mesh = _east_striking(edge_length=8e3)
    rng = np.random.default_rng(2)
    x, y = rng.uniform(-40e3, 40e3, 12), rng.uniform(-40e3, 40e3, 12)
    pairs = rng.normal(size=(mesh.n_elements, 2))
    flat = np.concatenate([pairs[:, 0], pairs[:, 1]])

    engine = HalfSpaceTDE(NU)
    np.testing.assert_allclose(
        engine.forward(mesh, pairs, x, y), engine.forward(mesh, flat, x, y), rtol=1e-12
    )
    with pytest.raises(ValueError, match="expected"):
        engine.forward(mesh, np.zeros(5), x, y)


def test_observations_on_the_trace_raise_a_useful_error():
    """A dislocation solution is singular on the fault, so this must not pass silently.

    Zeroing the NaN would leave a row of the inversion quietly meaningless; the
    message has to point at the exclusion buffer instead.
    """
    trace, frame, mesh = _east_striking()
    x, y = trace.to_local(frame)
    i = len(x) // 2
    with pytest.raises(ValueError, match="non-finite Green's function"):
        HalfSpaceTDE(NU).los_matrix(
            mesh,
            np.array([x[i]]), np.array([y[i]]),
            np.array([-0.5]), np.array([0.0]), np.array([np.sqrt(0.75)]),
            )


def test_greens_matrix_shape_and_block_order():
    _, _, mesh = _east_striking(edge_length=8e3)
    g = HalfSpaceTDE(NU).los_matrix(
        mesh, np.array([30e3]), np.array([30e3]),
        np.array([-0.5]), np.array([0.0]), np.array([np.sqrt(0.75)]),
    )
    assert g.shape == (1, 2 * mesh.n_elements)
    # Strike-slip block first, then dip-slip; both non-trivial.
    assert np.abs(g[0, : mesh.n_elements]).max() > 0
    assert np.abs(g[0, mesh.n_elements:]).max() > 0
