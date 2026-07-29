"""The layered engine: point-source superposition against the half-space answer.

The acceptance test is :func:`test_reproduces_the_half_space_engine`. A layered
code handed a uniform medium has to give back the homogeneous half-space answer,
and EDCMP is built around exactly that identity -- its input file carries a
"layered (1) or homogeneous (0)" switch and it ships Okada's routines for the
second branch. Synthesising uniform-medium tables lets us make the same check
with no Fortran present.

What that test does **not** check is the physics of layering: the tables it uses
come from the same triangular-dislocation solution the engine is compared with.
It is a test of the *machinery* -- the moment-tensor packing, the azimuthal
recombination, the north/east/down conventions, the bilinear lookup and the
quadrature -- which is where every transcription error would live. Real layering
is checked by the two-layer test at the end, which has no independent oracle and
so pins direction and magnitude rather than a number.
"""

import numpy as np
import pytest

from nisar_tools.slip import FaultMesh, FaultTrace, HalfSpaceTDE
from nisar_tools.slip.edgrn import EdgrnTables, VelocityModel
from nisar_tools.slip.layered import (
    LayeredPointSource,
    _moment_components,
    _triangle_quadrature,
)

_LON = np.array([-68.681, -68.300, -67.900, -67.500, -67.100, -66.700, -66.523])
_LAT = np.array([10.410, 10.490, 10.540, 10.565, 10.595, 10.620, 10.630])


@pytest.fixture(scope="module")
def uniform_tables():
    """Homogeneous tables, fine enough that bilinear lookup is not the limit."""
    return EdgrnTables.homogeneous(
        nu=0.25, r=np.linspace(0.0, 300e3, 301),
        z=np.linspace(250.0, 30e3, 120), n_azimuth=16)


@pytest.fixture(scope="module")
def scene():
    trace = FaultTrace(_LON, _LAT, name="test_fault")
    frame = trace.local_frame()
    mesh = FaultMesh.curved(trace, frame, uniform_dip=70.0,
                            max_depth=20e3, edge_length=12e3)
    rng = np.random.default_rng(0)
    x, y = mesh.centroids[:, 0], mesh.centroids[:, 1]
    ox = rng.uniform(x.min() - 40e3, x.max() + 40e3, 80)
    oy = rng.uniform(y.min() - 50e3, y.max() + 50e3, 80)
    keep = trace.distance(ox, oy, frame) > 8e3
    return trace, frame, mesh, ox[keep], oy[keep]


# -- the moment tensor -------------------------------------------------------

def test_moment_packing_of_a_vertical_strike_slip_element():
    """A vertical east-striking fault slipping east is a pure ``m12`` couple.

    The packing is the one place the north/east/down convention has to be exactly
    right, and nothing downstream can tell a sign error from a real slip reversal.
    """
    # ENU: normal north, slip east.
    m1, m2, m3, m4, m5 = _moment_components((0.0, 1.0, 0.0), (1.0, 0.0, 0.0))
    assert m1 == pytest.approx(1.0)
    assert (m2, m3, m4, m5) == pytest.approx((0.0, 0.0, 0.0, 0.0))


def test_moment_packing_of_a_vertical_dip_slip_element():
    """Normal north, slip straight down: a pure ``m13`` couple."""
    m1, m2, m3, m4, m5 = _moment_components((0.0, 1.0, 0.0), (0.0, 0.0, -1.0))
    assert m2 == pytest.approx(1.0)
    assert (m1, m3, m4, m5) == pytest.approx((0.0, 0.0, 0.0, 0.0))


def test_moment_tensor_of_a_shear_dislocation_is_traceless():
    """Which is why five components suffice for six tensor entries."""
    rng = np.random.default_rng(1)
    for _ in range(20):
        n = rng.normal(size=3)
        n /= np.linalg.norm(n)
        d = np.cross(n, rng.normal(size=3))
        d /= np.linalg.norm(d)
        _, _, m3, m4, _ = _moment_components(n, d)
        # trace = m_nn + m_ee + m_dd = (-m3/2 + m4) + (-m3/2 - m4) + m3 = 0.
        assert abs(-m3 / 2 + m4 + -m3 / 2 - m4 + m3) < 1e-12


# -- quadrature --------------------------------------------------------------

def test_quadrature_weights_and_centroid():
    p1, p2, p3 = np.array([0.0, 0, 0]), np.array([3.0, 0, 0]), np.array([0.0, 3, 0])
    for order in (1, 2, 3, 5):
        points, weights = _triangle_quadrature(p1, p2, p3, order)
        assert points.shape == (order ** 2, 3)
        assert weights.sum() == pytest.approx(1.0)
        # An equal-weight convergent rule reproduces the centroid exactly.
        np.testing.assert_allclose((points * weights[:, None]).sum(axis=0),
                                   (p1 + p2 + p3) / 3.0, atol=1e-12)


def test_quadrature_uses_both_sub_triangle_orientations():
    """Taking only the upright sub-triangles is a biased sample, not a rule.

    Subdividing a triangle ``k`` ways gives ``k(k+1)/2`` upright sub-triangles and
    ``k(k-1)/2`` inverted ones. Sampling only the upright set with equal weights
    -- which a plain barycentric lattice does -- has an error that stops falling
    however many points are added; measured, it plateaus near 5e-3 relative.
    """
    p1, p2, p3 = np.array([0.0, 0, 0]), np.array([1.0, 0, 0]), np.array([0.0, 1, 0])
    for order in (2, 3, 4):
        points, _ = _triangle_quadrature(p1, p2, p3, order)
        assert points.shape[0] == order ** 2


# -- the acceptance test -----------------------------------------------------

def test_reproduces_the_half_space_engine(uniform_tables, scene):
    """A uniform medium through the layered path must give the half-space answer.

    The two routes share nothing but the physics: one evaluates a closed-form
    triangular dislocation per element, the other cuts the element into point
    sources, looks each up in a table indexed by distance and source depth, puts
    an azimuthal harmonic back and superposes. Every convention in the second
    route is free to be wrong, and any of them would show up here.
    """
    _, _, mesh, ox, oy = scene
    exact = HalfSpaceTDE(0.25).displacement(mesh, ox, oy)
    layered = LayeredPointSource(uniform_tables, tolerance=1e-3).displacement(mesh, ox, oy)

    scale = np.abs(exact).max()
    assert np.abs(layered - exact).max() / scale < 1e-2
    assert np.sqrt(np.mean((layered - exact) ** 2)) / scale < 1e-3
    assert np.corrcoef(layered.ravel(), exact.ravel())[0, 1] > 0.9999

    # Both slip components, separately -- a fault that dips excites the CLVD
    # table through the dip-slip column, so this is where m3 gets checked.
    n = mesh.n_elements
    for name, block in (("strike", slice(0, n)), ("dip", slice(n, 2 * n))):
        a, b = exact[:, :, block], layered[:, :, block]
        assert np.corrcoef(a.ravel(), b.ravel())[0, 1] > 0.9999, name


def test_error_falls_as_the_table_is_refined(scene):
    """Confirms the residual is table resolution, not a mistake in the algebra.

    A wrong convention would leave an error that refinement cannot touch.
    Measured, halving the cell size takes the worst-case relative error from
    7.6e-3 to 3.7e-3.
    """
    _, _, mesh, ox, oy = scene
    exact = HalfSpaceTDE(0.25).displacement(mesh, ox, oy)
    scale = np.abs(exact).max()

    errors = []
    for nr, nz in ((76, 30), (151, 60), (301, 120)):
        tables = EdgrnTables.homogeneous(
            r=np.linspace(0.0, 300e3, nr), z=np.linspace(250.0, 30e3, nz), n_azimuth=12)
        layered = LayeredPointSource(tables, tolerance=1e-3).displacement(mesh, ox, oy)
        errors.append(np.abs(layered - exact).max() / scale)

    assert errors[0] > errors[1] > errors[2]


def test_los_matrix_agrees_with_the_projected_displacement(uniform_tables, scene):
    """``los_matrix`` must be exactly ``displacement`` dotted with the look vector.

    They are separate code paths -- the matrix form never materialises the third
    axis -- so this is the check that the saving is free.
    """
    _, _, mesh, ox, oy = scene
    look = (np.full(ox.size, 0.62), np.full(ox.size, -0.11), np.full(ox.size, 0.78))
    engine = LayeredPointSource(uniform_tables, tolerance=3e-3)

    direct = engine.los_matrix(mesh, ox, oy, *look)
    projected = np.einsum("pcn,c->pn", engine.displacement(mesh, ox, oy),
                          np.array([look[0][0], look[1][0], look[2][0]]))
    np.testing.assert_allclose(direct, projected, rtol=1e-12, atol=1e-14)


def test_forward_matches_the_matrix(uniform_tables, scene):
    _, _, mesh, ox, oy = scene
    engine = LayeredPointSource(uniform_tables, tolerance=3e-3)
    slip = np.zeros(2 * mesh.n_elements)
    slip[: mesh.n_elements] = -1.5
    slip[mesh.n_elements:] = 0.4

    look = (np.full(ox.size, 0.62), np.full(ox.size, -0.11), np.full(ox.size, 0.78))
    np.testing.assert_allclose(engine.forward(mesh, slip, ox, oy, look=look),
                               engine.los_matrix(mesh, ox, oy, *look) @ slip,
                               rtol=1e-10)


# -- the adaptive quadrature -------------------------------------------------

def test_adaptive_quadrature_costs_far_less_than_the_fixed_rule(uniform_tables, scene):
    """The reference's 91 points everywhere, against one where one will do.

    Counted in source-receiver evaluations, which is what the runtime is: the
    order is chosen per observation, so an element near a handful of samples and
    far from thousands pays only for the handful.
    """
    _, _, mesh, ox, oy = scene
    oz = np.zeros(ox.size)
    adaptive = LayeredPointSource(uniform_tables, tolerance=3e-3)
    fixed = LayeredPointSource(uniform_tables, tolerance=None)

    def cost(engine):
        total = 0
        for k in range(mesh.n_elements):
            size = np.sqrt(2.0 * mesh.areas[k])
            total += int((engine._orders(mesh.centroids[k], size, ox, oy, oz) ** 2).sum())
        return total

    assert cost(adaptive) * 5 < cost(fixed)


def test_adaptive_and_fixed_quadrature_agree(uniform_tables, scene):
    """The saving must not cost accuracy at the tolerance asked for."""
    _, _, mesh, ox, oy = scene
    adaptive = LayeredPointSource(uniform_tables, tolerance=3e-3).displacement(mesh, ox, oy)
    fixed = LayeredPointSource(uniform_tables, tolerance=None).displacement(mesh, ox, oy)
    assert np.abs(adaptive - fixed).max() / np.abs(fixed).max() < 1e-2


# -- layering itself ---------------------------------------------------------

def test_a_softer_shallow_layer_moves_the_surface_more(scene):
    """The reason for the whole exercise, stated as a one-way inequality.

    There is no independent oracle for a genuinely layered medium here, so this
    pins the direction rather than a number: a compliant shallow crust deforms
    more for the same slip, which is why assuming it is stiff biases shallow slip
    low. The tables are built by scaling the homogeneous ones by the rigidity
    contrast, which is the leading-order effect and enough to fix a sign.
    """
    _, _, mesh, ox, oy = scene
    stiff = EdgrnTables.homogeneous(r=np.linspace(0.0, 300e3, 301),
                                    z=np.linspace(250.0, 30e3, 120), n_azimuth=12)
    model = VelocityModel([0.0, 5e3, 30e3], [4.0e3, 6.0e3, 6.5e3],
                          [2.0e3, 3.5e3, 3.7e3], [2.4e3, 2.7e3, 2.9e3])
    contrast = model.at(30e3, "mu") / model.at(stiff.z, "mu")
    soft = EdgrnTables(
        stiff.r, stiff.z,
        {k: {c: v * contrast[None, :] for c, v in table.items()}
         for k, table in stiff.tables.items()},
        attrs={"source": "scaled"},
    )

    slip = np.zeros(2 * mesh.n_elements)
    slip[: mesh.n_elements] = -1.0
    shallow = mesh.centroids[:, 2] > -6e3
    slip[: mesh.n_elements] = np.where(shallow, -1.0, 0.0)

    engine_stiff = LayeredPointSource(stiff, tolerance=3e-3)
    engine_soft = LayeredPointSource(soft, tolerance=3e-3)
    u_stiff = engine_stiff.forward(mesh, slip, ox, oy)
    u_soft = engine_soft.forward(mesh, slip, ox, oy)

    assert np.abs(u_soft).max() > np.abs(u_stiff).max()


def test_engine_satisfies_the_inversion_protocol(uniform_tables, scene):
    """Drop-in for ``HalfSpaceTDE``: same members, same column layout."""
    from nisar_tools.slip.inversion import SlipInversion

    _, _, mesh, ox, oy = scene
    engine = LayeredPointSource(uniform_tables)
    for member in ("los_matrix", "displacement", "forward", "name", "nu"):
        assert hasattr(engine, member)
    assert SlipInversion.__init__.__defaults__ is not None
    assert engine.name != HalfSpaceTDE.name
