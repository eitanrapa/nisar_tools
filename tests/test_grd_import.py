"""Pins :func:`nisar_tools.geo.read_grd` and :meth:`LOSStack.from_grd`.

The route in for LOS products this package did not make -- GMTSAR, ISCE, an ALOS
``los_cm.grd`` from a colleague. Three things such a file cannot record have to
be declared (units, sign, look-vector direction), and each of them is silently
plausible when wrong, so most of what is pinned here is that the declarations are
applied and that a contradicted one raises.

The acceptance test is :func:`test_recovers_slip_from_alos_style_grids`: write
the three departures a NISAR product does *not* have -- lon/lat axes,
centimetres, a right-looking pass -- and require the planted slip back.
"""

import numpy as np
import pytest
import xarray as xr
from slip_synthetic import tapered_slip

from nisar_tools import FaultMesh, FaultTrace, LOSStack, Observations, SlipInversion
from nisar_tools.geo import read_grd, utm_epsg
from nisar_tools.slip.diagnostics import _geometry_consistent
from nisar_tools.slip.greens import HalfSpaceTDE

_LON = np.array([-68.681, -68.300, -67.900, -67.500, -67.100, -66.700, -66.523])
_LAT = np.array([10.410, 10.490, 10.540, 10.565, 10.595, 10.620, 10.630])

# ALOS is **right**-looking, where NISAR is left, so the sign of ``los_east``
# reverses between the two missions on the same pass direction.
_GEOMETRIES = {"asc": (34.0, -10.0), "desc": (38.0, 190.0)}


def _right_looking(incidence_deg, heading_deg):
    """Target->sensor ENU unit vector for a right-looking pass."""
    inc = np.deg2rad(incidence_deg)
    az = np.deg2rad(heading_deg + 90.0)
    return (-np.sin(inc) * np.sin(az), -np.sin(inc) * np.cos(az), np.cos(inc))


def _write_grd(arr, lon, lat, path):
    """A lon/lat GMT grid, as ``grdproject``/``proj_ra2ll`` leaves one."""
    xr.DataArray(
        np.asarray(arr, np.float32), coords={"lat": lat, "lon": lon},
        dims=["lat", "lon"], name="z",
    ).to_netcdf(path)
    return str(path)


@pytest.fixture(scope="module")
def alos_scenes(tmp_path_factory):
    """Two right-looking scenes over a planted right-lateral patch, as ``.grd``.

    Written in centimetres and positive *away* from the sensor -- the opposite of
    this package's convention on both counts -- so that loading them correctly
    exercises ``units`` and ``sign`` rather than passing by coincidence.
    """
    out = tmp_path_factory.mktemp("alos")
    trace = FaultTrace(_LON, _LAT, name="san_sebastian")
    frame = trace.local_frame()
    mesh = FaultMesh.vertical(trace, frame, max_depth=20e3, edge_length=6e3)
    truth = tapered_slip(mesh, peak=-2.0)

    lon = np.arange(-69.4, -65.8, 0.04)
    lat = np.arange(11.4, 9.79, -0.04)
    grid_lon, grid_lat = np.meshgrid(lon, lat)
    fx, fy = frame.to_local(grid_lon.ravel(), grid_lat.ravel())
    enu = HalfSpaceTDE(0.25).forward(mesh, truth, fx, fy)

    rng = np.random.default_rng(1)
    scenes = {}
    for name, geom in _GEOMETRIES.items():
        look = _right_looking(*geom)
        toward = (enu @ np.array(look)).reshape(grid_lon.shape)
        toward = toward + rng.normal(0.0, 0.004, toward.shape)
        scenes[name] = {
            "look": look,
            "los": _write_grd(-100.0 * toward, lon, lat, out / f"{name}_los_cm.grd"),
            **{
                f"look_{c}": _write_grd(np.full(grid_lon.shape, v), lon, lat,
                                        out / f"{name}_look_{c}.grd")
                for c, v in zip("enu", look)
            },
        }
    return trace, frame, mesh, truth, scenes


def _load(scene, **kwargs):
    kwargs.setdefault("units", "cm")
    kwargs.setdefault("sign", -1)
    return LOSStack.from_grd(
        scene["los"], scene["look_e"], scene["look_n"], scene["look_u"], **kwargs
    )


# -- the reader ---------------------------------------------------------------

def test_read_grd_round_trips_write_grd(tmp_path):
    """``read_grd`` is the inverse of ``write_grd``, coordinates included."""
    lon = np.linspace(-70.0, -68.0, 24)
    lat = np.linspace(11.0, 10.0, 17)
    values = np.outer(np.sin(lat), np.cos(lon)).astype(np.float32)
    path = _write_grd(values, lon, lat, tmp_path / "field.grd")

    da = read_grd(path)
    assert da.dims == ("y", "x")
    np.testing.assert_allclose(da["x"].values, lon)
    np.testing.assert_allclose(da["y"].values, lat)
    np.testing.assert_allclose(da.values, values)
    # lon/lat axes with no CRS in the file are taken as geographic, which is what
    # tells `from_grd` the grid has to be reprojected before it can be sampled.
    assert da.rio.crs.is_geographic


def test_utm_epsg_picks_the_containing_zone():
    assert utm_epsg(-67.5, 10.5) == 32619        # Venezuela, north
    assert utm_epsg(-67.5, -10.5) == 32719       # same zone, south
    assert utm_epsg(95.9, 21.9) == 32646         # Sagaing, Myanmar
    assert utm_epsg(96.0, 21.9) == 32647         # a zone starts at its west edge
    assert utm_epsg(-180.0, 0.0) == 32601
    assert utm_epsg(179.9, 0.0) == 32660


# -- the declarations ---------------------------------------------------------

def test_units_and_sign_are_applied_on_load(alos_scenes):
    """``los_cm.grd`` positive-away becomes metres positive-toward the sensor."""
    *_, scenes = alos_scenes
    raw = read_grd(scenes["desc"]["los"]).values
    los = _load(scenes["desc"]).ds["los"].values

    assert np.nanmax(np.abs(raw)) > 10.0            # tens of centimetres
    assert np.nanmax(np.abs(los)) < 1.0             # fractions of a metre
    # This is the *absolute* anchor -- that `sign=-1` means "the file is positive
    # away from the sensor, so flip it" -- and it can only be loose, because the
    # reprojection between the two resamples the field onto a different lattice.
    # The exactness of both declarations is pinned separately, on two loads of
    # the one file, where no resampling stands in the way.
    assert np.nanmean(los) * 100.0 / np.nanmean(raw) == pytest.approx(-1.0, rel=0.2)
    assert np.nanstd(los) * 100.0 / np.nanstd(raw) == pytest.approx(1.0, rel=0.2)


def test_units_and_sign_are_exactly_the_declared_factors(alos_scenes):
    """Nothing in a ``.grd`` records units or sign, so a wrong declaration is
    silent -- the load succeeds and only the recovered slip gives it away.

    Pinned as an exact relation between two loads of the same file, which is what
    both arguments promise and is unaffected by the reprojection they share.
    """
    *_, scenes = alos_scenes
    metres = _load(scenes["desc"]).ds["los"].values
    np.testing.assert_allclose(
        _load(scenes["desc"], units="m").ds["los"].values, metres * 100.0, rtol=1e-4
    )
    np.testing.assert_allclose(
        _load(scenes["desc"], sign=1).ds["los"].values, -metres, rtol=1e-6
    )


def test_look_convention_is_checked_against_the_data(alos_scenes):
    """A target->sensor vector points up; declaring the opposite must raise."""
    *_, scenes = alos_scenes
    with pytest.raises(ValueError, match="target->sensor unit vector"):
        _load(scenes["desc"], look_convention="sensor_to_target")


def test_sensor_to_target_grids_are_negated(alos_scenes, tmp_path):
    """The other convention loads correctly once declared."""
    *_, scenes = alos_scenes
    flipped = {}
    for c in "enu":
        da = read_grd(scenes["asc"][f"look_{c}"])
        flipped[c] = _write_grd(-da.values, da["x"].values, da["y"].values,
                                tmp_path / f"neg_{c}.grd")
    stack = LOSStack.from_grd(
        scenes["asc"]["los"], flipped["e"], flipped["n"], flipped["u"],
        units="cm", sign=-1, look_convention="sensor_to_target",
    )
    reference = _load(scenes["asc"])
    for v in ("los_east", "los_north", "los_up"):
        np.testing.assert_allclose(stack.ds[v].values, reference.ds[v].values,
                                   rtol=1e-5, equal_nan=True)


def test_non_unit_look_vector_raises(alos_scenes, tmp_path):
    """Three grids that are not a direction are a mistake, not a rescaling."""
    *_, scenes = alos_scenes
    da = read_grd(scenes["desc"]["look_u"])
    doubled = _write_grd(da.values * 2.0, da["x"].values, da["y"].values,
                         tmp_path / "big_u.grd")
    with pytest.raises(ValueError, match="not a unit vector"):
        LOSStack.from_grd(scenes["desc"]["los"], scenes["desc"]["look_e"],
                          scenes["desc"]["look_n"], doubled, units="cm", sign=-1)


@pytest.mark.parametrize("bad", [{"units": "km"}, {"sign": 0},
                                 {"look_convention": "toward"}])
def test_bad_declarations_raise(alos_scenes, bad):
    *_, scenes = alos_scenes
    with pytest.raises(ValueError):
        _load(scenes["desc"], **bad)


# -- the grid -----------------------------------------------------------------

def test_geographic_grids_are_reprojected_to_metres(alos_scenes):
    """The quadtree measures ``width_min`` in metres, so degrees cannot be fed to
    it: ``width_min=1000`` on a lon/lat lattice is a million columns."""
    *_, scenes = alos_scenes
    stack = _load(scenes["asc"])
    assert stack.epsg == 32619                       # UTM 19N, from the centre
    spacing = abs(np.diff(stack.ds["x"].values)[0])
    assert 1e3 < spacing < 1e4                       # metres, not degrees


def test_explicit_epsg_and_resolution_are_honoured(alos_scenes):
    *_, scenes = alos_scenes
    stack = _load(scenes["asc"], epsg=32620, resolution=6000.0)
    assert stack.epsg == 32620
    np.testing.assert_allclose(abs(np.diff(stack.ds["x"].values)[0]), 6000.0)


def test_look_grids_on_a_coarser_lattice_are_resampled(alos_scenes, tmp_path):
    """``SAT_look`` is often run on a decimated grid, so the components need not
    arrive on the displacement's lattice."""
    *_, scenes = alos_scenes
    coarse = {}
    for c in "enu":
        da = read_grd(scenes["desc"][f"look_{c}"])[::3, ::3]
        coarse[c] = _write_grd(da.values, da["x"].values, da["y"].values,
                               tmp_path / f"coarse_{c}.grd")
    stack = LOSStack.from_grd(scenes["desc"]["los"], coarse["e"], coarse["n"],
                              coarse["u"], units="cm", sign=-1)
    assert stack.ds["los_east"].shape == stack.ds["los"].shape[1:]
    e, n, u = _GEOMETRIES["desc"] and _right_looking(*_GEOMETRIES["desc"])
    np.testing.assert_allclose(np.nanmean(stack.ds["los_east"].values), e, atol=1e-3)
    np.testing.assert_allclose(np.nanmean(stack.ds["los_up"].values), u, atol=1e-3)


def test_look_vector_is_unit_length_and_gives_the_incidence_angle(alos_scenes):
    """``los_up == cos(incidence)`` is the invariant the rest of the package
    leans on, and only holds for a renormalised vector."""
    *_, scenes = alos_scenes
    ds = _load(scenes["desc"]).ds
    e, n, u = (ds[f"los_{c}"].values for c in ("east", "north", "up"))
    norm = np.sqrt(e ** 2 + n ** 2 + u ** 2)
    finite = np.isfinite(norm)
    np.testing.assert_allclose(norm[finite], 1.0, atol=1e-5)
    np.testing.assert_allclose(
        np.cos(np.radians(ds["incidence_angle"].values[finite])), u[finite], atol=1e-5
    )
    np.testing.assert_allclose(np.nanmean(ds["incidence_angle"].values), 38.0, atol=0.5)


def test_right_looking_passes_reverse_los_east(alos_scenes):
    """ALOS is right-looking, so ascending ``los_east`` is **negative** -- the
    opposite of NISAR. ``scene_report``'s geometry check has to agree, which is
    what makes it useful on a non-NISAR scene at all."""
    *_, scenes = alos_scenes
    asc = _load(scenes["asc"], direction="ascending", look_direction="right")
    desc = _load(scenes["desc"], direction="descending", look_direction="right")
    assert np.nanmean(asc.ds["los_east"].values) < 0
    assert np.nanmean(desc.ds["los_east"].values) > 0
    assert _geometry_consistent(asc.ds) == (True, -1)
    assert _geometry_consistent(desc.ds) == (True, +1)


# -- end to end ---------------------------------------------------------------

def test_recovers_slip_from_alos_style_grids(alos_scenes):
    """The acceptance test: ``.grd`` files in, the planted slip out.

    Two right-looking scenes on a lon/lat lattice in centimetres, positive away
    from the sensor. Getting any one of the three declarations wrong changes the
    answer's scale or its sense, so a high correlation *and* the right peak is a
    joint check on all of them.
    """
    trace, frame, mesh, truth, scenes = alos_scenes
    obs = Observations.concat(
        [
            Observations.from_los(
                _load(scene, direction=d, look_direction="right"),
                name=name, frame=frame, trace=trace,
                rms_min=0.006, width_min=6000.0, width_max=30000.0,
                exclude_within=5000.0,
            )
            for (name, scene), d in zip(scenes.items(), ("ascending", "descending"))
        ],
        normalize="sqrt_count",
    )
    model = SlipInversion(mesh, obs).solve(smoothing=0.3, polarity=(-1, 0, 0))

    assert model.converged
    assert model.variance_reduction > 95.0
    strike = truth[: mesh.n_elements]
    assert np.corrcoef(strike, model.strike_slip)[0, 1] > 0.95
    # Right-lateral in, right-lateral out -- and to within 20% of the true peak,
    # which is what catches a units or sign error that correlation would not.
    assert model.strike_slip.min() == pytest.approx(strike.min(), rel=0.2)
