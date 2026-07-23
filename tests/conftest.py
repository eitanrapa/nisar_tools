"""Shared test fixtures, including a synthetic GSLC HDF5 factory.

The synthetic granules mimic the real NISAR L2 GSLC layout closely enough that
the same code paths (epsg attribute, byte-string orbit direction, signed
y-spacing, descending y axis) are exercised without needing multi-GB files.
"""

import os

# Must be set before any OpenMP-linked library (numpy/scipy/pygmt) initialises;
# this conda env ships duplicate libomp copies that otherwise abort the process.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import h5py
import numpy as np
import pytest

# The equivalence-test oracle lives beside the tests as ``legacy_reference``;
# pytest's default import mode puts this directory on sys.path, so no manual
# path manipulation is needed.

GRID = "science/LSAR/GSLC/grids/frequencyA"
IDENT = "science/LSAR/identification"
RADAR_GRID = "science/LSAR/GSLC/metadata/radarGrid"
CENTER_FREQ_A = 1_239_000_000.0  # Hz, NISAR L-band frequency A -> lambda ~0.242 m
EARTH_RADIUS = 6_371_000.0       # m, spherical, for the synthetic look angle
PLATFORM_ALT = 747_000.0         # m, NISAR's nominal orbit altitude


def _write_synthetic_geometry(f, grid, ident, x_coords, y_coords, dx, dy, epsg):
    """Write a minimal ``metadata/radarGrid`` geometry cube + ``centerFrequency``.

    Incidence angle rises linearly across x (near->far range) from 30 to 45 deg,
    independent of y and height; the LOS unit vector is
    ``(east=-sin(inc), north=0, up=cos(inc))``. Being analytically known, tests
    can check the trilinear interpolation exactly.

    ``elevationAngle`` (the off-nadir look angle, which the reader exposes as
    ``look_angle``) follows the spherical-Earth relation
    ``sin(look) = Re/(Re+h) sin(incidence)`` at a NISAR-like altitude, so it is
    always the smaller of the two, as in a real product.
    """
    grid.create_dataset("centerFrequency", data=np.float64(CENTER_FREQ_A))
    ident.create_dataset("lookDirection", data=np.bytes_("Left"))

    heights = np.array([-500.0, 0.0, 500.0, 1000.0])
    cx = np.linspace(x_coords.min() - dx, x_coords.max() + dx, 12)
    cy = np.linspace(y_coords.min() - dy, y_coords.max() + dy, 10)
    frac = (cx - cx.min()) / (cx.max() - cx.min())
    inc_deg = (30.0 + 15.0 * frac).astype(np.float32)  # varies along x only
    inc = np.ascontiguousarray(
        np.broadcast_to(inc_deg, (len(heights), len(cy), len(cx)))
    )
    losx = np.ascontiguousarray((-np.sin(np.deg2rad(inc))).astype(np.float32))
    losy = np.zeros_like(losx)
    # Off-nadir look angle for a ~747 km orbit over a spherical Earth.
    look = np.degrees(
        np.arcsin(EARTH_RADIUS / (EARTH_RADIUS + PLATFORM_ALT)
                  * np.sin(np.deg2rad(inc)))
    ).astype(np.float32)

    rg = f.create_group(RADAR_GRID)
    rg.create_dataset("incidenceAngle", data=inc)
    rg.create_dataset("elevationAngle", data=np.ascontiguousarray(look))
    rg.create_dataset("losUnitVectorX", data=losx)
    rg.create_dataset("losUnitVectorY", data=losy)
    rg.create_dataset("heightAboveEllipsoid", data=heights)
    rg.create_dataset("xCoordinates", data=cx)
    rg.create_dataset("yCoordinates", data=cy)
    projd = rg.create_dataset("projection", data=np.int32(epsg))
    projd.attrs["epsg_code"] = np.int32(epsg)


def write_synthetic_gslc(
    path,
    *,
    ny=256,
    nx=256,
    epsg=32611,
    direction="Descending",
    x0=400000.0,
    y0=4_000_000.0,
    dx=10.0,
    dy=10.0,
    datetime_str="2025-11-28T02:32:50.000000000",
    polarization="HH",
    data=None,
    seed=0,
    write_geometry=False,
    compressed=True,
):
    """Write a minimal but structurally faithful GSLC HDF5 file.

    ``x`` is ascending; ``y`` is descending for a ``Descending`` pass and
    ascending otherwise (matching how NISAR stores geocoded grids). The
    ``yCoordinateSpacing`` is stored signed, as in real products.

    ``compressed`` (default, matching real products) writes the image gzip+
    shuffle filtered with a NaN fill value. Pass ``False`` for a plain
    uncompressed dataset, which exercises the h5py fallback read path.
    """
    path = str(path)
    x_coords = x0 + dx * np.arange(nx, dtype=np.float64)
    if direction == "Descending":
        y_coords = y0 - dy * np.arange(ny, dtype=np.float64)
        y_spacing = -dy
    else:
        y_coords = y0 + dy * np.arange(ny, dtype=np.float64)
        y_spacing = dy

    if data is None:
        rng = np.random.default_rng(seed)
        amp = rng.uniform(0.5, 1.5, size=(ny, nx))
        phase = rng.uniform(-np.pi, np.pi, size=(ny, nx))
        data = (amp * np.exp(1j * phase)).astype(np.complex64)
    else:
        data = np.asarray(data, dtype=np.complex64)

    # Real GSLCs are gzip+shuffle with a NaN fill, which is the layout
    # DirectChunkReader decodes itself; default to matching that so the fast
    # read path is what the suite exercises.
    filters = (
        {"compression": "gzip", "compression_opts": 1, "shuffle": True,
         "fillvalue": np.complex64(complex(np.nan, np.nan))}
        if compressed else {}
    )

    with h5py.File(path, "w") as f:
        grid = f.create_group(GRID)
        grid.create_dataset(polarization, data=data,
                            chunks=(min(64, ny), min(64, nx)), **filters)
        grid.create_dataset("xCoordinates", data=x_coords)
        grid.create_dataset("yCoordinates", data=y_coords)
        grid.create_dataset("xCoordinateSpacing", data=np.float64(dx))
        grid.create_dataset("yCoordinateSpacing", data=np.float64(y_spacing))
        proj = grid.create_dataset("projection", data=np.int32(epsg))
        proj.attrs["epsg_code"] = np.int32(epsg)

        ident = f.create_group(IDENT)
        ident.create_dataset("orbitPassDirection", data=np.bytes_(direction))
        ident.create_dataset("zeroDopplerStartTime", data=np.bytes_(datetime_str))

        if write_geometry:
            _write_synthetic_geometry(f, grid, ident, x_coords, y_coords, dx, dy, epsg)

    return path


@pytest.fixture
def gslc_factory(tmp_path):
    """Return a factory that writes synthetic GSLC files into ``tmp_path``."""
    counter = {"n": 0}

    def _make(**kwargs):
        counter["n"] += 1
        name = kwargs.pop("name", f"gslc_{counter['n']}.h5")
        path = tmp_path / name
        return write_synthetic_gslc(path, **kwargs)

    return _make


GUNW_GRID = "science/LSAR/GUNW/grids/frequencyA"
GUNW_UNW = GUNW_GRID + "/unwrappedInterferogram"        # grid coords live here
GUNW_RADAR_GRID = "science/LSAR/GUNW/metadata/radarGrid"


def write_synthetic_gunw(
    path,
    *,
    ny=64,
    nx=64,
    epsg=32611,
    direction="Descending",
    x0=400000.0,
    y0=4_000_000.0,
    dx=80.0,
    dy=80.0,
    ref_time="2025-11-28T02:32:16.000000000",
    sec_time="2025-12-10T02:32:16.000000000",
    polarization="HH",
    nan_border=4,
    spikes=0,
    iono_amp=0.5,
    mask_invalid_cols=0,
    seed=0,
):
    """Write a minimal but faithful NISAR L2 GUNW HDF5 file.

    Mirrors the real product's layout: the grid coordinates and projection sit
    on the ``unwrappedInterferogram`` group (not the frequency group, which
    holds only ``centerFrequency``), and a ``metadata/radarGrid`` cube provides
    the geometry so ``to_los`` is self-contained. ``unwrappedPhase`` is a smooth
    ramp in x with a ``nan_border`` of out-of-swath NaNs; ``connectedComponents``
    is 1 in the valid interior and the product's 65535 fill in the border, so
    the reader's fill -> 0 normalisation is exercised.

    A smooth ``ionospherePhaseScreen`` (a gentle plane in y, amplitude
    ``iono_amp``) is written so ``phase_screen`` / ``remove_phase_screen`` can be
    tested; ``spikes`` injects that many large isolated outliers into the
    interior for ``remove_outliers`` tests. The 3-digit ``mask`` layer is valid
    land (11) everywhere except the out-of-swath border (255); ``mask_invalid_cols``
    flags that many interior columns as out-of-subswath (10, secondary digit 0)
    to exercise ``mask_edges(use_builtin_mask=True)``.

    The geometry cube matches ``_write_synthetic_geometry``: incidence rises
    linearly 30 -> 45 deg across x, and ``elevationAngle`` (the look angle)
    follows the spherical-Earth relation, always smaller than the incidence.
    """
    path = str(path)
    x_coords = x0 + dx * np.arange(nx, dtype=np.float64)
    if direction == "Descending":
        y_coords = y0 - dy * np.arange(ny, dtype=np.float64)
    else:
        y_coords = y0 + dy * np.arange(ny, dtype=np.float64)

    rng = np.random.default_rng(seed)
    unw = np.tile(np.linspace(-3.0, 3.0, nx), (ny, 1)).astype(np.float32)
    unw += rng.normal(0.0, 0.01, size=(ny, nx)).astype(np.float32)
    if spikes:
        lo = (nan_border or 0) + 2
        iy = rng.integers(lo, ny - lo, size=int(spikes))
        ix = rng.integers(lo, nx - lo, size=int(spikes))
        unw[iy, ix] += 50.0  # large isolated outliers
    coh = np.full((ny, nx), 0.7, np.float32)
    cc = np.ones((ny, nx), dtype=np.uint16)
    # a smooth ionosphere-like phase screen (a gentle plane in y)
    iono = (iono_amp * np.linspace(-1.0, 1.0, ny)[:, None]
            * np.ones((1, nx))).astype(np.float32)
    # 3-digit water+subswath mask: 11 = land, both subswaths valid.
    swmask = np.full((ny, nx), 11, np.uint8)
    if mask_invalid_cols:
        c0 = nx // 2
        swmask[:, c0:c0 + int(mask_invalid_cols)] = 10  # secondary subswath 0 = invalid
    if nan_border:
        b = nan_border
        unw[:, :b] = unw[:, -b:] = np.nan
        unw[:b, :] = unw[-b:, :] = np.nan
        invalid = ~np.isfinite(unw)
        coh[invalid] = np.nan
        cc[invalid] = 65535  # product fill; the reader maps it to label 0
        iono[invalid] = np.nan
        swmask[invalid] = 255  # out-of-swath fill

    heights = np.array([-500.0, 0.0, 500.0, 1000.0])
    cx = np.linspace(x_coords.min() - dx, x_coords.max() + dx, 12)
    cy = np.linspace(y_coords.min() - dy, y_coords.max() + dy, 10)
    frac = (cx - cx.min()) / (cx.max() - cx.min())
    inc_deg = (30.0 + 15.0 * frac).astype(np.float32)  # varies along x only
    inc = np.ascontiguousarray(
        np.broadcast_to(inc_deg, (len(heights), len(cy), len(cx)))
    )
    losx = np.ascontiguousarray((-np.sin(np.deg2rad(inc))).astype(np.float32))
    losy = np.zeros_like(losx)
    look = np.degrees(
        np.arcsin(EARTH_RADIUS / (EARTH_RADIUS + PLATFORM_ALT)
                  * np.sin(np.deg2rad(inc)))
    ).astype(np.float32)

    with h5py.File(path, "w") as f:
        grid = f.create_group(GUNW_GRID)
        grid.create_dataset("centerFrequency", data=np.float64(CENTER_FREQ_A))

        ug = f.create_group(GUNW_UNW)
        ug.create_dataset("xCoordinates", data=x_coords)
        ug.create_dataset("yCoordinates", data=y_coords)
        proj = ug.create_dataset("projection", data=np.int32(epsg))
        proj.attrs["epsg_code"] = np.int32(epsg)
        ug.create_dataset("mask", data=swmask)  # sits above the polarisation group
        pol = ug.create_group(polarization)
        pol.create_dataset("unwrappedPhase", data=unw)
        pol.create_dataset("coherenceMagnitude", data=coh)
        pol.create_dataset("connectedComponents", data=cc)
        pol.create_dataset("ionospherePhaseScreen", data=iono)

        ident = f.create_group(IDENT)
        ident.create_dataset("orbitPassDirection", data=np.bytes_(direction))
        ident.create_dataset("lookDirection", data=np.bytes_("Left"))
        ident.create_dataset("referenceZeroDopplerStartTime",
                             data=np.bytes_(ref_time))
        ident.create_dataset("secondaryZeroDopplerStartTime",
                             data=np.bytes_(sec_time))

        rg = f.create_group(GUNW_RADAR_GRID)
        rg.create_dataset("incidenceAngle", data=inc)
        rg.create_dataset("elevationAngle", data=np.ascontiguousarray(look))
        rg.create_dataset("losUnitVectorX", data=losx)
        rg.create_dataset("losUnitVectorY", data=losy)
        rg.create_dataset("heightAboveEllipsoid", data=heights)
        rg.create_dataset("xCoordinates", data=cx)
        rg.create_dataset("yCoordinates", data=cy)
        projd = rg.create_dataset("projection", data=np.int32(epsg))
        projd.attrs["epsg_code"] = np.int32(epsg)

    return path


@pytest.fixture
def gunw_factory(tmp_path):
    """Return a factory that writes synthetic GUNW files into ``tmp_path``."""
    counter = {"n": 0}

    def _make(**kwargs):
        counter["n"] += 1
        name = kwargs.pop("name", f"gunw_{counter['n']}.h5")
        path = tmp_path / name
        return write_synthetic_gunw(path, **kwargs)

    return _make
