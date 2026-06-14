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
):
    """Write a minimal but structurally faithful GSLC HDF5 file.

    ``x`` is ascending; ``y`` is descending for a ``Descending`` pass and
    ascending otherwise (matching how NISAR stores geocoded grids). The
    ``yCoordinateSpacing`` is stored signed, as in real products.
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

    with h5py.File(path, "w") as f:
        grid = f.create_group(GRID)
        grid.create_dataset(polarization, data=data, chunks=(min(64, ny), min(64, nx)))
        grid.create_dataset("xCoordinates", data=x_coords)
        grid.create_dataset("yCoordinates", data=y_coords)
        grid.create_dataset("xCoordinateSpacing", data=np.float64(dx))
        grid.create_dataset("yCoordinateSpacing", data=np.float64(y_spacing))
        proj = grid.create_dataset("projection", data=np.int32(epsg))
        proj.attrs["epsg_code"] = np.int32(epsg)

        ident = f.create_group(IDENT)
        ident.create_dataset("orbitPassDirection", data=np.bytes_(direction))
        ident.create_dataset("zeroDopplerStartTime", data=np.bytes_(datetime_str))

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
