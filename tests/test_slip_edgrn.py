"""EDGRN table reading, writing and synthesis.

The file format is the whole risk here: a table is ``nr*nz`` rows of numbers with
no self-description beyond one metadata line, so a transposed reshape or a
miscounted column produces a perfectly plausible table that is wrong everywhere.
"""

import numpy as np
import pytest

from nisar_tools.slip.edgrn import (
    SOURCE_TYPES,
    EdgrnTables,
    VelocityModel,
    write_edgrn_input,
)


def _write_table(path, nr, nz, r_range, z_range, columns, n_columns=10):
    """A synthetic EDGRN database in the real format, values ``100*ir + iz``."""
    header = (f"# Green functions calculated with the program edgrn\n"
              f"# nr, r1[m], r2[m]; nzs, zs1[m], zs2[m]; obs.depth[m];"
              f" lambda[Pa], mu[Pa]\n"
              f"{nr:4d}{r_range[0]:13.5E}{r_range[1]:13.5E}"
              f"{nz:4d}{z_range[0]:13.5E}{z_range[1]:13.5E}"
              f"{0.0:13.5E}{3.0e10:13.5E}{3.0e10:13.5E}\n")
    rows = []
    for iz in range(nz):                       # z outer, r inner: edgmain.f's order
        for ir in range(nr):
            values = [columns[c][ir, iz] for c in ("uz", "ur", "ut")][:n_columns]
            values = values + [0.0] * (n_columns - len(values))
            rows.append("".join(f"{v:14.6E}" for v in values))
    path.write_text(header + "\n".join(rows) + "\n")


def _ramp(nr, nz):
    ir, iz = np.meshgrid(np.arange(nr), np.arange(nz), indexing="ij")
    return {"uz": 100.0 * ir + iz, "ur": -1.0 * ir + 10.0 * iz, "ut": 1.0 * ir * iz}


# -- the velocity model ------------------------------------------------------

def test_velocity_model_moduli():
    model = VelocityModel.uniform(vp=6000.0, vs=3464.0, rho=2670.0)
    np.testing.assert_allclose(model.mu, 2670.0 * 3464.0 ** 2)
    # vp/vs = sqrt(3) is a Poisson solid, nu = 0.25.
    assert abs(model.poisson() - 0.25) < 1e-3


def test_velocity_model_samples_and_clamps():
    model = VelocityModel([0.0, 10e3, 30e3], [4.0e3, 6.0e3, 6.5e3],
                          [2.0e3, 3.5e3, 3.7e3], [2.4e3, 2.7e3, 2.9e3])
    shallow, deep = model.at(0.0, "mu"), model.at(30e3, "mu")
    assert shallow < deep, "the shallow crust must be the softer end"
    # Below the deepest row the bottom layer continues as a half-space.
    assert model.at(100e3, "mu") == pytest.approx(deep)
    assert model.at(-5e3, "mu") == model.at(5e3, "mu"), "depth sign must not matter"


def test_velocity_model_from_file(tmp_path):
    path = tmp_path / "crust.txt"
    path.write_text("# depth vp vs rho\n0 5000 2900 2600\n15000 6500 3700 2900\n")
    model = VelocityModel.from_file(path)
    assert len(model) == 2
    np.testing.assert_allclose(model.depth, [0.0, 15e3])

    # EDGRN's own five-column form leads with a row index.
    indexed = tmp_path / "edgrn_style.txt"
    indexed.write_text("  1  0.0 5000 2900 2600\n  2  15000 6500 3700 2900\n")
    np.testing.assert_allclose(VelocityModel.from_file(indexed).vp, model.vp)


def test_velocity_model_rejects_ragged_input():
    with pytest.raises(ValueError, match="same length"):
        VelocityModel([0, 1], [1, 2], [1], [1, 2])


# -- reading -----------------------------------------------------------------

def test_table_reshape_is_r_fastest(tmp_path):
    """``do izs ... do j=1,nr`` -- so r is the inner loop and Fortran order applies.

    A C-order reshape here transposes the table, which still has the right shape
    when ``nr == nz`` and is wrong everywhere. The ramp values make the
    transposition visible.
    """
    nr, nz = 5, 4
    columns = _ramp(nr, nz)
    paths = {}
    for kind in SOURCE_TYPES:
        paths[kind] = tmp_path / f"edgrn.{kind}"
        _write_table(paths[kind], nr, nz, (0.0, 40e3), (1e3, 4e3), columns,
                     n_columns=7 if kind == "cl" else 10)

    tables = EdgrnTables.from_files(paths)
    assert tables.tables["ss"]["uz"].shape == (nr, nz)
    np.testing.assert_allclose(tables.tables["ss"]["uz"], columns["uz"])
    np.testing.assert_allclose(tables.r, np.linspace(0.0, 40e3, nr))
    np.testing.assert_allclose(tables.z, np.linspace(1e3, 4e3, nz))


def test_cl_table_may_have_seven_columns(tmp_path):
    """Stock EDGRN 2.0 writes 7 columns for cl and 10 for ss/ds.

    ``edgmain.f`` drops ``ut``, ``ert`` and ``etz`` from the CLVD file because an
    axisymmetric source has none of them. ``getedgrn.m`` and SlipSolve's own
    converter both read 10 regardless and assert the row count, so a stock file
    fails there -- hence reading the width from the file instead of assuming it.
    """
    nr, nz = 4, 3
    columns = _ramp(nr, nz)
    paths = {}
    for kind in SOURCE_TYPES:
        paths[kind] = tmp_path / f"edgrn.{kind}"
        _write_table(paths[kind], nr, nz, (0.0, 30e3), (1e3, 3e3), columns,
                     n_columns=7 if kind == "cl" else 10)

    tables = EdgrnTables.from_files(paths)
    np.testing.assert_allclose(tables.tables["cl"]["uz"], columns["uz"])
    assert not np.any(tables.tables["cl"]["ut"]), "a CLVD drives no transverse motion"


def test_input_file_locates_the_databases(tmp_path):
    """The directory and filenames are on the 5th line with no ``#`` in it."""
    nr, nz = 3, 3
    columns = _ramp(nr, nz)
    (tmp_path / "fcts").mkdir()
    for kind in SOURCE_TYPES:
        _write_table(tmp_path / "fcts" / f"izmhs.{kind}", nr, nz,
                     (0.0, 20e3), (1e3, 3e3), columns,
                     n_columns=7 if kind == "cl" else 10)

    inp = tmp_path / "edgrn.inp"
    inp.write_text(
        "# a comment line, skipped even though it holds numbers 1 2 3\n"
        "  0.00d+00\n"
        "  3  0.0d+00  20.0d+03\n"
        "  3  1.0d+03  3.0d+03\n"
        "  12.0\n"
        "  './fcts/'  'izmhs.ss'  'izmhs.ds'  'izmhs.cl'\n"
        "  1\n"
        "  1  0.0  5000  2900  2600\n"
    )
    tables = EdgrnTables.from_input_file(inp)
    np.testing.assert_allclose(tables.tables["ds"]["ur"], columns["ur"])


def test_fortran_exponents_parse(tmp_path):
    """Fortran writes ``1.0D+03``; Python needs ``1.0e+03``."""
    from nisar_tools.slip.edgrn import _fortran_floats

    assert _fortran_floats(" 3  0.0D+00  2.0d-03 ").split() == ["3", "0.0e+00", "2.0e-03"]


def test_row_count_mismatch_is_refused(tmp_path):
    path = tmp_path / "edgrn.ss"
    _write_table(path, 4, 3, (0.0, 30e3), (1e3, 3e3), _ramp(4, 3))
    text = path.read_text().splitlines()
    path.write_text("\n".join(text[:-2]) + "\n")          # drop two rows
    with pytest.raises(ValueError, match="data rows"):
        EdgrnTables.from_files(dict.fromkeys(SOURCE_TYPES, path))


# -- interpolation -----------------------------------------------------------

def test_bilinear_interpolation_is_exact_on_a_ramp():
    nr, nz = 6, 5
    r, z = np.linspace(0.0, 50e3, nr), np.linspace(1e3, 5e3, nz)
    rr, zz = np.meshgrid(r, z, indexing="ij")
    linear = 3.0 + 2.0e-4 * rr - 1.0e-3 * zz
    tables = EdgrnTables(r, z, {k: {"uz": linear, "ur": linear, "ut": linear}
                                for k in SOURCE_TYPES})

    probe_r = np.array([0.0, 12345.0, 50e3])
    probe_z = np.array([1e3, 2345.0, 5e3])
    np.testing.assert_allclose(
        tables.interpolate("ss", "uz", probe_r, probe_z),
        3.0 + 2.0e-4 * probe_r - 1.0e-3 * probe_z, rtol=1e-12,
    )


def test_interpolation_clamps_outside_the_table():
    """Outside the grid EDCMP clamps rather than extrapolating."""
    r, z = np.linspace(0.0, 10e3, 3), np.linspace(1e3, 3e3, 3)
    values = np.arange(9.0).reshape(3, 3)
    tables = EdgrnTables(r, z, {k: {"uz": values, "ur": values, "ut": values}
                                for k in SOURCE_TYPES})
    edge = tables.interpolate("ss", "uz", 10e3, 3e3)
    assert tables.interpolate("ss", "uz", 1e9, 1e9) == pytest.approx(edge)


def test_table_shape_is_validated():
    r, z = np.linspace(0, 1, 4), np.linspace(0, 1, 3)
    bad = {k: {"uz": np.zeros((3, 4)), "ur": np.zeros((3, 4)), "ut": np.zeros((3, 4))}
           for k in SOURCE_TYPES}
    with pytest.raises(ValueError, match=r"expected \(4, 3\)"):
        EdgrnTables(r, z, bad)


# -- synthesis ---------------------------------------------------------------

def test_synthetic_tables_have_the_right_symmetries():
    """Each source type's own azimuthal order, and the CLVD's silence.

    ``ut`` for the dip-slip table comes out zero at the free surface -- an m13
    double couple drives no transverse surface motion in a half-space -- which is
    physics, not an empty column. It is checked here so that a future non-zero
    value from a real layered table is understood as the layering doing something,
    not as a regression.
    """
    tables = EdgrnTables.homogeneous(
        r=np.linspace(0.0, 100e3, 41), z=np.linspace(1e3, 20e3, 11), n_azimuth=16)

    assert not np.any(tables.tables["cl"]["ut"])
    for kind in ("ss", "ds"):
        assert np.abs(tables.tables[kind]["uz"]).max() > 0
        assert np.abs(tables.tables[kind]["ur"]).max() > 0
    assert np.abs(tables.tables["ds"]["ut"]).max() < 1e-6 * np.abs(
        tables.tables["ds"]["ur"]).max()
    assert np.abs(tables.tables["ss"]["ut"]).max() > 0.05 * np.abs(
        tables.tables["ss"]["ur"]).max()


def test_synthetic_tables_decay_with_distance_and_depth():
    tables = EdgrnTables.homogeneous(
        r=np.linspace(1e3, 200e3, 60), z=np.linspace(2e3, 30e3, 15), n_azimuth=12)
    profile = np.abs(tables.tables["ss"]["ur"][:, 0])
    assert profile[5] > profile[-1] * 10, "the near field must dominate"
    deep = np.abs(tables.tables["ss"]["ur"][10, :])
    assert deep[0] > deep[-1], "a deeper source must move the surface less"


# -- writing -----------------------------------------------------------------

def test_write_edgrn_input_round_trips_the_grid(tmp_path):
    model = VelocityModel([0.0, 20e3], [5.5e3, 6.5e3], [3.2e3, 3.7e3], [2.6e3, 2.9e3])
    path = write_edgrn_input(tmp_path / "edgrn.inp", model,
                             nr=51, r_max=100e3, nz=21, z_min=1e3, z_max=25e3)
    lines = [ln for ln in path.read_text().splitlines() if "#" not in ln and ln.strip()]

    assert len(lines) >= 5
    assert int(lines[1].split()[0]) == 51
    assert int(lines[2].split()[0]) == 21
    assert lines[4].count("'") == 8, "a directory and three filenames, quoted"
    assert (tmp_path / "edgrnfcts").is_dir()


def test_run_edgrn_says_what_to_do_when_there_is_no_binary(tmp_path):
    from nisar_tools.slip.edgrn import run_edgrn

    model = VelocityModel.uniform()
    with pytest.raises(RuntimeError, match="pygrnwang|executable"):
        run_edgrn(model, tmp_path, executable=None if not _has_edgrn() else "/nonexistent")


def _has_edgrn():
    import shutil

    from nisar_tools.slip.edgrn import _pygrnwang_edgrn

    return shutil.which("edgrn") or _pygrnwang_edgrn()
