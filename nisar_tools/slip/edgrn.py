"""EDGRN Green's-function tables: what a layered crust does to a point source.

A homogeneous half-space has one elastic modulus everywhere, so its response to a
buried source is a closed form and :mod:`nisar_tools.slip._tde` evaluates it
directly. A layered crust has no closed form: the response has to be built by
wavenumber integration through the layer stack, which is what Rongjiang Wang's
**EDGRN** (Wang, Lorenzo-Martin & Roth 2003, *Computers & Geosciences* **29**,
195-207) does, once, into a table indexed by source depth and epicentral
distance. **EDCMP** then reads that table and superposes point sources over a
fault. This module is the table half; :mod:`nisar_tools.slip.layered` is the
superposition half.

Three tables are needed because a general moment tensor decomposes into three
sources with distinct azimuthal symmetry, and the tables store each with its
azimuthal dependence stripped out:

===== ============================== ================
file  source                         azimuthal order
===== ============================== ================
ss    ``m12 = m21 = M0``             2
ds    ``m13 = m31 = M0``             1
cl    ``m33 = M0, m11 = m22 = -M0/2``0
===== ============================== ================

so a table row is a function of ``(r, z_source)`` alone and the azimuth is put
back at evaluation time (``edgmoment.f`` sets ``ms = 2, 1, 0`` for exactly this
reason). :mod:`nisar_tools.slip.layered` does the putting back.

**Coordinates here are EDGRN's, not the package's.** ``x`` is north, ``y`` is
east and ``z`` is **down**; ``uz`` in a table is positive downward. The
conversion happens at the boundary of :mod:`nisar_tools.slip.layered`, which is
also where the reference does it -- the apparent axis swap at the top of
``calc_green_gps_3d_edcmp_xyz.m`` is that conversion and not, as it looks, a bug.

Running EDGRN itself is optional and external, exactly as in the reference: the
user owns the Earth model. :func:`run_edgrn` will drive it through ``pygrnwang``
if that is installed, and :meth:`EdgrnTables.homogeneous` synthesises a
uniform-medium table analytically so the whole layered pipeline can be tested
without any Fortran at all.
"""

import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

#: Source types, in EDGRN's own order (``istype`` 1, 2, 3).
SOURCE_TYPES = ("ss", "ds", "cl")

#: Displacement columns of a table row, in file order.
_DISPLACEMENT_COLUMNS = ("uz", "ur", "ut")


class VelocityModel:
    """A layered elastic Earth: depth, P and S velocity, density.

    The input EDGRN needs, and the source of the depth-dependent rigidity
    ``mu = rho * vs**2`` that a moment calculation wants. A discontinuity is
    written as two rows at the same depth; a gradient as two rows at different
    depths with different values -- EDGRN's own convention, preserved so a model
    can be round-tripped through an ``edgrn.inp`` unchanged.
    """

    def __init__(self, depth, vp, vs, rho, name=None):
        self.depth = np.asarray(depth, dtype=float).ravel()
        self.vp = np.asarray(vp, dtype=float).ravel()
        self.vs = np.asarray(vs, dtype=float).ravel()
        self.rho = np.asarray(rho, dtype=float).ravel()
        self.name = name
        sizes = {self.depth.size, self.vp.size, self.vs.size, self.rho.size}
        if len(sizes) != 1:
            raise ValueError("depth, vp, vs and rho must have the same length")
        if self.depth.size == 0:
            raise ValueError("A velocity model needs at least one layer")
        if np.any(np.diff(self.depth) < 0):
            raise ValueError("Velocity-model depths must be non-decreasing")
        if np.any(self.vs <= 0) or np.any(self.rho <= 0):
            raise ValueError("vs and rho must be positive")

    @classmethod
    def uniform(cls, vp=6000.0, vs=3464.0, rho=2670.0, max_depth=100e3):
        """One layer -- the model whose answer must match the half-space engine.

        The defaults give ``mu = rho * vs**2`` = 32.0 GPa and a Poisson ratio of
        0.25, i.e. a Poisson solid close to the 30 GPa this package assumes when
        no model is supplied.
        """
        return cls([0.0, max_depth], [vp, vp], [vs, vs], [rho, rho], name="uniform")

    @classmethod
    def from_file(cls, path):
        """Read ``depth vp vs rho`` rows (metres, m/s, m/s, kg/m^3)."""
        path = Path(path).expanduser()
        rows = []
        for line in path.read_text().splitlines():
            line = line.split("#")[0].strip()
            if not line:
                continue
            fields = [float(v) for v in re.split(r"[\s,]+", line)]
            # Tolerate EDGRN's own five-column form, whose first column is an index.
            rows.append(fields[-4:] if len(fields) >= 5 else fields[:4])
        table = np.asarray(rows, dtype=float)
        if table.ndim != 2 or table.shape[1] != 4:
            raise ValueError(f"{path} must have four columns: depth vp vs rho")
        return cls(*table.T, name=path.stem)

    @property
    def mu(self):
        """Shear modulus per row, ``rho * vs**2``, in pascals."""
        return self.rho * self.vs ** 2

    @property
    def lam(self):
        """First Lame parameter per row, ``rho * (vp**2 - 2 vs**2)``."""
        return self.rho * (self.vp ** 2 - 2.0 * self.vs ** 2)

    def poisson(self, depth=0.0):
        """Poisson's ratio at ``depth``."""
        lam, mu = self.at(depth, "lam"), self.at(depth, "mu")
        return float(lam / (2.0 * (lam + mu)))

    def at(self, depth, field="mu"):
        """A property sampled at one or more depths, piecewise-linearly.

        Depths below the deepest row take the deepest row's value, which is what
        a half-space bottom layer means.
        """
        values = getattr(self, field) if field in ("vp", "vs", "rho") else getattr(self, field)
        depth = np.abs(np.asarray(depth, dtype=float))
        # np.interp already clamps outside the range at both ends.
        return np.interp(depth, self.depth, values)

    def __len__(self):
        return self.depth.size

    def __repr__(self):
        return (f"<VelocityModel {self.name or ''} {len(self)} rows "
                f"0-{self.depth.max() / 1e3:.0f}km mu={self.mu.min() / 1e9:.1f}"
                f"-{self.mu.max() / 1e9:.1f}GPa>")


class EdgrnTables:
    """The three EDGRN databases on their shared ``(r, z_source)`` grid.

    ``ss``, ``ds`` and ``cl`` are each a dict of ``(nr, nz)`` arrays keyed by
    ``uz``, ``ur``, ``ut`` -- the displacement at the receiver depth ``zrec`` per
    unit source area times slip, with the azimuthal factor removed. ``cl``'s
    ``ut`` is identically zero: an axisymmetric source drives no transverse
    motion.
    """

    def __init__(self, r, z, tables, zrec=0.0, lam=None, mu=None, attrs=None):
        self.r = np.asarray(r, dtype=float).ravel()
        self.z = np.asarray(z, dtype=float).ravel()
        self.tables = {k: {c: np.asarray(v[c], dtype=float) for c in _DISPLACEMENT_COLUMNS}
                       for k, v in tables.items()}
        missing = set(SOURCE_TYPES) - set(self.tables)
        if missing:
            raise ValueError(f"Missing EDGRN table(s): {sorted(missing)}")
        shape = (self.r.size, self.z.size)
        for kind, table in self.tables.items():
            for component, values in table.items():
                if values.shape != shape:
                    raise ValueError(
                        f"{kind}.{component} has shape {values.shape}; expected {shape} "
                        "= (nr, nz)"
                    )
        self.zrec = float(zrec)
        self.lam = lam
        self.mu = mu
        self.attrs = dict(attrs or {})

    # -- reading -----------------------------------------------------------
    @classmethod
    def from_input_file(cls, path):
        """Read the three databases an ``edgrn.inp`` names.

        The directory and the three filenames sit on the **fifth line containing
        no** ``#``, single-quote delimited -- ``getedgrn.m``'s rule, kept because
        it is the only thing that locates the tables. Relative directories are
        resolved against the input file rather than the working directory, which
        is why the reference has to ``cd`` before calling its own reader.
        """
        path = Path(path).expanduser()
        lines = [ln for ln in path.read_text().splitlines() if "#" not in ln and ln.strip()]
        if len(lines) < 5:
            raise ValueError(
                f"{path} has {len(lines)} comment-free lines; an EDGRN input file "
                "names its Green's-function files on the fifth"
            )
        quoted = re.findall(r"'([^']*)'", lines[4])
        if len(quoted) < 4:
            raise ValueError(
                f"Line 5 of {path} should quote a directory and three filenames "
                f"(ss, ds, cl); found {quoted}"
            )
        directory = (path.parent / quoted[0]).resolve()
        return cls.from_files({k: directory / name
                               for k, name in zip(SOURCE_TYPES, quoted[1:4])})

    @classmethod
    def from_files(cls, paths):
        """Read three already-located database files, keyed ``ss``/``ds``/``cl``."""
        header, tables = None, {}
        for kind in SOURCE_TYPES:
            head, columns = _read_table(Path(paths[kind]).expanduser(), kind)
            if header is None:
                header = head
            elif not np.allclose(head[:7], header[:7]):
                raise ValueError(
                    f"The {kind} database is on a different grid than the ss one "
                    f"({head[:7]} vs {header[:7]}); they must share (nr, nz)."
                )
            tables[kind] = columns

        nr, r1, r2, nz, z1, z2, zrec, lam, mu = header
        return cls(np.linspace(r1, r2, int(nr)), np.linspace(z1, z2, int(nz)),
                   tables, zrec=zrec, lam=lam, mu=mu,
                   attrs={"source": "edgrn"})

    # -- synthesis ---------------------------------------------------------
    @classmethod
    def homogeneous(cls, nu=0.25, r=None, z=None, n_azimuth=16, source_size=1.0):
        """Synthesise the tables a *uniform* medium would produce.

        The point of this is testing. A layered code handed a one-layer model has
        to reproduce the homogeneous half-space answer, and EDCMP is built around
        that idea -- its own input file carries a "layered (1) or homogeneous (0)"
        switch and it ships Okada's routines for the second branch. Synthesising
        the tables here gives the same check without any Fortran, and it exercises
        every part of :mod:`nisar_tools.slip.layered`: the discretisation, the
        bilinear interpolation, the azimuthal recombination, the north/east/down
        conventions and the quadrature.

        What it does **not** test is the physics of layering, since the field it
        tabulates comes from :func:`nisar_tools.slip._tde.tde_disp_hs` -- the same
        solution the engine is compared against. That is a test of the machinery,
        deliberately, and it is the part with all the transcription risk.

        Each source is built as a dislocation small enough to be a point at the
        table's own resolution, and the azimuthal orders are projected out
        numerically rather than derived by hand -- the exact inverse of the
        recombination in :mod:`nisar_tools.slip.layered`. The CLVD, which is not a
        dislocation at all, comes from three tensile sources::

            CLVD = (1/2) * [ T_z - (T_x + T_y) / 2 ]

        because a unit tensile source with normal ``w`` has moment tensor
        ``lam*I + 2*mu*w(x)w``, so the isotropic parts cancel and what is left is
        ``mu*(z(x)z - x(x)x/2 - y(x)y/2)`` -- the CLVD, exactly.
        """
        from ._tde import tde_disp_hs

        r = np.linspace(0.0, 400e3, 201) if r is None else np.asarray(r, dtype=float)
        z = np.linspace(500.0, 50e3, 100) if z is None else np.asarray(z, dtype=float)
        phi = np.linspace(0.0, 2.0 * np.pi, int(n_azimuth), endpoint=False)

        # Receivers on the free surface. EDGRN's azimuth is measured from north
        # towards east, so (north, east) = r * (cos phi, sin phi).
        rr, pp = np.meshgrid(r, phi, indexing="ij")
        north, east = rr * np.cos(pp), rr * np.sin(pp)
        obs_x, obs_y = east.ravel(), north.ravel()          # ENU east, north
        obs_z = np.zeros(obs_x.size)

        area = 0.5 * source_size ** 2
        tables = {kind: {c: np.empty((r.size, z.size)) for c in _DISPLACEMENT_COLUMNS}
                  for kind in SOURCE_TYPES}

        for j, depth in enumerate(z):
            fields = {}
            for label, normal, slip in _SOURCE_GEOMETRY:
                p1, p2, p3, components = _point_dislocation(
                    normal, slip, depth, source_size)
                ue, un, uu = tde_disp_hs(obs_x, obs_y, obs_z, p1, p2, p3,
                                         *components, nu)
                fields[label] = np.stack([ue, un, uu]) / area

            combined = {
                "ss": fields["ss"],
                "ds": fields["ds"],
                "cl": 0.5 * (fields["tz"] - 0.5 * (fields["tx"] + fields["ty"])),
            }
            for kind, enu in combined.items():
                ue, un, uu = (c.reshape(r.size, phi.size) for c in enu)
                # ENU -> EDGRN's north/east/down, then to cylindrical.
                ux, uy, uz = un, ue, -uu
                ur = ux * np.cos(phi) + uy * np.sin(phi)
                ut = -ux * np.sin(phi) + uy * np.cos(phi)
                for component, values in (("uz", uz), ("ur", ur), ("ut", ut)):
                    tables[kind][component][:, j] = _project(values, phi, kind, component)

        return cls(r, z, tables, zrec=0.0,
                   mu=None, lam=None,
                   attrs={"source": "synthetic_homogeneous", "nu": float(nu)})

    # -- use ---------------------------------------------------------------
    def interpolate(self, kind, component, distance, depth):
        """Bilinear lookup in ``(r, z_source)``, EDCMP's own stencil.

        Outside the tabulated range the value is taken from the edge cell rather
        than extrapolated -- EDCMP clamps the distance index below ``r1`` and the
        table is padded with zeros beyond ``r2``, so a source outside the table
        contributes nothing rather than something arbitrary.
        """
        table = self.tables[kind][component]
        nr, nz = table.shape
        dr = (self.r[-1] - self.r[0]) / (nr - 1) if nr > 1 else 1.0
        dz = (self.z[-1] - self.z[0]) / (nz - 1) if nz > 1 else 1.0

        distance = np.asarray(distance, dtype=float)
        depth = np.asarray(depth, dtype=float)
        fi = np.clip((distance - self.r[0]) / dr, 0.0, nr - 1)
        fj = np.clip((depth - self.z[0]) / dz, 0.0, nz - 1)
        i0 = np.clip(np.floor(fi).astype(int), 0, nr - 2 if nr > 1 else 0)
        j0 = np.clip(np.floor(fj).astype(int), 0, nz - 2 if nz > 1 else 0)
        ti = fi - i0
        tj = fj - j0
        i1 = np.minimum(i0 + 1, nr - 1)
        j1 = np.minimum(j0 + 1, nz - 1)

        return ((1 - ti) * (1 - tj) * table[i0, j0]
                + ti * (1 - tj) * table[i1, j0]
                + (1 - ti) * tj * table[i0, j1]
                + ti * tj * table[i1, j1])

    def __repr__(self):
        return (f"<EdgrnTables nr={self.r.size} r={self.r[0] / 1e3:.0f}-"
                f"{self.r[-1] / 1e3:.0f}km nz={self.z.size} "
                f"z={self.z[0] / 1e3:.1f}-{self.z[-1] / 1e3:.0f}km "
                f"{self.attrs.get('source', '?')}>")


# Dislocation geometry for each canonical source, in ENU. `normal` is the fault
# plane's unit normal and `slip` the unit slip direction of the hanging wall.
#   ss  m12 = m21: plane normal north, slip east   (a vertical east-striking fault)
#   ds  m13 = m31: plane normal north, slip down
#   tx/ty/tz     : unit tensile sources, combined into the CLVD
_SOURCE_GEOMETRY = (
    ("ss", (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)),
    ("ds", (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)),
    ("tx", (0.0, 1.0, 0.0), (0.0, 1.0, 0.0)),      # normal north -> x(x)x in NED
    ("ty", (1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),      # normal east  -> y(x)y in NED
    ("tz", (0.0, 0.0, 1.0), (0.0, 0.0, 1.0)),      # normal up    -> z(x)z in NED
)


def _point_dislocation(normal, slip, depth, size):
    """A tiny triangle with the requested normal, and its ``(ss, ds, ts)``.

    Solving for the slip components rather than naming them avoids the sign trap
    in the triangular-dislocation basis: ``Vstrike = cross(eZ, Vnorm)`` can come
    out anti-parallel to the direction you meant, and the resulting moment tensor
    is then the negative of the one intended -- which produces a perfectly smooth,
    entirely wrong table. Projecting the desired slip vector onto the element's
    own orthonormal basis is correct whatever that basis turns out to be.
    """
    from ._tde import _element_basis

    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    # Any orthonormal pair (u, v) with u x v = n makes cross(P2-P1, P3-P1) = n.
    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(helper, n)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)

    centre = np.array([0.0, 0.0, -abs(depth)])
    p1 = centre - size * (u + v) / 3.0
    p2 = p1 + size * u
    p3 = p1 + size * v

    vnorm, vstrike, vdip = _element_basis(p1, p2, p3)
    d = np.asarray(slip, dtype=float)
    d = d / np.linalg.norm(d)
    return p1, p2, p3, (float(d @ vstrike), float(d @ vdip), float(d @ vnorm))


def _project(values, phi, kind, component):
    """Strip the azimuthal factor, inverting the recombination exactly.

    ``values`` is ``(nr, nphi)``. The factors are the ones
    :mod:`nisar_tools.slip.layered` puts back: order 2 for ``ss``, 1 for ``ds``,
    0 for ``cl``, with ``ut`` carrying the quadrature partner of the ``uz``/``ur``
    factor.
    """
    if kind == "cl":
        return np.zeros(values.shape[0]) if component == "ut" else values.mean(axis=1)
    order = 2 if kind == "ss" else 1
    basis = (np.cos(order * phi) if (kind == "ss") == (component == "ut")
             else np.sin(order * phi))
    return 2.0 * (values * basis).mean(axis=1)


def _read_table(path, kind):
    """One EDGRN database: its metadata line and its displacement columns.

    The column count is read from the file rather than assumed. Stock EDGRN 2.0
    writes **ten** columns for ``ss`` and ``ds`` but only **seven** for ``cl``
    (``edgmain.f`` drops ``ut``, ``ert`` and ``etz``, which vanish for an
    axisymmetric source), while ``getedgrn.m`` and SlipSolve's own converter both
    read ten and assert the row count -- so a stock ``cl`` file would fail there.
    Only the leading displacement columns are used either way; the six strain
    columns and ``duz/dr`` are read and discarded.
    """
    text = Path(path).read_text().splitlines()
    body = [ln for ln in text if ln.strip() and "#" not in ln]
    if not body:
        raise ValueError(f"{path} has no data lines")

    header = [float(v) for v in re.split(r"[\s,]+", _fortran_floats(body[0]).strip())]
    if len(header) < 9:
        raise ValueError(
            f"{path}: the metadata line should hold nr r1 r2 nz z1 z2 zrec lambda mu; "
            f"found {len(header)} values"
        )
    nr, nz = int(header[0]), int(header[3])

    rows = np.array([[float(v) for v in re.split(r"[\s,]+", _fortran_floats(ln).strip())]
                     for ln in body[1:]], dtype=float)
    if rows.shape[0] != nr * nz:
        raise ValueError(
            f"{path} has {rows.shape[0]} data rows; the header declares "
            f"nr*nz = {nr}*{nz} = {nr * nz}"
        )

    # Row order is z outer, r inner (`do izs ... do j=1,nr` in edgmain.f), which
    # is Fortran order for an (nr, nz) array.
    n_disp = 3 if rows.shape[1] >= 10 else 2
    columns = {}
    for index, name in enumerate(_DISPLACEMENT_COLUMNS):
        columns[name] = (rows[:, index].reshape(nr, nz, order="F") if index < n_disp
                         else np.zeros((nr, nz)))
    if kind == "cl":
        columns["ut"] = np.zeros((nr, nz))
    return header[:9], columns


def _fortran_floats(line):
    """Turn Fortran's ``1.0D+03`` exponent marker into one Python can parse."""
    return re.sub(r"([0-9])[dD]([+-]?[0-9])", r"\1e\2", line)


def write_edgrn_input(path, model, obs_depth=0.0, nr=201, r_max=400e3,
                      nz=100, z_min=500.0, z_max=50e3, srate=12.0,
                      directory="edgrnfcts/", names=("edgrn.ss", "edgrn.ds", "edgrn.cl")):
    """Write an ``edgrn.inp`` for ``model`` on the given ``(r, z)`` grid.

    The grid is the table's resolution, and the interpolation in
    :meth:`EdgrnTables.interpolate` is bilinear, so it wants to be fine enough
    that the response is nearly linear across a cell near the fault. ``z_min``
    is not zero because a source exactly at the free surface radiates nothing.
    """
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = "\n".join(
        f"  {i + 1:3d}  {d:14.4e}  {vp:14.4e}  {vs:14.4e}  {rho:14.4e}"
        for i, (d, vp, vs, rho) in enumerate(
            zip(model.depth, model.vp, model.vs, model.rho))
    )
    path.write_text(
        "# EDGRN input written by nisar_tools\n"
        f"  {obs_depth:.6e}\n"
        f" {int(nr)}  {0.0:.6e}  {r_max:.6e}\n"
        f" {int(nz)}  {z_min:.6e}  {z_max:.6e}\n"
        f" {srate:.4f}\n"
        f" '{directory}'  '{names[0]}'  '{names[1]}'  '{names[2]}'\n"
        f"  {len(model)}\n{rows}\n"
    )
    (path.parent / directory).mkdir(parents=True, exist_ok=True)
    return path


def run_edgrn(model, workdir, executable=None, **grid):
    """Generate tables by actually running EDGRN, and read them back.

    Optional and lazily resolved, in the same style as the package's other
    external tools. The reference implementation does not do this at all -- its
    stage 7 only converts EDGRN's text output, on the reasoning that the user owns
    the Earth model -- so this is a convenience, not a port.

    Resolution order: an explicit ``executable``, then ``edgrn`` on ``PATH``, then
    the copy ``pygrnwang`` bundles and builds. Raises with all three named if
    none is available.
    """
    workdir = Path(workdir).expanduser()
    workdir.mkdir(parents=True, exist_ok=True)
    inp = write_edgrn_input(workdir / "edgrn.inp", model, **grid)

    binary = executable or shutil.which("edgrn") or _pygrnwang_edgrn()
    if binary is None:
        raise RuntimeError(
            "No EDGRN executable found. Pass executable=, put `edgrn` on PATH, or "
            "`pip install pygrnwang` (which bundles Rongjiang Wang's Fortran and "
            "builds it with gfortran). Alternatively supply tables directly with "
            "EdgrnTables.from_input_file, or use EdgrnTables.homogeneous for a "
            "uniform medium."
        )

    # EDGRN reads the input file's name on stdin and writes relative to its own
    # working directory, so it has to be run from there.
    subprocess.run([str(binary)], input=f"{inp.name}\n", text=True, check=True,
                   cwd=workdir, capture_output=True)
    return EdgrnTables.from_input_file(inp)


def _pygrnwang_edgrn():
    """The EDGRN binary ``pygrnwang`` ships, if it is installed and built."""
    try:
        import pygrnwang
    except ImportError:
        return None
    root = Path(pygrnwang.__file__).parent
    for candidate in sorted(root.rglob("edgrn*")):
        if candidate.is_file() and candidate.stat().st_mode & 0o111:
            return candidate
    return None
