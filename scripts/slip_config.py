"""Shared configuration for the three detached slip-inversion stages.

The work splits into three because the costs are very different and only the
first is expensive to redo:

    run_sampling.py    resample every scene onto one lattice, then iterate
                       sample -> solve -> re-sample until the model settles.
                       Minutes. Writes the observations.
    run_lcurve.py      sweep the smoothing weight over those observations.
                       One Green's matrix, many solves.
    run_inversion.py   solve once at the weight the L-curve picked, and write
                       the model, its text table and the review figures.

They share this file so a mesh or a sampling parameter cannot drift between
them -- which would be silent, since each stage's output looks fine on its own.

The Green's matrix is **not** passed between stages: it is the largest object in
the problem and is deliberately never saved (``SlipModel.save`` omits it). Each
stage re-assembles it, which measured ~25 s at 1106 elements against 8000
observations -- far cheaper than the alternatives.

Everything here is edited in place, and the settings worth changing between runs
also read an environment variable, so a detached job can be re-launched at a new
value without touching the file:

    NISAR_WORK_DIR      where the workspace and every output live
    NISAR_FAULT         the fault trace (.kml or two-column lon/lat ASCII)
    NISAR_EDGE_LENGTH   element size, metres
    NISAR_MAX_DEPTH     bottom of the fault, metres
    NISAR_DIP           one dip, or a list -- vertical if unset (see DIPS)
    NISAR_SEGMENTS      segment files, one per dip (see SEGMENT_FILES)
    NISAR_BIAS_W        depth-level grading (see CURVE)
    NISAR_DOWN_DIP_LEVELS  node levels down dip -- what makes NISAR_BIAS_W mean
                        a definite ratio (see CURVE)
    NISAR_ENGINE        "layered" (default) or "halfspace" (see ENGINE_KIND)
    NISAR_SMOOTHING     the weight stage 3 solves at
    NISAR_MAX_ROUNDS    sampling rounds; 0 stops at the coarse data-driven set

⚠️ Keep the geometry variables identical across all three stages. Every stage
rebuilds the mesh from this file, and ``load_observations`` checks only the
*frame*, not the mesh -- so changing a dip between stages would sweep a different
mesh than was sampled for, with no error. Export them once per session.
"""

import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # headless: before pyplot is imported anywhere

import matplotlib.pyplot as plt  # noqa: E402

from nisar_tools import LOSStack, Workspace                                    # noqa: E402
from nisar_tools.slip import (                                                 # noqa: E402
    EdgrnTables, FaultMesh, FaultSegment, FaultTrace, LayeredPointSource,
    VelocityModel, run_edgrn, scene_report,
)
from nisar_tools.slip.plot import plot_coverage                                # noqa: E402


# -- parsing the environment overrides ---------------------------------------
def _dips(value):
    """``None``, one dip, or a list of dips -> ``None`` or a list of floats."""
    if value is None or value == "":
        return None
    if isinstance(value, str):
        value = [v for v in re.split(r"[,\s]+", value.strip()) if v]
    elif isinstance(value, (int, float)):
        value = [value]
    return [float(v) for v in value] or None


def _paths(value):
    """The same, for a list of file paths."""
    if value is None or value == "":
        return None
    if isinstance(value, str):
        value = [v for v in re.split(r"[,\s]+", value.strip()) if v]
    return [Path(v).expanduser() for v in value] or None


def _path(value):
    """The same, for a single optional file path."""
    paths = _paths(value)
    return None if paths is None else paths[0]


def _int_or_none(value):
    return None if value is None or value == "" else int(value)


def _float_or_none(value):
    return None if value is None or value == "" else float(value)


# -- where -------------------------------------------------------------------
WORK_DIR = Path(os.environ.get(
    "NISAR_WORK_DIR", "/raid/class239/erapaport/workdir")).expanduser()
OUT_DIR = WORK_DIR / "model_sampling"
FIG_DIR = OUT_DIR / "figures"
FAULT = Path(os.environ.get(
    "NISAR_FAULT",
    "/raid/class239/erapaport/share/Venezuela/Venezuela_fault_trace_2.kml",
)).expanduser()

#: Where the EDGRN tables are generated and then cached -- see :func:`engine`.
EDGRN_DIR = WORK_DIR / "edgrn_venezuela"

#: Stage names inside the workspace, so the three scripts agree on them.
OBS_STAGE = "slip_observations"
LOS_STAGE = "los_{name}_frame"

# -- what --------------------------------------------------------------------
#: Applied to every scene on load, as ``(lon_min, lon_max, lat_min, lat_max)``.
#: ``None`` keeps the full footprint; a per-scene ``"crop"`` key overrides it.
CROP = (-70.0, -66.0, 9.0, 11.5)

#: One entry per scene: ``"stage"`` for a persisted ``los`` stage in the
#: workspace, or ``"grd"`` for a directory of GMT grids.
#:
#: The **key is the track name**, and it is load-bearing -- ``ramp`` keys its
#: nuisance columns on it, so one entry per scene is what lets each carry its own
#: arbitrary offset. Do not merge scenes into one raster to save entries:
#: measured, that cost a factor of 3.5 in peak slip while variance reduction
#: stayed at 98%, so it does not warn you.
#:
#: This mapping is the *only* place the scenes are listed. ``run_sampling.py``
#: does the ``Observations.from_los`` per scene and the ``Observations.concat``
#: for you (inside ``iterate_sampling``, with ``normalize="sqrt_count"``), so
#: there is no parallel list of ``from_los`` calls to keep in step with this one.
#:
#: ``units``/``sign``/``look_direction`` matter only for ``grd`` entries: a
#: ``.grd`` cannot record them, and ``units`` and ``sign`` are silent when wrong.
_ALOS = "/raid/class239/yuri/A2/Venezuela/clean"
SCENES = {
    "A162": {"stage": "los_A162_full_mask"},
    "A061": {"stage": "los_A061_full_mask"},
    "D126": {"stage": "los_D126_full_mask"},
    "D054": {"stage": "los_D054_full_mask"},
    "D134": {"grd": f"{_ALOS}/D134", "units": "cm", "sign": +1,
             "direction": "descending", "look_direction": "right"},
    "D135": {"grd": f"{_ALOS}/D135", "units": "cm", "sign": +1,
             "direction": "descending", "look_direction": "right"},
}

#: Extra keyword arguments for :func:`scene_report`.
#:
#: ``min_distance`` is how far from the trace the noise floor is measured: near
#: blocks carry real deformation gradient and bias it upward. It defaults to
#: ``4 * max_depth``, which at a 40 km fault is **160 km** -- further than this
#: crop reaches (~138 km from the trace), so leaving it unset raises "No blocks
#: survive". This value is required here, not a preference.
SCENE_REPORT = dict(min_distance=50e3)

#: Per-scene sampling parameters, keyed by ``SCENES`` name, overriding what
#: :func:`scene_report` measures. ``{}`` (the default) measures all of them,
#: which is the right choice: they are statements about the data.
#:
#: ⚠️ Numbers pinned here must have been measured on the **resampled** stacks.
#: Stage 1 puts every scene on a common 10-arcsec lattice first, and the reachable
#: quadtree cell sizes are a dyadic ladder set by the pixel size -- so a
#: ``width_min`` read off a native-resolution report lands on a different rung
#: here, and block-averaging onto the common grid lowers the noise floor that
#: ``rms_min`` is measuring. ``exclude_within`` is the one worth pinning
#: deliberately: its floor is ``width_min / 2``, and anything above that is a
#: judgement about unwrapping and near-fault model error that no measurement can
#: make for you. Overriding is silent, so the log prints which keys were pinned.
#:
#:     SAMPLING = {"D126": dict(exclude_within=4000.0)}
SAMPLING = {}

#: The coarsest quadtree cell, shared by every scene. Rarely the binding limit.
WIDTH_MAX = 30_000.0

#: ``NISAR_EDGE_LENGTH`` overrides the element size without editing this file --
#: worth having on a detached runner, where a coarse pass is how you find out
#: whether the whole chain works before committing an hour to the fine one.
MESH = dict(max_depth=float(os.environ.get("NISAR_MAX_DEPTH", 40e3)),
            edge_length=float(os.environ.get("NISAR_EDGE_LENGTH", 5e3)))

#: Fault dip. ``None`` extrudes the trace straight down; one number dips the
#: whole fault; a list gives one dip per straight *deep segment*, in order along
#: strike. Edit the fallback below to fix it here, or set ``NISAR_DIP=75`` /
#: ``NISAR_DIP=70,80,85`` to override without touching the file.
#:
#: Dips **above 90 are meaningful**, not an error -- the fault leans the other
#: way (the reference's Myanmar configuration uses ``[75 75 70 80 85 90 100]``).
#: A vertical mesh has nowhere to put dip-slip signal except into strike-slip or
#: the residual, so this is worth setting whenever the fault is known to dip.
DIPS = _dips(os.environ.get("NISAR_DIP", None))

#: Where the deep segments come from when ``DIPS`` names more than one. ``None``
#: chops the trace into ``len(DIPS)`` equal chords, which is enough to say "the
#: western third dips 70, the rest 85". Otherwise a list of four-number segment
#: files (``x_begin y_begin x_end y_end``, metres in the local frame), one per
#: dip; ``NISAR_SEGMENTS`` takes the same list, comma- or space-separated.
SEGMENT_FILES = _paths(os.environ.get("NISAR_SEGMENTS", None))

#: A digitised *bottom* trace: the map-view line where the fault reaches
#: ``BOTTOM_DEPTH``. Set it (or ``NISAR_BOTTOM=/path/to/fault-bottom.kml``) and the
#: dip is whatever connects the two traces -- no angles at all. Mutually exclusive
#: with ``DIPS``, and read with the same ``FaultTrace.from_file`` as ``FAULT``, so
#: ``.kml`` and plain text both work.
#:
#: This is the better description whenever the bottom edge is actually known,
#: because the dip then varies continuously along strike and may **reverse**
#: where the bottom trace crosses the surface trace -- which ``DIPS`` can express
#: only with hand-tuned values straddling 90. The San Sebastian pair does exactly
#: that: the bottom edge runs ~10 km north of the trace for the western 33 km and
#: up to 7 km south for the remaining 230 km.
BOTTOM_TRACE = _path(os.environ.get("NISAR_BOTTOM", None))

#: What depth ``BOTTOM_TRACE`` sits at. ``None`` means the base of the mesh
#: (``MESH["max_depth"]``), which is the usual reading. A KML carries no usable
#: depth -- Google Earth writes every vertex at altitude 0 -- so this cannot be
#: taken from the file. Setting it *shallower* than ``max_depth`` is legitimate
#: and useful (a bottom trace digitised at a locking depth); the levels below it
#: continue the dip linearly. Deeper is refused.
BOTTOM_DEPTH = _float_or_none(os.environ.get("NISAR_BOTTOM_DEPTH", None))

#: Options that exist only on the curved constructor.
#:
#: ``bias_w`` thickens the depth levels geometrically downward, putting the fine
#: elements where surface data can actually resolve slip -- a patch at 2 km is
#: resolved far more sharply than one at 18 km, so even levels spend parameters
#: where they cannot be recovered. It is orthogonal to the geometry, so it applies
#: to a vertical fault too: ``FaultMesh.vertical`` does not take it, so a config
#: asking for it routes through ``curved(uniform_dip=90)``, which is bit-identical.
#:
#: ``down_dip_levels`` is what makes ``bias_w`` mean a definite amount of grading:
#: there are ``down_dip_levels - 1`` intervals with thicknesses
#: ``bias_w ** (0 .. down_dip_levels - 2)``, so
#:
#:     deepest / shallowest = bias_w ** (down_dip_levels - 2)
#:
#: ``5 ** (1/15)`` at 17 levels is therefore exactly 5x -- 0.99 km levels at the
#: surface to 4.96 km at 40 km depth. Left as ``None`` the count comes from
#: ``edge_length`` instead and the ratio is not the one the exponent was chosen
#: for, so the two must be set together.
#:
#: ⚠️ At 5 km along strike those top levels are 5:1 elements. ``basis="node"``
#: below is what makes that tolerable: its Laplace-Beltrami smoother is
#: cotangent-weighted by the actual triangle geometry, where element-basis
#: ``neighbor_smoothing`` weights every edge equally. Revisit the grading before
#: switching basis.
#:
#: ``smoothness`` is the surface fit's regularizer weight (the reference's 0.008).
#: Only two depths are constrained -- the trace and the segments' bottom lines --
#: so the regularizer *is* the dip profile between them, not a cosmetic knob.
#: ``None`` takes the reference default. Ignored on the ``uniform_dip`` paths,
#: which are closed-form and have nothing for a gridder to decide.
CURVE = dict(
    bias_w=float(os.environ.get("NISAR_BIAS_W", 5 ** (1 / 15))),
    down_dip_levels=_int_or_none(os.environ.get("NISAR_DOWN_DIP_LEVELS", 17)),
    smoothness=None,
)

#: The ``CURVE`` values ``FaultMesh.vertical`` already implements, so a config
#: using only these can stay on the frozen constructor. Anything else has to go
#: through ``curved(uniform_dip=90)`` -- which is bit-identical -- or it would be
#: silently dropped, since ``vertical()`` takes none of these keywords.
_CURVE_NOOP = {"bias_w": 1.0, "down_dip_levels": None, "smoothness": None}

#: Depth-dependent rigidity, from Crust2.0 for Venezuela. Passing it matters:
#: `moment_magnitude` falls back to a flat 30 GPa, and this crust runs 34-46 GPa
#: through the seismogenic zone, so Mw would come out too small.
VELOCITY_MODEL = VelocityModel(
    depth=[0, 2e3, 2e3, 10.58e3, 10.58e3, 19.25e3, 19.25e3, 27.92e3, 27.92e3, 60e3],
    vp=[3.75e3] * 2 + [6.10e3] * 2 + [6.50e3] * 2 + [7.00e3] * 2 + [8.20e3] * 2,
    vs=[1.95e3] * 2 + [3.50e3] * 2 + [3.65e3] * 2 + [3.90e3] * 2 + [4.70e3] * 2,
    rho=[2.37e3] * 2 + [2.75e3] * 2 + [2.87e3] * 2 + [3.01e3] * 2 + [3.40e3] * 2,
    name="Venezuela_crust2.0",
)

#: ``"layered"`` looks each element's point sources up in EDGRN tables built from
#: ``VELOCITY_MODEL``; ``"halfspace"`` uses the homogeneous engine.
#:
#: ``NISAR_ENGINE=halfspace`` checks that the whole chain runs without waiting on
#: the Fortran or paying the layered assembly cost. A half-space gives the whole
#: crust one rigidity, which biases shallow slip low and deep slip high, so it is
#: a smoke test rather than an answer -- **except in stage 1**, where the model
#: only chooses quadtree cells and the observations it writes carry no trace of
#: which engine picked them. Sampling with the half-space and inverting layered is
#: the same reasoning as Wang & Fialko's "preliminary model", and it is much faster.
ENGINE_KIND = os.environ.get("NISAR_ENGINE", "layered").lower()

#: Quadrature accuracy for the layered engine. The order is chosen per element
#: *and* per observation, which measured 52x fewer source-receiver evaluations at
#: this tolerance than the reference's fixed 91 points; ``None`` restores the
#: fixed rule.
EDGRN_TOLERANCE = 3e-3

#: ``ramp`` gives each *named* track its own nuisance terms. Without it an
#: interferogram's arbitrary constant lands in the slip as broad, deep, fictitious
#: patches -- and with two scenes merged into one raster it cost a factor of 3.5
#: in peak slip while variance reduction stayed at 98%.
#:
#: ``"offset"`` solves a constant per track; ``"linear"`` adds x/y gradients,
#: which is what absorbs an orbital or ionospheric ramp. With six tracks that is
#: 18 nuisance columns against 6 -- worth running both and comparing, since for a
#: long east-west strike-slip fault the far-field arctangent step and a gradient
#: perpendicular to strike genuinely compete, so ``"linear"`` can eat real signal.
#:
#: ``basis="node"`` solves for slip at the mesh nodes with a continuous
#: piecewise-linear field between them: fewer parameters than triangles, and the
#: smoothing operator switches to Laplace-Beltrami automatically.
#:
#: ``velocity_model`` belongs here rather than only at the ``moment()`` call:
#: :attr:`SlipModel.moment_magnitude` reads it off the *inversion*, so without it
#: the reported Mw silently falls back to a flat 30 GPa while an explicitly
#: computed M0 uses the real rigidity -- two numbers in one summary that do not
#: describe the same Earth.
#:
#: The engine is **not** listed here: :func:`inversion_kwargs` attaches it, so the
#: tables are built (or read from cache) once per process rather than at import.
INVERSION = dict(ramp="offset", basis="node", velocity_model=VELOCITY_MODEL)

#: Everything :meth:`SlipInversion.solve` takes except the weight, shared by the
#: L-curve sweep and the final solve so the two cannot disagree.
#:
#: Right-lateral faults (San Sebastian) need strike-slip pinned non-positive,
#: because positive strike-slip is LEFT-lateral here.
#:
#: ``max_iter`` is well below the 400 default. That is a deliberate trade on a
#: wide sweep -- its cost is dominated by the few weights that run to the cap, and
#: their statistics are meaningless anyway -- but it means "did not converge" is
#: expected at the rough end rather than exceptional. Stage 2 lists which weights
#: hit it; if the corner itself is capped, raise this before reading the curve.
BOUNDS = dict(polarity=(-1, 0, 0), strike=(-6.0, 6.0), dip=(-2.0, 2.0),
              max_iter=60)

#: The weight `run_inversion.py` solves at. Read it off `run_lcurve.py`'s corner.
#: ``NISAR_SMOOTHING`` overrides it, so stage 3 can be re-run at a new weight
#: without editing this file -- which is the common case after looking at the curve.
#:
#: ⚠️ Both failure modes are invisible in variance reduction: too much smoothing
#: and the model goes flat while VR still reads a plausible 22%; too little and
#: the strike bound saturates, so the *bound*, not the data, sets the answer.
#: Stage 2 flags both. `solve()` normalises the operator by its own row count, so
#: a weight stays roughly invariant under mesh refinement -- but **not** across a
#: change of basis or engine, so re-read the curve after either. The value below
#: is a placeholder in the middle of the sweep, not a measurement.
SMOOTHING = float(os.environ.get("NISAR_SMOOTHING", 200.0))

#: Weights `run_lcurve.py` sweeps -- wide enough to show both failure modes, so
#: the corner is visibly a corner and not just the end of the range. Swept
#: large -> small internally; cost is dominated by the roughest end.
LCURVE_WEIGHTS = [10000.0, 5000.0, 2000.0, 1000.0, 500.0, 200.0,
                  100.0, 50.0, 20.0, 10.0, 5.0, 2.0]

#: `iterate_sampling`: round 0 data-driven and coarse, the rest model-driven.
#:
#: ``NISAR_MAX_ROUNDS=0`` stops after round 0, so stage 1 writes the **coarse,
#: data-driven** sampling. That is the way to L-curve *before* letting a model
#: steer the sampling -- but read what it gives you with care: round 0 is
#: deliberately under-sampled, and where there are fewer observations than
#: parameters the smoothing is supplying missing rank rather than trading misfit
#: against roughness, so its corner sits at more smoothing than the final,
#: well-determined problem wants.
LOOP = dict(max_rounds=int(os.environ.get("NISAR_MAX_ROUNDS", 4)),
            spacing=2000.0, tol=0.01)


# -- shared helpers ----------------------------------------------------------
def workspace(create=True):
    return Workspace(WORK_DIR, create=create)


def geometry():
    """``(trace, frame, mesh)`` -- rebuilt identically by every stage.

    One :class:`LocalFrame` for the mesh and every track. Mixing frames is a
    silent kilometre-scale error rather than a crash, which is why every stage
    derives the frame from the same trace instead of storing one.
    """
    trace = FaultTrace.from_file(FAULT)
    frame = trace.local_frame()
    return trace, frame, fault_mesh(trace, frame)


def fault_mesh(trace, frame):
    """The mesh described by ``MESH`` / ``DIPS`` / ``SEGMENT_FILES`` / ``CURVE``.

    Vertical stays on :meth:`FaultMesh.vertical` rather than the equivalent
    ``curved(uniform_dip=90)`` so that a run configured as it was before this
    option existed goes down exactly the same code path -- but only while
    ``CURVE`` asks for nothing, since ``vertical()`` accepts none of its
    keywords and would drop them without a word.
    """
    if BOTTOM_TRACE is not None:
        if DIPS is not None:
            # `curved` would refuse this too, but naming the *settings* is what
            # makes it fixable from the log of a detached run.
            raise ValueError(
                f"BOTTOM_TRACE ({BOTTOM_TRACE.name}) and DIPS ({DIPS}) both set -- "
                "a bottom trace already says where the fault goes at depth, so a "
                "dip would be a second, contradicting answer. Unset NISAR_DIP to "
                "use the bottom trace, or NISAR_BOTTOM to use the dips."
            )
        return FaultMesh.curved(trace, frame,
                                bottom_trace=FaultTrace.from_file(BOTTOM_TRACE),
                                bottom_depth=BOTTOM_DEPTH, **MESH, **CURVE)

    if DIPS is None:
        if all(CURVE.get(k) == v for k, v in _CURVE_NOOP.items()) \
                and not set(CURVE) - set(_CURVE_NOOP):
            return FaultMesh.vertical(trace, frame, **MESH)
        return FaultMesh.curved(trace, frame, uniform_dip=90.0, **MESH, **CURVE)

    if len(DIPS) == 1:
        return FaultMesh.curved(trace, frame, uniform_dip=DIPS[0], **MESH, **CURVE)

    segments = (FaultSegment.from_files(SEGMENT_FILES) if SEGMENT_FILES
                else FaultSegment.from_trace(trace, frame, len(DIPS)))
    if len(segments) != len(DIPS):
        # `from_segments` broadcasts, so a mismatch would otherwise surface as a
        # numpy shape error naming neither of the two settings that disagree.
        raise ValueError(
            f"{len(DIPS)} dips {DIPS} against {len(segments)} segments -- "
            "DIPS and SEGMENT_FILES must have the same length (or leave "
            "SEGMENT_FILES=None to chop the trace into len(DIPS) equal chords)"
        )
    return FaultMesh.curved(trace, frame, segments=segments, dips=DIPS,
                            **MESH, **CURVE)


def mesh_summary(mesh):
    """A JSON-safe description of the geometry, for the log and ``summary.json``.

    Read off the *mesh* rather than off ``DIPS``, because the two can legitimately
    disagree: ``curved(uniform_dip=90)`` reports itself as vertical. Recording what
    ran is the point -- ``MESH`` alone would describe a dipping fault as though it
    were the vertical one.
    """
    out = {"kind": mesh.attrs.get("kind", "vertical"),
           "n_elements": int(mesh.n_elements),
           "n_nodes": int(mesh.n_nodes),
           "n_down": int(mesh.attrs.get("n_down", 0))}
    out.update({k: float(v) for k, v in MESH.items()})
    for key in ("dip_deg", "bias_w", "smoothness", "min_curvature_radius",
                "bottom_depth"):
        if key in mesh.attrs:
            out[key] = float(mesh.attrs[key])
    if "dips" in mesh.attrs:
        out["dips"] = [float(d) for d in mesh.attrs["dips"]]
    if "segments" in mesh.attrs:
        out["segments"] = [[float(v) for v in seg] for seg in mesh.attrs["segments"]]
    if "bottom_trace" in mesh.attrs:
        # The dip range and the flip flag are the whole point of recording this:
        # a bottom trace crossing the surface trace reverses which way the fault
        # leans, and nothing else in a run's output would ever say so.
        out["bottom_trace"] = str(mesh.attrs["bottom_trace"])
        out["bottom_samples"] = int(mesh.attrs["bottom_samples"])
        out["bottom_trimmed"] = int(mesh.attrs["bottom_trimmed"])
        out["bottom_dip_flips"] = bool(mesh.attrs["bottom_dip_flips"])
        out["bottom_cross_range"] = [float(v) for v in mesh.attrs["bottom_cross_range"]]
        out["dip_range_deg"] = [float(v) for v in mesh.attrs["dip_range_deg"]]
    return out


#: Built once per process by :func:`inversion_kwargs`; EDGRN tables are megabytes
#: and every stage needs the same ones.
_INVERSION_CACHE = None


def engine():
    """The forward engine named by ``ENGINE_KIND``.

    The EDGRN tables are cached on disk, so the Fortran runs once and the later
    stages read ``EDGRN_DIR/edgrn.inp`` back instead of regenerating identical
    tables. ``None`` is the half-space default, which :class:`SlipInversion`
    interprets as :class:`HalfSpaceTDE`.
    """
    if ENGINE_KIND in ("halfspace", "half_space", "tde"):
        print("    engine: HOMOGENEOUS HALF-SPACE (NISAR_ENGINE=halfspace) -- "
              "fine for stage 1, a smoke test for stages 2 and 3", flush=True)
        return None
    if ENGINE_KIND != "layered":
        raise ValueError(
            f"NISAR_ENGINE={ENGINE_KIND!r}; use 'layered' or 'halfspace'")

    inp = EDGRN_DIR / "edgrn.inp"
    if inp.exists():
        try:
            tables = EdgrnTables.from_input_file(inp)
            print(f"    engine: layered, cached tables from {inp}", flush=True)
            return LayeredPointSource(tables, tolerance=EDGRN_TOLERANCE)
        except Exception as exc:                                   # noqa: BLE001
            print(f"    cached tables at {inp} unusable ({exc}); regenerating",
                  flush=True)
    print(f"    engine: layered, running EDGRN into {EDGRN_DIR}", flush=True)
    tables = run_edgrn(VELOCITY_MODEL, EDGRN_DIR)
    return LayeredPointSource(tables, tolerance=EDGRN_TOLERANCE)


def inversion_kwargs():
    """``INVERSION`` with the engine attached.

    Every stage builds its ``SlipInversion`` through this rather than splatting
    ``INVERSION`` directly: the engine is the one setting that costs something to
    construct, and leaving it out of the dict is what would silently invert a
    layered configuration with half-space physics.
    """
    global _INVERSION_CACHE
    if _INVERSION_CACHE is None:
        _INVERSION_CACHE = dict(INVERSION, engine=engine())
    return _INVERSION_CACHE


def load_scene(spec, ws):
    """One scene, from a ``.grd`` quadruple or a persisted ``los`` stage."""
    if "stage" in spec:
        stack = LOSStack.from_zarr(ws.path(spec["stage"]))
    else:
        directory = Path(spec["grd"]).expanduser()
        stack = LOSStack.from_grd(
            directory / "los_cm.grd", directory / "look_e.grd",
            directory / "look_n.grd", directory / "look_u.grd",
            units=spec.get("units", "m"), sign=spec.get("sign", 1),
            direction=spec.get("direction"),
            look_direction=spec.get("look_direction"),
        )
    box = spec.get("crop", CROP)
    return stack.crop(*box) if box else stack


def sampling_parameters(gridded, trace, frame, mesh):
    """``{name: from_los kwargs}`` per scene: measured, then overridden.

    Measured rather than inherited: ``rms_min`` is a noise level, and set below it
    the quadtree cannot stop splitting and just runs down to ``width_min``.
    ``SAMPLING`` pins values on top, and this prints which -- an override is
    otherwise indistinguishable from a measurement in the log.

    Takes the *resampled* stacks, since that is the grid the quadtree will run on
    and the reachable cell sizes are a dyadic ladder set by the pixel size.
    """
    sampling = {}
    for name, stack in gridded.items():
        report = scene_report(stack, trace, frame, mesh=mesh, **SCENE_REPORT)
        measured = dict(
            rms_min=report.attrs["rms_min"], width_min=report.attrs["width_min"],
            width_max=WIDTH_MAX, exclude_within=report.attrs["exclude_within"],
        )
        pinned = dict(SAMPLING.get(name, {}))
        sampling[name] = {**measured, **pinned}

        print(f"{name}: noise {1e3 * report.attrs['noise_floor']:.1f} mm, "
              f"two-sided {100 * report.attrs['two_sided_fraction']:.0f}%, "
              f"geometry_ok={report.attrs['geometry_consistent']}", flush=True)
        print(f"      measured {measured}", flush=True)
        if pinned:
            print(f"      PINNED   {pinned}  (from SAMPLING -- the measured line "
                  "above is what this resampled scene actually says)", flush=True)
        save_figure(plot_coverage(report, name=name)[0], f"coverage_{name}")
    return sampling


def load_observations(ws, frame):
    """The observations `run_sampling.py` wrote, checked against this frame.

    The check is the point: an ``Observations`` built in a different frame will
    invert perfectly happily and put the slip in the wrong place.
    """
    from nisar_tools.slip import Observations

    obs = Observations.from_zarr(ws.path(OBS_STAGE))
    frame.require_match(obs.ds.attrs["frame"], "The stored observations")
    return obs


def sampling_kind(obs):
    """``"model"`` or ``"bootstrap"``, read off the observations themselves.

    An ``Observations`` records ``quadtree["field"] = "model"`` when its cells were
    chosen from a predicted field; a coarse round-0 set has no such key. Deriving
    the label from that rather than from a flag the caller passes is the point --
    the two samplings produce L-curves that mean different things, and a mislabelled
    one is indistinguishable from a correct one six months later.
    """
    return "model" if obs.ds.attrs.get("quadtree", {}).get("field") == "model" \
        else "bootstrap"


def save_figure(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}", flush=True)
    return path


def banner(stage):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== {stage}\n    work dir {WORK_DIR}\n    fault    {FAULT}", flush=True)
