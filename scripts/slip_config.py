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
    NISAR_SMOOTHING     the weight stage 3 solves at
    NISAR_MAX_ROUNDS    sampling rounds; 0 stops at the coarse data-driven set
"""

import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # headless: before pyplot is imported anywhere

import matplotlib.pyplot as plt  # noqa: E402

from nisar_tools import LOSStack, Workspace                                    # noqa: E402
from nisar_tools.slip import (                                                 # noqa: E402
    FaultMesh, FaultSegment, FaultTrace, VelocityModel,
)


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


# -- where -------------------------------------------------------------------
WORK_DIR = Path(os.environ.get("NISAR_WORK_DIR", "workdir")).expanduser()
OUT_DIR = WORK_DIR / "model_sampling"
FIG_DIR = OUT_DIR / "figures"
FAULT = Path(os.environ.get("NISAR_FAULT",
                            "~/Downloads/Venezuela_fault_trace.kml")).expanduser()

#: Stage names inside the workspace, so the three scripts agree on them.
OBS_STAGE = "slip_observations"
LOS_STAGE = "los_{name}_frame"

# -- what --------------------------------------------------------------------
#: One entry per scene: a ``.grd`` directory, or a persisted ``los`` stage name.
SCENES = {
    "D134": {
        "grd": "~/Downloads/D134",
        "units": "cm", "sign": +1,
        "direction": "descending", "look_direction": "right",
    },
    # "D126": {"stage": "los_D126"},
}

#: ``NISAR_EDGE_LENGTH`` overrides the element size without editing this file --
#: worth having on a detached runner, where a coarse pass is how you find out
#: whether the whole chain works before committing an hour to the fine one.
MESH = dict(max_depth=float(os.environ.get("NISAR_MAX_DEPTH", 20e3)),
            edge_length=float(os.environ.get("NISAR_EDGE_LENGTH", 3e3)))

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

#: Options that exist only on the curved constructor.
#:
#: ``bias_w`` thickens the depth levels geometrically downward, putting the fine
#: elements where surface data can actually resolve slip -- a patch at 2 km is
#: resolved far more sharply than one at 18 km, so even levels spend parameters
#: where they cannot be recovered. Measured on the real trace, 8 levels over
#: 20 km: 1.15 runs 1.8 km levels at the surface to 4.2 km at the base, 1.3 runs
#: 1.1 km to 5.5 km. It is orthogonal to the geometry, so it applies to a
#: vertical fault too -- ``FaultMesh.vertical`` does not take it, so ``DIPS=None``
#: with ``bias_w != 1`` routes through ``curved(uniform_dip=90)``, which is
#: bit-identical to ``vertical()``. Note ``neighbor_smoothing`` weights every
#: edge equally, so graded levels make the smoother anisotropic with depth;
#: ``ds_ratio`` is the knob if that matters.
#:
#: ``smoothness`` is the surface fit's regularizer weight (the reference's
#: 0.008). Only two depths are constrained -- the trace and the segments' bottom
#: lines -- so the regularizer *is* the dip profile between them, not a cosmetic
#: knob. ``None`` takes the reference default.
CURVE = dict(bias_w=float(os.environ.get("NISAR_BIAS_W", 1.0)), smoothness=None)

#: ``ramp`` gives each *named* track its own nuisance terms. Without it an
#: interferogram's arbitrary constant lands in the slip as broad, deep, fictitious
#: patches -- and with two scenes merged into one raster it cost a factor of 3.5
#: in peak slip while variance reduction stayed at 98%.
#:
#: ``velocity_model`` belongs here rather than only at the ``moment()`` call:
#: :attr:`SlipModel.moment_magnitude` reads it off the *inversion*, so without it
#: the reported Mw silently falls back to a flat 30 GPa while an explicitly
#: computed M0 uses the real rigidity -- two numbers in one summary that do not
#: describe the same Earth.
INVERSION = dict(ramp="offset")

#: Bounds and polarity. Right-lateral faults (San Sebastian) need strike-slip
#: pinned non-positive, because positive strike-slip is LEFT-lateral here.
BOUNDS = dict(polarity=(-1, 0, 0), strike=(-6.0, 6.0), dip=(-2.0, 2.0))

#: The weight `run_inversion.py` solves at. Read it off `run_lcurve.py`'s corner.
#: ``NISAR_SMOOTHING`` overrides it, so stage 3 can be re-run at a new weight
#: without editing this file -- which is the common case after looking at the curve.
#:
#: Measured on the real D134 scene, 100 elements, `ramp="offset"`:
#:
#:     lambda   1000    100      10       3       1     0.3     0.1    0.03
#:     VR %     21.96  22.15   36.70   80.10   95.40   97.68   98.37   98.48
#:     max |s|  0.000  0.004   0.308   1.678   2.862   3.926   5.569   6.000
#:
#: so the corner is around **0.3-1.0**. Above ~30 the smoothing wins outright and
#: the model comes back flat zero at a plausible-looking 22% VR; below ~0.03 the
#: strike bound saturates and the *bound*, not the data, is setting the answer.
#: `solve()` normalises the operator by its own row count, so this stays roughly
#: mesh-refinement invariant.
SMOOTHING = float(os.environ.get("NISAR_SMOOTHING", 0.3))

#: Weights `run_lcurve.py` sweeps -- wide enough to show both failure modes, so
#: the corner is visibly a corner and not just the end of the range. Swept
#: large -> small internally; cost is dominated by the roughest end.
LCURVE_WEIGHTS = [30.0, 10.0, 3.0, 1.0, 0.5, 0.3, 0.1, 0.03]

#: `iterate_sampling`: round 0 data-driven and coarse, the rest model-driven.
#:
#: ``NISAR_MAX_ROUNDS=0`` stops after round 0, so stage 1 writes the **coarse,
#: data-driven** sampling. That is the way to L-curve *before* letting a model
#: steer the sampling -- but read what it gives you with care: round 0 is
#: deliberately under-sampled, and on the test mesh it produced 154 samples
#: against 240 slip parameters. A corner picked on an under-determined problem
#: sits at more smoothing than one picked on the final, well-determined one,
#: because there the regularization is supplying missing rank rather than
#: trading misfit against roughness.
LOOP = dict(max_rounds=int(os.environ.get("NISAR_MAX_ROUNDS", 4)),
            spacing=2000.0, tol=0.01)

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

INVERSION["velocity_model"] = VELOCITY_MODEL


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
    option existed goes down exactly the same code path.
    """
    if DIPS is None:
        if CURVE["bias_w"] == 1.0:
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
           "n_elements": int(mesh.n_elements)}
    out.update({k: float(v) for k, v in MESH.items()})
    for key in ("dip_deg", "bias_w", "smoothness", "min_curvature_radius"):
        if key in mesh.attrs:
            out[key] = float(mesh.attrs[key])
    if "dips" in mesh.attrs:
        out["dips"] = [float(d) for d in mesh.attrs["dips"]]
    if "segments" in mesh.attrs:
        out["segments"] = [[float(v) for v in seg] for seg in mesh.attrs["segments"]]
    return out


def load_scene(spec, ws):
    """One scene, from a ``.grd`` quadruple or a persisted ``los`` stage."""
    if "stage" in spec:
        return LOSStack.from_zarr(ws.path(spec["stage"]))
    directory = Path(spec["grd"]).expanduser()
    return LOSStack.from_grd(
        directory / "los_cm.grd", directory / "look_e.grd",
        directory / "look_n.grd", directory / "look_u.grd",
        units=spec.get("units", "m"), sign=spec.get("sign", 1),
        direction=spec.get("direction"), look_direction=spec.get("look_direction"),
    )


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
