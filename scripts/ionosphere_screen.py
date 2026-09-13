#!/usr/bin/env python
"""Dual-frequency (split-spectrum) ionospheric screen, one track at a time.

The screen is a *measurement*, not a fit: the ionosphere is dispersive and
everything else is not, so NISAR's two sub-bands separate them.  That
measurement needs both bands **unwrapped** -- the combination at NISAR's
carriers is ``12.12*phi_A - 11.61*phi_B``, so an unknown 2*pi in either input
moves the screen by ~76 rad, and there is no wrapped-domain shortcut.

So this runs your band-A recipe twice, once per sub-band, and combines the two.
Band B's parameters are derived from band A's by **ground distance**, never
copied as pixel counts: band B is posted ~8x coarser in x, so the same number
means a different thing on its grid.

What is deliberately NOT done here
----------------------------------
``deramp``.  Two reasons, both measured: the screen is ~94% a plane on real
data, so deramping first throws away most of what you are trying to measure;
and a deramped band A against an undramped band B gives a screen with std
15.07 rad, because the ~12x coefficients amplify the mismatch.  Deramp *after*
the correction, or let the inversion's own ramp columns absorb it.

Usage
-----
    python scripts/ionosphere_screen.py D054
    python scripts/ionosphere_screen.py D054 --dry-run   # print the plan only
    python scripts/ionosphere_screen.py --all

Stores already present with matching parameters are reused (``Workspace.store``
returns the existing stage on a hash match), so pointing ``names`` at your
existing band-A stores makes the band-A half free.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from nisar_tools import GSLC, GSLCStack, Workspace, ionosphere
from nisar_tools.interferogram import InterferogramStack
from nisar_tools.unwrap import UnwrappedStack

# Interferometric phase per TECU of differential TEC, at band A's carrier.
# Two-way range change from the ionosphere is dr = -40.31 * TEC / f**2, and
# phi = -(4*pi*f/c) * dr, so phi = 4*pi*40.31/(c*f) * TEC.  ~13.6 rad/TECU at
# 1.239 GHz -- a sanity yardstick: a real screen here ran 0.3-0.5 TECU.
K_IONO = 40.31  # m**3 s**-2
C_LIGHT = 299_792_458.0



# ---------------------------------------------------------------------------
# Finding granules
# ---------------------------------------------------------------------------
# A NISAR L2 filename is underscore-delimited and positional:
#
#   NISAR_L2_PR_GSLC_023_054_D_083_4005_DHDH_A_20260617T230235_...._001.h5
#     0    1  2   3   4   5  6  7    8    9  10       11
#                    ^cycle ^track ^dir ^frame        ^start
#
# The TRACK sits before the direction and the FRAME after it -- the pair
# ``054_D_083`` is track 54 descending, frame 83.  Confirmed by the clock: frame
# 083 runs 230235-230310 and frame 084 runs 230309-230343, i.e. consecutive
# along-track slices of one pass.
#
# Keying on token 3 is what keeps a GUNW out, and it is not paranoia: a GUNW
# sits in the same tree, carries the same ``_D_`` fields in its name, and has an
# *extra* field after the frame, so every later position shifts.  A glob like
# ``*D_054*.h5`` would match it and then fail somewhere far less legible.


@dataclass(frozen=True)
class GranuleName:
    """The fields of a NISAR L2 filename that identify a stack."""

    path: Path
    product: str
    direction: str
    track: str
    frame: str
    start: str

    @classmethod
    def parse(cls, path):
        """Parse a filename, or return ``None`` if it is not a NISAR L2 product."""
        parts = Path(path).stem.split("_")
        if len(parts) < 12 or parts[0] != "NISAR":
            return None
        return cls(Path(path), parts[3], parts[6], parts[5], parts[7], parts[11])


def find_granules(track_id: str, root, frame=None, product: str = "GSLC"):
    """Every ``product`` granule of one track under ``root``, grouped by frame.

    Returns ``{frame: [paths in acquisition order]}``.  A track is often more
    than one frame -- consecutive along-track slices of the same pass -- and
    each is its own grid, so they are kept apart here and merged as stacks
    later.  ``frame=`` narrows to one; pass a list for a subset.

    ``track_id`` is the spelling you use for the track -- ``"D054"`` is
    direction ``D``, track ``054``.
    """
    root = Path(root).expanduser()
    if not root.is_dir():
        raise SystemExit(f"Granule root {root} does not exist.")

    direction = track_id[0].upper()
    number = track_id[1:].lstrip("_").zfill(3)
    wanted = None if frame is None else {
        str(f).zfill(3) for f in ([frame] if isinstance(frame, (str, int)) else frame)
    }

    found = {}
    for path in root.rglob("*.h5"):
        g = GranuleName.parse(path)
        if g is None or g.product != product:
            continue
        if g.direction != direction or g.track.zfill(3) != number:
            continue
        if wanted is not None and g.frame.zfill(3) not in wanted:
            continue
        found.setdefault(g.frame.zfill(3), []).append(g)

    if not found:
        raise SystemExit(
            f"No {product} granules for track {track_id} under {root}. "
            f"Searched recursively for NISAR_*_{product}_*_{number}_{direction}_*.h5 "
            "-- set granules= explicitly on the Track, or point granule_root= "
            "somewhere else."
        )

    out = {}
    for name in sorted(found):
        granules = sorted(found[name], key=lambda g: g.start)
        if len(granules) < 2:
            raise SystemExit(
                f"Track {track_id} frame {name}: found only "
                f"{granules[0].path.name}. An interferogram needs two dates."
            )
        out[name] = [g.path for g in granules]
    return out


@dataclass
class Track:
    """One track's configuration: your band-A recipe, plus where the data is."""

    name: str
    # Granules: {frame: [paths]} for a multi-frame track, a flat list for a
    # single frame, or None to discover them by track id under granule_root.
    # The track id is this Track's own name -- "D054" is direction D, track 054
    # -- so nothing has to be spelled twice.
    granules: object = None
    granule_root: str = "~/Downloads"
    frame: object = None                 # narrow discovery to one frame or a few
    bbox: tuple | None = None            # (lon_min, lon_max, lat_min, lat_max)

    # -- band A: exactly your settings -----------------------------------
    looks: int = 10
    pairs: str = "sequential"
    convolution: str = "Gaussian"
    water_resolution: str = "f"
    water_spacing: str | None = None     # GMT -I increment; None -> derived
    goldstein_alpha: object = "adaptive"
    edge_pixels: int = 300               # multilooked px of band A
    edges: object = "left"
    outlier_scale: float = 16.0          # px of band A (the method default)
    outlier_threshold: float = 3.0       # radians
    nproc: int = 20

    # -- band B: None = derived from band A by ground distance ------------
    looks_b: int | None = None
    outlier_threshold_b: float | None = None

    # -- the screen -------------------------------------------------------
    smooth_metres: float | None = None   # None -> ionosphere.DEFAULT_SMOOTH_METRES
    min_coherence: float | None = 0.3

    # -- store names; point band A's at your existing stores to reuse them -
    names: dict = field(default_factory=dict)

    # Already-persisted SLC stacks, per band.  A GSLCStack holds ONE sub-band
    # (from_gslcs raises on a mixed stack), so a saved slc_stack is band A only
    # and band B has to be built from the granules the first time.
    slc_stores: dict = field(default_factory=dict)

    def store(self, stage: str, band: str) -> str:
        """Store name for ``stage`` of sub-band ``band``."""
        return self.names.get((stage, band), f"{stage}_{self.name}_{band}")

    def granule_frames(self):
        """``{frame: [paths]}`` for this track, explicit or discovered (cached).

        A flat ``granules`` list is taken to be a single frame, so the
        one-frame case needs no dict.
        """
        cached = getattr(self, "_frames", None)
        if cached is None:
            if isinstance(self.granules, dict):
                cached = {str(k): [Path(p).expanduser() for p in v]
                          for k, v in sorted(self.granules.items())}
            elif self.granules:
                cached = {"single": [Path(p).expanduser() for p in self.granules]}
            else:
                cached = find_granules(self.name, self.granule_root,
                                       frame=self.frame)
            object.__setattr__(self, "_frames", cached)
        return cached


# ---------------------------------------------------------------------------
# Your tracks.  The band-A names below point at the stores you already have,
# so only band B actually costs anything on a re-run.
# ---------------------------------------------------------------------------
_R = "/raid/class239/yuri/NI/Venezuela"

TRACKS = {
    # D054 is two consecutive along-track frames of one descending pass, each
    # its own grid, merged into one stack.  Band A already exists as
    # slc_stack_D054; band B is built from these granules once and persisted as
    # slc_stack_D054_B.
    "D054": Track(
        name="D054",
        granules={
            "083": [
                "/raid/class239/yuri/NI/Venezuela/D054/083/NISAR_L2_PR_GSLC_023_054_D_083_4005_DHDH_A_20260617T230235_20260617T230310_P05012_N_F_J_001/NISAR_L2_PR_GSLC_023_054_D_083_4005_DHDH_A_20260617T230235_20260617T230310_P05012_N_F_J_001.h5",
                "/raid/class239/yuri/NI/Venezuela/D054/083/NISAR_L2_PR_GSLC_024_054_D_083_4005_DHDH_A_20260629T230235_20260629T230309_P05023_N_F_J_001/NISAR_L2_PR_GSLC_024_054_D_083_4005_DHDH_A_20260629T230235_20260629T230309_P05023_N_F_J_001.h5",
            ],
            "084": [
                "/raid/class239/yuri/NI/Venezuela/D054/084/NISAR_L2_PR_GSLC_023_054_D_084_4005_DHDH_A_20260617T230309_20260617T230343_P05012_N_F_J_001/NISAR_L2_PR_GSLC_023_054_D_084_4005_DHDH_A_20260617T230309_20260617T230343_P05012_N_F_J_001.h5",
                "/raid/class239/yuri/NI/Venezuela/D054/084/NISAR_L2_PR_GSLC_024_054_D_084_4005_DHDH_A_20260629T230308_20260629T230342_P05023_N_F_J_001/NISAR_L2_PR_GSLC_024_054_D_084_4005_DHDH_A_20260629T230308_20260629T230342_P05023_N_F_J_001.h5",
            ],
        },
        # Or drop `granules` entirely and let it discover both frames:
        #   granule_root="/raid/class239/yuri/NI/Venezuela",
        bbox = (-70.34, -67.40, 7.55, 11.6),                      
        slc_stores={"A": "slc_stack_D054"},
        looks=10,
        edge_pixels=300,
        edges="left",
        outlier_threshold=3.0,
        nproc=20,
        names={
            ("igrams", "A"): "igrams_D054",
            ("igrams_filt", "A"): "igrams_filt_D054_full_mask",
            ("unwrapped", "A"): "unwrapped_D054_full_mask",
            ("clean", "A"): "unwrapped_D054_clean_full_mask_no_deramp",
        },
    ),
}


# --- the other Venezuela tracks -------------------------------------------
# No slc_stores/names are set for these, so every stage is built and persisted
# under the default <stage>_<track>_<band> names.  If you already have band-A
# products for any of them, point names={("clean", "A"): "..."} etc. at those
# stores -- otherwise this re-unwraps band A from scratch, which is the
# expensive half.
#
# !! edges="left" came from D054, which is DESCENDING.  Which side is near
#    range depends on the pass, so check the trimmed side on the ASCENDING
#    tracks (A061, A162) before trusting it -- the run prints the direction
#    next to the side for exactly this reason.

TRACKS["D126"] = Track(
    name="D126",
    granules={
        "083": [
            f"{_R}/D126/083/{_s}/{_s}.h5" for _s in [
                "NISAR_L2_PR_GSLC_022_126_D_083_4005_DHDH_A_20260610T225417_20260610T225451_P05012_N_F_J_001",
                "NISAR_L2_PR_GSLC_024_126_D_083_4005_DHDH_A_20260704T225415_20260704T225450_P05023_N_F_J_001",
            ]
        ],
        "084": [
            f"{_R}/D126/084/{_s}/{_s}.h5" for _s in [
                "NISAR_L2_PR_GSLC_022_126_D_084_4005_DHDH_A_20260610T225450_20260610T225524_P05012_N_F_J_001",
                "NISAR_L2_PR_GSLC_024_126_D_084_4005_DHDH_A_20260704T225449_20260704T225523_P05023_N_F_J_001",
            ]
        ],
    },
        slc_stores={"A": "slc_stack_D126"},
        looks=10,
        edge_pixels=350,
        edges="left",
        outlier_threshold=3.0,
        nproc=20,
        names={
            ("igrams", "A"): "igrams_D126",
            ("igrams_filt", "A"): "igrams_filt_D0126_full_mask",
            ("unwrapped", "A"): "unwrapped_D126_full_mask",
            ("clean", "A"): "unwrapped_D126_clean_full_mask_no_deramp",
        },
    )

TRACKS["A061"] = Track(
    name="A061",
    granules={
        "006": [
            f"{_R}/A061/006/{_s}/{_s}.h5" for _s in [
                "NISAR_L2_PR_GSLC_023_061_A_006_4005_DHDH_A_20260618T095803_20260618T095837_P05012_N_F_J_001",
                "NISAR_L2_PR_GSLC_024_061_A_006_4005_DHDH_A_20260630T095802_20260630T095836_P05023_N_F_J_001",
            ]
        ],
        "007": [
            f"{_R}/A061/007/{_s}/{_s}.h5" for _s in [
                "NISAR_L2_PR_GSLC_023_061_A_007_4005_DHDH_A_20260618T095836_20260618T095911_P05012_N_F_J_001",
                "NISAR_L2_PR_GSLC_024_061_A_007_4005_DHDH_A_20260630T095835_20260630T095910_P05023_N_F_J_001",
            ]
        ],
    },
        slc_stores={"A": "slc_stack_A061"},
        looks=10,
        edge_pixels=300,
        edges="right",
        outlier_threshold=3.0,
        nproc=20,
        names={
            ("igrams", "A"): "igrams_A061",
            ("igrams_filt", "A"): "igrams_filt_A061_full_mask",
            ("unwrapped", "A"): "unwrapped_A061_full_mask",
            ("clean", "A"): "unwrapped_A061_clean_full_mask_no_deramp",
        },
    )

# A162 is a single frame, so a flat list would do; the dict just keeps the
# frame number visible.
TRACKS["A162"] = Track(
    name="A162",
    granules={
        "007": [
            f"{_R}/A162/007/{_s}/{_s}.h5" for _s in [
                "NISAR_L2_PR_GSLC_022_162_A_007_4005_DHDH_A_20260613T100656_20260613T100731_P05012_N_F_J_001",
                "NISAR_L2_PR_GSLC_023_162_A_007_4005_DHDH_A_20260625T100655_20260625T100730_P05023_N_F_J_001",
            ]
        ],
    },
        slc_stores={"A": "slc_stack_A061"},
        looks=10,
        edge_pixels=100,
        edges="right",
        outlier_threshold=3.0,
        nproc=20,
        names={
            ("igrams", "A"): "igrams_A162",
            ("igrams_filt", "A"): "igrams_filt_A162_full_mask",
            ("unwrapped", "A"): "unwrapped_A162_full_mask",
            ("clean", "A"): "unwrapped_A162_clean_full_mask_no_deramp",
        },
    )

# ---------------------------------------------------------------------------
# Band-B parameters, derived by ground distance
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# The SLC stack for one band
# ---------------------------------------------------------------------------
def load_or_build_slc(track: Track, band: str, ws: Workspace,
                      persist: bool = True, reference=None):
    """Band ``band``'s ``GSLCStack``: the saved store if there is one, else built.

    A ``GSLCStack`` carries exactly one sub-band -- ``from_gslcs`` raises on a
    mixed one -- so a ``slc_stack_*`` you already have is band A, and band B has
    to be read from the granules the first time.  It is persisted afterwards, so
    that cost is paid once.

    **Frames are built separately and merged**, in frame order, because each is
    its own grid; the first frame wins where two overlap.  ``GSLCStack.merge``
    pairs acquisitions by nearest time within ten minutes, which is what makes
    consecutive frames of one pass (34 s apart here) line up.

    When ``bbox`` is not set and a ``reference`` stack is given, band B inherits
    band A's **extent**: a saved stack was already cropped when it was built, and
    reading band B over the whole frame would put the two bands on different
    ground.  The lon/lat box is edge-densified from A's native corners, so it
    covers a rotated grid rather than clipping its corners off.
    """
    from nisar_tools import geo

    name = track.slc_stores.get(band)
    if name and ws.exists(name):
        stack = GSLCStack.from_zarr(ws.path(name))
        # A store written before the dual-frequency work records no 'frequency'
        # at all -- from_gslcs only started stamping it (with center_frequency
        # and wavelength) in 2026-08.  Absent is therefore "older store", not
        # "wrong band", and the configuration is taken at its word; only a
        # recorded band that actually disagrees is an error.
        recorded = stack.ds.attrs.get("frequency")
        if recorded is not None and str(recorded) != band:
            raise SystemExit(
                f"Store '{name}' is sub-band {str(recorded)!r}, not {band!r}. A "
                "stack holds one band; point slc_stores at the right store per "
                "band."
            )
        if recorded is None:
            print(f"    band {band}: reusing {name} (records no sub-band; "
                  "trusting slc_stores, reading its posting and carrier from "
                  "the granules)")
        else:
            print(f"    band {band}: reusing {name}")
        return stack

    bbox = track.bbox
    if bbox is None and reference is not None:
        x = reference.ds["x"].values
        y = reference.ds["y"].values
        bbox = geo.native_bbox_to_lonlat(float(x.min()), float(x.max()),
                                         float(y.min()), float(y.max()),
                                         int(reference.epsg))
        print(f"    band {band}: inheriting band A's extent "
              f"({bbox[0]:.3f}..{bbox[1]:.3f} lon, {bbox[2]:.3f}..{bbox[3]:.3f} lat)")

    frames = track.granule_frames()
    opened = []
    try:
        built = []
        for frame_name, paths in frames.items():
            gslcs = [GSLC(path, frequency=band) for path in paths]
            opened.extend(gslcs)
            part = GSLCStack.from_gslcs(gslcs, bbox=bbox)
            if min(part.sizes["y"], part.sizes["x"]) < 2:
                print(f"    band {band}: frame {frame_name} is empty under this "
                      "bbox, skipped")
                continue
            print(f"    band {band}: frame {frame_name}, {len(paths)} granule(s), "
                  f"{part.sizes['y']} x {part.sizes['x']} px")
            built.append(part)

        if not built:
            raise SystemExit(
                f"Every frame of {track.name} came back empty for band {band} "
                "under this bbox."
            )

        stack = built[0]
        for part in built[1:]:            # self wins in the overlap
            stack = stack.merge(part)
        if len(built) > 1:
            print(f"    band {band}: merged {len(built)} frames -> "
                  f"{stack.sizes['y']} x {stack.sizes['x']} px")

        if persist:                       # also severs the HDF5 handles
            stack = stack.persist(ws, name or f"slc_stack_{track.name}_{band}")
        # else: left lazy and unpersisted -- a --dry-run must not write.
    finally:
        for g in opened:
            try:
                g.close()
            except Exception:
                pass
    return stack


def band_metadata(track: Track, band: str, stack):
    """``((dy, dx) native, carrier Hz)`` for one band, from the store or a granule.

    A ``slc_stack`` written before the dual-frequency work carries no
    ``center_frequency`` -- ``from_gslcs`` only started recording it (and
    ``frequency``/``wavelength``) once the ionosphere code needed it, and
    deliberately kept it out of the persist hash so old stores stayed valid.
    Rather than refuse such a store, the two numbers are read straight from a
    granule, which is their source in any case.  The carrier is then passed
    explicitly to ``correct_ionosphere``, so nothing downstream has to find it.
    """
    attrs = stack.ds.attrs
    carrier = attrs.get("center_frequency")
    spacings = (attrs.get("y_spacing"), attrs.get("x_spacing"))
    if carrier is not None and None not in spacings:
        return (abs(float(spacings[0])), abs(float(spacings[1]))), float(carrier)

    frames = track.granule_frames()
    path = next(iter(frames.values()))[0]
    g = GSLC(path, frequency=band)
    try:
        if g.center_frequency is None:
            raise SystemExit(
                f"Granule {path.name} records no centerFrequency for band "
                f"{band}. Pass the carriers explicitly (NISAR: A 1.239e9, "
                "B 1.2935e9 Hz)."
            )
        return (abs(float(g.y_spacing)), abs(float(g.x_spacing))), \
               float(g.center_frequency)
    finally:
        g.close()


def water_spacing_for(track: Track, stacks: dict, plan: dict) -> str:
    """A coastline increment that resolves the data without melting GMT.

    ``mask_water``'s default increment tracks the stack's **own** pixel, at
    twice its resolution.  That is right for an isotropic multilooked grid and
    catastrophic for band B, whose y axis stays at the GSLC's native posting
    while x is multilooked 8x coarser: the finer axis sets the increment, so a
    10 m x 80 m grid asks GMT for a 5 m shoreline.  On a Venezuela-sized region
    that is ~6e9 nodes, 25 GB -- the same "e means METRES" trap that once cost a
    639x slowdown, reached from the other direction.

    So: never finer than half the **coarsest** posting either band carries (the
    GSHHG shoreline does not resolve better than that anyway), then backed off
    by factors of two until the node count clears ``mask``'s own budget.  The
    same increment is used for both bands, because a coastline that moves
    between them is a difference the screen would have to explain.
    """
    if track.water_spacing is not None:
        return track.water_spacing

    from nisar_tools.mask import _MASK_NODE_BUDGET

    spacing = max(max(plan["A"]["multilooked"]),
                  max(plan["B"]["multilooked"])) / 2.0

    x = stacks["A"].ds["x"].values
    y = stacks["A"].ds["y"].values
    buffer_m = 2 * 0.05 * 111_000.0          # make_water_mask's own 0.05 deg
    span_x = float(x.max() - x.min()) + buffer_m
    span_y = float(y.max() - y.min()) + buffer_m
    while (span_x / spacing) * (span_y / spacing) > _MASK_NODE_BUDGET:
        spacing *= 2.0
    return f"{spacing:.0f}e"


def band_plan(track: Track, stacks: dict, workdir: Path) -> dict:
    """Resolve both bands' postings and parameters without computing anything.

    Reads each band's native posting and carrier off its stack, then converts
    every pixel-valued knob from band A's multilooked grid to band B's.  This is
    the whole reason the script exists: ``edge_pixels=300`` is 15 km on band A's
    50 m grid and 24 km on band B's 80 m one, and trimming the two bands to
    different footprints puts an *extrapolation* into the screen (the low-pass
    is a normalized convolution, so it fills across a gap rather than
    propagating NaN) with the ~12x coefficient on top of it.

    ``x_spacing``/``y_spacing`` on a stack are the **native** GSLC posting, not
    the multilooked one, which is exactly what sizing ``looks`` needs.
    """
    plan = {}
    for band in ("A", "B"):
        native, carrier = band_metadata(track, band, stacks[band])
        plan[band] = {"native": native, "carrier": carrier}

    dy_a, dx_a = plan["A"]["native"]
    dy_b, dx_b = plan["B"]["native"]

    looks_a = int(track.looks)
    ml_dx_a = dx_a * looks_a

    # Round UP to the next whole look rather than to nearest.  A coarser band B
    # costs nothing -- the screen is low-passed to kilometres regardless -- and
    # buys looks, which band B (5 MHz, the noisy one) needs to unwrap cleanly.
    # Rounding to nearest can land on looks=1, i.e. no multilooking at all.
    looks_b = track.looks_b or max(1, math.ceil(ml_dx_a / dx_b))

    ml_a = (dy_a * looks_a, dx_a * looks_a)
    ml_b = (dy_b * looks_b, dx_b * looks_b)

    # Ground distance is the invariant, so both conversions go through metres.
    edge_b = max(1, int(round(track.edge_pixels * ml_a[1] / ml_b[1])))
    scale_b = max(2.0, track.outlier_scale * ml_a[1] / ml_b[1])

    plan["A"].update(looks=looks_a, multilooked=ml_a,
                     edge_pixels=int(track.edge_pixels),
                     outlier_scale=float(track.outlier_scale),
                     outlier_threshold=float(track.outlier_threshold))
    plan["B"].update(looks=looks_b, multilooked=ml_b,
                     edge_pixels=edge_b,
                     outlier_scale=float(scale_b),
                     outlier_threshold=float(track.outlier_threshold_b
                                             or track.outlier_threshold))
    plan["direction"] = stacks["A"].ds.attrs.get("direction")
    plan["water_spacing"] = water_spacing_for(track, stacks, plan)
    plan["smooth_metres"] = (track.smooth_metres
                             or ionosphere.DEFAULT_SMOOTH_METRES)
    plan["workdir"] = workdir
    return plan


def describe(track: Track, plan: dict) -> str:
    lines = []
    for band in ("A", "B"):
        p = plan[band]
        f = p["carrier"]
        lines.append(
            f"  band {band}: carrier {f / 1e9:.4f} GHz  "
            f"native {p['native'][0]:.0f} x {p['native'][1]:.0f} m  "
            f"looks={p['looks']}  -> {p['multilooked'][0]:.0f} x "
            f"{p['multilooked'][1]:.0f} m"
        )
        lines.append(
            f"           mask_edges({p['edge_pixels']} px = "
            f"{p['edge_pixels'] * p['multilooked'][1] / 1000:.1f} km, "
            f"edges={track.edges!r})  "
            f"remove_outliers(scale={p['outlier_scale']:.0f} px, "
            f"threshold={p['outlier_threshold']:.1f} rad)"
        )
    (d_a, d_b), _ = ionosphere.split_coefficients(plan["A"]["carrier"],
                                                  plan["B"]["carrier"])
    lines.append(
        f"  screen: {d_a:+.2f}*phi_A {d_b:+.2f}*phi_B   "
        f"noise gain {math.hypot(d_a, d_b):.1f}x   "
        f"low-pass sigma {plan['smooth_metres']:.0f} m"
    )
    # Which side is near range depends on the pass, so the two are printed
    # together: edges= is grid-relative by design and cannot be got backwards,
    # but it can be pointed at the wrong side of an unfamiliar track.
    sides = track.edges if isinstance(track.edges, str) else sorted(track.edges)
    lines.append(f"  pass: {plan['direction']}   trimming {sides}   "
                 f"coastline {plan['water_spacing']}")
    coarsest = max(max(plan["A"]["multilooked"]), max(plan["B"]["multilooked"]))
    if plan["smooth_metres"] < 2.0 * coarsest:
        lines.append(
            f"  !! low-pass is under 2x the coarsest posting ({coarsest:.0f} m); "
            "the regrid will dominate the screen"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# One band, all the way to a cleaned UnwrappedStack
# ---------------------------------------------------------------------------
def configured_store(track: Track, stage: str, band: str, ws: Workspace):
    """The store for ``stage`` if YOU named it in ``names`` and it exists.

    Only an **explicitly configured** name is reused blind.  Band A is normally
    already built -- by you, with your own settings, in your own session -- and
    re-deriving it would waste the unwrap and, worse, hash-mismatch against a
    store written by a slightly different call.  A store the script wrote itself
    is left to ``persist()`` instead, which hash-checks and so catches a changed
    parameter rather than silently reusing a stale stage.

    ⚠️ The consequence: for a band loaded this way, the Track's ``looks`` /
    ``edge_pixels`` / ``outlier_*`` knobs are no longer *applied* -- they only
    **describe** what that store already contains, so band B's ground-equivalent
    values can be derived from them.  Keep them matching what you actually ran,
    or band B will be cleaned to a different footprint than band A.
    """
    name = track.names.get((stage, band))
    if name and ws.exists(name):
        return name
    return None


def unwrapped_band(track: Track, band: str, plan: dict, ws: Workspace,
                   stack):
    """Your recipe, run for one sub-band, resuming from the furthest stage built."""
    p = plan[band]

    existing = configured_store(track, "clean", band, ws)
    if existing:
        print(f"    band {band}: reusing {existing} (cleaned; skipping the chain)")
        return UnwrappedStack.from_zarr(ws.path(existing))

    existing = configured_store(track, "unwrapped", band, ws)
    if existing:
        print(f"    band {band}: reusing {existing}")
        unw = UnwrappedStack.from_zarr(ws.path(existing))
    else:
        existing = configured_store(track, "igrams_filt", band, ws)
        if existing:
            print(f"    band {band}: reusing {existing}")
            igrams = InterferogramStack.from_zarr(ws.path(existing))
        else:
            existing = configured_store(track, "igrams", band, ws)
            if existing:
                print(f"    band {band}: reusing {existing}")
                igrams = InterferogramStack.from_zarr(ws.path(existing))
            else:
                igrams = stack.form_interferograms(
                    pairs=track.pairs, looks=p["looks"], downsample=True,
                    convolution=track.convolution,
                ).persist(ws, track.store("igrams", band))

            igrams = igrams.mask_water(
                mask_cache=ws, resolution=track.water_resolution,
                spacing=plan["water_spacing"],
            ).remove_unconnected_regions()
            igrams = igrams.filter_goldstein(
                alpha=track.goldstein_alpha,
            ).persist(ws, track.store("igrams_filt", band))

        unw = igrams.unwrap(ws, name=track.store("unwrapped", band),
                            nproc=track.nproc)

    # Both of these are REJECTIONS, not rewrites -- they only set pixels to NaN
    # (_kernels.remove_outliers: np.where(residual > threshold, nan, out)), so
    # they never alter a phase value that survives.  That is what makes them
    # safe to run before the screen, unlike a deramp.
    clean = (
        unw.mask_edges(edge_pixels=p["edge_pixels"], edges=track.edges)
           .remove_outliers(scale=p["outlier_scale"],
                            threshold=p["outlier_threshold"])
    ).persist(ws, track.store("clean", band))
    return clean


# ---------------------------------------------------------------------------
# Diagnostics -- the screen's silent failure modes
# ---------------------------------------------------------------------------
def report_screen(screen, carrier: float) -> str:
    """Amplitude, planarity and TEC of the screen, per pair.

    A real screen here ran 0.3-0.5 TECU and correlated +0.94 with its own
    best-fit plane.  Far outside that -- especially a screen that is *not*
    mostly a plane -- usually means a 2*pi blunder in one band rather than an
    interesting ionosphere.
    """
    rad_per_tecu = 4.0 * math.pi * K_IONO / (C_LIGHT * carrier) * 1e16
    out = []
    values = np.asarray(screen.compute())
    for i in range(values.shape[0]):
        plane = values[i]
        finite = np.isfinite(plane)
        if finite.sum() < 100:
            out.append(f"    pair {i}: empty")
            continue
        yy, xx = np.mgrid[: plane.shape[0], : plane.shape[1]]
        design = np.column_stack([np.ones(finite.sum()), yy[finite], xx[finite]])
        coef, *_ = np.linalg.lstsq(design, plane[finite], rcond=None)
        fitted = design @ coef
        corr = np.corrcoef(fitted, plane[finite])[0, 1]
        p2p = np.ptp(plane[finite])
        out.append(
            f"    pair {i}: std {np.nanstd(plane):6.3f} rad  "
            f"p2p {p2p:6.3f} rad = {p2p / rad_per_tecu:.2f} TECU  "
            f"plane corr {corr:+.2f}  valid {100 * finite.mean():.1f}%"
        )
    return "\n".join(out)


def conncomp_agreement(a, b) -> str:
    """How much of the common footprint the two bands both unwrapped.

    Nothing in ``dual_frequency_screen`` looks at ``conncomp``: it reads ``unw``
    and optionally gates on coherence.  A 2*pi seam that SNAPHU placed
    differently in the two bands is therefore invisible to it and lands in the
    screen at ~12x -- 73 rad from a single cycle.  This is the cheap check.
    """
    va = np.isfinite(np.asarray(a.ds["unw"].isel(pair=0).compute()))
    vb_da = b.ds["unw"].isel(pair=0)
    vb = np.isfinite(np.asarray(vb_da.compute()))
    parts = [f"    band A valid {100 * va.mean():.1f}%, "
             f"band B valid {100 * vb.mean():.1f}% (on their own grids)"]
    for label, stack in (("A", a), ("B", b)):
        if "conncomp" in stack.ds:
            cc = np.asarray(stack.ds["conncomp"].isel(pair=0).compute())
            labels = np.unique(cc[cc > 0])
            parts.append(f"    band {label}: {labels.size} connected component(s)")
    parts.append("    -> more than one component in either band means the two "
                 "unwraps can disagree by 2*pi; check before trusting the screen")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
def run(track: Track, workdir: Path, grddir: Path, dry_run: bool = False):
    ws = Workspace(workdir)
    print(f"--- {track.name} -------------------------------------------")
    # Band A first: band B inherits its extent when no bbox is configured, and
    # passing the loaded stack in is what keeps A from being built twice.
    stack_a = load_or_build_slc(track, "A", ws, persist=not dry_run)
    stack_b = load_or_build_slc(track, "B", ws, persist=not dry_run,
                                reference=stack_a)
    stacks = {"A": stack_a, "B": stack_b}
    plan = band_plan(track, stacks, workdir)
    print(describe(track, plan))

    # Which configured stores exist decides how much actually runs: the first
    # one found from the end of the chain is where each band resumes.
    for band in ("A", "B"):
        for stage in ("clean", "unwrapped", "igrams_filt", "igrams"):
            name = track.names.get((stage, band))
            if name:
                mark = "found" if ws.exists(name) else "MISSING"
                print(f"  band {band} {stage:12s} {name:45s} {mark}")
    if dry_run:
        return None

    print(f"  [{track.name}] band A ...")
    unw_a = unwrapped_band(track, "A", plan, ws, stacks["A"])
    print(f"  [{track.name}] band B ...")
    unw_b = unwrapped_band(track, "B", plan, ws, stacks["B"])

    print(conncomp_agreement(unw_a, unw_b))

    # scale= is in pixels of the corrected stack, so convert the metres here.
    scale_px = plan["smooth_metres"] / plan["A"]["multilooked"][1]
    # Passed explicitly rather than left to ionosphere._carrier: it reads them
    # off the stack attrs, which a store written before the dual-frequency work
    # does not carry.  We already resolved both, so hand them over.
    corrected = unw_a.correct_ionosphere(
        unw_b, scale=scale_px, min_coherence=track.min_coherence,
        frequency=plan["A"]["carrier"], other_frequency=plan["B"]["carrier"],
    ).persist(ws, f"{track.store('clean', 'A')}_iono")

    print(report_screen(corrected.ds["phase_screen"], plan["A"]["carrier"]))

    outdir = Path(grddir) / track.name
    written = corrected.to_grd(outdir, fields=["phase_screen", "unw"])
    print(f"  wrote {len(written)} grids to {outdir}")
    return corrected


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tracks", nargs="*", default=None,
                    help="track names from TRACKS (default: --all)")
    ap.add_argument("--all", action="store_true", help="every configured track")
    ap.add_argument("--workdir", type=Path,
                    default=Path("/raid/class239/erapaport/workdir/"))
    ap.add_argument("--grddir", default="grd", type=Path)
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve and print both bands' parameters, compute nothing")
    args = ap.parse_args()

    names = list(TRACKS) if (args.all or not args.tracks) else args.tracks
    unknown = [n for n in names if n not in TRACKS]
    if unknown:
        raise SystemExit(f"Unknown track(s) {unknown}; configured: {list(TRACKS)}")

    for name in names:
        run(TRACKS[name], args.workdir, args.grddir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
