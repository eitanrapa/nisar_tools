"""Move stage-3 outputs aside so the next solve does not overwrite them.

    python scripts/archive_run.py       # then re-run run_inversion.py

``run_inversion.py`` writes to fixed names, so re-solving at a new weight or a
different ``ramp`` replaces the previous result. Run this in between and each
solve keeps its own directory.

The tag is read off ``summary.json`` rather than passed in -- ``run_lam1000_
offset_1696el`` -- so it describes the run that actually happened rather than
what the caller meant to type. The ramp *mode* is not recorded in the summary, so
it is inferred from the ramp labels -- see ``_ramp_mode``, which tests the highest
degree first because a cubic run still carries ``:dx`` labels.

A subdirectory rather than a flat rename, because ``fit_*.png`` would otherwise
re-match files renamed on an earlier pass and quietly build up
``fit_A061_lam1000_lam500.png``. Directories cannot do that.

Only stage 3 is touched. ``bootstrap.slip.zip``, ``history.csv``, ``l_curve*``
and the mesh/samples/coverage figures describe the *sampling* and the *sweep*,
not one solve, and stay where they are.
"""

import json
import shutil
from pathlib import Path

from slip_config import OUT_DIR

#: Everything `run_inversion.py` writes, and nothing else.
STAGE_E_FILES = ("slip_model.slip.zip", "slip_model.txt", "summary.json",
                 "vert_nodes.txt", "vert_elements.txt",
                 "slip_model_UNCONVERGED.slip.zip")
STAGE_E_FIGS = ("slip.png", "slip_strike.png", "fit_*.png")


def _ramp_mode(ramp):
    """Name the ramp mode from the coefficient labels ``summary.json`` records.

    Highest degree first. A label's suffix does not identify its degree on its own
    -- ``ramp_columns`` writes ``:dx``, ``:dxx``, ``:dxxx`` -- so testing for the
    linear suffixes before the higher ones would match a cubic run's ``:dx`` and
    file it under the wrong name.
    """
    keys = list(ramp)
    for name, suffixes in (
        ("cubic", (":dxxx", ":dxxy", ":dxyy", ":dyyy")),
        ("quadratic", (":dxx", ":dxy", ":dyy")),
        ("linear", (":dx", ":dy")),
    ):
        if any(k.endswith(suffixes) for k in keys):
            return name
    return "offset"


def main():
    summary = OUT_DIR / "summary.json"
    if not summary.exists():
        raise SystemExit(f"No {summary} -- nothing from stage 3 to archive.")
    s = json.loads(summary.read_text())
    ramp = _ramp_mode(s["ramp"])
    tag = f"lam{s['smoothing']:g}_{ramp}_{s['mesh']['n_elements']}el"

    # Never clobber an earlier archive: two runs can share a tag when only
    # something the tag does not capture changed (bounds, velocity model).
    dest, n = OUT_DIR / f"run_{tag}", 2
    while dest.exists():
        dest, n = OUT_DIR / f"run_{tag}_v{n}", n + 1
    (dest / "figures").mkdir(parents=True)

    moved = []
    for name in STAGE_E_FILES:
        path = OUT_DIR / name
        if path.exists():
            shutil.move(str(path), str(dest / name))
            moved.append(name)
    for pattern in STAGE_E_FIGS:
        for path in sorted((OUT_DIR / "figures").glob(pattern)):
            shutil.move(str(path), str(dest / "figures" / path.name))
            moved.append(f"figures/{path.name}")

    print(f"archived {len(moved)} files -> {dest}")
    print(f"  VR {s['variance_reduction']:.2f}%  Mw {s['moment_magnitude']:.2f}  "
          f"peak {s['peak_strike_slip_m']:+.2f} m  lambda {s['smoothing']:g}  "
          f"ramp {ramp}")
    for name in moved:
        print("   ", name)


if __name__ == "__main__":
    main()
