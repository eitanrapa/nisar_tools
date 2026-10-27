# Split-spectrum ionosphere near surface ruptures — paper draft

A focused paper on `nisar_tools.split_spectrum`. The NISAR PR GUNW ionosphere screens copy coseismic deformation
near ruptures. This draft shows where that comes from, and why removing the scaled, unwrapped main band from the
side band *before* anything is unwrapped keeps it out. Started 2026-10-21 (user: "let's do that, and maybe we can run
some tests on synthetic data too. Included it all in a focused paper").

**Novelty is narrow, and the draft says so.** The estimator is algebraically Wegmüller et al. (2018)'s reformulated
split spectrum, which is also ISCE3's `main_diff_ms_band`, the GUNW's. What is new is the order of operations, the
near-rupture diagnosis and the error budget. The prior-art check is in the memory file
`ionosphere-screen-prior-art.md` (2026-10-21).

## What the experiments found (2026-10-21)

- **Benign geometry, every form agrees.** A rupture that ends inside the scene, with the fault zone excluded: every
  estimator without cycle corrections equals ours, even the two-band form with the side band unwrapped on its own.
  The error is 8–9 mm of LOS at 2–5 km from the trace, the same as at zero slip, so it is the filter's.
- **ISCE3's cycle-correction passes are the systematic leak.** The PR GUNWs ran with `unwrap_correction: True` (their
  stored runconfig).
  - On synthetics they copy −0.045 to −0.048 of the high-passed deformation at every slip, −0.06 without the
    fault-zone exclusion.
  - On NISAR, run on our cells, they leak 0.009–0.048 more than ours.
  - They also move cycles at data edges, where their one-sided low-pass lags any gradient of ≥2.7 rad/km
    (`selftest.py`).
- **Unwrapping deformation fails when the rupture splits the scene.** Forms that unwrap a field carrying deformation
  (the side band, or the double difference Δ at 4.4%) fail once Δ's share of the cross-fault step exceeds ~π:
  - the reformulated form with our own unwrapper fails at 4 m of slip, with SNAPHU at 6 m;
  - the two-band form fails at every slip;
  - on D141, the 3.8 m step puts 1.38 cycles into Δ, and every reformulated arm leaves the cut-off south block one Δ
    cycle off (73 rad, 1.4 m of LOS).
- **The products' own main band** (`run_nisar.py --gunw-phase`):
  - On D141 the PR GUNW's main band is −25 cycles on the south block and +3..+9 north of the trace, relative to ours
    (tied by the side band, 34). With it, even our estimator leaks −0.061 (the product leaks −0.104).
  - On the four Venezuela tracks it agrees with ours to within 0.3% of the cells.
- **Footprint mismatch** (`--footprint`; the side band widened in range as the GUNW's): changes every leak by ≤0.004.
- **Venezuela products leak −0.021 to −0.123** (0.042–0.092 beyond ours). The recipe on our cells reproduces 25–52%
  of that; the rest is not attributed.
- **The reordered form stays at its zero-slip error in every case.** It reproduces the production v6 screens on all
  five tracks (to 4e-6 rad).
- **Smaller effects:**
  - Band co-location (ISCE3's ~9 m decimation offset) adds 0.3 mm beyond 2 km in a D141-like geometry.
  - The production 10 km plane fill gives 16 mm rms inside the 2 km fault zone on the synthetics, a 1.5 km plane
    2.0 mm; on real ionosphere they differ by about a millimetre (Section 7, below).

## The side-band tie (2026-10-21)

User, after the limitations were listed: "Yes let's do it" (the side band as the tie of a cut-off block).

- **Principle.** A block whose main-band integer is off by eps carries 3.21 eps rad of extra raw estimate, and the
  ionosphere is continuous across the gap. So the raw estimate's step across the gap reads eps.
  - Estimator: in 20 km windows along the boundary, one quadratic over both blocks plus a step on the block. Cells
    within 12 km of the other block; Cramér–Rao weights, Tukey biweight; the median over windows.
  - Library: `nisar_tools.split_spectrum.block_cycles` (three tests in `tests/test_split_spectrum.py`).
  - Kriging cross-check (Matérn 3/2, REML per window): `experiments/dispersive_tie.gp_tie`.
- **Fake boundaries** (`run_tie_null.py`, 8 scenes, 6082 windows): |e| median 0.035, p95 0.26 (0.15 without D054),
  mean +0.001.
  - Per line (the median of its windows, 565 lines): no wrong integer, max 0.47.
  - Most windows over 0.5 sit on real main-band errors: sharp patches 1–3 cycles off on coastal strips, islands, and
    the Llanos floodplain on D126 (66.2–66.6 W, 7.7–8.0 N) and A162 (68.4 W, 8.3 N).
  - On D054 they are ionospheric waves (10–20 km), which bias a window when the line runs along them.
- **Controls** (`run_tie_real.py`):
  - Venezuela A162 and D054, main band unwrapped around the rupture's west end: +0.02 / −0.04 (kriging 0.00 / +0.03).
  - D141 uncut (v5): +3.0 to +3.2 in every window, 2 cycles below the tied v6, which reads +0.6 to +1.7.
- **The Panama ties** (production cuts):

  | Pair | Side band (polynomial) | Kriging | Offsets' tie | Side band's tie |
  |---|---|---|---|---|
  | D141 v8 | +1.19 | +1.12 | 26 | **27** |
  | D040 v8 | +1.28 (flagged: one window +2.15 at the east branches) | +1.00 | 20 | **21** |
  | A076 v8 | +0.93 | +0.91 | −23 | **−22** |
  | D141 v3 (the paper's Section 5) | | | 33 | **34** |

  A004's south block has one usable window (−0.30, kriging −0.11): not confirmed.
- **Why the offsets were off is not established.**
  - m + 2 x screen cancels the integer and the ionosphere, and still steps by 0.6–1.5 cycles across the three ruptures
    (`run_tie_offsets.py`). So the offsets and the phase disagree by 7–18 cm of range there.
  - Pointing to the offsets:
    - the side band sits within 0.07–0.28 of an integer (kriging within 0.12), the offsets 0.13–0.41 from one;
    - the offsets' tie moves by up to 1.4 cycles with the cells used;
    - m departs by 1.0–1.5 cycles within 2 km of the trace;
    - used as a phase model, the offsets put D141's block 2 cycles from where their planes do.
  - Caveat: the real-rupture controls step at most 26 rad 2–4 km from the trace, against 70–208 rad in Panama.
- **Consequence** (`tie_screen_effect.csv`):
  - Adding the cycle changes the corrected LOS (positive toward the satellite), beyond each block's constant, by
    +36..+38 mm at 0–1 km north, +8..+11 mm at 6–8 km, and −18..−20 mm just south. So the production v8 LOS is that
    much low north of the trace and high south of it: a 5–6 cm fake step. (`tie_screen_effect.csv` holds the change of
    the corrected phase in mm, the opposite sign; the figure plots the LOS.)
  - The 59 mm block constant goes into the block's own offset.
  - pm2 runs on these screens.
- **Synthetic** (`run_tie_synthetic.py`, `run_tie_iono_only.py`):
  - Planted ties are exact shifts (3e-5 cycles), and slip does not matter.
  - Without local cycle errors, the polynomial reads rms 0.08–0.12 and kriging 0.03–0.06.
  - In a rough ionosphere (8 rad at 8 km), over 20 fields: polynomial rms 0.44 with 5 of 20 ties wrong; kriging rms
    0.05, none wrong. Kriging is the better estimator there; it is in the experiments, not yet in the library.
  - With 35% of one side's near-gap cells off by a cycle, both are biased +0.2..+0.4, and a few ties come out wrong.
  - Leaving out the cells within 6 km of the gap makes it worse (rms 0.27–0.44).

## Near the trace: the cells, the fill, the filter and the integers (2026-10-21)

User, after the side-band tie: "Let's move on to item 2 and 3" -- taking the scaled main band out pixel by pixel before
the cell sums, and matching the fill to the gap. Section 7 of the paper. In the library (`nisar_tools.split_spectrum`,
all opt-in, production unchanged): `gslc_cells(phase_model=)`, the `fill_m` ladder with `fill_lever`, and
`ionosphere_filter(order=1)`.

- **Synthetic** (`run_pixel.py`: the through-going rupture, 4 m, five seeds; with and without a rough near field as
  steep as D141's, 16-21 rad/km within 1 km): error within 2 km of the trace 15.6 / 19.0 mm (production) -> 4.4 / 4.6
  (fill matched to the gap) -> 2.0 / 2.6 (+ first-order filter) -> 2.0 / 2.2 (+ per pixel) -> 1.5 / 1.6 (no
  exclusion); the 2-4 km step 6.9 / 11.9 -> 0.5 / 0.2 mm; 1.3 mm beyond 20 km.
  - The production filter's weighted mean lags the ionosphere's gradient where the damage zone lowers its weights:
    raw estimate unbiased to +-2 mm, screen +7..+15 mm north / -8..-11 mm south at 0.5-5 km. A weighted local plane
    (Savitzky-Golay; Piromthong et al. 2026 fit such planes at scene edges) removes it.
  - Per pixel: cells kept within 0-0.5 / 0.5-1 / 1-2 / 2-5 km 34 / 58 / 96 / 100% against 6 / 19 / 45 / 83% per
    cell; raw error at 2-5 km 17 against 29 mm. With the Okada field alone (~3 rad/km) the two forms are the same.
    The model's own filtering does not matter (true phase vs smoothed: 0.05 mm), so the Gaussian multilook and
    Goldstein reach the screen only through the main band's integers.
- **Fills, synthetic** (`run_fill_synthetic.py`): in a 2 km zone a 1.5 km plane misses by 2.0 / 11 / 6 mm (default /
  rough / waves), the 10 km plane by 15 / 60 / 29.
- **Fills, real ionosphere** (`run_fill_null.py`: 34 windows of 100 km in the eight production screens, fake zones of
  1-4 km and one-sided strips): all fills 1.1-3.1 mm in 1-2 km zones; best a 2.6 km plane or the ladder 2.6-5-10 km,
  the production 10 km plane 0.6-1.1 mm worse; the 1.5 km ladder too narrow (3.5-7 mm in 4 km zones); at a one-sided
  edge 6-10 km planes best. The first-order filter changes the fills' error by <= 0.4 mm there.
- **D141** (`run_nisar_pixel.py`, `run_d141_integers.py`): per-pixel cells double the cells kept north of the trace
  within 5 km (5 -> 20% at 0.5-1 km, 18 -> 35% at 2-5 km). But narrower fills raise the screen near the trace by 7-16
  mm against production, and that is the main band's local integers: the cut (v3) and uncut (v5) unwrappings disagree
  on 78 / 62 / 42 / 18% of the cells at 0-1 / 1-2 / 2-4 / 4-6 km north (3% beyond). With those cells on v5's integers
  every recipe is within 5 mm of production, and production itself moves 2.4 mm or less: its 10 km fill averages the
  near-trace cells away. The first-order filter moves D141's screen by at most 3 mm.
- **So:** on real data the near-trace screen is set by the main band's local integers; the fill, filter and cells are
  millimetre corrections. Keep the production fill and exclusion until those integers can be checked (a patch-level
  side-band tie would be the next step).

## Ice: the Antarctic pairs of nisarErrors (2026-10-26)

User: "can we download and process the Antarctica scenes that they are looking at and calculate our own ionosphere to
see if it helps", then "yes, add it to the paper". The source is Joughin's nisarErrors
(github.com/fastice/nisarErrors, `Documents/carrierCorrectionFrames.md`, a draft; the paper carries an author note to
check with him before citing it). Section 8 of the paper, `sections/08_antarctica.tex`; the discussion, conclusions and
appendix moved to 09/10/11. The pipeline and its own README are in `~/Documents/data/antarctica_t094_f129/`.

- **The product's screen is ours plus integers** (both pairs, 19–31 Aug and 31 Aug–12 Sep, track 94 D, frame 129):
  - component 1: 9.013 / 8.004 main-band cycles (the main band's split, carried at 3.21 rad);
  - a 11,980 / 11,304 km² block: 1.0006 / 0.9995 cycles of the double difference (72.96 rad): the region nisarErrors found ~40 rad off, which they measured against a different reference;
  - elsewhere 0.00 (robust sd 0.07).
  - Every gap around the block carries more than π of Δ: the non-dispersive steps are 167 / 120 / 151 rad, which is
    1.17 / 0.84 / 1.06 Δ cycles. SNAPHU on our cells also slips a Δ cycle, but at component 1, not the block.
- **What is left after the integers** (5,400 / 5,078 km² above 3 rad):
  - the filter's smear of the block's step within 3 km of its edge (+17.7 / +16.4 rad);
  - a plateau over the tributary's trunk, 5–20 km outside the block: +6.50 / +5.44 rad = 2.03 / 1.70 main-band cycles.
  - ISCE3's correction passes on our cells move cycles at 21% of the cells and put +3.60 / +3.52 rad there: same place
    and sign, about 60% of the size. Elsewhere they do more than the product shows (the product's own grid again, as
    on Venezuela).
  - Away from the integer steps the product's screen regresses on the high-passed non-dispersive phase at −0.066
    (r −0.65 / −0.67); ours at +0.002 / +0.001.
- **The range offsets side with ours** where the screens differ (offsets' ionosphere − ours ±0.3 rad; − product −5.6 /
  +3.5 rad), calibrated at α 2.01 / 1.83 (theory 2).
- **The split is 9 / 8 cycles, not GrIMP's 7**: the side band reads 8.95 / 7.97 and the GUNW's range offsets 8.90 /
  7.96, window by window. Velocity maps read low (ITS_LIVE 8.0, GrIMP 7.5): the main side of the boundary is fast ice.
  The non-dispersive step across that boundary is 74–188 rad, so this is the large-step control Section 6 lacked;
  Section 6's caveat now points to it.
  - ⚠️ The GUNW's offsets are NOT raw (ISCE3's dense rubbersheet branch; the solid-earth GUNWs take the polynomial
    branch instead, see the Panama re-check). ISCE3's rubbersheet culled them (9×9 median), refilled them by smoothing, and
    boxcar-filtered them. Our own (`ant_roff.py`: `nisar_tools.roff`, 64 px windows every 150 px, a gross cull only)
    finished on 2026-10-27, and the section and Table 9 carry them (`run_antarctica.py ours`): our own offsets give 9.06 / 8.15 (spread 0.34 / 0.21) against the side band's 8.95 / 7.97 and the GUNW's 8.90 / 7.96; over the fast ice they are +0.11 / −0.20 rad from ours and −4.95 / −4.55 from the product where it is > 3 rad above, +0.26 / 0.00 and +3.31 / +2.73 where below; α 2.04 / 1.86.
- **The residual against ITS_LIVE**, slow ice, where the slope is 0.998 / 0.994 (σ / robust σ): ours 1.84 / 1.22 and
  1.84 / 1.21; uncorrected 3.50 / 3.15 and 2.09 / 1.69; the product 15.2 / 14.2 and 15.1 / 14.1; the product with its
  integers repaired 4.0 / 1.4. Not comparable with nisarErrors' numbers (GrIMP's map, all ice).
- **How good each correction is** (Section 8.6, added 2026-10-26, user: "Sure why not"; `run_antarctica.py quality` reads the Antarctic folder's `scripts/corrected_maps.py` → `work/corrected_metrics.json`):
  - Screen minus the offsets' ionosphere ((φ − kρ)/2, uncalibrated, 6 km smoothing, rms): ours 0.63 / 0.68 rad; the product 18.9 as delivered and 1.29 / 1.21 repaired.
  - Pair 1 − pair 2 (σ / robust): uncorrected 3.94 / 4.13, product 1.30 / 0.99, ours 1.02 / 0.81. ⚠️ Blind to errors that repeat in both pairs, as the product's do.
  - The table rows come from `antarctica.json["quality"]`. The floats now sit before 8.2 as [p], so no near-empty page is left.
- `ours` in `estimators.py` reproduces the pipeline's tied screen to 0.0013–0.0017 rad rms (max 0.2 at a few cells:
  the two bands' cell means of the phase model differ where their valid pixels do).

## The Panama ties, re-checked after Section 8 (2026-10-26)

User: "What do these say about the Panama ties?", then "sure" to the checks. `experiments/run_tie_panama_checks.py` →
`results/tie_panama_checks.csv`. Every number below is in cycles to add to the offsets' ties (26 / 20 / −23).

| | D141 | D040 | A076 |
|---|---|---|---|
| side band, from 2 km / from 6 km / the pipelines' planes | +1.19 / +1.05 / +1.31 | +1.28 / +1.21 / +1.44 | +0.93 / +0.95 / +1.00 |
| our registration offsets (smoothed) | +0.84 / +0.19 / −0.08 | +1.71 / +1.03 / +0.85 | +0.84 / +0.57 / +0.33 |
| our offsets, raw (gross cull only) | +0.84 / +0.40 / +0.12 | +1.06 / +1.00 / +0.67 | +0.99 / +0.64 / +0.28 |
| the GUNW's offsets (one quadratic per frame) | −35 | −19 | +37 |

- **Smoothing is not the cause.** Raw and smoothed offsets match to −0.2..−0.3 cm (median, robust sd ~9 cm).
- **The reading is.** Read like the side band from 2 km, the offsets agree with it to −0.06..+0.35 cycles. The
  pipelines' far-field planes put the ties a cycle short.
- **Plain cell sums are not the cause.** On D141's per-pixel cells (Section 7.3), with the south block at 34, the tie
  reads +0.37 (+0.11 from 6 km), against +0.19 (+0.10) on plain sums.
- **The GUNW's offsets cannot referee Panama: they are not measurements** (corrected 2026-10-27).
  - Every Venezuela and Panama GUNW (PR and UR) was made with ISCE3's polynomial rubbersheet (`polyfitting: True`,
    degree 2). It fits one quadratic per frame for each of the along-track and slant-range offsets to a 200 × 200 grid
    of the dense offsets, weighted by correlation peak, with outliers removed one at a time. The secondary is resampled
    with it, and the `pixelOffsets` layers ARE it: a cubic in map x, y fits them to ≤ 3 mm rms. The Antarctic products
    take the dense branch and leave 1.1–1.7 m. The cull, fill and filter settings in the run configuration are never
    read in the polynomial branch (ISCE3 v0.25.16 `nisar/workflows/rubbersheet.py`, `run()`).
  - ⚠️ This replaces "valid on 15–45% of the frames, correlation peaks ~0.05, rubbersheet refill near the ruptures",
    which had the cause wrong. Near the ruptures the quadratic spans 0.8–1.5 m where ours span −1.8..+2.3 m.
  - Their "ties" are the uncancelled coseismic step.
  - More than 20 km out, on coherent ground, they track the long-wavelength part of ours (r 0.82–0.90, ~8 cm).
  - Section 6's item now says so (2026-10-27, user: "Ok let's fix section 6").
- **Section 6 rewritten (2026-10-26, user: "Sure, rewrite it").** The Panama subsection now says the offsets agree with the side band when read the same way, and that the pipelines' far-field planes put the ties a cycle short. The three checks rule out the alternatives. New Table 6 (`tab_tie_readings`, from `tables.tab_tie_readings`). The abstract, introduction, discussion and conclusions now say "the pipelines' range-offset ties".

## Layout

| path | what |
|---|---|
| `main.tex`, `sections/*.tex`, `sections/tables/*.tex`, `references.bib` | the draft; every bib entry checked against Crossref on 2026-10-21 |
| `main.pdf` | the compiled draft |
| `experiments/estimators.py` | the eight estimators, all on one set of map cells (see its docstring) |
| `experiments/synth.py` | the synthetic generator: Okada rupture, ionosphere, per-pixel Wishart looks, 240 m cells |
| `experiments/metrics.py` | the scores: leak (15 km high-pass slope within 10 km), error by distance band, step |
| `experiments/selftest.py` | noise-free checks: the forms agree to 1e-13 rad; one main-band cycle = 3.21 / 76.2 rad |
| `experiments/run_synthetic.py` | the synthetic experiments → `results/synthetic.csv`, `synthetic_cycle.csv`, `synthetic_example_*.npz` |
| `experiments/run_nisar.py` | the NISAR ablation (D141 + the four Venezuela tracks) → `results/nisar_<scene>.{csv,npz}` |
| `experiments/dispersive_tie.py` | the side-band tie: the robust polynomial (`block_tie`) and kriging (`gp_tie`) |
| `experiments/panama_scenes.py`, `venezuela_scenes.py` | the production screens as the tie reads them (blocks, trace, ROFF) |
| `experiments/run_tie_null.py` | fake boundaries in 8 scenes → `results/tie_null.csv` (~1.5 h) |
| `experiments/run_tie_real.py` | the Panama ties, the controls, the screens' change → `results/tie_real*.csv`, `tie_screen_effect.csv`, `tie_profiles.npz` |
| `experiments/run_tie_offsets.py` | the offsets' tie reproduced on the cells, and both fields' departures near the trace → `results/tie_offsets.csv`, `tie_departures.csv` |
| `experiments/run_tie_synthetic.py` | planted ties and local cycle errors on synthetic ruptures → `results/tie_synthetic.csv` |
| `experiments/run_pixel.py` | the near-trace recipe on synthetics: per cell / per pixel, four exclusions, three filters → `results/pixel.csv`, `pixel_example.npz` (~35 min on 3 cores) |
| `experiments/run_fill_synthetic.py` | fills of synthetic gaps → `results/fill_synthetic.csv` (~45 min on 3 cores) |
| `experiments/run_fill_null.py` | fills of fake gaps in the eight production screens → `results/fill_null.csv` (two passes, `fills=1` and `fills=2`, ~20 min each) |
| `experiments/run_nisar_pixel.py` | D141 with the main band taken out per pixel (cells cached in `results/cells_pixel_*`, ~25 min of GSLC reading) → `results/nisar_pixel_D141.{csv,npz}` |
| `experiments/run_d141_integers.py` | D141's screens with the main band as cut and with its local cycles from the uncut unwrapping → `results/d141_integers.csv`, `d141_repair.csv` |
| `experiments/run_antarctica.py` | Section 8, the Antarctic pairs: reads the pipeline in `~/Documents/data/antarctica_t094_f129` (its `ant_pipeline.py`, `tie_check.py`, `ant_roff.py`), runs `ours` / `diff_snaphu` / `diff_isce3` on its cells → `results/antarctica.json`, `antarctica_{windows,zones,arms,profile}.csv`, `antarctica_P{1,2}.npz` (~10 min a pair; `post` and `ours` modes re-run only their part) |
| `experiments/figures.py`, `experiments/tables.py` | every figure → `figures/`, every table → `sections/tables/` |

## How to rerun

```
PY=~/miniforge3/envs/remote_sensing/bin/python
cd docs/ionosphere_paper/experiments
$PY selftest.py
$PY -u run_synthetic.py workers=3          # ~1.2 h on 3 cores; builds the Okada caches first (results/synth_unit_slip_*.npz)
$PY -u run_nisar.py D141 A061 D126 A162 D054   # 35-60 min per scene on a loaded machine
$PY -u run_nisar.py --gunw-phase D141 A061 D126 A162 D054   # after: the products' own main band
$PY -u run_nisar.py --footprint D141 A061 D126 A162 D054    # after: the products' side-band footprint
$PY -u run_tie_null.py                     # the side-band tie: fake boundaries (~1.5 h)
$PY -u run_tie_real.py && $PY -u run_tie_offsets.py   # the Panama ties and controls (~25 min)
$PY -u run_tie_synthetic.py workers=3      # synthetic ties (~35 min)
$PY -u run_pixel.py workers=3 && $PY -u run_pixel.py example=1     # near the trace, synthetic (~40 min)
$PY -u run_fill_synthetic.py && $PY -u run_fill_null.py && $PY -u run_fill_null.py fills=2
$PY -u run_nisar_pixel.py D141 && $PY -u run_d141_integers.py        # D141 near the trace (~1.5 h)
# Section 8 (after the Antarctic pipeline: cd ~/Documents/data/antarctica_t094_f129; scripts/run_pairs.sh, then
# ant_pipeline.py compare P / residual P, and ant_roff.py P for our own offsets)
$PY -u run_antarctica.py 1 2 && $PY -u run_antarctica.py ours 1 2
$PY figures.py && $PY tables.py
```

`run_nisar.py D141` gives the south block one more cycle than the offsets did (`D141_TIE_EXTRA`, default 1: the side
band's tie). Its reproduction check runs `ours` again on the offsets' tie, against the saved screen.

`run_nisar.py` rebuilds each scene's inputs from the pipelines' cached cells and stores
(`~/Documents/data/D141_panama/scripts/panama_pipeline.py`, `~/Documents/data/Venezuela_reprocess/scripts/vz_pipeline.py`).
It checks that its `ours` arm reproduces the saved v6 screen.

## Notes

- The filter's 3 x 3 nanmedian (`scipy.ndimage.generic_filter`) took 12 s per call on 400 x 400 cells.
  `estimators.py` replaces it, in its own process only, with a stacked version that is bit-identical and ~50x
  faster. The library on disk is unchanged.
- Results that the scripts rebuild (`results/*.npz`, logs) are git-ignored.
- Built with TinyTeX (installed 2026-10-21 in `~/Library/TinyTeX`, not on PATH):
  `PATH=~/Library/TinyTeX/bin/universal-darwin:$PATH latexmk -pdf main.tex` → `main.pdf` (37 pages with Section 8, a clean log).
  TinyTeX is minimal: add a missing package with `tlmgr install <name>`; `tlmgr search --global --file /<file>.sty`
  finds which package holds a file. If latexmk keeps reporting an error you have already fixed, run `pdflatex` once by
  hand, then latexmk again.
