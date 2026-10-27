# Our NISAR InSAR pipeline, end to end, and how it differs from the ISCE3 PR products

*Written 2026-10-14. Covers the coseismic processing used for Panama (D141, A004) and the Venezuela rebuild (A061,
A162, D054, D126). Numbers are from the measurements recorded in `CLAUDE.md` and the per-folder READMEs; where a step
has a version number (v3, v6) it is the pipelines' stage name.*

## In short

What we do differently from the PR GUNW, and why:

1. **Registration.** We form each pair from ASF's date-1 GSLC and our own date-2 GSLC, geocoded onto the same grid with
   a measured registration field. Two ASF GSLCs as downloaded are misregistered by about a resolution cell by their own
   TEC corrections.
2. **The rupture.** We cut a ±200 m strip along the mapped trace before unwrapping, so the unwrapper never crosses the
   rupture. A block that the cut isolates is either tied by pixel offsets or given its own offset in the inversion.
3. **The ionosphere.** We unwrap only band A. The side band enters through a residual that contains no deformation,
   summed on the same map cells as band A. The GUNW's screen copies part of the deformation near a rupture (up to 13%);
   ours does not.
4. **Partial focus.** NISAR's band A is partially focused over the ~10 km nearest the near-range edge, and at the start
   of a datatake. The GUNW drops those samples. We keep them in the data and drop them only where they bias the screen.
5. **Few masks.** No coherence mask, no outlier removal, no cycle "repair" and no smoothing before sampling (the user's
   defaults, 2026-10-13/14).

## Inputs

- **ASF PR GSLCs** (L2, 5 m posting, frame grids): band A (40 MHz at f_A = 1239 MHz) and band B (5 MHz side band at
  f_B = 1293.5 MHz). Each carries a per-product TEC and solid-earth-tide (SET) geolocation correction.
- **ASF PR RSLCs** of the second date, to geocode ourselves.
- **The NISAR DEM**, staged per frame (`dem.stage_nisar_dem`).
- **A mapped fault trace**: Panama, the main trace drawn in Google Earth on the slant-range offset step; Venezuela,
  `Venezuela_fault_trace_2.kml`.

## The pipeline, step by step

### 1. Registration: our own date-2 GSLC on date 1's grid

- **The problem.** ASF's GSLCs each apply their own TEC + SET geolocation correction. Between two dates those
  corrections differ, and on D141 they moved date 2 by +2.5..+2.7 m along track and +3.1..+3.2 m in slant range
  against the true offsets of −1.3..−2.0 / +0.5..+2.0 m: about a resolution cell. The pair's coherence fell to 0.29,
  against 0.62–0.75 once registered. Along-track misregistration also biases the phase (−2π f_dc Δt, about 1.7 cm LOS
  per metre).
- **What we do.** Keep ASF's date-1 GSLC. Geocode the date-2 RSLC with `geocode_rslc` (our port of ISCE3 v0.25.16's
  GSLC workflow; 99.80% / 99.87% of pixels bit-identical to ASF's own product in bands A / B), onto date 1's grid
  (`like=`), with `tec_file=None`, `solid_earth_tides=False` and a `RegistrationField`:
  date 1's own TEC+SET shift plus the measured offsets.
- **The offsets** come from `roff.py` + `ampcor.py`, our numpy port of ISCE3's ROFF (PyCuAmpcor is GPU-only): 99.7–99.9%
  of windows bit-identical to the UR ROFF on the same inputs. For band B the same metres are converted to B's pixels,
  with B's own TEC shift (TEC scales by (f_A/f_B)² = 0.917; SET does not).
- **Mixing a corrected GSLC with an uncorrected one is safe**: the corrections only move where pixels are sampled. Both
  ISCE3 and our port flatten the phase with the uncorrected range.
- **The ISCE3 GUNW** coregisters the two RSLCs in radar geometry (geometry, refined with dense offsets) and geocodes
  the result, so it is not misregistered either. The difference is where the pair is made: ours on the 5 m GSLC grid.

### 2. Interferograms

- `GSLCStack.form_interferograms(pairs="sequential", looks=10, convolution="Gaussian", sigma=...)`: reference ×
  conj(secondary) at full resolution, multilooked 10 × 10 onto a **50 m** lattice anchored to absolute coordinates, with
  Gaussian weights (σ = 5 pixels = 25 m for Panama; σ = 10 pixels for Venezuela). Coherence from the same looks.
- **Frames are merged before forming** (Panama 084 + 085, A004 005 + 006, Venezuela's two frames per track). There is
  one interferogram across the frame boundary, so one unwrapping and one integer.
- **The GUNW** posts at 80 m and processes each frame on its own. Each frame then carries its own integer and ramp
  (Venezuela's GUNW frames disagree by up to 4.6 cm).

### 3. Masks and regions

- **Water**: `mask_water(resolution="f")` (full-resolution coastline). Nothing else is masked: **no coherence mask**.
- **The fault cut** (coseismic pairs only): a ±200 m NaN strip along the trace. Panama cuts the whole drawn main trace
  (not the secondary strands). Venezuela cuts only the airport stretch (arc 180–191 km of trace_2).
- **Regions**: `remove_unconnected_regions` (regions under 10 km² go), then the region rule: keep the largest region,
  the block(s) the fault cut isolates, and, in Panama, the big islands (Coiba, Cébaco). Every other region is dropped.
- **The GUNW** masks water, invalid and partially focused samples, and flags anomalies; it does not cut the rupture.

### 4. Filtering and unwrapping

- **Adaptive Goldstein** filter on the 50 m interferogram, then **SNAPHU** in smooth mode (snaphu-py, SNAPHU 2.0.7,
  8 processes).
- SNAPHU's defo mode was tried on D141 (2026-10-14) and did not change the integer across the rupture.
- **The GUNW** unwraps with ISCE3, frame by frame, without a cut, across the rupture.

### 5. Tying a cut-off block (Panama)

- The block the cut isolates (D141's and A004's south blocks) is unwrapped on its own, with an arbitrary integer. We tie
  it with our ROFF slant-range offsets: `m = (4π/λ) · ROFF − φ` holds no deformation, only the block's whole cycles plus
  twice the ionosphere (the offsets see it with the opposite sign). Planes fitted to m at 6–12 km on each side are
  extrapolated to the trace, and the nearest integer is applied.
- **This tie is uncertain by about ±1 cycle** (see Limitations), so in the slip inversion each region keeps its own
  offset whatever the tie.
- **The side band ties the block itself** (`split_spectrum.block_cycles`, 2026-10-21).
  - A block whose integer is off by e cycles moves its raw dispersive estimate by 3.21 e rad, and the ionosphere is
    continuous across the gap. So the raw estimate's step across the gap reads e.
  - Estimator: a quadratic over both blocks plus a step, in 20 km windows along the trace; the median over windows.
  - On fake boundaries in 8 scenes, no tie read a wrong integer.
  - It finds the offset ties of D141, D040 and A076 (v8) one cycle short: they should be 27, 21 and −22, not 26, 20
    and −23.
  - Details and caveats: `docs/ionosphere_paper/` (section 6, README).
- **v9 (2026-10-21; D141, D040, A076; v8 untouched).** v8's cut unwrapping with the side band's ties, then the screen from
  per-pixel cells (`panama_pipeline.py` stages unwrap9m, patch9, iono9, los9, compare9).
  - The south blocks take +1 cycle (27, 21, −22). Read again on the v9 screens: D141 +0.23 (kriging −0.12), A076
    +0.06 (+0.27), D040 +0.01 / +0.62 / +1.48 in its three windows west to east (kriging −0.48).
    - D040's east window is not the tie but two patches north of the trace that its side band flags, beside the branch cuts.
      With them left out, D040 reads +0.10 (kriging −0.07): 21 stands.
    - x 541–545 km (11 km², 3–7 km north of the main): band A is wrong in BOTH tracks, the opposite ways.
      - D040: side band −3.1 cycles, own range offsets −3.7, graded −1..−6 across the patch.
      - A076: +1.4 / +1.7, graded 0..+4.
      - Both tracks' offsets have the ground moving ~0.75 m east and ~0.73 m down against its surroundings: −1.0 m of D040's
        LOS, and nearly nothing of A076's.
      - No single integer corrects it; mask it in both.
    - x 555–565 km: A076's phase is right there (side band 0.0, offsets 0.0) and sees −0.76 m of real LOS. D040's is
      mostly right (72% of pixels exact, ~20% a cycle high). This is a real feature, not an unwrapping error.
    - Every SNAPHU variant makes the same choice in both: ROFF-flattened, ROFF + 2 × screen-flattened, and a box without the
      dead-end branch cuts (unwrap9m, unwrap9m2, unwrap9nb). Both are still in v9.
  - Patch by patch, where another unwrapping of the pair disagrees by whole cycles, the side band reads which one is
    right (`split_spectrum.cycle_patches` / `patch_cycles` / `patch_verdict`).
    - Alternatives: v8's cut interferogram flattened by the registration ROFF before SNAPHU, and, on D141 only, the uncut
      v5 / v5defo.
    - One switch: D141, 26 km² centred 3 km north of the trace, a cycle high in v8 (read −0.81 ± 0.08 on 129 cells
      against v5, confirmed against the flattened one). After it, D141's tie fraction fell from 0.19 to 0.03.
    - On D040 the flattened alternative was wrong over 1,269 km² and the side band kept v8 there; its model fits
      D040's unwrapping poorly (slope 0.1, the ionosphere dominating).
    - Undecided: 115–146 km² per track, almost all within 2 km of the trace, where the cells are decorrelated or the
      per-cell raw estimate is unusable. These cells are listed in `cycle_patches_v9.{csv,npz}`.
  - v9 − v8 LOS (positive toward the satellite), beyond each block's constant (south −58 to −60 mm):
    - north of the trace: +37–40 mm at 0–1 km, +23–32 mm at 2–4 km, +6–10 mm at 6–10 km;
    - south of it: −17 to −18 mm at 0–1 km.
    - On D141 the repaired patch's pixels moved by a whole cycle (121 mm): 7–14% of the north side within 6 km.
- Venezuela's airport blocks are not tied: they had their own offset, and model 4d drops them.
- **The GUNW** has no cut, so nothing to tie; its unwrapper chooses the integer across the rupture.

### 6. The ionosphere: our split spectrum (`split_spectrum.py`)

The physics is the same for any split spectrum: the phase in band X is
`φ_X = (f_X/f_A) φ_nd + (f_A/f_X) φ_io + 2π n_X`, with φ_nd everything non-dispersive (deformation, topography,
troposphere) and φ_io the ionosphere at f_A. What differs is how the bands are combined.

- **Cells.** Both bands' interferograms are summed over **identical 240 m map cells** from the registered GSLCs
  (`gslc_cells`). Since v6 (2026-10-14), pixels the product mask marks partially focused are dropped from the cells
  **only in band A's near-range strip** (within 11 km of the near-range edge; 12 km on D141, measured from the masks:
  the strip is ≤ 10.1 km, ≤ 11.05 km on D141). The partially focused start of a datatake is kept: A004's rupture lies in
  one, and dropping it extrapolated the screen 20–40 km.
  `gslc_cells(phase_model=)` (2026-10-21, not used by the pipelines) can also take our unwrapped band A out of every
  pixel of both bands before the sums, each scaled to its frequency, so steep fringes inside a cell neither decorrelate
  it nor weight the bands' cell phases differently. North of D141's trace it doubles the cells kept within 5 km
  (5 → 20% at 0.5–1 km, 18 → 35% at 2–5 km). The multilook and Goldstein filter of the unwrapped band A reach the screen
  only through its integers, in either form: the cells' values are the GSLC pixels.
- **Band A from our unwrapping.** φ_A is our unwrapped band-A phase averaged onto the cells, from the main region (and a
  tied block) only: an island's arbitrary integer would shift its screen by 3.2 rad per cycle.
- **The residual.** Band B is never unwrapped. It enters through
  `r = wrap(φ_B − (f_B/f_A) φ_A) = wrap(k φ_io − 0.044 · 2π n_A)`, with k = f_A/f_B − f_B/f_A = −0.0861.
  The non-dispersive term, deformation included, cancels exactly, so r is small and smooth. It is unwrapped
  (`unwrap_smooth_phase`) and the screen is `d = r / k`. The unwrapping low-passes exp(i r), fills gaps with wider
  Gaussians, and runs SNAPHU twice: over the whole grid, and over the data's reach alone (~2 km past the cells). Each
  connected piece of the reach takes the whole-grid solution's integer. The second run was added on 2026-10-17 (D040, below).
- **Which cells.** Coherence above 0.2 in both bands, at least half the pixels valid, and not within 2 km of the trace.
- **The filter.** ISCE3's `IonosphereFilter.low_pass_filter`, ported in metres: Gaussian σ = 2.0 × 2.6 km, a guide,
  outliers beyond 3.5 MAD, inverse-variance weights. **One change** (v6): masked cells are filled with a local plane
  (σ 10 km, at most 12 km from the data) instead of ISCE3's zero-order guide. The guide lags a gradient at a one-sided
  edge, such as a coast along the fault plus the 2 km exclusion (+25.7 mm LOS at D126's trace on a noise-free test).
  The plane width was chosen by withholding an 11 km band of A004's own data: 90th-percentile error 33 mm with the guide's
  2.6 km plane, 7.4 mm at 10 km. Two options since 2026-10-21, neither used by the pipelines: a ladder of fill widths
  (`fill_m=(2600, 5000, 10000)`: each filled cell takes the narrowest plane whose data lie around it) and a first-order
  final step (`order=1`: a weighted local plane instead of ISCE3's weighted mean, which lags the ionosphere's gradient
  where its weights change, as across a damage zone). On 34 fake gaps cut out of eight production screens the fills
  differ by about a millimetre in 1–2 km zones (the ladder best, the 10 km plane 0.6–1.1 mm worse), and the first-order
  step by ≤ 0.4 mm. Next to D141's rupture the narrower fills follow the main band's local cycle errors (below).
- **Applied.** The screen is interpolated to the 50 m grid and subtracted from φ_A.

What this buys, and what it still depends on:

- **No deformation in the screen**, because r has none. On a synthetic test, 80 rad of near-fault deformation with a
  step put less than 0.05 rad into the screen.
- **No differential errors**, because band B has no integer of its own to get wrong.
- **No cycle corrections**, so nothing can mistake deformation for a cycle error.
- **It does depend on φ_A's integers**: a cycle of φ_A on a region moves the screen there by 0.511 × 2π = 3.21 rad
  (62 mm LOS). Confirmed 2026-10-14: where SNAPHU defo and smooth chose different integers, the raw screen moved by
  exactly 3.21 rad per cycle.
- **And on r's integers**: one cycle of r is 2π/|k| = 73 rad of screen (1.4 m LOS). r is smooth, but a large gap's
  fill averages data whose r differs by up to π and can leave a phase singularity in the gap. Over the whole grid,
  SNAPHU must cut from it to the grid's edge and takes the shortest route. On D040 (2026-10-17) a singularity in Parita
  Bay, 28 km from the nearest cell, cut across the 20 km neck joining the Azuero peninsula to the mainland: Azuero's
  north block came out 1.3 m off the mainland in LOS. Unwrapping over the data's reach removes this within any
  connected piece of data. Pieces further apart than the reach are still tied through the fill. D141, A004 and the
  four Venezuela v6 screens had no singularity and are unchanged.

**The GUNW's split spectrum** (ISCE3 `main_diff_ms_band`, `d = 0.511 φ_A + 11.61 (φ_A − φ_B)`):

1. both bands are unwrapped, and the band difference, which still carries 4.4% of the deformation, is unwrapped on
   its own;
2. a second pass corrects φ_A by whole cycles against a ~2 km low-pass of the non-dispersive field, which near a
   rupture reads the deformation as cycle errors (3.2 rad of screen per false cycle);
3. band A is decimated to band B's columns by stride, a constant ~9 m range offset;
4. the two bands have very different footprints (40 m vs 325 m);
5. the filter fills gaps with ISCE3's zero-order guide.

The result is a sign-flipped copy of the deformation near ruptures. Within 10 km of the trace it is 8.1 / 13.2 / 4.6 /
1.4% of the data's high-passed signal for A061 / D126 / A162 / D054, and +9% north / −5% south on D141. The GUNW's
`unwrappedPhase` is **not** corrected: its screen is a separate layer, `ionospherePhaseScreen`.

### 7. Line of sight

- `to_los`: `d = −(λ/4π) φ`, positive toward the sensor, with the look vectors from the products' geometry cubes. The
  50 m LOS stores go to slipsolve, which samples them with its quadtree **without any pre-smoothing**.
- The ROFF offsets can enter as data too (`roff.offset_los_stacks`: slant range on the LOS vector, along-track on the
  along-track vector). Model 4d does not use them.

## Side by side

| | ISCE3 PR GUNW | Ours |
|---|---|---|
| Pair | RSLCs coregistered in radar geometry | ASF date-1 GSLC + our date-2 GSLC on its grid, registration field from measured offsets |
| Posting | 80 m | 50 m (10 × 10 looks on 5 m GSLC pixels, Gaussian) |
| Frames | each on its own | merged before forming, one unwrapping |
| Partially focused samples | dropped (near-range strip, datatake start) | kept in the data; dropped from the screen's cells only in band A's near-range strip |
| Rupture | unwrapped across | ±200 m cut; isolated block tied by offsets or given its own offset |
| Masks | water, validity, anomalies; coherence and components layers | water only |
| Unwrapping | ISCE3, per frame | SNAPHU smooth on the Goldstein-filtered 50 m interferogram |
| Ionosphere | both bands unwrapped, band difference unwrapped, pass-2 cycle corrections, decimated band A, zero-order fill | band A only; deformation-free residual on identical cells; no cycle corrections; 2 km fault zone filled; plane fill |
| Ionosphere in the phase | separate layer, not applied | subtracted |

## How well it works

- **Agreement where both are clean** (each screen less a cubic):
  - D141 frame 084: correlation 0.9992, 9.6 mm rms.
  - Venezuela, four tracks: 5.1–5.6 mm.
  - A004 frame 006: 6.2 mm, and within ±3 mm out to 60 km inside the GUNW's valid edge.
- **Small-scale structure that is real ionosphere is in both.** D054's 10 km high-pass matches the GUNW's with
  correlation 0.94 / 0.92 (frames 083 / 084), the same amplitude, 3–5 mm apart: bands 15–30 km apart on a dusk pass,
  about twice the fine structure of A061 and A162 (5–6 mm rms).
- **Leak near the rupture**: ours 0.3% on D141 (GUNW +9% / −5%), 1.0–2.6% on Venezuela except D054's 5.0% (GUNW
  1.4–13%).

## Limitations and open items

- **Tie ambiguity.** The offset tie measures `π n + φ_io`, and so does the split-spectrum screen (adding twice the
  screen to the tie leaves only 0.022 cycles per cycle of n), so their combination cannot separate the integer from
  the ionosphere. The screen's own continuity across the gap does: `split_spectrum.block_cycles` (2026-10-21) reads the
  integer to about a tenth of a cycle, and finds three Panama offset ties a cycle short. Earlier numbers:
  - D141: 32.65 → k = 33, but 32.41–32.90 along strike (±1).
  - A004: −18.43, with −19.45..−17.41 along strike (a two-cycle spread).
  - Per wrong cycle: the block's corrected LOS is off by a constant 59 mm far from the trace, which its own offset
    absorbs. The screen filter also spreads the 3.21 rad step across the trace: on D141 that leaves a fake near-trace
    step of about 5 cm (in phase-equivalent mm, −40 at 0.25 km north and +14 just south, gone within 5–7 km; the LOS,
    positive toward the satellite, has the opposite signs).
    `D141_panama/scripts/tie_cycle_screen.py`.
- **Near-trace integers.** Next to D141's rupture the cut (v3) and uncut (v5) band-A unwrappings disagree on 78 / 62 /
  42 / 18% of the cells at 0–1 / 1–2 / 2–4 / 4–6 km north of the trace (≤ 3% beyond), each cycle 3.21 rad of raw
  estimate. The production screen moves ≤ 2.4 mm when those cells take v5's integers (its 10 km fill averages them
  away); narrower fills move 7–16 mm. So near a trace the screen is limited by band A's local integers, and the wide fill
  plus the 2 km exclusion are the robust choice until they can be checked. Paper Section 7,
  `docs/ionosphere_paper/experiments/run_d141_integers.py`. v9 (§5) checks them patch by patch where an alternative
  unwrapping disagrees. It repaired one 26 km² patch on D141, but within ~2 km of the trace most disagreements stay
  undecided.
  - ⚠️ A north-side excess of cells +0.5 cycle above the screen in all three v8 screens was mostly an artefact of
    summing steep fringes per cell. With per-pixel cells it drops from +26–34 to +6–8 points at 2–4 km. Only on D141 was
    it shown to be cycles.
- **Partially focused data.** A004's rupture area lies in the partially focused start of the datatake in both dates.
  It is kept, and no GUNW can check it there: the GUNW has no valid sample within 21 km of the trace.
- **Coastal spots.** A004's screen has spots 1–3 km inland around an inlet (raw estimate 0.9–2.5 cycles off its
  surroundings). They are not our unwrapping (it matches the GUNW's to 0.02 cycles there), and the GUNW's screen has
  them too. The cause is not established.
- **The filter's own uncertainty** (ISCE3's output sigma, 21–23 mm everywhere) exceeds every measured small-scale
  rms, so it cannot say whether a feature is noise.

## Where things are

- **Package** (`~/Documents/nisar_tools/nisar_tools/`): `gslc_geocode.py` (RSLC → GSLC), `roff.py` + `ampcor.py`
  (offsets), `stack.py` / `interferogram.py` (forming, Goldstein, masks, regions), `unwrap.py` (SNAPHU, `to_los`),
  `split_spectrum.py` (the screen), `los.py`.
- **Pipelines**:
  - Panama: `~/Documents/data/D141_panama/scripts/panama_pipeline.py` (`TRACK=D141|A004`). Stages `igrams clean3
    unwrap3 tie3 iono6 los6` (and `keep6` for D141).
  - Venezuela: `~/Documents/data/Venezuela_reprocess/scripts/vz_pipeline.py T ...`. Stages `igrams clean3 unwrap3
    iono6 los_iono6` for the three airport tracks; D054, with no cut, screens its `reg` unwrapping.
  - Registration: `frame_registration.py` (Venezuela), `A004_panama/scripts/a004_registration.py` and the D141
    equivalents.
- **Products** (screen version v6): Panama `workspace_reg/*_v6k_s5` (D141), `workspace_reg_005_006/*_v6_s5` (A004);
  Venezuela `workspace/{unwrapped,los}_{T}_iono6`. Each screen's cells are `iono_split_spectrum_*.nc`.
