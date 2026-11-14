# Refocusing an RSLC from its L0B

Most defects of a delivered RSLC are corrected where geocoding samples it (see [Checking an RSLC before
use](checking-rslcs.md)): band B's range offset, and whole-pulse slips that cover the area of interest. Three kinds
change how the echoes were added up, and only refocusing the L0B undoes them:

- a **wrong PRF**, from a squeezed span of pulse tags anywhere in the datatake: the whole frame is defocused and
  misplaced (`timing_report(rslc).verdict == "refocus"`);
- the **edge of a slip** inside the area of interest, where one synthetic aperture mixes good and slipped echoes;
- **echo removed by RFI mitigation**, which ISCE3 v0.25.16 does to clean echo beside swath edges and gaps.

Otherwise use the RSLC, or another date.

```python
from nisar_tools.focus import focus_inputs, run_focus

run = focus_inputs(rslc, "work/", window={"start_time": "2025-01-01T10:10:52", "end_time": "2025-01-01T10:10:59",
                                          "start_range": 1008e3, "end_range": 1040e3})
refocused = run_focus(run, python="/path/to/isce3-env/bin/python", polarizations=["HH"])
gslc = geocode_rslc(refocused, "data/", dem, tec_file=tec)
```

`focus_inputs` gathers what production used, from the RSLC's own metadata: the L0B(s) it names (10-90 GB each, from
ASF), its calibration files, its orbit and attitude (written from the RSLC as NISAR XML, since the processing
system's ephemeris files are not distributed), the NISAR DEM v1.2, and production's run configuration with local
paths and CPU processing. It also writes `pulse_timing.txt`, what `nisar_tools.l0b.pulse_timing` finds in each L0B.
The output is an ordinary RSLC named `NISAR_L1_OD_RSLC_...`, so it cannot pass for a product of the processing
system.

## Which ISCE3

nisar_tools does not import ISCE3. `run_focus` runs its focusing workflow, on the CPU, with the interpreter of an
environment that has it (`python=`, or the `NISAR_TOOLS_ISCE3_PYTHON` variable).

Stock ISCE3 v0.25.16, the release that focused the provisional products, reproduces production, defects included;
`run_focus(require_fork=False)` runs it, which is how a refocus is checked against production. Repairing the defects
needs ISCE3 patched, and `run_focus` refuses an ISCE3 that does not repair pulse tags. Our patched v0.25.16:

- replaces pulse tags that disagree with the commanded pulse sequence (slips and squeezed spans) with the sequence's
  times, fitted to the tags that agree;
- joins sub-swath pieces that overlap by a sample (under NumPy 2 they were not, and a line lost most of its swath);
- normalises the transmit weights of the lines before the first calibration-bypass line;
- fits the RFI threshold of tone-rank mitigation over valid echo only, and stops one block's inverse FFT leaking
  into the next;
- trims the receive channel-imbalance mean by magnitude rather than by record position;
- gives the image the calibration factor's phase rather than its conjugate (HV, VH and VV; HH is unaffected);
- centres the biquintic spline that interpolates the DEM;
- estimates the noise-equivalent backscatter from fully compressed samples, and fills and extrapolates it in dB so it
  stays positive.

For HH interferometry the pulse tags and the RFI fixes matter; the transmit and receive weights move the antenna
pattern smoothly across range, and the spline changes the focusing geometry negligibly.

## Costs

ISCE3's CPU backprojection is slow; production focuses on GPUs. Measured on one NISAR 40 + 5 MHz datatake (HH, with
the defaults above):

- range compression and RFI mitigation of the whole raw swath: about 5 minutes per channel for 10 s of echoes;
- backprojection, band A (40 MHz): 18 minutes per 1024 lines x 10,247 samples on a 10-core Xeon W-2150B, and 30
  minutes on a 6-core i7-8700B, about 1-1.7 x 10^-4 s per output pixel. Band B adds about a sixth of that.

A 7 s x 32 km window takes about 3.5-5.5 hours per polarization; a full frame (about 40,000 x 52,600 band-A
pixels) would take 2.5-4 days. Refocus a window around the area of interest, and only the polarizations needed
(`polarizations=["HH"]`): each polarization costs as much as the first.
