# Source-space strategy: resolvability first, then a literal translation of the sensor analysis

_2026-10-21. Decisions taken with the user in this session; commands are ready to run on the workstation._

## 1. Why the current source results have no story

The custom-ROI arm (LCMV, 6 functional-localizer spheres, leakage-corrected, logistic tuned, n = 20) shows **no regional selectivity**:

| overtProd / prodDiff | peak acc | peak (ms) |
|---|---|---|
| awfa | 0.616 | 13 |
| ifc | 0.594 | 48 |
| tpc | 0.582 | 9 |
| pmc | 0.569 | 18 |
| vwfa (visual control) | 0.588 | 9 |
| owfa (visual control) | 0.577 | 13 |

Perception/percDiff is the same picture: all six ROIs peak within 0.02 of each other at 170–200 ms. One signal is copied into every sphere. GC between the spheres is at chance in every clean-baseline arm (see `project_gc_final_config` memory), ROIs are rank-1 (PC1 86–89 %), and the raw-vs-leakage arms of TRGC are essentially uncorrelated (`exploratory/compare_gc_leakage_arms.py`, r ≈ 0.16).

Swapping the localizer spheres for another parcellation of the same size does not change this: the limit is the point-spread of a 64-channel template-anatomy inverse, not where the spheres sit. So the plan tests resolvability first and translates the sensor analysis at its own granularity before refining anything.

## 2. What was built (all in `exploratory/` unless noted)

| Piece | File | What it does |
|---|---|---|
| eLORETA / sLORETA | `inverse_pipelines.py` (core) | `run_minimum_norm_lowram(method=…)` on the shared fixed-orientation inverse; `run_inverse_lowram()` dispatcher; `--method {dSPM,sLORETA,eLORETA,LCMV}` in every runner. eLORETA output stored in nA·m (×1e9) to keep the float32 caches and absolute regularizers sane. Smoke-tested on EEGPROD4003/perception. |
| `sensorEquiv` atlas | `config.py` (core) | The four sensor pseudo-channel ROIs (Temporal, Inferior_Frontal, Superior_Frontal, Superior_Parietal; HCPMMP1 composites) + `Visual` (V1–V3) null ROI. Own atlas name ⇒ own cache dirs, no collision with the 16-ROI HCPMMP1 caches. `VISUAL_CONTROL_ROI` gives a visual ROI for aparc/HCPMMP1/Schaefer200 too. |
| 500 Hz caches | `--fs 500` on the runners | Source-localizes the continuous-resampled 500 Hz EEGLAB exports into a separate `raw_fs500` / `leakage_corrected_fs500` cache dir (4× smaller and faster; same 0.1–30 Hz band). Smoke-tested for both tasks. **Rule (user, 2026-10-22): decoding runs on the 2000 Hz caches, GC on the 500 Hz caches** — the driver localizes both rates (`DEC_FS=2000`, `GC_FS=500`). |
| Windows of interest | `config.GC_WINDOWS_OF_INTEREST` | overtProd `phonloop` −1.2..−0.8 s, `preonset` −0.5..0 s; perception `auditory` 0.15..0.30 s. Used by the latency script, the leakage comparison and the translation-arm stats. |
| Resolution analysis | `resolution_analysis.py` | ROI-level CTF/PSF of the pipeline's own inverses (no full resolution matrix): pattern overlap \|cos\| = expected zero-lag correlation, cross-talk %, se_own, spatial dispersion, peak location; sensor pseudo-channels on the same footing; brain maps optional. |
| Translation arm | `run_translation_arm.sh` | localize (2000 Hz for decoding, 500 Hz for GC) → decode → latency → GC → stats for `sensorEquiv` × {dSPM, eLORETA, sLORETA}, raw, sensor BSMART settings (order 10 / 40 ms / fs 500 / normalize none / 1 PC), plain GC + TRGC, stats on the full span and per window of interest. |
| Decoding timing | `decode_onset_latency.py` | Jackknife onset/peak latencies per ROI, ROI-vs-control cluster test over time, ROI × time RM-ANOVA cluster test, fixed-window accuracy vs control. Works on the existing custom results with `--control-roi vwfa-lh`. |
| Leakage check | `compare_gc_leakage_arms.py` | Raw vs leakage-corrected arms of one GC config: overlays, arm similarity, fixed-window tests. Already run on `order10_win40ms_fs500` (results below). |

## 3. Results already in hand (local, no estimation runs)

**Raw vs leakage-corrected TRGC** (`GC_source_space/_figures_final_review/leakage_arms_order10_win40ms_fs500/`): the two arms are nearly unrelated. Median r between the subject-mean net-TRGC curves is 0.16 (IQR 0.08–0.27); pointwise-significant windows overlap in 0.6 % of the span (raw 8.3 %, leak 6.2 %, chance 5 %). In the fixed windows the group sign agrees in only 61–72 % of edge × band cells. Leakage correction is therefore load-bearing for TRGC, not cosmetic. Decision: the translation arm runs **raw + TRGC** (one leakage control, Pellegrini 2023); the leakage arm is a sensitivity analysis (`LEAK=--leakage-correction`).

**Decoding timing on the custom ROIs** (`DECODE_source_space/overtProd/LCMV/custom/vertex_selectkbest/leakage_corrected/no_pseudo/40_5/logistic_tuned/_onset_latency/`): see `prodDiff_summary.txt` after the run; the question it answers is whether any speech ROI decodes earlier or better than vwfa.

## 4. Workstation run order

```bash
cd code/source_estimation && conda activate mne

# A. resolvability (light; ~2 min per subject per method; writes RESOLUTION_ANALYSIS/)
python exploratory/resolution_analysis.py --task overtProd --stim-class prodDiff \
    --methods dSPM sLORETA eLORETA LCMV --atlases custom sensorEquiv --n-jobs 8 --brain-maps
python exploratory/resolution_analysis.py --task perception --stim-class percDiff \
    --methods dSPM sLORETA eLORETA LCMV --atlases custom sensorEquiv --n-jobs 8

# B. translation arm, all stages (~600 GB of caches per method: 2000 Hz for decoding + 500 Hz for GC)
bash exploratory/run_translation_arm.sh
#   or stage by stage:
STEPS=localize bash exploratory/run_translation_arm.sh
STEPS="decode latency" bash exploratory/run_translation_arm.sh
STEPS="gc stats" bash exploratory/run_translation_arm.sh

# C. sensitivity arms (optional)
LEAK=--leakage-correction STEPS="localize gc stats" bash exploratory/run_translation_arm.sh
METHODS=LCMV bash exploratory/run_translation_arm.sh

# D. the same leakage comparison on the new arm once C has run
python exploratory/compare_gc_leakage_arms.py --tag order10_win40ms_fs500 \
    --method eLORETA --atlas sensorEquiv \
    --pairs Temporal,Inferior_Frontal Temporal,Superior_Frontal Temporal,Superior_Parietal \
            Inferior_Frontal,Superior_Frontal Inferior_Frontal,Superior_Parietal \
            Superior_Frontal,Superior_Parietal
```

Outputs to look at, in order:

1. `RESOLUTION_ANALYSIS/{task}_{stim}/summary.txt` and `pattern_overlap.png` — which ROI sets and inverses give pairwise |cos| well below 0.5, and how the sensor pseudo-channels compare.
2. `DECODE_source_space/{task}/{method}/sensorEquiv/.../_onset_latency/{stim}_onset_latency.png` — do the speech ROIs separate from Visual in time or in the fixed windows?
3. `GC_source_space/{task}/{method}/sensorEquiv/vertex_selectkbest/raw/order10_win40ms_fs500/all_rois/{stim}/group_stats*/gc_permutation_summary_ttest.log` — full span (sensor-faithful) and `group_stats_win_{phonloop,preonset,auditory}`.

## 5. Decision rule

* If the resolution analysis shows the four lobe-scale ROIs resolvable (|cos| ≲ 0.3–0.4, se_own well above chance) and the spheres not, the source chapter becomes a coarser confirmation of the sensor network; the localizer ROIs are reported as a documented resolution limit.
* If even the four are not resolvable, keep sensor GC as the primary result, report source decoding as timing confirmation (latency script), and present source GC as a resolution-limited negative with the CTF figure as the explanation.
* Inverse choice is made on the resolution numbers plus the Visual-control decoding, not on which arm gives more stars.

## 6. Notes and caveats

* **Where the big outputs go.** The only large outputs are the vertex caches; they are written to `ROI_TIMESERIES_SAVE_ROOT` from `config.env` (on the workstation: `/media/maxlab_bigdrive/.../DECODE_source_space_timeseries`) and found again through the same root, so nothing of the ~600 GB touches the project drive. Decoding CSVs, GC npz and the resolution-analysis caches are small (MBs to a few GB in total) and stay under `derivatives/source_estimation/`.
* `sensorEquiv` composites share parcel lists with `SPEECH_ROIS['HCPMMP1']`; Temporal and Superior_Parietal are large (13 and 12 parcels).
* The decoding and GC result paths do not encode `--fs`; the arm's convention is fixed: decoding results come from the 2000 Hz caches, GC results from the 500 Hz caches.
* `run_decode.py` requests every ROI of the atlas; a cache written with `--roi-subset` lacks the others and the subject is skipped — localize the full `sensorEquiv` set.
* LCMV at 500 Hz input is a new arm, not comparable to the 2048-Hz LCMV caches (data-adaptive filter; see `project_gc_downsampling_order`).
* The stats baselines for the windows-of-interest passes are the `_bl` ones (overtProd −1.47..−1.37 s, perception −0.08..0 s); the full-span pass uses the sensor-faithful leading 100 ms.
