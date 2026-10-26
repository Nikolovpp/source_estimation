# Source-space findings so far (2026-10-26)

A one-page record of what the resolution analysis, the inverse sweep, the
first translation-arm decoding and the subject QC have established, each
claim with the figure that backs it. Figure paths are relative to this file;
`R` = `../../../derivatives/source_estimation/RESOLUTION_ANALYSIS`,
`D` = `../../../derivatives/source_estimation/DECODE_source_space`.
Both tasks (overtProd/prodDiff, perception/percDiff) give the same picture;
the overtProd figures are linked, the perception twins sit next to them.

## How the numbers are made

* **CTF (cross-talk function) of ROI i** = the row of the resolution matrix
  R = K·G that belongs to ROI i's time course: the ROI weights applied to
  the inverse kernel K, times the whole forward model G. It says how much of a
  unit source anywhere on cortex ends up in that ROI's signal. The forward
  model is the fsaverage template for every subject, so the only
  subject-specific ingredients are the baseline noise covariance (whitens K)
  and the task-window data covariance (LCMV weights, pc1 ROI weights).
* **|cos(CTF_i, CTF_j)|** = expected zero-lag correlation between ROI i and
  ROI j under spatially white sources; the quantity that matters for both
  decoding (shared signal) and TRGC (instantaneous mixing).
* **se_own** = share of CTF_i energy on ROI i's own vertices.
  **peak_in_roi**, **sd_mm** = where the CTF peaks and how far it spreads.
  **noise_gain** = sensor noise SD reaching the ROI time course per unit
  own-source gain, on one fixed reference covariance for every configuration.
  **PC1 share** = fraction of ROI variance in one component of K·C_data·Kᵀ;
  near 1 = the beamformer collapse (one global time course on every vertex).
* **sensorEquiv ROIs** = the four 3-electrode pseudo-channels of the sensor
  manuscript (Temporal FT7/T7/TP7, Inferior_Frontal F5/FC5/FC3,
  Superior_Frontal FCz/FC1/F1, Superior_Parietal CPz/CP1/P1) mapped to the
  cortex under them as HCPMMP1 composites (the lists in
  `config.SPEECH_ROIS['HCPMMP1']`), plus `Visual` (V1–V3) as a null ROI.
  The sensor pseudo-channels themselves enter the resolution analysis as
  mean leadfield rows, so sensors and sources are scored on one footing.
* **DeFleCT** (Hauk & Stenroos 2014) = an ROI spatial filter built to pass
  the ROI's own leadfield pattern with unit gain while explicitly nulling
  chosen other patterns: w = M⁻¹F(FᵀM⁻¹F)⁻¹d with M = GGᵀ + C/snr², F = [own
  pattern, null patterns], d = e₁. "null2 other_rois" = the first two
  singular patterns of every other ROI's leadfield are zeroed. A virtual
  channel, not a whole-brain inverse, so it fits the GC stage only.
* **LCMV collapse** was found 2026-07 by inspecting the vertex caches: for
  many subjects every vertex carried the identical time course
  (corr = 1.000, std across vertices ~1e-7; file size a third of a healthy
  cache). Cause: rank 64 passed for rank-63 average-referenced data, plus a
  degenerate noise baseline; fixed with `mne.compute_rank`. The sweep's
  PC1-share metric is the operator-level version of the same test, and it
  says the fixed beamformer still spreads each ROI over ~50 mm.

## Findings

**1. The four lobe-scale ROIs are resolvable under dSPM/eLORETA, better than
the sensor clusters.** Mean |cos| 0.08–0.10 (worst pair Inferior Frontal ↔
Temporal 0.17–0.21), own-energy share 0.34–0.40, against 0.24 mean / 0.63
worst for the sensor pseudo-channels. The translation arm is therefore a
better-separated version of the sensor network, not a blurred one.
Figures: `R/overtProd_prodDiff/pattern_overlap.png` (all inverses, both ROI
sets and the sensor panel), `R/overtProd_prodDiff/crosstalk.png`,
`R/overtProd_prodDiff/ctf_maps_dSPM_sensorEquiv.png`.

**2. LCMV is unusable here in every conditioning tried (24 variants).**
Mean |cos| ≥ 0.41, own-energy ≤ 0.13, CTF spread ~50 mm, PC1 share
0.7–0.85, CTF peaks outside the ROI in most cases. Removing unit-noise-gain
or the noise covariance helps by 0.05–0.1; normal orientation hurts by
0.14–0.24. It is retired; the old custom-ROI LCMV results are
resolution-limited by construction.
Figures: `R/overtProd_prodDiff/ctf_maps_LCMV_sensorEquiv.png`,
`R/overtProd_prodDiff/sweep/report/pareto_sensorEquiv.png` (orange cloud),
`R/overtProd_prodDiff/subject_qc/qc_lcmv_collapse_sensorEquiv.png`.

**3. Of the five inverse levers, only two move anything.** Assumed SNR 3→10
halves the cross-talk (0.10→0.04–0.06) and adds 0.05 own-energy for ×1.8–2
noise gain; DeFleCT with nulls on the other ROIs reaches the lowest worst
pair of all (0.16–0.18) and se_own 0.48 for ×1.4 noise. Depth weighting,
loose orientation and Ledoit–Wolf change nothing (< 0.01); ad hoc /
identity / full-epoch noise covariances and SNR ≤ 2 are worse. Inferior
Frontal ↔ Temporal never drops below 0.13 (adjacent across the Sylvian
fissure).
Figures: `R/overtProd_prodDiff/sweep/report/factors_sensorEquiv.png`
(one-knob curves), `matrices_sensorEquiv.png` (production vs best per
family), `summary.txt` (rankings and marginal effects), and the brain maps
`ctf_maps_sensorEquiv_mn_dSPM_snr10_d0.8_l0_shrunk.png`,
`ctf_maps_sensorEquiv_deflect_snr10_null2_other_rois.png` in the same folder.

| sensorEquiv, overtProd | speech |cos| | worst pair | se_own | noise ×prod | peak in ROI |
|---|---|---|---|---|---|
| production dSPM (SNR 3) | 0.10 | 0.58 | 0.34 | 1.0 | 68 % |
| production eLORETA | 0.09 | 0.54 | 0.38 | 1.1 | 61 % |
| dSPM SNR 10 | 0.06 | 0.34 | 0.39 | 1.9 | 76 % |
| DeFleCT SNR 10, null2 other ROIs | 0.05 | 0.18 | 0.48 | 1.4 | 80 % |
| best LCMV | 0.45 | 0.86 | 0.11 | 1.2 | 43 % |

**4. The six localizer spheres stay resolution-limited whatever the
inverse.** Best mean |cos| 0.09, but owfa ↔ vwfa 0.35–0.51 in every
configuration and awfa/vwfa never put a CTF peak inside their own sphere.
DeFleCT nulls do not help (small, adjacent spheres: nulling the neighbors
removes most of the own leadfield).
Figures: `R/overtProd_prodDiff/sweep/report/matrices_custom.png`,
`pareto_custom.png`, `R/overtProd_prodDiff/ctf_maps_dSPM_custom.png`.

**5. First translation-arm decoding (overtProd/prodDiff, n = 20, dSPM =
sLORETA, eLORETA): no regional selectivity at lobe scale either.** Every
ROI peaks at speech onset (Temporal 0.617, Visual 0.609, IFG 0.607, SFG
0.573, SPL 0.550); no speech ROI beats Visual in any cluster or fixed window
(phonological-loop window: Visual 0.529 vs 0.506–0.515). Because the ROIs
are resolvable from each other (finding 1), pairwise leakage cannot explain
this; a common source outside all five (sensorimotor strip is the candidate)
can, since ~65 % of each ROI's energy comes from outside it.
Figures: `D/overtProd/dSPM/sensorEquiv/vertex_selectkbest/raw/no_pseudo/40_5/logistic_tuned/_onset_latency/prodDiff_onset_latency.png`,
`.../figures/overtProd_dSPM_<ROI>_SW40_5_CLUSTER.png`.

**6. No subject is reject-worthy at the inverse stage.** All 20 sit within
×0.5–×2 of the group-median noise gain in both tasks; configuration
rankings agree with the group (Spearman median 0.87–0.92, min 0.69).
EEGPROD4014 and 4020 have the noisiest overtProd baselines (broadest CTFs,
spread 24 mm vs 16 mm; unremarkable in perception); 4001/4004/4005 the
quietest; 4019 mildly off in both tasks.
Figures: `R/subject_qc/overview.png`,
`R/overtProd_prodDiff/subject_qc/qc_heatmap_sensorEquiv.png`,
`qc_noise_gain_sensorEquiv.png`, `qc_config_agreement_sensorEquiv.png`.

**7. Sensitivity check without 4014/4020 (overtProd/prodDiff, dSPM):** peak
accuracies rise by ~0.01 in every ROI (Temporal 0.617→0.628, IFG
0.610→0.622, SFG 0.574→0.582, SPL 0.550→0.558), the ROI ordering and the
cluster structure are unchanged. Outputs in `.../logistic_tuned/prodDiff/
excl_4014_4020/` and `.../figures/excl_4014_4020/` (via
`source_stats_viz.py --exclude-subjects 4014 4020`).

**8. Whole-brain leakage maps (where the cross-talk energy comes from;
`exploratory/ctf_leakage_map.py`, both tasks agree).** Most of each ROI's
leaked energy is its own neighborhood: Temporal from TE1p/TE1m/OP4/PF,
Inferior Frontal from 46/p9-46v/8Av/8C/6r, Superior Frontal from 3b/4/SCEF/
8Av, Superior Parietal from VIP/area 1/PFm/area 2. The sensorimotor strip
(areas 1, 4, 3b, 3a, 2) is the only region that feeds *all four* speech ROIs
and tops the common-leak ranking, but it is a modest share: 10–16 % of each
speech ROI's energy, 20 % of all leaked energy. Right hemisphere: 1–4 % for
Temporal/IFG, 10–18 % for SFG/SPL (midline ROIs). The Visual control takes
28–33 % of its energy from the *right* early visual cortex and < 0.5 % from
the sensorimotor strip, so Visual decoding as well as the speech ROIs
cannot be sensorimotor leakage; it is either genuine visual discrimination
of the stimuli or a signal that reaches every ROI through the sensors
rather than through cortex (articulation EMG, eye movements).
Figures: `R/overtProd_prodDiff/leakage_map/brain_sensorEquiv_mn_dSPM_snr3_d0.8_l0_shrunk.png`
(per-ROI leak sources, total, common; lateral + medial),
`parcels_sensorEquiv_mn_dSPM_snr3_d0.8_l0_shrunk.png` (top-20 parcels
stacked by receiving ROI), `summary.txt` and the parcel CSVs in that folder.
DeFleCT with nulls on the other ROIs (same folder, `*_deflect_*`) lowers the
sensorimotor share to 6–10 % and leaves the neighborhood leak as is.

## Decisions and what runs next

* Keep dSPM/eLORETA (SNR 3) as the production inverse; drop LCMV.
* Run the sharper variant as a comparison arm on sensorEquiv, both tasks:
  `METHODS="dSPM_snr10" nohup bash exploratory/run_translation_arm.sh > logs/translation_arm_dSPM_snr10.log 2>&1 &`
  (own caches/results under `{method} = dSPM_snr10`; resumes like the
  production arm). Whether halved cross-talk at doubled noise helps decoding
  is empirical; the resolution numbers cannot say.
* DeFleCT with nulls is the candidate for the GC stage (virtual-channel
  path), to be wired after the decoding comparison.
* Group stats: report n = 20 and n = 18 (without 4014/4020) side by side.
* The sensorimotor check is done (finding 8): the strip is the one common
  source but only 10–16 % per ROI, so a DeFleCT null on areas 1/4/3b/3a/2 is
  a cheap, targeted addition for the GC stage; adding the strip as a sixth
  ROI is the alternative if its own time course is of interest. Neither
  explains the Visual control, which needs the sensor-level (EMG / ocular)
  check instead.
