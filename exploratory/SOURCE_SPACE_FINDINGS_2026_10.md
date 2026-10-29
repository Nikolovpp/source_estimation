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
* **CTF brain maps** (`ctf_maps_*.png`) show |CTF_i| on the fsaverage
  surface, one panel per ROI, each panel divided by its own maximum: 1 =
  the vertex that ROI listens to most, 0.5 = a vertex whose unit source
  enters the ROI time course at half that gain, < 0.05 not drawn. Compare
  the *shape* across panels, not the brightness. A tight map with its peak
  inside the ROI = well resolved; the LCMV maps paint half the hemisphere.
  The leakage maps (`leakage_map/brain_*.png`) use CTF² with the own-ROI
  vertices masked, again scaled to each map's max.
* **dSPM, sLORETA and eLORETA maps look identical because they nearly are.**
  dSPM and sLORETA are one minimum-norm kernel with different per-vertex
  scale factors (their PSF ratio is constant across ROIs at every vertex),
  so an ROI's CTF changes only through the within-ROI weights; eLORETA is a
  different kernel that lands in the same place at 64 channels. Normalized
  maps agree to cos 0.98–0.99 for the speech ROIs (mean |diff| 0.01–0.02,
  max 0.06–0.16); Visual is the exception (cos 0.93–0.95, eLORETA tighter).
  `ctf_maps_diff_dSPM-eLORETA_{atlas}.png` (`exploratory/ctf_method_diff_map.py`)
  shows the signed difference on a diverging scale.
* **|cos(CTF_i, CTF_j)|** = expected zero-lag correlation between ROI i and
  ROI j under spatially white sources; the quantity that matters for both
  decoding (shared signal) and TRGC (instantaneous mixing).
* **ctf_xtalk** (`crosstalk.png`, in %) = mean CTF_i² over ROI j's vertices
  divided by the mean over ROI i's own: the per-vertex source power in j
  that ROI i reports, relative to its own; > 100 % (LCMV) means it hears j
  better than itself. The sensor panel uses the same definition with the
  pseudo-channel's leadfield row as its CTF and the sensorEquiv ROI of the
  same name as its "own" vertices.
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

**9. Why dSPM and LCMV decode alike although their CTFs are nothing alike
(checked 2026-10-26 in `DECODE_source_space`).** overtProd/prodDiff, raw,
selectkbest, logistic_tuned, n = 20: LCMV/custom (July, post rank fix) peaks
0.55–0.59 at speech onset, dSPM/sensorEquiv peaks 0.55–0.62 at speech onset;
the accuracy curves of *any* LCMV ROI and *any* dSPM ROI correlate 0.83–0.97
over time, and within one arm the ROI curves correlate 0.90–0.93. Perception
(old matched arm, custom/vertex_pca/leakage, n = 18–20) is the same story at
chance level. Reason: a linear classifier is blind to the inverse. An ROI's
vertex time courses are the sensors passed through the ROI block of the
kernel, and that block spans only 9–16 sensor-space directions at 1 % of
the leading singular value (5–6 at 10 %) for every ROI; the leading
directions are the smooth topographies over that part of the head, and the
ROIs share them (50–80 % subspace overlap between any two ROIs, against
8–16 % for random subspaces). What the classifier sees is the onset-locked
sensor pattern projected into that shared subspace, under dSPM, eLORETA or
LCMV alike. The CTF decides where the signal is *attributed*, not whether
it is decodable, so resolution improvements change interpretation and GC,
and should not be expected to move decoding peaks.

**10. Translation arm complete, vertex dSPM vs DeFleCT (2026-10-29; both tasks,
both contrasts, n = 20; review figures in
`derivatives/source_estimation/TRANSLATION_ARM_REVIEW/`, made by
`exploratory/review_translation_arm.py`).** Arms: *vertex* = dSPM SNR 3,
200 selected vertices per ROI (sensitivity: the signal reaches the ROI);
*deflect* = `deflect_k3_null2_other_rois` at SNR 10, three virtual channels
per ROI with the two leading topographies of each other ROI nulled
(specificity: the signal lives in the ROI's own sensor directions).

*Decoding.* overtProd/prodDiff, vertex: every ROI peaks at speech onset
(Temporal 0.617, IFG 0.610, Visual 0.609, SFG 0.573, SPL 0.550; clusters from
−630…−260 ms to the end). DeFleCT keeps Temporal 0.566, IFG 0.572 and Visual
0.551 with the same timing but drops SFG (0.525) and SPL (0.526) to the
noise floor; paired loss at the vertex peak −0.04…−0.07 (p ≤ .009) except
SPL (−0.03, p = .06). overtProd/percDiff: vertex 0.53–0.56 at onset,
DeFleCT ≤ 0.524. perception/percDiff, vertex: all five ROIs decode at
175–200 ms (SPL 0.592, Visual 0.576, SFG 0.567, Temporal 0.562, IFG 0.540,
clusters ~80–440 ms); DeFleCT: only SPL survives (0.555, cluster 120–257 ms,
and the first ROI > Visual cluster of the whole source program, 130–223 ms,
p = .021); the others fall to 0.513–0.525. perception/prodDiff: chance in both
arms. Arm curves correlate 0.57–0.97 (production) and 0.33–0.95 (perception)
over time; the DeFleCT loss is largest for Visual in three of four cells.
Figures: `TRANSLATION_ARM_REVIEW/decoding/decoding_curves_{task}.png`,
`decoding_rois_{task}.png`, `decoding_deflect_minus_vertex_{task}.png`;
tables `decoding_summary.md`, `decoding_arm_comparison.csv`.

*Why Superior Parietal stands out in perception/percDiff.* Not auditory
leakage: per vertex, SPL's dSPM filter is 0.3 % as sensitive to right
auditory cortex and 1.4 % to the left Temporal ROI as to its own vertices
(`exploratory/ctf_region_shares.py`, `RESOLUTION_ANALYSIS/perception_percDiff/
ctf_region_{shares,gains}_sensorEquiv.csv`). What SPL does hear is the
peri-central cortex: 26 % per-vertex sensitivity to the left sensorimotor
strip and 26 % to the right SPL (17 % of its energy is right-hemispheric),
and the DeFleCT nulls, which target the other four ROIs only, leave that
untouched (22 % / 24 % after nulling). At the sensor level the percDiff
discrimination at 100–350 ms peaks over central/fronto-central electrodes
(F6, FC3, C3, FT8, C4, FC1 at 0.55, 17 subjects, 20 ms windows) while the
CPz/CP1/P1 pseudo-channel is the weakest of the four (0.525) — so the source
SPL is not reporting "its" electrodes; the template inverse attributes a
bilateral central component to the superficial dorsal parietal/postcentral
cortex, and of the five ROIs SPL (areas 5, 7PC, AIP border the postcentral
gyrus) is the one whose own leadfield spans it. Decisive test: the
`deflect_null4_other_rois_sm` variant (adds the sensorimotor-strip null, finding 11); if SPL
keeps its percDiff peak, it is parietal; if it falls to the others' level,
it was central.

*GC.* DeFleCT arm: theta GC rises task-vs-baseline in 17–20 of the 20
directed edges in all four cells, in both directions, including
perception/prodDiff where nothing decodes; mean Δ +0.06–0.07 against
+0.00–0.01 in the vertex arm, with the rise starting at stimulus/speech
onset and peaking ~250 ms in every pair — the evoked response entering the
3-channel blocks (sensor-faithful config = no ERP removal, 1 PC in the
vertex arm dilutes it). Net TRGC cancels it: 1–3 edges per cell at
uncorrected p < .05, none survive FDR, no consistent pair across arms or
contrasts. Vertex arm: 3–8 theta edges (perception/percDiff: IFG→Temporal,
IFG→SFG, SPL→Temporal, Visual→IFG survive FDR), nothing in beta, TRGC
nothing. The vertex-arm GC for overtProd/prodDiff is missing: the directory
holds one subject and its stats were computed with n = 1.
Figures: `TRANSLATION_ARM_REVIEW/gc/gc_matrix_{task}_{stim}.png` (overview),
`gc_trgc_*.png`, `gc_directed_*_{arm}.png`; tables `gc_edge_summary.csv`,
`gc_significant_edges.md`.

**11. "Null the adjacent cortex" is not possible with 64 channels
(2026-10-29).** The question was whether each left ROI's DeFleCT bank should
null its *neighbors* rather than the other ROIs of the set (the current
default never touches the right hemisphere; the nulls are the four other
left ROIs). Added `ring` / `ring_rois` null modes (every HCPMMP1 parcel that
borders the ROI on the fsaverage mesh — Temporal 13 parcels, IFG 13, SFG 15,
SPL 21, Visual 9 — each a null region; `deflect.build_ring_labels`) and
designed the banks on the template leadfield (`exploratory/ctf_region_shares.py
--designs ...`, identity noise covariance). Hard nulls on the ring destroy
the filter: own-energy 0.00–0.01 and noise ×6–600 for the four speech ROIs
(Visual, which has a distinct posterior field, survives at 0.44). The
neighbors' topographies *are* the ROI's topographies at this channel count,
so nulling them nulls the ROI — the mechanism behind finding 4 (nulls never
helped the adjacent custom spheres). Softer versions do not rescue it: the
ring as one region with a single nulled pattern halves IFG's own energy
(0.29 → 0.13), costs ×2–4 noise and *raises* the sensitivity to the
sensorimotor strip (SPL 24 → 34 %), because the filter pushes its passband
wherever it is not constrained. The one adjacent region that can be nulled
is the sensorimotor strip, an elongated patch with its own field: with 4
patterns per null region (`deflect_k3_null4_other_rois_sm`) SPL's per-vertex
sensitivity to the strip drops from 22 % to 5 % and SFG's from 17 % to 7 %,
at own-energy 0.22–0.23 and ×1.9–2.5 noise (2 patterns: 16 % / 10 % at
×1.5; 3 patterns no better than 2). So "one left ROI nulling its adjacent
ROIs" is realized as: other ROIs of the set + the sensorimotor strip, 4
patterns each. The ring modes stay in the code as a documented negative
result.

| template leadfield, SNR 10, k 3 | n nulls (SPL) | se_own SPL | noise × | SPL sens. to SM strip | SFG sens. to SM strip |
|---|---|---|---|---|---|
| no nulls | 0 | 0.32 | 1.0 | 24 % | 18 % |
| other_rois, 2 patterns (arm run) | 8 | 0.33 | 1.1 | 22 % | 17 % |
| ring, 1 pattern per parcel | 21 | 0.01 | 33 | — | — |
| ring as one region, 1 pattern | 1 | 0.25 | 1.6 | 34 % | 17 % |
| other_rois_sm, 2 patterns | 10 | 0.31 | 1.5 | 16 % | 10 % |
| other_rois_sm, 4 patterns | 20 | 0.22 | 2.5 | 5 % | 7 % |

## Decisions and what runs next

* Keep dSPM/eLORETA (SNR 3) as the production inverse; drop LCMV.
* Run the sharper variant as a comparison arm on sensorEquiv, both tasks:
  `METHODS="dSPM_snr10" bash exploratory/run_translation_arm.sh 2>&1 | tee logs/translation_arm_dSPM_snr10.log`
  (own caches/results under `{method} = dSPM_snr10`; resumes like the
  production arm). Whether halved cross-talk at doubled noise helps decoding
  is empirical; the resolution numbers cannot say.
* DeFleCT is wired (2026-10-26, `deflect.py`, feature mode `deflect`): K
  virtual channels per ROI, other ROIs' topographies nulled, caches ~1 % of
  the vertex caches. Run as its own arm, decoding *and* GC:
  `FEAT=deflect METHODS="dSPM_snr10" N_PCS=3 bash exploratory/run_translation_arm.sh 2>&1 | tee logs/translation_arm_deflect.log`
  Read the decoding of this arm as the specificity test (finding 9): the
  vertex arm is the upper bound (signal reaches the ROI), the DeFleCT arm the
  lower bound (signal is the ROI's own); a region is implicated when it
  passes both. `deflect_other_rois_sm` adds the sensorimotor-strip null.
* Group stats: report n = 20 and n = 18 (without 4014/4020) side by side.
* The sensorimotor check is done (finding 8): the strip is the one common
  source but only 10–16 % per ROI, so a DeFleCT null on areas 1/4/3b/3a/2 is
  a cheap, targeted addition for the GC stage; adding the strip as a sixth
  ROI is the alternative if its own time course is of interest. Neither
  explains the Visual control, which needs the sensor-level (EMG / ocular)
  check instead.
* 2026-10-29, after finding 10: (a) rerun the missing vertex-arm GC cell on
  the workstation — `STEPS="gc stats" METHODS="dSPM" TASKS=overtProd
  STIMS=prodDiff bash exploratory/run_translation_arm.sh 2>&1 | tee logs/translation_arm_dSPM_gc_fix.log`
  (resumes past EEGPROD4001) — and delete the stray copy
  `GC_source_space/dSPM/` (an overtProd/percDiff duplicate without the task
  level) plus the decode-tree copies under `GC_source_space/{task}/{method}/
  sensorEquiv/*/raw/no_pseudo/`; (b) run the sensorimotor-null variant (finding 11: the only
  adjacent null that works at 64 channels), both tasks, decoding + GC:
  `FEAT=deflect_null4_other_rois_sm METHODS="dSPM_snr10" N_PCS=3 bash exploratory/run_translation_arm.sh 2>&1 | tee logs/translation_arm_deflect_sm.log`
  — if Superior Parietal keeps its perception/percDiff peak with the strip
  nulled it is parietal, if it falls to the others' level it was central;
  (c) one leakage control per arm (user decision 2026-10-29): the vertex
  arm keeps directed GC + net TRGC (TRGC is its control); a DeFleCT arm has
  its nulls as the control and reports the directed GC only, which requires
  the evoked response removed first (finding 10: with `normalize none` the
  DeFleCT directed theta GC is the ERP in every edge). Rerun the DeFleCT GC
  as `FEAT=deflect METHODS="dSPM_snr10" N_PCS=3 NORMALIZE=demean TRGC=0
  STEPS="gc stats" bash exploratory/run_translation_arm.sh 2>&1 | tee logs/translation_arm_deflect_gc_demean.log`
  (own `order10_win40ms_fs500_demean_pc3` directories); the review script
  takes the arm as `deflect=dSPM_snr10:deflect:3:demean:gc`.
* 2026-10-29, sensorimotor strip as a sixth ROI and the custom-ROI
  comparison (user request). Atlas `sensorEquivSM` = sensorEquiv +
  `Sensorimotor` (areas 4/3a/3b/1/2, the same parcels as the DeFleCT `_sm`
  null; map in `ROI_maps/sensorEquiv/`). Nothing that has run contains its
  time course: in the `deflect_null4_other_rois_sm` arm it is only nulled.
  Cheapest route for the vertex arm (the five old ROIs' decoding is
  unchanged, so only the new ROI is decoded; GC needs all six in one cache,
  which at 500 Hz is small):
  `ATLAS=sensorEquivSM ROI_SUBSET=Sensorimotor GC_FS=2000 STEPS=localize METHODS=dSPM bash exploratory/run_translation_arm.sh 2>&1 | tee logs/sm_localize2000.log`
  (2000 Hz cache, strip only);
  `ATLAS=sensorEquivSM DEC_FS=500 GC_FS=500 STEPS=localize METHODS=dSPM bash exploratory/run_translation_arm.sh 2>&1 | tee logs/sm_localize500.log`
  (500 Hz cache, all six);
  `ATLAS=sensorEquivSM ROI_SUBSET=Sensorimotor STEPS=decode METHODS=dSPM bash exploratory/run_translation_arm.sh 2>&1 | tee logs/sm_decode.log`
  (one ROI, ~1/5 of a full decode per cell); then
  `python exploratory/merge_decode_rois.py --from-atlas sensorEquiv --to-atlas sensorEquivSM`
  and `ATLAS=sensorEquivSM STEPS="latency gc stats" GC_JOBS=15 METHODS=dSPM bash exploratory/run_translation_arm.sh 2>&1 | tee logs/sm_gc.log`
  (15 pairs). A DeFleCT arm on `sensorEquivSM` is one driver call
  (`ATLAS=sensorEquivSM FEAT=deflect ...`): with the strip inside the set the
  plain `other_rois` design nulls it for every other ROI, so it replaces the
  `_sm` variant. Review: `python exploratory/review_translation_arm.py --atlas sensorEquivSM`.
* Custom-ROI comparison (same arms on the six functional-localizer spheres,
  `ATLAS=custom CONTROL_ROI=vwfa-lh`): dSPM vertex caches for `custom` at
  2000 Hz do not exist (the custom arm was LCMV), so this is a full
  localize (both rates) + four decodes (~1700 min each) + GC; the DeFleCT
  arm is cheap. Expect from findings 3–4 and 11: owfa ↔ vwfa stay
  unresolvable (|cos| 0.35–0.51 in every inverse) and the DeFleCT nulls will
  partly cancel the spheres' own topographies, so the comparison mostly
  shows what lobe-scale composites buy over 10 mm spheres.
