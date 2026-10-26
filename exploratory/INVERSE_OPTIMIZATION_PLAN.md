# Inverse-model optimization: choose the source estimation settings on resolvability, before decoding and GC

_2026-10-25. Follows `SOURCE_SPACE_STRATEGY_PLAN.md`. The first translation-arm results (overtProd / prodDiff, dSPM / eLORETA / sLORETA) showed no regional selectivity at lobe scale either: every ROI peaks at speech onset, the Visual control decodes as well as or better than the speech ROIs, and no ROI beats it in time. Before spending another ~1400 min per decode cell, the inverse stage is optimized on a quantity that costs minutes per subject: the ROI cross-talk of the inverse itself._

## 1. Logic

The inverse cannot add information to 64 channels, but it decides how the information is assigned to regions. `resolution_analysis.py` measures that assignment directly (CTF/PSF of the pipeline's own operators; see its docstring). Every candidate setting below changes the resolution matrix, so each can be scored on the same three numbers **without decoding anything**:

| metric | meaning | want |
|---|---|---|
| speech \|cos\| | expected zero-lag correlation between two speech ROI estimates under independent white sources | low (0 = separable) |
| se_own | share of an ROI's cross-talk energy coming from its own vertices | high |
| noise_gain | sensor-noise SD reaching the ROI time course per unit own-source gain, same units for every configuration | low |
| k_pc1_frac | share of the ROI's variance in one component (1.0 = the beamformer collapse) | < 0.9 |

Sharper filters always cost noise, so the decision is a trade-off read off a Pareto plot, not a single winner. Two scripts implement it: `resolution_sweep.py` (workstation, computes) and `resolution_sweep_report.py` (local, reads CSVs, draws figures, ranks).

## 2. The five levers and what the sweep does with each

| # | lever | what changes | knob in `resolution_sweep.py` | expected effect |
|---|---|---|---|---|
| 1 | **DeFleCT ROI filters** (Hauk & Stenroos 2014) | the ROI weights: instead of PC1 of whatever the inverse puts in the ROI, a sensor filter is *designed* with unit gain on the ROI's dominant patch topography, hard nulls on the other ROIs' topographies (1 or 2 SVD components each), minimum response over the rest of cortex plus a noise penalty | family `deflect`: `snr` {1, 3, 10}, `null_comp` {1, 2}, `nulls` {other_rois, none} | pairwise \|cos\| between the ROI patch means → 0 by construction; noise_gain rises; the question is by how much |
| 2 | **regularization + depth** | λ² = 1/SNR² (sharpness vs noise), depth exponent (where CTF peaks sit) | family `mn`: `snr` {1, 2, 3, 5, 10}, `depth` {0, 0.4, 0.8} (`--grid full` adds the factorial) | lower λ² sharpens PSF, raises noise; depth moves peaks toward/away from the surface |
| 3 | **noise covariance** | the whitening inside the operator | `noise_cov` {shrunk (production), auto, ledoit_wolf, adhoc, fullepoch, identity} via `inverse_pipelines.make_noise_cov` | a 100 ms baseline is a small sample for 64×64; better-conditioned whitening changes K; `fullepoch` is the standard choice without a clean baseline |
| 4 | **LCMV conditioning** | beamformer regularization, weight normalization, orientation, data-covariance window, noise covariance; `k_pc1_frac` flags the spatial collapse that produced the 2026-07 washout (rank is already data-driven, see `inverse_pipelines._compute_data_rank`) | family `lcmv`: `reg` {0.01, 0.05, 0.1, 0.3}, `weight_norm` {unit-noise-gain, nai, none}, `pick_ori` {max-power, normal}, `data_cov` {task, full}, `noise_cov` as above + none | the pilot's LCMV (\|cos\| 0.6, no CTF peak inside its ROI) is the collapse signature; a conditioned beamformer is the only method that actively suppresses correlated neighbours |
| 5 | **loose orientation** | fixed normal (production) vs loose {0.2, 0.4, 0.6}; applied with `pick_ori='normal'` so the time courses stay linear (magnitude output would wreck GC) | family `mn`: `loose` | relaxes the fsaverage-normal assumption; usually less spurious spread on sulcal walls |

The default `--grid star` varies one knob at a time around the production settings (dSPM / eLORETA: SNR 3, depth 0.8, fixed, shrunk; LCMV: reg 0.05, unit-noise-gain, max-power, task window, shrunk). That is ~55 configurations per subject and about 10 min per subject with the ico-5 forward. sLORETA is not swept: it equals dSPM up to a per-vertex scale, which PC1 weights and the standardizing classifier both remove.

What the sweep does **not** change: the forward model (template anatomy, template biosemi64 positions, 3-layer BEM). There are no digitized electrode positions, so that lever is closed; it is listed here so the limitation is explicit.

## 3. Workstation

```bash
cd code/source_estimation && git pull && conda activate mne
nohup bash exploratory/run_resolution_sweep.sh > logs/resolution_sweep_driver.log 2>&1 &
#   = baseline resolution_analysis (dSPM/sLORETA/eLORETA/LCMV x custom/sensorEquiv, brain maps)
#   + resolution_sweep (mn/lcmv/deflect star grid) + report, for overtProd/prodDiff and perception/percDiff
#   knobs: N_JOBS=8 GRID=star|full ATLASES="custom sensorEquiv" STEPS="baseline sweep report" BRAIN_MAPS=0
```

Outputs under `derivatives/source_estimation/RESOLUTION_ANALYSIS/{task}_{stim}/`: `summary.txt` + `pattern_overlap.png` + `ctf_maps_*.png` (baseline), and `sweep/report/` with `factors_{atlas}.png`, `pareto_{atlas}.png`, `matrices_{atlas}.png`, `ranking_{atlas}.csv`, `deltas.csv`, `summary.txt`. Total size is small (CSV + a few hundred MB of CTF vectors), so it stays on the project drive and syncs back for local reading. The report alone re-runs locally:

```bash
python exploratory/resolution_sweep_report.py --task overtProd --stim-class prodDiff [--max-noise-ratio 2] [--brain-maps]
```

## 4. Reading the results

1. `factors_sensorEquiv.png` — per knob, does any level move speech \|cos\| down or se_own up at an acceptable noise ratio? Marginal effects are tabulated in `deltas.csv` / `summary.txt`.
2. `pareto_sensorEquiv.png` — the production dSPM star is the reference; candidates are points left of the dashed noise budget (default 2× production) and below the star. DeFleCT points should sit at \|cos\| ≈ 0; their x-position says what that costs.
3. `matrices_*.png` and the `ctf_maps_*` of the best configurations — sanity-check that the numbers correspond to CTFs that actually sit in their ROIs.
4. Repeat for `custom`: if the spheres become separable under some setting, the localizer ROIs come back into play; if not, the lobe-scale ROIs remain the unit of analysis.
5. Compare tasks: perception has shorter epochs and a 100 ms baseline; a setting that only wins on one task is suspect.

Decision rule (default; override with reasons): take the lowest speech \|cos\| configuration within the noise budget whose se_own is not below production and whose k_pc1_frac is below 0.9. If no configuration beats production by a meaningful margin (Δ\|cos\| < 0.02 at lobe scale), keep dSPM/eLORETA as they are and treat the resolution figure as the chapter's explanation of the source results.

## 5. Porting a winner into the pipeline

* Levers 2, 3, 5 are already parameters of `inverse_pipelines.make_minimum_norm_inverse(depth=, loose=, noise_cov_method=)`; lever 4 of `make_lcmv_filters(reg=, weight_norm=, pick_ori=, data_cov_window=, noise_cov_method=)`. Production defaults are unchanged. Wiring a winner means adding the chosen values to `config.py` (next to `LAMBDA2`) and passing them from `run_minimum_norm_lowram` / `run_lcmv_lowram`, plus — if `loose > 0` — applying with `pick_ori='normal'` and giving the new arm its own cache tag so it never collides with the existing caches.
* Lever 1 (DeFleCT) is a *virtual-channel* method: one time course per ROI computed directly from the sensor epochs, so it bypasses the vertex caches. It fits the GC stage (which uses one PC per ROI anyway) as a new `run_granger.py` input path; for decoding it provides one feature per ROI, i.e. a different, lower-dimensional analysis. Implement only if the sweep shows DeFleCT's noise cost is acceptable.
* Any change to the inverse requires re-localizing (2000 Hz for decoding, 500 Hz for GC) under a new cache tag; the translation-arm driver handles the rest.
* **Lever 2 is wired (2026-10-26):** a minimum-norm method string may carry an `_snr<N>` suffix (`config.parse_method`): `--method dSPM_snr10` runs with λ² = 1/100 and uses `dSPM_snr10` as the `{method}` path segment everywhere (caches, decode CSVs, latency, GC), so `METHODS="dSPM_snr10" bash exploratory/run_translation_arm.sh` is the whole port. Results and reading: `exploratory/SOURCE_SPACE_FINDINGS_2026_10.md`.

## 6. Caveats

* CTF/PSF are properties of the operator and the forward model, not of the data; they say which ROIs **can** be separated, not whether the brain separates them. A configuration with low cross-talk and high noise can still decode worse.
* The CTF uses the fixed-normal leadfield as the source model even for loose operators; tangential source components are not scored.
* DeFleCT nulls are exact only for the patch topographies they were built from (1–2 SVD components per ROI); residual leakage from other spatial patterns within a nulled ROI remains and is what the metrics report.
* `noise_gain` uses the shrunk baseline covariance as a fixed reference for every configuration, including the ones built with a different noise covariance; this keeps the axis comparable.
