# Exploratory analyses, sweeps, and figure scripts

Everything in this directory *consumes* the core toolbox (the modules and
runners at the repo root) but is not part of it: one-off investigations,
parameter sweeps, estimator comparisons, validation scripts, and figure
generators. Nothing at the root imports from here (the one exception is
`methods_paper/gc_config_decision.py`, which adds this directory to its path
for the MNE-comparison modules).

All Python scripts here put the repo root on `sys.path` themselves, and the
shell scripts `cd` to the repo root — so run everything from anywhere as
`python exploratory/<script>.py ...` / `bash exploratory/<script>.sh`.

New one-off, sweep, or figure code belongs here, not at the repo root.

## GC estimator validation & MNE comparison

- `validate_granger.py`, `validate_granger_conditional.py`,
  `validate_granger_statespace.py`, `validate_granger_normalize.py` —
  synthetic-ground-truth checks of the BSMART-port MVAR GC, its conditional
  and state-space variants, and the normalization arms.
- `granger_mne.py`, `run_granger_mne.py` — MNE-Connectivity-based GC pipeline
  (validated as directionally consistent with the BSMART port; not used for
  the final analysis).
- `compare_bsmart_vs_mne.py`, `compare_gc_mne_configs.py`,
  `run_gc_mne_compare.sh`, `compare_damera_bsmart_vs_mne.sh` — the
  BSMART-vs-MNE comparison battery.

## GC sweeps & batch drivers

- `run_gc_window_order_sweep.sh` — the 352-config window x order sweep.
- `run_gc_normalize_ab.sh` — none / demean / zscore normalization A/B.
- `run_gc_theta_config.sh` — theta-sized config trial (superseded by the
  theta arm of the root-level `run_gc_final.sh`).
- `run_gc_all.sh`, `rerun_GC.sh` — older batch drivers.
- `run_gc_stats_figs.sh` — stats + review figures for `run_gc_final.sh`
  results; takes the same knobs (`ORDER`, `WINS`, `TARGET_FS`, `NORMALIZE`,
  `NPCS`, `NPCS_ROI`), one pass per window.
- `plot_gc_window_sweep.py` — overlays the window lengths of a
  `run_gc_final.sh` sweep per task x contrast x pair (from the stats CSVs),
  plus an overview of the pointwise t-test in a chosen time range.
- `compare_normalize_arms.py`, `compare_gc_conditions.py` — cross-run
  aggregation and comparison of sweep outputs.
- `report_gc_diagnostics.py` — per-run diagnostics (stability, consistency).
- `clean_nan_bivariate_gc.py` — one-off cache cleanup after the NaN bug.

## GC routes (triple-wise) and sensor-space arms

- `run_granger_routes.py`, `granger_routes_stats.py`,
  `plot_granger_routes.py`, `plot_granger_story.py`, `run_gc_routes_batch.sh`
  — the conditional "routes" analysis (sweep verdict: conditioning on a third
  ROI changes nothing; final analysis is pairwise).
- `run_granger_sensor.py` — sensor-space GC baseline.

## GC figures

- `plot_gc_edge_review.py` — per-edge 3-panel review (Fxy / Fyx / dTRGC) with
  pointwise bars and FWER cluster stars; the final-analysis review figure.
- `plot_gc_subject_lines.py` — pointwise stats under group-scalar vs
  own-baseline referencing (supplement to the review figures).
- `plot_gc_fixpc_compare.py` — FIXPC1 vs FIXPC-k (or a per-ROI policy) on the
  same edge, bands x (Fxy / Fyx / dTRGC).
- `plot_gc_edge_extent.py` — how far the leading-edge GC inflation reaches into
  the epoch, per band and task; used to place the stats baseline.
- `plot_gc_pathway_timecourses.py` — theta-arm pathway figure.
- `plot_gc_sweep_summary.py`, `plot_gc_sweep_headline.py` — sweep grids.
- `plot_gc_pairs.py` — pair time-course panels.
- `plot_gc_baseline_choice.py`, `plot_gc_baseline_sensitivity.py`,
  `plot_gc_story5_baseline.py` — baseline-window investigations.
- `plot_lcmv_order_sweep.py`, `plot_sampling_rate_tradeoff.py` — LCMV order
  and sampling-rate analyses behind the fs200 decision.

## Source-space strategy 2026-10 (resolvability + translation arm)

See `SOURCE_SPACE_STRATEGY_PLAN.md` in this directory for the rationale,
the workstation run order and the decision rule.

- `resolution_analysis.py` — ROI-level cross-talk / point-spread of the
  pipeline's own inverses (dSPM, sLORETA, eLORETA, LCMV) for the custom and
  `sensorEquiv` ROI sets, plus the sensor pseudo-channels on the same
  footing; answers "can 64 channels separate these ROIs at all?".
- `run_translation_arm.sh` — the sensor-space manuscript analysis re-done
  literally in source space: `sensorEquiv` atlas (4 pseudo-channel ROIs +
  Visual control), minimum-norm inverses, raw, 500 Hz, sensor BSMART GC
  settings, stats on the full span and on `config.GC_WINDOWS_OF_INTEREST`.
- `decode_onset_latency.py` — decoding TIMING per ROI against a null ROI:
  jackknife onsets, ROI-vs-control cluster test, ROI x time RM-ANOVA,
  fixed-window accuracy.
- `compare_gc_leakage_arms.py` — raw vs leakage-corrected arms of one GC
  config (overlays, similarity, fixed-window tests).
- `resolution_sweep.py` — inverse-settings sweep scored with the same CTF
  machinery (the five levers of `INVERSE_OPTIMIZATION_PLAN.md`: DeFleCT ROI
  filters, λ²/depth, noise covariance, LCMV conditioning, loose
  orientation); adds `noise_gain` and `k_pc1_frac` so configurations are
  comparable on the noise axis. Workstation compute; per-subject CSV/npz.
- `resolution_sweep_report.py` — reads the sweep CSVs: one-knob factor
  curves, Pareto (noise vs cross-talk), overlap matrices production vs best,
  rankings, marginal effects, recommendation; `--brain-maps` for the best
  configurations. Runs locally.
- `resolution_subject_qc.py` — per-subject QC from the sweep CSVs (the
  forward model is shared, so subject spread = that subject's noise / data
  covariance): robust-z heatmap of the production dSPM/eLORETA/LCMV metrics,
  noise-gain per ROI, LCMV collapse detector, every subject's |cos| matrix,
  configuration-ranking agreement; `subject_qc/` per task + cross-task
  `overview.png`. Runs locally.
- `ctf_leakage_map.py` — whole-brain answer to "where does each ROI's
  cross-talk energy come from": energy-fraction maps from the saved sweep
  CTF vectors, own ROI masked, summed and geometric-mean ("common") across
  the speech ROIs, aggregated onto the 360 HCPMMP1 parcels (ranked table,
  stacked bars, lateral+medial brain maps). Decides DeFleCT nulls / extra
  ROIs without guessing. `leakage_map/` per task. Runs locally.
- `run_resolution_sweep.sh` — workstation driver: baseline
  `resolution_analysis.py` + `resolution_sweep.py` + report for both tasks.
- `INVERSE_OPTIMIZATION_PLAN.md` — the plan behind the three scripts above.
- `SOURCE_SPACE_FINDINGS_2026_10.md` — one-page record of the findings so
  far (resolution, sweep, first decoding, subject QC), each with its figure.

## Decoding exploration

- `explore_hyperparams_summary.py`, `explore_viz_stats.py` — summaries and
  visualization of `explore_decoding.py` sweeps.
- `explore_to_decode.py` — promote explore results into the decode layout.
- `migrate_svm_to_decode.py` — one-time result-directory migration.
