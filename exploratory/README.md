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
- `plot_gc_pathway_timecourses.py` — theta-arm pathway figure.
- `plot_gc_sweep_summary.py`, `plot_gc_sweep_headline.py` — sweep grids.
- `plot_gc_pairs.py` — pair time-course panels.
- `plot_gc_baseline_choice.py`, `plot_gc_baseline_sensitivity.py`,
  `plot_gc_story5_baseline.py` — baseline-window investigations.
- `plot_lcmv_order_sweep.py`, `plot_sampling_rate_tradeoff.py` — LCMV order
  and sampling-rate analyses behind the fs200 decision.

## Decoding exploration

- `explore_hyperparams_summary.py`, `explore_viz_stats.py` — summaries and
  visualization of `explore_decoding.py` sweeps.
- `explore_to_decode.py` — promote explore results into the decode layout.
- `migrate_svm_to_decode.py` — one-time result-directory migration.
