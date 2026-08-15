# MATLAB sensor-space GC vs this Python source-space pipeline

Read from `code/bsmart_GC/`:
`task_overtProd_bivariate_GC_pseudoChan_seed2target_parfor_v3_all_trials_combined.m`
(+ the `task_perception_..._v2_...` twin) for the computation, and
`production_pwgc_data_to_python.m` / `perception_pwgc_data_to_python.m` for the
banding and statistics.

**The estimator is the same. Almost everything feeding it is not.**

## Side by side

| | MATLAB (sensor) | Python (source) |
|---|---|---|
| signal | unweighted mean of 3 named electrodes | LCMV source estimate → ROI first PC, fixed spatial filter |
| reference | mastoid | average |
| sampling rate | 500 Hz | 200 Hz |
| window | 40 ms = **20 samples** | 40/60/80 ms = **8/12/16 samples** |
| model order | 10 | swept 2/4/6/10 (canonical 6) |
| normalisation | **none** | **zscore**: ERP + per-time-point ensemble SD |
| trials | all good trials pooled | split by stimulus class (4 task×contrast cells) |
| GC routine | BSMART `mov_bi_ga` | verified port, matches to 2.8e-16 |
| nodes | 3 seeds × 6 targets, sensor pseudochannels | 4 ROIs (awfa, ifc, pmc, tpc), all 6 pairs both ways |
| subjects | 20 | identical list, verified |

## The four differences that change numbers

**1. Samples per fit.** Both use a 40 ms window, but 40 ms is 20 samples at
500 Hz and only 8 at 200 Hz. Fitting order 10 on 8 samples is not the same
estimation problem as fitting it on 20. The sweep measured the consequence
directly: GC falls monotonically as the window lengthens (40 > 60 > 80 ms at
every order and band), which is finite-sample bias, so the MATLAB 40 ms values
sit at a different point on that bias curve than any Python cell. The nearest
match in samples-per-fit is the Python **80 ms** window (16 samples), not 40.

**2. Normalisation.** MATLAB passes the raw pseudochannel pair straight into
`mov_bi_ga` — no ERP removal, no demeaning, no scaling. Python defaults to
`zscore`. That is not a cosmetic difference: the A/B measured it inverting the
dominant band (theta → high beta at order 6) and cutting order-sensitivity
four-fold. MATLAB's setting corresponds to the Python `--normalize none` arm.

**3. Band edges.** theta agrees; the beta bands do not.

| | MATLAB | Python |
|---|---|---|
| theta | 4:7 Hz | [4, 8) → 4–7 |
| low beta | 13:20 | [12, 18) → 12–17 |
| high beta | 21:30 | [18, 30] |

18, 19 and 20 Hz are **low beta in MATLAB and high beta in Python**, and 12 Hz
enters low beta only in Python. Any "low beta" or "high beta" value is not
comparable across the two pipelines. (Python's alpha band has no MATLAB
counterpart — `Fxy_alpha_subj` is declared there and never assigned.)

**4. Spatial definition.** A sensor pseudochannel is an unweighted average of
three electrodes; a source ROI is a first principal component over vertices
after an inverse solution, optionally leakage-corrected. These are different
quantities, and the ROI sets only partly correspond (`awfa`≈temporal,
`ifc`≈inferior frontal, `pmc`≈motor, `tpc`≈parietal).

## Statistics

**MATLAB** does exactly one test, per band and direction:

```matlab
[~, p(t)] = ttest(Fxy_task(:, t), Fxy_subjAvg_baseline_mean, ...
                  'Alpha', 0.05, 'Tail', 'right');
```

A right-tailed one-sample *t* of the 20 subjects at each task window against a
**scalar** — the baseline-window mean of the subject-averaged curve. No
correction of any kind.

**Python** reproduces that exactly (`granger_stats.task_vs_baseline(test='ttest')`,
plotted by `plot_gc_pairs.py` into `_figures_pairwise_ttest/`) and adds two
things MATLAB has no equivalent for:

- a sign-flip **cluster permutation** over the task span with each subject
  referenced to its *own* baseline, which controls FWER across windows;
- **FDR/Bonferroni** across the whole edge × band family.

Both are reported separately, because on this data they disagree: 5–8 of 144
tests reach p<0.05 uncorrected (~7 expected by chance) and none survive FDR,
while the pointwise test finds `ifc→tpc` theta in perception at 59/125 windows
against a chance expectation of 6, with 0/125 in the reverse direction.

### Windows tested

| | MATLAB | Python |
|---|---|---|
| overtProd baseline | −1600 … −1502 ms | −1500 … −1400 ms |
| overtProd task | −1500 … 398 ms | −1400 … 320 ms |
| perception baseline | −200 … −100 ms | −200 … −100 ms |
| perception task | **0** … 598 ms | **−100** … 520 ms |

The baseline *rule* matches — the leading ~100 ms of the epoch, no edge guard
(the Python default is now 0). The absolute windows do not:

- **Python's overtProd axis begins at −1500 ms, MATLAB's at −1600.** The last
  window still ends at +400 = `PRODUCTION_TMAX`, so ~100 ms is missing from the
  START of the source-space epoch relative to `PRODUCTION_TMIN = -1.6`. Worth
  checking whether the ROI cache is cropped; it shifts the baseline 100 ms later
  than the sensor analysis.
- **Perception task start differs three ways.** MATLAB begins the task span at
  stimulus onset (`start_task_t = 0`). `granger_stats` derives it from the
  baseline end (−100 ms), so it tests 100 ms of pre-stimulus windows as task.
  And `config.GC_TASK_START` says −50 ms — but **`granger_stats` never reads
  `GC_TASK_START` or `GC_BASELINE_WINDOWS`**; only `run_granger_mne.py` and the
  methods-paper scripts do. Three definitions coexist; the stats path uses the
  derived one.
- Python stops the task span at `GC_TASK_END` (320 / 520 ms), MATLAB at
  398 / 598, so Python drops the trailing ~78 ms.

## If you want the two to be comparable

Closest Python configuration to the MATLAB analysis:
`--normalize none --win-ms 80 --order 10` (80 ms ≈ MATLAB's samples-per-fit),
band means re-derived on the MATLAB edges, and the pointwise *t* only. That is
a deliberately un-normalised, uncorrected analysis — reasonable as a
reproduction check, not as the primary result.
