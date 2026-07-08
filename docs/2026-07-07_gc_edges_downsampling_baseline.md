# GC edges, downsampling order, and the baseline bug — session notes (2026-07-07)

Reference write-up of an investigation into source-space Granger causality (GC)
edge artifacts, the downsample-vs-source-estimation ordering question, and a
task-vs-baseline bug in `granger_stats.py`. Covers findings, decisions, the
code changes made, and open questions.

## TL;DR

- **Sensor-space GC pseudo-channels** are hardcoded in `run_granger_sensor.py`
  (`SENSOR_PSEUDOCHANNELS`), transcribed by hand from the MATLAB BSMART
  scripts. Faithful to the 4 groups it copies, but a **partial** copy of the
  MATLAB `v3 all_trials_combined` design (drops 2 parietal target groups; uses
  all-pairs instead of the MATLAB 3-seed×6-target directed design).
- **GC task-vs-baseline stat** was a right-tailed one-sample **Student's t-test**
  (matches `production_pwgc_data_to_python.m` / MATLAB v4 figures). Added a
  `--test {ttest,signrank}` switch; MATLAB v3 figures used the Wilcoxon
  signed-rank instead. Wilcoxon is not simply "more conservative" — it's ~95%
  as efficient under normality but can be *more* powerful under skew/outliers.
- **GC sliding window is forward-looking / left-aligned** (timestamp = window
  start; value at t uses [t, t+win]). **Decoding is center-aligned.** They
  differ by half a window (~20 ms for a 40 ms window) — matters if comparing
  GC vs decoding latencies.
- **Edge artifacts** in GC have (≥) two causes: (1) scipy `resample_poly`'s
  default **zero-padding** injects a ringing transient at every epoch edge, and
  (2) the intrinsic **moving-window MVAR fit at the epoch boundary**. Fix #1
  addresses (1); (2) is handled by cropping the outermost windows.
- **EEGLAB `pop_resample`** uses a polyphase FIR (MATLAB `resample`) with
  **DC-hold edge padding** and, in the original pipeline, was run on the
  **continuous** data **before** epoching (so no per-epoch transient). The
  Python pipeline resampled per-epoch after epoching — the source of the
  reproduction artifact.
- **Downsample order (before vs after source estimation):** for a *linear dSPM*
  inverse the orders commute (equivalent). For an **LCMV beamformer they do
  not** — the spatial filter is built from the sampling-rate-dependent data
  covariance. Empirically (n=8): LCMV source time courses are **bimodal** —
  near-identical for ~half the subjects, badly divergent for the other half.
  GC tracks the sources (r=0.93). **Re-running source estimation on `fs_500`
  is NOT simply "unnecessary"** (an earlier n=1 claim, now retracted).
- **Baseline bug:** `granger_stats` reused the covariance `BASELINE_WINDOWS`
  as the GC reference. For overtProd that window (−1.6/−1.5) is out-of-range
  for the actual **−1.5..0.4 s** epoch, so the baseline collapsed onto the
  single leading-edge window → **~99% spurious significance**. Fixed by a
  decoupled `GC_BASELINE_WINDOWS` / `GC_TASK_START` / `GC_TASK_END`.

## Commits made today

| commit | summary |
|---|---|
| `9c25d01` | `granger_stats`: add `--test {ttest,signrank}` for task-vs-baseline |
| `b8cfe46` | `run_granger`: edge-pad resample to remove per-epoch transient (**fix #1**) |
| `4241b42` | `data_loader`: load `fs_500` data + derive sfreq from times array (**bug fix**) |
| `b70b49e` | add `compare_downsample_order.py`: A/B check for LCMV source GC |
| `d941862` | `granger_stats`: decouple GC task-vs-baseline window from covariance baseline |
| `78943f3` | `granger_stats`: add GC task-end crop to drop trailing MVAR edge window |

---

## 1. Sensor-space GC pseudo-channels

`run_granger_sensor.py` does **not** discover the MATLAB pseudo-channels — the
dict `SENSOR_PSEUDOCHANNELS` is hardcoded (Temporal `FT7/T7/TP7`,
Inferior_Frontal `F5/FC5/FC3`, Superior_Frontal `FCz/FC1/F1`, Superior_Parietal
`CPz/CP1/P1`). These match the active (uncommented) seed/target definitions in
`code/bsmart_GC/task_overtProd_bivariate_GC_pseudoChan_seed2target_parfor_v3_all_trials_combined.m`,
but it is a **partial** reproduction:
- The MATLAB v3 script ran **3 seeds × 6 targets** (directed); two target groups
  are dropped in Python ("superior parietal candidate #2" `CP1/CP3/P1`,
  "inferior parietal" `CP5/P5/P7`).
- Python computes symmetric all-pairs GC among 4 channels; MATLAB was directed
  seed→target.
- Earlier MATLAB variants (v1/v2, perception) used a single seed vs all 61
  individual channels — a different design entirely.

## 2. GC statistics — t-test vs Wilcoxon

The task-vs-baseline test is a **right-tailed one-sample test** of each task
time-point's across-subject GC against a **scalar** baseline (the mean over the
baseline windows of the subject-averaged GC). Python (`granger_stats.task_vs_baseline`)
uses `scipy.stats.ttest_1samp(..., alternative='greater')`, matching MATLAB
`ttest(x, m, 'Tail','right')` in `production_pwgc_data_to_python.m` and the v4
figure scripts.

The **v3 figure scripts used Wilcoxon signed-rank** (`signrank(..., 'tail','right')`)
instead — same one-sample/right-tailed/vs-scalar-baseline design, differing only
in parametric vs non-parametric. Added `--test {ttest,signrank}` to
`granger_stats.py`; outputs are tagged with the test name so both can coexist.

**Conservativeness:** parametric-vs-nonparametric and conservative are separate
axes. Under normality the t-test is optimal and Wilcoxon is ~95.5% as efficient
(ARE 3/π) — marginally more conservative. Under skew/outliers Wilcoxon can be
*more* powerful (ARE ≥ 0.864 always, unbounded above). At small n the discrete
signed-rank is mildly conservative. For our non-negative, right-skewed GC with
n≈20, which flags more time-points is empirical — run both and compare.

## 3. Sliding-window convention

- **GC** (`granger.py` / `run_granger.py`): forward-looking / **left-aligned**.
  `window_ms = win_times[starts]` — the window START time. GC at t is computed
  from data in **[t, t+win_ms)**. The GC time axis is truncated by one window
  at the right (last window start = tmax − win_ms).
- **Decoding** (`decoding.py:prepare_windowed_data`): **center-aligned**.
  `window_center_ms = (start_time + end_time)/2`. Value at t uses
  **[t−win/2, t+win/2]**.

Consequence: a GC curve and a decoding curve on the same data are offset by
~win/2 (20 ms for a 40 ms window). GC appears to *lead* decoding by that amount
purely from labeling — align before comparing latencies.

## 4. Edge artifacts and the resample fix (fix #1)

The GC results show sharp excursions at both epoch ends (empirically the
global-max GC sat at the 2nd window). Causes:

1. **Resample zero-padding.** `run_granger.resample_channels` called scipy
   `resample_poly` with the default (zero) padding, so the polyphase FIR sees a
   step to zero at each epoch edge → ringing. **Fix #1** (commit `b8cfe46`) pads
   each signal (`reflect`, EEGLAB-style, with `edge` fallback) before the FIR
   and crops after; output length and interior are unchanged. On a DC-signal
   test the edge deviation dropped from **1.89 → 0.0001** (~1e4×).
2. **Intrinsic MVAR-at-boundary.** The order-10 MVAR fit on a 40 ms (20-sample)
   window sitting at the epoch cut is unreliable regardless of resampling. Fix
   #1 does **not** remove this — it is handled by cropping the outermost windows
   (see §6, task-start / task-end).

### EEGLAB `pop_resample` (verified from source, EEGLAB 2024.0)
- Uses MATLAB Signal Processing Toolbox `resample()` — a **polyphase FIR** with
  EEGLAB's own Kaiser-windowed-sinc anti-alias kernel (`firws`, β=5, same β as
  scipy's default).
- Pads each segment with **DC-hold** (repeat first/last sample) by ~half the
  filter length, then crops — specifically to suppress edge transients
  (`myresample`, Widmann; bug 1017/1757).
- Explicitly **warns against resampling epoched data** ("due to anti-aliasing
  filtering").
- In `EEGPreProc_speechProd_perceptionV13.m` the resample (line 405) runs on the
  **continuous** `EEG` *before* `pop_epoch` (line 460+). So the original MATLAB
  GC had no per-epoch resample transient — the artifact is purely a property of
  the Python reproduction, which resamples per-epoch after epoching.

## 5. Downsampling order vs source estimation

Question: does it matter whether you downsample 2048→500 Hz **before** the
inverse (pipeline A) or **after**, on the per-epoch ROI time courses (pipeline
B, the current `run_granger` path)? Both end at 500 Hz.

**Deep-research verdict (cited synthesis):**
- **dSPM/MNE (linear):** the inverse is a fixed per-sample operator
  `s(t)=M·x(t)`; it commutes with linear resampling. On 0.1–30 Hz data (≪ 250 Hz
  Nyquist) the orders are effectively equivalent. Aliasing is handled either way.
- **Beamformers (LCMV/DICS):** the order **genuinely matters** — the spatial
  filter `W=(LᵀC⁻¹L)⁻¹LᵀC⁻¹` is built from the data covariance, which depends on
  bandwidth and sample count. Recommendation in the literature: downsample
  *before* beamforming, or compute the covariance on identically-preprocessed
  data (Westner et al. 2022; MNE LCMV tutorial; Gross et al. 2013).
- No published head-to-head A-vs-B for source GC exists.

Our LCMV (`inverse_pipelines.run_lcmv_lowram`) uses `pick_ori='max-power'`,
`weight_norm='unit-noise-gain'`, `reg=0.05`, data_cov from the task window,
shrunk noise_cov — fully data-adaptive, so the dSPM commutation argument does
**not** apply.

### A/B experiment (`compare_downsample_order.py`, commit `b70b49e`)
Pipeline A = LCMV on the `fs_500` continuous-resampled epochs; pipeline B = LCMV
on `fs_2000` then resample the ROI courses to 500 Hz (with fix #1). Same forward
model, ROI labels, LCMV settings — only the covariance sampling rate differs.
Includes a **matched white-noise control** (perturb A's TCs to the same
magnitude as the A-vs-B difference, recompute GC).

**Result at n=8 (overtProd/prodDiff, HCPMMP1, 4 ROIs) — corrects an initial
atypical n=1 (subject 4001) reading:**

| subject | trials | source-TC \|r\| | GC A-vs-B r |
|---|---|---|---|
| 4001 | 206 | 0.996 | 0.528 |
| 4003 | 222 | 0.986 | 0.720 |
| 4009 | 220 | 0.978 | 0.472 |
| 4005 | 171/170 ⚠ | 0.813 | 0.709 |
| 4006 | 176 | 0.258 | GC failed (LinAlgError) |
| 4008 | 128 | 0.241 | −0.002 |
| 4007 | 218 | 0.199 | 0.023 |
| 4004 | 147 | 0.167 | 0.003 |

- Source time courses are **bimodal**: near-identical for ~half the subjects
  (|r| 0.98–0.996), badly divergent for the other half (|r| 0.17–0.26,
  `frac_diverge`=1.00 — every epoch differs in shape, not just sign).
- **GC tracks the sources**: per-subject corr(source |r|, GC r) = **0.93**.
- Matched-noise baseline r consistently *below* A-vs-B r → the order effect is
  milder than an equivalent white-noise perturbation (GC is fragile to any
  sub-percent change), but for the divergent subjects the sources genuinely
  differ.
- Subject 4006's GC failed entirely (MVAR covariance not positive-definite).

**Interpretation / two candidate mechanisms (unresolved):**
- (a) genuine max-power **orientation flips** (→ downsample *before* beamform,
  per literature); vs
- (b) the `fs_500` covariance is more **marginally conditioned** (~14
  samples/channel at 500 Hz vs ~56 at 2048; 4006 failed) → the low-rate
  covariance is the problem, keep the 2048 Hz one.

These point in opposite directions, so **which order is preferable is open**.
The earlier "sources equivalent, `fs_500` re-run unnecessary" is **retracted**.

## 6. The baseline bug and the fix

Root cause (verified across all 20 subjects): the overtProd epoch is
**−1.5 to +0.4 s** (filename `_-1.5_0.4_-1499_-1400_`, times −1500..399 ms), not
−1.6. `config.PRODUCTION_TMIN=-1.6` and `BASELINE_WINDOWS['overtProd']=(-1.6,-1.5)`
are **stale** relative to the data. Perception is −0.2..0.6 (config-correct).

`BASELINE_WINDOWS` serves the **inverse noise-covariance window**, but
`granger_stats` **reused it** as the GC task-vs-baseline reference. Because
(−1.6,−1.5) is out-of-range for the −1.5 epoch, the baseline mask matched only
the single −1500 ms window — the leading-edge artifact (the lowest point). So
the "baseline" was ~0.02 vs an interior GC of ~0.09 → **99–100% of windows
"significant"**. The widespread significance was a baseline artifact.

### Fix (commits `d941862`, `78943f3`)
New, GC-specific config decoupled from the covariance window:

```python
GC_BASELINE_WINDOWS = {'perception': (-0.150, -0.050), 'overtProd': (-1.450, -1.350)}
GC_TASK_START       = {'perception': -0.050,           'overtProd': -1.350}
GC_TASK_END         = {'perception':  0.520,           'overtProd':  0.320}   # tmax - 2*win
```

`granger_stats.py` uses these by default, with CLI overrides
`--baseline-start / --baseline-end / --task-start / --task-end` (all in seconds).
The task window is `[task_start, task_end]`; the leading segment (epoch start →
baseline) and the trailing boundary window are excluded from both baseline and
task. Figures shade the baseline window and mark the task start/end.

**Validation (existing overtProd LCMV/custom caches, awfa-lh→ifc-lh):** baseline
0.02 (edge) → **0.09 (interior level)**; significant windows **99% → ~0%**;
tested windows now `[−1350, +318]` ms (trailing +356..+360 spike excluded). So
this edge shows **no** task-related GC increase once compared to a proper
reference — expect this to change many "results."

**Design caveat:** overtProd is production-onset-locked; −1.45/−1.35 is still
~1.4 s pre-onset but may already be task-active (planning), so it is not a true
*rest* baseline. If GC is tonically elevated across the epoch, a task-vs-baseline
test will legitimately find little. Choose the window per the design (flags).

## 7. Bug found & fixed: `data_loader` sfreq

The EEGLAB `.mat` exports **omit the `srate` field**, so `data_loader` fell back
to a hardcoded 2048 Hz and, on conflict, *kept* it — silently mislabeling any
non-2048 file (e.g. the 500 Hz set) as 2048 Hz. Fixed (commit `4241b42`) to
trust the (authoritative) times array when it disagrees with the stored/fallback
rate. The 2048 files are unaffected (no conflict). Also added an `fs` argument
(default 2000) to `load_subject_epochs` and the two data-path helpers to load
the `fs_500` continuous-resampled files.

## 8. Data facts established

- **Preprocessed coverage:** 20/20 subjects for `perception` and `overtProd`,
  each at both `fs_2000` and `fs_500` (`derivatives/EEGLAB/<task>/<subj>/average_ref/.../fs_{2000,500}/`).
- **Epochs:** overtProd −1.5..0.4 s (baseline −1.499..−1.400 in preprocessing);
  perception −0.2..0.6 s. Native rate 2048 Hz; `fs_500` = continuous-resampled.
- **Source estimation was run only on `fs_2000`.** Source ROI caches live under
  `DECODE_source_space_timeseries` (and external `maxlab` drives, not mounted here).
- **LCMV source-GC grid** (from `GC_source_space/`): 2 tasks × 2 stim-classes
  (prodDiff, percDiff) × 2 atlases (custom, HCPMMP1) × `vertex_selectkbest` ×
  {raw, leakage_corrected}.
- **Timing estimate to redo LCMV source estimation on `fs_500`:** ~4–5 min per
  inverse; ~80 distinct inverses minimum (subj×task×stim), or up to ~320
  as-structured (per atlas/leakage). ~1.5–3 h wall-clock at `--n-jobs 8`, or
  ~45 min with a runner tweak that extracts both atlases/leakage from one inverse.

## 9. Open questions / next steps

1. **Resolve LCMV mechanism (a) vs (b)** — per-subject covariance condition
   numbers at 2048 vs 500 Hz, and which pipeline's sources are more reproducible.
   Decides whether `fs_500` re-estimation is the fix or the problem.
2. **Regenerate all `group_stats`** with the corrected baseline (cheap; reads
   existing `.npz`) and re-read which edges actually show modulation.
3. **Full 20-subject A/B run** + window/order stability sweep (does agreement
   recover at 100–200 ms windows / lower order?).
4. **Filter-vs-resample decomposition** (A2 = scipy-resample sensors then LCMV)
   to separate the LCMV-filter change from the resample-method change.
5. **Choose GC baselines per design** (the config defaults are placeholders).
6. Investigate subject **4006's LinAlgError** (non-PD MVAR covariance).

## 10. Files changed / added

- `run_granger.py` — edge-padded `resample_channels` (fix #1).
- `data_loader.py`, `config.py` — `fs_500` loading, sfreq bug fix,
  `GC_BASELINE_WINDOWS` / `GC_TASK_START` / `GC_TASK_END`.
- `granger_stats.py` — `--test`, decoupled GC baseline + task start/end,
  figure annotations.
- `compare_downsample_order.py` (new) — A/B downsampling-order harness with
  matched-noise control. Outputs under
  `derivatives/source_estimation/GC_downsample_order_check/`.
