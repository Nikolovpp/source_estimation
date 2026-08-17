# Source-space Granger causality — build log

**23 March – 15 August 2026.** How a BSMART port grew into a validated
source-space GC pipeline: the configurations tried, the bugs that invalidated
whole runs, and the evidence behind each surviving decision. §0 covers the
source-estimation work the GC pipeline is built on; §1 onward is GC proper.

Reconstructed from the repository's full git history (167 commits) and the
working session log. Commit hashes given per phase are representative, not
exhaustive. Every quantity below was measured on the pipeline's own output;
inferences are marked as such.

**Branch coverage.** `main` is an ancestor of
`gc-routes-parametric-vs-statespace` — verified with
`git merge-base --is-ancestor main HEAD` — so every commit on `main` is already
in this history. `main`'s tip (`f4fa242`, 29 July) is the merge base; it has no
commits the branch lacks, and neither does `granger-band-erp-fixes`. The GC work
lives entirely on the branch and has not yet been merged back.

| | |
|---|---|
| Repository commits | 167 |
| — foundation (Mar–May, §0) | 86 |
| — GC pipeline (Jul–Aug, §1–9) | 81 |
| Final sweep | 352 configs |
| MVAR fits in the final sweep | 1,803,560 |
| Subjects | 20 |
| Runs invalidated by a bug | 2 |

Companion documents: `granger_preprocessing_references.md` (the literature
basis for ERP removal), `matlab_vs_python_gc.md` (differences from the sensor
pipeline), `optimizing_exploratory_analyses.md`.

---

## 0 · The foundation the GC pipeline reads from
*23 March – 10 May 2026 — 86 commits, all on `main`*

No GC code exists before 6 July. These 86 commits build the source-estimation
and decoding pipeline that GC later consumes — its vertex ROI caches *are* the
GC input, so decisions taken here constrain everything after.

**Source estimation (March).** fsaverage ico-5 source space with a 3-layer BEM,
dSPM and LCMV inverses in standard and low-RAM variants, spatial leakage
correction by Löwdin orthogonalisation and regression, and four atlas options
— aparc, Schaefer200, HCPMMP1, and a custom functional-localizer atlas
projected from volumetric NIfTI onto the fsaverage surface (Chang et al.). The
custom atlas is the one every GC analysis uses; its four ROIs are `awfa`,
`ifc`, `pmc`, `tpc`.

Two fixes here matter downstream: `reduce_rank=True` on `make_lcmv` to avoid a
singular leadfield, and a singular-matrix fix in vertex leakage correction. Both
concern the same failure mode — rank deficiency in the inverse — that resurfaced
in July as the LCMV spatial collapse (§2), so the March fixes were necessary but
not sufficient.

**Output-path discipline (March–May).** A long run of commits threaded atlas,
leakage correction, pseudo-trial size, classifier and C through every output
path. That looks like housekeeping and is not: it is the same invariant the GC
pipeline later violated twice — `demean_trials` missing from `gc_tag`, and the
sweep-summary grouping that merged A/B arms (§7). Every parameter that changes
the numbers must appear in the path.

**Decoding and infrastructure (April–May).** Classifier options and ROI-subset
filtering, `explore_decoding.py` with (ROI × config × window) parallelism,
cluster-permutation statistics ported verbatim between the two stats
viewers, a shared ROI cache across `vertex_*` feature modes, external-drive
cache fallback, BLAS pinned to one thread per worker, and the split of the
combined runner into `run_source_localize.py` and `run_decode.py` — the
subject-parallel / cell-parallel split that the GC sweep later mirrors at the
config level.

The cluster-permutation test the GC statistics use is the same one written here
for the decoding curves, which is why GC and decoding results are testable on
equal terms.

## 1 · Porting BSMART, and building a statistics layer around it
*6–8 July — `2340333`, `7854208`, `9c25d01`, `78943f3`, `5ebbaf7`*

The pipeline began as a Python port of the MATLAB sensor-space analysis:
`armorf`'s Morf recursion and `mov_bi_ga`'s moving-window driver, reading the
vertex ROI caches the decoding already produced. The port was verified
byte-faithful against BSMART to 2.8e-16, and that verification has held through
every change since — the four validators still pass 26/26, 11/11, 8/8.

Everything around the estimator had to be built. `--roi-subset` runs got their
own output folders so a two-ROI probe could not overwrite a full run; subset
names were validated against the actual cache rather than assumed. The stats
layer gained both the MATLAB test (`--test ttest`) and a non-parametric
alternative (`signrank`), a task-end crop to drop the trailing MVAR edge
window, and a baseline decoupled from the covariance baseline used by the
inverse.

**Decision (kept).** The baseline is the epoch's *actual* pre-stimulus period,
derived from the data rather than hardcoded — an earlier fixed interior window
had been cutting into it. A low baseline is the signal, not an artifact: a rest
period should have low directed connectivity.

## 2 · Two source-estimation bugs, then a search for the right configuration
*9–12 July — `ce73423`, `b247fa9`, `237153d`, `b298c73`, `b8cfe46`*

Two defects upstream of GC surfaced and were fixed. Some subjects' LCMV
beamformers were collapsing spatially — one global time course on every vertex
— which a data-driven rank estimate (`mne.compute_rank`) resolved. And the
overtProd noise-covariance baseline window sat *outside* the epoch entirely; it
now validates and logs.

With those fixed, the question became which GC configuration to run. A
methods-paper suite compared virtual-sensor definitions, swept MVAR order, and
contrasted plain GC against time-reversed GC; a diagonal sweep tested the rule
of thumb that window = 2 × model order. The per-epoch resample was found to
inject a ringing transient at both epoch edges and was fixed with edge padding,
mirroring EEGLAB's `pop_resample`.

**Finding.** TRGC proved order-stable where plain GC was order-fragile — an
early signal of the parameter sensitivity that would eventually justify the
whole window × order sweep.

## 3 · A citable estimator, and the commit that quietly broke everything
*21–29 July — `fa0e91c`, `26d1c87`, `12a39a3`, `669f8b7`, `ff2cdd6`*

A home-brewed BSMART port is hard to defend in a methods section, so
MNE-Connectivity's GC was evaluated as a citable replacement. Direction
reproduced; magnitude did not. Two things explained the gap: MNE's `gc` is a
VAR of `gc_n_lags`, not Barnett–Seth state-space GC as assumed, and the 4–30 Hz
CWT grid understated GC roughly sevenfold — the estimate has to run
DC-to-Nyquist with band averaging afterwards.

A systematic bug hunt on 29 July found real defects and fixed most of them.
`band_average`'s bands overlapped, counting 8, 12 and 18 Hz in two bands each,
so peak alpha contributed a fifth of the theta mean; that fix changed every band
value ever reported. ERP removal was made the default in both runners.

**Bug introduced, found 15 days later.** The same commit "fixed"
`--normalize` on the belief that `V` is `(n_trials, n_chan, n_times)`. It is
`(n_roi, n_trials, n_times)`. Iterating `V[:, c, :]` instead of `V[r]` turned
ERP removal into a common-average reference across the very ROIs being related.
The comment added alongside it described that exact failure mode as the thing
to avoid.

## 4 · Parametric versus state-space, across the speech routes
*6–11 August — `ec308cb`, `307d031`, `31fcc72`, `6fdf94d`, `eeeff43`*

A route-comparison runner computed conditional spectral GC by three paths —
parametric MVAR, Barnett–Seth state-space via the DARE, and nonparametric
Wilson factorisation — over the dual-stream ROIs. Two unpacking bugs were fixed
along the way: the ROI cache axis order, and the FIXPC-*k* block-GC unpack.

The estimators agree when only a sub-block of the model is needed and diverge
when a factorisation is required, which is the expected behaviour: the reduced
model of a VAR is a VARMA, so no finite reduced VAR order is correct and the
parametric route fits it at an order known to be wrong. A group statistics layer
was added that reports non-random missingness *before* any group mean.

**Decision (narrowed scope).** Time-domain GC was dropped from the default
path. The project reports spectral GC; the time-domain variant cost runtime and
produced numbers nobody read.

## 5 · The window × order sweep
*11–12 August — `332b5f4`, `0232668`, `8a890bd`, `658af5f`, `dc1b8a3`*

Every prior configuration varied window and order together — `order6_win60ms`
against `order10_win120ms` changes both at once, so neither can be blamed for a
difference. The sweep crosses them independently: windows 40/60/80 ms against
orders 2/4/6/10 at 200 Hz, over two tasks and two contrasts.

### Why 200 Hz, and why no 20 ms window

These two exclusions did most of the work in defining the grid, so the
reasoning is worth setting out. Figure:
`GC_source_space/_figures_methods/sampling_rate_tradeoff.png`, from
`plot_sampling_rate_tradeoff.py`.

**Model order is a duration, not a unitless measure of model richness.** An
order-*p* fit sees *p* lags spaced 1/*f_s* apart, so it spans *p*/*f_s* seconds
of history. The same order means different things at different rates:

| order | @500 Hz | @200 Hz |
|---|---|---|
| 2 | 4 ms | 10 ms |
| 6 | 12 ms | 30 ms |
| 10 | 20 ms | 50 ms |

To represent an oscillation the model must span a real fraction of its cycle,
and high beta at 30 Hz is a 33 ms cycle — the shortest anything in this study
reports. Fraction of one cycle spanned at order 10, the deepest in the grid:

| band | cycle (fastest edge) | @500 Hz (20 ms) | @200 Hz (50 ms) |
|---|---|---|---|
| theta | 125 ms | 0.16× | 0.40× |
| alpha | 83 ms | 0.24× | 0.60× |
| low beta | 56 ms | 0.36× | 0.90× |
| high beta | 33 ms | 0.60× | **1.50×** |

At 500 Hz the deepest model in the sweep does not span one full cycle of *any*
reported band.

**The extra samples at 500 Hz do not compensate.** The obvious objection is
that 500 Hz gives 2.5× more samples per window, so the fit should be better
conditioned — true only if those samples were independent. The data is bandpass
filtered 0.1–30 Hz (per the EEGLAB epoch filenames), and a signal band-limited
to *B* has autocorrelation sinc(2*Bτ*):

| *f_s* | lag | r(adjacent) | oversampling vs Nyquist(30 Hz) |
|---|---|---|---|
| 500 Hz | 2.0 ms | **0.976** | 8.3× |
| 200 Hz | 5.0 ms | 0.858 | 3.3× |

(Those are the ideal brick-wall values. Measured through the actual 4th-order
Butterworth they are 0.979 and 0.876 — the idealisation slightly *understates*
the redundancy, so the argument is if anything conservative.)

Adjacent 500 Hz samples share ~95% of their variance, while every extra lag
still costs *n*² coefficients — 40 at order 10 bivariate. Full parameter price
for near-duplicate observations, which shows up as ill-conditioning rather than
as an error.

**Measured on a known process.** A VAR resonating at 22 Hz in *x* drives *y*
15 ms later, low-passed at 30 Hz, then decimated to each rate so both see the
same signal; 300 trials, averaged over nine window placements:

| *f_s* | win | samples | order | reach | ratio x→y : y→x | rho |
|---|---|---|---|---|---|---|
| 500 | 40 | 20 | 6 | **12 ms** | 3.4 | 0.992 |
| 500 | 80 | 40 | 6 | **12 ms** | 3.2 | 0.992 |
| 500 | 40 | 20 | 10 | 20 ms | 26.7 | 0.977 |
| 500 | 80 | 40 | 10 | 20 ms | 28.2 | 0.976 |
| 200 | 40 | 8 | 6 | 30 ms | 10.0 | 0.957 |
| 200 | 80 | 16 | 6 | 30 ms | 9.7 | 0.956 |
| 200 | 80 | 16 | 10 | **50 ms** | **333** | 0.933 |

The two 500 Hz order-6 rows span 12 ms while trying to detect a 15 ms
interaction, and doubling the window from 40 to 80 ms — 20 samples to 40 —
moves the ratio 3.4 → 3.2. **Samples cannot substitute for reach.** The best row
has 16 samples and beats the row with 40. Spectral radius falls monotonically as
reach grows, so the oversampled fits are also the ones straining against the
stability boundary — the same near-unit-root regime the real diagnostics show.

*Caveat: one synthetic process with one lag, built to make the mechanism
visible. It shows why reach matters; it does not establish the optimal
configuration for real data, which is what the sweep itself was for.*

**The 20 ms exclusion is a separate, arithmetic constraint.** At 200 Hz that is
four samples, and `armorf`'s Morf recursion has prediction-error arrays of width
`Nl − m − 1`, so it runs out of data once *p* + 1 ≥ *Nl*; the sweep enforces
`samples > order + 1`, capping four samples at order 2. The trial ensemble does
not rescue it — 234 trials give plenty of observation equations, but each
contributes the same few lag configurations. Running 20 ms at 500 Hz instead
(10 samples, order 6 feasible) was rejected on design grounds: it would make
20 ms the only column at a different sampling rate, putting a sampling-rate
confound inside a grid built to separate window from order.

**What this costs.** Theta is under-covered at every configuration in the grid —
0.40× of a cycle at best. Reaching one theta cycle would need order 25 at
200 Hz, i.e. the canonical 250 ms / order-25 configuration, which this grid does
not include. Theta results here should be read with that in mind.

### The rest of the grid

One estimator runs throughout, with the *analysis* chosen by subset size — two
ROIs give bivariate GC, three give A→B|C conditioned on the remaining region.
An ill-conditioned window now degrades to NaN and is counted rather than killing
every subject in the joblib pool.

**Infrastructure.** Parallelism moved from inside a config to across configs.
Only 1–4 cores were being used because subjects run sequentially and only
windows were parallel. The ceiling turned out to be shared-drive I/O, not cores:
`PARALLEL=56` crashed the workstation, and 8 became the default.

## 6 · The normalize bug, and what the literature actually says
*13 August — `95bd838`, `2e42d7a`, `5cb824a`*

The pairwise arm came back with 96–99% of windows marked unstable, flat across
every window length and model order — the giveaway, since sample count was
clearly not the issue. The cross-ROI average introduced in July makes a two-ROI
pair exactly antisymmetric: (a−b)/2 and −(a−b)/2, the same signal negated. The
ensemble is rank 1, every covariance in the Morf recursion is singular, and the
Cholesky raises for every window.

| Arm | Windows lost | Detected by |
|---|---|---|
| Bivariate (2 ROI) | 1,244,994 / 1,245,572 | Loud — a crash per window |
| Triple-wise (3 ROI) | 93% of values | **Silent — looked like data** |

A prior diagnosis of mine — that a two-ROI subset in conditional mode
degenerates on an empty conditioning set — was checked against the MVGC source
and proved wrong. `autocov_to_smvgc.m` branches `if isempty(z)` to the
unconditional formula, and our state-space code has the same branch. With the
normalize bug fixed the two modes agree to 4.4e-16. One cause, not two.

**Is ERP removal even defensible?** MVGC's `demean.m` subtracts one scalar per
variable over pooled time and trials — never across variables, which confirmed
the fix — and the toolbox has no ERP-removal function at all. But its authors
endorse the step in print (Seth, Barrett & Barnett 2015), and Ding, Bressler,
Yang & Liang (2000) — the BSMART lineage this code ports — call it essential and
apply it per channel. MVGC's only event-related citation is that same paper.

**Decision (keep ERP removal).** Reverting to MVGC's bare demean would be the
*harder* position to defend for event-related data. The objection to prepare for
is Wang, Chen & Ding (2008) — average-ERP subtraction ignores trial-to-trial
variability — not that the step is unwarranted. Full quotes in
`granger_preprocessing_references.md`.

## 7 · Diagnostics, and a review that found nine more bugs
*13–14 August — `98987a9`, `a8ed24f`, `9619d95`, `8039d18`, `5eaa08c`*

Nothing in the pipeline detected a non-minimum-phase model — a fit that returns
without complaint but whose spectral GC is undefined. Two diagnostics were
ported from MVGC and now run per window: the companion spectral radius
(`var_specrad.m`) and the consistency statistic (`stats/consistency.m`, Ding
Eq. 12). Reusing the existing fit rather than refitting kept the cost at 2%,
not 100%.

A full code review followed. Nine defects, ranked by what they would have cost:

| Defect | Consequence had it shipped |
|---|---|
| Sweep-summary glob required `*conditional` | "0 configs" for the entire bivariate arm |
| Grouping on (window, order) alone | Would have averaged the A/B arms into one line |
| Band edges never saved | No result file could say what "theta" meant in it |
| Three plot scripts hardcoded `/mnt/r` | "No data" on the workstation, not "wrong machine" |
| Run header omitted the preprocessing flags | A/B arm logs indistinguishable from each other |
| Diagnostics indexed by arrival order | Latent: rho attributed to the wrong pair |
| `demean_trials` absent from the output path | Two arms silently overwriting each other |
| ERP removed pooled across stimulus classes | Between-class evoked difference left in the residual |
| Stats not NaN-aware | **One NaN killed a real effect** |

**The worst of them.** `granger_stats` used plain `mean`/`std`. A single NaN in
one subject's baseline window made the group baseline NaN and took the whole
edge down. Measured on synthetic data carrying a real effect: **20/20
significant windows became 0/20, silently.** The permutation test was worse —
one NaN anywhere discarded the pair entirely, the test the module's own
docstring calls "the test to report".

Edge windows were measured rather than asserted. Two scripts disagreed — one
dropped three leading windows, the other dropped none — and both docstrings
turned out to be right about the configuration they were measured on and wrong
as general claims. The contaminated span of signal is roughly fixed at 20–30 ms,
so the number of affected *windows* is that span over the step while the *depth*
scales inversely with window length.

## 8 · The preprocessing A/B, and the sweep that finally ran clean
*14–15 August — `e24a671`, `078a4ba`, `0b94980`, `a97e06d`, `fe6f196`*

Rather than argue the preprocessing from first principles, five arms were run on
both tasks at two orders. Trial counts were read from the caches rather than
assumed: **234 epochs mean for perception, 202 for production** — roughly 2.5×
my estimate, which materially weakened my own argument against ensemble-SD
normalisation.

| Arm | Δ vs demean | Band flips (order 6→10) | Mean abs. change |
|---|---|---|---|
| 0 · no ERP removal | <1% | 2/4 | 92–121% |
| A · demean *(was default)* | — | 2/4 | 92–119% |
| C · no per-trial demean | ~3% | 2/4 | 93–118% |
| D · ERP removed per class | <1% | 2/4 | 92–119% |
| **B · zscore** | up to 300% | **0/4** | **21–33%** |

ERP removal turned out not to matter numerically for this ROI pair — under 1% in
every cell — so that choice rests on principle and the literature, not on the
data. Only `zscore` changed anything, and it changed the one thing the sweep
exists to test: under every other arm the dominant band flips with the model
order in half the cells.

**Decision (zscore, with a stated cost).** Ensemble-SD normalisation is Ding's
own step 3, it is citable, and it is the only arm under which the robustness
question has a stable answer. It buys that partly by whitening — spectral radius
0.99 → 0.94 — and it specifically suppresses low-frequency GC. Both belong in
the methods section.

The full sweep then ran clean: **352/352 configs, zero unstable windows,
0 of 1,803,560 fits non-minimum-phase, consistency 96.8%.**

A late correction: the 30 ms edge guard had been measured on un-normalised data.
Under zscore the SD normalisation already compensates for the edge transient —
only the first window is materially low (66–80% of plateau vs 46–68%) — so the
guard was discarding good baseline and the default is now zero.

## 9 · Baseline definition and the interpretable figures
*15 August — `a97e06d`, `fe6f196`, `99cdcd4`, `ca0b663`*

Three separate figure sets now exist, kept apart from the sweep's robustness
figures: `_figures_pairwise/` and `_figures_triplewise/` (cluster permutation),
`_figures_pairwise_ttest/` and `_figures_triplewise_ttest/` (the a-priori MATLAB
pointwise test), and `_figures_*_bl50/` (50 ms baseline).

Two baseline corrections. **Perception task now starts at 0** — the task span
was being derived from the baseline end, so it tested 100 ms of pre-stimulus
windows as task. overtProd is deliberately exempt: its t=0 is articulation
onset, and the planning epoch before it is the period of interest. And the
**source epoch cannot match the sensor epoch**: LCMV consumes a 100 ms
pre-stimulus segment for its covariance estimate, so that segment is not
available to GC and the source axis necessarily starts 100 ms later. Taking "the
leading 100 ms of whatever axis exists" is what keeps the two comparable in rule.

`plot_gc_baseline_choice.py` asks whether the baseline should scale with the GC
configuration. A flat duration holds the same *number* of windows at every
window length but not the same independent evidence: at a 1-sample step, 100 ms
spans 2.5 window-lengths at 40 ms and 1.25 at 80 ms. Measured on
perception/percDiff ifc↔tpc, order 6, the SEM of the baseline estimate stops
falling at roughly **one window length**, and the significant-window count swings
fourfold across durations (80 ms window: 29 at 25 ms, 72 at 75 ms, 17 at 200 ms —
the fall past 100 ms is task contamination, since a 200 ms baseline in perception
reaches t=0). This suggests a baseline of ~1–1.5 × window length rather than a
flat 100 ms. Measured, not adopted: it is a design decision to make once and
state.

---

## Where it stands

### Supported

- **Edge ranking is grid-stable** once order 2 is dropped — Spearman 0.88–0.97
  across the nine remaining window × order cells (0.70–0.84 including order 2,
  which is degenerate at 3–4× lower GC).
- **tpc→awfa exceeds its reverse in all 132 grid cells**, across both tasks,
  both contrasts and all three bands.
- **Conditioning changes nothing.** Bivariate against A→B|C lies on the
  diagonal, so the 88 triple-wise configs added no information beyond the
  bivariate arm.

### Not supported

- **Absolute GC values.** The across-grid SD is 30–60% of the mean, and
  window 40 > 60 > 80 ms monotonically — finite-sample bias, not signal.
- **Any task-vs-baseline effect under family correction.** 5–8 of 144 tests
  reach p<0.05 against ~7 expected by chance; none survive FDR.
- **The earlier awfa→ifc directional headline.** That edge is symmetric here
  (0.042 vs 0.041 in high beta).

### Open questions

- The pointwise *t* — the a-priori MATLAB design — finds `ifc→tpc` theta in
  perception at 59/125 windows against a chance expectation of 6, with 0/125
  reverse. FWER across 144 tests buries it. Which family is the right one is a
  judgement call, not a computation.
- `tpc` and `awfa` are both temporal-lobe ROIs, so proximity-driven leakage is
  the live alternative for the one robust asymmetry. The correction applied is
  zero-lag only and would not remove a lagged spread. A
  `--no-leakage-correction` comparison on that pair would settle it.
- Baseline duration swings the significant-window count fourfold, and the
  precision of the baseline estimate stops improving at roughly one window
  length (see §9).
- Only band means are persisted, so changing band edges later means recomputing
  all 352 configs. The MNE path persists frequency-resolved GC; `run_granger.py`
  does not.
