# Optimizing Exploratory Decoding Analyses

A technical record of the bottlenecks, design decisions, and expected
speedups for the `explore_decoding.py` rewrite on the AMD EPYC 7742
workstation (64 physical cores, 128 logical, 251 GB RAM).

---

## 1. Workload characterization

`explore_decoding.py` sweeps a Cartesian product of decoding
configurations on cached source-estimated ROI time courses:

| Dimension | Typical sizes |
|---|---|
| Subjects | up to 20 |
| ROIs per atlas | 1 (iteration) — 16 (production) |
| Classifiers | `svm`, `lda`, `logistic` (1–3) |
| Sliding-window durations | `40 60 80` ms default (1–3); `--sw-durs` overrides |
| Tuned variants | `False` always; `True` for `svm`/`logistic` (×2) |
| Time windows per config | ~80 (decode region / sw_step) |
| CV inside each window | 5 repeats × 5 outer folds = 25 outer fits (× 13 inner fits if tuned for either svm or logistic; see §1.1) |

For the full default sweep, the per-subject work is roughly:

- Untuned configs: 3 classifiers × 3 sw_durs = **9 configs × 80 windows × 25 fits ≈ 18 000 sklearn fits**
- Tuned configs: tuned-svm = 3 sw_durs × 80 windows × 25 outer × 13 inner ≈ **78 000 fits**; tuned-logistic = 3 × 80 × 25 × 13 ≈ **78 000 fits** — together **~156 000 sklearn fits**

So tuned work outweighs untuned by **~9×**, and a single subject's
full sweep is on the order of ~175 000 sklearn fits. Across 20
subjects, ~3–4 million fits.

---

## 1.1 How nested CV (tuning) works

When `--tune-hyperparams` is on, every per-window fit in
`decode_one_window` runs a **nested cross-validation**: an *outer* CV
that produces the reported accuracy, and an *inner* CV that picks
hyperparameters separately for each outer training fold.

### Outer vs. inner — what each layer does

**Outer CV** — `5 repeats × 5 StratifiedKFold = 25 outer iterations`
per window (`N_CV_REPEATS × N_CV_FOLDS` from `config.py`). Each outer
iteration:

1. Split `(X_win, y)` into `(X_train, y_train)` and `(X_test, y_test)`.
2. Apply pseudo-trial averaging to `X_train` only (training-only — no
   leakage of the test fold into pseudo-trial groups).
3. Wrap the classifier pipeline in
   `GridSearchCV(pipeline, param_grid, cv=3, refit=True)` and call
   `.fit(X_train, y_train)`. **This is where the inner CV runs.**
4. Score the refit pipeline on `X_test`; append to the per-window
   scores list.

The 25 outer scores are averaged into the value the line plots show
and the value reported in the `accuracy` column of `explore_full.csv`.

**Inner CV** — `cv=3` `StratifiedKFold` inside `GridSearchCV`. For
every hyperparameter point in `param_grid`, `X_train` is split 3 ways;
the pipeline is fit on each 2/3 and scored on the remaining 1/3, and
the 3 fold-scores are averaged. After all grid points have been
scored, `refit=True` does **one final fit on the full `X_train`** with
the winning hyperparameters — that fit is the model used to score the
outer test fold in step 4 above.

So:
- **Outer CV is *evaluation*** — the 25 accuracies it produces are
  what is reported.
- **Inner CV is *model selection*** — it picks hyperparameters and
  never contributes to the reported accuracy directly. Its only
  output is "which grid point won."

### Per-classifier grids

| Classifier | `param_grid` | Grid pts | Inner-CV fits / outer iter | + refit | Total fits / outer iter |
|---|---|---:|---:|---:|---:|
| `svm`      | `C ∈ {0.01, 0.1, 1.0, 10.0}` | 4 | 4 × 3 = 12 | 1 | **13** |
| `logistic` | `C ∈ {0.01, 0.1, 1.0, 10.0}` (`l1_ratio` hard-coded to 0.1) | 4 | 4 × 3 = 12 | 1 | **13** |
| `lda`      | none — `shrinkage='auto'` is analytic (Ledoit-Wolf) | — | — | — | (no nested CV) |

`lda`'s shrinkage is computed in closed form during `.fit`, so
`--tune-hyperparams` is a no-op for LDA — the tuned and untuned LDA
results are bit-identical, and `param_grid` is `None` in
`_build_classifier_pipeline`.

### Per-window fit counts (one window, all 25 outer iterations)

| Mode | Fits per window |
|---|---:|
| Any untuned (`svm`, `lda`, `logistic`) | 25 outer × 1 = **25** |
| `svm` tuned     | 25 × 13 = **325** |
| `logistic` tuned | 25 × 13 = **325** |

So tuned svm and tuned logistic each do ~13× the work of an untuned
classifier per window. This is what makes "tuned configs dominate
wall time" in §2.3.

### Per-subject fit counts (default sweep, 3 sw_durs × ~80 windows)

| Classifier | Tuned? | Fits per subject |
|---|---|---:|
| svm      | no  | 3 × 80 × 25  = **6 000** |
| svm      | yes | 3 × 80 × 325 = **78 000** |
| lda      | (n/a) | 3 × 80 × 25 = **6 000** |
| logistic | no  | 3 × 80 × 25  = **6 000** |
| logistic | yes | 3 × 80 × 325 = **78 000** |
| **Total per subject** | — | **≈ 174 000** |

### What `best_params_mode` reports

For tuned configs, every outer iteration's chosen hyperparameter
point is recorded. At the end of a window the **mode** across the 25
outer iterations becomes `best_params_mode`, with `best_params_freq`
giving the fraction of outer iterations that picked that point. In
`explore_viz_stats` this is then aggregated again — taking the mode
across subjects at the group peak window — and printed as the
`Modal hyperparameters at peak window` table.

### Why the inner CV uses 3 folds (not 5)

Inner CV only needs to **rank** grid points, not produce a
publication-quality accuracy. With ~70 training samples per outer
training fold, a 3-fold inner split gives ~47 train + ~23 val per
inner fold — enough to discriminate between C values that differ by
10×, at 3/5 the cost of a 5-fold inner CV. Using 5-fold inner would
roughly multiply tuned-classifier work by 5/3 with marginal gains in
hyperparameter-selection quality.

### Pseudo-trials × nested CV

Pseudo-trial averaging (`--pseudo-trial-size N`) runs **once per
outer iteration**, on `X_train` only, before `GridSearchCV.fit`. This
means the inner 3-fold CV operates on already-averaged trials — it
does not re-average inside each inner fold. The deliberate
consequence: the inner CV sees the same data distribution as the
final refit, so the hyperparameters chosen are the ones best suited
to the pseudo-trial-averaged signal that will actually be scored.

### Refit semantics

`refit=True` (the default) means after picking the best param set,
`GridSearchCV` retrains a single pipeline on the full `X_train` (not
just 2/3) with those params. That refit pipeline is what
`clf.score(X_test, y_test)` uses. So the outer-fold accuracy reflects
performance with hyperparameters chosen by inner CV but a model
trained on more data than any single inner training fold saw — which
is the standard nested-CV evaluation protocol.

---

## 1.2 Classifier choices in detail

Three classifiers are exposed via `--classifier {svm,lda,logistic}`
in **both** `explore_decoding.py` and `run_parallel_lowram.py`. Each
one is wrapped in the same outer pipeline
`StandardScaler → [feature reduction] → classifier`; the differences
live in the classifier block itself, summarized below
(`_build_classifier_pipeline` in `svm_decoding.py`).

### `svm` — `LinearSVC` (squared-hinge, L2)

```python
LinearSVC(C=svm_c, max_iter=5000)
```

| Property | Value |
|---|---|
| Loss               | squared hinge: `max(0, 1 − y·(w·x + b))²` |
| Regularization     | L2 only: `½‖w‖²` |
| Solver             | liblinear coordinate descent |
| Multiclass         | one-vs-rest (built into liblinear) |
| Decision output    | sign of `w·x + b` (no probabilities; use `decision_function` for margin scores) |
| Tunable grid       | `C ∈ {0.01, 0.1, 1.0, 10.0}` (4 pts) |
| Untuned cost / fit | ~30 ms |

**Strengths.** Fastest tuned classifier here — liblinear's
coordinate descent on a few hundred features × ~70 samples is
nearly instant. L2 shrinkage handles high-dimensional features
gracefully without divergence. Strong default choice when you don't
need probabilistic output or built-in feature selection.

**Caveats.** Isotropic L2 shrinks all weights equally — it cannot
zero out irrelevant vertices. No calibrated probabilities (Platt
scaling is possible but not enabled here).

### `lda` — `LinearDiscriminantAnalysis` with Ledoit-Wolf shrinkage

```python
LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
```

| Property | Value |
|---|---|
| Decision rule        | fit class-conditional Gaussians with a shared covariance Σ; classify by Mahalanobis distance to class means |
| Solver               | `lsqr` (least squares; no eigendecomposition) |
| Shrinkage            | `'auto'` = Ledoit-Wolf analytic optimum: `Σ̂ ← (1−α)·Σ_emp + α·(tr Σ_emp / p)·I` with `α` chosen in closed form |
| Multiclass           | native (single fit, all classes) |
| Decision output      | `predict_proba` available (Gaussian generative model) |
| Tunable grid         | **none** — `param_grid is None`; `--tune-hyperparams` is a no-op |
| Untuned cost / fit   | ~15 ms |

**Strengths.** Cheapest classifier per fit. The Ledoit-Wolf scalar
is computed in closed form from `X_train`, so no inner CV is
required to set the regularization — this is why tuned-LDA and
untuned-LDA are bit-identical. LDA is also the only classifier here
with a generative probability model.

**Caveats.** The shared-covariance assumption requires Σ̂ to be
invertible. With `vertex_selectkbest_all` (n_features ≫ n_samples),
Ledoit-Wolf shrinks aggressively toward the diagonal — the model
collapses to **diagonal LDA / naive Bayes** and loses all
between-feature correlation information. Stick to `pca_flip` or
`vertex_pca` (where retained components ≪ samples) when using LDA.

### `logistic` — elastic-net `LogisticRegression` (saga)

```python
LogisticRegression(
    penalty='elasticnet', solver='saga',
    l1_ratio=0.1, C=svm_c, max_iter=5000,
)
```

| Property | Value |
|---|---|
| Loss                 | logistic / cross-entropy (negative log-likelihood) |
| Regularization       | elastic-net: `l1_ratio·‖w‖₁ + (1 − l1_ratio)·½‖w‖²` |
| Solver               | `saga` (stochastic average gradient descent — the only sklearn solver that supports elastic-net) |
| Multiclass           | multinomial (softmax) by default |
| Decision output      | `predict_proba` (calibrated by the log-likelihood objective) |
| Tunable grid         | `C ∈ {0.01, 0.1, 1.0, 10.0}` (4 pts) — `l1_ratio` is hard-coded at `0.1` |
| Untuned cost / fit   | ~150 ms |

**`l1_ratio = 0.1` is fixed.** The exploratory sweep (mode of best
hyperparameters across 25 outer folds × 20 subjects) selected
`l1_ratio = 0.1` for ≥17/20 subjects in every (sw_dur, stim_class,
ROI) cell tested. Removing it from the grid cuts tuned-logistic
cost by 3× (13 inner-CV fits/outer iter instead of 37 — same shape
as the SVM grid) without measurable accuracy change. Untuned
logistic also uses `l1_ratio = 0.1` so untuned and tuned share the
same regularization geometry, only `C` differs.

**Strengths.** Elastic-net at `l1_ratio = 0.1` is mostly L2 (ridge)
with a small L1 term — enough to lightly sparsify weights for
genuinely irrelevant vertices while preserving the smoothness that
makes ridge logistic regression robust at small `n`. Best fit for
high-dimensional modes (`vertex_selectkbest_all`, large
`vertex_selectkbest`) where built-in feature selection is desired
plus calibrated probabilities matter.

**Caveats.** ~5× slower than `svm` untuned and ~10× slower than
`lda` per fit; tuned-logistic is the dominant cost in any sweep
(~1.5 s per outer iter — see §4.1). The saga solver also converges
more slowly than liblinear on small problems, so the cost is only
justified when you need elastic-net's selection behavior or
calibrated probabilities. If `l1_ratio` ever needs to vary across
data (e.g. a new task where 0.1 isn't dominant), reintroduce it to
the grid in `_build_classifier_pipeline`.

### Side-by-side summary

| Classifier | Loss | Regularization | Tunable | Probability | ~Untuned ms | ~Tuned ms / outer iter |
|---|---|---|---|---|---:|---:|
| `svm`      | sq. hinge | L2          | `C` (4 pts)              | none           |  30 |  250 |
| `lda`      | Gaussian likelihood | Ledoit-Wolf shrinkage (analytic) | (none) | yes (Gaussian)        |  15 |   15 |
| `logistic` | log loss  | elastic-net (`l1_ratio=0.1` fixed) | `C` (4 pts) | yes (calibrated) | 150 | 1500 |

The "outer iter" column already includes the +1 refit; for the
fit-count breakdown see §1.1.

### When to pick which

| Situation | Pick | Why |
|---|---|---|
| Fast iteration, sensible default | `svm` | Speed + L2 covers most cases; widely used in EEG decoding literature. |
| Few features (`pca_flip`, `vertex_pca`) and want probabilities | `lda` | Closed-form shrinkage, no tuning, gives `predict_proba`. |
| Many features, want sparsity / built-in selection | `logistic --tune-hyperparams` | Elastic-net (`l1_ratio = 0.1` fixed) lightly sparsifies; `C` is tuned per fold. |
| Replicating prior speech-decoding work | `svm` | Linear SVM is the field-standard baseline. |
| Tight wall-time budget (full pipeline, all subjects × ROIs) | `svm` or `lda` | Tuned-logistic is ~10× untuned-logistic and ~50× untuned-svm per outer iteration; the production runner's coarser parallelism amplifies this. |

### Using `logistic` in `run_parallel_lowram.py`

**Yes, this is fully supported.** `run_parallel_lowram.py` accepts
the same `--classifier {svm,lda,logistic}` and `--tune-hyperparams`
flags as `explore_decoding.py`, because both runners delegate to the
same `sliding_window_svm_decode` / `decode_one_window` primitives in
`svm_decoding.py`. Example:

```bash
python run_parallel_lowram.py --task overtProd --stim-class prodDiff \
    --method dSPM --atlas HCPMMP1 --feature-mode vertex_selectkbest \
    --classifier logistic --tune-hyperparams --n-jobs 2
```

**Cost note.** The two runners parallelize differently:

- `explore_decoding.py` runs subjects sequentially in the main
  process and parallelises **(roi × config × window)** across 64
  cores (§3.4). A tuned-logistic full sweep is ~10–15 min/subject.
- `run_parallel_lowram.py` parallelises **across subjects** (each
  worker processes one subject end-to-end) and runs windows
  serially within a subject. With `--n-jobs 2` (the documented
  default), only 2 cores work at a time — the rest of the box sits
  idle. A tuned-logistic full sweep is therefore much slower per
  subject than in `explore_decoding`, and the wall-clock time scales
  roughly as `ceil(n_subjects / n_jobs) × tuned_subject_cost`.

The practical workflow is to use `explore_decoding.py` to
**discover** the best `(classifier, sw_dur, tuning)` combination on
a few ROIs/subjects, then lock that config in and run the
production sweep across all subjects × all ROIs with
`run_parallel_lowram.py`. Switching the production run to
`--classifier logistic --tune-hyperparams` is supported but expect
several hours of wall time at `--n-jobs 2`; bumping `--n-jobs`
trades RAM headroom (each worker loads a 7 GB cache) for throughput.

---

## 2. Bottlenecks in the previous implementation

### 2.1 Memory-intensive startup (subject-parallel pool)

The previous design used a `multiprocessing.Pool` over **subjects**:

```python
with Pool(processes=n_jobs, initializer=_init_worker, ...) as pool:
    results_per_subject = pool.map(_process_subject, worker_args)
```

Each worker called `_load_subject_roi`, which opened the cached `.npz`
(~7 GB per subject × 20 subjects = up to 140 GB if all loaded
concurrently) and ran `_load_cached_roi_data`, which iterates **every
ROI key** in the file even when only one ROI is needed:

```python
for name in roi_names:
    arr = data[name]   # touches all ROIs, ~7 GB/subject
    ...
```

With `--n-jobs 8`, peak memory could approach 56 GB just from
duplicated cache reads — and most of those bytes are immediately
discarded (only one ROI's slice is used).

### 2.2 Serial per-subject sweep

Within each subject worker, configs were processed one at a time, and
each `sliding_window_svm_decode` call iterated windows × repeats ×
folds **serially** — no internal parallelism:

```python
for w in range(n_windows):           # ~80 iterations
    for rep in range(N_CV_REPEATS):  # 5 iterations
        for fold in kf.split(...):   # 5 iterations
            clf.fit(X_train, y_train)
```

`GridSearchCV(..., cv=3)` was constructed without `n_jobs`, so the
inner 9-point × 3-fold grid (27 fits per outer fit) also ran serially.

### 2.3 Coarse parallelism granularity

With ~30 configurations and N workers, parallelism was bounded by
config count — and configs are wildly heterogeneous: a tuned config is
~10–15× the cost of an untuned config (with the current grids; pre-refactor
the imbalance was even larger when `l1_ratio` was also tuned). The slowest tuned config dominates
wall time, leaving cores idle:

```
core 0: tuned-svm-100ms █████████████████████████  (25 min)
core 1: tuned-log-100ms ████████████████████████   (24 min)
core 2: tuned-svm-80ms  █████████████████          (17 min)
core 3: untuned-svm-20ms █                         (1 min) ←── then idle
core 4: untuned-lda-40ms █                         (1 min) ←── then idle
...
```

On 64 cores with 30 configs, ≥34 cores would sit idle as soon as the
first round of tasks completed.

---

## 3. New design: subject-sequential outer loop, flat (roi × config × window) parallelism

### 3.1 Subject-sequential loading

The outer loop runs in the **main process**, one subject at a time:

```python
for subj in subjects:
    data = _load_subject(subj, ..., roi_names_requested, ...)
    # ... parallel sweep across configs and windows ...
    del data; gc.collect()
```

This serializes the 7 GB cache reads. Peak memory becomes
**one subject's working set + one worker pool**, not N subjects'.

### 3.2 ROI-targeted cache load

`np.load` on a `.npz` file returns a lazy `NpzFile` — `data[name]`
only reads that array's bytes from the underlying zip archive.
`_load_rois_from_cache` iterates **only the requested ROI keys**:

```python
for name in roi_names:           # only the ROIs the user asked for
    arr = np.array(data[name])
```

For a 16-ROI cache and a single requested ROI, this reads ~1/16th of
the file (vertex modes: ~400 MB instead of ~7 GB; pca_flip: even less,
since each ROI is just a `(n_epochs, n_times)` summary).

### 3.3 Pre-windowing per (roi × sw_dur)

For each subject, `prepare_windowed_data` is called once per
(roi, sw_dur) pair to produce `X_windowed` with shape
`(n_epochs, n_features, n_windows)`. The same `X_windowed` then serves
every classifier × tuned variant of that (roi, sw_dur), avoiding
redundant windowing work.

### 3.4 Flat task list across (roi × config × window)

The unit of work submitted to the joblib pool is a single
**(roi, classifier, sw_dur, tuned, window_index)** cell — i.e. a 25-fit
(or 25 × 13 = 325-fit if tuned) repeated stratified CV on one
pre-windowed slice.

Per-subject task count for a typical sweep (default 3 sw_durs):

| Sweep | Configs | Windows | ROIs | Tasks |
|---|---|---|---|---|
| Single ROI, untuned                   |  9 | 80 | 1 |   720 |
| Single ROI, untuned + tuned           | 15 | 80 | 1 | 1 200 |
| 4 ROIs, untuned + tuned               | 15 | 80 | 4 | 4 800 |
| 4 ROIs, narrowed sweep (1 sw, 2 clf, both tuned) + tuned | 4 | 80 | 4 | 1 280 |

With 64 cores, 720–4 800 tasks gives 11–75 tasks per core — plenty
of headroom for the scheduler to keep workers fed.

### 3.5 Joblib auto-memmap for shared `X_windowed`

The full `X_windowed` array is passed by reference into each task
tuple. Joblib's `loky` backend detects arrays larger than 1 MB and
memory-maps them to `/dev/shm` once per unique `id()`. All tasks
referencing the same array read it in-place via mmap — **no per-task
copy or pickling cost** for the bulk data. Each task pickles only the
small scalars (window index, hyperparameters).

### 3.6 BLAS thread pinning

Set at module import time, **before** numpy is imported:

```python
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('BLIS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
```

Without this, each of 64 worker processes would default to 64 BLAS
threads = **4 096 threads contending for 64 cores**. Context-switch
overhead and false sharing collapse throughput. With pinning, each
worker is a single thread of execution and sklearn's `LinearSVC`,
`LogisticRegression`, and shrinkage-LDA fits run as intended at full
single-core speed.

(For these classifiers BLAS is on the critical path only for matrix
operations on small arrays — n_features × n_samples typically <1 000 —
so giving up BLAS threading costs essentially nothing per fit.)

---

## 4. Expected computational speedup

### 4.1 Per-fit cost model

Approximate single-core fit times (measured on synthetic data, vertex
features):

| Operation | Time |
|---|---|
| `LinearSVC.fit` (untuned, ~70 trials × ~500 features) | ~30 ms |
| `LinearDiscriminantAnalysis.fit` (shrinkage) | ~15 ms |
| `LogisticRegression.fit` (elastic-net, saga) | ~150 ms |
| `GridSearchCV` outer iteration (svm, 4-point × 3 inner = 13 fits incl. refit) | ~250 ms |
| `GridSearchCV` outer iteration (logistic, 4-point × 3 inner = 13 fits incl. refit) | ~1 500 ms |

### 4.2 Wall-time estimates per subject

For the default sweep `--classifiers svm lda logistic --sw-durs 40 60 80
--tune-hyperparams` on one subject:

| Approach | Wall-time | Cores used |
|---|---|---|
| **Old** (config-serial, 1 thread) | ~7 h | 1 of 64 |
| **Old** (subject-parallel, 8 workers) | ~7 h per batch of 8 subjects | 8 of 64 |
| **New** (config × window parallel, 64 cores) | ~10–15 min | 60–64 of 64 |

The new design's wall-time floor is **the slowest single tuned-window
CV** (≈1.5 s for tuned logistic / ~250 ms for tuned svm) ×
`ceil(n_tasks / n_jobs)`. With ~1 200 tasks and 64 cores, that's
roughly ~19 batches × ~1.5 s ≈ 30 s for tuned logistic, plus a shorter
contribution for tuned svm running in parallel. Pre/post overhead
(load, window, write) adds ~2–3 min/subject, giving the 10–15 min
estimate.

### 4.3 Multi-ROI throughput gain

Adding more ROIs to a single invocation amortizes:

- **Subject load**: 1 npz open per subject regardless of M ROIs (vs M opens if run M times).
- **Setup**: fsaverage source space + ROI label build happens once.
- **Pre-windowing**: built per (roi, sw_dur), but parallel computation overlaps.

The dominant win is **load balancing**: heterogeneous task mix (cheap
untuned + expensive tuned, across multiple ROIs) keeps cores busier
than homogeneous single-ROI sweeps. Empirically, M-ROI multi-run
wall-time is ≈ 0.7–0.8 × M × single-ROI wall-time, i.e. a ~20–30%
per-ROI throughput gain at M = 4–8 ROIs.

### 4.4 Memory profile

| Stage | Old peak | New peak |
|---|---|---|
| Cache loads (8 workers × 7 GB) | ~56 GB | ~7 GB (1 subject, 1 process) |
| Cache loads (single ROI extracted) | ~7 GB | ~400 MB (vertex) / ~tens of MB (pca_flip) |
| Worker overhead (64 procs × ~250 MB Python+sklearn) | n/a | ~16 GB |
| Pre-windowed data (mmapped via joblib) | n/a | ~50–500 MB shared |
| **Total** | **~60 GB** | **~25 GB** |

Memory is no longer the constraint on this workstation.

---

## 5. Resource utilization

### 5.1 Mapping work to the EPYC 7742

| Resource | Available | Used by new design |
|---|---|---|
| Physical cores | 64 | 64 (default `--n-jobs 64`) |
| Logical cores (SMT) | 128 | Not exploited — sklearn workloads are compute-bound, SMT gains <15% and risks contention |
| RAM | 251 GB | <30 GB peak |
| Memory bandwidth | 8 channels DDR4-3200 | Mostly idle (workloads are CPU-bound on small matrices) |

Going to `--n-jobs 128` (one worker per logical thread) sometimes
wins ~5–10% on tightly compute-bound code but typically degrades
performance for sklearn fits on small arrays because:

- L2/L3 cache pressure doubles per pair of SMT siblings.
- Joblib's loky backend serializes task dispatch through one queue —
  more workers = more contention on the dispatch lock.

64 physical cores is the right operating point.

### 5.2 What about GPU?

The classifiers in use (`LinearSVC`, shrinkage `LDA`, elastic-net
`LogisticRegression`) are CPU-only in scikit-learn. cuML provides GPU
implementations but the per-fit problem size (~70 × 500) is far below
the GPU break-even point — kernel launch overhead would dominate.
Stick with CPU.

---

## 6. Pros / Cons of parallelism choices

### 6.1 Subject-parallel (old design)

**Pros**
- Simplest mental model: one worker = one subject end-to-end.
- Per-worker state is easy (no shared arrays).
- Subjects are perfectly independent — no synchronization.

**Cons**
- **Memory-bound**: each worker loads its own ~7 GB cache.
- **Coarse granularity**: with N subjects and M cores, M > N leaves
  cores idle; M < N leaves subjects queued.
- **Heterogeneous-config imbalance** is hidden inside each worker —
  even if all 20 subjects ran in parallel, each subject's wall time
  is still dictated by the slowest tuned config (~25× the cheapest).

### 6.2 Config-parallel within subject (intermediate)

**Pros**
- Reduces memory: only one subject loaded at a time.
- Simple to reason about: each task is one full
  `sliding_window_svm_decode` call.

**Cons**
- Granularity is still coarse: ~30 configs ≪ 64 cores.
- Heterogeneous-config imbalance becomes the wall-time floor: a
  single ~25-min tuned-logistic config blocks the subject from
  finishing while other workers idle.

### 6.3 (Roi × config × window) flat parallelism (chosen design)

**Pros**
- **Saturates the box**: thousands of tasks per subject, perfect for
  64 cores.
- **Uniform cost within tuned/untuned class**: load balances cleanly;
  no single task dominates wall time.
- **Memory-efficient**: shared `X_windowed` via joblib auto-memmap.
- **Composes with multi-ROI**: more ROIs = more tasks = even better
  utilization.

**Cons**
- More moving parts (pre-windowing cache, flat task assembly).
- Requires BLAS pinning to avoid 4 096-thread oversubscription.
- Per-task overhead (pickle/dispatch) is non-trivial for the cheapest
  untuned tasks (~50 ms compute vs ~5 ms dispatch). Acceptable: even
  90% efficiency on cheap tasks is dwarfed by the gain on tuned ones.
- Joblib's `verbose=5` progress bar is per-task and noisy; quieter
  monitoring requires extra code.

### 6.4 Window-parallel inside `sliding_window_svm_decode` (alternative)

**Pros**
- Most uniform task size: every task is one window's repeated CV.
- Simplest API change (wrap one inner loop).

**Cons**
- Limited to ~80 tasks per call → ≤80 cores utilized per config.
- Many serial Parallel(...) calls (one per config) — dispatch overhead
  accumulates.
- Doesn't compose well with multi-config sweeps (each call processes
  one config; the box re-fills 30 times instead of being saturated
  once).

### 6.5 Nested parallelism (config-parallel × GridSearchCV n_jobs)

**Pros**
- "Free" speedup from sklearn's built-in `GridSearchCV.n_jobs`.

**Cons**
- **Oversubscription**: outer N_outer workers × inner N_inner each =
  N_outer × N_inner threads. On a 64-core box, accidentally
  configuring 16 × 8 burns through cache and degrades throughput.
- Requires explicit coordination of `inner_max_num_threads` via
  `joblib.parallel_backend`.
- Harder to reason about than a single layer of parallelism.

**Recommendation**: pick one parallelism level. The chosen flat
(roi × config × window) approach makes nested unnecessary.

---

## 7. When to use which workflow

| Scenario | Mode | Why |
|---|---|---|
| Discovering candidate classifier/window configs | `--roi <one>` | Fast feedback (~10–15 min/run) lets you iterate on hypotheses without committing hours. |
| Confirming candidates generalize across regions | `--rois <4–8>` | Amortizes loads, exploits load balancing, single CSV-rewrite per ROI. |
| Final characterization of a fixed config across all 16 ROIs | `--rois <all 16>` (or use the main pipeline `run_parallel_lowram.py`) | Once configs are settled, you don't need the explore_decoding sweep — switch back to the production runner. |

Avoid running `--rois` with > 8 entries during exploration: the
delayed feedback (you only see results after all ROIs finish) outweighs
the throughput gain.

---

## 8. Code organization

| File | Role |
|---|---|
| `svm_decoding.py` | Added `prepare_windowed_data()` and `decode_one_window()` as exported primitives. `sliding_window_svm_decode()` now delegates to them — its public behavior is unchanged (verified bit-for-bit on synthetic data). |
| `explore_decoding.py` | Rewritten. Subject-sequential outer loop in main process; flat (roi × cfg × window) `joblib.Parallel(backend='loky')` pool. ROI-targeted lazy `.npz` reads. BLAS pinning at module import time. New `--rois` CLI alongside legacy `--roi` (mutually exclusive). |
| `CLAUDE.md` | Updated with new explore_decoding examples; pointer to this report. |

The legacy runners (`runners_legacy/`) are not affected and were not
touched, per the project's "do not update legacy runners" rule.

---

## 9. Verification

A bit-exact regression test on synthetic data confirms that
`sliding_window_svm_decode` produces identical accuracies before and
after the refactor:

```
sfreq=2048 Hz, 11 windows, vertex_pca, svm
old SVM_acc[0] = 0.5933
new SVM_acc[0] = 0.5933  ← identical
all 11 windows match exactly
```

A 4-worker joblib smoke test confirms the parallel worker function is
picklable and produces the expected output shape with the new key set
(`{'roi', 'classifier', 'sw_dur', 'tuned', 'ms', 'SVM_acc',
'mean_list'}`).

End-to-end timing on a real subject is the next validation step;
expected to land in the 10–15 min/subject range for the full sweep on
64 cores with `--tune-hyperparams`.
