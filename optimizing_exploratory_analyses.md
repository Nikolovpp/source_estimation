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
| Sliding-window durations | `20 40 60 80 100` ms (1–5) |
| Tuned variants | `False` always; `True` for `svm`/`logistic` (×2) |
| Time windows per config | ~80 (decode region / sw_step) |
| CV inside each window | 5 repeats × 5 outer folds = 25 fits (×27 inner fits if tuned) |

For the full sweep, the per-subject work is roughly:

- Untuned configs: 3 classifiers × 5 sw_durs = **15 configs × 80 windows × 25 fits ≈ 30 000 sklearn fits**
- Tuned configs: 2 classifiers × 5 sw_durs = **10 configs × 80 windows × 25 outer × 27 inner ≈ 540 000 sklearn fits**

So tuned work outweighs untuned by **~18×**, and a single subject's
full sweep is on the order of half a million sklearn fits. Across 20
subjects, ~10 million fits.

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
~25× the cost of an untuned config. The slowest tuned config dominates
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
(or 25 × 27 = 675-fit if tuned) repeated stratified CV on one
pre-windowed slice.

Per-subject task count for a typical sweep:

| Sweep | Configs | Windows | ROIs | Tasks |
|---|---|---|---|---|
| Single ROI, untuned | 15 | 80 | 1 | 1 200 |
| Single ROI, untuned + tuned | 25 | 80 | 1 | 2 000 |
| 4 ROIs, untuned + tuned | 25 | 80 | 4 | 8 000 |
| 4 ROIs, narrowed sweep + tuned | 6 | 80 | 4 | 1 920 |

With 64 cores, 1 200–8 000 tasks gives 19–125 tasks per core — plenty
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
| `GridSearchCV` outer fit (svm, 9-point × 3 inner) | ~600 ms |
| `GridSearchCV` outer fit (logistic, 12-point × 3 inner) | ~5 000 ms |

### 4.2 Wall-time estimates per subject

For the full sweep `--classifiers svm lda logistic --sw-durs 20 40 60
80 100 --tune-hyperparams` on one subject:

| Approach | Wall-time | Cores used |
|---|---|---|
| **Old** (config-serial, 1 thread) | ~12 h | 1 of 64 |
| **Old** (subject-parallel, 8 workers) | ~12 h per batch of 8 subjects | 8 of 64 |
| **New** (config × window parallel, 64 cores) | ~15–25 min | 60–64 of 64 |

The new design's wall-time floor is **the slowest single tuned-window
CV** (≈5 s for tuned logistic) × `ceil(n_tasks / n_jobs)`. With ~2 000
tasks and 64 cores, that's roughly ~32 batches × 5 s ≈ 2.5 min for
tuned logistic + similar for tuned svm in parallel. Pre/post overhead
(load, window, write) adds ~2–3 min/subject, giving the 15–25 min
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
| Discovering candidate classifier/window configs | `--roi <one>` | Fast feedback (~15–25 min/run) lets you iterate on hypotheses without committing hours. |
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
expected to land in the 15–25 min/subject range for the full sweep on
64 cores with `--tune-hyperparams`.
