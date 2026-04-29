#!/usr/bin/env python3
"""
Exploratory analysis: compare classifiers, sliding window durations, and
hyperparameter tuning strategies across one or more ROIs.

Architecture
------------
For each subject (sequentially in the main process):
    1. Load only the requested ROIs from the multi-ROI cache (.npz).
       The .npz is a zip archive — accessing ``data[roi_name]`` reads
       only that ROI's bytes, so we never materialize the other ~15 ROIs
       packed into the file.
    2. For each (roi × sw_dur), pre-window the data once.
    3. Run all (roi × classifier × sw_dur × tuned × window) cells in
       parallel via joblib.  This is the unit of work that fills the box
       on a 64-core machine: typically thousands of uniform tasks per
       subject, perfect for high-core-count load balancing.

BLAS thread pinning
-------------------
``OMP_NUM_THREADS=1`` (and friends) are set at module import time so
that each joblib worker runs sklearn/numpy single-threaded.  Without
this, 64 workers × 64 BLAS threads each = 4096 threads competing for
cores — measurable slowdown from context switching.

Workflow
--------
- Single-ROI mode (fast iteration): use ``--roi`` to discover good
  configs on a representative region.
- Multi-ROI mode (production characterization): use ``--rois`` to
  amortize subject loads and exploit heterogeneous-task load balancing
  across cores.

Results accumulate across runs: rerunning this script with different
``--classifiers``, ``--sw-durs``, or ``--tune-hyperparams`` values
merges new rows into the existing ``explore_full.csv`` /
``explore_summary.csv`` files (one pair per ROI), deduplicated on
(subject, classifier, sw_dur, sw_step, tuned [, ms]).

Usage
-----
    # Single ROI (fast iteration during model discovery)
    python explore_decoding.py --task overtProd --stim-class prodDiff \\
        --method dSPM --atlas HCPMMP1 --roi Temporal \\
        --classifiers svm lda logistic --sw-durs 40 60 80

    # Multi-ROI (characterization once configs are narrowed)
    python explore_decoding.py --task overtProd --stim-class prodDiff \\
        --method dSPM --atlas HCPMMP1 --rois Temporal vSMC IFG DLPFC \\
        --classifiers svm logistic --sw-durs 40 60 --tune-hyperparams

Output
------
For each ROI:
    .../<roi_name>/explore_full.csv     — accuracy at every time window
    .../<roi_name>/explore_summary.csv  — peak/mean per (subject x config)
"""
import os
# Pin BLAS threads to 1 BEFORE numpy import — otherwise each of 64
# workers spawns 64 BLAS threads and the box thrashes.  Joblib's loky
# backend propagates these env vars to child processes.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('BLIS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import argparse
import gc
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    SUBJECT_IDS, SW_STEP_SIZE, SVM_OUTPUT_ROOT,
    SPEECH_ROIS, BASELINE_WINDOWS, DECODE_TMIN,
    SVM_C, PSEUDO_TRIAL_SIZE,
    find_cached_npz, explore_run_segment,
)
from data_loader import load_subject_epochs
from forward_model import setup_fsaverage, make_forward, build_roi_labels
from inverse_pipelines import run_dspm, run_lcmv
from svm_decoding import (
    extract_roi_data_vertices,
    extract_roi_data_pca_flip,
    prepare_windowed_data,
    decode_one_window,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Explore decoding configurations for one or more ROIs',
    )
    parser.add_argument('--task', required=True,
                        choices=['perception', 'overtProd'])
    parser.add_argument('--stim-class', required=True,
                        choices=['prodDiff', 'percDiff'])
    parser.add_argument('--method', required=True,
                        choices=['dSPM', 'LCMV'])
    parser.add_argument('--atlas', default='aparc',
                        choices=['aparc', 'HCPMMP1', 'Schaefer200', 'custom'])
    roi_grp = parser.add_mutually_exclusive_group(required=True)
    roi_grp.add_argument('--roi',
                         help='Single ROI (fast-iteration mode, case-insensitive)')
    roi_grp.add_argument('--rois', nargs='+',
                         help='Multiple ROIs in one invocation (case-insensitive). '
                              'Amortizes subject loads and improves core utilization.')
    parser.add_argument('--feature-mode', default='pca_flip',
                        choices=['pca_flip', 'vertex_pca', 'vertex_selectkbest',
                                 'vertex_selectkbest_all'])
    parser.add_argument('--subjects', nargs='+', default=None,
                        help='Subjects to process (default: all)')
    parser.add_argument('--classifiers', nargs='+',
                        default=['svm', 'lda', 'logistic'],
                        choices=['svm', 'lda', 'logistic'])
    parser.add_argument('--sw-durs', nargs='+', type=int,
                        default=[40, 60, 80],
                        help='Sliding window durations in ms')
    parser.add_argument('--sw-step', type=int, default=SW_STEP_SIZE,
                        help=f'Sliding window step in ms (default: {SW_STEP_SIZE})')
    parser.add_argument('--tune-hyperparams', action='store_true', default=False,
                        help='Add nested-CV-tuned variant for each non-LDA config')
    parser.add_argument('--svm-c', type=float, default=SVM_C)
    parser.add_argument('--leakage-correction', action='store_true', default=False)
    parser.add_argument('--pseudo-trial-size', type=int, default=PSEUDO_TRIAL_SIZE)
    parser.add_argument('--n-jobs', type=int, default=64,
                        help='Worker processes for the (roi × config × window) '
                             'pool.  Default 64 (one per physical core on the '
                             '64-core EPYC workstation).  Workers are pinned to '
                             '1 BLAS thread each — do not exceed the physical '
                             'core count.')
    parser.add_argument('--random-state', type=int, default=42)
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────
# ROI resolution and cache loading
# ──────────────────────────────────────────────────────────────────────
def _resolve_roi_names(roi_dict, requested, atlas):
    """Case-insensitive resolution to canonical ROI names in roi_dict."""
    lower_map = {k.lower(): k for k in roi_dict}
    resolved, missing = [], []
    for name in requested:
        actual = lower_map.get(name.lower())
        if actual:
            resolved.append(actual)
        else:
            missing.append(name)
    if missing:
        print(f'ERROR: ROIs not found in {atlas} atlas: {missing}')
        print(f'Available ROIs: {sorted(roi_dict.keys())}')
        sys.exit(1)
    return resolved


def _load_rois_from_cache(npz_path, roi_names, feature_mode):
    """Load only the requested ROIs from a multi-ROI .npz cache.

    The .npz is a zip archive — ``data[name]`` only reads that array's
    bytes from the zip.  Iterating over only the requested ROI keys
    avoids materializing the other ROIs (a 7 GB cache with ~16 ROIs
    means ~400-500 MB per ROI, so loading 1 of 16 reads ~17× less data).
    """
    data = np.load(npz_path, allow_pickle=True)
    available = set(data.files)
    missing = [r for r in roi_names if r not in available]
    if missing:
        data.close()
        return None, missing

    y = np.array(data['y'])
    times = np.array(data['times'])
    sfreq = float(data['sfreq'])

    rois = {}
    for name in roi_names:
        arr = np.array(data[name])
        if feature_mode == 'pca_flip':
            if arr.ndim == 3 and arr.shape[2] == 1:
                arr = arr[:, :, 0]
        else:
            if arr.ndim == 3:
                arr = arr.transpose(0, 2, 1)
        rois[name] = arr
    data.close()
    return {'rois': rois, 'y': y, 'times': times, 'sfreq': sfreq}, []


def _compute_subject_rois(subj_id, task, stim_class, method, feature_mode,
                          roi_dict_subset, fwd, src):
    """Compute the requested ROIs from scratch when no cache exists."""
    baseline_tmin, baseline_tmax = BASELINE_WINDOWS[task]
    try:
        epochs, y, sfreq = load_subject_epochs(subj_id, task, stim_class)
    except FileNotFoundError as e:
        print(f'  SKIPPING {subj_id}: {e}')
        return None

    if method == 'dSPM':
        stcs = run_dspm(epochs, fwd, baseline_tmin, baseline_tmax)
    elif method == 'LCMV':
        stcs = run_lcmv(epochs, fwd, baseline_tmin, baseline_tmax)

    rois = {}
    if feature_mode == 'pca_flip':
        roi_labels = list(roi_dict_subset.values())
        X_all = extract_roi_data_pca_flip(stcs, roi_labels, src)
        for i, name in enumerate(roi_dict_subset):
            rois[name] = X_all[:, i, :]
    else:
        for name, label in roi_dict_subset.items():
            rois[name] = extract_roi_data_vertices(stcs, label)

    return {'rois': rois, 'y': y, 'times': epochs.times, 'sfreq': sfreq}


def _load_subject(subj_id, task, stim_class, method, feature_mode,
                  atlas, leakage_correction, roi_names, roi_dict, fwd, src):
    """Load requested ROIs for one subject — cache if available, else compute."""
    cached_npz = find_cached_npz(task, method, atlas, feature_mode,
                                 leakage_correction, subj_id, stim_class)
    if cached_npz is not None:
        print(f'  Loading cache: {cached_npz.name}')
        t0 = time.time()
        result, missing = _load_rois_from_cache(
            cached_npz, roi_names, feature_mode,
        )
        if result is None:
            print(f'  WARNING: ROIs missing from cache: {missing} — skipping subject')
            return None
        print(f'  Loaded {len(result["rois"])} ROI(s) in {time.time() - t0:.1f}s')
        return result

    print(f'  No cache — computing inverse for {subj_id}...')
    sub_dict = {n: roi_dict[n] for n in roi_names}
    return _compute_subject_rois(
        subj_id, task, stim_class, method, feature_mode,
        sub_dict, fwd, src,
    )


# ──────────────────────────────────────────────────────────────────────
# Parallel decode worker
# ──────────────────────────────────────────────────────────────────────
def _build_configs(classifiers, sw_durs, tune_hyperparams):
    """Enumerate (classifier, sw_dur, tuned) combinations."""
    configs = []
    for clf in classifiers:
        for sw in sw_durs:
            configs.append((clf, sw, False))
            if tune_hyperparams and clf != 'lda':
                configs.append((clf, sw, True))
    return configs


def _decode_window_task(roi_name, classifier, sw_dur, tuned, w_idx, ms,
                        X_windowed, y, feature_mode, n_features,
                        svm_c, pseudo_trial_size, random_state):
    """Joblib worker: decode one (roi, config, window) cell.

    X_windowed is the full per-(roi, sw_dur) array; only column ``w_idx``
    is used.  Passing the full array lets joblib's auto-memmap share one
    copy across all workers.
    """
    entry = decode_one_window(
        X_windowed[:, :, w_idx], y, classifier, feature_mode, n_features,
        svm_c=svm_c, tune_hyperparams=tuned,
        pseudo_trial_size=pseudo_trial_size, random_state=random_state,
    )
    entry.update({
        'roi': roi_name,
        'classifier': classifier,
        'sw_dur': sw_dur,
        'tuned': tuned,
        'ms': float(ms),
    })
    return entry


def _process_subject(subj_id, subject_data, configs, args, decode_tmin):
    """Drive the parallel sweep for one subject across all requested ROIs.

    Pre-windows data once per (roi, sw_dur), builds a flat (roi × cfg ×
    window) task list, then submits one Parallel(...) call so the worker
    pool stays saturated and load-balanced.
    """
    rois_data = subject_data['rois']
    y = subject_data['y']
    times = subject_data['times']
    sfreq = subject_data['sfreq']

    # Pre-window per (roi, sw_dur) so the same X_windowed serves every
    # classifier × tuned variant.  Each X_windowed is shared by ~6
    # configs × n_windows tasks via joblib auto-memmap.
    print(f'  Pre-windowing for {len(rois_data)} ROI(s) × '
          f'{len(args.sw_durs)} sw_dur(s)...')
    windowed = {}
    for roi_name in rois_data:
        for sw_dur in args.sw_durs:
            X_w, ms_arr, n_feats = prepare_windowed_data(
                rois_data[roi_name], sfreq, sw_dur, args.sw_step,
                tmin=times[0], decode_tmin=decode_tmin, times=times,
                verbose=False,
            )
            windowed[(roi_name, sw_dur)] = (X_w, ms_arr, n_feats)

    # Build flat (roi × cfg × window) task list
    tasks = []
    for roi_name in rois_data:
        for clf, sw_dur, tuned in configs:
            X_w, ms_arr, n_feats = windowed[(roi_name, sw_dur)]
            n_windows = X_w.shape[2]
            for w in range(n_windows):
                tasks.append((
                    roi_name, clf, sw_dur, tuned, w, ms_arr[w],
                    X_w, y, args.feature_mode, n_feats, args.svm_c,
                    args.pseudo_trial_size, args.random_state,
                ))

    n_tasks = len(tasks)
    print(f'  Dispatching {n_tasks} (roi × cfg × window) tasks '
          f'across {args.n_jobs} workers...')
    t0 = time.time()
    results = Parallel(
        n_jobs=args.n_jobs, backend='loky', verbose=5,
    )(
        delayed(_decode_window_task)(*t) for t in tasks
    )
    elapsed = time.time() - t0
    print(f'  Subject {subj_id} done in {elapsed/60:.2f} min '
          f'({n_tasks / elapsed:.1f} tasks/s)')

    rows = []
    for entry in results:
        row = {
            'subject': subj_id,
            'roi': entry['roi'],
            'classifier': entry['classifier'],
            'sw_dur': entry['sw_dur'],
            'sw_step': args.sw_step,
            'tuned': entry['tuned'],
            'ms': entry['ms'],
            'accuracy': entry['SVM_acc'],
        }
        for pname, pval in entry.get('best_params_mode', {}).items():
            row[f'best_{pname}'] = pval
            row[f'best_{pname}_freq'] = entry['best_params_freq'][pname]
        rows.append(row)

    del windowed, tasks, results
    gc.collect()
    return rows


# ──────────────────────────────────────────────────────────────────────
# CSV merge
# ──────────────────────────────────────────────────────────────────────
def _merge_csv(df_new, csv_path, dedupe_keys):
    """Merge new rows into existing CSV, deduping on the given keys.

    New rows override matching rows from disk (``keep='last'``), so a
    rerun with different code or parameters supersedes stale values for
    the same (subject, classifier, sw_dur, sw_step, tuned [, ms]) cell.
    """
    if csv_path.exists():
        df_existing = pd.read_csv(csv_path)
        combined = pd.concat([df_existing, df_new], ignore_index=True)
        combined = combined.drop_duplicates(subset=dedupe_keys, keep='last')
    else:
        combined = df_new
    combined.to_csv(csv_path, index=False)
    return combined


def _save_results_for_roi(df_roi, args, roi_name):
    """Write per-ROI explore_full.csv and explore_summary.csv."""
    run_seg = explore_run_segment(
        args.leakage_correction, args.pseudo_trial_size, args.svm_c,
    )
    out_dir = (
        SVM_OUTPUT_ROOT / 'explore' / args.task / args.method
        / args.atlas / args.feature_mode / args.stim_class
        / run_seg / roi_name
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    full_csv = out_dir / 'explore_full.csv'
    df_full = _merge_csv(
        df_roi, full_csv,
        dedupe_keys=['subject', 'classifier', 'sw_dur', 'sw_step', 'tuned', 'ms'],
    )
    print(f'  [{roi_name}] full: {full_csv.name} '
          f'({len(df_full)} rows total, '
          f'{df_roi["subject"].nunique()} subjects this run)')

    summary_rows = []
    for (subj, clf, sw, tuned), grp in df_roi.groupby(
        ['subject', 'classifier', 'sw_dur', 'tuned'],
    ):
        peak_idx = grp['accuracy'].idxmax()
        summary_rows.append({
            'subject': subj,
            'classifier': clf,
            'sw_dur': sw,
            'sw_step': args.sw_step,
            'tuned': tuned,
            'peak_acc': grp['accuracy'].max(),
            'peak_ms': grp.loc[peak_idx, 'ms'],
            'mean_acc': grp['accuracy'].mean(),
        })
    df_summary_new = pd.DataFrame(summary_rows)

    summary_csv = out_dir / 'explore_summary.csv'
    df_summary = _merge_csv(
        df_summary_new, summary_csv,
        dedupe_keys=['subject', 'classifier', 'sw_dur', 'sw_step', 'tuned'],
    )
    print(f'  [{roi_name}] summary: {summary_csv.name} ({len(df_summary)} rows)')
    return df_summary_new


def _print_run_summary(df_summary_new, roi_name, atlas, n_subjects):
    print(f'\n{"=" * 60}')
    print(f'Run Summary: {roi_name} ({atlas})')
    print(f'{"=" * 60}')
    if n_subjects > 1:
        group_avg = df_summary_new.groupby(
            ['classifier', 'sw_dur', 'tuned'],
        ).agg(
            mean_peak_acc=('peak_acc', 'mean'),
            std_peak_acc=('peak_acc', 'std'),
            mean_mean_acc=('mean_acc', 'mean'),
        ).reset_index()
        print(f'\nGroup averages (n={n_subjects}):')
        print(f'{"Classifier":<12} {"SW_dur":<8} {"Tuned":<7} '
              f'{"Peak Acc":<16} {"Mean Acc":<10}')
        print('-' * 60)
        for _, row in group_avg.iterrows():
            tuned_str = 'Yes' if row['tuned'] else 'No'
            std = row['std_peak_acc'] if not np.isnan(row['std_peak_acc']) else 0.0
            print(f'{row["classifier"]:<12} {int(row["sw_dur"]):<8} '
                  f'{tuned_str:<7} '
                  f'{row["mean_peak_acc"]:.3f}+/-{std:.3f}  '
                  f'{row["mean_mean_acc"]:.3f}')
    else:
        print(f'\n{"Classifier":<12} {"SW_dur":<8} {"Tuned":<7} '
              f'{"Peak Acc":<10} {"Peak ms":<10} {"Mean Acc":<10}')
        print('-' * 60)
        for _, row in df_summary_new.iterrows():
            tuned_str = 'Yes' if row['tuned'] else 'No'
            print(f'{row["classifier"]:<12} {int(row["sw_dur"]):<8} '
                  f'{tuned_str:<7} '
                  f'{row["peak_acc"]:<10.3f} {row["peak_ms"]:<10.1f} '
                  f'{row["mean_acc"]:<10.3f}')


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    subjects = args.subjects if args.subjects else SUBJECT_IDS
    decode_tmin = DECODE_TMIN[args.task]
    requested_rois = [args.roi] if args.roi else args.rois

    configs = _build_configs(args.classifiers, args.sw_durs, args.tune_hyperparams)

    print(f'Explore decoding configurations')
    print(f'  Task:         {args.task}')
    print(f'  Stim class:   {args.stim_class}')
    print(f'  Method:       {args.method}')
    print(f'  Atlas:        {args.atlas}')
    print(f'  ROIs:         {requested_rois}')
    print(f'  Feature mode: {args.feature_mode}')
    print(f'  Classifiers:  {args.classifiers}')
    print(f'  SW durations: {args.sw_durs} ms')
    print(f'  SW step:      {args.sw_step} ms')
    print(f'  Tune HP:      {args.tune_hyperparams}')
    print(f'  Workers:      {args.n_jobs} (BLAS pinned to 1/worker)')
    print(f'  Subjects:     {len(subjects)}')
    print(f'  Configs/ROI:  {len(configs)} '
          f'({"untuned + tuned" if args.tune_hyperparams else "untuned only"})')
    print()

    print('Setting up fsaverage source space and ROI labels...')
    subjects_dir, fs_dir, src, bem = setup_fsaverage()
    if args.atlas in SPEECH_ROIS:
        roi_dict_full = build_roi_labels(
            subjects_dir, atlas=args.atlas,
            composite_rois=SPEECH_ROIS[args.atlas],
        )
    else:
        roi_dict_full = build_roi_labels(subjects_dir, atlas=args.atlas)

    roi_names = _resolve_roi_names(roi_dict_full, requested_rois, args.atlas)
    print(f'Resolved ROIs: {roi_names}')

    # Forward model only built if at least one subject lacks cache
    any_uncached = any(
        find_cached_npz(args.task, args.method, args.atlas, args.feature_mode,
                        args.leakage_correction, s, args.stim_class) is None
        for s in subjects
    )
    if any_uncached:
        print('\nBuilding forward solution (uncached subjects detected)...')
        first_epochs, _, _ = load_subject_epochs(
            subjects[0], args.task, args.stim_class,
        )
        fwd = make_forward(first_epochs.info, src, bem)
        del first_epochs
    else:
        print('\nAll subjects cached — skipping forward model build.')
        fwd = None
    del bem
    gc.collect()

    # ── Subject-sequential outer loop ────────────────────────────────
    total_start = time.time()
    rows_by_roi = {roi: [] for roi in roi_names}

    for s_idx, subj_id in enumerate(subjects, 1):
        print(f'\n{"=" * 60}')
        print(f'Subject {s_idx}/{len(subjects)}: {subj_id}')
        print(f'{"=" * 60}')

        subject_data = _load_subject(
            subj_id, args.task, args.stim_class, args.method,
            args.feature_mode, args.atlas, args.leakage_correction,
            roi_names, roi_dict_full, fwd, src,
        )
        if subject_data is None:
            continue

        subj_rows = _process_subject(
            subj_id, subject_data, configs, args, decode_tmin,
        )

        for row in subj_rows:
            rows_by_roi[row['roi']].append(row)

        del subject_data, subj_rows
        gc.collect()

    total_time = (time.time() - total_start) / 60.0

    if not any(rows_by_roi.values()):
        print('\nNo results to save (all subjects skipped).')
        return

    # ── Save / merge results per ROI ─────────────────────────────────
    for roi_name in roi_names:
        rows = rows_by_roi[roi_name]
        if not rows:
            continue
        df_roi = pd.DataFrame(rows)
        df_summary_new = _save_results_for_roi(df_roi, args, roi_name)
        _print_run_summary(
            df_summary_new, roi_name, args.atlas,
            df_roi['subject'].nunique(),
        )

    print(f'\nDone in {total_time:.1f} minutes')

    # ── Replayable hint for explore_viz_stats ────────────────────────
    clf_str = ' '.join(args.classifiers)
    sw_str = ' '.join(str(s) for s in args.sw_durs)
    extra = []
    if args.leakage_correction:
        extra.append('--leakage-correction')
    if args.pseudo_trial_size != PSEUDO_TRIAL_SIZE:
        extra.append(f'--pseudo-trial-size {args.pseudo_trial_size}')
    if args.svm_c != SVM_C:
        extra.append(f'--svm-c {args.svm_c}')
    extra_str = (' ' + ' '.join(extra)) if extra else ''
    print('\nTo produce figures and cluster-based permutation stats, run:')
    for roi_name in roi_names:
        print(f'  python explore_viz_stats.py \\')
        print(f'      --task {args.task} --stim-class {args.stim_class} \\')
        print(f'      --method {args.method} --atlas {args.atlas} \\')
        print(f'      --feature-mode {args.feature_mode} --roi {roi_name} \\')
        print(f'      --classifiers {clf_str} \\')
        print(f'      --sw-durs {sw_str}{extra_str}')


if __name__ == '__main__':
    main()
