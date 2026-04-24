#!/usr/bin/env python3
"""
Exploratory analysis: compare classifiers, sliding window durations, and
hyperparameter tuning strategies for a single ROI.

Runs the decoding sweep and writes per-time-window accuracy to CSV.
Figures and group-level statistics are produced separately by
``explore_viz_stats.py`` so the same decoding output can be re-plotted
and re-analyzed without re-running the SVM loop.

Results accumulate across runs: rerunning this script with different
``--classifiers``, ``--sw-durs``, or ``--tune-hyperparams`` values
merges new rows into the existing ``explore_full.csv`` /
``explore_summary.csv`` files, deduplicated on
(subject, classifier, sw_dur, sw_step, tuned [, ms]).  Running SVM first
and LDA second leaves both classifiers in the CSV.

Usage:
    # Quick comparison of classifiers on one subject
    python explore_decoding.py --task overtProd --stim-class prodDiff \
        --method dSPM --atlas HCPMMP1 --roi Temporal \
        --subjects EEGPROD4001

    # Full sweep with hyperparameter tuning
    python explore_decoding.py --task overtProd --stim-class prodDiff \
        --method dSPM --atlas HCPMMP1 --roi Temporal \
        --classifiers svm lda logistic \
        --sw-durs 20 40 60 80 100 \
        --tune-hyperparams

    # Multiple subjects for group-level comparison
    python explore_decoding.py --task overtProd --stim-class prodDiff \
        --method dSPM --atlas Schaefer200 --roi vSMC \
        --subjects EEGPROD4001 EEGPROD4003 EEGPROD4005

    # Then produce figures and cluster-permutation stats:
    python explore_viz_stats.py --task overtProd --stim-class prodDiff \
        --method dSPM --atlas HCPMMP1 --roi Temporal

Output:
    explore_full.csv    — accuracy at every time window for every config
    explore_summary.csv — peak/mean accuracy per (subject x config)
"""
import argparse
import gc
import os
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    SUBJECT_IDS, SW_STEP_SIZE, SVM_OUTPUT_ROOT,
    SPEECH_ROIS, BASELINE_WINDOWS, DECODE_TMIN,
    SVM_C, PSEUDO_TRIAL_SIZE,
    find_cached_npz,
)
from data_loader import load_subject_epochs
from forward_model import setup_fsaverage, make_forward, build_roi_labels
from inverse_pipelines import run_dspm, run_lcmv
from svm_decoding import (
    extract_roi_data_vertices,
    extract_roi_data_pca_flip,
    sliding_window_svm_decode,
)
from run_source_svm import _load_cached_roi_data, filter_roi_dict


def parse_args():
    parser = argparse.ArgumentParser(
        description='Explore decoding configurations for a single ROI'
    )
    parser.add_argument('--task', required=True,
                        choices=['perception', 'overtProd'])
    parser.add_argument('--stim-class', required=True,
                        choices=['prodDiff', 'percDiff'])
    parser.add_argument('--method', required=True,
                        choices=['dSPM', 'LCMV'])
    parser.add_argument('--atlas', default='aparc',
                        choices=['aparc', 'HCPMMP1', 'Schaefer200', 'custom'])
    parser.add_argument('--roi', required=True,
                        help='ROI name to decode (case-insensitive)')
    parser.add_argument('--feature-mode', default='pca_flip',
                        choices=['pca_flip', 'vertex_pca', 'vertex_selectkbest',
                                 'vertex_selectkbest_all'])
    parser.add_argument('--subjects', nargs='+', default=None,
                        help='Subjects to process (default: all)')
    parser.add_argument('--classifiers', nargs='+',
                        default=['svm', 'lda', 'logistic'],
                        choices=['svm', 'lda', 'logistic'],
                        help='Classifiers to compare (default: svm lda logistic)')
    parser.add_argument('--sw-durs', nargs='+', type=int,
                        default=[20, 40, 60, 80, 100],
                        help='Sliding window durations in ms (default: 20 40 60 80 100)')
    parser.add_argument('--sw-step', type=int, default=SW_STEP_SIZE,
                        help=f'Sliding window step in ms (default: {SW_STEP_SIZE})')
    parser.add_argument('--tune-hyperparams', action='store_true', default=False,
                        help='Also run each classifier with nested CV tuning '
                             '(doubles the number of configs, no effect for lda)')
    parser.add_argument('--svm-c', type=float, default=SVM_C,
                        help=f'Default C for SVM/logistic (default: {SVM_C})')
    parser.add_argument('--leakage-correction', action='store_true', default=False)
    parser.add_argument('--pseudo-trial-size', type=int, default=PSEUDO_TRIAL_SIZE)
    parser.add_argument('--n-jobs', type=int, default=1,
                        help='Number of parallel workers across subjects (default: 1)')
    return parser.parse_args()


def _load_subject_roi(subj_id, task_cond, stim_class, method, feature_mode,
                      atlas, leakage_correction, roi_name,
                      fwd, src, roi_dict):
    """Load ROI data for one subject, from cache or by computing inverse.

    Returns (X_roi, y, times, sfreq) or None if subject must be skipped.
    """
    baseline_tmin, baseline_tmax = BASELINE_WINDOWS[task_cond]

    # Try loading from cache
    cached_npz = find_cached_npz(task_cond, method, atlas, feature_mode,
                                 leakage_correction, subj_id, stim_class)
    if cached_npz is not None:
        print(f'  Loading cached data: {cached_npz}')
        roi_data, y, times, sfreq = _load_cached_roi_data(
            cached_npz, feature_mode,
        )
        if roi_name not in roi_data:
            print(f'  WARNING: {roi_name} not in cached file, skipping')
            return None
        return roi_data[roi_name], y, times, sfreq

    # Compute from scratch
    print(f'  No cache found, computing inverse for {subj_id}...')
    try:
        epochs, y, sfreq = load_subject_epochs(subj_id, task_cond, stim_class)
    except FileNotFoundError as e:
        print(f'  SKIPPING {subj_id}: {e}')
        return None

    if method == 'dSPM':
        stcs = run_dspm(epochs, fwd, baseline_tmin, baseline_tmax)
    elif method == 'LCMV':
        stcs = run_lcmv(epochs, fwd, baseline_tmin, baseline_tmax)

    if feature_mode == 'pca_flip':
        roi_labels = [roi_dict[roi_name]]
        X_all = extract_roi_data_pca_flip(stcs, roi_labels, src)
        X_roi = X_all[:, 0, :]
    else:
        X_roi = extract_roi_data_vertices(stcs, roi_dict[roi_name])

    return X_roi, y, epochs.times, sfreq


# ──────────────────────────────────────────────────────────────
# Parallel worker infrastructure
# ──────────────────────────────────────────────────────────────
_fwd = None
_src = None
_roi_dict_full = None


def _init_worker(fwd, src, roi_dict_full):
    """Initialize each worker process with shared forward model data."""
    global _fwd, _src, _roi_dict_full
    _fwd = fwd
    _src = src
    _roi_dict_full = roi_dict_full


def _process_subject(args):
    """Worker: load data for one subject and sweep all configs."""
    (subj_id, task_cond, stim_class, method, feature_mode,
     atlas, leakage_correction, roi_name, configs,
     sw_step, svm_c, pseudo_trial_size, decode_tmin) = args

    result = _load_subject_roi(
        subj_id, task_cond, stim_class, method, feature_mode,
        atlas, leakage_correction, roi_name,
        _fwd, _src, _roi_dict_full,
    )
    if result is None:
        return []

    X_roi, y, times, sfreq = result
    tmin = times[0]
    rows = []

    for cfg_idx, (clf_name, sw_dur, tuned) in enumerate(configs):
        tuned_str = ' [tuned]' if tuned else ''
        print(f'  [{subj_id}] Config {cfg_idx + 1}/{len(configs)}: '
              f'{clf_name}{tuned_str}, sw_dur={sw_dur}ms')

        cfg_start = time.time()
        results = sliding_window_svm_decode(
            X_roi, y, sfreq, sw_dur, sw_step, tmin, decode_tmin,
            feature_mode=feature_mode, times=times,
            classifier=clf_name, svm_c=svm_c,
            tune_hyperparams=tuned,
            pseudo_trial_size=pseudo_trial_size,
        )
        cfg_time = time.time() - cfg_start

        for r in results:
            row = {
                'subject': subj_id,
                'classifier': clf_name,
                'sw_dur': sw_dur,
                'sw_step': sw_step,
                'tuned': tuned,
                'ms': r['ms'],
                'accuracy': r['SVM_acc'],
            }
            # When tuning was active, flatten modal hyperparameters across
            # the 25 outer folds into per-parameter columns (NaN for rows
            # where tuning was not used, e.g. untuned or LDA).
            for pname, pval in r.get('best_params_mode', {}).items():
                row[f'best_{pname}'] = pval
                row[f'best_{pname}_freq'] = r['best_params_freq'][pname]
            rows.append(row)

        accs = [r['SVM_acc'] for r in results]
        peak_acc = max(accs)
        peak_ms = results[np.argmax(accs)]['ms']
        print(f'    [{subj_id}] Peak: {peak_acc:.3f} at {peak_ms:.1f} ms '
              f'({cfg_time:.1f}s)')

    del X_roi, y
    gc.collect()
    return rows


# ──────────────────────────────────────────────────────────────
# Merge helpers
# ──────────────────────────────────────────────────────────────
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


def main():
    args = parse_args()
    subjects = args.subjects if args.subjects else SUBJECT_IDS
    decode_tmin = DECODE_TMIN[args.task]

    # ── Header ──────────────────────────────────────────────────────
    n_configs = len(args.classifiers) * len(args.sw_durs)
    if args.tune_hyperparams:
        # Add tuned variants (except lda which has no tunable params)
        n_tunable = sum(1 for c in args.classifiers if c != 'lda')
        n_configs += n_tunable * len(args.sw_durs)

    print(f'Explore decoding configurations')
    print(f'  Task:         {args.task}')
    print(f'  Stim class:   {args.stim_class}')
    print(f'  Method:       {args.method}')
    print(f'  Atlas:        {args.atlas}')
    print(f'  ROI:          {args.roi}')
    print(f'  Feature mode: {args.feature_mode}')
    print(f'  Classifiers:  {args.classifiers}')
    print(f'  SW durations: {args.sw_durs} ms')
    print(f'  SW step:      {args.sw_step} ms')
    print(f'  Tune HP:      {args.tune_hyperparams}')
    print(f'  Workers:      {args.n_jobs}')
    print(f'  Subjects:     {len(subjects)}')
    print(f'  Total configs: {n_configs} per subject')
    print()

    # ── Setup ───────────────────────────────────────────────────────
    print('Setting up fsaverage source space and ROI labels...')
    subjects_dir, fs_dir, src, bem = setup_fsaverage()

    if args.atlas in SPEECH_ROIS:
        roi_dict = build_roi_labels(subjects_dir, atlas=args.atlas,
                                     composite_rois=SPEECH_ROIS[args.atlas])
    else:
        roi_dict = build_roi_labels(subjects_dir, atlas=args.atlas)

    # Resolve ROI name (case-insensitive)
    roi_dict_full = roi_dict
    roi_dict = filter_roi_dict(roi_dict, [args.roi], args.atlas)
    roi_name = list(roi_dict.keys())[0]

    # Only build the forward solution if at least one subject lacks cached data
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

    # ── Build configuration list ────────────────────────────────────
    configs = []
    for clf_name in args.classifiers:
        for sw_dur in args.sw_durs:
            configs.append((clf_name, sw_dur, False))
            if args.tune_hyperparams and clf_name != 'lda':
                configs.append((clf_name, sw_dur, True))

    # ── Sweep ───────────────────────────────────────────────────────
    total_start = time.time()

    worker_args = [
        (subj_id, args.task, args.stim_class, args.method,
         args.feature_mode, args.atlas, args.leakage_correction,
         roi_name, configs, args.sw_step, args.svm_c,
         args.pseudo_trial_size, decode_tmin)
        for subj_id in subjects
    ]

    if args.n_jobs > 1:
        print(f'Running {len(subjects)} subjects across {args.n_jobs} workers...')
        with Pool(
            processes=args.n_jobs,
            initializer=_init_worker,
            initargs=(fwd, src, roi_dict_full),
        ) as pool:
            results_per_subject = pool.map(_process_subject, worker_args)
        all_rows = [row for subj_rows in results_per_subject for row in subj_rows]
    else:
        # Sequential — use module-level globals directly
        _init_worker(fwd, src, roi_dict_full)
        all_rows = []
        for wa in worker_args:
            all_rows.extend(_process_subject(wa))

    total_time = (time.time() - total_start) / 60.0

    if not all_rows:
        print('\nNo results to save (all subjects skipped).')
        return

    # ── Save / merge results ───────────────────────────────────────
    df_new = pd.DataFrame(all_rows)

    out_dir = (
        SVM_OUTPUT_ROOT / 'explore' / args.task / args.method
        / args.atlas / args.feature_mode / args.stim_class / roi_name
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    full_csv = out_dir / 'explore_full.csv'
    df_full = _merge_csv(
        df_new, full_csv,
        dedupe_keys=['subject', 'classifier', 'sw_dur', 'sw_step', 'tuned', 'ms'],
    )
    print(f'\nFull results: {full_csv} '
          f'({len(df_full)} rows, {df_new["subject"].nunique()} subjects this run)')

    # Per-subject/per-config summary: peak acc, peak ms, mean acc
    summary_rows = []
    for (subj, clf, sw, tuned), grp in df_new.groupby(
        ['subject', 'classifier', 'sw_dur', 'tuned']
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
    print(f'Summary:      {summary_csv} ({len(df_summary)} rows total)')

    # ── Print per-run summary table ─────────────────────────────────
    print(f'\n{"="*60}')
    print(f'Run Summary: {roi_name} ({args.atlas})')
    print(f'{"="*60}')

    if df_new['subject'].nunique() > 1:
        group_avg = df_summary_new.groupby(
            ['classifier', 'sw_dur', 'tuned']
        ).agg(
            mean_peak_acc=('peak_acc', 'mean'),
            std_peak_acc=('peak_acc', 'std'),
            mean_mean_acc=('mean_acc', 'mean'),
        ).reset_index()

        print(f'\nGroup averages (n={df_new["subject"].nunique()}):')
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

    print(f'\nDone in {total_time:.1f} minutes')
    clf_str = ' '.join(args.classifiers)
    sw_str = ' '.join(str(s) for s in args.sw_durs)
    hint_lines = [
        '',
        'To produce figures and cluster-based permutation stats, run:',
        f'  python explore_viz_stats.py \\',
        f'      --task {args.task} --stim-class {args.stim_class} \\',
        f'      --method {args.method} --atlas {args.atlas} \\',
        f'      --feature-mode {args.feature_mode} --roi {roi_name} \\',
        f'      --classifiers {clf_str} \\',
        f'      --sw-durs {sw_str}',
    ]
    print('\n'.join(hint_lines))


if __name__ == '__main__':
    main()
