#!/usr/bin/env python3
"""
Exploratory analysis: compare classifiers, sliding window durations, and
hyperparameter tuning strategies for a single ROI.

Sweeps configurations and exports results for easy comparison.  Designed
for rapid parameter testing before committing to full-pipeline runs.

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
from scipy.stats import ttest_1samp
from mne.stats import permutation_cluster_1samp_test
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    SUBJECT_IDS, SW_DUR, SW_STEP_SIZE, SVM_OUTPUT_ROOT,
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
                        choices=['pca_flip', 'vertex_pca', 'vertex_selectkbest'])
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
            rows.append({
                'subject': subj_id,
                'classifier': clf_name,
                'sw_dur': sw_dur,
                'sw_step': sw_step,
                'tuned': tuned,
                'ms': r['ms'],
                'accuracy': r['SVM_acc'],
            })

        accs = [r['SVM_acc'] for r in results]
        peak_acc = max(accs)
        peak_ms = results[np.argmax(accs)]['ms']
        print(f'    [{subj_id}] Peak: {peak_acc:.3f} at {peak_ms:.1f} ms '
              f'({cfg_time:.1f}s)')

    del X_roi, y
    gc.collect()
    return rows


# ──────────────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────────────
CLF_COLORS = {'svm': 'tab:blue', 'lda': 'tab:green', 'logistic': 'tab:orange'}


def plot_classifier_comparison(df, sw_dur, roi_name, n_subjects):
    """Overlay classifier accuracy curves for one sw_dur value.

    Shows group-mean accuracy with SEM bands (when n_subjects > 1).
    """
    sub = df[df['sw_dur'] == sw_dur]

    fig, ax = plt.subplots(figsize=(12, 7))

    for (clf, tuned), grp in sub.groupby(['classifier', 'tuned']):
        time_avg = grp.groupby('ms')['accuracy'].agg(['mean', 'std', 'count'])
        time_avg = time_avg.reset_index()
        time_avg['sem'] = time_avg['std'] / np.sqrt(time_avg['count'])

        color = CLF_COLORS.get(clf, 'gray')
        ls = '--' if tuned else '-'
        label = clf + (' [tuned]' if tuned else '')

        ax.plot(time_avg['ms'], time_avg['mean'], color=color,
                linestyle=ls, linewidth=2, label=label)
        if n_subjects > 1:
            ax.fill_between(time_avg['ms'],
                            time_avg['mean'] - time_avg['sem'],
                            time_avg['mean'] + time_avg['sem'],
                            alpha=0.2, color=color)

    ax.axhline(0.5, color='black', linestyle='--', linewidth=0.8, label='chance')
    ax.axvline(0, color='black', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Time (ms)', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_title(f'{roi_name} — Classifier Comparison '
                 f'(sw_dur={sw_dur}ms, n={n_subjects})', fontsize=16)
    ax.legend(loc='upper left', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)
    plt.tight_layout()
    return fig


def plot_sw_dur_comparison(df, classifier, tuned, roi_name, n_subjects,
                           sw_durs):
    """Overlay accuracy curves for different sw_durs (one classifier)."""
    sub = df[(df['classifier'] == classifier) & (df['tuned'] == tuned)]

    cmap = plt.cm.viridis
    colors = {sw: cmap(i / max(1, len(sw_durs) - 1))
              for i, sw in enumerate(sw_durs)}

    fig, ax = plt.subplots(figsize=(12, 7))

    for sw_dur in sw_durs:
        grp = sub[sub['sw_dur'] == sw_dur]
        if grp.empty:
            continue
        time_avg = grp.groupby('ms')['accuracy'].agg(['mean', 'std', 'count'])
        time_avg = time_avg.reset_index()
        time_avg['sem'] = time_avg['std'] / np.sqrt(time_avg['count'])

        ax.plot(time_avg['ms'], time_avg['mean'], color=colors[sw_dur],
                linewidth=2, label=f'{sw_dur}ms')
        if n_subjects > 1:
            ax.fill_between(time_avg['ms'],
                            time_avg['mean'] - time_avg['sem'],
                            time_avg['mean'] + time_avg['sem'],
                            alpha=0.15, color=colors[sw_dur])

    ax.axhline(0.5, color='black', linestyle='--', linewidth=0.8, label='chance')
    ax.axvline(0, color='black', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Time (ms)', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    tuned_str = ' [tuned]' if tuned else ''
    ax.set_title(f'{roi_name} — Window Duration Sweep '
                 f'({classifier}{tuned_str}, n={n_subjects})', fontsize=16)
    ax.legend(loc='upper left', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)
    plt.tight_layout()
    return fig


def plot_peak_accuracy_heatmap(df_summary, roi_name, n_subjects):
    """Heatmap of mean peak accuracy by (classifier x sw_dur)."""
    avg = df_summary.groupby(
        ['classifier', 'sw_dur', 'tuned']
    )['peak_acc'].mean().reset_index()

    avg['config'] = avg.apply(
        lambda r: r['classifier'] + (' [tuned]' if r['tuned'] else ''),
        axis=1,
    )

    pivot = avg.pivot(index='config', columns='sw_dur', values='peak_acc')

    fig, ax = plt.subplots(
        figsize=(max(8, len(pivot.columns) * 1.5),
                 max(4, len(pivot) * 0.8 + 1)),
    )
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax,
                vmin=0.48, linewidths=0.5,
                cbar_kws={'label': 'Peak Accuracy'})
    ax.set_xlabel('Sliding Window Duration (ms)', fontsize=13)
    ax.set_ylabel('Configuration', fontsize=13)
    ax.set_title(f'{roi_name} — Peak Accuracy by Configuration '
                 f'(n={n_subjects})', fontsize=15)
    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────
# Group-level statistics
# ──────────────────────────────────────────────────────────────
N_EXPLORE_PERMUTATIONS = 512


def find_contiguous_clusters(mask):
    """Find start/end indices of contiguous True runs."""
    clusters = []
    in_cluster = False
    start = None
    for i, val in enumerate(mask):
        if val and not in_cluster:
            start = i
            in_cluster = True
        elif not val and in_cluster:
            clusters.append((start, i - 1))
            in_cluster = False
    if in_cluster:
        clusters.append((start, len(mask) - 1))
    return clusters


def compute_explore_stats(df, subjects, roi_name):
    """Cluster-based permutation test against chance for each config.

    Requires >= 3 subjects.  Returns a stats DataFrame and a dict of
    per-config cluster details for the summary log.
    """
    n_subjects = len(subjects)
    if n_subjects < 3:
        print('  Skipping stats: need at least 3 subjects for permutation test')
        return None

    stats_rows = []

    for (clf, sw_dur, tuned), grp in df.groupby(
        ['classifier', 'sw_dur', 'tuned']
    ):
        ms_values = np.array(sorted(grp['ms'].unique()))
        n_times = len(ms_values)

        # Build accuracy matrix: (n_subjects, n_times)
        acc_matrix = np.full((n_subjects, n_times), np.nan)
        for i, subj in enumerate(subjects):
            subj_data = grp[grp['subject'] == subj].sort_values('ms')
            if len(subj_data) == n_times:
                acc_matrix[i] = subj_data['accuracy'].values

        # Drop subjects with missing data
        valid = ~np.isnan(acc_matrix).any(axis=1)
        X = acc_matrix[valid] - 0.5  # center at chance
        n_valid = X.shape[0]
        if n_valid < 3:
            continue

        # Cluster-based permutation test (one-sided: accuracy > chance)
        T_obs, clusters, pv, _ = permutation_cluster_1samp_test(
            X, threshold=None, n_permutations=N_EXPLORE_PERMUTATIONS,
            tail=1, out_type='mask', verbose=False,
        )

        n_sig = sum(1 for p in pv if p < 0.05)
        mean_acc = acc_matrix[valid].mean(axis=0)
        peak_acc = mean_acc.max()
        peak_ms = ms_values[np.argmax(mean_acc)]
        earliest_onset = None

        if n_sig > 0:
            for ic, cpv in enumerate(pv):
                if cpv < 0.05:
                    sig_idx = np.where(clusters[ic])[0]
                    onset = ms_values[sig_idx[0]]
                    if earliest_onset is None or onset < earliest_onset:
                        earliest_onset = onset

        tuned_str = 'Yes' if tuned else 'No'
        stats_rows.append({
            'classifier': clf,
            'sw_dur': sw_dur,
            'tuned': tuned_str,
            'n_subjects': n_valid,
            'n_sig_clusters': n_sig,
            'earliest_onset_ms': earliest_onset,
            'peak_acc': peak_acc,
            'peak_ms': peak_ms,
        })

    return pd.DataFrame(stats_rows)


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

    # ── Save results ────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)

    out_dir = (
        SVM_OUTPUT_ROOT / 'explore' / args.task / args.method
        / args.atlas / args.feature_mode / args.stim_class / roi_name
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    full_csv = out_dir / 'explore_full.csv'
    df.to_csv(full_csv, index=False)
    print(f'\nFull results: {full_csv}')

    # ── Summary ─────────────────────────────────────────────────────
    summary_rows = []
    for (subj, clf, sw, tuned), grp in df.groupby(
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

    df_summary = pd.DataFrame(summary_rows)
    summary_csv = out_dir / 'explore_summary.csv'
    df_summary.to_csv(summary_csv, index=False)
    print(f'Summary:      {summary_csv}')

    # ── Print summary table ─────────────────────────────────────────
    print(f'\n{"="*60}')
    print(f'Exploration Summary: {roi_name} ({args.atlas})')
    print(f'{"="*60}')

    # Group-level averages if multiple subjects
    if len(subjects) > 1:
        group_avg = df_summary.groupby(
            ['classifier', 'sw_dur', 'tuned']
        ).agg(
            mean_peak_acc=('peak_acc', 'mean'),
            std_peak_acc=('peak_acc', 'std'),
            mean_mean_acc=('mean_acc', 'mean'),
        ).reset_index()

        print(f'\nGroup averages (n={len(subjects)}):')
        print(f'{"Classifier":<12} {"SW_dur":<8} {"Tuned":<7} '
              f'{"Peak Acc":<12} {"Mean Acc":<10}')
        print('-' * 55)
        for _, row in group_avg.iterrows():
            tuned_str = 'Yes' if row['tuned'] else 'No'
            print(f'{row["classifier"]:<12} {int(row["sw_dur"]):<8} '
                  f'{tuned_str:<7} '
                  f'{row["mean_peak_acc"]:.3f}+/-{row["std_peak_acc"]:.3f}  '
                  f'{row["mean_mean_acc"]:.3f}')
    else:
        print(f'\n{"Classifier":<12} {"SW_dur":<8} {"Tuned":<7} '
              f'{"Peak Acc":<10} {"Peak ms":<10} {"Mean Acc":<10}')
        print('-' * 60)
        for _, row in df_summary.iterrows():
            tuned_str = 'Yes' if row['tuned'] else 'No'
            print(f'{row["classifier"]:<12} {int(row["sw_dur"]):<8} '
                  f'{tuned_str:<7} '
                  f'{row["peak_acc"]:<10.3f} {row["peak_ms"]:<10.1f} '
                  f'{row["mean_acc"]:<10.3f}')

    # ── Figures ──────────────────────────────────────────────────────
    n_subjects = df['subject'].nunique()
    fig_dir = out_dir / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Classifier comparison (one figure per sw_dur)
    for sw_dur in args.sw_durs:
        fig = plot_classifier_comparison(df, sw_dur, roi_name, n_subjects)
        fname = fig_dir / f'clf_comparison_sw{sw_dur}ms.svg'
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f'  Saved: {fname}')

    # SW duration sweep (one figure per classifier × tuned combo)
    for clf_name in args.classifiers:
        for tuned in [False] + ([True] if args.tune_hyperparams and clf_name != 'lda' else []):
            fig = plot_sw_dur_comparison(df, clf_name, tuned, roi_name,
                                         n_subjects, args.sw_durs)
            t_str = '_tuned' if tuned else ''
            fname = fig_dir / f'sw_sweep_{clf_name}{t_str}.svg'
            fig.savefig(fname, dpi=150)
            plt.close(fig)
            print(f'  Saved: {fname}')

    # Peak accuracy heatmap
    fig = plot_peak_accuracy_heatmap(df_summary, roi_name, n_subjects)
    fname = fig_dir / 'peak_accuracy_heatmap.svg'
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f'  Saved: {fname}')

    # ── Group stats ────────────────────────────────────────────────
    if n_subjects >= 3:
        print(f'\nRunning cluster-based permutation tests '
              f'({N_EXPLORE_PERMUTATIONS} permutations)...')
        df_stats = compute_explore_stats(df, subjects, roi_name)
        if df_stats is not None and len(df_stats) > 0:
            stats_csv = out_dir / 'explore_stats.csv'
            df_stats.to_csv(stats_csv, index=False)
            print(f'Stats:        {stats_csv}')

            print(f'\n{"Classifier":<12} {"SW_dur":<8} {"Tuned":<7} '
                  f'{"Sig Clusters":<14} {"Onset (ms)":<12} '
                  f'{"Peak Acc":<10} {"Peak ms":<10}')
            print('-' * 75)
            for _, row in df_stats.iterrows():
                onset_str = (f'{row["earliest_onset_ms"]:.1f}'
                             if row['earliest_onset_ms'] is not None
                             else '—')
                print(f'{row["classifier"]:<12} {int(row["sw_dur"]):<8} '
                      f'{row["tuned"]:<7} '
                      f'{row["n_sig_clusters"]:<14} {onset_str:<12} '
                      f'{row["peak_acc"]:<10.3f} {row["peak_ms"]:<10.1f}')
    else:
        print('\nSkipping group stats (need >= 3 subjects for permutation test)')

    print(f'\nDone in {total_time:.1f} minutes')


if __name__ == '__main__':
    main()
