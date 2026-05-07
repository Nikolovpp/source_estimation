#!/usr/bin/env python3
"""
Decoding runner: subject-sequential, (ROI x window) parallel.

Splits off the decoding half of the former ``run_parallel_lowram.py``.
Source localization is expected to have already been run via
``run_source_localize.py`` so the per-subject ROI .npz caches exist —
this script consumes those caches and runs sliding-window CV.

Why subject-sequential here?
----------------------------
Decoding produces thousands of cheap (ROI x window) tasks per subject
that fill a 64-core box uniformly via joblib's loky backend.  Running
multiple subjects in parallel processes each at a fraction of the cores
and wastes the heterogeneous-task load balancing — see
``optimizing_exploratory_analyses.md``.  Source localization, by
contrast, is heavyweight per subject and runs in
``run_source_localize.py`` with multiprocessing.Pool over subjects.

If a subject's cache does not exist this script errors and points at
``run_source_localize.py`` rather than silently re-running the inverse.

Usage
-----
    python run_decode.py --task overtProd --stim-class prodDiff \\
        --method dSPM --atlas HCPMMP1 --feature-mode vertex_selectkbest \\
        --leakage-correction --classifier logistic --tune-hyperparams \\
        --pseudo-trial-size 5 --n-jobs 64

    # Subset to specific ROIs for faster iteration
    python run_decode.py --task overtProd --stim-class prodDiff \\
        --method dSPM --atlas HCPMMP1 --classifier svm \\
        --roi-subset Temporal vSMC --n-jobs 64
"""
import os
# Pin BLAS threads to 1 BEFORE numpy import — joblib's loky backend
# propagates these env vars to child processes.  Without this each of
# 64 workers spawns 64 BLAS threads and the box thrashes.
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

from joblib import Parallel, delayed

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    SUBJECT_IDS, SW_DUR, SW_STEP_SIZE, DECODE_OUTPUT_ROOT,
    DECODE_TMIN, SPEECH_ROIS,
    DEFAULT_C, PSEUDO_TRIAL_SIZE,
    find_cached_npz,
)
from log_utils import setup_logging
from forward_model import setup_fsaverage, build_roi_labels
from decoding import prepare_windowed_data, decode_one_window
from decoding_io import (
    _save_results, filter_roi_dict, _load_cached_roi_data,
)
from plotting import save_svm_results


def _decode_window_task(roi_name, w_idx, ms, X_windowed, y,
                        feature_mode, n_features, classifier, c,
                        tune_hyperparams, pseudo_trial_size, random_state):
    """Joblib worker: decode one (roi, window) cell.

    ``X_windowed`` is the full per-ROI array; only column ``w_idx`` is
    used.  Passing the full array lets joblib's auto-memmap share one
    copy across all workers.
    """
    entry = decode_one_window(
        X_windowed[:, :, w_idx], y, classifier, feature_mode, n_features,
        c=c, tune_hyperparams=tune_hyperparams,
        pseudo_trial_size=pseudo_trial_size, random_state=random_state,
    )
    entry['roi'] = roi_name
    entry['ms'] = float(ms)
    return entry


def _decode_subject(subj_id, roi_data, y, times, sfreq, args,
                    decode_tmin, c):
    """Run (ROI x window) parallel decoding for one subject's ROI dict.

    Pre-windows each ROI once, builds a flat (roi x window) task list,
    then dispatches one Parallel(...) call so the worker pool stays
    saturated and load-balanced across heterogeneous tasks.
    """
    roi_names = list(roi_data.keys())
    print(f'  Pre-windowing {len(roi_names)} ROI(s)...')
    windowed = {}
    for roi_name in roi_names:
        X_w, ms_arr, n_feats = prepare_windowed_data(
            roi_data[roi_name], sfreq, args.sw_dur, args.sw_step,
            tmin=times[0], decode_tmin=decode_tmin, times=times,
            verbose=False,
        )
        windowed[roi_name] = (X_w, ms_arr, n_feats)

    tasks = []
    for roi_name in roi_names:
        X_w, ms_arr, n_feats = windowed[roi_name]
        n_windows = X_w.shape[2]
        for w in range(n_windows):
            tasks.append((
                roi_name, w, ms_arr[w], X_w, y, args.feature_mode,
                n_feats, args.classifier, c, args.tune_hyperparams,
                args.pseudo_trial_size, args.random_state,
            ))

    n_tasks = len(tasks)
    print(f'  Dispatching {n_tasks} (roi x window) tasks across '
          f'{args.n_jobs} workers...')
    t0 = time.time()
    entries = Parallel(
        n_jobs=args.n_jobs, backend='loky', verbose=5,
    )(
        delayed(_decode_window_task)(*t) for t in tasks
    )
    elapsed = time.time() - t0
    print(f'  {subj_id}: {n_tasks} tasks in {elapsed/60:.2f} min '
          f'({n_tasks / elapsed:.1f} tasks/s)')

    # Reassemble entries into the {roi: [per-window dict, ...]} shape
    # expected by save_svm_results / _save_results.
    results_all_rois = {name: [] for name in roi_names}
    for entry in entries:
        results_all_rois[entry['roi']].append({
            'ms': entry['ms'],
            'mean_list': entry['mean_list'],
            'decode_acc': entry['decode_acc'],
        })
    for name in roi_names:
        results_all_rois[name].sort(key=lambda r: r['ms'])

    del windowed, tasks, entries
    gc.collect()
    return results_all_rois


def parse_args():
    parser = argparse.ArgumentParser(
        description='Subject-sequential, (ROI x window) parallel decoding'
    )
    parser.add_argument('--task', required=True,
                        choices=['perception', 'overtProd'])
    parser.add_argument('--stim-class', required=True,
                        choices=['prodDiff', 'percDiff'])
    parser.add_argument('--method', required=True,
                        choices=['dSPM', 'LCMV'])
    parser.add_argument('--feature-mode', default='pca_flip',
                        choices=['pca_flip', 'vertex_pca', 'vertex_selectkbest',
                                 'vertex_selectkbest_all'])
    parser.add_argument('--subjects', nargs='+', default=None)
    parser.add_argument('--n-jobs', type=int, default=64,
                        help='Worker processes for the (ROI x window) pool. '
                             'Default 64 (one per physical core on the 64-core '
                             'EPYC workstation).  Workers are pinned to 1 BLAS '
                             'thread each — do not exceed the physical core count.')
    parser.add_argument('--sw-dur', type=int, default=SW_DUR)
    parser.add_argument('--sw-step', type=int, default=SW_STEP_SIZE)
    parser.add_argument('--atlas', default='aparc',
                        choices=['aparc', 'HCPMMP1', 'Schaefer200', 'custom'])
    parser.add_argument('--leakage-correction', action='store_true', default=False,
                        help='Read leakage-corrected caches '
                             '(must match what was produced by run_source_localize.py)')
    parser.add_argument('--pseudo-trial-size', type=int, default=PSEUDO_TRIAL_SIZE,
                        help='Pseudo-trial group size; 0 = disabled (default: 0)')
    parser.add_argument('--c', type=float, default=None,
                        help='Regularization parameter C for svm/logistic. '
                             'When omitted, use the per-classifier default '
                             f'from config.DEFAULT_C ({DEFAULT_C}).  '
                             'Ignored when classifier=lda.')
    parser.add_argument('--roi-subset', nargs='+', default=None, metavar='ROI',
                        help='Subset of ROI names to decode (default: all). '
                             'Case-insensitive, e.g., --roi-subset Temporal vSMC')
    parser.add_argument('--classifier', default='svm',
                        choices=['svm', 'lda', 'logistic'],
                        help='Classifier algorithm (default: svm)')
    parser.add_argument('--tune-hyperparams', action='store_true', default=False,
                        help='Enable nested CV for hyperparameter tuning')
    parser.add_argument('--random-state', type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    subjects = args.subjects if args.subjects else SUBJECT_IDS
    decode_tmin = DECODE_TMIN[args.task]

    setup_logging(args.task, args.stim_class, args.method, args.atlas,
                  args.feature_mode, runner_name='decode')

    effective_c = args.c if args.c is not None else DEFAULT_C.get(args.classifier, 1.0)
    c_source = 'explicit' if args.c is not None else f'default for {args.classifier}'

    print(f'Decoding (subject-sequential, ROI x window parallel)')
    print(f'  Task:         {args.task}')
    print(f'  Stim class:   {args.stim_class}')
    print(f'  Method:       {args.method}')
    print(f'  Atlas:        {args.atlas}')
    print(f'  Feature mode: {args.feature_mode}')
    print(f'  Classifier:   {args.classifier}')
    print(f'  Tune HP:      {args.tune_hyperparams}')
    print(f'  C:            {effective_c} ({c_source})')
    print(f'  Pseudo-trial: {args.pseudo_trial_size if args.pseudo_trial_size > 0 else "disabled"}')
    print(f'  Leakage corr: {args.leakage_correction}')
    print(f'  ROI subset:   {args.roi_subset if args.roi_subset else "all"}')
    print(f'  Workers:      {args.n_jobs} (BLAS pinned to 1/worker)')
    print(f'  Subjects:     {len(subjects)}')
    print()

    print('Resolving ROI label set...')
    subjects_dir, _fs_dir, _src, _bem = setup_fsaverage()
    if args.atlas in SPEECH_ROIS:
        roi_dict = build_roi_labels(subjects_dir, atlas=args.atlas,
                                    composite_rois=SPEECH_ROIS[args.atlas])
    else:
        roi_dict = build_roi_labels(subjects_dir, atlas=args.atlas)
    if args.roi_subset:
        roi_dict = filter_roi_dict(roi_dict, args.roi_subset, args.atlas)
    requested_roi_names = list(roi_dict.keys())
    del _src, _bem
    gc.collect()

    total_start = time.time()
    n_ok = 0
    failed = []

    for s_idx, subj_id in enumerate(subjects, 1):
        print(f'\n{"="*60}')
        print(f'Subject {s_idx}/{len(subjects)}: {subj_id}')
        print(f'{"="*60}')

        cached_npz = find_cached_npz(
            args.task, args.method, args.atlas, args.feature_mode,
            args.leakage_correction, subj_id, args.stim_class,
        )
        if cached_npz is None:
            print(f'  ERROR: no cached ROI .npz for {subj_id}.')
            print(f'  Run source localization first:')
            print(f'    python run_source_localize.py --task {args.task} '
                  f'--stim-class {args.stim_class} \\')
            print(f'        --method {args.method} --atlas {args.atlas} '
                  f'--feature-mode {args.feature_mode}'
                  f'{" --leakage-correction" if args.leakage_correction else ""}')
            failed.append(subj_id)
            continue

        print(f'  Loading cache: {cached_npz.name}')
        t0 = time.time()
        roi_data, y, times, sfreq = _load_cached_roi_data(
            cached_npz, args.feature_mode, roi_subset=requested_roi_names,
        )
        if roi_data is None:
            failed.append(subj_id)
            continue
        print(f'  Loaded {len(roi_data)} ROI(s) in {time.time() - t0:.1f}s')

        try:
            results_all_rois = _decode_subject(
                subj_id, roi_data, y, times, sfreq, args,
                decode_tmin, effective_c,
            )
        except Exception as e:
            print(f'  FAILED {subj_id}: {e}')
            failed.append(subj_id)
            del roi_data
            gc.collect()
            continue

        del roi_data
        gc.collect()

        save_svm_results(
            results_all_rois, subj_id, args.task, args.stim_class,
            args.method, args.feature_mode, args.sw_dur, args.sw_step,
            atlas=args.atlas, leakage_correction=args.leakage_correction,
            pseudo_trial_size=args.pseudo_trial_size,
        )
        _save_results(
            subj_id, args.task, args.stim_class, args.method,
            args.feature_mode, args.sw_dur, args.sw_step,
            results_all_rois, DECODE_OUTPUT_ROOT,
            atlas=args.atlas, c=effective_c,
            leakage_correction=args.leakage_correction,
            pseudo_trial_size=args.pseudo_trial_size,
            classifier=args.classifier,
        )

        del results_all_rois
        gc.collect()
        n_ok += 1

    total_time = (time.time() - total_start) / 60.0
    print(f'\n{n_ok}/{len(subjects)} subjects done in {total_time:.1f} minutes')
    if failed:
        print(f'FAILED subjects: {", ".join(failed)}')


if __name__ == '__main__':
    main()
