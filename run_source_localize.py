#!/usr/bin/env python3
"""
Source-localization runner: parallel over subjects.

Splits off the source-estimation half of the former
``run_parallel_lowram.py``.  This script computes the inverse solution
and saves per-ROI source time series as .npz caches; it does NOT run
decoding.  Use ``run_decode.py`` afterwards to consume the cached
time series.

Why split?
----------
Source localization and decoding have very different parallelism
sweet spots:

  * Source localization is dominated by the per-subject inverse + ROI
    extraction step, which is heavyweight per subject but trivially
    parallel across subjects.  Multiprocessing.Pool over subjects is
    the right shape; each worker holds one subject's epochs/STC chain
    and writes one .npz.
  * Decoding (sliding-window CV) is the opposite — light per
    (ROI, window) cell, but thousands of cells per subject.  See
    ``run_decode.py`` for the (ROI x window) joblib pool.

Once the .npz caches exist, source localization typically does not
need to re-run.  Most iteration happens in ``run_decode.py`` /
``explore_decoding.py``.

Usage
-----
    python run_source_localize.py --task overtProd --stim-class prodDiff \\
        --method dSPM --atlas HCPMMP1 --feature-mode vertex_selectkbest \\
        --leakage-correction --n-jobs 2

    # Force a re-run even if caches exist
    python run_source_localize.py --task overtProd --stim-class prodDiff \\
        --method dSPM --atlas HCPMMP1 --overwrite-timeseries --n-jobs 2
"""
import os
# Pin BLAS threads to 1 BEFORE numpy import — otherwise each worker
# spawns ~N_CORES BLAS threads and the box thrashes (especially when
# multiple instances of this runner are launched concurrently).  The
# child processes spawned by multiprocessing.Pool inherit these env vars.
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
from multiprocessing import Pool

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    SUBJECT_IDS, SPEECH_ROIS, find_cached_npz,
)
from log_utils import setup_logging
from data_loader import load_subject_epochs
from forward_model import setup_fsaverage, make_forward, build_roi_labels
from inverse_pipelines import run_dspm_lowram, run_lcmv_lowram
from decoding_io import _save_roi_timeseries, filter_roi_dict
from plotting import save_sensor_erp


# Module-level globals set by the initializer so each worker can access them
_fwd = None
_src = None
_roi_dict = None


def _init_worker(fwd, src, roi_dict):
    """Initialize each worker process with shared forward model data."""
    global _fwd, _src, _roi_dict
    _fwd = fwd
    _src = src
    _roi_dict = roi_dict


def _process_subject(subj_id, task_cond, stim_class, method,
                     feature_mode, atlas, leakage_correction,
                     overwrite_timeseries):
    """Run inverse + ROI extraction for one subject and save .npz cache."""
    subj_start = time.time()
    print(f'\n{"="*60}')
    print(f'Source localizing: {subj_id} | {task_cond} | {stim_class} | {method}')
    print(f'{"="*60}')

    # Skip if cache already exists (unless we're forced to overwrite)
    cached_npz = find_cached_npz(task_cond, method, atlas, feature_mode,
                                 leakage_correction, subj_id, stim_class)
    if cached_npz is not None and not overwrite_timeseries:
        print(f'  Cache already exists, skipping: {cached_npz}')
        return subj_id

    from config import BASELINE_WINDOWS
    baseline_tmin, baseline_tmax = BASELINE_WINDOWS[task_cond]

    try:
        epochs, y, sfreq = load_subject_epochs(subj_id, task_cond, stim_class)
    except FileNotFoundError as e:
        print(f'  SKIPPING {subj_id}: {e}')
        return None

    save_sensor_erp(epochs, y, subj_id, task_cond, stim_class,
                    method, feature_mode, atlas=atlas,
                    leakage_correction=leakage_correction)

    roi_labels = list(_roi_dict.values())
    roi_names = list(_roi_dict.keys())
    extract_mode = 'pca_flip' if feature_mode == 'pca_flip' else 'vertex'

    print(f'  Running {method} inverse (low-RAM)...')
    if method == 'dSPM':
        X_roi, stc_times = run_dspm_lowram(
            epochs, _fwd, baseline_tmin, baseline_tmax,
            roi_labels, _src, feature_mode=extract_mode,
        )
    elif method == 'LCMV':
        X_roi, stc_times = run_lcmv_lowram(
            epochs, _fwd, baseline_tmin, baseline_tmax,
            roi_labels, _src, feature_mode=extract_mode,
        )

    times = stc_times
    del epochs
    gc.collect()

    if leakage_correction:
        if feature_mode == 'pca_flip':
            from leakage_correction import apply_leakage_correction
            print('  Applying symmetric orthogonalization (leakage correction)...')
            X_roi = apply_leakage_correction(X_roi)
        else:
            from leakage_correction import (
                compute_pca_summaries_from_vertices,
                apply_vertex_leakage_correction,
            )
            print('  Computing PCA summaries for vertex leakage correction...')
            n_times_lc = X_roi[0].shape[2]
            X_all_pca = compute_pca_summaries_from_vertices(X_roi, n_times_lc)
            roi_data_tmp = {roi_names[i]: X_roi[i] for i in range(len(roi_names))}
            print('  Applying regression-based vertex leakage correction...')
            apply_vertex_leakage_correction(roi_data_tmp, X_all_pca, roi_names)
            for i, roi_name in enumerate(roi_names):
                X_roi[i] = roi_data_tmp[roi_name]
            del X_all_pca, roi_data_tmp

    roi_data = {}
    if feature_mode == 'pca_flip':
        for i, roi_name in enumerate(roi_names):
            roi_data[roi_name] = X_roi[:, i, :]
    else:
        for i, roi_name in enumerate(roi_names):
            roi_data[roi_name] = X_roi[i]

    _save_roi_timeseries(subj_id, task_cond, stim_class, method,
                         feature_mode, roi_data, y, times, sfreq,
                         overwrite=overwrite_timeseries, atlas=atlas,
                         leakage_correction=leakage_correction)

    del roi_data, X_roi
    gc.collect()

    subj_time = (time.time() - subj_start) / 60.0
    print(f'  {subj_id} done in {subj_time:.1f} minutes')
    return subj_id


def _worker(args):
    (subj_id, task_cond, stim_class, method, feature_mode,
     atlas, leakage_correction, overwrite_timeseries) = args
    try:
        return _process_subject(
            subj_id, task_cond, stim_class, method, feature_mode,
            atlas, leakage_correction, overwrite_timeseries,
        )
    except Exception as e:
        print(f'\n  FAILED {subj_id}: {e}')
        return None


def parse_args():
    parser = argparse.ArgumentParser(
        description='Subject-parallel source localization (writes ROI .npz caches)'
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
    parser.add_argument('--n-jobs', type=int, default=2,
                        help='Number of parallel worker subjects (default: 2 for low-RAM)')
    parser.add_argument('--atlas', default='aparc',
                        choices=['aparc', 'HCPMMP1', 'Schaefer200', 'custom'],
                        help='Cortical atlas for ROI parcellation (default: aparc). '
                             '"custom" uses volumetric NIfTI masks from config.CUSTOM_ROI_DIR')
    parser.add_argument('--leakage-correction', action='store_true', default=False,
                        help='Apply leakage correction (orthogonalization for pca_flip, '
                             'regression for vertex modes)')
    parser.add_argument('--roi-subset', nargs='+', default=None, metavar='ROI',
                        help='Subset of ROI names to process (default: all). '
                             'Case-insensitive, e.g., --roi-subset Temporal vSMC')
    parser.add_argument('--overwrite-timeseries', action='store_true',
                        help='Overwrite existing .npz ROI time series files')
    return parser.parse_args()


def main():
    args = parse_args()
    subjects = args.subjects if args.subjects else SUBJECT_IDS

    setup_logging(args.task, args.stim_class, args.method, args.atlas,
                  args.feature_mode, runner_name='source_localize')

    print(f'Source localization (subject-parallel)')
    print(f'  Task:         {args.task}')
    print(f'  Stim class:   {args.stim_class}')
    print(f'  Method:       {args.method}')
    print(f'  Atlas:        {args.atlas}')
    print(f'  Feature mode: {args.feature_mode}')
    print(f'  Leakage corr: {args.leakage_correction}')
    print(f'  ROI subset:   {args.roi_subset if args.roi_subset else "all"}')
    print(f'  Workers:      {args.n_jobs}')
    print(f'  Subjects:     {len(subjects)}')
    print(f'  Overwrite:    {args.overwrite_timeseries}')
    print()

    print('Setting up fsaverage source space and ROI labels...')
    subjects_dir, fs_dir, src, bem = setup_fsaverage()

    if args.atlas in SPEECH_ROIS:
        roi_dict = build_roi_labels(subjects_dir, atlas=args.atlas,
                                    composite_rois=SPEECH_ROIS[args.atlas])
    else:
        roi_dict = build_roi_labels(subjects_dir, atlas=args.atlas)

    if args.roi_subset:
        roi_dict = filter_roi_dict(roi_dict, args.roi_subset, args.atlas)

    # Only build forward solution if at least one subject lacks cached data
    any_uncached = any(
        find_cached_npz(args.task, args.method, args.atlas, args.feature_mode,
                        args.leakage_correction, s, args.stim_class) is None
        or args.overwrite_timeseries
        for s in subjects
    )
    if any_uncached:
        print('\nBuilding forward solution (uncached subjects detected)...')
        first_epochs, _, _ = load_subject_epochs(
            subjects[0], args.task, args.stim_class
        )
        fwd = make_forward(first_epochs.info, src, bem)
        del first_epochs
    else:
        print('\nAll subjects cached — nothing to do.')
        return
    del bem
    gc.collect()

    worker_args = [
        (subj_id, args.task, args.stim_class, args.method,
         args.feature_mode, args.atlas, args.leakage_correction,
         args.overwrite_timeseries)
        for subj_id in subjects
    ]

    total_start = time.time()

    with Pool(
        processes=args.n_jobs,
        initializer=_init_worker,
        initargs=(fwd, src, roi_dict),
    ) as pool:
        results = pool.map(_worker, worker_args)

    total_time = (time.time() - total_start) / 60.0
    failed = [s for s, r in zip(subjects, results) if r is None]
    n_ok = len(subjects) - len(failed)
    print(f'\n{n_ok}/{len(subjects)} subjects done in {total_time:.1f} minutes')
    if failed:
        print(f'FAILED subjects: {", ".join(failed)}')


if __name__ == '__main__':
    main()
