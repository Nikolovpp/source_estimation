#!/usr/bin/env python3
"""
Low-RAM parallel runner: processes multiple subjects with minimal memory.

Key differences from run_parallel.py:
  - Uses generator-based inverse solvers (never holds all STCs in memory)
  - Explicit garbage collection between subjects
  - Defaults to 2 workers (not 4) to reduce peak memory
  - Each worker frees data aggressively after processing
  - Uses float32 for ROI data arrays

Usage:
    python run_parallel_lowram.py --task overtProd --stim-class prodDiff --method dSPM
    python run_parallel_lowram.py --task overtProd --stim-class prodDiff --method dSPM \
        --n-jobs 2 --feature-mode vertex_pca
"""
import argparse
import gc
import os
import sys
import time
import warnings
from multiprocessing import Pool

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    SUBJECT_IDS, SW_DUR, SW_STEP_SIZE, SVM_OUTPUT_ROOT,
    BASELINE_WINDOWS, DECODE_TMIN, SPEECH_ROIS,
    SVM_C, PSEUDO_TRIAL_SIZE, LEAKAGE_CORRECTION,
)
from log_utils import setup_logging
from data_loader import load_subject_epochs
from forward_model import setup_fsaverage, make_forward, build_roi_labels
from inverse_pipelines import run_dspm_lowram, run_lcmv_lowram
from svm_decoding import sliding_window_svm_decode
from run_source_svm import _save_results, _save_roi_timeseries
from plotting import save_sensor_erp, save_source_erp, save_svm_results


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


def _process_subject_lowram(subj_id, task_cond, stim_class, method,
                            feature_mode, sw_dur, sw_step, save_dir,
                            skip_svm=False, skip_save_timeseries=False,
                            overwrite_timeseries=False,
                            atlas='aparc', leakage_correction=False,
                            pseudo_trial_size=0, svm_c=1.0):
    """Process a single subject using low-RAM inverse pipeline."""
    subj_start = time.time()
    print(f'\n{"="*60}')
    print(f'[low-RAM] Processing: {subj_id} | {task_cond} | {stim_class} | {method}')
    print(f'{"="*60}')

    # Step 1: Load data
    try:
        epochs, y, sfreq = load_subject_epochs(subj_id, task_cond, stim_class)
    except FileNotFoundError as e:
        print(f'  SKIPPING {subj_id}: {e}')
        return None

    baseline_tmin, baseline_tmax = BASELINE_WINDOWS[task_cond]
    decode_tmin = DECODE_TMIN[task_cond]

    # Save sensor-space ERP figure
    save_sensor_erp(epochs, y, subj_id, task_cond, stim_class,
                    method, feature_mode)

    roi_labels = list(_roi_dict.values())
    roi_names = list(_roi_dict.keys())
    extract_mode = 'pca_flip' if feature_mode == 'pca_flip' else 'vertex'

    # Step 2: Inverse + ROI extraction (low-RAM — generator based)
    print(f'\n  Running {method} inverse (low-RAM)...')
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

    # Free epochs immediately
    tmin_epoch = stc_times[0]
    times = stc_times
    del epochs
    gc.collect()

    # Leakage correction
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
            # Build temporary dict for correction
            roi_data_tmp = {roi_names[i]: X_roi[i] for i in range(len(roi_names))}
            print('  Applying regression-based vertex leakage correction...')
            apply_vertex_leakage_correction(roi_data_tmp, X_all_pca, roi_names)
            # Write back to list
            for i, roi_name in enumerate(roi_names):
                X_roi[i] = roi_data_tmp[roi_name]
            del X_all_pca, roi_data_tmp

    # Build roi_data dict for saving figures and time series
    roi_data = {}
    if feature_mode == 'pca_flip':
        for i, roi_name in enumerate(roi_names):
            roi_data[roi_name] = X_roi[:, i, :]  # (n_epochs, n_times)
    else:
        for i, roi_name in enumerate(roi_names):
            roi_data[roi_name] = X_roi[i]  # (n_epochs, n_vertices, n_times)

    # Save source-space ERP figure
    save_source_erp(roi_data, y, times, subj_id, task_cond, stim_class,
                    method, feature_mode, decode_tmin)

    # Save ROI time series for use with original SVM notebooks
    if not skip_save_timeseries:
        _save_roi_timeseries(subj_id, task_cond, stim_class, method,
                             feature_mode, roi_data, y, times, sfreq,
                             overwrite=overwrite_timeseries, atlas=atlas,
                             leakage_correction=leakage_correction)

    # Step 3: SVM decoding per ROI
    results_all_rois = {}

    if not skip_svm:
        if feature_mode == 'pca_flip':
            for i, roi_name in enumerate(roi_names):
                print(f'\n  Decoding ROI: {roi_name} (pca_flip)')
                X_roi_i = X_roi[:, i, :]
                results = sliding_window_svm_decode(
                    X_roi_i, y, sfreq, sw_dur, sw_step,
                    tmin_epoch, decode_tmin, feature_mode='pca_flip',
                    times=times, svm_c=svm_c,
                    pseudo_trial_size=pseudo_trial_size,
                )
                results_all_rois[roi_name] = results
        else:
            for i, roi_name in enumerate(roi_names):
                print(f'\n  Decoding ROI: {roi_name} ({feature_mode})')
                X_vert = X_roi[i]
                results = sliding_window_svm_decode(
                    X_vert, y, sfreq, sw_dur, sw_step,
                    tmin_epoch, decode_tmin, feature_mode=feature_mode,
                    times=times, svm_c=svm_c,
                    pseudo_trial_size=pseudo_trial_size,
                )
                results_all_rois[roi_name] = results

    # Free ROI data
    del X_roi, roi_data
    gc.collect()

    if not skip_svm:
        # Save SVM accuracy figure
        save_svm_results(results_all_rois, subj_id, task_cond, stim_class,
                         method, feature_mode, sw_dur, sw_step)

        # Save SVM results CSV
        _save_results(subj_id, task_cond, stim_class, method, feature_mode,
                      sw_dur, sw_step, results_all_rois, save_dir,
                      atlas=atlas, svm_c=svm_c,
                      leakage_correction=leakage_correction)

    subj_time = (time.time() - subj_start) / 60.0
    print(f'\n  {subj_id} done in {subj_time:.1f} minutes')

    del results_all_rois
    gc.collect()
    return subj_id


def _worker(args):
    """Worker function for parallel processing."""
    (subj_id, task_cond, stim_class, method, feature_mode,
     sw_dur, sw_step, save_dir, skip_svm,
     skip_save_timeseries, overwrite_timeseries,
     atlas, leakage_correction, pseudo_trial_size, svm_c) = args
    return _process_subject_lowram(
        subj_id, task_cond, stim_class, method, feature_mode,
        sw_dur, sw_step, save_dir, skip_svm=skip_svm,
        skip_save_timeseries=skip_save_timeseries,
        overwrite_timeseries=overwrite_timeseries,
        atlas=atlas, leakage_correction=leakage_correction,
        pseudo_trial_size=pseudo_trial_size, svm_c=svm_c,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description='Low-RAM parallel source-space SVM decoding pipeline'
    )
    parser.add_argument('--task', required=True,
                        choices=['perception', 'overtProd'])
    parser.add_argument('--stim-class', required=True,
                        choices=['prodDiff', 'percDiff'])
    parser.add_argument('--method', required=True,
                        choices=['dSPM', 'LCMV'])
    parser.add_argument('--feature-mode', default='pca_flip',
                        choices=['pca_flip', 'vertex_pca', 'vertex_selectkbest'])
    parser.add_argument('--subjects', nargs='+', default=None)
    parser.add_argument('--n-jobs', type=int, default=2,
                        help='Number of parallel workers (default: 2 for low-RAM)')
    parser.add_argument('--sw-dur', type=int, default=SW_DUR)
    parser.add_argument('--sw-step', type=int, default=SW_STEP_SIZE)
    parser.add_argument('--skip-svm', action='store_true',
                        help='Only save ROI time series and figures, skip SVM decoding')
    parser.add_argument('--skip-save-timeseries', action='store_true',
                        help='Skip saving .npz ROI time series (faster iteration on SVM decoding)')
    parser.add_argument('--overwrite-timeseries', action='store_true',
                        help='Overwrite existing .npz ROI time series files')
    # Advanced pipeline options
    parser.add_argument('--atlas', default='aparc',
                        choices=['aparc', 'HCPMMP1', 'Schaefer200', 'custom'],
                        help='Cortical atlas for ROI parcellation (default: aparc). '
                             '"custom" uses volumetric NIfTI masks from config.CUSTOM_ROI_DIR')
    parser.add_argument('--leakage-correction', action='store_true', default=False,
                        help='Apply leakage correction (orthogonalization for pca_flip, regression for vertex modes)')
    parser.add_argument('--pseudo-trial-size', type=int, default=PSEUDO_TRIAL_SIZE,
                        help='Pseudo-trial group size; 0 = disabled (default: 0)')
    parser.add_argument('--svm-c', type=float, default=SVM_C,
                        help=f'SVM regularization parameter C (default: {SVM_C})')
    return parser.parse_args()


def main():
    args = parse_args()
    subjects = args.subjects if args.subjects else SUBJECT_IDS

    setup_logging(args.task, args.stim_class, args.method, args.atlas,
                  args.feature_mode, runner_name='parallel_lowram')

    print(f'Low-RAM parallel source-space SVM decoding')
    print(f'  Task:         {args.task}')
    print(f'  Stim class:   {args.stim_class}')
    print(f'  Method:       {args.method}')
    print(f'  Atlas:        {args.atlas}')
    print(f'  Feature mode: {args.feature_mode}')
    print(f'  SVM C:        {args.svm_c}')
    print(f'  Pseudo-trial: {args.pseudo_trial_size if args.pseudo_trial_size > 0 else "disabled"}')
    print(f'  Leakage corr: {args.leakage_correction}')
    print(f'  Workers:      {args.n_jobs}')
    print(f'  Subjects:     {len(subjects)}')
    print()

    # Build forward model (once, in the main process)
    print('Setting up fsaverage forward model...')
    subjects_dir, fs_dir, src, bem = setup_fsaverage()

    if args.atlas in SPEECH_ROIS:
        roi_dict = build_roi_labels(subjects_dir, atlas=args.atlas,
                                     composite_rois=SPEECH_ROIS[args.atlas])
    else:
        roi_dict = build_roi_labels(subjects_dir, atlas=args.atlas)

    print('\nBuilding forward solution...')
    first_epochs, _, _ = load_subject_epochs(
        subjects[0], args.task, args.stim_class
    )
    fwd = make_forward(first_epochs.info, src, bem)
    del first_epochs, bem
    gc.collect()

    # Build parameter list for workers
    worker_args = [
        (subj_id, args.task, args.stim_class, args.method,
         args.feature_mode, args.sw_dur, args.sw_step, SVM_OUTPUT_ROOT,
         args.skip_svm, args.skip_save_timeseries, args.overwrite_timeseries,
         args.atlas, args.leakage_correction, args.pseudo_trial_size,
         args.svm_c)
        for subj_id in subjects
    ]

    total_start = time.time()

    with Pool(
        processes=args.n_jobs,
        initializer=_init_worker,
        initargs=(fwd, src, roi_dict),
    ) as pool:
        pool.map(_worker, worker_args)

    total_time = (time.time() - total_start) / 60.0
    print(f'\nAll {len(subjects)} subjects done in {total_time:.1f} minutes')


if __name__ == '__main__':
    main()
