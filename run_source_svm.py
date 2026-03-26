#!/usr/bin/env python3
"""
Main script: source estimation + SVM decoding pipeline.

Processes each subject through:
  1. Load EEGLAB-preprocessed EEG data → MNE Epochs
  2. Build forward model (fsaverage template, shared across subjects)
  3. Run inverse pipeline (dSPM or LCMV) → source estimates
  4. Extract ROI time courses
  5. Run sliding-window SVM decoding per ROI
  6. Save results in CSV format compatible with existing analysis scripts

Usage:
    python run_source_svm.py --task overtProd --stim-class prodDiff --method dSPM
    python run_source_svm.py --task perception --stim-class percDiff --method LCMV
    python run_source_svm.py --task overtProd --stim-class prodDiff --method dSPM \
        --feature-mode vertex_pca --subjects EEGPROD4001 EEGPROD4003
"""
import argparse
import os
import sys
import time
import statistics
import warnings

import numpy as np
import pandas as pd

# Suppress convergence warnings from LinearSVC
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

# Add source_estimation directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    SUBJECT_IDS, SW_DUR, SW_STEP_SIZE, SVM_OUTPUT_ROOT, ROI_TIMESERIES_ROOT,
    SPEECH_ROIS, BASELINE_WINDOWS, DECODE_TMIN,
    SVM_C, PSEUDO_TRIAL_SIZE, LEAKAGE_CORRECTION,
)
from log_utils import setup_logging
from data_loader import load_subject_epochs
from forward_model import setup_fsaverage, make_forward, build_roi_labels
from inverse_pipelines import run_dspm, run_lcmv
from svm_decoding import (
    extract_roi_data_vertices,
    extract_roi_data_pca_flip,
    sliding_window_svm_decode,
)
from plotting import save_sensor_erp, save_source_erp, save_svm_results


def parse_args():
    parser = argparse.ArgumentParser(
        description='Source-space SVM decoding pipeline'
    )
    parser.add_argument(
        '--task', required=True,
        choices=['perception', 'overtProd'],
        help='Task condition'
    )
    parser.add_argument(
        '--stim-class', required=True,
        choices=['prodDiff', 'percDiff'],
        help='Stimulus class contrast'
    )
    parser.add_argument(
        '--method', required=True,
        choices=['dSPM', 'LCMV'],
        help='Source estimation method'
    )
    parser.add_argument(
        '--feature-mode', default='pca_flip',
        choices=['pca_flip', 'vertex_pca', 'vertex_selectkbest'],
        help='Feature extraction strategy for SVM (default: pca_flip)'
    )
    parser.add_argument(
        '--subjects', nargs='+', default=None,
        help='Subset of subject IDs to process (default: all)'
    )
    parser.add_argument(
        '--sw-dur', type=int, default=SW_DUR,
        help=f'Sliding window duration in ms (default: {SW_DUR})'
    )
    parser.add_argument(
        '--sw-step', type=int, default=SW_STEP_SIZE,
        help=f'Sliding window step in ms (default: {SW_STEP_SIZE})'
    )
    parser.add_argument(
        '--skip-svm', action='store_true',
        help='Only save ROI time series, skip SVM decoding'
    )
    parser.add_argument(
        '--skip-save-timeseries', action='store_true',
        help='Skip saving .npz ROI time series (faster iteration on SVM decoding)'
    )
    parser.add_argument(
        '--overwrite-timeseries', action='store_true',
        help='Overwrite existing .npz ROI time series files'
    )
    # Advanced pipeline options
    parser.add_argument(
        '--atlas', default='aparc',
        choices=['aparc', 'HCPMMP1', 'Schaefer200', 'custom'],
        help='Cortical atlas for ROI parcellation (default: aparc). '
             '"custom" uses volumetric NIfTI masks from config.CUSTOM_ROI_DIR'
    )
    parser.add_argument(
        '--leakage-correction', action='store_true', default=False,
        help='Apply leakage correction (orthogonalization for pca_flip, regression for vertex modes)'
    )
    parser.add_argument(
        '--pseudo-trial-size', type=int, default=PSEUDO_TRIAL_SIZE,
        help='Pseudo-trial group size; 0 = disabled (default: 0)'
    )
    parser.add_argument(
        '--svm-c', type=float, default=SVM_C,
        help=f'SVM regularization parameter C (default: {SVM_C})'
    )
    return parser.parse_args()


def process_subject(subj_id, task_cond, stim_class, method, feature_mode,
                    fwd, src, roi_dict, sw_dur, sw_step, save_dir,
                    skip_svm=False, skip_save_timeseries=False,
                    overwrite_timeseries=False,
                    atlas='aparc', leakage_correction=False,
                    pseudo_trial_size=0, svm_c=1.0):
    """Process a single subject through the full pipeline."""
    subj_start = time.time()
    print(f'\n{"="*60}')
    print(f'Processing: {subj_id} | {task_cond} | {stim_class} | {method}')
    print(f'{"="*60}')

    # Step 1: Load data
    try:
        epochs, y, sfreq = load_subject_epochs(subj_id, task_cond, stim_class)
    except FileNotFoundError as e:
        print(f'  SKIPPING {subj_id}: {e}')
        return None

    tmin = epochs.tmin
    times = epochs.times

    # Task-specific baseline and decode windows
    baseline_tmin, baseline_tmax = BASELINE_WINDOWS[task_cond]
    decode_tmin = DECODE_TMIN[task_cond]

    # Save sensor-space ERP figure
    save_sensor_erp(epochs, y, subj_id, task_cond, stim_class,
                    method, feature_mode)

    # Step 2: Run inverse pipeline
    print(f'\n  Running {method} inverse...')
    if method == 'dSPM':
        stcs = run_dspm(epochs, fwd, baseline_tmin, baseline_tmax)
    elif method == 'LCMV':
        stcs = run_lcmv(epochs, fwd, baseline_tmin, baseline_tmax)

    # Step 3: Extract features per ROI
    # Build dict of {roi_name: X_roi} for both saving and decoding
    roi_data = {}

    if feature_mode == 'pca_flip':
        roi_labels = list(roi_dict.values())
        roi_names = list(roi_dict.keys())
        X_all = extract_roi_data_pca_flip(stcs, roi_labels, src)
        # X_all shape: (n_epochs, n_rois, n_times)

        # Leakage correction: symmetric orthogonalization
        if leakage_correction:
            from leakage_correction import apply_leakage_correction
            print('  Applying symmetric orthogonalization (leakage correction)...')
            X_all = apply_leakage_correction(X_all)

        for i, roi_name in enumerate(roi_names):
            roi_data[roi_name] = X_all[:, i, :]  # (n_epochs, n_times)
    else:
        for roi_name, roi_label in roi_dict.items():
            X_roi = extract_roi_data_vertices(stcs, roi_label)
            # X_roi shape: (n_epochs, n_vertices, n_times)
            roi_data[roi_name] = X_roi

        # Vertex-level leakage correction: regress out other ROIs' signals
        if leakage_correction:
            from leakage_correction import apply_vertex_leakage_correction
            print('  Computing pca_flip summaries for vertex leakage correction...')
            roi_labels = list(roi_dict.values())
            roi_names = list(roi_dict.keys())
            X_all_pca = extract_roi_data_pca_flip(stcs, roi_labels, src)
            print('  Applying regression-based vertex leakage correction...')
            roi_data = apply_vertex_leakage_correction(
                roi_data, X_all_pca, roi_names
            )
            del X_all_pca

    # Step 3b: Save ROI time series for use with original SVM notebooks
    if not skip_save_timeseries:
        _save_roi_timeseries(subj_id, task_cond, stim_class, method,
                             feature_mode, roi_data, y, times, sfreq,
                             overwrite=overwrite_timeseries, atlas=atlas,
                             leakage_correction=leakage_correction)

    # Save source-space ERP figure
    save_source_erp(roi_data, y, times, subj_id, task_cond, stim_class,
                    method, feature_mode, decode_tmin)

    # Step 4: SVM decoding (optional)
    results_all_rois = {}
    if not skip_svm:
        for roi_name, X_roi in roi_data.items():
            print(f'\n  Decoding ROI: {roi_name} ({feature_mode})')
            results = sliding_window_svm_decode(
                X_roi, y, sfreq, sw_dur, sw_step, tmin, decode_tmin,
                feature_mode=feature_mode, times=times,
                svm_c=svm_c, pseudo_trial_size=pseudo_trial_size,
            )
            results_all_rois[roi_name] = results

        # Save SVM accuracy figure
        save_svm_results(results_all_rois, subj_id, task_cond, stim_class,
                         method, feature_mode, sw_dur, sw_step)

        # Step 5: Save SVM results
        _save_results(subj_id, task_cond, stim_class, method, feature_mode,
                      sw_dur, sw_step, results_all_rois, save_dir,
                      atlas=atlas, svm_c=svm_c,
                      leakage_correction=leakage_correction)

    subj_time = (time.time() - subj_start) / 60.0
    print(f'\n  {subj_id} done in {subj_time:.1f} minutes')
    return results_all_rois


def _save_roi_timeseries(subj_id, task_cond, stim_class, method,
                         feature_mode, roi_data, y, times, sfreq,
                         overwrite=False, atlas='aparc',
                         leakage_correction=False):
    """
    Save source-estimated ROI time series as .npz files for use with
    the original sensor-space SVM notebooks.

    If the file already exists and *overwrite* is False, the save is
    skipped to avoid repeating this time-consuming step.

    The saved arrays use the same axis convention as the original pipeline:
      - pca_flip:  (n_trials, n_timepoints, 1)    — 1 virtual sensor per ROI
      - vertex_*:  (n_trials, n_timepoints, n_vertices) — vertices as "channels"

    To load in the original notebooks:
        data = np.load('EEGPROD4001_perception_percDiff.npz', allow_pickle=True)
        roi_names = data['roi_names']           # array of ROI name strings
        y = data['y']                           # binary class labels
        times = data['times']                   # time vector in seconds
        sfreq = float(data['sfreq'])            # sampling frequency
        X_Temporal = data['Temporal']           # (trials, timepoints, features)
    """
    leakage_tag = 'leakage_corrected' if leakage_correction else 'raw'
    ts_dir = (
        ROI_TIMESERIES_ROOT / task_cond / method / atlas
        / feature_mode / leakage_tag
    )
    ts_dir.mkdir(parents=True, exist_ok=True)

    npz_file = ts_dir / f'{subj_id}_{task_cond}_{stim_class}.npz'

    if npz_file.exists() and not overwrite:
        print(f'  ROI time series already exists, skipping: {npz_file}')
        return

    save_dict = {
        'y': y,
        'times': times,
        'sfreq': np.array(sfreq),
        'roi_names': np.array(list(roi_data.keys())),
    }

    for roi_name, X_roi in roi_data.items():
        if X_roi.ndim == 2:
            # pca_flip: (n_epochs, n_times) → (n_epochs, n_times, 1)
            save_dict[roi_name] = X_roi[:, :, np.newaxis]
        else:
            # vertex modes: (n_epochs, n_vertices, n_times) → (n_epochs, n_times, n_vertices)
            save_dict[roi_name] = X_roi.transpose(0, 2, 1)

    np.savez_compressed(npz_file, **save_dict)
    print(f'  Saved ROI time series: {npz_file}')


def _save_results(subj_id, task_cond, stim_class, method, feature_mode,
                  sw_dur, sw_step, results_all_rois, save_dir,
                  atlas='aparc', svm_c=1.0, leakage_correction=False):
    """Save results in CSV format matching the existing pipeline."""
    # Create output directory (includes atlas and leakage tag for separation)
    leakage_tag = 'leakage_corrected' if leakage_correction else 'raw'
    csv_save_path = (
        save_dir / task_cond / method / atlas / feature_mode
        / leakage_tag / f'{sw_dur}_{sw_step}' / stim_class
    )
    csv_save_path.mkdir(parents=True, exist_ok=True)

    csv_file = (
        csv_save_path
        / f'{subj_id}_{task_cond}_{stim_class}_{sw_dur}_{sw_step}.csv'
    )

    rows = []
    for roi_name, results in results_all_rois.items():
        # Sanitize ROI names: replace spaces with underscores for CSV compat
        csv_key = roi_name.replace(' ', '_')
        for r in results:
            rows.append({
                'key': csv_key,
                'ms': r['ms'],
                'mean_list': r['mean_list'],
                'SVM_acc': r['SVM_acc'],
                'best_params': f'C={svm_c}',
            })

    df = pd.DataFrame(rows, columns=['key', 'ms', 'mean_list', 'SVM_acc', 'best_params'])
    df.to_csv(csv_file, index=False)
    print(f'  Saved: {csv_file}')


def main():
    args = parse_args()

    subjects = args.subjects if args.subjects else SUBJECT_IDS
    task_cond = args.task
    stim_class = args.stim_class
    method = args.method
    feature_mode = args.feature_mode
    sw_dur = args.sw_dur
    sw_step = args.sw_step
    skip_svm = args.skip_svm
    skip_save_timeseries = args.skip_save_timeseries
    overwrite_timeseries = args.overwrite_timeseries
    atlas = args.atlas
    leakage_correction = args.leakage_correction
    pseudo_trial_size = args.pseudo_trial_size
    svm_c = args.svm_c

    setup_logging(task_cond, stim_class, method, atlas, feature_mode,
                  runner_name='sequential')

    print(f'Source-space SVM decoding pipeline')
    print(f'  Task:         {task_cond}')
    print(f'  Stim class:   {stim_class}')
    print(f'  Method:       {method}')
    print(f'  Atlas:        {atlas}')
    print(f'  Feature mode: {feature_mode}')
    print(f'  SW duration:  {sw_dur} ms')
    print(f'  SW step:      {sw_step} ms')
    print(f'  SVM C:        {svm_c}')
    print(f'  Pseudo-trial: {pseudo_trial_size if pseudo_trial_size > 0 else "disabled"}')
    print(f'  Leakage corr: {leakage_correction}')
    print(f'  Subjects:     {len(subjects)}')
    print(f'  Skip SVM:     {skip_svm}')
    print(f'  Save .npz:    {not skip_save_timeseries}')
    print()

    # Step 0: Build forward model (shared across subjects)
    print('Setting up fsaverage forward model...')
    subjects_dir, fs_dir, src, bem = setup_fsaverage()

    if atlas in SPEECH_ROIS:
        roi_dict = build_roi_labels(subjects_dir, atlas=atlas,
                                     composite_rois=SPEECH_ROIS[atlas])
    else:
        roi_dict = build_roi_labels(subjects_dir, atlas=atlas)

    # We need to build the forward solution using the first subject's info
    # to get the correct channel configuration. Since all subjects share
    # the same montage, we build it once.
    print('\nLoading first subject to build forward solution...')
    first_epochs, _, _ = load_subject_epochs(subjects[0], task_cond, stim_class)
    fwd = make_forward(first_epochs.info, src, bem)

    # Process all subjects
    total_start = time.time()
    for subj_id in subjects:
        process_subject(
            subj_id, task_cond, stim_class, method, feature_mode,
            fwd, src, roi_dict, sw_dur, sw_step, SVM_OUTPUT_ROOT,
            skip_svm=skip_svm,
            skip_save_timeseries=skip_save_timeseries,
            overwrite_timeseries=overwrite_timeseries,
            atlas=atlas,
            leakage_correction=leakage_correction,
            pseudo_trial_size=pseudo_trial_size,
            svm_c=svm_c,
        )

    total_time = (time.time() - total_start) / 60.0
    print(f'\n{"="*60}')
    print(f'All subjects done in {total_time:.1f} minutes')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
