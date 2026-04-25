#!/usr/bin/env python3
"""
Validation script: runs the full pipeline on one subject to verify
everything works end-to-end before launching the full batch.

Usage:
    python validate_pipeline.py
    python validate_pipeline.py --subject EEGPROD4003 --task perception
"""
import argparse
import os
import sys
import warnings

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import mne


def validate_data_loading(subj_id, task_cond, stim_class):
    """Step 1: Validate data loading."""
    print('\n--- Step 1: Data Loading ---')
    from data_loader import load_subject_epochs

    epochs, y, sfreq = load_subject_epochs(subj_id, task_cond, stim_class)

    print(f'  Epochs shape:  {epochs.get_data().shape}')
    print(f'  Labels shape:  {y.shape}')
    print(f'  Label balance: {np.bincount(y)}')
    print(f'  sfreq:         {sfreq} Hz')
    print(f'  tmin:          {epochs.tmin} s')
    print(f'  tmax:          {epochs.tmax} s')
    print(f'  Channel names: {epochs.ch_names[:5]}... ({len(epochs.ch_names)} total)')
    print(f'  Data range:    [{epochs.get_data().min():.2e}, {epochs.get_data().max():.2e}] V')

    # Verify data is in Volts (should be ~1e-5 to 1e-6)
    max_val = np.abs(epochs.get_data()).max()
    assert max_val < 1e-3, f'Data appears to be in µV, not V: max={max_val}'
    assert max_val > 1e-10, f'Data values suspiciously small: max={max_val}'
    print('  ✓ Data scale looks correct (Volts)')

    # Verify average reference
    projs = epochs.info['projs']
    has_avg_ref = any('Average EEG' in p['desc'] for p in projs)
    print(f'  ✓ Average reference projection: {has_avg_ref}')

    return epochs, y, sfreq


def validate_forward_model(epochs):
    """Step 2: Validate forward model setup."""
    print('\n--- Step 2: Forward Model ---')
    from forward_model import setup_fsaverage, make_forward, build_roi_labels

    subjects_dir, fs_dir, src, bem = setup_fsaverage()
    print(f'  fsaverage dir: {fs_dir}')
    print(f'  Source space:  {src}')

    fwd = make_forward(epochs.info, src, bem)
    print(f'  Forward:       {fwd["nsource"]} sources')

    roi_dict = build_roi_labels(subjects_dir)
    for name, label in roi_dict.items():
        print(f'  ROI {name}: {len(label.vertices)} vertices')

    return fwd, src, roi_dict, subjects_dir


def validate_inverse_dspm(epochs, fwd, baseline_tmin, baseline_tmax):
    """Step 3a: Validate dSPM inverse."""
    print('\n--- Step 3a: dSPM Inverse ---')
    from inverse_pipelines import run_dspm

    stcs = run_dspm(epochs, fwd, baseline_tmin, baseline_tmax)
    stc0 = stcs[0]
    print(f'  STC data shape: {stc0.data.shape}')
    print(f'  STC data range: [{stc0.data.min():.4f}, {stc0.data.max():.4f}]')
    print(f'  STC times:      [{stc0.times[0]:.3f}, {stc0.times[-1]:.3f}] s')
    return stcs


def validate_inverse_lcmv(epochs, fwd, baseline_tmin, baseline_tmax):
    """Step 3b: Validate LCMV inverse."""
    print('\n--- Step 3b: LCMV Inverse ---')
    from inverse_pipelines import run_lcmv

    stcs = run_lcmv(epochs, fwd, baseline_tmin, baseline_tmax)
    stc0 = stcs[0]
    print(f'  STC data shape: {stc0.data.shape}')
    print(f'  STC data range: [{stc0.data.min():.4f}, {stc0.data.max():.4f}]')
    print(f'  STC times:      [{stc0.times[0]:.3f}, {stc0.times[-1]:.3f}] s')
    return stcs


def validate_svm_decoding(stcs, y, sfreq, src, roi_dict, tmin, decode_tmin,
                          times=None):
    """Step 4: Validate SVM decoding on one ROI."""
    print('\n--- Step 4: SVM Decoding ---')
    from svm_decoding import (
        extract_roi_data_vertices,
        extract_roi_data_pca_flip,
        sliding_window_svm_decode,
    )
    from config import SW_DUR, SW_STEP_SIZE

    # Test pca_flip mode
    first_roi_name = list(roi_dict.keys())[0]
    first_roi_label = roi_dict[first_roi_name]
    roi_labels = list(roi_dict.values())

    print(f'  Decode starts at: {decode_tmin} s (baseline excluded)')
    print(f'  Testing pca_flip on ROI: {first_roi_name}')
    X_pca = extract_roi_data_pca_flip(stcs, roi_labels, src)
    print(f'  pca_flip data shape: {X_pca.shape}')

    X_single = X_pca[:, 0, :]  # first ROI
    results = sliding_window_svm_decode(
        X_single, y, sfreq, SW_DUR, SW_STEP_SIZE, tmin, decode_tmin,
        feature_mode='pca_flip', times=times
    )
    print(f'  Number of time windows: {len(results)}')
    accs = [r["SVM_acc"] for r in results]
    print(f'  Accuracy range: [{min(accs):.3f}, {max(accs):.3f}]')
    print(f'  First window: ms={results[0]["ms"]:.1f}, acc={results[0]["SVM_acc"]:.3f}')
    print(f'  Peak window:  ms={results[np.argmax(accs)]["ms"]:.1f}, '
          f'acc={max(accs):.3f}')

    # Test vertex_pca mode
    print(f'\n  Testing vertex_pca on ROI: {first_roi_name}')
    X_vert = extract_roi_data_vertices(stcs, first_roi_label)
    print(f'  Vertex data shape: {X_vert.shape}')

    results_v = sliding_window_svm_decode(
        X_vert, y, sfreq, SW_DUR, SW_STEP_SIZE, tmin, decode_tmin,
        feature_mode='vertex_pca', times=times
    )
    accs_v = [r["SVM_acc"] for r in results_v]
    print(f'  Accuracy range: [{min(accs_v):.3f}, {max(accs_v):.3f}]')

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', default='EEGPROD4001')
    parser.add_argument('--task', default='overtProd',
                        choices=['perception', 'overtProd'])
    parser.add_argument('--stim-class', default='prodDiff',
                        choices=['prodDiff', 'percDiff'])
    parser.add_argument('--skip-lcmv', action='store_true',
                        help='Skip LCMV validation (faster)')
    args = parser.parse_args()

    from config import BASELINE_WINDOWS, DECODE_TMIN

    print('='*60)
    print('Source Estimation Pipeline Validation')
    print(f'  Subject:    {args.subject}')
    print(f'  Task:       {args.task}')
    print(f'  Stim class: {args.stim_class}')
    print('='*60)

    baseline_tmin, baseline_tmax = BASELINE_WINDOWS[args.task]
    decode_tmin = DECODE_TMIN[args.task]

    # Step 1
    epochs, y, sfreq = validate_data_loading(
        args.subject, args.task, args.stim_class
    )

    # Step 2
    fwd, src, roi_dict, subjects_dir = validate_forward_model(epochs)

    # Step 3a
    stcs_dspm = validate_inverse_dspm(epochs, fwd, baseline_tmin, baseline_tmax)

    # Step 3b
    if not args.skip_lcmv:
        stcs_lcmv = validate_inverse_lcmv(epochs, fwd, baseline_tmin, baseline_tmax)

    # Step 4
    validate_svm_decoding(stcs_dspm, y, sfreq, src, roi_dict,
                          epochs.tmin, decode_tmin, times=epochs.times)

    print('\n' + '='*60)
    print('ALL VALIDATION STEPS PASSED')
    print('='*60)


if __name__ == '__main__':
    main()
