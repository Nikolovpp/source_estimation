#!/usr/bin/env python3
"""
Parallel runner: processes multiple subjects concurrently.

Uses multiprocessing to parallelize across subjects, mirroring the
existing sensor-space SVM pipeline's parallel execution pattern.

Usage:
    python run_parallel.py --task overtProd --stim-class prodDiff --method dSPM
    python run_parallel.py --task overtProd --stim-class prodDiff --method dSPM \
        --n-jobs 4 --feature-mode vertex_pca
"""
import argparse
import os
import sys
import time
import warnings
from multiprocessing import Pool

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    SUBJECT_IDS, SW_DUR, SW_STEP_SIZE, SVM_OUTPUT_ROOT, SPEECH_ROIS,
    SVM_C, PSEUDO_TRIAL_SIZE,
)
from data_loader import load_subject_epochs
from forward_model import setup_fsaverage, make_forward, build_roi_labels
from run_source_svm import process_subject


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


def _worker(args):
    """Worker function for parallel processing."""
    (subj_id, task_cond, stim_class, method, feature_mode,
     sw_dur, sw_step, save_dir, skip_svm,
     atlas, leakage_correction, pseudo_trial_size, svm_c) = args
    return process_subject(
        subj_id, task_cond, stim_class, method, feature_mode,
        _fwd, _src, _roi_dict, sw_dur, sw_step, save_dir,
        skip_svm=skip_svm,
        atlas=atlas, leakage_correction=leakage_correction,
        pseudo_trial_size=pseudo_trial_size, svm_c=svm_c,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description='Parallel source-space SVM decoding pipeline'
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
    parser.add_argument('--n-jobs', type=int, default=4,
                        help='Number of parallel workers (default: 4)')
    parser.add_argument('--sw-dur', type=int, default=SW_DUR)
    parser.add_argument('--sw-step', type=int, default=SW_STEP_SIZE)
    parser.add_argument('--skip-svm', action='store_true',
                        help='Only save ROI time series, skip SVM decoding')
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

    print(f'Parallel source-space SVM decoding')
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

    if args.atlas == 'aparc':
        roi_dict = build_roi_labels(subjects_dir, atlas='aparc',
                                     composite_rois=SPEECH_ROIS['aparc'])
    else:
        roi_dict = build_roi_labels(subjects_dir, atlas=args.atlas)

    print('\nBuilding forward solution...')
    first_epochs, _, _ = load_subject_epochs(
        subjects[0], args.task, args.stim_class
    )
    fwd = make_forward(first_epochs.info, src, bem)

    # Build parameter list for workers
    worker_args = [
        (subj_id, args.task, args.stim_class, args.method,
         args.feature_mode, args.sw_dur, args.sw_step, SVM_OUTPUT_ROOT,
         args.skip_svm,
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
