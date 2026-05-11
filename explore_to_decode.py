#!/usr/bin/env python3
"""Convert explore_decoding CSVs to the run_decode per-subject layout.

The two runners feed the same ``decode_one_window`` with the same
default ``random_state=42`` and same per-window CV setup, so explore's
``accuracy`` is bit-identical to run_decode's ``decode_acc`` for a
matching ``(classifier, sw_dur, sw_step, tuned, leakage, pseudo,
feature_mode, atlas, c)``.  Re-running ``run_decode --tune-hyperparams``
just to consume the explore numbers in ``source_stats_viz`` is wasted
compute — this converter reshapes the existing explore_full.csv files
into the per-subject layout ``source_stats_viz`` expects.

Layout mismatch this script fixes
---------------------------------
- explore: one CSV per **ROI**, all subjects in ``subject`` column,
  columns ``subject, roi, classifier, sw_dur, sw_step, tuned, ms,
  accuracy[, best_C, best_C_freq]``.
- run_decode: one CSV per **subject**, all ROIs in ``key`` column,
  columns ``key, ms, mean_list, decode_acc, best_params``.

Usage
-----
    python explore_to_decode.py --task overtProd --stim-class prodDiff \\
        --method dSPM --atlas HCPMMP1 --feature-mode vertex_selectkbest \\
        --classifier logistic --tune-hyperparams \\
        --sw-dur 40 --sw-step 5
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DECODE_OUTPUT_ROOT, DEFAULT_C, SW_DUR, SW_STEP_SIZE,
    classifier_path_segment, explore_run_segment,
)


def parse_args():
    p = argparse.ArgumentParser(
        description='Reshape explore_full.csv into run_decode per-subject CSVs'
    )
    p.add_argument('--task', required=True,
                   choices=['perception', 'overtProd'])
    p.add_argument('--stim-class', required=True,
                   choices=['prodDiff', 'percDiff'])
    p.add_argument('--method', required=True, choices=['dSPM', 'LCMV'])
    p.add_argument('--atlas', required=True,
                   choices=['aparc', 'HCPMMP1', 'Schaefer200', 'custom'])
    p.add_argument('--feature-mode', required=True,
                   choices=['pca_flip', 'vertex_pca', 'vertex_selectkbest',
                            'vertex_selectkbest_all'])
    p.add_argument('--classifier', required=True,
                   choices=['svm', 'lda', 'logistic'])
    p.add_argument('--c', type=float, default=None,
                   help='Regularization C used in the explore run. '
                        'Omit to fall back to DEFAULT_C[classifier] (matches '
                        'explore_run_segment Cdef behavior).  Ignored for lda '
                        'and when --tune-hyperparams is set.')
    p.add_argument('--tune-hyperparams', action='store_true', default=False,
                   help='Pick the tuned=True explore rows.')
    p.add_argument('--leakage-correction', action='store_true', default=False)
    p.add_argument('--pseudo-trial-size', type=int, default=0)
    p.add_argument('--sw-dur', type=int, default=SW_DUR)
    p.add_argument('--sw-step', type=int, default=SW_STEP_SIZE)
    p.add_argument('--overwrite', action='store_true', default=False,
                   help='Overwrite existing per-subject CSVs.  Without '
                        'this flag the script skips already-converted '
                        'subjects (idempotent).')
    return p.parse_args()


def _best_params_string(classifier, tune_hyperparams, effective_c, row):
    """Build the best_params descriptive string for one row."""
    if classifier == 'lda':
        return 'lda(shrinkage=auto)'
    if tune_hyperparams:
        # ``best_C`` may be NaN if the explore row predates best_params capture
        best_c = row.get('best_C')
        freq = row.get('best_C_freq')
        if pd.notna(best_c):
            elastic = ', elasticnet' if classifier == 'logistic' else ''
            freq_str = f', freq={freq:.2f}' if pd.notna(freq) else ''
            return f'{classifier}_tuned(best_C={best_c}{freq_str}{elastic})'
        return f'{classifier}_tuned(best_C=?)'
    if classifier == 'logistic':
        return f'logistic(C={effective_c}, elasticnet)'
    return f'svm(C={effective_c})'


def main():
    args = parse_args()
    effective_c = (
        args.c if args.c is not None else DEFAULT_C.get(args.classifier, 1.0)
    )

    # Where explore_decoding wrote its per-ROI CSVs
    run_seg = explore_run_segment(
        args.leakage_correction, args.pseudo_trial_size, args.c,
    )
    explore_root = (
        DECODE_OUTPUT_ROOT / 'explore' / args.task / args.method
        / args.atlas / args.feature_mode / args.stim_class / run_seg
    )
    if not explore_root.is_dir():
        sys.exit(f'ERROR: explore tree not found: {explore_root}')

    # Where source_stats_viz expects per-subject CSVs
    clf_tag = classifier_path_segment(
        args.classifier, effective_c, args.tune_hyperparams,
    )
    leakage_tag = 'leakage_corrected' if args.leakage_correction else 'raw'
    pseudo_tag = (
        f'pseudo_{args.pseudo_trial_size}'
        if args.pseudo_trial_size > 0 else 'no_pseudo'
    )
    out_dir = (
        DECODE_OUTPUT_ROOT / args.task / args.method / args.atlas
        / args.feature_mode / leakage_tag / pseudo_tag
        / f'{args.sw_dur}_{args.sw_step}' / clf_tag / args.stim_class
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Reading from:  {explore_root}')
    print(f'Writing to:    {out_dir}')
    print(f'Filter:        classifier={args.classifier} '
          f'sw_dur={args.sw_dur} sw_step={args.sw_step} '
          f'tuned={args.tune_hyperparams}')
    print()

    # Walk per-ROI explore_full.csv files and concat
    roi_dfs = []
    for roi_dir in sorted(p for p in explore_root.iterdir() if p.is_dir()):
        # explore_decoding writes sibling helper dirs like
        # ``_hyperparam_summary`` that aren't ROIs — skip silently.
        if roi_dir.name.startswith('_'):
            continue
        csv_path = roi_dir / 'explore_full.csv'
        if not csv_path.exists():
            print(f'  WARNING: {csv_path} missing, skipping ROI {roi_dir.name}')
            continue
        df = pd.read_csv(csv_path)
        df = df[
            (df['classifier'] == args.classifier)
            & (df['sw_dur'] == args.sw_dur)
            & (df['sw_step'] == args.sw_step)
            & (df['tuned'] == args.tune_hyperparams)
        ]
        if df.empty:
            print(f'  WARNING: no matching rows in {csv_path.name} for the '
                  f'requested config, skipping ROI {roi_dir.name}')
            continue
        # explore stores roi as the directory name; ensure column matches
        if 'roi' not in df.columns:
            df = df.assign(roi=roi_dir.name)
        roi_dfs.append(df)

    if not roi_dfs:
        sys.exit('ERROR: no rows matched the requested config across any ROI.')

    big = pd.concat(roi_dfs, ignore_index=True)
    n_subjects = big['subject'].nunique()
    n_rois = big['roi'].nunique()
    print(f'Matched {len(big)} rows across {n_subjects} subject(s) '
          f'and {n_rois} ROI(s)')
    print()

    written, skipped = 0, 0
    for subj, sdf in big.groupby('subject'):
        out_csv = out_dir / (
            f'{subj}_{args.task}_{args.stim_class}_'
            f'{args.sw_dur}_{args.sw_step}.csv'
        )
        if out_csv.exists() and not args.overwrite:
            print(f'  SKIP (exists): {out_csv.name}')
            skipped += 1
            continue

        # Canonical row ordering: (key ascending, ms ascending) so every
        # per-subject CSV agrees on the (key, ms) index that
        # source_stats_viz.load_subject_csvs / compute_stats rely on.
        out = sdf.sort_values(['roi', 'ms']).copy()
        out['key'] = out['roi'].astype(str).str.replace(' ', '_')
        out['decode_acc'] = out['accuracy']
        out['mean_list'] = ''  # explore does not persist per-fold scores
        out['best_params'] = out.apply(
            lambda r: _best_params_string(
                args.classifier, args.tune_hyperparams, effective_c, r,
            ),
            axis=1,
        )

        cols = ['key', 'ms', 'mean_list', 'decode_acc', 'best_params']
        # Pass through any explore best_* columns (best_C, best_C_freq,
        # best_l1_ratio, best_l1_ratio_freq, …).  source_stats_viz
        # ignores unknown columns; downstream scripts can mine them.
        for extra in sorted(c for c in out.columns if c.startswith('best_')):
            if extra not in cols:
                cols.append(extra)
        out[cols].to_csv(out_csv, index=False)
        print(f'  wrote: {out_csv.name} ({len(out)} rows)')
        written += 1

    print()
    print(f'Done: {written} written, {skipped} skipped'
          f'{" (use --overwrite to refresh)" if skipped else ""}')


if __name__ == '__main__':
    main()
