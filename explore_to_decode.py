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
    classifier_path_segment,
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
                   help='Regularization C the explore run was launched '
                        'with (used to locate the source dir, since '
                        'explore_decoding stamps it into the path even '
                        'for tuned runs).  Required for untuned runs '
                        '(--c picks both source and output paths).  '
                        'Optional for --tune-hyperparams: when omitted, '
                        'the converter scans lc*_pt*_C* dirs and picks '
                        'the matching one — tuned accuracies are bit-'
                        'identical across c values since C is chosen per '
                        'fold by inner CV.  Ignored for lda.')
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


def _csv_has_matching_rows(csv_path, classifier, sw_dur, sw_step, tuned):
    """Quick check whether an explore_full.csv contains target rows."""
    df = pd.read_csv(
        csv_path,
        usecols=['classifier', 'sw_dur', 'sw_step', 'tuned'],
    )
    return (
        (df['classifier'] == classifier)
        & (df['sw_dur'] == sw_dur)
        & (df['sw_step'] == sw_step)
        & (df['tuned'] == tuned)
    ).any()


def _discover_explore_root(base, leakage, pseudo, c, classifier,
                           sw_dur, sw_step, tuned):
    """Locate the explore_run_segment dir, auto-discovering when c is None.

    explore_decoding encodes the input ``c`` in its output path
    (``lc{}_pt{}_C{c}``) even for tuned runs, where the actual C is
    overridden per fold by inner GridSearchCV.  For tuned rows the
    path-c is cosmetic — any matching candidate gives bit-identical
    accuracies.  Auto-discovery means callers don't have to remember
    what c was passed to the explore invocation.
    """
    leakage_bit = int(bool(leakage))
    pseudo_int = int(pseudo)

    # Explicit c: use as-is and let a missing dir speak for itself.
    if c is not None:
        c_tag = f'{c:g}'
        target = base / f'lc{leakage_bit}_pt{pseudo_int}_C{c_tag}'
        if not target.is_dir():
            sys.exit(f'ERROR: explore dir not found: {target}')
        return target

    # Auto-discover: scan lc{}_pt{}_C* dirs for a match.
    pattern = f'lc{leakage_bit}_pt{pseudo_int}_C*'
    candidates = []
    for cand in sorted(base.glob(pattern)):
        if not cand.is_dir():
            continue
        # The (classifier, sw_dur, tuned) schema is shared across ROIs
        # within a run, so a single peek is enough.
        for roi_dir in sorted(cand.iterdir()):
            if roi_dir.name.startswith('_') or not roi_dir.is_dir():
                continue
            csv = roi_dir / 'explore_full.csv'
            if csv.exists() and _csv_has_matching_rows(
                csv, classifier, sw_dur, sw_step, tuned,
            ):
                candidates.append(cand)
            break

    if not candidates:
        sys.exit(
            f'ERROR: no explore dirs under {base} matching {pattern} '
            f'contain rows for (classifier={classifier}, sw_dur={sw_dur}, '
            f'sw_step={sw_step}, tuned={tuned}).'
        )

    if len(candidates) > 1:
        names = ', '.join(c.name for c in candidates)
        if tuned:
            print(f'  NOTE: multiple candidate explore dirs ({names}); '
                  f'using {candidates[0].name} (tuned accuracies are '
                  f'bit-identical across them — C is chosen per fold)')
        else:
            sys.exit(
                f'ERROR: untuned rows for classifier={classifier} exist '
                f'in multiple explore dirs ({names}).  Specify --c to '
                f'disambiguate (each C is a distinct decoding).'
            )

    return candidates[0]


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

    # Where explore_decoding wrote its per-ROI CSVs.  When --c is
    # given, this matches the explore_run_segment exactly; when --c is
    # omitted (only meaningful for tuned runs), the helper scans the
    # sibling lc*_pt*_C* dirs for one containing matching rows.
    base = (
        DECODE_OUTPUT_ROOT / 'explore' / args.task / args.method
        / args.atlas / args.feature_mode / args.stim_class
    )
    if not base.is_dir():
        sys.exit(f'ERROR: explore tree not found: {base}')
    explore_root = _discover_explore_root(
        base, args.leakage_correction, args.pseudo_trial_size, args.c,
        args.classifier, args.sw_dur, args.sw_step, args.tune_hyperparams,
    )

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
