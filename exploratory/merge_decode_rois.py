#!/usr/bin/env python
"""Merge per-subject decoding CSVs of two atlases into one result set.

Used when an ROI is ADDED to an existing ROI set (e.g. sensorEquivSM =
sensorEquiv + Sensorimotor): the five old ROIs' decoding does not change
(each ROI is decoded on its own features), so only the new ROI is decoded
(``ROI_SUBSET=Sensorimotor`` in the driver) and the old ROIs' rows are
copied in from the source atlas.  Afterwards the target atlas has complete
per-subject CSVs and ``source_stats_viz.py`` / ``decode_onset_latency.py``
run on it as usual.

    python exploratory/merge_decode_rois.py --from-atlas sensorEquiv --to-atlas sensorEquivSM \\
        --method dSPM --feature-mode vertex_selectkbest

Rows for an ROI that already exists in the target CSV are never overwritten
(the target wins); subjects missing in either atlas are reported and skipped.
"""
import os
import sys
import glob
import argparse
from pathlib import Path

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from config import DECODE_OUTPUT_ROOT, SPEECH_ROIS, classifier_path_segment, feature_mode_arg, method_arg


def result_dir(task, method, atlas, feat, clf, stim, leak='raw', pseudo='no_pseudo', sw='40_5'):
    return Path(DECODE_OUTPUT_ROOT) / task / method / atlas / feat / leak / pseudo / sw / clf / stim


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--from-atlas', required=True)
    ap.add_argument('--to-atlas', required=True)
    ap.add_argument('--method', type=method_arg, default='dSPM')
    ap.add_argument('--feature-mode', type=feature_mode_arg, default='vertex_selectkbest')
    ap.add_argument('--classifier', default='logistic')
    ap.add_argument('--c', type=float, default=1.0)
    ap.add_argument('--tune-hyperparams', action='store_true', default=True)
    ap.add_argument('--tasks', nargs='+', default=['overtProd', 'perception'])
    ap.add_argument('--stim-classes', nargs='+', default=['prodDiff', 'percDiff'])
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    clf = classifier_path_segment(args.classifier, args.c, args.tune_hyperparams)
    target_rois = list(SPEECH_ROIS[args.to_atlas].keys())
    for task in args.tasks:
        for stim in args.stim_classes:
            src = result_dir(task, args.method, args.from_atlas, args.feature_mode, clf, stim)
            dst = result_dir(task, args.method, args.to_atlas, args.feature_mode, clf, stim)
            src_files = {os.path.basename(f): f for f in glob.glob(str(src / 'EEGPROD4*.csv'))}
            dst_files = {os.path.basename(f): f for f in glob.glob(str(dst / 'EEGPROD4*.csv'))}
            if not src_files:
                print(f'{task}/{stim}: no source CSVs in {src} — skipped'); continue
            if not dst_files:
                print(f'{task}/{stim}: no target CSVs in {dst} (decode the new ROI first) — skipped'); continue
            n_ok = 0
            for name, fdst in sorted(dst_files.items()):
                if name not in src_files:
                    print(f'  {name}: not in source atlas — skipped'); continue
                a = pd.read_csv(src_files[name]); b = pd.read_csv(fdst)
                have = set(b.key.unique())
                add = a[~a.key.isin(have) & a.key.isin(target_rois)]
                merged = pd.concat([b, add], ignore_index=True)
                merged['_o'] = merged.key.map({r: i for i, r in enumerate(target_rois)})
                merged = merged.sort_values(['_o', 'ms']).drop(columns='_o')
                missing = [r for r in target_rois if r not in set(merged.key.unique())]
                if missing:
                    print(f'  {name}: still missing {missing}')
                if not args.dry_run:
                    merged.to_csv(fdst, index=False)
                n_ok += 1
            print(f'{task}/{stim}: {n_ok} subjects merged -> {dst}  (ROIs: {", ".join(target_rois)})'
                  + ('  [dry run]' if args.dry_run else ''))


if __name__ == '__main__':
    main()
