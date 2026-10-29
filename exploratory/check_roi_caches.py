#!/usr/bin/env python
"""Audit the ROI time-series caches of one arm: present, readable, size.

A cache file that exists but is not a readable npz (write cut off by a full
disk or a killed process) makes every stage fail at that subject while the
localize stage skips it as "already cached".  This lists, per subject and
contrast, the cache path, its size and whether it opens, prints the free
space of the cache volume, and with ``--delete-bad`` removes the corrupt
files so ``run_source_localize.py`` rebuilds exactly those.

    python exploratory/check_roi_caches.py --task overtProd --method dSPM --fs 500
    python exploratory/check_roi_caches.py --task overtProd --method dSPM --fs 500 --delete-bad
"""
import os
import sys
import shutil
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from config import (SUBJECT_IDS, ROI_TIMESERIES_ROOT, find_cached_npz, npz_is_valid,
                    method_arg, feature_mode_arg)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--task', required=True, choices=['overtProd', 'perception'])
    ap.add_argument('--method', type=method_arg, default='dSPM')
    ap.add_argument('--atlas', default='sensorEquiv')
    ap.add_argument('--feature-mode', type=feature_mode_arg, default='vertex_selectkbest')
    ap.add_argument('--leakage-correction', action='store_true')
    ap.add_argument('--fs', type=int, default=2000, choices=[2000, 500])
    ap.add_argument('--stim-classes', nargs='+', default=['prodDiff', 'percDiff'])
    ap.add_argument('--subjects', nargs='+', default=None)
    ap.add_argument('--delete-bad', action='store_true', help='remove the corrupt files')
    args = ap.parse_args()

    subjects = args.subjects or list(SUBJECT_IDS)
    bad, missing, ok = [], [], 0
    for stim in args.stim_classes:
        print(f'\n== {args.task}/{stim}  {args.method}/{args.atlas}/{args.feature_mode}  fs {args.fs}')
        for s in subjects:
            p = find_cached_npz(args.task, args.method, args.atlas, args.feature_mode,
                                args.leakage_correction, s, stim, fs=args.fs)
            if p is None:
                print(f'  {s}  MISSING'); missing.append((stim, s)); continue
            size = p.stat().st_size / 1e6
            valid = npz_is_valid(p)
            print(f'  {s}  {size:8.0f} MB  {"ok" if valid else "CORRUPT"}  {p if not valid else ""}')
            if valid:
                ok += 1
            else:
                bad.append(p)
    root = ROI_TIMESERIES_ROOT
    try:
        du = shutil.disk_usage(root)
        print(f'\ncache volume {root}: {du.free / 1e9:.0f} GB free of {du.total / 1e9:.0f} GB')
    except OSError:
        pass
    print(f'{ok} ok, {len(bad)} corrupt, {len(missing)} missing')
    if bad:
        if args.delete_bad:
            for p in bad:
                p.unlink(); print(f'  deleted {p}')
            print('rerun run_source_localize.py (or the driver localize step) to rebuild them')
        else:
            print('rerun with --delete-bad to remove the corrupt files, then rebuild with run_source_localize.py')


if __name__ == '__main__':
    main()
