#!/usr/bin/env python
"""Delete the all-NaN bivariate GC outputs from the conditional-mode sweep.

WHY THESE EXIST. run_gc_window_order_sweep.sh forced --gc-mode conditional on
every subset. For a 2-ROI subset the conditioning set is EMPTY, which
granger_statespace cannot factor: cholesky(parcov(SIG, w, x)) degenerates and
every window returns NaN. It never raises, so the NaN guard wrote full-size
files containing nothing. Fixed in ee563e9 (2 ROIs now use --gc-mode pairwise).

WHAT IT TOUCHES. Only ``*_conditional/rois_<A>-<B>/`` — config directories in
conditional mode whose subset holds exactly two ROIs. Triple-wise output is
never a candidate, and neither is anything written in pairwise mode.

SAFETY. A file is deleted only if it is verified all-NaN on disk. Matching the
path is not enough: if any real value is present the file is KEPT and reported,
because that would mean this diagnosis is wrong for that cell. Dry-run is the
default; deleting requires --delete.

    conda activate mne
    python clean_nan_bivariate_gc.py              # report only, deletes nothing
    python clean_nan_bivariate_gc.py --delete     # actually remove
"""
import os
import sys
import glob
import argparse
import numpy as np

ROOT = ('/mnt/r/phd_thesis/Research/SpeechProduction/EEG/derivatives/'
        'source_estimation/GC_source_space')


def two_roi_conditional_dirs(root):
    """Config dirs in conditional mode whose subset holds exactly 2 ROIs."""
    out = []
    for d in glob.glob(f'{root}/**/*_conditional/rois_*', recursive=True):
        if not os.path.isdir(d):
            continue
        subset = os.path.basename(d)[len('rois_'):]
        if subset.count('-lh') == 2:
            out.append(d)
    return sorted(out)


def all_nan(path):
    """True if every GC value in the file is NaN. Errs toward KEEPING."""
    try:
        z = np.load(path, allow_pickle=True)
    except Exception as e:
        print(f'    ! unreadable, keeping: {os.path.basename(path)} ({e})')
        return False
    keys = [k for k in z.files if k.startswith(('fxy_', 'fyx_', 'dtrgc_'))]
    if not keys:
        return False                      # nothing recognisable — do not touch
    return all(np.all(np.isnan(np.asarray(z[k], dtype=float))) for k in keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--delete', action='store_true',
                    help='actually remove the verified all-NaN files')
    args = ap.parse_args()

    dirs = two_roi_conditional_dirs(args.root)
    if not dirs:
        print('no 2-ROI conditional directories found — nothing to do')
        return 0

    doomed, kept, n_files = [], [], 0
    for d in dirs:
        for f in glob.glob(f'{d}/**/*.npz', recursive=True):
            n_files += 1
            (doomed if all_nan(f) else kept).append(f)

    print(f'{len(dirs)} two-ROI conditional directories, {n_files} files\n')
    print(f'  verified all-NaN : {len(doomed)}')
    print(f'  holding data     : {len(kept)}')

    if kept:
        print('\n  KEPT — these contain real values, so the all-NaN diagnosis')
        print('  does not hold for them. Inspect before doing anything else:')
        for f in kept[:10]:
            print(f'    {f.split("GC_source_space/")[1]}')
        if len(kept) > 10:
            print(f'    ... and {len(kept) - 10} more')

    if not doomed:
        print('\nnothing to delete')
        return 0

    subsets = sorted({os.path.basename(os.path.dirname(os.path.dirname(f)))
                      for f in doomed})
    print(f'\n  subsets affected: {", ".join(subsets)}')

    if not args.delete:
        print(f'\nDRY RUN — {len(doomed)} files would be deleted. '
              f'Re-run with --delete to remove them.')
        return 0

    n = 0
    for f in doomed:
        os.remove(f)
        n += 1
    # Drop directories left empty, deepest first, but never a non-empty one.
    for d in sorted(dirs, key=len, reverse=True):
        for sub, _, _ in os.walk(d, topdown=False):
            if not os.listdir(sub):
                os.rmdir(sub)
    print(f'\ndeleted {n} files')
    return 0


if __name__ == '__main__':
    sys.exit(main())
