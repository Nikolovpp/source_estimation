#!/usr/bin/env python
"""Delete every bivariate (2-ROI) GC output so the sweep can recompute them.

WHY. run_gc_window_order_sweep.sh forced --gc-mode conditional on every subset.
For a 2-ROI subset the conditioning set is EMPTY, which granger_statespace
cannot factor: cholesky(parcov(SIG, w, x)) degenerates. Most windows came back
NaN; the few that survived returned ln 2 (0.6931), the value GC takes when the
variance ratio degenerates to exactly 2. Either way the numbers are junk.
Fixed in ee563e9 — 2-ROI subsets now run with --gc-mode pairwise.

WHAT IT TOUCHES. Every ``rois_<A>-<B>/`` directory under GC_source_space whose
subset holds exactly two ROIs, in ANY gc-mode. Triple-wise output is never a
candidate, so the 88 good triple cells are untouched and will be skipped by the
next sweep rather than recomputed.

    conda activate mne
    python clean_nan_bivariate_gc.py              # report only, deletes nothing
    python clean_nan_bivariate_gc.py --delete     # actually remove
"""
import os
import sys
import glob
import shutil
import argparse
import collections
import numpy as np

def _default_root():
    """Derive the GC root from config.env, like every other runner here.

    Hardcoding an absolute path silently found nothing on the workstation,
    whose project root is /media/maxlab_sharedrive/... rather than /mnt/r/...
    Reuse run_granger.py's own GC_OUTPUT_ROOT so the two cannot disagree about
    where results live.
    """
    try:
        from run_granger import GC_OUTPUT_ROOT
        return str(GC_OUTPUT_ROOT)
    except Exception:
        try:
            import config
            return str(config.DERIVATIVES_ROOT / 'source_estimation'
                       / 'GC_source_space')
        except Exception:
            return ''


ROOT = _default_root()


def bivariate_dirs(root):
    """Every rois_* directory whose subset holds exactly two ROIs."""
    out = []
    for d in glob.glob(f'{root}/**/rois_*', recursive=True):
        if os.path.isdir(d) and os.path.basename(d)[len('rois_'):].count('-lh') == 2:
            out.append(d)
    return sorted(out)


def summarise(files):
    """How much of what we are about to delete was actually usable."""
    n_nan = n_data = 0
    for f in files[:400]:                       # cap: this is a report, not a proof
        try:
            z = np.load(f, allow_pickle=True)
        except Exception:
            continue
        keys = [k for k in z.files if k.startswith(('fxy_', 'fyx_'))]
        if not keys:
            continue
        finite = sum(int(np.isfinite(np.asarray(z[k], dtype=float)).sum())
                     for k in keys)
        if finite:
            n_data += 1
        else:
            n_nan += 1
    return n_nan, n_data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--delete', action='store_true',
                    help='actually remove the bivariate directories')
    args = ap.parse_args()

    if not args.root:
        print('could not resolve the GC root from config.env — '
              'pass --root explicitly', file=sys.stderr)
        return 2
    print(f'root: {args.root}')
    if not os.path.isdir(args.root):
        print(f'that directory does not exist — pass --root explicitly',
              file=sys.stderr)
        return 2

    dirs = bivariate_dirs(args.root)
    if not dirs:
        n_any = len(glob.glob(f'{args.root}/**/rois_*', recursive=True))
        print(f'no bivariate (2-ROI) directories found among {n_any} '
              f'rois_* directories — nothing to do')
        return 0

    files = [f for d in dirs for f in glob.glob(f'{d}/**/*.npz', recursive=True)]
    by_subset = collections.Counter(os.path.basename(d) for d in dirs)
    by_mode = collections.Counter(
        'pairwise' if '_pairwise' in d else
        'conditional' if '_conditional' in d else 'other' for d in dirs)

    print(f'{len(dirs)} bivariate directories, {len(files)} files\n')
    print('  by subset:')
    for s, n in sorted(by_subset.items()):
        print(f'    {s:<34} {n:>3} config dirs')
    print('\n  by gc-mode:')
    for m, n in sorted(by_mode.items()):
        print(f'    {m:<34} {n:>3} config dirs')

    n_nan, n_data = summarise(files)
    print(f'\n  sampled contents: {n_nan} all-NaN, {n_data} with some finite '
          f'values (of {min(len(files), 400)} checked)')

    if not args.delete:
        print(f'\nDRY RUN — {len(files)} files in {len(dirs)} directories would '
              f'be deleted.\nRe-run with --delete to remove them.')
        return 0

    for d in dirs:
        shutil.rmtree(d)
    # Prune config directories left empty by the removal, deepest first.
    for _ in range(4):
        for d in glob.glob(f'{args.root}/**/', recursive=True):
            try:
                if os.path.isdir(d) and not os.listdir(d):
                    os.rmdir(d)
            except OSError:
                pass
    print(f'\ndeleted {len(files)} files in {len(dirs)} directories')
    print('the next sweep will recompute the bivariate arm in pairwise mode')
    return 0


if __name__ == '__main__':
    sys.exit(main())
