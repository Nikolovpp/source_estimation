#!/usr/bin/env python
"""Delete GC subset outputs invalidated by the --normalize axis bug.

WHY. Every ``_demean`` directory was written with the ff2cdd6 regression in
compute_subject_gc, which subtracted a cross-ROI average instead of the ERP.
For 2 ROIs that makes the pair exactly antisymmetric (rank 1) and every MVAR
fit fails; for 3 ROIs it costs one rank of three and 93% of windows are NaN,
with the surviving 7% no longer ranking the true edge first. See
validate_granger_normalize.py. Fixed 2026-08-13; all of it must be recomputed.

NOT a second bug. The 2-ROI wipeout was once blamed on the sweep forcing
--gc-mode conditional, on the theory that an empty conditioning set makes
cholesky(parcov(SIG, w, x)) degenerate. That was wrong. ss_conditional_gc
branches on an empty z exactly as MVGC's autocov_to_smvgc.m does ("if
isempty(z) % unconditional"), and with the normalize bug fixed a 2-ROI subset
in conditional mode reproduces pairwise GC to 4.4e-16. One cause, not two.

WHAT IT TOUCHES. By default subset sizes 2, 3 and 4, in ANY gc-mode:
  2 ROIs — the normalize bug; recomputed by the next sweep in pairwise mode.
  3 ROIs — the triple-wise arm. Was good data before 2026-07-29; every cell on
           disk now postdates the normalize bug. Recomputed by SCOPE=triplewise.
  4 ROIs — leftovers from before the SUBSETS loop existed, i.e. the all-ROI
           conditioning this project deliberately moved away from. Nothing
           regenerates these; they are simply stale.

Pass --sizes explicitly to keep an arm (e.g. ``--sizes 2`` to rerun only the
pairs first). The pre-2026-07-29 directories carry no ``_demean`` in their
config tag, so they are distinguishable if any turn out to be worth keeping.

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


def target_dirs(root, sizes):
    """Every rois_* directory whose subset size is in ``sizes``."""
    out = []
    for d in glob.glob(f'{root}/**/rois_*', recursive=True):
        if os.path.isdir(d) and os.path.basename(d)[len('rois_'):].count('-lh') in sizes:
            out.append(d)
    return sorted(out)


def summarise(files):
    """How much of what we are about to delete was actually usable.

    Report the finite FRACTION, not a has-any-finite-value count: under the
    normalize bug a file is typically 93% NaN, which "400 files with some
    finite values" would have made look healthy.
    """
    n_nan = n_data = 0
    tot = fin = 0
    for f in files[:400]:                       # cap: this is a report, not a proof
        try:
            z = np.load(f, allow_pickle=True)
        except Exception:
            continue
        keys = [k for k in z.files if k.startswith(('fxy_', 'fyx_'))]
        if not keys:
            continue
        v = np.concatenate([np.asarray(z[k], dtype=float).ravel() for k in keys])
        tot += v.size
        n_f = int(np.isfinite(v).sum())
        fin += n_f
        if n_f:
            n_data += 1
        else:
            n_nan += 1
    return n_nan, n_data, (100.0 * fin / tot if tot else float('nan'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--sizes', type=int, nargs='+', default=[2, 3, 4],
                    help='subset sizes to remove. Default 2, 3 and 4 — every '
                         'cell on disk was written with the ff2cdd6 normalize '
                         'bug and has to be recomputed. Narrow it (e.g. '
                         '--sizes 2) to rerun one arm at a time.')
    ap.add_argument('--delete', action='store_true',
                    help='actually remove the matching directories')
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

    sizes = set(args.sizes)
    print(f'removing subset sizes: {sorted(sizes)}')
    dirs = target_dirs(args.root, sizes)
    if not dirs:
        n_any = len(glob.glob(f'{args.root}/**/rois_*', recursive=True))
        print(f'no matching directories found among {n_any} rois_* '
              f'directories — nothing to do')
        return 0

    files = [f for d in dirs for f in glob.glob(f'{d}/**/*.npz', recursive=True)]
    by_subset = collections.Counter(os.path.basename(d) for d in dirs)
    by_mode = collections.Counter(
        'pairwise' if '_pairwise' in d else
        'conditional' if '_conditional' in d else 'other' for d in dirs)

    print(f'{len(dirs)} directories, {len(files)} files\n')
    print('  by subset:')
    for s, n in sorted(by_subset.items()):
        n_roi = s[len('rois_'):].count('-lh')
        print(f'    {s:<36} {n:>3} config dirs  ({n_roi} ROIs)')
    print('\n  by gc-mode:')
    for m, n in sorted(by_mode.items()):
        print(f'    {m:<34} {n:>3} config dirs')

    n_nan, n_data, pct = summarise(files)
    print(f'\n  sampled contents ({min(len(files), 400)} files): {n_nan} '
          f'all-NaN, {n_data} partly finite — {pct:.1f}% of all GC values '
          f'are finite')

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
    print('the next sweep recomputes the bivariate arm in pairwise mode; '
          '4-ROI subsets are no longer in SUBSETS and will not come back')
    return 0


if __name__ == '__main__':
    sys.exit(main())
