#!/usr/bin/env python
"""Compare the preprocessing arms of run_gc_normalize_ab.sh on the GC itself.

The logs already answer the MODEL-VALIDATION half (spectral radius,
consistency, non-minimum-phase count). This answers the half that needs the
.npz: does the reported effect survive each arm?

    arm 0  none                    no ERP removal (BSMART-faithful)
    arm A  demean                  ERP removed, pooled over classes
    arm B  zscore                  + point-by-point ensemble-SD normalisation
    arm C  demean, no per-trial    --no-demean-trials
    arm D  demean, per class       --normalize-per-class

For each arm it reports, per band and direction, the group mean GC over the
non-edge windows, the paired difference against arm A, and whether the
strongest band and the sign of the edge are preserved. Arm A is the reference
because it is the current default, not because it is correct.

EDGE WINDOWS are dropped by the measured rule (30 ms at each end; see
plot_gc_sweep_summary.EDGE_MS), and averaging is NaN-aware.

    conda activate mne
    python compare_normalize_arms.py                    # both tasks, order 6+10
    python compare_normalize_arms.py --order 10
"""
import os
import sys
import glob
import argparse
import collections
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_granger import GC_OUTPUT_ROOT
from granger import DEFAULT_BANDS

EDGE_MS = 30.0
# (label, the config-tag suffix gc_tag produces)
ARM_TAGS = [
    ('0 none',            ''),
    ('A demean',          '_demean'),
    ('B zscore',          '_zscore'),
    ('C demean noTrial',  '_demean_notrialdemean'),
    ('D demean perClass', '_demean_perclass'),
]
TASKS = [('perception', 'percDiff'), ('overtProd', 'prodDiff')]


def edge_drop(window_ms):
    if window_ms.size < 2:
        return 0
    step = float(np.median(np.diff(window_ms)))
    return int(min(np.ceil(EDGE_MS / max(step, 1e-9)), window_ms.size // 4))


def load_arm(task, stim, order, win_ms, fs, suffix, rois):
    """-> {(src, tgt, band): {subject: value}} averaged over non-edge windows."""
    cfg = f'order{order}_win{win_ms:g}ms_fs{fs:g}{suffix}'
    pat = (f'{GC_OUTPUT_ROOT}/{task}/**/{cfg}/rois_{rois}/{stim}/*.npz')
    files = sorted(glob.glob(pat, recursive=True))
    out = collections.defaultdict(dict)
    for f in files:
        subj = os.path.basename(f).split('_')[0]
        z = np.load(f, allow_pickle=True)
        k = edge_drop(np.asarray(z['window_ms'], float))
        names = [str(r) for r in z['roi_names']]
        for a, (i, j) in enumerate(zip(z['pair_i'], z['pair_j'])):
            for key, edge in ((f'fxy_', (names[i], names[j])),
                              (f'fyx_', (names[j], names[i]))):
                for b in DEFAULT_BANDS:
                    kk = key + b
                    if kk not in z.files:
                        continue
                    v = np.asarray(z[kk], float)
                    v = v[a] if v.ndim == 2 else v
                    v = v[k:len(v) - k] if k else v
                    out[(edge[0], edge[1], b)][subj] = float(np.nanmean(v))
    return dict(out), len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--orders', type=int, nargs='+', default=[6, 10])
    ap.add_argument('--win-ms', type=float, default=60.0)
    ap.add_argument('--target-fs', type=float, default=200.0)
    ap.add_argument('--rois', default='awfa-lh-ifc-lh')
    args = ap.parse_args()

    print(f'root: {GC_OUTPUT_ROOT}')
    print(f'edge guard: {EDGE_MS:g} ms at each end;  ROIs {args.rois}\n')

    for task, stim in TASKS:
        for order in args.orders:
            arms = {}
            for label, suf in ARM_TAGS:
                d, n = load_arm(task, stim, order, args.win_ms,
                                args.target_fs, suf, args.rois)
                if d:
                    arms[label] = (d, n)
            if 'A demean' not in arms:
                print(f'{task}/{stim} order{order}: reference arm A missing '
                      f'— found {list(arms)}\n')
                continue
            ref, n_ref = arms['A demean']
            edges = sorted({(s, t) for s, t, _ in ref})

            print('=' * 78)
            print(f'{task} / {stim}   order {order}, {args.win_ms:g} ms window')
            print('=' * 78)
            for src, tgt in edges:
                print(f'\n  {src} -> {tgt}')
                print(f'    {"arm":<20}' + ''.join(f'{b:>12}' for b in DEFAULT_BANDS)
                      + f'{"top band":>12}')
                for label, _ in ARM_TAGS:
                    if label not in arms:
                        continue
                    d, n = arms[label]
                    means, cells = [], {}
                    for b in DEFAULT_BANDS:
                        vals = d.get((src, tgt, b), {})
                        m = np.nanmean(list(vals.values())) if vals else np.nan
                        cells[b] = m
                        means.append(m)
                    top = max(DEFAULT_BANDS, key=lambda b: (cells[b]
                              if np.isfinite(cells[b]) else -np.inf))
                    print(f'    {label:<20}'
                          + ''.join(f'{m:>12.4f}' for m in means)
                          + f'{top:>12}   (n={n})')

                # paired difference vs arm A, per band
                for label, _ in ARM_TAGS:
                    if label == 'A demean' or label not in arms:
                        continue
                    d, _ = arms[label]
                    diffs = []
                    for b in DEFAULT_BANDS:
                        common = sorted(set(d.get((src, tgt, b), {}))
                                        & set(ref.get((src, tgt, b), {})))
                        if not common:
                            diffs.append(np.nan); continue
                        a = np.array([ref[(src, tgt, b)][s] for s in common])
                        c = np.array([d[(src, tgt, b)][s] for s in common])
                        with np.errstate(invalid='ignore', divide='ignore'):
                            diffs.append(100 * np.nanmean((c - a) / np.abs(a)))
                    print(f'    {"  " + label + " vs A":<20}'
                          + ''.join(f'{v:>11.1f}%' for v in diffs))
            print()

    order_stability(args)


def order_stability(args):
    """Which arm survives changing the model order — the sweep's own question.

    The arms are nearly indistinguishable on magnitude; where they differ is
    whether the answer holds when an arbitrary modelling choice changes. For
    each arm this reports the top band at each order, whether it flips, and the
    mean absolute change in band GC.
    """
    if len(args.orders) < 2:
        return
    lo, hi = min(args.orders), max(args.orders)
    print('=' * 78)
    print(f'ORDER STABILITY: order {lo} vs order {hi}, same data')
    print('=' * 78)
    print(f'{"task":<11} {"edge":<20} {"arm":<20} {"top@" + str(lo):>10} '
          f'{"top@" + str(hi):>10} {"flip":>6} {"mean |d|":>9}')
    bands = list(DEFAULT_BANDS)
    flips = collections.Counter()
    tot = 0
    for task, stim in TASKS:
        per = {}
        for o in (lo, hi):
            for lab, suf in ARM_TAGS:
                per[(o, lab)], _ = load_arm(task, stim, o, args.win_ms,
                                            args.target_fs, suf, args.rois)
        if not per.get((lo, 'A demean')):
            continue
        for src, tgt in sorted({(a, b) for a, b, _ in per[(lo, 'A demean')]}):
            for lab, _ in ARM_TAGS:
                v = {}
                for o in (lo, hi):
                    d = per[(o, lab)]
                    if not d:
                        v = None; break
                    v[o] = {b: np.nanmean(list(
                        d.get((src, tgt, b), {0: np.nan}).values()))
                        for b in bands}
                if v is None:
                    continue
                t_lo = max(bands, key=lambda b: v[lo][b])
                t_hi = max(bands, key=lambda b: v[hi][b])
                rel = 100 * np.nanmean([abs(v[hi][b] - v[lo][b]) / abs(v[lo][b])
                                        for b in bands])
                flips[lab] += t_lo != t_hi
                print(f'{task:<11} {src + "->" + tgt:<20} {lab:<20} '
                      f'{t_lo:>10} {t_hi:>10} '
                      f'{"YES" if t_lo != t_hi else "-":>6} {rel:>8.0f}%')
            tot += 1
            print()
    if tot:
        print(f'band-ranking flips over {tot} task x direction cells:')
        for lab, _ in ARM_TAGS:
            print(f'  {lab:<22} {flips[lab]}/{tot}')


if __name__ == '__main__':
    sys.exit(main())
