#!/usr/bin/env python
"""How far into the epoch does the leading-edge GC inflation reach?

The moving-window GC is inflated in the first windows of every epoch (both
directions, strongest in theta).  The stats baseline is the leading 100 ms of
the axis, so it sits inside that inflation.  This figure measures the extent
so a clean baseline can be chosen.

Method, per task x arm (FIXPC1 / FIXPC-k) x band:

  1. reference = a stretch well clear of both epoch edges (``REF_MS``);
  2. each subject's GC curve is divided by that subject's own reference
     mean, so 1.0 = plateau;
  3. average over all ROI pairs, both directions and both contrasts, then
     mean and 95% CI over subjects;
  4. settle time = first window from which the group mean stays within
     +/- ``--tol`` of 1.0 for the next ``--hold`` ms.

Grid: rows = bands, columns = tasks, x = window start in epoch time (the
first ``--xmax`` ms of each task's axis).  Shaded:
the baseline the stats currently use (edge guard .. 100 ms).  In perception
the dotted line marks the first window that reaches stimulus onset.

    conda activate mne
    python exploratory/plot_gc_edge_extent.py --normalize none --n-pcs 4
"""
import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from granger_stats import load_gc_group                        # noqa: E402
from run_granger import gc_tag, GC_OUTPUT_ROOT                  # noqa: E402

BANDS = ['theta', 'low_beta', 'high_beta']
BAND_LABEL = {'theta': 'theta 4–8 Hz', 'low_beta': 'low beta 12–18 Hz',
              'high_beta': 'high beta 18–30 Hz'}
TASKS = ['overtProd', 'perception']
# reference stretch (window start, ms), clear of both epoch edges
REF_MS = {'overtProd': (-1200.0, -700.0), 'perception': (100.0, 400.0)}
ARM_COLOR = ['#0072B2', '#D55E00']


def edge_ratio(root, task):
    """Per-subject GC / own reference mean, averaged over every edge.

    Returns offset_ms (n_win,), {band: (n_subj, n_win)}, n_edges.
    """
    acc = {b: [] for b in BANDS}
    wm = None
    for d in sorted(glob.glob(os.path.join(root, 'rois_*', '*Diff'))):
        if not glob.glob(os.path.join(d, '*.npz')):
            continue
        agg = load_gc_group(d, BANDS)
        if len(agg['roi_names']) != 2:
            continue                                   # pairwise dirs only
        wm = agg['window_ms']
        for b in BANDS:
            acc[b] += [agg['fxy'][b], agg['fyx'][b]]
    if wm is None:
        raise SystemExit(f'no pairwise GC results under {root}')
    ref = (wm >= REF_MS[task][0]) & (wm <= REF_MS[task][1])
    out = {}
    for b in BANDS:
        A = np.concatenate(acc[b], axis=1)             # subj, edges, win
        r = A / np.nanmean(A[:, :, ref], axis=2, keepdims=True)
        out[b] = np.nanmean(r, axis=1)
    return wm - wm[0], out, A.shape[1], float(wm[0])


def settle_ms(off, g, tol, hold):
    n = int(round(hold / (off[1] - off[0])))
    for i in range(len(g) - n):
        if np.all(np.abs(g[i:i + n] - 1.0) <= tol):
            return float(off[i])
    return np.nan


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--method', default='LCMV')
    p.add_argument('--atlas', default='custom')
    p.add_argument('--feature-mode', default='vertex_selectkbest')
    p.add_argument('--leakage', default='leakage_corrected')
    p.add_argument('--order', type=int, default=10)
    p.add_argument('--win-ms', type=float, default=80.0)
    p.add_argument('--fs', type=float, default=200.0)
    p.add_argument('--normalize', default='none')
    p.add_argument('--n-pcs', type=int, default=4,
                   help='second arm; FIXPC1 is always the first')
    p.add_argument('--edge-guard', type=float, default=30.0,
                   help='ms; only used to draw the current baseline span')
    p.add_argument('--tol', type=float, default=0.10)
    p.add_argument('--hold', type=float, default=60.0)
    p.add_argument('--xmax', type=float, default=300.0)
    p.add_argument('--out-dir', default=os.path.join(
        GC_OUTPUT_ROOT, '_figures_final_review'))
    args = p.parse_args()

    arms = [1] if args.n_pcs == 1 else [1, args.n_pcs]
    fig, axes = plt.subplots(len(BANDS), len(TASKS), figsize=(11, 8.5),
                             sharex='col', sharey=True)
    rows = []
    for ci, task in enumerate(TASKS):
        onset_off = None
        for ai, k in enumerate(arms):
            tag = gc_tag(args.order, args.win_ms, args.fs,
                         normalize=args.normalize, n_pcs=k)
            root = os.path.join(GC_OUTPUT_ROOT, task, args.method, args.atlas,
                                args.feature_mode, args.leakage, tag)
            off, ratio, n_edges, t0 = edge_ratio(root, task)
            onset_off = -t0 - args.win_ms      # last window ending by t = 0
            keep = off <= args.xmax
            for ri, b in enumerate(BANDS):
                rs = ratio[b]
                g = np.nanmean(rs, axis=0)
                n = np.sum(np.isfinite(rs), axis=0)
                ci95 = (stats.t.ppf(0.975, n - 1)
                        * np.nanstd(rs, axis=0, ddof=1) / np.sqrt(n))
                st = settle_ms(off, g, args.tol, args.hold)
                ax = axes[ri, ci]
                t = off + t0                   # epoch time of window start
                ax.fill_between(t[keep], (g - ci95)[keep], (g + ci95)[keep],
                                color=ARM_COLOR[ai], alpha=0.18, lw=0)
                ax.plot(t[keep], g[keep], color=ARM_COLOR[ai], lw=2,
                        label=f'FIXPC{k}')
                if np.isfinite(st) and st <= args.xmax:
                    ax.plot([st + t0], [g[off == st][0]], 'o', ms=8,
                            color=ARM_COLOR[ai], mec='white', mew=1.5,
                            zorder=5)
                ipk = int(np.nanargmax(g[off <= 100]))
                rows.append({'task': task, 'arm': f'FIXPC{k}', 'band': b,
                             'n_subj': rs.shape[0], 'n_edges': n_edges,
                             'peak_ratio': g[ipk], 'peak_at_ms': off[ipk],
                             'settle_after_start_ms': st,
                             'settle_epoch_ms': st + t0,
                             'current_baseline_ratio': np.nanmean(
                                 g[(off >= args.edge_guard) & (off <= 100)])})
        for ri, b in enumerate(BANDS):
            ax = axes[ri, ci]
            ax.axhspan(1 - args.tol, 1 + args.tol, color='0.5', alpha=0.13,
                       lw=0)
            ax.axhline(1.0, color='0.35', lw=0.8)
            ax.axvspan(t0 + args.edge_guard, t0 + 100, color='#b08900',
                       alpha=0.13, lw=0)
            if task == 'perception' and onset_off is not None:
                ax.axvline(t0 + onset_off, color='0.2', ls=':', lw=1.2)
                ax.axvline(0.0, color='0.2', lw=0.8)
            ax.set_xlim(t0, t0 + args.xmax)
            for s in ('top', 'right'):
                ax.spines[s].set_visible(False)
            ax.grid(axis='y', color='0.9', lw=0.6)
            if ci == 0:
                ax.set_ylabel(f'{BAND_LABEL[b]}\nGC / plateau')
            if ri == 0:
                ax.set_title(task,
                             fontsize=11, loc='left')
                ax.text(t0 + 65, 0.98, 'current\nbaseline', transform=
                        ax.get_xaxis_transform(), ha='center', va='top',
                        fontsize=8, color='#6b5300')
                if task == 'perception':
                    ax.text(t0 + onset_off + 4, 0.98,
                            'windows from here\nreach stimulus onset',
                            transform=ax.get_xaxis_transform(), ha='left',
                            va='top', fontsize=8, color='0.2')
            if ri == len(BANDS) - 1:
                ax.set_xlabel('window start (ms, epoch time)')
    axes[0, 0].legend(frameon=False, loc='upper right', fontsize=9)
    fig.suptitle(
        'Leading-edge GC inflation: GC relative to each subject\'s plateau\n'
        f'order {args.order} / {args.win_ms:g} ms @ {args.fs:g} Hz / '
        f'{args.normalize} · mean ± 95% CI over subjects, all pairs, both '
        f'directions, both contrasts\n'
        f'grey band: ±{100 * args.tol:.0f}% of plateau · '
        f'dot: settled (stays within the band for the next {args.hold:g} ms)',
        fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    os.makedirs(args.out_dir, exist_ok=True)
    tag = gc_tag(args.order, args.win_ms, args.fs, normalize=args.normalize)
    stem = os.path.join(args.out_dir, f'edge_extent_{tag}')
    fig.savefig(stem + '.png', dpi=160)
    pd.DataFrame(rows).to_csv(stem + '.csv', index=False)
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    print('wrote', stem + '.png')


if __name__ == '__main__':
    main()
