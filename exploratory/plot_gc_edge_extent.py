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
from run_granger import (gc_tag, GC_OUTPUT_ROOT,              # noqa: E402
                         normalize_n_pcs_roi)

BANDS = ['theta', 'low_beta', 'high_beta']
BAND_LABEL = {'theta': 'theta 4–8 Hz', 'low_beta': 'low beta 12–18 Hz',
              'high_beta': 'high beta 18–30 Hz'}
TASKS = ['overtProd', 'perception']
# reference stretch (window start, ms), clear of both epoch edges
REF_MS = {'overtProd': (-1200.0, -700.0), 'perception': (100.0, 400.0)}
ARM_COLOR = ['#0072B2', '#D55E00', '#009E73']


def edge_ratio(root, task, stim='*Diff'):
    """Per-subject GC / own reference mean, averaged over every edge.

    Returns offset_ms (n_win,), {band: (n_subj, n_win)}, n_edges.
    """
    acc = {b: [] for b in BANDS}
    wm = None
    for d in sorted(glob.glob(os.path.join(root, 'rois_*', stim))):
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
    p.add_argument('--n-pcs-roi', nargs='+', default=None, metavar='ROI=K',
                   help='per-ROI override of --n-pcs for the second arm')
    p.add_argument('--extra-n-pcs', type=int, nargs='+', default=None,
                   help='further uniform FIXPC-k arms to overlay')
    p.add_argument('--ref-normalize', default=None,
                   help='normalize mode of the FIXPC1 reference arm when it '
                        'was only run under another mode (e.g. zscore for '
                        'the 200 ms theta arm); default: --normalize')
    p.add_argument('--edge-guard', type=float, default=30.0,
                   help='ms; draws the default stats baseline span '
                        '(epoch start + guard .. + 100 ms) unless a '
                        '--baseline-* range is given for that task')
    p.add_argument('--baseline-overtprod', type=float, nargs=2, default=None,
                   metavar=('START', 'END'),
                   help='baseline window-start range drawn for overtProd '
                        '(s, epoch time), as passed to granger_stats.py')
    p.add_argument('--baseline-perception', type=float, nargs=2,
                   default=None, metavar=('START', 'END'),
                   help='same for perception (s)')
    p.add_argument('--tasks', nargs='+', default=TASKS, choices=TASKS,
                   help='tasks to show (one column each)')
    p.add_argument('--stim-class', default=None,
                   choices=['prodDiff', 'percDiff'],
                   help='restrict to one contrast (default: pool both)')
    p.add_argument('--tol', type=float, default=0.10)
    p.add_argument('--hold', type=float, default=60.0)
    p.add_argument('--xmax', type=float, default=300.0)
    p.add_argument('--out-dir', default=os.path.join(
        GC_OUTPUT_ROOT, '_figures_final_review'))
    args = p.parse_args()

    # arms: FIXPC1, the --n-pcs[/--n-pcs-roi] policy, then any --extra-n-pcs
    over = normalize_n_pcs_roi(args.n_pcs_roi, args.n_pcs)
    ref_norm = args.ref_normalize or args.normalize
    ref_lab = ('FIXPC1' if ref_norm == args.normalize
               else f'FIXPC1 ({ref_norm})')
    arms = [(1, {}, ref_lab, ref_norm)]
    if args.n_pcs != 1 or over:
        lab = f'FIXPC{args.n_pcs}' + ''.join(
            f', {r} {v}' for r, v in sorted(over.items()))
        arms.append((args.n_pcs, over, lab, args.normalize))
    arms += [(k, {}, f'FIXPC{k}', args.normalize)
             for k in (args.extra_n_pcs or [])]
    tasks = list(args.tasks)
    stim = args.stim_class or '*Diff'
    fig, axes = plt.subplots(len(BANDS), len(tasks),
                             figsize=(5.5 * len(tasks) + 0.5, 8.5),
                             sharex='col', sharey=True, squeeze=False)
    rows = []
    bl_arg = {'overtProd': args.baseline_overtprod,
              'perception': args.baseline_perception}
    for ci, task in enumerate(tasks):
        onset_off = None
        for ai, (k, k_over, arm_lab, norm) in enumerate(arms):
            tag = gc_tag(args.order, args.win_ms, args.fs,
                         normalize=norm, n_pcs=k, n_pcs_roi=k_over)
            root = os.path.join(GC_OUTPUT_ROOT, task, args.method, args.atlas,
                                args.feature_mode, args.leakage, tag)
            try:
                off, ratio, n_edges, t0 = edge_ratio(root, task, stim)
            except SystemExit as e:
                print(f'skip {arm_lab} / {task}: {e}')
                continue
            onset_off = -t0 - args.win_ms      # last window ending by t = 0
            keep = off <= args.xmax
            # baseline actually used by the stats, as window-start offsets
            if bl_arg[task] is not None:
                bl = (1000.0 * bl_arg[task][0] - t0,
                      1000.0 * bl_arg[task][1] - t0)
            else:
                bl = (args.edge_guard, 100.0)
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
                        label=arm_lab)
                if np.isfinite(st) and st <= args.xmax:
                    ax.plot([st + t0], [g[off == st][0]], 'o', ms=8,
                            color=ARM_COLOR[ai], mec='white', mew=1.5,
                            zorder=5)
                ipk = int(np.nanargmax(g[off <= 100]))
                rows.append({'task': task, 'arm': arm_lab, 'band': b,
                             'n_subj': rs.shape[0], 'n_edges': n_edges,
                             'peak_ratio': g[ipk], 'peak_at_ms': off[ipk],
                             'settle_after_start_ms': st,
                             'settle_epoch_ms': st + t0,
                             'baseline_start_ms': bl[0] + t0,
                             'baseline_end_ms': bl[1] + t0,
                             'baseline_ratio': np.nanmean(
                                 g[(off >= bl[0]) & (off <= bl[1])])})
        for ri, b in enumerate(BANDS):
            ax = axes[ri, ci]
            ax.axhspan(1 - args.tol, 1 + args.tol, color='0.5', alpha=0.13,
                       lw=0)
            ax.axhline(1.0, color='0.35', lw=0.8)
            ax.axvspan(t0 + bl[0], t0 + bl[1], color='#b08900',
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
                ax.text(t0 + 0.5 * (bl[0] + bl[1]), 0.98,
                        'stats\nbaseline', transform=ax.get_xaxis_transform(),
                        ha='center', va='top', fontsize=8, color='#6b5300')
                if task == 'perception':
                    # bottom-left of the onset line, so it never collides
                    # with the baseline label at the top
                    ax.text(t0 + onset_off + 4, 0.03,
                            'windows from here\nreach stimulus onset',
                            transform=ax.get_xaxis_transform(), ha='left',
                            va='bottom', fontsize=8, color='0.2')
            if ri == len(BANDS) - 1:
                ax.set_xlabel('window start (ms, epoch time)')
    # one legend for the figure, in the title band, clear of every panel
    handles, labels = axes[0, 0].get_legend_handles_labels()
    top = 0.905 if len(tasks) > 1 else 0.865
    fig.legend(handles, labels, frameon=False, loc='upper center',
               bbox_to_anchor=(0.5, top), ncol=len(labels), fontsize=9)
    # wrap the title into more, shorter lines when there is one column
    sep = '\n' if len(tasks) == 1 else ' · '
    fig.suptitle(
        'Leading-edge GC inflation: GC relative to each subject\'s plateau\n'
        f'order {args.order} / {args.win_ms:g} ms @ {args.fs:g} Hz / '
        f'{args.normalize}{sep}mean ± 95% CI over subjects, all pairs, both '
        f'directions, {args.stim_class or "both contrasts"}\n'
        f'grey band: ±{100 * args.tol:.0f}% of plateau{sep}'
        f'dot: settled (stays within the band for the next {args.hold:g} ms)',
        fontsize=10)
    fig.tight_layout()
    fig.subplots_adjust(top=top - 0.07)

    os.makedirs(args.out_dir, exist_ok=True)
    tag = gc_tag(args.order, args.win_ms, args.fs, normalize=args.normalize)
    if over:
        tag = gc_tag(args.order, args.win_ms, args.fs,
                     normalize=args.normalize, n_pcs=args.n_pcs,
                     n_pcs_roi=over)
    sfx = ''
    if tasks != TASKS:
        sfx += '_' + '_'.join(tasks)
    if args.stim_class:
        sfx += f'_{args.stim_class}'
    stem = os.path.join(args.out_dir, f'edge_extent_{tag}{sfx}')
    fig.savefig(stem + '.png', dpi=160)
    pd.DataFrame(rows).to_csv(stem + '.csv', index=False)
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    print('wrote', stem + '.png')


if __name__ == '__main__':
    main()
