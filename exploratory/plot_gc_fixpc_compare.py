#!/usr/bin/env python
"""FIXPC1 vs FIXPC-k on the same edge: does keeping more PCs per ROI change GC?

Companion to plot_gc_edge_review.py.  One figure per ROI pair x task x
contrast, a grid of

    rows     bands (theta, low beta, high beta, combined beta)
    columns  Fxy (i->j)  |  Fyx (j->i)  |  net dTRGC (positive = i->j)

Each panel overlays the two aggregation arms of the SAME run configuration
(order / window / fs / normalize): FIXPC1 = one first-PC virtual channel per
ROI and bivariate GC (dashed, grey); FIXPC-k = top-k fixed-filter PCs per ROI
and block GC (solid, colour).  Mean ± SEM over subjects, full moving-window
axis.  Below each panel: bars where the pointwise test is p<0.05 uncorrected
and a star at the centre of every FWER-significant permutation cluster, one
row per arm, read from each arm's own ``group_stats`` CSV (run
``granger_stats.py`` on both first, or the markers are simply absent).

Block GC is not on the same scale as bivariate GC — with k components per
ROI the target block has k channels' worth of innovations to explain, so
the log-determinant ratio is generally larger.  Compare the SHAPE over time
and the task-vs-baseline contrast, not the raw level; the dTRGC panel is
the fairest comparison because TRGC is a within-arm difference.

    conda activate mne
    python exploratory/plot_gc_fixpc_compare.py --task overtProd \\
        --stim-class prodDiff --roi-subset awfa-lh tpc-lh --normalize none \\
        --n-pcs 4 --edge-guard 30
"""
import os
import sys
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
from run_granger import (GC_OUTPUT_ROOT, gc_tag, roiset_tag,
                         normalize_n_pcs_roi)
from granger import DEFAULT_BANDS
from granger_stats import load_gc_group
from plot_gc_edge_review import (band_stack, sig_spans_from_csv, SHOW_BANDS,
                                 BAND_COLOR, BAND_LABEL)

ARM_STYLE = {1: dict(color='0.45', ls='--', lw=1.6),
             'k': dict(ls='-', lw=1.9)}


def load_arm(args, n_pcs, n_pcs_roi=None):
    gc_dir = (GC_OUTPUT_ROOT / args.task / args.method / args.atlas
              / args.feature_mode / 'leakage_corrected'
              / gc_tag(args.order, args.win_ms, args.target_fs, args.normalize,
                       n_pcs=n_pcs, n_pcs_roi=n_pcs_roi)
              / roiset_tag(args.roi_subset) / args.stim_class)
    agg = load_gc_group(str(gc_dir), DEFAULT_BANDS)
    if 'dtrgc' not in agg:
        raise SystemExit(f'no dtrgc arrays in {gc_dir} — run with --trgc first')
    first_npz = sorted(glob.glob(os.path.join(gc_dir, '*.npz')))[0]
    with np.load(first_npz, allow_pickle=True) as d:
        freqs = d['freqs']
        n_comp = d['n_comp'].tolist() if 'n_comp' in d.files else None
    csv_path = os.path.join(gc_dir, 'group_stats',
                            'gc_task_vs_baseline_stats_ttest.csv')
    return agg, freqs, csv_path, n_comp


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--task', required=True, choices=['perception', 'overtProd'])
    p.add_argument('--stim-class', required=True)
    p.add_argument('--roi-subset', nargs=2, required=True, metavar='ROI')
    p.add_argument('--method', default='LCMV')
    p.add_argument('--atlas', default='custom')
    p.add_argument('--feature-mode', default='vertex_selectkbest')
    p.add_argument('--order', type=int, default=10)
    p.add_argument('--win-ms', type=float, default=80.0)
    p.add_argument('--target-fs', type=float, default=200.0)
    p.add_argument('--normalize', default='none')
    p.add_argument('--n-pcs', type=int, default=4,
                   help='the FIXPC-k arm compared against FIXPC1')
    p.add_argument('--n-pcs-roi', nargs='+', default=None, metavar='ROI=K',
                   help='per-ROI override of --n-pcs in the FIXPC-k arm '
                        '(run_granger.py --n-pcs-roi), e.g. pmc-lh=3')
    p.add_argument('--edge-guard', type=float, default=30.0,
                   help='ms; only draws the stats baseline span')
    p.add_argument('--format', default='png', choices=['png', 'svg'])
    args = p.parse_args()
    over = normalize_n_pcs_roi(args.n_pcs_roi, args.n_pcs)
    # arm label: the number for a uniform run, the policy for a mixed one
    k = str(args.n_pcs) + ''.join(f'_{r}{v}' for r, v in sorted(over.items()))

    agg1, freqs, csv1, _ = load_arm(args, 1)
    aggk, _, csvk, n_comp = load_arm(args, args.n_pcs, over)
    wm = agg1['window_ms']
    if aggk['window_ms'].shape != wm.shape or not np.allclose(aggk['window_ms'], wm):
        raise SystemExit('the two arms have different window axes')
    roi = agg1['roi_names']
    i, j = int(agg1['pair_i'][0]), int(agg1['pair_j'][0])
    n1, nk = len(agg1['subjects']), len(aggk['subjects'])
    baseline = (float(wm[0]) + args.edge_guard, float(wm[0]) + 100.0)

    cols = [('fxy', f'{roi[i]} → {roi[j]}', 'gc', roi[i], roi[j]),
            ('fyx', f'{roi[j]} → {roi[i]}', 'gc', roi[j], roi[i]),
            ('dtrgc', f'net ΔTRGC (+ = {roi[i]} → {roi[j]})', 'dtrgc',
             roi[i], roi[j])]
    arms = [(1, agg1, csv1), (k, aggk, csvk)]

    fig, axes = plt.subplots(len(SHOW_BANDS), 3, figsize=(15, 11),
                             sharex=True)
    for r, b in enumerate(SHOW_BANDS):
        c = BAND_COLOR[b]
        for cc, (key, label, measure, src, tgt) in enumerate(cols):
            ax = axes[r, cc]
            # Block GC (k channels per ROI) sits on a much larger scale than
            # bivariate GC, so FIXPC1 gets its own right-hand axis: the
            # comparison is of SHAPE over time, not of level.
            ax1 = ax.twinx()
            ax1.spines[['top']].set_visible(False)
            ax1.tick_params(axis='y', colors='0.45', labelsize=8)
            for n_pcs, agg, csv_path in arms:
                X = band_stack(agg, key, b, freqs)
                m = np.nanmean(X, axis=0)
                n_fin = np.maximum(np.isfinite(X).sum(axis=0), 1)
                se = np.nanstd(X, axis=0, ddof=1) / np.sqrt(n_fin)
                st = dict(ARM_STYLE[1]) if n_pcs == 1 else dict(ARM_STYLE['k'], color=c)
                tgt_ax = ax1 if n_pcs == 1 else ax
                tgt_ax.plot(wm, m, label=f'FIXPC{n_pcs}', **st)
                tgt_ax.fill_between(wm, m - se, m + se, color=st['color'],
                                    alpha=0.14, lw=0)
            if key == 'dtrgc':
                # Both arms' zero lines must coincide: make each axis
                # symmetric about 0 so the sign of net TRGC reads the same.
                for a in (ax, ax1):
                    lim = max(abs(v) for v in a.get_ylim())
                    a.set_ylim(-lim, lim)
            ax.axvspan(*baseline, color='0.55', alpha=0.18, lw=0)
            ax.axvline(0, color='k', lw=0.8, alpha=0.6)
            if key == 'dtrgc':
                ax.axhline(0, color='k', lw=0.6, alpha=0.6)
            # significance rows: one per arm (FIXPC1 grey, FIXPC-k colour)
            y0, y1 = ax.get_ylim()
            row_h = 0.05 * (y1 - y0)
            drew = False
            for rr, (n_pcs, agg, csv_path) in enumerate(arms):
                y_row = y0 - (rr + 1) * row_h
                col_ = '0.45' if n_pcs == 1 else c
                for t0, t1 in sig_spans_from_csv(csv_path, measure, src, tgt,
                                                 b, wm, col='sig'):
                    ax.plot([t0, t1], [y_row] * 2, color=col_, lw=3,
                            solid_capstyle='butt', clip_on=False)
                    drew = True
                for t0, t1 in sig_spans_from_csv(csv_path, measure, src, tgt,
                                                 b, wm, col='sig_cluster'):
                    ax.plot(0.5 * (t0 + t1), y_row, marker='*', color='k',
                            ms=10, mew=0, clip_on=False, zorder=5)
                    drew = True
            if drew:
                ax.set_ylim(y0 - 3.2 * row_h, y1)
            if r == 0:
                ax.set_title(label, fontsize=11, loc='left')
            if cc == 0:
                ax.set_ylabel(BAND_LABEL[b], fontsize=10)
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(axis='y', color='0.9', lw=0.6)
            ax.set_axisbelow(True)
    for ax in axes[-1]:
        ax.set_xlabel('window start (ms)')
    from matplotlib.lines import Line2D
    axes[0, 0].legend(handles=[
        Line2D([], [], label='FIXPC1 (right axis)', **ARM_STYLE[1]),
        Line2D([], [], label=f'FIXPC{k} (left axis)', color='k',
               **ARM_STYLE['k'])], fontsize=9, loc='upper right', frameon=False)

    pair_lbl = f"{roi[i].replace('-lh', '')}–{roi[j].replace('-lh', '')}"
    comp_lbl = (f'components used {n_comp}' if n_comp else '')
    fig.suptitle(
        f'{pair_lbl}   {args.task}/{args.stim_class}   order {args.order} / '
        f'{args.win_ms:g} ms @ {args.target_fs:g} Hz / {args.normalize}   '
        f'FIXPC1 (dashed grey, right axis, n={n1}) vs FIXPC{k} (solid, left axis, n={nk}; {comp_lbl})\n'
        'mean ± SEM over subjects, full axis · gray span: stats baseline · '
        'bars: pointwise p<0.05 uncorrected (upper row FIXPC1, lower row '
        f'FIXPC{k}) · ★: FWER-significant permutation cluster', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out_dir = GC_OUTPUT_ROOT / '_figures_final_review'
    os.makedirs(out_dir, exist_ok=True)
    tag = gc_tag(args.order, args.win_ms, args.target_fs, args.normalize)
    out = os.path.join(
        out_dir, f'fixpc1_vs_fixpc{k}_{args.task}_{args.stim_class}_'
                 f"{pair_lbl.replace('–', '+')}_{tag}.{args.format}")
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
