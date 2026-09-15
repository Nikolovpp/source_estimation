#!/usr/bin/env python
"""One-page review figure for a single ROI pair: Fxy / Fyx / Diff-TRGC.

Three stacked panels over the FULL moving-window axis (no baseline/task-end
crop — this is the review view, so the edges are shown too):

    top     Fxy   directed GC  roi_i -> roi_j
    middle  Fyx   directed GC  roi_j -> roi_i
    bottom  dTRGC net Diff-TRGC, oriented i->j (positive = net i->j)

All four bands are drawn in each panel (subject mean ± SEM, n from the cache).
The gray span is the stats baseline (axis start + [edge_guard, 100] ms); the
colored bars under each panel mark the FWER-controlled significant permutation
clusters from that pair's ``group_stats`` CSV (GC right-tailed, TRGC
two-tailed) — run ``granger_stats.py`` first or the bars are simply absent.

    conda activate mne
    python plot_gc_edge_review.py --task overtProd --stim-class prodDiff \
        --roi-subset pmc-lh tpc-lh --order 10 --win-ms 80
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_granger import GC_OUTPUT_ROOT, gc_tag, roiset_tag
from granger import DEFAULT_BANDS, band_masks
from granger_stats import load_gc_group

# The bands of interest (alpha deliberately not shown), plus 'beta' — the
# combined 12–30 Hz band, reconstructed below from the two stored beta means.
SHOW_BANDS = ['theta', 'low_beta', 'high_beta', 'beta']

# Fixed band -> hue assignment; color follows the entity, never re-cycled.
BAND_COLOR = {'theta': '#2a78d6', 'low_beta': '#1baf7a',
              'high_beta': '#eda100', 'beta': '#e87ba4'}
BAND_LABEL = {'theta': 'theta 4–8 Hz', 'low_beta': 'low beta 12–18 Hz',
              'high_beta': 'high beta 18–30 Hz',
              'beta': 'beta 12–30 Hz (combined)'}


def band_stack(agg, key, band, freqs):
    """(n_subj, n_win) band trace; 'beta' = bin-weighted union of the betas.

    The npz caches persist band MEANS only, so the combined beta cannot be
    re-averaged from frequency-resolved GC.  It does not need to be: each
    stored band value is the plain mean over that band's frequency bins, so
    the mean over the union of bins is the bin-count-weighted average of the
    two stored means.  The weights come from band_masks on the run's own
    frequency grid (6 bins for [12,18), 13 for [18,30] on the 1 Hz grid).
    """
    if band != 'beta':
        return agg[key][band][:, 0, :]
    masks = band_masks(freqs, DEFAULT_BANDS)
    n_lo, n_hi = masks['low_beta'].sum(), masks['high_beta'].sum()
    return (n_lo * agg[key]['low_beta'][:, 0, :]
            + n_hi * agg[key]['high_beta'][:, 0, :]) / (n_lo + n_hi)


def contiguous_spans(mask):
    mask = np.asarray(mask, bool)
    if not mask.any():
        return []
    d = np.diff(mask.astype(int))
    starts = list(np.where(d == 1)[0] + 1)
    ends = list(np.where(d == -1)[0])
    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        ends.append(mask.size - 1)
    return list(zip(starts, ends))


def sig_spans_from_csv(csv_path, measure, src, tgt, band, window_ms,
                       col='sig_cluster'):
    """[(t0, t1), ...] spans (ms) where ``col`` is True for this edge x band.

    ``col='sig'`` gives the pointwise per-window test at p<0.05, uncorrected;
    ``col='sig_cluster'`` the FWER-controlled permutation clusters.
    """
    if not os.path.exists(csv_path):
        return []
    df = pd.read_csv(csv_path)
    g = df[(df['measure'] == measure) & (df['src'] == src)
           & (df['tgt'] == tgt) & (df['band'] == band)
           & (df[col] == True)]                                # noqa: E712
    if g.empty:
        return []
    hit = np.isin(np.round(window_ms, 3), np.round(g['window_ms'].to_numpy(), 3))
    return [(window_ms[a], window_ms[b]) for a, b in contiguous_spans(hit)]


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
    p.add_argument('--normalize', default='zscore')
    p.add_argument('--edge-guard', type=float, default=5.0)
    p.add_argument('--format', default='png', choices=['png', 'svg'])
    args = p.parse_args()

    gc_dir = (GC_OUTPUT_ROOT / args.task / args.method / args.atlas
              / args.feature_mode / 'leakage_corrected'
              / gc_tag(args.order, args.win_ms, args.target_fs, args.normalize)
              / roiset_tag(args.roi_subset) / args.stim_class)
    agg = load_gc_group(str(gc_dir), DEFAULT_BANDS)
    if 'dtrgc' not in agg:
        raise SystemExit(f'no dtrgc arrays in {gc_dir} — run with --trgc first')
    wm = agg['window_ms']
    roi = agg['roi_names']
    i, j = int(agg['pair_i'][0]), int(agg['pair_j'][0])
    n_subj = len(agg['subjects'])
    import glob
    first_npz = sorted(glob.glob(os.path.join(gc_dir, '*.npz')))[0]
    freqs = np.load(first_npz, allow_pickle=True)['freqs']
    csv_path = os.path.join(gc_dir, 'group_stats',
                            'gc_task_vs_baseline_stats_ttest.csv')
    baseline = (float(wm[0]) + args.edge_guard, float(wm[0]) + 100.0)

    panels = [('fxy', f'{roi[i]} → {roi[j]}', 'gc', roi[i], roi[j]),
              ('fyx', f'{roi[j]} → {roi[i]}', 'gc', roi[j], roi[i]),
              ('dtrgc', f'net ΔTRGC (positive = {roi[i]} → {roi[j]})',
               'dtrgc', roi[i], roi[j])]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for ax, (key, label, measure, src, tgt) in zip(axes, panels):
        for b in SHOW_BANDS:
            X = band_stack(agg, key, b, freqs)             # (n_subj, n_win)
            m = np.nanmean(X, axis=0)
            n_fin = np.maximum(np.isfinite(X).sum(axis=0), 1)
            se = np.nanstd(X, axis=0, ddof=1) / np.sqrt(n_fin)
            c = BAND_COLOR[b]
            ax.plot(wm, m, color=c, lw=1.8, label=BAND_LABEL[b])
            ax.fill_between(wm, m - se, m + se, color=c, alpha=0.16, lw=0)
        ax.axvspan(*baseline, color='0.55', alpha=0.18, lw=0)
        ax.axvline(0, color='k', lw=0.8, alpha=0.6)
        if key == 'dtrgc':
            ax.axhline(0, color='k', lw=0.6, alpha=0.6)
        # Significance rows under the data, one per band: thin bars where the
        # POINTWISE test is p<0.05 (uncorrected), a black asterisk at the
        # centre of each FWER-significant permutation cluster.
        y0, y1 = ax.get_ylim()
        row_h = 0.035 * (y1 - y0)
        drew_any = False
        for r, b in enumerate(SHOW_BANDS):
            y_row = y0 - (r + 1) * row_h
            for t0, t1 in sig_spans_from_csv(csv_path, measure, src, tgt, b,
                                             wm, col='sig'):
                ax.plot([t0, t1], [y_row] * 2, color=BAND_COLOR[b], lw=3,
                        solid_capstyle='butt', clip_on=False)
                drew_any = True
            for t0, t1 in sig_spans_from_csv(csv_path, measure, src, tgt, b,
                                             wm, col='sig_cluster'):
                ax.plot(0.5 * (t0 + t1), y_row, marker='*', color='k',
                        ms=11, mew=0, clip_on=False, zorder=5)
                drew_any = True
        if drew_any:
            ax.set_ylim(y0 - 5.5 * row_h, y1)
        ax.set_ylabel('ΔTRGC' if key == 'dtrgc' else 'GC')
        ax.set_title(label, fontsize=11, loc='left')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', color='0.9', lw=0.6)
        ax.set_axisbelow(True)
    axes[-1].set_xlabel('window start (ms)')
    axes[0].legend(fontsize=9, ncol=4, loc='upper right', frameon=False)

    pair_lbl = f"{roi[i].replace('-lh', '')}–{roi[j].replace('-lh', '')}"
    cfg_lbl = (f'order {args.order} / {args.win_ms:g} ms @ {args.target_fs:g} Hz'
               f' / {args.normalize}')
    fig.suptitle(
        f'{pair_lbl}   {args.task}/{args.stim_class}   {cfg_lbl}   n={n_subj}\n'
        'mean ± SEM over subjects, full axis (uncropped) · gray span: stats '
        'baseline · bars: pointwise p<0.05 uncorrected · ★: FWER-significant '
        'permutation cluster (GC right-tailed, TRGC two-tailed)', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out_dir = GC_OUTPUT_ROOT / '_figures_final_review'
    os.makedirs(out_dir, exist_ok=True)
    tag = gc_tag(args.order, args.win_ms, args.target_fs, args.normalize)
    out = os.path.join(
        out_dir, f'{args.task}_{args.stim_class}_'
                 f"{pair_lbl.replace('–', '+')}_{tag}.{args.format}")
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
