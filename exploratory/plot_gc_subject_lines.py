#!/usr/bin/env python
"""Pointwise task-vs-baseline for one edge x band, under BOTH referencings.

Companion to plot_gc_edge_review.py.  Two panels, same data, same test
(right-tailed one-sample t per window, p<0.05 UNCORRECTED), differing only in
what each subject is referenced to:

    top     raw GC tested against the GROUP baseline scalar — the MATLAB
            pointwise design reported in the review figures.
    bottom  GC minus each subject's OWN baseline mean, tested against 0 —
            the referencing the cluster permutation test uses.  Removing the
            per-subject level typically shrinks the across-subject SD, so
            this panel shows what the group-scalar design costs in power.

Red bars under each panel mark the windows with p<0.05; the panel titles
count them.  Mean ± SEM of the plotted quantity in both panels.

    conda activate mne
    python plot_gc_subject_lines.py --task overtProd --stim-class prodDiff \
        --roi-subset ifc-lh tpc-lh --direction fxy --band high_beta
"""
import os
import sys
import glob
import argparse
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GC_TASK_END
from run_granger import GC_OUTPUT_ROOT, gc_tag, roiset_tag
from granger_stats import load_gc_group, REPORT_BANDS, TASK_ONSET_MS, MIN_SUBJECTS
from plot_gc_edge_review import band_stack, BAND_LABEL, BAND_COLOR, contiguous_spans


def pointwise_p(X, ref, task_mask):
    """Right-tailed one-sample t of X[:, w] against ref per task window.

    ref : scalar (group baseline) or (n_subj,) vector (own baselines).
    Returns (n_win,) p-values, NaN outside the task span.
    """
    n_win = X.shape[1]
    p = np.full(n_win, np.nan)
    ref = np.asarray(ref, dtype=float)
    for w in np.where(task_mask)[0]:
        x = X[:, w] - (ref if ref.ndim == 0 else ref)
        x = x[np.isfinite(x)]
        if x.size < MIN_SUBJECTS or not np.any(x != 0.0):
            continue
        _t, pv = stats.ttest_1samp(x, 0.0, alternative='greater')
        p[w] = pv
    return p


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--task', required=True, choices=['perception', 'overtProd'])
    p.add_argument('--stim-class', required=True)
    p.add_argument('--roi-subset', nargs=2, required=True, metavar='ROI')
    p.add_argument('--direction', default='fxy', choices=['fxy', 'fyx', 'dtrgc'])
    p.add_argument('--band', default='high_beta', choices=list(REPORT_BANDS))
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
    agg = load_gc_group(str(gc_dir), REPORT_BANDS)
    wm = agg['window_ms']
    roi = agg['roi_names']
    i, j = int(agg['pair_i'][0]), int(agg['pair_j'][0])
    freqs = np.load(sorted(glob.glob(os.path.join(gc_dir, '*.npz')))[0],
                    allow_pickle=True)['freqs']
    X = band_stack(agg, args.direction, args.band, freqs)   # (n_subj, n_win)
    n_subj = X.shape[0]

    baseline = (float(wm[0]) + args.edge_guard, float(wm[0]) + 100.0)
    base_mask = (wm >= baseline[0]) & (wm <= baseline[1])
    task_start = baseline[1]
    onset = TASK_ONSET_MS.get(args.task)
    if onset is not None:
        task_start = max(task_start, onset)
    task_end = GC_TASK_END[args.task] * 1000.0
    task_mask = (wm >= task_start) & (wm <= task_end)

    own_base = np.nanmean(X[:, base_mask], axis=1)              # (n_subj,)
    group_base = float(np.nanmean(np.nanmean(X, axis=0)[base_mask]))
    Y = X - own_base[:, None]

    p_group = pointwise_p(X, np.float64(group_base), task_mask)
    p_own = pointwise_p(X, own_base, task_mask)

    def msem(A):
        m = np.nanmean(A, axis=0)
        n_fin = np.maximum(np.isfinite(A).sum(axis=0), 1)
        return m, np.nanstd(A, axis=0, ddof=1) / np.sqrt(n_fin)

    c = BAND_COLOR[args.band]
    src, tgt = (roi[j], roi[i]) if args.direction == 'fyx' else (roi[i], roi[j])
    head = (f'net ΔTRGC (positive = {src} → {tgt})' if args.direction == 'dtrgc'
            else f'{src} → {tgt}')

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    panels = [
        (X, group_base, p_group,
         'vs GROUP baseline scalar (the reported pointwise design)',
         f'group baseline scalar ({group_base:.4f})'),
        (Y, 0.0, p_own,
         "vs each subject's OWN baseline mean (the permutation test's "
         'referencing)', "reference: each subject's baseline mean"),
    ]
    for ax, (A, ref, pv, title, ref_label) in zip(axes, panels):
        m, se = msem(A)
        ax.plot(wm, m, color=c, lw=2)
        ax.fill_between(wm, m - se, m + se, color=c, alpha=0.22, lw=0)
        ax.axhline(ref, color='#b2182b', ls='--', lw=1.4, label=ref_label)
        ax.axvspan(*baseline, color='0.55', alpha=0.18, lw=0)
        ax.axvline(0, color='k', lw=0.8, alpha=0.6)
        sig = pv < 0.05
        n_sig, n_task = int(np.nansum(sig)), int(task_mask.sum())
        y0, y1 = ax.get_ylim()
        for a, b in contiguous_spans(sig):
            ax.plot([wm[a], wm[b]], [y0 + 0.02 * (y1 - y0)] * 2,
                    color='#b2182b', lw=4, solid_capstyle='butt')
        ax.set_title(f'{title} — {n_sig}/{n_task} task windows p<0.05 '
                     '(uncorrected, right-tailed)', fontsize=11, loc='left')
        ax.set_ylabel('GC' if A is X else 'GC − own baseline mean')
        ax.legend(fontsize=9, loc='upper right', frameon=False)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', color='0.9', lw=0.6)
        ax.set_axisbelow(True)
    axes[-1].set_xlabel('window start (ms)')

    pair_lbl = f"{roi[i].replace('-lh', '')}–{roi[j].replace('-lh', '')}"
    fig.suptitle(
        f'{head}   {BAND_LABEL[args.band]}   {args.task}/{args.stim_class}   '
        f'order {args.order} / {args.win_ms:g} ms @ {args.target_fs:g} Hz / '
        f'{args.normalize}   n={n_subj}\n'
        'mean ± SEM · same one-sample t per window in both panels; only the '
        'baseline referencing differs · red bars: p<0.05 uncorrected',
        fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out_dir = GC_OUTPUT_ROOT / '_figures_final_review'
    os.makedirs(out_dir, exist_ok=True)
    tag = gc_tag(args.order, args.win_ms, args.target_fs, args.normalize)
    out = os.path.join(
        out_dir, f'pointwise_ref_{args.task}_{args.stim_class}_'
                 f"{pair_lbl.replace('–', '+')}_{args.direction}_"
                 f'{args.band}_{tag}.{args.format}')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
