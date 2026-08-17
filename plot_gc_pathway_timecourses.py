#!/usr/bin/env python
"""Directed GC per pathway — the theta-sized config against the sweep's grid.

Rebuilds the GC_qc/theta_sweep_pmc/pathway_pmc_timecourses.png comparison on
the current data. That figure's generator was never committed, so this is a
reimplementation of its design, not a port: one row per ROI pair, forward
(solid) against reverse (dashed) with SEM bands, columns comparing configs.

    rows     every ROI pair present in the data
    columns  high beta @ the sweep config  |  theta @ the sweep config  |
             theta @ the theta-sized config (MO25/SW200)

The point of the original was that theta needs more VAR memory than the short
config gives it: order 10 at 200 Hz spans 50 ms, which is 0.40 of a theta cycle,
while order 25 spans 125 ms — a full cycle at 8 Hz. If that holds here the third
column should be visibly smoother and better separated than the second.

Two figures per scope, one per task/contrast, for pairwise and triple-wise.

    conda activate mne
    python plot_gc_pathway_timecourses.py
    python plot_gc_pathway_timecourses.py --ref-order 6 --ref-win 60
"""
import os
import sys
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_granger import GC_OUTPUT_ROOT
from granger_stats import load_gc_group

D = str(GC_OUTPUT_ROOT)
OUT = f'{D}/_figures_pathway'
FWD, REV = '#c0392b', '#2471a3'
CONDS = [('overtProd', 'prodDiff'), ('perception', 'percDiff')]


def sh(r):
    return str(r).replace('-lh', '')


def load_cfg(task, stim, scope, win_ms, order):
    """-> (window_ms, {(src,tgt): (n_subj, n_win)}) for one config, or (None, {})."""
    n_roi = 2 if scope == 'biv' else 3
    suffix = '_zscore' if scope == 'biv' else '_zscore_conditional'
    cfg = f'order{order}_win{win_ms:g}ms_fs200{suffix}'
    out, w_ref = {}, None
    for d in sorted(glob.glob(f'{D}/{task}/*/*/*/*/{cfg}/rois_*/{stim}')):
        if d.split('/')[-2].count('-lh') != n_roi or not glob.glob(f'{d}/*.npz'):
            continue
        agg = load_gc_group(d)
        w = np.asarray(agg['window_ms'], float)
        if w_ref is None:
            w_ref = w
        elif w.shape != w_ref.shape:
            continue
        roi = [str(r) for r in agg['roi_names']]
        for k, (i, j) in enumerate(zip(agg['pair_i'], agg['pair_j'])):
            for key, e in (('fxy', (roi[i], roi[j])), ('fyx', (roi[j], roi[i]))):
                out[e] = {b: agg[key][b][:, k, :] for b in agg['fxy']}
    return w_ref, out


def panel(ax, w, fwd, rev, lab_f, lab_r, band):
    for stack, col, lab, ls in ((fwd, FWD, lab_f, '-'), (rev, REV, lab_r, '--')):
        if stack is None:
            continue
        m = np.nanmean(stack[band], axis=0)
        e = np.nanstd(stack[band], axis=0) / np.sqrt(stack[band].shape[0])
        ax.plot(w, m, ls, color=col, lw=1.5, label=lab)
        ax.fill_between(w, m - e, m + e, color=col, alpha=0.18, lw=0)
    ax.axvline(0, color='0.35', ls=':', lw=1.1)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7.5, frameon=False, loc='upper left')


def build(task, stim, scope, cfgs, out_png):
    loaded = []
    for lab, win, order, band in cfgs:
        w, d = load_cfg(task, stim, scope, win, order)
        loaded.append((lab, band, w, d))
    have = [x for x in loaded if x[2] is not None and x[3]]
    if not have:
        print(f'  {task}/{stim} {scope}: no data'); return None
    missing = [x[0] for x in loaded if x[2] is None or not x[3]]

    # pairs present in the reference config
    ref = have[0][3]
    pairs, seen = [], set()
    for (s, t) in ref:
        if (t, s) not in seen:
            pairs.append((s, t)); seen.add((s, t))
    pairs.sort()

    n_r, n_c = len(pairs), len(loaded)
    fig, axes = plt.subplots(n_r, n_c, figsize=(5.2 * n_c, 2.5 * n_r),
                             squeeze=False)
    n_subj = None
    for c, (lab, band, w, d) in enumerate(loaded):
        for r, (s, t) in enumerate(pairs):
            ax = axes[r][c]
            if w is None or not d or (s, t) not in d:
                ax.text(0.5, 0.5, 'not computed', ha='center', va='center',
                        transform=ax.transAxes, fontsize=10, color='0.55')
                ax.set_xticks([]); ax.set_yticks([])
                continue
            panel(ax, w, d[(s, t)], d.get((t, s)),
                  f'{sh(s)}→{sh(t)}', f'{sh(t)}→{sh(s)}', band)
            n_subj = d[(s, t)][band].shape[0]
            if c == 0:
                ax.set_ylabel(f'{sh(s)}↔{sh(t)}\nGC', fontsize=9)
            if r == 0:
                ax.set_title(lab, fontsize=10.5)
            if r == n_r - 1:
                ax.set_xlabel('window start (ms, 0 = onset)', fontsize=9)

    scope_lab = ('bivariate' if scope == 'biv'
                 else 'triple-wise, A→B|C conditioned on the third ROI')
    note = ''
    if missing:
        note = ('\nnot computed yet: ' + ', '.join(missing)
                + ' — run run_gc_theta_config.sh')
    fig.suptitle(
        f'Directed GC per pathway — {task}/{stim},  {scope_lab}\n'
        f'zscore, n={n_subj}; solid = forward, dashed = reverse, shading = SEM '
        f'over subjects{note}', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.965 - 0.004 * n_r])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref-win', type=float, default=60.0)
    ap.add_argument('--ref-order', type=int, default=6)
    ap.add_argument('--theta-win', type=float, default=200.0)
    ap.add_argument('--theta-order', type=int, default=25)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    cfgs = [
        (f'high beta · sweep config (MO{args.ref_order}/SW{args.ref_win:g})',
         args.ref_win, args.ref_order, 'high_beta'),
        (f'theta · sweep config (MO{args.ref_order}/SW{args.ref_win:g})',
         args.ref_win, args.ref_order, 'theta'),
        (f'theta · theta-sized (MO{args.theta_order}/SW{args.theta_win:g})',
         args.theta_win, args.theta_order, 'theta'),
    ]
    print(f'root: {D}')
    for scope, tag in (('biv', 'pairwise'), ('cond', 'triplewise')):
        for task, stim in CONDS:
            p = f'{OUT}/pathway_{tag}_{task}_{stim}.png'
            print(f'{tag} · {task}/{stim}')
            r = build(task, stim, scope, cfgs, p)
            if r:
                print(f'  wrote {os.path.basename(r)}')
    print(f'\nfigures in {OUT}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
