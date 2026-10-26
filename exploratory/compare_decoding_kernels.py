#!/usr/bin/env python
"""Are two inverse kernels giving different decoding results?

Overlays the group-mean sliding-window accuracy of several ``--method``
arms (same task / contrast / atlas / feature mode / classifier) ROI by ROI,
and quantifies the differences with paired per-subject statistics, so the
question "is it worth running kernel B at all" is answered from the result
files rather than by eye.

Per ROI and kernel pair (reference = first method):
    r(time)        correlation of the two group curves over time
    max |dgroup|   largest absolute difference of the group curves
    peak A / B     peak group accuracy and its time
    mean |dsubj|   per-subject mean absolute difference over time, averaged
    d at peak      paired difference at the reference peak, mean +- SEM, and
                   a paired t-test p (n subjects)
    subj-noise     SEM of the group curve (the yardstick a difference has to
                   clear to matter)

Outputs (DECODE_source_space/{task}/_kernel_comparison/):
    kernels_{atlas}_{stim}_{A}-vs-{B,...}.png / .csv

Usage (local, result files only):
    python exploratory/compare_decoding_kernels.py --task overtProd --stim-class prodDiff \
        --methods dSPM eLORETA sLORETA
"""
import os
import sys
import re
import glob
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import DECODE_OUTPUT_ROOT, classifier_path_segment, method_arg, feature_mode_arg


def arm_dir(task, method, atlas, feat, leak, pseudo, sw, clf, stim):
    return (Path(DECODE_OUTPUT_ROOT) / task / method / atlas / feat
            / ('leakage_corrected' if leak else 'raw')
            / (f'pseudo_{pseudo}' if pseudo else 'no_pseudo') / sw / clf / stim)


def load_arm(d):
    out = {}
    for f in sorted(glob.glob(str(d / 'EEGPROD4*.csv'))):
        s = re.search(r'(EEGPROD\d+)', os.path.basename(f)).group(1)
        t = pd.read_csv(f, usecols=['key', 'ms', 'decode_acc'])
        out[s] = t.pivot_table(index='ms', columns='key', values='decode_acc')
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--task', required=True, choices=['overtProd', 'perception'])
    ap.add_argument('--stim-class', required=True, choices=['prodDiff', 'percDiff'])
    ap.add_argument('--methods', nargs='+', type=method_arg, default=['dSPM', 'eLORETA', 'sLORETA'],
                    help='first = reference')
    ap.add_argument('--atlas', default='sensorEquiv')
    ap.add_argument('--feature-mode', default='vertex_selectkbest', type=feature_mode_arg)
    ap.add_argument('--leakage-correction', action='store_true')
    ap.add_argument('--pseudo-trial-size', type=int, default=0)
    ap.add_argument('--sw', default='40_5')
    ap.add_argument('--classifier', default='logistic')
    ap.add_argument('--c', type=float, default=1.0)
    ap.add_argument('--tune-hyperparams', action='store_true', default=True)
    ap.add_argument('--out-dir', default=None)
    args = ap.parse_args()

    clf = classifier_path_segment(args.classifier, args.c, args.tune_hyperparams)
    arms = {}
    for m in args.methods:
        d = arm_dir(args.task, m, args.atlas, args.feature_mode, args.leakage_correction,
                    args.pseudo_trial_size, args.sw, clf, args.stim_class)
        a = load_arm(d)
        if not a:
            print(f'  {m}: no result CSVs in {d} — skipped')
            continue
        arms[m] = a
    if len(arms) < 2:
        raise SystemExit('need at least two arms with results')
    methods = list(arms)
    ref = methods[0]
    common = sorted(set.intersection(*(set(a) for a in arms.values())))
    rois = [r for r in arms[ref][common[0]].columns
            if all(r in arms[m][common[0]].columns for m in methods)]
    t = arms[ref][common[0]].index
    for m in methods:
        for s in common:
            arms[m][s] = arms[m][s].reindex(t)
    print(f'{args.task}/{args.stim_class} {args.atlas} {args.feature_mode} {clf}: '
          f'{len(common)} common subjects, {len(rois)} ROIs, {len(t)} windows')

    # group curves and per-subject stacks: X[m][roi] -> (n_subj, n_t)
    X = {m: {r: np.stack([arms[m][s][r].values for s in common]) for r in rois} for m in methods}
    rows = []
    for r in rois:
        ga = X[ref][r].mean(0)
        sem_ref = X[ref][r].std(0, ddof=1) / np.sqrt(len(common))
        ipk = int(np.nanargmax(ga))
        for m in methods[1:]:
            gb = X[m][r].mean(0)
            dsub = X[m][r] - X[ref][r]
            dpk = dsub[:, ipk]
            tt = stats.ttest_1samp(dpk, 0.0)
            rows.append(dict(
                roi=r, ref=ref, other=m,
                r_time=float(np.corrcoef(ga, gb)[0, 1]),
                max_abs_dgroup=float(np.abs(gb - ga).max()),
                peak_ref=float(ga.max()), t_peak_ref=float(t[ipk]),
                peak_other=float(gb.max()), t_peak_other=float(t[int(np.nanargmax(gb))]),
                mean_abs_dsubj=float(np.abs(dsub).mean()),
                d_at_peak=float(dpk.mean()), d_at_peak_sem=float(dpk.std(ddof=1) / np.sqrt(len(dpk))),
                p_at_peak=float(tt.pvalue), sem_group_at_peak=float(sem_ref[ipk]),
                n=len(common)))
    tab = pd.DataFrame(rows)
    pd.set_option('display.width', 220)
    cols = ['roi', 'other', 'r_time', 'max_abs_dgroup', 'peak_ref', 'peak_other',
            't_peak_ref', 't_peak_other', 'mean_abs_dsubj', 'd_at_peak', 'd_at_peak_sem',
            'p_at_peak', 'sem_group_at_peak']
    print(tab[cols].round(3).to_string(index=False))

    out = Path(args.out_dir) if args.out_dir else Path(DECODE_OUTPUT_ROOT) / args.task / '_kernel_comparison'
    out.mkdir(parents=True, exist_ok=True)
    stem = f'kernels_{args.atlas}_{args.stim_class}_{ref}-vs-{"-".join(methods[1:])}'
    tab.to_csv(out / f'{stem}.csv', index=False)

    # figure: top row curves, bottom row differences vs reference
    n = len(rois)
    fig, axes = plt.subplots(2, n, figsize=(3.6 * n, 6.2), sharex=True, layout='constrained')
    axes = np.atleast_2d(axes)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for k, r in enumerate(rois):
        ax = axes[0, k]
        for i, m in enumerate(methods):
            g = X[m][r].mean(0); se = X[m][r].std(0, ddof=1) / np.sqrt(len(common))
            ax.plot(t, g, color=colors[i], lw=1.4, label=m)
            if i == 0:
                ax.fill_between(t, g - se, g + se, color=colors[i], alpha=0.15, lw=0)
        ax.axhline(0.5, color='k', lw=0.6, ls=':')
        ax.axvline(0, color='k', lw=0.6, ls='--')
        ax.set_title(r.replace('_', ' '), fontsize=10)
        if k == 0:
            ax.set_ylabel('accuracy (group mean; band = SEM of ref)')
            ax.legend(fontsize=8, frameon=False)
        ax = axes[1, k]
        for i, m in enumerate(methods[1:], start=1):
            d = X[m][r] - X[ref][r]
            g = d.mean(0); se = d.std(0, ddof=1) / np.sqrt(len(common))
            ax.plot(t, g, color=colors[i], lw=1.2, label=f'{m} − {ref}')
            ax.fill_between(t, g - se, g + se, color=colors[i], alpha=0.2, lw=0)
        ax.axhline(0, color='k', lw=0.6)
        ax.axvline(0, color='k', lw=0.6, ls='--')
        ax.set_xlabel('ms')
        if k == 0:
            ax.set_ylabel('paired difference (mean ± SEM)')
            ax.legend(fontsize=8, frameon=False)
    fig.suptitle(f'{args.task}/{args.stim_class} · {args.atlas} · {args.feature_mode} · {clf} · n = {len(common)}\n'
                 f'inverse kernels compared on identical subjects, windows and classifier', fontsize=11)
    fig.savefig(out / f'{stem}.png', dpi=150)
    print(f'-> {out / stem}.png / .csv')


if __name__ == '__main__':
    main()
