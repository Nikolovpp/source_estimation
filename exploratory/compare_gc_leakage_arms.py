#!/usr/bin/env python3
"""
Raw vs leakage-corrected source GC: does the leakage arm change TRGC?

Loads the two arms of ONE GC configuration (same task, contrast, method,
atlas, order/window/fs tag) from ``GC_source_space/.../raw/<tag>/<roiset>``
and ``.../leakage_corrected/<tag>/<roiset>`` and compares them edge by edge:

  * group-mean time courses (change from each arm's own baseline) overlaid,
    raw vs leakage, for dTRGC and for the two directed GCs;
  * similarity of the arms: Pearson r between the subject-mean curves over
    the task span, and the mean within-subject r;
  * the a-priori fixed windows (config.GC_WINDOWS_OF_INTEREST): per-subject
    window mean of the baseline-referenced measure in each arm, one-sample
    t vs 0 per arm (two-tailed) and a paired t raw-vs-leakage;
  * pointwise task-vs-baseline significance share per arm (granger_stats.
    task_vs_baseline, same tails as the production stats) and its overlap.

TRGC already removes zero-lag mixing, so the leakage-corrected arm is a
double correction (Pellegrini et al. 2023 recommend one).  If the arms agree
(high r, same window signs) the choice is cosmetic; if they disagree, the
raw+TRGC arm is the principled one and the leakage arm the sensitivity check.

No GC is estimated here — it reads existing result npz only.

Usage
-----
    conda activate mne
    # the n=20 PC1 / normalize-none arms that exist for both raw and leakage
    python exploratory/compare_gc_leakage_arms.py --tag order10_win40ms_fs500
    python exploratory/compare_gc_leakage_arms.py --tag order10_win40ms_fs500 \\
        --tasks overtProd --stims prodDiff --bands low_beta high_beta

Output
------
    GC_source_space/_figures_final_review/leakage_arms_{tag}/
        {task}_{stim}_dtrgc.png / _gc.png   overlays, pairs x bands
        arm_similarity_{task}_{stim}.png    r(raw, leak) heat table
        leakage_arms_{tag}.csv              every number, long format
        summary.txt
"""
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import GC_WINDOWS_OF_INTEREST
from run_granger import GC_OUTPUT_ROOT
from granger_stats import load_gc_group, task_vs_baseline

ARM_COLORS = {'raw': '#2a78d6', 'leakage_corrected': '#eb6834'}
ARM_LABEL = {'raw': 'raw + TRGC', 'leakage_corrected': 'leakage-corrected + TRGC'}
DEFAULT_PAIRS = ['awfa-lh,ifc-lh', 'awfa-lh,pmc-lh', 'awfa-lh,tpc-lh',
                 'ifc-lh,pmc-lh', 'ifc-lh,tpc-lh', 'pmc-lh,tpc-lh']
# _bl baselines of the final analysis (window START times, ms)
DEFAULT_BASELINE = {'overtProd': (-1470.0, -1370.0), 'perception': (-80.0, 0.0)}


def short(name):
    return str(name).replace('-lh', '')


def edges_for(group, pairs):
    """[(row_index, src, tgt, measure)] for the requested undirected pairs."""
    names = [str(n) for n in group['roi_names']]
    want = [tuple(sorted(p.split(','))) for p in pairs]
    out = []
    for k, (i, j) in enumerate(zip(group['pair_i'], group['pair_j'])):
        a, b = names[int(i)], names[int(j)]
        if tuple(sorted((a, b))) in want:
            out.append((k, a, b))
    return out


def analyse_cell(task, stim, arms, bands, pairs, edge_guard_ms, tail_guard_ms,
                 baseline_ms, alpha):
    """Return (records, curves) for one task x contrast."""
    g = {arm: load_gc_group(d, bands) for arm, d in arms.items()}
    w_raw, w_leak = g['raw']['window_ms'], g['leakage_corrected']['window_ms']
    if not np.allclose(w_raw, w_leak):
        raise ValueError(f'{task}/{stim}: window axes differ between arms')
    window_ms = np.asarray(w_raw, float)
    subj = {arm: list(g[arm]['subjects']) for arm in g}
    common = sorted(set(subj['raw']) & set(subj['leakage_corrected']))
    sel = {arm: [subj[arm].index(s) for s in common] for arm in g}
    task_start = baseline_ms[1]
    task_end = window_ms[-1] - tail_guard_ms
    win_mask = (window_ms >= task_start) & (window_ms <= task_end)
    win_mask &= window_ms >= window_ms[0] + edge_guard_ms
    base_mask = (window_ms >= baseline_ms[0]) & (window_ms <= baseline_ms[1])
    windows = GC_WINDOWS_OF_INTEREST[task]

    edges = edges_for(g['raw'], pairs)
    if [e[1:] for e in edges] != [e[1:] for e in edges_for(g['leakage_corrected'], pairs)]:
        raise ValueError(f'{task}/{stim}: pair sets differ between arms')

    records, curves = [], {}
    for band in bands:
        for measure in ('dtrgc', 'fxy', 'fyx'):
            if measure == 'dtrgc' and 'dtrgc' not in g['raw']:
                continue
            tail = 'two' if measure == 'dtrgc' else 'right'
            for k, a, b in edges:
                src, tgt = (a, b) if measure != 'fyx' else (b, a)
                X = {}
                for arm in g:
                    x = g[arm][measure][band][sel[arm], k, :]       # (n_subj, n_win)
                    bl = np.nanmean(x[:, base_mask], axis=1, keepdims=True)
                    X[arm] = x - bl                                   # own-baseline referenced
                mean = {arm: np.nanmean(X[arm], axis=0) for arm in X}
                sem = {arm: np.nanstd(X[arm], axis=0, ddof=1) / np.sqrt(len(common))
                       for arm in X}
                r_group = float(np.corrcoef(mean['raw'][win_mask],
                                            mean['leakage_corrected'][win_mask])[0, 1])
                r_subj = []
                for si in range(len(common)):
                    xa, xb = X['raw'][si, win_mask], X['leakage_corrected'][si, win_mask]
                    ok = np.isfinite(xa) & np.isfinite(xb)
                    if ok.sum() > 10:
                        r_subj.append(np.corrcoef(xa[ok], xb[ok])[0, 1])
                # pointwise significance share per arm (production test)
                sig = {}
                for arm in g:
                    stack = g[arm][measure][band][sel[arm]][:, [k], :]
                    tv = task_vs_baseline(stack, window_ms, baseline_ms, task_start,
                                          alpha=alpha, test='ttest',
                                          task_end_ms=task_end, tail=tail)
                    sig[arm] = tv['sig'][0] & win_mask
                n_test = int(win_mask.sum())
                rec = dict(task=task, stim=stim, band=band, measure=measure,
                           src=short(src), tgt=short(tgt), n_subj=len(common),
                           r_group=r_group, r_subj_mean=float(np.nanmean(r_subj)),
                           sig_share_raw=sig['raw'].sum() / n_test,
                           sig_share_leak=sig['leakage_corrected'].sum() / n_test,
                           sig_share_both=(sig['raw'] & sig['leakage_corrected']).sum() / n_test)
                for wname, (lo, hi) in windows.items():
                    m = (window_ms >= lo * 1000) & (window_ms <= hi * 1000) & win_mask
                    vals = {arm: np.nanmean(X[arm][:, m], axis=1) for arm in X}
                    for arm, key in (('raw', 'raw'), ('leakage_corrected', 'leak')):
                        v = vals[arm][np.isfinite(vals[arm])]
                        t, p = stats.ttest_1samp(v, 0.0) if v.size > 2 else (np.nan, np.nan)
                        rec[f'{wname}_mean_{key}'] = float(np.mean(v)) if v.size else np.nan
                        rec[f'{wname}_p_{key}'] = float(p)
                    d = vals['raw'] - vals['leakage_corrected']
                    d = d[np.isfinite(d)]
                    t, p = stats.ttest_1samp(d, 0.0) if d.size > 2 else (np.nan, np.nan)
                    rec[f'{wname}_p_paired'] = float(p)
                    rec[f'{wname}_same_sign'] = bool(
                        np.sign(rec[f'{wname}_mean_raw']) == np.sign(rec[f'{wname}_mean_leak']))
                records.append(rec)
                curves[(band, measure, k)] = dict(mean=mean, sem=sem, sig=sig,
                                                  src=short(src), tgt=short(tgt))
    return records, curves, window_ms, win_mask, windows, edges


def plot_overlays(out_dir, task, stim, curves, window_ms, win_mask, windows,
                  edges, bands, measure_set, fname):
    """Grid: rows = edges, cols = bands; raw vs leak overlays."""
    keys = [(k, a, b) for k, a, b in edges]
    if measure_set == 'dtrgc':
        rows = [('dtrgc', k, a, b) for k, a, b in keys]
    else:
        rows = [(m, k, a, b) for k, a, b in keys for m in ('fxy', 'fyx')]
    rows = [r for r in rows if (bands[0], r[0], r[1]) in curves]
    if not rows:
        return
    fig, axes = plt.subplots(len(rows), len(bands),
                             figsize=(3.4 * len(bands), 1.9 * len(rows) + 0.8),
                             sharex=True, squeeze=False)
    t = window_ms[win_mask]
    for ri, (m, k, a, b) in enumerate(rows):
        for bi, band in enumerate(bands):
            ax = axes[ri, bi]
            c = curves[(band, m, k)]
            for arm in ('raw', 'leakage_corrected'):
                mu, se = c['mean'][arm][win_mask], c['sem'][arm][win_mask]
                ax.fill_between(t, mu - se, mu + se, color=ARM_COLORS[arm], alpha=0.18, lw=0)
                ax.plot(t, mu, color=ARM_COLORS[arm], lw=1.4, label=ARM_LABEL[arm])
            ax.axhline(0, color='#c3c2b7', lw=0.8)
            for lo, hi in windows.values():
                ax.axvspan(lo * 1000, hi * 1000, color='#eeeeee', zorder=0)
            # pointwise significance above the traces: raw row, leak row
            ymax = max(np.nanmax(c['mean'][arm][win_mask] + c['sem'][arm][win_mask])
                       for arm in ARM_COLORS)
            ymin = min(np.nanmin(c['mean'][arm][win_mask] - c['sem'][arm][win_mask])
                       for arm in ARM_COLORS)
            span = max(ymax - ymin, 1e-6)
            for li, arm in enumerate(('raw', 'leakage_corrected')):
                s = c['sig'][arm][win_mask]
                if s.any():
                    ax.plot(t[s], np.full(s.sum(), ymax + (0.08 + 0.1 * li) * span),
                            ls='none', marker='|', ms=4, color=ARM_COLORS[arm])
            ax.set_ylim(ymin - 0.05 * span, ymax + 0.3 * span)
            arrow = '<>' if m == 'dtrgc' else '->'
            if bi == 0:
                ax.set_ylabel(f'{c["src"]}{arrow}{c["tgt"]}', fontsize=8)
            if ri == 0:
                ax.set_title(band, fontsize=10)
            ax.tick_params(labelsize=7)
            ax.spines[['top', 'right']].set_visible(False)
    axes[-1, 0].set_xlabel('window start (ms)', fontsize=8)
    axes[0, 0].legend(fontsize=7, frameon=False, loc='upper left')
    what = 'net TRGC (change from baseline)' if measure_set == 'dtrgc' else \
        'directed GC (change from baseline)'
    fig.suptitle(f'{task} / {stim} — {what}; ticks above = pointwise p<.05 '
                 f'(upper row leakage arm); grey = windows of interest', fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=160)
    plt.close(fig)


def plot_similarity(out_dir, df, task, stim, bands):
    d = df[(df.task == task) & (df.stim == stim)]
    fig, axes = plt.subplots(1, 3, figsize=(11, 0.35 * d.src.nunique() * 2 + 2.2))
    for ax, m, title in zip(axes, ('dtrgc', 'fxy', 'fyx'),
                            ('net TRGC', 'GC src->tgt', 'GC tgt->src')):
        dm = d[d.measure == m]
        if dm.empty:
            ax.set_axis_off()
            continue
        labels = list(dict.fromkeys(dm.src + '>' + dm.tgt))
        M = np.full((len(labels), len(bands)), np.nan)
        for _, r in dm.iterrows():
            M[labels.index(r.src + '>' + r.tgt), bands.index(r.band)] = r.r_group
        im = ax.imshow(M, vmin=-1, vmax=1, cmap='RdBu_r')
        ax.set_xticks(range(len(bands)))
        ax.set_xticklabels(bands, fontsize=8, rotation=30, ha='right')
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(f'r(raw, leak) — {title}', fontsize=10)
        for i in range(len(labels)):
            for j in range(len(bands)):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center',
                            fontsize=7, color='white' if abs(M[i, j]) > 0.6 else '#0b0b0b')
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, pad=0.02)
    fig.suptitle(f'{task} / {stim}: similarity of the subject-mean curves between arms',
                 fontsize=10)
    fig.savefig(out_dir / f'arm_similarity_{task}_{stim}.png', dpi=160,
                bbox_inches='tight')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tag', default='order10_win40ms_fs500',
                   help='GC config directory tag (run_granger.gc_tag)')
    p.add_argument('--roiset', default='all_rois')
    p.add_argument('--method', default='LCMV')
    p.add_argument('--atlas', default='custom')
    p.add_argument('--feature-mode', default='vertex_selectkbest')
    p.add_argument('--tasks', nargs='+', default=['overtProd', 'perception'])
    p.add_argument('--stims', nargs='+', default=['prodDiff', 'percDiff'])
    p.add_argument('--bands', nargs='+',
                   default=['theta', 'alpha', 'low_beta', 'high_beta'])
    p.add_argument('--pairs', nargs='+', default=DEFAULT_PAIRS, metavar='ROI,ROI')
    p.add_argument('--edge-guard', type=float, default=30.0)
    p.add_argument('--tail-guard', type=float, default=50.0)
    p.add_argument('--baseline-overtprod', type=float, nargs=2, default=None,
                   help='window-start baseline in s (default -1.47 -1.37)')
    p.add_argument('--baseline-perception', type=float, nargs=2, default=None,
                   help='window-start baseline in s (default -0.08 0)')
    p.add_argument('--alpha', type=float, default=0.05)
    p.add_argument('--out-dir', default=None)
    args = p.parse_args()

    base = GC_OUTPUT_ROOT
    out_dir = Path(args.out_dir) if args.out_dir else \
        base / '_figures_final_review' / f'leakage_arms_{args.tag}'
    out_dir.mkdir(parents=True, exist_ok=True)
    bl = dict(DEFAULT_BASELINE)
    if args.baseline_overtprod:
        bl['overtProd'] = tuple(1000 * v for v in args.baseline_overtprod)
    if args.baseline_perception:
        bl['perception'] = tuple(1000 * v for v in args.baseline_perception)

    all_rec = []
    for task in args.tasks:
        for stim in args.stims:
            arms = {arm: base / task / args.method / args.atlas / args.feature_mode
                    / arm / args.tag / args.roiset / stim
                    for arm in ('raw', 'leakage_corrected')}
            missing = [a for a, d in arms.items() if not d.is_dir()]
            if missing:
                print(f'{task}/{stim}: missing arm(s) {missing} — skipped '
                      f'({arms[missing[0]]})')
                continue
            print(f'{task}/{stim}: comparing arms under {args.tag}')
            rec, curves, window_ms, win_mask, windows, edges = analyse_cell(
                task, stim, arms, args.bands, args.pairs, args.edge_guard,
                args.tail_guard, bl[task], args.alpha)
            all_rec += rec
            plot_overlays(out_dir, task, stim, curves, window_ms, win_mask,
                          windows, edges, args.bands, 'dtrgc',
                          f'{task}_{stim}_dtrgc.png')
            plot_overlays(out_dir, task, stim, curves, window_ms, win_mask,
                          windows, edges, args.bands, 'gc',
                          f'{task}_{stim}_gc.png')
    if not all_rec:
        raise SystemExit('nothing compared')
    df = pd.DataFrame(all_rec)
    df.to_csv(out_dir / f'leakage_arms_{args.tag}.csv', index=False)
    for task in args.tasks:
        for stim in args.stims:
            if ((df.task == task) & (df.stim == stim)).any():
                plot_similarity(out_dir, df, task, stim, args.bands)

    lines = [f'raw vs leakage-corrected arms — {args.method}/{args.atlas}/{args.tag}/'
             f'{args.roiset}, n = {int(df.n_subj.min())}-{int(df.n_subj.max())}', '']
    for m, title in (('dtrgc', 'net TRGC'), ('fxy', 'GC src->tgt'), ('fyx', 'GC tgt->src')):
        dm = df[df.measure == m]
        if dm.empty:
            continue
        lines.append(f'{title}: r(raw, leak) of subject-mean curves — median '
                     f'{dm.r_group.median():.2f}, IQR {dm.r_group.quantile(.25):.2f}..'
                     f'{dm.r_group.quantile(.75):.2f}; within-subject mean r '
                     f'{dm.r_subj_mean.mean():.2f}')
        lines.append(f'    pointwise p<.05 share: raw {100 * dm.sig_share_raw.mean():.1f}%, '
                     f'leak {100 * dm.sig_share_leak.mean():.1f}%, both '
                     f'{100 * dm.sig_share_both.mean():.1f}% (chance 5%)')
    lines.append('')
    lines.append('Fixed windows (window mean of the baseline-referenced measure):')
    for task in args.tasks:
        for wname in GC_WINDOWS_OF_INTEREST[task]:
            d = df[df.task == task]
            if d.empty:
                continue
            pr, pl, pp = (d[f'{wname}_p_raw'], d[f'{wname}_p_leak'],
                          d[f'{wname}_p_paired'])
            lines.append(f'  {task} {wname}: cells p<.05 vs 0 — raw {int((pr < .05).sum())}'
                         f'/{len(d)}, leak {int((pl < .05).sum())}/{len(d)}; '
                         f'raw-vs-leak paired p<.05 in {int((pp < .05).sum())}/{len(d)}; '
                         f'same sign {100 * d[f"{wname}_same_sign"].mean():.0f}%')
            top = d[(d.measure == 'dtrgc')].copy()
            if not top.empty:
                top['pmin'] = top[[f'{wname}_p_raw', f'{wname}_p_leak']].min(axis=1)
                for _, r in top.nsmallest(4, 'pmin').iterrows():
                    lines.append(f'      {r.src}<>{r.tgt} {r.band}: raw {r[f"{wname}_mean_raw"]:+.4f} '
                                 f'(p {r[f"{wname}_p_raw"]:.3f}), leak {r[f"{wname}_mean_leak"]:+.4f} '
                                 f'(p {r[f"{wname}_p_leak"]:.3f}), paired p {r[f"{wname}_p_paired"]:.3f}')
    text = '\n'.join(lines)
    print('\n' + text)
    (out_dir / 'summary.txt').write_text(text + '\n')
    print(f'\nWrote {out_dir}')


if __name__ == '__main__':
    main()
