#!/usr/bin/env python
"""Forward-and-reverse GC time courses with baseline-referenced statistics.

This is the interpretable view: for one ROI pair, both directions on the same
axes, so the asymmetry (A->B vs B->A) can be read directly rather than
reconstructed from two separate figures.

Writes to THREE directories under GC_source_space, kept apart from the sweep's
own robustness figures:

    _figures_pairwise/     bivariate GC, one figure per pair
    _figures_triplewise/   conditional GC A->B|C, one figure per pair-in-triple
    _figures_edgeguard/    the same pair with and without the edge guard

STATISTICS. Both families from granger_stats, per direction and band:
  * pointwise right-tailed test of each task window against the group baseline
    scalar (uncorrected — plotted as a thin marker row, not as a claim);
  * sign-flip cluster permutation over the task span, each subject referenced
    to its OWN baseline (FWER-controlled across windows — the bars).
Only the cluster result is drawn as a solid significance bar.

EDGE GUARD. ``--edge-guard`` ms of epoch onset are excluded before the
baseline starts. Measured on the pre-bug files: at a 60 ms window the leading
4-6 windows sit at 46-68% of the interior plateau, so including them biases
the baseline LOW, and the test is right-tailed "task > baseline". The A/B
figures show what that does here rather than asserting it.

    conda activate mne
    python plot_gc_pairs.py                      # bivariate + triple, guard 30
    python plot_gc_pairs.py --edge-guard-ab      # add the guard 0 vs 30 figures
    python plot_gc_pairs.py --win-ms 80 --order 6
"""
import os
import re
import sys
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import GC_TASK_END
from granger import DEFAULT_BANDS
from run_granger import GC_OUTPUT_ROOT
from granger_stats import (load_gc_group, task_vs_baseline, TASK_ONSET_MS,
                           permutation_task_vs_baseline, _contiguous_spans,
                           bh_fdr)

D = str(GC_OUTPUT_ROOT)
BANDS = ['theta', 'low_beta', 'high_beta']
LABEL = {'theta': 'theta 4–8 Hz', 'alpha': 'alpha 8–12 Hz',
         'low_beta': 'low beta 12–18 Hz', 'high_beta': 'high beta 18–30 Hz'}
FWD, REV = '#c0392b', '#2471a3'
CONDS = [('overtProd', 'prodDiff'), ('overtProd', 'percDiff'),
         ('perception', 'percDiff'), ('perception', 'prodDiff')]
BASELINE_DUR = 100.0


def sh(r):
    return str(r).replace('-lh', '')


def analyse(gc_dir, task, edge_guard, n_perm, tfce, seed=42,
            baseline_dur=BASELINE_DUR):
    """-> (agg, baseline_ms, task_start, task_end, {band: (stats, perm)})

    Baseline = the leading ``baseline_dur`` ms of the moving-window axis. The
    source epoch starts 100 ms later than the sensor one because LCMV consumes
    a 100 ms pre-stimulus segment for its covariance estimate, so the two can
    match in RULE but never in absolute time.

    Task start is the baseline end EXCEPT where t=0 is a stimulus onset
    (perception), where nothing before 0 is task — see
    granger_stats.TASK_ONSET_MS.
    """
    agg = load_gc_group(gc_dir)
    w = np.asarray(agg['window_ms'], float)
    baseline_ms = (float(w[0]) + edge_guard, float(w[0]) + baseline_dur)
    task_start = baseline_ms[1]
    onset = TASK_ONSET_MS.get(task)
    if onset is not None:
        task_start = max(task_start, onset)
    task_end = GC_TASK_END[task] * 1000.0 if task in GC_TASK_END else None
    out = {}
    for key in ('fxy', 'fyx'):
        for b in BANDS:
            st = task_vs_baseline(agg[key][b], w, baseline_ms, task_start,
                                  task_end_ms=task_end)
            pm = permutation_task_vs_baseline(
                agg[key][b], w, baseline_ms, task_start, task_end_ms=task_end,
                n_permutations=n_perm, tfce=tfce, seed=seed)
            out[(key, b)] = (st, pm)
    return agg, baseline_ms, task_start, task_end, out


def plot_pair(agg, res, baseline_ms, task_start, task_end, pair_idx,
              src, tgt, title, out_png, scope):
    """Forward and reverse on shared axes, one row per band."""
    w = np.asarray(agg['window_ms'], float)
    lo = baseline_ms[0]
    hi = task_end if task_end is not None else float(w[-1])
    keep = (w >= lo) & (w <= hi)

    fig, axes = plt.subplots(len(BANDS), 1, figsize=(9.2, 2.5 * len(BANDS)),
                             sharex=True, squeeze=False)
    for r, band in enumerate(BANDS):
        ax = axes[r][0]
        ax.axvspan(baseline_ms[0], baseline_ms[1], color='0.6', alpha=0.16,
                   lw=0, zorder=0)
        ax.axvline(task_start, color='0.4', ls=':', lw=1, zorder=0)
        ymax = 0.0
        for key, col, lab in (('fxy', FWD, f'{sh(src)} → {sh(tgt)}'),
                              ('fyx', REV, f'{sh(tgt)} → {sh(src)}')):
            st, pm = res[(key, band)]
            m = st['mean'][pair_idx]
            e = st['sem'][pair_idx]
            ax.plot(w[keep], m[keep], color=col, lw=1.7, label=lab)
            ax.fill_between(w[keep], (m - e)[keep], (m + e)[keep],
                            color=col, alpha=0.18, lw=0)
            ax.axhline(st['baseline_mean'][pair_idx], color=col, ls='--',
                       lw=0.9, alpha=0.65)
            if np.isfinite(m[keep]).any():
                ymax = max(ymax, np.nanmax((m + e)[keep]))
        # significance bars: cluster permutation only
        for k, (key, col) in enumerate((('fxy', FWD), ('fyx', REV))):
            _, pm = res[(key, band)]
            sig = pm['sig_cluster'][pair_idx]
            y = -0.055 * ymax * (k + 1)
            for a, b_ in _contiguous_spans(sig & keep):
                ax.plot([w[a], w[b_]], [y, y], color=col, lw=3.5,
                        solid_capstyle='butt')
        ax.set_ylim(-0.055 * ymax * 2.7, ymax * 1.12)
        ax.set_ylabel(f'{LABEL[band]}\nGC', fontsize=9)
        ax.tick_params(labelsize=8)
        if r == 0:
            ax.legend(fontsize=9, frameon=False, ncol=2, loc='upper right')
    axes[-1][0].set_xlabel('window start (ms)', fontsize=9)
    fig.suptitle(title, fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_pair_ttest(agg, res, baseline_ms, task_start, task_end, pair_idx,
                    src, tgt, title, out_png):
    """Same curves, but marked with the POINTWISE one-sample Student's t.

    Right-tailed, each task window against the group baseline scalar. This is
    the ``production_pwgc_data_to_python.m`` design and is UNCORRECTED across
    windows — at alpha=0.05 over M tested windows roughly 0.05*M cross by
    chance, so the count and that expectation are printed on every panel.
    Isolated single-window hits are drawn lighter than runs of three or more,
    because a sustained run is the part that is hard to get by chance.
    """
    w = np.asarray(agg['window_ms'], float)
    lo = baseline_ms[0]
    hi = task_end if task_end is not None else float(w[-1])
    keep = (w >= lo) & (w <= hi)

    fig, axes = plt.subplots(len(BANDS), 1, figsize=(9.2, 2.6 * len(BANDS)),
                             sharex=True, squeeze=False)
    for r, band in enumerate(BANDS):
        ax = axes[r][0]
        ax.axvspan(baseline_ms[0], baseline_ms[1], color='0.6', alpha=0.16,
                   lw=0, zorder=0)
        ax.axvline(task_start, color='0.4', ls=':', lw=1, zorder=0)
        ymax = 0.0
        note = []
        for key, col, lab in (('fxy', FWD, f'{sh(src)} → {sh(tgt)}'),
                              ('fyx', REV, f'{sh(tgt)} → {sh(src)}')):
            st, _ = res[(key, band)]
            m, e = st['mean'][pair_idx], st['sem'][pair_idx]
            ax.plot(w[keep], m[keep], color=col, lw=1.7, label=lab)
            ax.fill_between(w[keep], (m - e)[keep], (m + e)[keep],
                            color=col, alpha=0.18, lw=0)
            ax.axhline(st['baseline_mean'][pair_idx], color=col, ls='--',
                       lw=0.9, alpha=0.65)
            if np.isfinite(m[keep]).any():
                ymax = max(ymax, np.nanmax((m + e)[keep]))
            n_sig = int(st['sig'][pair_idx].sum())
            n_tested = int(np.isfinite(st['pval'][pair_idx]).sum())
            note.append(f'{lab}: {n_sig}/{n_tested}')
        for k, (key, col) in enumerate((('fxy', FWD), ('fyx', REV))):
            st, _ = res[(key, band)]
            sig = st['sig'][pair_idx] & keep
            y = -0.06 * ymax * (k + 1)
            # runs of >=3 windows solid, isolated hits faint
            for a, b_ in _contiguous_spans(sig):
                run = b_ - a + 1
                ax.plot([w[a], w[b_]], [y, y], color=col,
                        lw=3.5 if run >= 3 else 2.0,
                        alpha=1.0 if run >= 3 else 0.35,
                        solid_capstyle='butt')
        ax.set_ylim(-0.06 * ymax * 2.9, ymax * 1.14)
        ax.set_ylabel(f'{LABEL[band]}\nGC', fontsize=9)
        ax.tick_params(labelsize=8)
        n_tested = int(np.isfinite(res[('fxy', band)][0]['pval'][pair_idx]).sum())
        ax.text(0.005, 0.97, '  |  '.join(note)
                + f'  (chance ≈ {0.05 * n_tested:.0f})',
                transform=ax.transAxes, fontsize=7.5, va='top', color='0.25')
        if r == 0:
            ax.legend(fontsize=9, frameon=False, ncol=2, loc='upper right')
    axes[-1][0].set_xlabel('window start (ms)', fontsize=9)
    fig.suptitle(title, fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def configs(scope, win_ms, order):
    """Yield (task, stim, subset_dir, gc_dir) for the requested scope."""
    n_roi = 2 if scope == 'biv' else 3
    suffix = '_zscore' if scope == 'biv' else '_zscore_conditional'
    cfg = f'order{order}_win{win_ms:g}ms_fs200{suffix}'
    for d in sorted(glob.glob(f'{D}/*/*/*/*/*/{cfg}/rois_*/*')):
        p = d.split('GC_source_space/')[1].split('/')
        if p[6].count('-lh') != n_roi or not glob.glob(f'{d}/*.npz'):
            continue
        yield p[0], p[7], p[6], d


def run_scope(scope, args, outdir, tt_dir=None):
    """Two passes: test everything, correct across the family, THEN plot.

    A per-figure significance bar is uncorrected across the other edges, bands
    and conditions tested alongside it. With 6 pairs x 2 directions x 3 bands x
    4 conditions that family is 144 tests, where ~7 cross p<0.05 by chance —
    so an uncorrected bar drawn on its own would read as a finding at exactly
    the rate noise produces one. Every figure therefore carries the FDR-adjusted
    p for its own test and the family size it came from.
    """
    os.makedirs(outdir, exist_ok=True)
    if tt_dir is not None:
        os.makedirs(tt_dir, exist_ok=True)
    analysed, fam_keys, fam_p = {}, [], []
    for task, stim, subset, gc_dir in configs(scope, args.win_ms, args.order):
        out = analyse(gc_dir, task, args.edge_guard, args.n_permutations,
                      args.tfce, baseline_dur=args.baseline_dur)
        analysed[(task, stim, subset)] = (gc_dir, out)
        agg, _, _, _, res = out
        for pi in range(len(agg['pair_i'])):
            for key in ('fxy', 'fyx'):
                for b in BANDS:
                    fam_keys.append((task, stim, subset, pi, key, b))
                    fam_p.append(res[(key, b)][1]['p_cluster_min'][pi])
        print(f'  tested {task}/{stim} {subset}', flush=True)
    fdr = dict(zip(fam_keys, bh_fdr(np.asarray(fam_p, float))))
    n_fam = len(fam_keys)
    n_raw = int(np.sum(np.asarray(fam_p, float) < 0.05))
    n_fdr = int(np.sum(np.asarray(list(fdr.values()), float) < 0.05))
    print(f'  family: {n_fam} edge x band tests, {n_raw} at p<0.05 '
          f'uncorrected (~{0.05 * n_fam:.0f} expected by chance), '
          f'{n_fdr} surviving FDR')

    n = 0
    for (task, stim, subset), (gc_dir, out) in analysed.items():
        agg, base, tstart, tend, res = out
        roi = agg['roi_names']
        for pi, (i, j) in enumerate(zip(agg['pair_i'], agg['pair_j'])):
            src, tgt = str(roi[i]), str(roi[j])
            cond = ('A→B | C, conditioned on the third ROI' if scope == 'cond'
                    else 'bivariate')
            others = [sh(r) for k, r in enumerate(roi)
                      if k not in (i, j)] if scope == 'cond' else []
            ttl = (f'{sh(src)} ↔ {sh(tgt)}   {task} / {stim}   [{cond}'
                   + (f' = {", ".join(others)}]' if others else ']')
                   + f'\norder {args.order}, {args.win_ms:g} ms window, '
                     f'zscore, n={len(agg["subjects"])};  shaded = baseline '
                     f'[{base[0]:.0f}, {base[1]:.0f}] ms, edge guard '
                     f'{args.edge_guard:g} ms'
                     f'\nbars = cluster-permutation significant vs baseline '
                     f'(sign-flip, {args.n_permutations} perms, FWER across '
                     f'windows)')
            fdr_txt = []
            for key, lab in (('fxy', f'{sh(src)}→{sh(tgt)}'),
                             ('fyx', f'{sh(tgt)}→{sh(src)}')):
                for b in BANDS:
                    q = fdr[(task, stim, subset, pi, key, b)]
                    raw = res[(key, b)][1]['p_cluster_min'][pi]
                    if np.isfinite(raw) and raw < 0.05:
                        fdr_txt.append(f'{lab} {b}: p={raw:.3f}, '
                                       f'FDR q={q:.2f}'
                                       + ('' if q < 0.05 else ' (n.s.)'))
            ttl += ('\nfamily = ' + str(n_fam) + ' tests; '
                    + ('; '.join(fdr_txt) if fdr_txt
                       else 'no cluster reaches p<0.05 here'))
            png = (f'{outdir}/{task}_{stim}_{sh(src)}_{sh(tgt)}'
                   + (f'_cond{"".join(others)}' if others else '')
                   + f'_win{args.win_ms:g}_order{args.order}.png')
            plot_pair(agg, res, base, tstart, tend, pi, src, tgt, ttl, png,
                      scope)
            if tt_dir is not None:
                tt_ttl = (f'{sh(src)} ↔ {sh(tgt)}   {task} / {stim}   [{cond}'
                          + (f' = {", ".join(others)}]' if others else ']')
                          + f'\norder {args.order}, {args.win_ms:g} ms window, '
                            f'zscore, n={len(agg["subjects"])};  shaded = '
                            f'baseline [{base[0]:.0f}, {base[1]:.0f}] ms'
                            f'\nbars = pointwise one-sample t vs baseline, '
                            f'right-tailed, uncorrected'
                            f'\n(solid = run of ≥3 windows, faint = isolated)')
                plot_pair_ttest(agg, res, base, tstart, tend, pi, src, tgt,
                                tt_ttl, f'{tt_dir}/{os.path.basename(png)}')
            n += 1
    print(f'{n} figures -> {outdir}')


def run_edge_guard_ab(args, outdir):
    """The same pair at two edge-guard settings, side by side."""
    os.makedirs(outdir, exist_ok=True)
    guards = [0.0, args.edge_guard]
    n = 0
    for task, stim, subset, gc_dir in configs('biv', args.win_ms, args.order):
        if args.ab_pairs and subset not in args.ab_pairs:
            continue
        per = {}
        for g in guards:
            per[g] = analyse(gc_dir, task, g, args.n_permutations, args.tfce,
                             baseline_dur=args.baseline_dur)
        agg = per[guards[0]][0]
        roi = agg['roi_names']
        w = np.asarray(agg['window_ms'], float)
        for pi, (i, j) in enumerate(zip(agg['pair_i'], agg['pair_j'])):
            src, tgt = str(roi[i]), str(roi[j])
            fig, axes = plt.subplots(len(BANDS), len(guards),
                                     figsize=(6.0 * len(guards),
                                              2.4 * len(BANDS)),
                                     sharex=True, sharey='row', squeeze=False)
            for c, g in enumerate(guards):
                _, base, tstart, tend, res = per[g]
                keep = (w >= base[0]) & (w <= (tend if tend is not None
                                               else w[-1]))
                for r, band in enumerate(BANDS):
                    ax = axes[r][c]
                    ax.axvspan(base[0], base[1], color='0.6', alpha=0.16, lw=0)
                    ymax = 0.0
                    for key, col, lab in (('fxy', FWD, f'{sh(src)}→{sh(tgt)}'),
                                          ('fyx', REV, f'{sh(tgt)}→{sh(src)}')):
                        st, pm = res[(key, band)]
                        m, e = st['mean'][pi], st['sem'][pi]
                        ax.plot(w[keep], m[keep], color=col, lw=1.5, label=lab)
                        ax.fill_between(w[keep], (m - e)[keep], (m + e)[keep],
                                        color=col, alpha=0.16, lw=0)
                        ax.axhline(st['baseline_mean'][pi], color=col, ls='--',
                                   lw=0.9, alpha=0.7)
                        if np.isfinite(m[keep]).any():
                            ymax = max(ymax, np.nanmax((m + e)[keep]))
                    for k, (key, col) in enumerate((('fxy', FWD),
                                                    ('fyx', REV))):
                        _, pm = res[(key, band)]
                        yy = -0.055 * ymax * (k + 1)
                        for a, b_ in _contiguous_spans(
                                pm['sig_cluster'][pi] & keep):
                            ax.plot([w[a], w[b_]], [yy, yy], color=col, lw=3.2,
                                    solid_capstyle='butt')
                    ax.set_ylim(-0.055 * ymax * 2.7, ymax * 1.12)
                    ax.tick_params(labelsize=8)
                    if c == 0:
                        ax.set_ylabel(f'{LABEL[band]}\nGC', fontsize=9)
                    if r == 0:
                        nb = int(((w >= base[0]) & (w <= base[1])).sum())
                        bm = res[('fxy', BANDS[0])][0]['baseline_mean'][pi]
                        ax.set_title(f'edge guard {g:g} ms — baseline '
                                     f'[{base[0]:.0f}, {base[1]:.0f}] ms, '
                                     f'{nb} windows', fontsize=10,
                                     weight='bold')
                        if c == 0:
                            ax.legend(fontsize=8, frameon=False, ncol=2)
                    if r == len(BANDS) - 1:
                        ax.set_xlabel('window start (ms)', fontsize=9)
            # what actually changed
            deltas = []
            for band in BANDS:
                for key in ('fxy', 'fyx'):
                    b0 = per[guards[0]][4][(key, band)][0]['baseline_mean'][pi]
                    b1 = per[guards[1]][4][(key, band)][0]['baseline_mean'][pi]
                    s0 = int(per[guards[0]][4][(key, band)][1]
                             ['sig_cluster'][pi].sum())
                    s1 = int(per[guards[1]][4][(key, band)][1]
                             ['sig_cluster'][pi].sum())
                    deltas.append((band, key, b0, b1, s0, s1))
            db = np.nanmean([100 * (d[3] - d[2]) / abs(d[2]) for d in deltas
                             if np.isfinite(d[2]) and d[2] != 0])
            ds = sum(d[5] for d in deltas) - sum(d[4] for d in deltas)
            fig.suptitle(
                f'{sh(src)} ↔ {sh(tgt)}   {task} / {stim}  — edge guard A/B'
                f'\norder {args.order}, {args.win_ms:g} ms, zscore, '
                f'n={len(agg["subjects"])}'
                f'\ndropping the leading {args.edge_guard:g} ms moves the '
                f'baseline {db:+.1f}% and the significant-window count '
                f'{ds:+d} (3 bands x 2 directions)'
                f'\nunder zscore only the FIRST window is materially low '
                f'(66-80% of plateau), so this guard is mostly discarding '
                f'good baseline', fontsize=10)
            fig.tight_layout(rect=[0, 0, 1, 0.90])
            png = (f'{outdir}/AB_{task}_{stim}_{sh(src)}_{sh(tgt)}'
                   f'_win{args.win_ms:g}_order{args.order}.png')
            fig.savefig(png, dpi=150); plt.close(fig)
            n += 1
        print(f'  {task}/{stim} {subset}', flush=True)
    print(f'{n} A/B figures -> {outdir}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--win-ms', type=float, default=60.0)
    ap.add_argument('--order', type=int, default=6,
                    help='default 6; order 2 is degenerate (3-4x lower GC)')
    ap.add_argument('--edge-guard', type=float, default=0.0,
                    help='ms of epoch onset excluded before the '
                         'baseline. Default 0: under zscore only '
                         'the first window is materially low.')
    ap.add_argument('--baseline-dur', type=float, default=100.0,
                    help='ms of the leading window axis used as baseline. '
                         'Default 100. A shorter baseline is a real question '
                         'here: the number of INDEPENDENT window-lengths it '
                         'holds depends on --win-ms, so a flat 100 ms is not '
                         'the same amount of evidence across the sweep.')
    ap.add_argument('--suffix', default='',
                    help='appended to every output directory name, so a '
                         'variant lands beside the default instead of over it')
    ap.add_argument('--n-permutations', type=int, default=1024)
    ap.add_argument('--no-tfce', dest='tfce', action='store_false', default=True)
    ap.add_argument('--edge-guard-ab', action='store_true',
                    help='also write the guard 0 vs --edge-guard comparison')
    ap.add_argument('--ab-pairs', nargs='+', default=None,
                    help='limit the A/B to these rois_* directory names')
    ap.add_argument('--skip-ttest', action='store_true',
                    help='do not write the pointwise t-test set')
    ap.add_argument('--skip-pairwise', action='store_true')
    ap.add_argument('--skip-triplewise', action='store_true')
    args = ap.parse_args()

    print(f'root: {D}')
    print(f'config: order {args.order}, {args.win_ms:g} ms, zscore, '
          f'edge guard {args.edge_guard:g} ms\n')
    if not args.skip_pairwise:
        print('bivariate:')
        run_scope('biv', args, f'{D}/_figures_pairwise{args.suffix}',
                  None if args.skip_ttest else f'{D}/_figures_pairwise_ttest{args.suffix}')
    if not args.skip_triplewise:
        print('\ntriple-wise (A->B | C):')
        run_scope('cond', args, f'{D}/_figures_triplewise{args.suffix}',
                  None if args.skip_ttest else f'{D}/_figures_triplewise_ttest{args.suffix}')
    if args.edge_guard_ab:
        print('\nedge-guard A/B:')
        run_edge_guard_ab(args, f'{D}/_figures_edgeguard{args.suffix}')


if __name__ == '__main__':
    sys.exit(main())
