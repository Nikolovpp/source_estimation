#!/usr/bin/env python
"""Baseline sensitivity for the zscore sweep — one figure per task and scope.

Same question as GC_routes/story5_baseline_sensitivity, asked of the current
results: does the task-vs-baseline verdict depend on where the baseline is put
and how wide it is, rather than on the data?

Three panels per figure:

  (a) THE CAUSE. Group-mean GC over all directed edges, with the epoch-edge
      region marked. Under --normalize zscore the ensemble-SD division largely
      compensates for the edge transient (measured: first window 66-80% of
      plateau, second 86-96%, rest within 10%), so this panel is expected to
      show a much smaller edge than the un-normalised GC_routes version did.

  (b) SLIDING THE BASELINE. Percentage of task windows significant against a
      100 ms baseline slid away from the epoch edge. Flat = the verdict does
      not come from anchoring on the edge. Collapsing = it does.

  (c) WIDENING THE BASELINE. The same, varying baseline WIDTH from the epoch
      start. The precision of the baseline estimate stops improving at about
      one window length (see plot_gc_baseline_choice.py), so anything past that
      buys overlap rather than information — but it also risks reaching into
      the task.

STATISTIC. Pointwise right-tailed one-sample t against the group baseline
scalar, uncorrected across windows — the a-priori MATLAB design, and the one
the _figures_*_ttest sets use. The chance rate (5% of tested windows) is drawn
so a curve sitting at chance is identifiable.

PERCEPTION vs PRODUCTION. In perception t=0 is stimulus onset, so the task span
starts there and a baseline that slides past 0 is no longer pre-stimulus; the
crossing is marked. In overtProd t=0 is ARTICULATION onset and the planning
period before it is the period of interest, so the task span starts where the
baseline ends.

    conda activate mne
    python plot_gc_baseline_sensitivity.py
    python plot_gc_baseline_sensitivity.py --order 10 --win-ms 80
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
from config import GC_TASK_END
from run_granger import GC_OUTPUT_ROOT
from granger_stats import load_gc_group, task_vs_baseline, TASK_ONSET_MS

D = str(GC_OUTPUT_ROOT)
OUT = f'{D}/_figures_baseline_sensitivity'
BANDS = ['theta', 'low_beta', 'high_beta']
BCOL = {'theta': '#2e7d32', 'low_beta': '#1565c0', 'high_beta': '#c62828'}
LABEL = {'theta': 'theta 4–8 Hz', 'low_beta': 'low beta 12–18 Hz',
         'high_beta': 'high beta 18–30 Hz'}
CONDS = [('perception', 'percDiff'), ('overtProd', 'prodDiff')]
STARTS = [0, 25, 50, 75, 100, 150, 200]        # ms after the epoch begins
WIDTHS = [25, 50, 60, 75, 100, 150, 200]       # ms
ALPHA = 0.05


def sh(r):
    return str(r).replace('-lh', '')


def gather(task, stim, scope, win_ms, order):
    """-> (window_ms, {band: (n_subj, n_edges, n_win)}, [edge labels])"""
    n_roi = 2 if scope == 'biv' else 3
    suffix = '_zscore' if scope == 'biv' else '_zscore_conditional'
    cfg = f'order{order}_win{win_ms:g}ms_fs200{suffix}'
    stacks = {b: [] for b in BANDS}
    labels, w_ref = [], None
    for d in sorted(glob.glob(f'{D}/{task}/*/*/*/*/{cfg}/rois_*/{stim}')):
        subset = d.split('/')[-2]
        if subset.count('-lh') != n_roi or not glob.glob(f'{d}/*.npz'):
            continue
        agg = load_gc_group(d)
        w = np.asarray(agg['window_ms'], float)
        if w_ref is None:
            w_ref = w
        elif w.shape != w_ref.shape:
            print(f'    skipping {subset}: {w.size} windows vs {w_ref.size}')
            continue
        roi = agg['roi_names']
        for k, (i, j) in enumerate(zip(agg['pair_i'], agg['pair_j'])):
            for key, (s, t) in (('fxy', (roi[i], roi[j])),
                                ('fyx', (roi[j], roi[i]))):
                for b in BANDS:
                    stacks[b].append(agg[key][b][:, k, :])
                labels.append(f'{sh(s)}→{sh(t)}')
    if w_ref is None:
        return None, None, None
    return w_ref, {b: np.stack(stacks[b], axis=1) for b in BANDS}, labels


def pct_significant(stack, w, base, task_end, onset):
    """-> (per-edge % of tested task windows significant, n_tested).

    PER EDGE, not averaged. Averaging over all 12 directed edges buries the one
    or two that carry anything: on perception/percDiff the strongest edge sits
    near 50% while the mean over edges sits near 4%, i.e. below chance. The
    GC_routes story5 figure this follows plotted one line per edge for the same
    reason.
    """
    ts = base[1] if onset is None else max(base[1], onset)
    if ts >= (task_end if task_end is not None else w[-1]):
        return None, 0
    res = task_vs_baseline(stack, w, base, ts, task_end_ms=task_end)
    n_t = int(np.isfinite(res['pval'][0]).sum())
    if not n_t:
        return None, 0
    return 100.0 * res['sig'].sum(axis=1) / n_t, n_t


def figure(task, stim, scope, win_ms, order, out_png):
    w, stacks, labels = gather(task, stim, scope, win_ms, order)
    if w is None:
        print(f'  {task}/{stim} {scope}: no data')
        return None
    n_edges = stacks[BANDS[0]].shape[1]
    n_subj = stacks[BANDS[0]].shape[0]
    task_end = GC_TASK_END.get(task)
    task_end = task_end * 1000.0 if task_end is not None else None
    onset = TASK_ONSET_MS.get(task)
    step = float(np.median(np.diff(w)))

    fig, axes = plt.subplots(1, 3, figsize=(17.2, 4.9))

    # ---- (a) the cause: where is the edge, and how big is it? -----------
    ax = axes[0]
    for b in BANDS:
        m = np.nanmean(stacks[b], axis=(0, 1))
        e = (np.nanstd(np.nanmean(stacks[b], axis=1), axis=0)
             / np.sqrt(n_subj))
        ax.plot(w, m, color=BCOL[b], lw=1.5, label=LABEL[b])
        ax.fill_between(w, m - e, m + e, color=BCOL[b], alpha=0.16, lw=0)
    q = w.size // 4
    plateau = np.nanmedian(np.nanmean(stacks['theta'], axis=(0, 1))[q:-q])
    ax.axhline(plateau, color='0.55', ls='--', lw=1)
    prof = np.nanmean(stacks['theta'], axis=(0, 1))
    n_low = int(np.sum(prof[:10] < 0.9 * plateau))
    ax.axvspan(w[0], w[0] + max(n_low, 1) * step, color='#c62828', alpha=0.10,
               lw=0)
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo, yhi + 0.16 * (yhi - ylo))     # headroom for the legend
    ax.text(w[0] + max(n_low, 1) * step + 0.03 * (w[-1] - w[0]),
            ylo + 0.03 * (yhi - ylo),
            f'{n_low} window(s) below 90% of plateau', fontsize=8,
            color='#c62828', va='bottom')
    if onset is not None:
        ax.axvline(onset, color='0.3', ls=':', lw=1.2)
    ax.set_xlabel('window start (ms)')
    ax.set_ylabel('GC, mean over all directed edges')
    ax.set_title('(a) the cause — how big is the epoch edge?', fontsize=10.5,
                 loc='left')
    ax.legend(fontsize=8, frameon=False, loc='upper right', ncol=3,
              columnspacing=1.0, handlelength=1.3)

    # ---- (b) slide a fixed-width baseline off the edge ------------------
    # and (c) widen it. Both draw EVERY edge faintly and name the strongest,
    # so a curve that collapses is visible as such rather than averaged away.
    ref_base = (float(w[0]), float(w[0]) + 100.0)
    ref_top = {}
    for b in BANDS:
        pct, _ = pct_significant(stacks[b], w, ref_base, task_end, onset)
        if pct is not None:
            ref_top[b] = int(np.argmax(pct))
    for ax, xs_all, mode in ((axes[1], STARTS, 'slide'),
                             (axes[2], WIDTHS, 'widen')):
        curves = {}                       # (band, edge_idx) -> [pct per x]
        xs_ok = []
        for x in xs_all:
            base = ((float(w[0]) + x, float(w[0]) + x + 100.0) if mode == 'slide'
                    else (float(w[0]), float(w[0]) + x))
            got = False
            for b in BANDS:
                pct, _ = pct_significant(stacks[b], w, base, task_end, onset)
                if pct is None:
                    continue
                got = True
                for e in range(n_edges):
                    curves.setdefault((b, e), []).append(pct[e])
            if got:
                xs_ok.append(x)
        if not xs_ok:
            continue
        top = ref_top          # same edge in both panels, so they compare
        for (b, e), ys in curves.items():
            if len(ys) != len(xs_ok):
                continue
            if top.get(b) == e:
                ax.plot(xs_ok, ys, 'o-', color=BCOL[b], lw=2.2, ms=5,
                        zorder=3,
                        label=f'{labels[e]}  {LABEL[b].rsplit(" ", 1)[0]}')
            else:
                ax.plot(xs_ok, ys, '-', color=BCOL[b], lw=0.8, alpha=0.28,
                        zorder=1)
        ax.axhline(100 * ALPHA, color='0.35', ls=':', lw=1.3, zorder=2)
        ax.text(xs_ok[-1], 100 * ALPHA, ' chance', fontsize=8, color='0.35',
                va='bottom', ha='right')
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8, frameon=False, loc='upper right',
                  title='strongest edge at the default baseline',
                  title_fontsize=8)
        if mode == 'slide':
            if onset is not None:
                cross = onset - float(w[0]) - 100.0
                if xs_ok[0] <= cross <= xs_ok[-1]:
                    ax.axvspan(cross, xs_ok[-1], color='#c62828', alpha=0.09,
                               lw=0, zorder=0)
                    ax.text(cross + 3, ax.get_ylim()[1] * 0.55,
                            'baseline reaches\npast t = 0', fontsize=8,
                            color='#c62828', va='top')
            ax.set_xlabel('baseline start, ms after the epoch begins')
            ax.set_title('(b) slide a 100 ms baseline off the edge',
                         fontsize=10.5, loc='left')
        else:
            ax.axvline(win_ms, color='#6a1b9a', ls='--', lw=1.3, zorder=2)
            ax.text(win_ms + 2, ax.get_ylim()[1] * 0.55,
                    f'one window\nlength ({win_ms:g} ms)', fontsize=8,
                    color='#6a1b9a', va='top')
            if onset is not None:
                reach = onset - float(w[0])
                if xs_ok[0] <= reach <= xs_ok[-1]:
                    ax.axvspan(reach, xs_ok[-1], color='#c62828', alpha=0.09,
                               lw=0, zorder=0)
            ax.set_xlabel('baseline width, from the epoch start (ms)')
            ax.set_title('(c) widen the baseline', fontsize=10.5, loc='left')
        ax.set_ylabel('% of task windows significant')

    scope_lab = ('bivariate' if scope == 'biv'
                 else 'triple-wise, A→B|C conditioned on the third ROI')
    onset_lab = ('task span starts at t=0 (stimulus onset)' if onset is not None
                 else 'task span starts where the baseline ends '
                      '(t=0 is articulation)')
    fig.suptitle(
        f'Baseline sensitivity — {task} / {stim},  {scope_lab}\n'
        f'order {order}, {win_ms:g} ms window, zscore, n={n_subj}, '
        f'{n_edges} directed edges;  {onset_lab}\n'
        f'statistic: pointwise right-tailed one-sample t vs the group baseline '
        f'scalar, uncorrected across windows',
        fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png, n_edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--win-ms', type=float, default=60.0)
    ap.add_argument('--order', type=int, default=6)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    print(f'root: {D}\norder {args.order}, {args.win_ms:g} ms, zscore\n')
    for scope, tag in (('biv', 'pairwise'), ('cond', 'triplewise')):
        for task, stim in CONDS:
            p = (f'{OUT}/baseline_sensitivity_{tag}_{task}_{stim}'
                 f'_win{args.win_ms:g}_order{args.order}.png')
            print(f'{tag} · {task}/{stim}')
            res = figure(task, stim, scope, args.win_ms, args.order, p)
            if res:
                print(f'  wrote {os.path.basename(res[0])}  '
                      f'({res[1]} directed edges)')
    print(f'\nfigures in {OUT}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
