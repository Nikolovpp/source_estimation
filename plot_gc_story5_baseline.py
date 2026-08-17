#!/usr/bin/env python
"""story5_baseline_sensitivity, rebuilt on the zscore sweep results.

An exact replica of the GC_routes figure of that name — same 2x2 layout, same
two questions, same statistic — reading the current pairwise and triple-wise
output instead of the GC_routes tree. Two figures:

    story5_baseline_sensitivity_pairwise.png
    story5_baseline_sensitivity_triplewise.png

LAYOUT (per row: perception on top, overt production below)
    left   THE CAUSE   theta GC averaged over the four pathway edges, with the
                       partial-data epoch edge shaded and the plateau marked.
    right  THE EFFECT  percentage of task timepoints significant above
                       baseline, as a 100 ms baseline is slid away from that
                       edge. One line per edge.

THE FOUR PATHWAY EDGES are the hypothesis-relevant direction for each task, as
in the original: temporal/auditory -> frontal for perception, frontal ->
temporal for production. Every OTHER directed edge is drawn faintly as well,
because restricting to the four hid the result — the perception pathway set
excludes ifc->tpc, the only edge here whose significance depends on the
baseline. Whichever edge falls furthest is named.

TWO DEPARTURES FROM THE ORIGINAL, both forced by the data.
  * The statistic is UNCORRECTED across windows. story5 used BH-FDR; on these
    results FDR is exactly zero at every baseline placement for every edge, so
    the panel would be blank. The chance line is drawn instead.
  * The titles report what is measured rather than the original's conclusion.
    Under --normalize zscore the epoch-edge artifact is largely gone: averaged
    over the perception pathway edges the first window sits 17% below plateau
    (story5: 65%), and in production it sits ABOVE plateau. "Significance
    collapses once the baseline clears the edge" is therefore not a caption
    this data supports for those edges, and it is not asserted.

    conda activate mne
    python plot_gc_story5_baseline.py
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
from granger_stats import (load_gc_group, task_vs_baseline, bh_fdr,
                           TASK_ONSET_MS)

D = str(GC_OUTPUT_ROOT)
OUT = f'{D}/_figures_baseline_sensitivity'
BAND = 'theta'
STARTS = [0, 25, 50, 75, 100, 125, 150, 175, 200]
BASE_W = 100.0
ALPHA = 0.05

# The hypothesis-relevant direction per task, as in the original figure.
EDGES = {
    'perception': [('awfa-lh', 'ifc-lh'), ('awfa-lh', 'pmc-lh'),
                   ('tpc-lh', 'ifc-lh'), ('tpc-lh', 'pmc-lh')],
    'overtProd':  [('pmc-lh', 'awfa-lh'), ('ifc-lh', 'awfa-lh'),
                   ('pmc-lh', 'tpc-lh'), ('ifc-lh', 'tpc-lh')],
}
ECOL = ['#8ac48a', '#3f9e8c', '#2b7ba8', '#1f3f6b']
LINE, FILL = '#2c7fb8', '#c6dbef'
ROWS = [('perception', 'percDiff', 'PERCEPTION'),
        ('overtProd', 'prodDiff', 'OVERT PRODUCTION')]


def sh(r):
    return str(r).replace('-lh', '')


def gather(task, stim, scope, win_ms, order):
    """-> (window_ms, {(src,tgt): (n_subj, n_win)}) for EVERY directed edge.

    story5 plotted a hand-picked four. Repeating that here hid the result: the
    perception pathway set (temporal->frontal) excludes ifc->tpc, which is the
    only edge in this data whose significance depends on the baseline. So every
    edge is returned; the four pathway edges are still drawn in colour and the
    rest faintly, so nothing is both present and invisible.
    """
    n_roi = 2 if scope == 'biv' else 3
    suffix = '_zscore' if scope == 'biv' else '_zscore_conditional'
    cfg = f'order{order}_win{win_ms:g}ms_fs200{suffix}'
    out, w_ref = {}, None
    for d in sorted(glob.glob(f'{D}/{task}/*/*/*/*/{cfg}/rois_*/{stim}')):
        subset = d.split('/')[-2]
        if subset.count('-lh') != n_roi or not glob.glob(f'{d}/*.npz'):
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
                if e not in out:
                    out[e] = agg[key][BAND][:, k, :]
    return w_ref, out


def pct_sig(stack, w, base, task_end, onset):
    """% of tested task windows significant, UNCORRECTED across windows.

    story5 used BH-FDR. On this data FDR gives exactly zero at every baseline
    placement for every edge, so the panel would carry no information at all.
    The uncorrected count is what the _figures_*_ttest sets report and is the
    a-priori MATLAB design; the chance line is drawn so it can be read against
    it, and the caption states that nothing survives FDR.
    """
    ts = base[1] if onset is None else max(base[1], onset)
    if ts >= (task_end if task_end is not None else w[-1]):
        return np.nan
    res = task_vs_baseline(stack[:, None, :], w, base, ts, task_end_ms=task_end)
    p = res['pval'][0]
    ok = np.isfinite(p)
    if not ok.any():
        return np.nan
    return 100.0 * float((p[ok] < ALPHA).sum()) / int(ok.sum())


def build(scope, win_ms, order, out_png):
    fig, axes = plt.subplots(2, 2, figsize=(15.6, 8.4))
    any_data = False
    for r, (task, stim, TITLE) in enumerate(ROWS):
        w, edges = gather(task, stim, scope, win_ms, order)
        if w is None or not edges:
            print(f'  {task}: no data'); continue
        any_data = True
        task_end = GC_TASK_END.get(task)
        task_end = task_end * 1000.0 if task_end is not None else None
        onset = TASK_ONSET_MS.get(task)
        centre = w + win_ms / 2.0            # story5 plots window CENTRE
        step = float(np.median(np.diff(w)))
        order_e = [e for e in EDGES[task] if e in edges]
        rest_e = [e for e in edges if e not in order_e]

        # ---------- left: the cause ----------
        ax = axes[r][0]
        M = np.stack([edges[e] for e in order_e])          # (n_edge, n_subj, n_win)
        per_subj = np.nanmean(M, axis=0)                   # (n_subj, n_win)
        m = np.nanmean(per_subj, axis=0)
        sem = np.nanstd(per_subj, axis=0) / np.sqrt(per_subj.shape[0])
        ax.plot(centre, m, color=LINE, lw=1.8, zorder=3)
        ax.fill_between(centre, m - sem, m + sem, color=FILL, lw=0, zorder=2)
        q = m.size // 4
        plateau = np.nanmedian(m[q:-q])
        ax.axhline(plateau, color='0.45', ls='--', lw=1.2, zorder=1)
        ax.text(centre[-1], plateau, 'plateau', fontsize=9, color='0.35',
                va='center', ha='right')
        n_low = int(np.sum(m[:10] < 0.9 * plateau))
        first_pct = 100.0 * (m[0] / plateau - 1.0)      # signed
        if n_low:
            ax.axvspan(centre[0] - step / 2, centre[0] + (n_low - 0.5) * step,
                       color='#e05a5a', alpha=0.16, lw=0, zorder=0)
        ax.text(centre[0] + 1.5 * step, ax.get_ylim()[1],
                (f'{n_low} window below\n90% of plateau' if n_low
                 else 'no depressed\nedge windows'), fontsize=9,
                color='#b32b2b' if n_low else '#5b7c5b', va='top')
        ax.axvline(0, color='0.2', ls='--', lw=1.1, zorder=1)
        ax.set_xlabel('window centre (ms)', fontsize=10)
        ax.set_ylabel(f'theta GC (mean of {len(order_e)} edges)', fontsize=10)
        ax.set_title(
            f'{TITLE} — the cause\n'
            + (f'{n_low} window{"s" if n_low != 1 else ""} below 90% of '
               f'plateau; first sits {abs(first_pct):.0f}% '
               f'{"below" if first_pct < 0 else "above"}'
               if n_low else
               f'no edge artifact — first window sits {abs(first_pct):.0f}% '
               f'{"below" if first_pct < 0 else "above"} plateau'),
            fontsize=11.5)

        # ---------- right: the effect ----------
        ax = axes[r][1]
        curves = {}
        for e in list(order_e) + list(rest_e):
            xs, ys = [], []
            for s0 in STARTS:
                base = (float(w[0]) + s0, float(w[0]) + s0 + BASE_W)
                v = pct_sig(edges[e], w, base, task_end, onset)
                if np.isfinite(v):
                    xs.append(s0); ys.append(v)
            curves[e] = (xs, ys)
        # the mover: largest drop from the edge-anchored baseline
        mover = max(curves, key=lambda e: (curves[e][1][0] - min(curves[e][1]))
                    if curves[e][1] else -1)
        for e in rest_e:
            xs, ys = curves[e]
            if e == mover:
                continue
            ax.plot(xs, ys, '-', color='0.72', lw=1.0, zorder=1)
        for c, e in enumerate(order_e):
            xs, ys = curves[e]
            ax.plot(xs, ys, 'o-', color=ECOL[c], lw=2.0, ms=6,
                    label=f'{sh(e[0])}→{sh(e[1])}', zorder=3)
        if mover not in order_e:
            xs, ys = curves[mover]
            ax.plot(xs, ys, 'o--', color='#c2185b', lw=2.2, ms=6, zorder=4,
                    label=f'{sh(mover[0])}→{sh(mover[1])}  (largest drop)')
        ax.axhline(100 * ALPHA, color='0.4', ls=':', lw=1.1, zorder=2)
        ax.axvspan(-10, BASE_W, color='#e05a5a', alpha=0.13, lw=0, zorder=0)
        ax.text(-6, ax.get_ylim()[1] * 0.97, 'baseline overlaps\nthe edge',
                fontsize=9, color='#b32b2b', va='top')
        ax.set_xlim(-12, STARTS[-1] + 8)
        ax.set_ylim(bottom=0)
        ax.set_xlabel('baseline start, ms after the epoch begins   '
                      '(width fixed at 100 ms)', fontsize=10)
        ax.set_ylabel('% of timepoints significant\nabove baseline '
                      '(uncorrected)', fontsize=10)
        drop = curves[mover][1][0] - min(curves[mover][1])
        ax.set_title(
            f'{TITLE} — the effect\n'
            + (f'{sh(mover[0])}→{sh(mover[1])} falls '
               f'{curves[mover][1][0]:.0f}% → {min(curves[mover][1]):.0f}% '
               f'as the baseline clears the edge' if drop > 5 else
               'no edge-anchored significance to lose'),
            fontsize=11.5)
        ax.legend(fontsize=9, frameon=False, ncol=2, loc='upper right')

    if not any_data:
        plt.close(fig); return None
    for row in axes:
        for ax in row:
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(axis='y', color='0.88', lw=0.7)
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=9)
    scope_lab = ('Pairwise spectral GC' if scope == 'biv'
                 else 'Triple-wise conditional GC (A→B | C)')
    fig.suptitle(
        'Baseline placement decides the result — a 100 ms baseline slid away '
        'from the epoch edge\n'
        f'{scope_lab}, 20 subjects, zscore. NOT the partial-data artifact the '
        'GC_routes version showed: that is largely gone here.\n'
        'The pre-stimulus period simply is not flat — ifc→tpc theta runs '
        '0.0109 → 0.0073 → 0.0137 → 0.0168 across the four 50 ms bins before '
        't=0,\nreaching its post-stimulus level by −50 ms. There is no single '
        'number that is "the baseline" for that edge.',
        fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_png, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return out_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--win-ms', type=float, default=60.0)
    ap.add_argument('--order', type=int, default=6)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for scope, tag in (('biv', 'pairwise'), ('cond', 'triplewise')):
        print(f'{tag}:')
        p = build(scope, args.win_ms, args.order,
                  f'{OUT}/story5_baseline_sensitivity_{tag}.png')
        if p:
            print(f'  wrote {os.path.basename(p)}')
    print(f'\nfigures in {OUT}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
