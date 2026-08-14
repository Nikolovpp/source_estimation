#!/usr/bin/env python
"""Window x order sweep summary — one figure per ROI subset.

Reads GC_source_space output from run_gc_window_order_sweep.sh. Each subset is
its own ``rois_*`` directory, and its SIZE sets the analysis:

    2 ROIs -> bivariate GC (identical to parametric BSMART pairwise)
    3 ROIs -> A->B|C, each edge conditioned on the single remaining ROI

Every panel answers one question: does the reported GC survive changing the
window and the model order? x = model order, one line per window. If the lines
sit on top of each other the measure is robust; if they cross or invert, the
parameter choice is doing the work.

EDGE WINDOWS. An epoch-onset/offset transient depresses the first and last
windows. Measured on the pre-bug _none files: at a 60 ms window the leading
4-6 windows and trailing 6-16 sit at 46-79% of the interior plateau; at 120 ms
only the first window is affected, because a longer window dilutes the same
contaminated span. So the guard is expressed in MILLISECONDS of epoch edge
(EDGE_MS) and converted to windows using each run's own step.

INCOMPLETE CELLS. A config with fewer than 20 subjects is skipped and listed,
never silently averaged over a different n than its neighbours.

    conda activate mne
    python plot_gc_sweep_summary.py
"""
import os
import re
import glob
import collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def _root():
    """Derive the GC root from config.env like the runners do.

    A hardcoded /mnt/r path silently found nothing on the workstation, whose
    project root is /media/maxlab_sharedrive/... — and "0 completed configs"
    reads as "the sweep produced nothing", not as "wrong machine".
    """
    from run_granger import GC_OUTPUT_ROOT
    return str(GC_OUTPUT_ROOT)


D = _root()
OUT = f'{D}/_figures_sweep'
os.makedirs(OUT, exist_ok=True)

BANDS = ['theta', 'low_beta', 'high_beta']
LABEL = {'theta': 'theta 4–8 Hz', 'low_beta': 'low beta 12–18 Hz',
         'high_beta': 'high beta 18–30 Hz'}
WCOL = {40: '#4575b4', 60: '#1a9850', 80: '#d73027'}
N_EXPECT = 20
# MEASURED, not assumed. On the pre-bug _none files (120 subject files, both
# tasks, n=20 each) the epoch-onset transient contaminates a fixed span of
# SIGNAL, so the number of affected WINDOWS is that span divided by the step:
#   win60  : leading 4-6 windows (20-30 ms), trailing 6-16 windows
#   win120 : leading 1 window, trailing 0-2  (a longer window dilutes it)
# The old EDGE_DROP = 3 was too few at 60 ms and dropped nothing at the
# trailing end, where the depression is deeper (down to 46% of plateau).
EDGE_MS = 30.0                     # ms of epoch edge to discard, each end


def sh(r):
    return r.replace('-lh', '')


def scan():
    """-> {(task, stim, subset, config): {(win, order): [files]}}

    Matches ANY gc-mode. The glob used to require ``*conditional`` in the
    config directory, from when the sweep forced conditional mode on every
    subset. 2-ROI subsets now run pairwise, whose tag carries no mode suffix,
    so that glob would have found none of the bivariate arm and reported
    "0 completed configs" — silently, right after the rerun that produced it.

    Everything AFTER the order/window/fs stem — the normalize mode, the
    gc-mode, _perclass, _notrialdemean — is part of the group key, not folded
    into the cell. Grouping on (window, order) alone would have averaged the
    A/B arms together into a single line, which is worse than finding nothing.
    """
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    for f in glob.glob(f'{D}/**/rois_*/*/*.npz', recursive=True):
        p = f.split('GC_source_space/')[1].split('/')
        if len(p) < 8:
            continue
        task, cfg, subset, stim = p[0], p[5], p[6].replace('rois_', ''), p[7]
        m = re.match(r'order(\d+)_win(\d+)ms_fs(\d+)_?(.*)$', cfg)
        if m is None:
            continue
        arm = m.group(4) or 'none'
        out[(task, stim, subset, arm)][(int(m.group(2)), int(m.group(1)))].append(f)
    return out


def edge_drop(window_ms):
    """How many windows EDGE_MS covers at this run's step. Never more than a
    quarter of the axis, so a short run degrades rather than empties."""
    if window_ms.size < 2:
        return 0
    step = float(np.median(np.diff(window_ms)))
    return int(min(np.ceil(EDGE_MS / max(step, 1e-9)), window_ms.size // 4))


def band_mean(files, band):
    """-> {(src, tgt): (n_subj,)} band GC, averaged over non-edge windows."""
    acc = collections.defaultdict(list)
    for f in files:
        z = np.load(f, allow_pickle=True)
        k_drop = edge_drop(np.asarray(z['window_ms'], float))
        rois = [str(r) for r in z['roi_names']]
        for a, (i, j) in enumerate(zip(z['pair_i'], z['pair_j'])):
            for key, edge in ((f'fxy_{band}', (rois[i], rois[j])),
                              (f'fyx_{band}', (rois[j], rois[i]))):
                if key not in z.files:
                    continue
                v = np.asarray(z[key])
                v = v[a] if v.ndim == 2 else v
                acc[edge].append(np.nanmean(
                    v[k_drop:len(v) - k_drop] if k_drop else v))
    return {e: np.array(v) for e, v in acc.items()}


def figure(task, stim, subset, arm, cells):
    good = {k: v for k, v in cells.items() if len(v) == N_EXPECT}
    skipped = {k: len(v) for k, v in cells.items() if len(v) != N_EXPECT}
    if not good:
        return None
    orders = sorted({o for _, o in good})
    wins = sorted({w for w, _ in good})

    edges = sorted(band_mean(next(iter(good.values())), 'theta'))
    n_roi = subset.count('-lh')
    scope = 'triple-wise  A→B|C' if n_roi == 3 else 'bivariate'

    fig, axes = plt.subplots(len(BANDS), len(edges),
                             figsize=(2.5 * len(edges) + 1.4, 2.5 * len(BANDS)),
                             squeeze=False, sharex=True)
    for r, band in enumerate(BANDS):
        vals = {k: band_mean(v, band) for k, v in good.items()}
        ymax = 0.0
        for c, e in enumerate(edges):
            ax = axes[r][c]
            for w in wins:
                m, s, xs = [], [], []
                for o in orders:
                    if (w, o) not in vals or e not in vals[(w, o)]:
                        continue
                    a = vals[(w, o)][e]
                    xs.append(o)
                    m.append(np.nanmean(a))
                    s.append(np.nanstd(a) / np.sqrt(np.isfinite(a).sum()))
                if not xs:
                    continue
                m, s = np.array(m), np.array(s)
                ax.errorbar(xs, m, yerr=s, color=WCOL.get(w, '0.3'), lw=1.7,
                            marker='o', ms=4, capsize=2, label=f'{w} ms')
                ymax = max(ymax, np.nanmax(m + s))
            ax.axhline(0, color='0.7', lw=0.6, zorder=0)
            if r == 0:
                ax.set_title(f'{sh(e[0])} → {sh(e[1])}', fontsize=10,
                             weight='bold')
            if c == 0:
                ax.set_ylabel(f'{LABEL[band]}\nGC', fontsize=9)
            if r == len(BANDS) - 1:
                ax.set_xlabel('model order', fontsize=9)
            ax.set_xticks(orders)
            ax.tick_params(labelsize=8)
        for c in range(len(edges)):
            axes[r][c].set_ylim(0, ymax * 1.1 if ymax > 0 else 1)

    axes[0][-1].legend(fontsize=8, frameon=False, title='window',
                       title_fontsize=8, loc='best')
    note = ''
    if skipped:
        note = ('\nincomplete cells skipped: '
                + ', '.join(f'{w}ms/o{o} (n={n})'
                            for (w, o), n in sorted(skipped.items())))
    fig.suptitle(
        f'{task} · {stim} · {" + ".join(sh(r) for r in subset.split("-lh")[:-1])}'
        f'   [{scope}]   config: {arm}\n'
        f'LCMV, n={N_EXPECT}, mean±SEM over subjects; '
        f'GC averaged over time after dropping {EDGE_MS:g} ms at each '
        f'epoch edge{note}',
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.99 - 0.035 * len(BANDS) / 3])
    p = f'{OUT}/sweep_{task}_{stim}_{subset}_{arm}.png'
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p, len(good), skipped


if __name__ == '__main__':
    cells = scan()
    print(f'{sum(len(v) for v in cells.values())} completed configs '
          f'across {len(cells)} (task, stim, subset, config) groups')
    if not cells:
        print(f'nothing found under {D} — wrong machine, or the sweep has '
              f'not written anything yet')
    print()
    for key in sorted(cells):
        res = figure(*key, cells[key])
        if res is None:
            print(f'  {key}: no complete cell, skipped')
            continue
        p, n, sk = res
        flag = f'   [{len(sk)} incomplete cell(s) skipped]' if sk else ''
        print(f'  {os.path.basename(p)}   {n} cells{flag}')
    print(f'\nfigures in {OUT}')
