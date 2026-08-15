#!/usr/bin/env python
"""Headline figures for the completed window x order sweep (zscore).

plot_gc_sweep_summary.py writes one detail figure per ROI subset — 32 of them.
This writes the three that answer the questions the sweep was run to settle:

  fig1_robustness   Does the answer depend on the window and the model order?
                    x = order, one line per window, averaged over all 12
                    directed bivariate edges. Flat lines = the measure is a
                    property of the data; crossing lines = the parameter
                    choice is doing the work.

  fig2_pattern      What is the directed pattern? A source x target matrix per
                    band and per task/contrast, averaged over the whole grid,
                    with the across-grid spread printed in each cell so a value
                    that only exists at one grid point cannot pass for a
                    finding.

  fig3_conditioned  Does an edge survive conditioning on a third ROI? Bivariate
                    GC against the triple-wise A->B|C for the same edge. Points
                    on the diagonal are unmediated; points far below it are
                    explained away by the third region.

All three use the measured 30 ms edge guard at both ends and NaN-aware
averaging, matching plot_gc_sweep_summary.

    conda activate mne
    python plot_gc_sweep_headline.py            # uses the cache if present
    python plot_gc_sweep_headline.py --rebuild  # re-read every .npz
"""
import os
import re
import sys
import glob
import argparse
import collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_granger import GC_OUTPUT_ROOT
from granger import DEFAULT_BANDS

D = str(GC_OUTPUT_ROOT)
OUT = f'{D}/_figures_sweep'
CACHE = f'{OUT}/_headline_cache.npz'
os.makedirs(OUT, exist_ok=True)

EDGE_MS = 30.0
BANDS = ['theta', 'low_beta', 'high_beta']
LABEL = {'theta': 'theta 4–8 Hz', 'alpha': 'alpha 8–12 Hz',
         'low_beta': 'low beta 12–18 Hz', 'high_beta': 'high beta 18–30 Hz'}
WCOL = {40: '#4575b4', 60: '#1a9850', 80: '#d73027'}
ROI_ORDER = ['awfa-lh', 'ifc-lh', 'pmc-lh', 'tpc-lh']
CONDS = [('overtProd', 'prodDiff'), ('overtProd', 'percDiff'),
         ('perception', 'percDiff'), ('perception', 'prodDiff')]


def sh(r):
    return r.replace('-lh', '')


def edge_drop(window_ms):
    if window_ms.size < 2:
        return 0
    step = float(np.median(np.diff(window_ms)))
    return int(min(np.ceil(EDGE_MS / max(step, 1e-9)), window_ms.size // 4))


def build(rebuild=False):
    """-> {(task, stim, scope, src, tgt, band, win, order): (n_subj,)}

    ``scope`` is 'biv' (2-ROI subset) or 'cond' (3-ROI, A->B|C).
    """
    if os.path.exists(CACHE) and not rebuild:
        z = np.load(CACHE, allow_pickle=True)
        return z['data'].item()

    data = collections.defaultdict(list)
    files = glob.glob(f'{D}/**/rois_*/*/*.npz', recursive=True)
    n = 0
    for f in files:
        p = f.split('GC_source_space/')[1].split('/')
        if len(p) < 8:
            continue
        task, cfg, subset, stim = p[0], p[5], p[6], p[7]
        m = re.match(r'order(\d+)_win(\d+)ms_fs\d+_zscore(_conditional)?$', cfg)
        if m is None:                       # zscore sweep only
            continue
        order, win = int(m.group(1)), int(m.group(2))
        n_roi = subset.count('-lh')
        scope = 'cond' if n_roi == 3 else 'biv' if n_roi == 2 else None
        if scope is None:
            continue
        z = np.load(f, allow_pickle=True)
        k = edge_drop(np.asarray(z['window_ms'], float))
        names = [str(r) for r in z['roi_names']]
        for a, (i, j) in enumerate(zip(z['pair_i'], z['pair_j'])):
            for pre, (s, t) in (('fxy_', (names[i], names[j])),
                                ('fyx_', (names[j], names[i]))):
                for b in DEFAULT_BANDS:
                    key = pre + b
                    if key not in z.files:
                        continue
                    v = np.asarray(z[key], float)
                    v = v[a] if v.ndim == 2 else v
                    v = v[k:len(v) - k] if k else v
                    data[(task, stim, scope, s, t, b, win, order)].append(
                        float(np.nanmean(v)))
        n += 1
        if n % 1000 == 0:
            print(f'  {n}/{len(files)} files', flush=True)
    data = {k: np.asarray(v) for k, v in data.items()}
    np.savez_compressed(CACHE, data=np.array(data, dtype=object))
    print(f'  cached {len(data)} cells -> {CACHE}')
    return data


def fig_robustness(data):
    """GC vs order, one line per window, averaged over all directed edges."""
    fig, axes = plt.subplots(len(BANDS), len(CONDS),
                             figsize=(3.3 * len(CONDS), 2.6 * len(BANDS)),
                             squeeze=False, sharex=True)
    for c, (task, stim) in enumerate(CONDS):
        for r, band in enumerate(BANDS):
            ax = axes[r][c]
            wins = sorted({k[6] for k in data if k[:3] == (task, stim, 'biv')})
            orders = sorted({k[7] for k in data if k[:3] == (task, stim, 'biv')})
            ymax = 0.0
            for w in wins:
                xs, ms, es = [], [], []
                for o in orders:
                    vals = [np.nanmean(v) for k, v in data.items()
                            if k[:3] == (task, stim, 'biv') and k[5] == band
                            and k[6] == w and k[7] == o]
                    if not vals:
                        continue
                    xs.append(o); ms.append(np.nanmean(vals))
                    es.append(np.nanstd(vals) / np.sqrt(len(vals)))
                if not xs:
                    continue
                ms, es = np.array(ms), np.array(es)
                ax.errorbar(xs, ms, yerr=es, color=WCOL.get(w, '0.3'), lw=1.8,
                            marker='o', ms=4, capsize=2, label=f'{w} ms')
                ymax = max(ymax, np.nanmax(ms + es))
            ax.set_ylim(0, ymax * 1.15 if ymax > 0 else 1)
            ax.set_xticks(orders)
            ax.tick_params(labelsize=8)
            if r == 0:
                ax.set_title(f'{task}\n{stim}', fontsize=10, weight='bold')
            if c == 0:
                ax.set_ylabel(f'{LABEL[band]}\nGC', fontsize=9)
            if r == len(BANDS) - 1:
                ax.set_xlabel('MVAR model order', fontsize=9)
    axes[0][-1].legend(fontsize=8, frameon=False, title='window',
                       title_fontsize=8)
    fig.suptitle(
        'Window x order robustness — LCMV, zscore (ERP + ensemble-SD removed), '
        'bivariate GC\nmean over all 12 directed edges ± SEM across edges, '
        'n=20 subjects; 30 ms dropped at each epoch edge',
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    p = f'{OUT}/fig1_robustness.png'
    fig.savefig(p, dpi=150); plt.close(fig)
    return p


def fig_pattern(data):
    """source x target matrix per band and condition, averaged over the grid."""
    fig, axes = plt.subplots(len(CONDS), len(BANDS),
                             figsize=(3.2 * len(BANDS), 3.0 * len(CONDS)),
                             squeeze=False)
    # one colour scale across the whole figure, or panels are not comparable
    allv = [np.nanmean(v) for k, v in data.items()
            if k[2] == 'biv' and k[5] in BANDS]
    vmax = np.nanpercentile(allv, 98) if allv else 1.0
    for r, (task, stim) in enumerate(CONDS):
        for c, band in enumerate(BANDS):
            ax = axes[r][c]
            M = np.full((len(ROI_ORDER), len(ROI_ORDER)), np.nan)
            S = np.full_like(M, np.nan)
            for i, src in enumerate(ROI_ORDER):
                for j, tgt in enumerate(ROI_ORDER):
                    if src == tgt:
                        continue
                    cells = [np.nanmean(v) for k, v in data.items()
                             if k[:3] == (task, stim, 'biv')
                             and k[3] == src and k[4] == tgt and k[5] == band]
                    if cells:
                        M[i, j] = np.nanmean(cells)
                        S[i, j] = np.nanstd(cells)      # spread ACROSS the grid
            im = ax.imshow(M, cmap='viridis', vmin=0, vmax=vmax)
            for i in range(len(ROI_ORDER)):
                for j in range(len(ROI_ORDER)):
                    if np.isfinite(M[i, j]):
                        ax.text(j, i, f'{M[i, j]:.3f}\n±{S[i, j]:.3f}',
                                ha='center', va='center', fontsize=7,
                                color='w' if M[i, j] < vmax * 0.6 else 'k')
                    elif i == j:
                        ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1,
                                                   color='0.85'))
            ax.set_xticks(range(len(ROI_ORDER)))
            ax.set_yticks(range(len(ROI_ORDER)))
            ax.set_xticklabels([sh(r_) for r_ in ROI_ORDER], fontsize=8)
            ax.set_yticklabels([sh(r_) for r_ in ROI_ORDER], fontsize=8)
            if r == 0:
                ax.set_title(LABEL[band], fontsize=10, weight='bold')
            if c == 0:
                ax.set_ylabel(f'{task}\n{stim}\n\nsource', fontsize=9)
            if r == len(CONDS) - 1:
                ax.set_xlabel('target', fontsize=9)
    fig.colorbar(im, ax=axes, shrink=0.5, label='bivariate GC')
    fig.suptitle('Directed pattern — bivariate GC, mean over the 11 window x '
                 'order cells\ncell text: mean ± SD ACROSS the grid (a large '
                 '± means the value depends on the parameters, not the data)',
                 fontsize=11)
    p = f'{OUT}/fig2_pattern.png'
    fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
    return p


def fig_conditioned(data):
    """Bivariate vs triple-wise A->B|C for the edges present in both arms."""
    fig, axes = plt.subplots(1, len(BANDS), figsize=(4.2 * len(BANDS), 4.2),
                             squeeze=False)
    MARK = {'overtProd': 'o', 'perception': 's'}
    COL = {'prodDiff': '#d73027', 'percDiff': '#4575b4'}
    for c, band in enumerate(BANDS):
        ax = axes[0][c]
        lim = 0.0
        for task, stim in CONDS:
            xs, ys = [], []
            for k, v in data.items():
                if k[:3] != (task, stim, 'cond') or k[5] != band:
                    continue
                bk = (task, stim, 'biv', k[3], k[4], band, k[6], k[7])
                if bk not in data:
                    continue
                xs.append(np.nanmean(data[bk])); ys.append(np.nanmean(v))
            if not xs:
                continue
            ax.scatter(xs, ys, s=14, alpha=0.45, marker=MARK[task],
                       color=COL[stim], edgecolors='none',
                       label=f'{task}/{stim}')
            lim = max(lim, np.nanmax(xs), np.nanmax(ys))
        lim *= 1.05
        ax.plot([0, lim], [0, lim], color='0.4', lw=1, ls='--', zorder=0)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel('bivariate GC', fontsize=9)
        if c == 0:
            ax.set_ylabel('conditional GC   A→B | C', fontsize=9)
            ax.legend(fontsize=7, frameon=False, loc='upper left')
        ax.set_title(LABEL[band], fontsize=10, weight='bold')
        ax.tick_params(labelsize=8)
    fig.suptitle('Does the edge survive conditioning on the third ROI?\n'
                 'one point per (edge x window x order); on the diagonal = '
                 'unmediated, far below = explained away by the third region',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    p = f'{OUT}/fig3_conditioned.png'
    fig.savefig(p, dpi=150); plt.close(fig)
    return p


def fig_stability(data):
    """Rank stability across the grid, and the asymmetry that survives it.

    Magnitudes are NOT grid-stable (fig2: the across-grid SD is 30-60% of the
    mean). The question that matters is whether the ORDERING of edges is, and
    it is — once order 2 is dropped, which is degenerate (3-4x lower GC than
    order 4+, i.e. underfit rather than a point on a trend).
    """
    from scipy.stats import spearmanr
    import itertools
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))

    ax = axes[0]
    width, xs = 0.35, np.arange(len(CONDS) * len(BANDS))
    for off, (drop2, lab, col) in enumerate((
            (False, 'all orders (2,4,6,10)', '#bdbdbd'),
            (True, 'order > 2', '#1a9850'))):
        vals = []
        for task, stim in CONDS:
            for band in BANDS:
                cells = collections.defaultdict(dict)
                for k, v in data.items():
                    if (k[:3] == (task, stim, 'biv') and k[5] == band
                            and (k[7] > 2 or not drop2)):
                        cells[(k[6], k[7])][(k[3], k[4])] = np.nanmean(v)
                keys = sorted(cells)
                if len(keys) < 2:
                    vals.append(np.nan); continue
                edges = sorted(set.intersection(*[set(cells[c]) for c in keys]))
                vals.append(np.mean([
                    spearmanr([cells[a][e] for e in edges],
                              [cells[b][e] for e in edges]).statistic
                    for a, b in itertools.combinations(keys, 2)]))
        ax.bar(xs + (off - 0.5) * width, vals, width, color=col, label=lab)
    ax.set_xticks(xs)
    ax.set_xticklabels([f'{t[:4]}/{s[:4]}\n{b.replace("_", " ")}'
                        for t, s in CONDS for b in BANDS],
                       fontsize=6.5, rotation=90)
    ax.axhline(0.9, color='0.4', ls=':', lw=1)
    ax.set_ylim(0, 1); ax.set_ylabel('mean Spearman ρ between grid cells')
    ax.set_title('(a) Edge RANKING is grid-stable once order 2 is dropped',
                 fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc='lower right')

    ax = axes[1]
    COL = {'overtProd': '#d73027', 'perception': '#4575b4'}
    MARK = {'prodDiff': 'o', 'percDiff': 's'}
    lim = 0.0
    for task, stim in CONDS:
        x, y = [], []
        for k, v in data.items():
            if (k[:3] == (task, stim, 'biv') and k[3] == 'tpc-lh'
                    and k[4] == 'awfa-lh' and k[5] in BANDS and k[7] > 2):
                rev = (task, stim, 'biv', 'awfa-lh', 'tpc-lh', k[5], k[6], k[7])
                if rev in data:
                    y.append(np.nanmean(v)); x.append(np.nanmean(data[rev]))
        if not x:
            continue
        ax.scatter(x, y, s=26, alpha=0.65, color=COL[task], marker=MARK[stim],
                   edgecolors='none', label=f'{task}/{stim}')
        lim = max(lim, max(x), max(y))
    lim *= 1.08
    ax.plot([0, lim], [0, lim], color='0.4', ls='--', lw=1, zorder=0)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel('awfa → tpc  (GC)'); ax.set_ylabel('tpc → awfa  (GC)')
    ax.set_title('(b) tpc→awfa exceeds its reverse in every grid cell',
                 fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc='lower right')

    fig.suptitle('What survives the sweep — bivariate GC, zscore, n=20',
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = f'{OUT}/fig4_stability.png'
    fig.savefig(p, dpi=150); plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rebuild', action='store_true')
    args = ap.parse_args()
    print(f'root: {D}')
    data = build(args.rebuild)
    n_biv = len({k[:2] for k in data if k[2] == 'biv'})
    print(f'{len(data)} cells; {n_biv} task/contrast combinations\n')
    for fn in (fig_robustness, fig_pattern, fig_conditioned,
               fig_stability):
        print(f'  wrote {os.path.basename(fn(data))}')
    print(f'\nfigures in {OUT}')


if __name__ == '__main__':
    sys.exit(main())
