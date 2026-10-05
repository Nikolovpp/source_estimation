#!/usr/bin/env python
"""Window-length sweep: the same edge at each sliding-window length, overlaid.

For every task x contrast x ROI pair this draws one figure:

    rows     the three tested series   roi_i -> roi_j GC
                                       roi_j -> roi_i GC
                                       net Diff-TRGC (positive = i -> j)
    columns  the four report bands (theta, low beta, high beta, combined beta)
    lines    one per window length (``--wins``), subject mean +/- SEM

Everything comes from the ``granger_stats.py`` CSVs (run
``exploratory/run_gc_stats_figs.sh`` first), so the figure shows exactly what
was tested:

  * the y-axis is the CHANGE FROM BASELINE (mean minus that run's
    ``baseline_mean``).  Raw GC cannot be overlaid usefully: the finite-sample
    floor grows as the window shrinks, so a 40 ms run sits ~2x higher than an
    80 ms run at every time point.  The raw baseline levels are printed in
    each panel instead.
  * the x-axis is the window CENTER (window start + length / 2), so an event
    lands at the same x for every window length.  The stats themselves are
    defined on window starts; the bars below follow each window's own axis.
  * ABOVE the traces, per window length (in its color): asterisks = the
    pointwise one-sample t-test at p < 0.05, uncorrected (GC right-tailed,
    Diff-TRGC two-tailed), and under them a thin bar = windows inside an
    FWER-significant permutation cluster (per edge x band, no correction
    across cells).  A run of significant windows gets evenly spaced
    asterisks rather than one per 2 ms window.
  * under the traces, gray bars = each run's baseline windows.
  * only baseline start .. task end is drawn; the leading-edge transient and
    the trailing windows are outside the tested range and would set the scale.

It also writes one overview figure and the table behind it: for every edge x
band cell and window length, the share of windows whose CENTER falls in
``--overview-range`` (default 150..300 ms) that pass the pointwise t-test.

    conda activate mne
    python exploratory/plot_gc_window_sweep.py                   # order 15, 40/60/80 ms
    python exploratory/plot_gc_window_sweep.py --stats-subdir group_stats
    python exploratory/plot_gc_window_sweep.py --tasks overtProd --pairs awfa-lh,ifc-lh
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_granger import GC_OUTPUT_ROOT, gc_tag, roiset_tag          # noqa: E402

BANDS = ['theta', 'low_beta', 'high_beta', 'beta']
BAND_LABEL = {'theta': 'theta 4–8 Hz', 'low_beta': 'low beta 12–18 Hz',
              'high_beta': 'high beta 18–30 Hz', 'beta': 'beta 12–30 Hz'}
PAIRS = ['awfa-lh,ifc-lh', 'awfa-lh,pmc-lh', 'awfa-lh,tpc-lh',
         'ifc-lh,pmc-lh', 'ifc-lh,tpc-lh', 'pmc-lh,tpc-lh']
# One fixed hue per window length, in sweep order (never re-assigned).
WIN_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
INK, MUTED, GRID, EMPTY = '#1f1f1e', '#6b6a63', '#e7e6e0', '#eeede8'
CSV = 'gc_task_vs_baseline_stats_ttest.csv'


def short(roi):
    return roi.replace('-lh', '')


def pair_dir(args, task, stim, pair, win):
    tag = gc_tag(args.order, win, args.target_fs, args.normalize,
                 n_pcs=args.n_pcs, n_pcs_roi=args.n_pcs_roi)
    return (GC_OUTPUT_ROOT / task / args.method / args.atlas
            / args.feature_mode / 'leakage_corrected' / tag
            / roiset_tag(pair.split(',')) / stim)


def baseline_ms(args, task, t0):
    """Baseline (window-start ms) the stats in ``--stats-subdir`` used.

    Mirrors run_gc_stats_figs.sh: ``group_stats`` is granger_stats.py's
    default, epoch start + [edge guard, 100] ms; any other subdir is the
    explicit one (overtProd: ``--baseline-dur`` ms right after the edge guard,
    perception: -80..0 ms), unless given on the command line.
    """
    given = {'overtProd': args.baseline_overtprod,
             'perception': args.baseline_perception}[task]
    if given is not None:
        return 1000.0 * given[0], 1000.0 * given[1]
    if args.stats_subdir == 'group_stats':
        return t0 + args.edge_guard, t0 + 100.0
    if task == 'overtProd':
        return t0 + args.edge_guard, t0 + args.edge_guard + args.baseline_dur
    return -80.0, 0.0


def load_series(args, task, stim, pair):
    """{win: {(measure, src, tgt, band): DataFrame sorted by window_ms}}."""
    out = {}
    for win in args.wins:
        path = os.path.join(pair_dir(args, task, stim, pair, win),
                            args.stats_subdir, CSV)
        if not os.path.exists(path):
            print(f'  missing {path}')
            continue
        df = pd.read_csv(path)
        out[win] = {k: g.sort_values('window_ms').reset_index(drop=True)
                    for k, g in df.groupby(['measure', 'src', 'tgt', 'band'])}
    return out


def spans(mask, x):
    """[(x_first, x_last), ...] for each run of True in ``mask``."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    cuts = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[cuts + 1]]
    ends = np.r_[idx[cuts], idx[-1]]
    return [(x[a], x[b]) for a, b in zip(starts, ends)]


def star_x(mask, x, min_dx):
    """Asterisk positions for the True runs of ``mask``: one per run, more
    for a long run, never closer than ``min_dx`` so they stay distinct."""
    out = []
    for a, b in spans(mask, x):
        n = int((b - a) // min_dx) + 1
        out += [0.5 * (a + b)] if n == 1 else list(np.linspace(a, b, n))
    return out


def plot_pair(args, task, stim, pair, series, out_dir, sfx):
    ri, rj = pair.split(',')
    rows = [('gc', ri, rj, f'{short(ri)} → {short(rj)}', 'ΔGC'),
            ('gc', rj, ri, f'{short(rj)} → {short(ri)}', 'ΔGC'),
            ('dtrgc', ri, rj,
             f'net {short(ri)} ↔ {short(rj)}\n(+ = {short(ri)} → {short(rj)})',
             'ΔTRGC')]
    wins = [w for w in args.wins if w in series]
    color = {w: WIN_COLORS[args.wins.index(w)] for w in wins}
    fig, axes = plt.subplots(len(rows), len(BANDS), figsize=(17, 9.6),
                             sharex=True, sharey='row', squeeze=False)
    n_subj, bl_note, x_lo, x_hi = 0, {}, np.inf, -np.inf
    for r, (measure, src, tgt, row_lbl, unit) in enumerate(rows):
        for c, band in enumerate(BANDS):
            ax = axes[r, c]
            marks, base_txt = [], []
            for w in wins:
                g = series[w].get((measure, src, tgt, band))
                if g is None:
                    continue
                wm = g['window_ms'].to_numpy()
                bl = baseline_ms(args, task, float(wm[0]))
                bmask = (wm >= bl[0]) & (wm <= bl[1])
                tested = g['pval'].notna().to_numpy()
                base = float(g['baseline_mean'].iloc[0])
                # the stats' baseline_mean must be the mean over our mask, or
                # the baseline drawn here is not the one that was tested
                ok = bmask.any() and np.isclose(
                    g['gc_mean'].to_numpy()[bmask].mean(), base,
                    rtol=1e-6, atol=1e-9)
                bl_note[w] = bl if ok else None
                show = (wm >= bl[0]) if ok else tested
                show &= wm <= wm[tested].max()
                x = wm + w / 2.0                         # window center
                m = g['gc_mean'].to_numpy() - base
                se = g['gc_sem'].to_numpy()
                ax.fill_between(x[show], (m - se)[show], (m + se)[show],
                                color=color[w], alpha=0.14, lw=0)
                ax.plot(x[show], m[show], color=color[w], lw=1.8)
                x_lo, x_hi = min(x_lo, x[show][0]), max(x_hi, x[show][-1])
                n_subj = max(n_subj, int(g['n_subj_perm'].max()))
                marks.append((w, x, bmask if ok else None,
                              g['sig_cluster'].to_numpy(bool),
                              g['sig'].to_numpy(bool)))
                base_txt.append(f'{base:.3f}')
            ax.axhline(0, color=MUTED, lw=0.8)
            ax.axvline(0, color=INK, lw=0.8, alpha=0.55)
            ax.grid(axis='y', color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            ax.spines[['top', 'right']].set_visible(False)
            ax.tick_params(colors=MUTED, labelsize=9)
            ax.text(0.99, 0.012, 'baseline ' + ' / '.join(base_txt),
                    transform=ax.transAxes, ha='right', va='bottom',
                    fontsize=8, color=MUTED)
            ax._sweep_marks = marks
            if r == 0:
                ax.set_title(BAND_LABEL[band], fontsize=11, loc='left',
                             color=INK)
            if c == 0:
                ax.set_ylabel(f'{row_lbl}\n{unit} vs baseline', fontsize=10,
                              color=INK)
            if r == len(rows) - 1:
                ax.set_xlabel('window center (ms)', fontsize=10, color=INK)
    # Stats go ABOVE the traces: per window length a row of asterisks
    # (pointwise t-test) with its thin cluster bar right under it, the
    # shortest window on top.  The baseline bars stay under the traces.
    min_dx = 0.016 * (x_hi - x_lo)
    for r in range(len(rows)):
        y0, y1 = axes[r, 0].get_ylim()
        h = 0.04 * (y1 - y0)
        axes[r, 0].set_ylim(y0 - (0.7 * len(wins) + 1.6) * h,
                            y1 + (2 * len(wins) + 0.6) * h)
        for c in range(len(BANDS)):
            ax = axes[r, c]
            for k, (w, x, bmask, sig_cl, sig_pt) in enumerate(ax._sweep_marks):
                yb = y0 - (0.7 * k + 0.7) * h
                if bmask is not None:
                    for a, b in spans(bmask, x):
                        ax.plot([a, b], [yb, yb], color='0.62', lw=3,
                                solid_capstyle='butt')
                top = y1 + 2 * (len(wins) - k) * h
                xs = star_x(sig_pt, x, min_dx)
                ax.plot(xs, [top] * len(xs), ls='none', marker=(6, 2, 0),
                        ms=5.5, mew=0.9, color=color[w])
                for a, b in spans(sig_cl, x):
                    ax.plot([a, max(b, a + 2)], [top - 0.85 * h] * 2,
                            color=color[w], lw=2, solid_capstyle='butt')
    axes[0, 0].set_xlim(x_lo, x_hi)

    handles = [Line2D([], [], color=color[w], lw=2.2, label=f'{w:g} ms window')
               for w in wins]
    handles += [Line2D([], [], ls='none', marker=(6, 2, 0), ms=7, mew=1.1,
                       color=INK, label='pointwise t-test p < 0.05 '
                                        '(uncorrected)'),
                Line2D([], [], color=INK, lw=2,
                       label='significant cluster (permutation)'),
                Line2D([], [], color='0.62', lw=3, label='baseline windows')]
    fig.legend(handles=handles, loc='upper center', ncol=len(handles),
               frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.925))
    over = ''.join(f', {o.split("=")[0].replace("-lh", "")} {o.split("=")[1]}'
                   for o in (args.n_pcs_roi or []))
    bl_txt = ' · '.join(
        f'{w:g} ms: ' + ('not reproduced' if bl_note.get(w) is None
                         else f'{bl_note[w][0]:g}..{bl_note[w][1]:g}')
        for w in wins)
    fig.suptitle(
        f'{short(ri)}–{short(rj)}   {task} / {stim}   window sweep at order '
        f'{args.order} @ {args.target_fs:g} Hz   ({args.normalize}, '
        f'{args.n_pcs} PCs{over}, n={n_subj})',
        fontsize=13, color=INK, y=0.992)
    fig.text(0.5, 0.948,
             'subject mean ± SEM, change from each run\'s baseline · stats '
             'in the window color: GC one-tailed (task > baseline), ΔTRGC '
             'two-tailed\n'
             f'baseline window starts (ms)  {bl_txt} · '
             '"baseline a / b / c" = raw baseline level per window length',
             ha='center', va='center', fontsize=9, color=MUTED,
             linespacing=1.5)
    fig.tight_layout(rect=(0, 0, 1, 0.895))
    out = os.path.join(out_dir, f'{task}_{stim}_{short(ri)}+{short(rj)}'
                                f'_window_sweep{sfx}.{args.format}')
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def summarize(args, task, stim, pair, series):
    """One row per tested series x band, with each window's cluster result."""
    ri, rj = pair.split(',')
    rows = []
    for measure, src, tgt in (('gc', ri, rj), ('gc', rj, ri),
                              ('dtrgc', ri, rj)):
        for band in BANDS:
            row = {'task': task, 'stim': stim, 'measure': measure,
                   'src': src, 'tgt': tgt, 'band': band}
            for w in args.wins:
                g = series.get(w, {}).get((measure, src, tgt, band))
                if g is None:
                    continue
                sig = g['sig_cluster'].to_numpy(bool)
                d = g['gc_mean'].to_numpy() - float(g['baseline_mean'].iloc[0])
                row[f'p_cluster_min_{w:g}'] = float(g['p_cluster_min'].iloc[0])
                row[f'n_sig_cluster_{w:g}'] = int(sig.sum())
                row[f'n_sig_pointwise_{w:g}'] = int(g['sig'].sum())
                # pointwise test inside the overview range (window centers)
                xc = g['window_ms'].to_numpy() + w / 2.0
                rng = ((xc >= args.overview_range[0])
                       & (xc <= args.overview_range[1])
                       & g['pval'].notna().to_numpy())
                hit = rng & g['sig'].to_numpy(bool)
                row[f'n_win_range_{w:g}'] = int(rng.sum())
                row[f'n_sig_pointwise_range_{w:g}'] = int(hit.sum())
                row[f'frac_sig_pointwise_range_{w:g}'] = (
                    hit.sum() / rng.sum() if rng.any() else np.nan)
                row[f'range_mean_delta_{w:g}'] = (float(d[hit].mean())
                                                  if hit.any() else np.nan)
                row[f'baseline_mean_{w:g}'] = float(g['baseline_mean'].iloc[0])
                # sign of the effect inside the significant windows
                row[f'sig_mean_delta_{w:g}'] = (float(d[sig].mean())
                                                if sig.any() else np.nan)
            row['n_windows_sig'] = sum(
                row.get(f'n_sig_cluster_{w:g}', 0) > 0 for w in args.wins)
            rows.append(row)
    return rows


def plot_overview(args, S, out_dir, tag, sfx):
    """Tiles: pointwise t-test inside ``--overview-range``, per cell and
    window length.  Number = % of the windows in the range with p < 0.05
    (uncorrected); the fill darkens with it.  Diff-TRGC is two-tailed, so its
    number carries the sign of the effect (+ = net src -> tgt)."""
    lo, hi = args.overview_range
    tasks = [t for t in args.tasks if (S.task == t).any()]
    stims = [s for s in args.stims if (S.stim == s).any()]
    nw = len(args.wins)
    fig, axes = plt.subplots(len(tasks), len(stims), squeeze=False,
                             figsize=(max(13.0, 7.6 * len(stims)),
                                      1.6 + 6.2 * len(tasks)))
    for a, task in enumerate(tasks):
        for b, stim in enumerate(stims):
            ax = axes[a, b]
            sub = S[(S.task == task) & (S.stim == stim)]
            edges = list(dict.fromkeys(
                zip(sub.measure, sub.src, sub.tgt)))
            for y, (measure, src, tgt) in enumerate(edges):
                for c, band in enumerate(BANDS):
                    cell = sub[(sub.measure == measure) & (sub.src == src)
                               & (sub.tgt == tgt) & (sub.band == band)]
                    for k, w in enumerate(args.wins):
                        col = f'frac_sig_pointwise_range_{w:g}'
                        f = float(cell[col].iloc[0]) if col in cell else 0.0
                        f = 0.0 if np.isnan(f) else f
                        x0 = c * (nw + 0.6) + k
                        ax.add_patch(Rectangle(
                            (x0 + 0.04, y + 0.06), 0.92, 0.88, lw=0,
                            color=WIN_COLORS[k] if f > 0 else EMPTY,
                            alpha=0.22 + 0.78 * f if f > 0 else 1.0))
                        if f > 0:
                            txt = f'{max(1, round(100 * f)):d}'
                            if measure == 'dtrgc':
                                d = cell[f'range_mean_delta_{w:g}'].iloc[0]
                                txt = ('+' if d > 0 else '−') + txt
                            ax.text(x0 + 0.5, y + 0.5, txt, ha='center',
                                    va='center', fontsize=7.5,
                                    color='white' if f > 0.55 else INK)
                if y % 3 == 0 and y:
                    ax.axhline(y, color=MUTED, lw=0.5)
            ax.set_xlim(0, len(BANDS) * (nw + 0.6) - 0.6)
            ax.set_ylim(len(edges), 0)
            ax.set_yticks(np.arange(len(edges)) + 0.5)
            ax.set_yticklabels(
                [f'{short(s)} → {short(t)}' if m == 'gc'
                 else f'{short(s)} ↔ {short(t)} ΔTRGC'
                 for m, s, t in edges], fontsize=9, color=INK)
            ax.set_xticks([c * (nw + 0.6) + nw / 2 for c in range(len(BANDS))])
            ax.set_xticklabels([BAND_LABEL[x].replace(' Hz', '\nHz', 1)
                                .replace(' 1', '\n1', 1).replace(' 4', '\n4', 1)
                                for x in BANDS], fontsize=9, color=INK)
            ax.xaxis.tick_top()
            ax.tick_params(length=0)
            for s in ax.spines.values():
                s.set_visible(False)
            hits = ', '.join(
                f'{w:g} ms '
                f'{int((sub[f"n_sig_pointwise_range_{w:g}"] > 0).sum())}'
                for w in args.wins
                if f'n_sig_pointwise_range_{w:g}' in sub)
            ax.set_title(f'{task} / {stim}\ncells with any significant '
                         f'window: {hits} of {len(sub)}', fontsize=10.5,
                         loc='left', color=INK, pad=40)
    handles = [Rectangle((0, 0), 1, 1, color=WIN_COLORS[k],
                         label=f'{w:g} ms window')
               for k, w in enumerate(args.wins)]
    handles.append(Rectangle((0, 0), 1, 1, color=EMPTY,
                             label='no significant window'))
    fig.legend(handles=handles, loc='upper center', ncol=len(handles),
               frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle(
        f'Pointwise t-test in {lo:g}–{hi:g} ms (window center), by window '
        f'length   order {args.order} @ {args.target_fs:g} Hz, '
        f'{args.stats_subdir}', fontsize=13, color=INK, y=0.99)
    fig.text(0.5, 0.925,
             f'number = % of the windows in {lo:g}–{hi:g} ms with p < 0.05, '
             'uncorrected (darker = more; ≈5% expected by chance) · GC '
             'one-tailed (task > baseline) · ΔTRGC two-tailed, sign = '
             'direction (+ = first → second ROI)',
             ha='center', fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.915), h_pad=2.5)
    out = os.path.join(out_dir, f'window_sweep_overview_{tag}{sfx}.'
                                f'{args.format}')
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--order', type=int, default=15)
    p.add_argument('--wins', type=float, nargs='+', default=[40, 60, 80],
                   help='window lengths (ms) to overlay, at most '
                        f'{len(WIN_COLORS)}')
    p.add_argument('--target-fs', type=float, default=500.0)
    p.add_argument('--normalize', default='none')
    p.add_argument('--n-pcs', type=int, default=2)
    p.add_argument('--n-pcs-roi', nargs='*', default=['pmc-lh=3'],
                   metavar='ROI=K')
    p.add_argument('--method', default='LCMV')
    p.add_argument('--atlas', default='custom')
    p.add_argument('--feature-mode', default='vertex_selectkbest')
    p.add_argument('--tasks', nargs='+', default=['overtProd', 'perception'])
    p.add_argument('--stims', nargs='+', default=['prodDiff', 'percDiff'])
    p.add_argument('--pairs', nargs='+', default=PAIRS, metavar='ROI,ROI')
    p.add_argument('--stats-subdir', default='group_stats_bl',
                   help='granger_stats.py output dir under each pair dir; '
                        'group_stats = default leading baseline')
    p.add_argument('--edge-guard', type=float, default=30.0,
                   help='ms; as passed to run_gc_stats_figs.sh')
    p.add_argument('--baseline-dur', type=float, default=100.0,
                   help='ms; run_gc_stats_figs.sh BL_DUR_MS')
    p.add_argument('--baseline-overtprod', type=float, nargs=2, default=None,
                   metavar=('START', 'END'),
                   help='s; only if the stats used an explicit BL_OVERTPROD')
    p.add_argument('--baseline-perception', type=float, nargs=2, default=None,
                   metavar=('START', 'END'),
                   help='s; only if the stats used a non-default '
                        'BL_PERCEPTION')
    p.add_argument('--overview-range', type=float, nargs=2,
                   default=[150.0, 300.0], metavar=('START', 'END'),
                   help='ms, window CENTER; the overview figure reports the '
                        'pointwise t-test inside this range')
    p.add_argument('--format', default='png', choices=['png', 'svg'])
    p.add_argument('--out-dir', default=None)
    args = p.parse_args()
    if len(args.wins) > len(WIN_COLORS):
        p.error(f'at most {len(WIN_COLORS)} window lengths')
    args.n_pcs_roi = args.n_pcs_roi or None

    # run label = the gc_tag without its window segment
    tag = gc_tag(args.order, 0, args.target_fs, args.normalize,
                 n_pcs=args.n_pcs, n_pcs_roi=args.n_pcs_roi).replace(
                     '_win0ms', '')
    sfx = ('' if args.stats_subdir == 'group_stats'
           else '_' + args.stats_subdir.replace('group_stats_', ''))
    out_dir = args.out_dir or os.path.join(
        GC_OUTPUT_ROOT, '_figures_final_review', f'window_sweep_{tag}{sfx}')
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for task in args.tasks:
        for stim in args.stims:
            for pair in args.pairs:
                series = load_series(args, task, stim, pair)
                if not series:
                    continue
                print('wrote', plot_pair(args, task, stim, pair, series,
                                         out_dir, sfx))
                rows += summarize(args, task, stim, pair, series)
    if not rows:
        raise SystemExit('no stats CSVs found — run '
                         'exploratory/run_gc_stats_figs.sh first')
    S = pd.DataFrame(rows)
    csv = os.path.join(out_dir, f'window_sweep_summary_{tag}{sfx}.csv')
    S.to_csv(csv, index=False)
    print('wrote', csv)
    print('wrote', plot_overview(args, S, out_dir, tag, sfx))

    n = len(S)
    lo, hi = args.overview_range
    print(f'\n{n} edge x band cells ({args.stats_subdir}); pointwise t-test, '
          f'window centers {lo:g}..{hi:g} ms (about 5% of windows expected '
          'by chance)')
    fr = [f'frac_sig_pointwise_range_{w:g}' for w in args.wins
          if f'frac_sig_pointwise_range_{w:g}' in S]
    for w in args.wins:
        col = f'frac_sig_pointwise_range_{w:g}'
        if col in S:
            print(f'  {w:g} ms: {int((S[col] > 0).sum())} cells with any '
                  f'significant window, {int((S[col] >= 0.5).sum())} with '
                  f'half or more; mean share {100 * S[col].mean():.1f}%')
    top = S[(S[fr] >= 0.25).all(axis=1)].copy()
    if len(top):
        top[fr] = (100 * top[fr]).round(0)
        print('\ncells with >= 25% of the range significant at every window '
              'length (% of windows):')
        print(top[['task', 'stim', 'measure', 'src', 'tgt', 'band'] + fr]
              .rename(columns={c: c.replace('frac_sig_pointwise_range_', '')
                               + ' ms' for c in fr}).to_string(index=False))


if __name__ == '__main__':
    main()
