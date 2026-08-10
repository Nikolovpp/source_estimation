#!/usr/bin/env python3
"""Figures for the parametric-vs-state-space GC comparison, all 20 subjects.

Built to be *read*, not to argue: every panel shows both estimators, the three
bands of interest (theta, low beta, high beta) are foregrounded, and nothing is
pre-selected for significance. Group mean +/- SEM across subjects throughout.

    conda activate mne && python plot_granger_routes.py

Writes into <GC_routes>/_figures/.
"""
from __future__ import annotations

import os
import sys
import itertools

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from granger_routes_stats import load_dir, bh_fdr
from scipy.stats import ttest_rel

ROOT = ('/mnt/r/phd_thesis/Research/SpeechProduction/EEG/derivatives/'
        'source_estimation/GC_routes')
OUT = os.path.join(ROOT, '_figures')

KEY_BANDS = ['theta', 'low_beta', 'high_beta']       # the ones that matter here
ALL_BANDS = ['theta', 'alpha', 'low_beta', 'high_beta', 'upper']
BAND_LAB = {'theta': 'theta 4-8', 'alpha': 'alpha 8-12',
            'low_beta': 'low beta 12-18', 'high_beta': 'high beta 18-30',
            'upper': 'upper 30-38'}
PRIMARY_EDGES = [('awfa-lh', 'tpc-lh'), ('tpc-lh', 'ifc-lh'),
                 ('ifc-lh', 'pmc-lh'), ('awfa-lh', 'ifc-lh'),
                 ('tpc-lh', 'pmc-lh'), ('pmc-lh', 'tpc-lh')]

sns.set_theme(context='notebook', style='white', font_scale=0.95,
              rc={'figure.facecolor': 'white', 'axes.titlepad': 8})
_P = sns.color_palette('colorblind')
PAR, SS = _P[3], _P[0]           # parametric (orange-red), state-space (blue)
INK, MUTED = '#222222', _P[7]
BAND_C = dict(zip(KEY_BANDS, sns.color_palette('crest', 3)))


def cfg(task, method, tag, sub, stim):
    return os.path.join(ROOT, task, method, 'custom', tag, sub, stim)


def band_idx(bands, name):
    return bands.index(name) if name in bands else None


def gm(arr):
    """group mean, SEM over subjects (axis 0), NaN-safe."""
    n = np.sum(np.isfinite(arr), axis=0)
    m = np.nanmean(arr, axis=0)
    s = np.nanstd(arr, axis=0) / np.sqrt(np.maximum(n, 1))
    return m, s


def _style(ax, xlab=None, ylab=None, title=None):
    if title: ax.set_title(title, fontsize=10, loc='left', color=INK)
    if xlab: ax.set_xlabel(xlab, fontsize=9)
    if ylab: ax.set_ylabel(ylab, fontsize=9)
    sns.despine(ax=ax)
    ax.grid(axis='y', color=MUTED, alpha=0.22, lw=0.6)
    ax.set_axisbelow(True)


# ══════════════════════════════════════════════════════════════════════
# 1. GC time courses, per edge, both arms, three bands
# ══════════════════════════════════════════════════════════════════════
def fig_timecourses(task, stim, method='dSPM', tag='A_canonical',
                    sub='win60ms_order6_fs200_pc1'):
    d = cfg(task, method, tag, sub, stim)
    if not os.path.isdir(d):
        return
    subs, bands, win, store = load_dir(d)
    edges = [e for e in PRIMARY_EDGES
             if f'm1_par__{e[0]}__{e[1]}' in store]
    if not edges:
        return
    fig, axes = plt.subplots(len(edges), len(KEY_BANDS),
                             figsize=(3.6 * len(KEY_BANDS), 2.0 * len(edges)),
                             sharex=True, squeeze=False)
    for r, (a, b) in enumerate(edges):
        for c, bd in enumerate(KEY_BANDS):
            ax = axes[r][c]
            bi = band_idx(bands, bd)
            if bi is None:
                continue
            for key, col, lab in ((f'm1_par__{a}__{b}', PAR, 'parametric'),
                                  (f'm1_ss__{a}__{b}', SS, 'state-space')):
                m, s = gm(store[key][:, bi])
                ax.plot(win, m, color=col, lw=1.5, zorder=3, label=lab)
                ax.fill_between(win, m - s, m + s, color=col, alpha=0.18, lw=0)
            ax.axvline(0, color=INK, lw=0.9, ls=(0, (3, 3)), zorder=1)
            if r == 0:
                ax.set_title(BAND_LAB[bd], fontsize=10, color=INK)
            if c == 0:
                ax.set_ylabel(f'{a.replace("-lh","")}→{b.replace("-lh","")}',
                              fontsize=9)
            if r == len(edges) - 1:
                ax.set_xlabel('window centre (ms)', fontsize=9)
            if r == 0 and c == 0:
                ax.legend(fontsize=7.5, frameon=False, loc='upper left')
            sns.despine(ax=ax); ax.grid(axis='y', color=MUTED, alpha=0.2, lw=0.6)
            ax.set_axisbelow(True)
    fig.suptitle(f'Conditional GC over time — {task} / {stim} / {method} '
                 f'({len(subs)} subjects, mean ± SEM). 0 ms = speech onset.',
                 fontsize=11, y=1.005)
    fig.tight_layout()
    p = os.path.join(OUT, f'fig1_timecourses_{task}_{stim}_{method}.png')
    fig.savefig(p, dpi=170, bbox_inches='tight', facecolor='white')
    plt.close(fig); print('  ', os.path.basename(p))


# ══════════════════════════════════════════════════════════════════════
# 2. Arm contrast — parametric vs state-space, every edge x band
# ══════════════════════════════════════════════════════════════════════
def fig_arm_contrast(task, stim, method='dSPM', tag='A_canonical',
                     sub='win60ms_order6_fs200_pc1'):
    d = cfg(task, method, tag, sub, stim)
    if not os.path.isdir(d):
        return
    subs, bands, win, store = load_dir(d)
    meas = [('m1', 'conditional spectral'), ('m2', 'pairwise time-domain')]
    fig, axes = plt.subplots(1, len(meas), figsize=(5.4 * len(meas), 5.0))
    for ax, (mk, mlab) in zip(np.atleast_1d(axes), meas):
        xs, ys, cs = [], [], []
        for key in sorted(k for k in store if k.startswith(f'{mk}_par__')):
            edge = key.split('__', 1)[1]
            ssk = f'{mk}_ss__{edge}'
            if ssk not in store:
                continue
            for bd in KEY_BANDS:
                bi = band_idx(bands, bd)
                if bi is None:
                    continue
                xs.append(np.nanmean(store[key][:, bi]))
                ys.append(np.nanmean(store[ssk][:, bi]))
                cs.append(BAND_C[bd])
        if not xs:
            continue
        xs, ys = np.array(xs), np.array(ys)
        lim = float(np.nanmax([xs.max(), ys.max()])) * 1.08
        ax.plot([0, lim], [0, lim], color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
        ax.scatter(xs, ys, s=22, c=cs, alpha=0.75, lw=0, zorder=3)
        med = float(np.nanmedian(xs / np.where(ys == 0, np.nan, ys)))
        ax.text(0.04, 0.95, f'{mlab}\nmedian parametric/state-space = {med:.3f}\n'
                            f'{len(xs)} edge × band cells, {len(subs)} subjects',
                transform=ax.transAxes, va='top', fontsize=9, color=INK)
        for bd in KEY_BANDS:
            ax.scatter([], [], s=22, color=BAND_C[bd], label=BAND_LAB[bd])
        ax.legend(fontsize=8, frameon=False, loc='lower right')
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_aspect('equal', adjustable='box')
        _style(ax, 'parametric', 'state-space', mlab)
    fig.suptitle(f'Parametric vs state-space — {task} / {stim} / {method}. '
                 'Points on the dashed line mean the two agree.',
                 fontsize=11, y=1.01)
    fig.tight_layout()
    p = os.path.join(OUT, f'fig2_armcontrast_{task}_{stim}_{method}.png')
    fig.savefig(p, dpi=170, bbox_inches='tight', facecolor='white')
    plt.close(fig); print('  ', os.path.basename(p))


# ══════════════════════════════════════════════════════════════════════
# 3/4. Sweeps — GC and arm ratio against order or window
# ══════════════════════════════════════════════════════════════════════
def _sweep(task, stim, method, tag, subdirs, xvals, xlabel, fname, span=None):
    rows = []
    for sd, xv in zip(subdirs, xvals):
        d = cfg(task, method, tag, sd, stim)
        if not os.path.isdir(d):
            continue
        subs, bands, win, store = load_dir(d)
        sl = slice(None)
        if span:
            i = np.where((win >= span[0]) & (win <= span[1]))[0]
            if i.size: sl = slice(int(i[0]), int(i[-1]) + 1)
        for bd in KEY_BANDS:
            bi = band_idx(bands, bd)
            if bi is None:
                continue
            for arm, pre in (('parametric', 'm1_par'), ('state-space', 'm1_ss')):
                vals, nanf = [], []
                for k in (k for k in store if k.startswith(pre + '__')):
                    a = store[k][:, bi, sl]
                    vals.append(np.nanmean(a, axis=1))
                    nanf.append(np.isnan(a).mean())
                if not vals:
                    continue
                v = np.nanmean(np.stack(vals), axis=0)     # (n_subj,)
                rows.append(dict(x=xv, band=bd, arm=arm,
                                 mean=float(np.nanmean(v)),
                                 sem=float(np.nanstd(v) / np.sqrt(len(v))),
                                 nan_frac=float(np.mean(nanf)),
                                 n=len(subs)))
    if not rows:
        return
    import pandas as pd
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(2, len(KEY_BANDS), figsize=(3.7 * len(KEY_BANDS), 6.0),
                             sharex=True, squeeze=False)
    for c, bd in enumerate(KEY_BANDS):
        ax = axes[0][c]
        sub = df[df.band == bd]
        for arm, col in (('parametric', PAR), ('state-space', SS)):
            s = sub[sub.arm == arm].sort_values('x')
            if s.empty: continue
            ax.errorbar(s.x, s['mean'], yerr=s['sem'], color=col, lw=1.8,
                        marker='o', ms=4.5, capsize=3, label=arm, zorder=3)
        ax.set_title(BAND_LAB[bd], fontsize=10, color=INK)
        if c == 0:
            ax.set_ylabel('conditional GC\n(mean over edges ± SEM)', fontsize=9)
            ax.legend(fontsize=8, frameon=False)
        _style(ax)
        # ratio + NaN rate
        ax2 = axes[1][c]
        p_ = sub[sub.arm == 'parametric'].sort_values('x')
        s_ = sub[sub.arm == 'state-space'].sort_values('x')
        if len(p_) and len(s_):
            ax2.plot(p_.x.values, p_['mean'].values / s_['mean'].values,
                     color=INK, lw=1.8, marker='s', ms=4.5, zorder=3)
        ax2.axhline(1.0, color=MUTED, lw=1.0, ls=(0, (4, 3)))
        if sub.nan_frac.max() > 0:
            axn = ax2.twinx()
            axn.plot(p_.x.values, p_.nan_frac.values * 100, color=_P[2],
                     lw=1.2, ls=(0, (2, 2)), marker='^', ms=4)
            axn.set_ylabel('% NaN', fontsize=8, color=_P[2])
            axn.tick_params(labelsize=7, colors=_P[2])
        if c == 0:
            ax2.set_ylabel('parametric ÷ state-space', fontsize=9)
        ax2.set_xlabel(xlabel, fontsize=9)
        _style(ax2)
    fig.suptitle(f'{fname.split("_")[1].title()} sweep — {task} / {stim} / '
                 f'{method}, {df.n.iloc[0]} subjects. '
                 'Top: absolute GC. Bottom: how far the two estimators diverge.',
                 fontsize=11, y=1.005)
    fig.tight_layout()
    p = os.path.join(OUT, f'{fname}_{task}_{stim}_{method}.png')
    fig.savefig(p, dpi=170, bbox_inches='tight', facecolor='white')
    plt.close(fig); print('  ', os.path.basename(p))


# ══════════════════════════════════════════════════════════════════════
# 6. Mediation — the pathway question
# ══════════════════════════════════════════════════════════════════════
def fig_mediation(task, stim, method='dSPM', tag='F_triples',
                  sub='win60ms_order6_fs200_pc1', top=18):
    d = cfg(task, method, tag, sub, stim)
    if not os.path.isdir(d):
        return
    try:
        from run_granger_routes import PRIMARY_TRIPLES
        prim = set(PRIMARY_TRIPLES)
    except Exception:
        prim = set()
    subs, bands, win, store = load_dir(d)
    rows = []
    for arm in ('par', 'ss'):
        for k in (k for k in store if k.startswith(f'f_{arm}__')):
            _, a, b, c = k.split('__')
            pk = f'f_pair__{a}__{c}'
            if pk not in store:
                continue
            for bd in KEY_BANDS:
                bi = band_idx(bands, bd)
                if bi is None:
                    continue
                cd = np.nanmean(store[k][:, bi], axis=1)
                un = np.nanmean(store[pk][:, bi], axis=1)
                ok = np.isfinite(cd) & np.isfinite(un) & (un > 0)
                if ok.sum() < 3:
                    continue
                M = 1.0 - cd[ok] / un[ok]
                t = ttest_rel(un[ok], cd[ok])
                rows.append(dict(arm=arm, a=a, b=b, c=c, band=bd,
                                 M=float(np.mean(M)),
                                 sem=float(np.std(M) / np.sqrt(ok.sum())),
                                 p=float(t.pvalue), n=int(ok.sum()),
                                 primary=(a, b, c) in prim))
    if not rows:
        return
    import pandas as pd
    df = pd.DataFrame(rows)
    for arm in df.arm.unique():
        for fam in (True, False):
            s = (df.arm == arm) & (df.primary == fam)
            if s.any():
                df.loc[s, 'p_fdr'] = bh_fdr(df.loc[s, 'p'].to_numpy())
    df.to_csv(os.path.join(OUT, f'mediation_{task}_{stim}_{method}.csv'),
              index=False)

    fig, axes = plt.subplots(1, len(KEY_BANDS),
                             figsize=(4.6 * len(KEY_BANDS), 6.4), sharex=True)
    for ax, bd in zip(np.atleast_1d(axes), KEY_BANDS):
        sub_p = df[(df.band == bd) & df.primary]
        piv = sub_p.pivot_table(index=['a', 'b', 'c'], columns='arm',
                                values=['M', 'p_fdr'])
        if piv.empty:
            continue
        piv = piv.sort_values(('M', 'ss'))
        lbl = [f'{a.replace("-lh","")}→{c.replace("-lh","")} | {b.replace("-lh","")}'
               for a, b, c in piv.index]
        y = np.arange(len(piv))
        # exploratory background
        expl = df[(df.band == bd) & ~df.primary & (df.arm == 'ss')]['M']
        if len(expl):
            ax.axvspan(np.percentile(expl, 5), np.percentile(expl, 95),
                       color=MUTED, alpha=0.16, zorder=0)
        for arm, col, off in (('par', PAR, -0.16), ('ss', SS, 0.16)):
            if ('M', arm) not in piv:
                continue
            ax.scatter(piv[('M', arm)], y + off, s=34, color=col, zorder=3,
                       label={'par': 'parametric', 'ss': 'state-space'}[arm])
            sig = piv[('p_fdr', arm)] < 0.05
            ax.scatter(piv[('M', arm)][sig], (y + off)[sig.to_numpy()], s=90,
                       facecolors='none', edgecolors=col, lw=1.4, zorder=4)
        ax.axvline(0, color=INK, lw=1.0)
        ax.axvline(1, color=MUTED, lw=1.0, ls=(0, (3, 3)))
        ax.set_yticks(y); ax.set_yticklabels(lbl, fontsize=8)
        ax.set_xlabel('mediation index  M = 1 − F(a→c|b)/F(a→c)', fontsize=9)
        ax.set_title(BAND_LAB[bd], fontsize=10, loc='left', color=INK)
        if bd == KEY_BANDS[0]:
            ax.legend(fontsize=8, frameon=False, loc='lower right')
        sns.despine(ax=ax); ax.grid(axis='x', color=MUTED, alpha=0.22, lw=0.6)
        ax.set_axisbelow(True)
    fig.suptitle(f'Mediation, pre-registered triples — {task} / {stim} / {method}, '
                 f'{len(subs)} subjects.\nM→1 means the mediator explains the '
                 'influence away. Grey band = 5–95% of the 120 exploratory '
                 'triples. Rings = FDR<0.05 within family.',
                 fontsize=11, y=1.02)
    fig.tight_layout()
    p = os.path.join(OUT, f'fig6_mediation_{task}_{stim}_{method}.png')
    fig.savefig(p, dpi=170, bbox_inches='tight', facecolor='white')
    plt.close(fig); print('  ', os.path.basename(p))


# ══════════════════════════════════════════════════════════════════════
def main():
    os.makedirs(OUT, exist_ok=True)
    combos = [('overtProd', 'prodDiff'), ('perception', 'percDiff'),
              ('overtProd', 'percDiff'), ('perception', 'prodDiff')]

    print('fig1 time courses + fig2 arm contrast (A canonical, both inverses)')
    for task, stim in combos:
        for method in ('dSPM', 'LCMV'):
            fig_timecourses(task, stim, method)
            fig_arm_contrast(task, stim, method)

    print('fig3 order sweep')
    for task, stim in [('overtProd', 'prodDiff'), ('perception', 'percDiff')]:
        _sweep(task, stim, 'dSPM', 'C_ordersweep_win60',
               [f'win60ms_order{o}_fs200_pc1' for o in (2, 4, 6, 8)],
               [2, 4, 6, 8], 'model order (60 ms window)', 'fig3_order60')
        _sweep(task, stim, 'dSPM', 'C_ordersweep_win120',
               [f'win120ms_order{o}_fs200_pc1' for o in (4, 8, 12, 16, 20)],
               [4, 8, 12, 16, 20], 'model order (120 ms window)', 'fig3_order120')

    print('fig4 window sweep')
    for task, stim in combos:
        for method in ('dSPM', 'LCMV'):
            _sweep(task, stim, method, 'B_winsweep',
                   [f'win{w}ms_order6_fs200_pc1' for w in (40, 60, 80, 120)],
                   [40, 60, 80, 120], 'window (ms), order 6', 'fig4_window')

    print('fig5 FIXPC1 vs FIXPC4')
    for task, stim in [('overtProd', 'prodDiff'), ('perception', 'percDiff')]:
        _sweep(task, stim, 'dSPM', 'E_fixpc',
               ['win60ms_order6_fs200_pc1', 'win60ms_order6_fs200_pc4'],
               [1, 4], 'PCs per ROI (FIXPC-k)', 'fig5_fixpc')

    print('fig6 mediation')
    for task, stim in [('overtProd', 'prodDiff'), ('perception', 'percDiff')]:
        fig_mediation(task, stim)

    print(f'\nall figures in {OUT}')


if __name__ == '__main__':
    main()


# ══════════════════════════════════════════════════════════════════════
# 7/8. Grid of time courses across a sweep — one row per parameter value.
#      The sweeps above collapse each config to a point; this shows the
#      whole time course at every combination, which is what you need when
#      the summary is non-monotonic.
# ══════════════════════════════════════════════════════════════════════
def fig_sweep_grid(task, stim, method, tag, subdirs, labels, rowlab, fname,
                   edge=('awfa-lh', 'ifc-lh')):
    dirs = [(l, cfg(task, method, tag, sd, stim))
            for l, sd in zip(labels, subdirs)
            if os.path.isdir(cfg(task, method, tag, sd, stim))]
    if not dirs:
        return
    a, b = edge
    fig, axes = plt.subplots(len(dirs), len(KEY_BANDS),
                             figsize=(3.5 * len(KEY_BANDS), 1.85 * len(dirs)),
                             sharex=True, squeeze=False)
    for r, (lab, d) in enumerate(dirs):
        subs, bands, win, store = load_dir(d)
        for c, bd in enumerate(KEY_BANDS):
            ax = axes[r][c]
            bi = band_idx(bands, bd)
            kp, ks = f'm1_par__{a}__{b}', f'm1_ss__{a}__{b}'
            if bi is None or kp not in store:
                ax.axis('off'); continue
            for k, col, ll in ((kp, PAR, 'parametric'), (ks, SS, 'state-space')):
                m, s = gm(store[k][:, bi])
                ax.plot(win, m, color=col, lw=1.4, zorder=3, label=ll)
                ax.fill_between(win, m - s, m + s, color=col, alpha=0.18, lw=0)
            # the divergence, as a number, on every panel
            rp = np.nanmean(store[kp][:, bi]); rs = np.nanmean(store[ks][:, bi])
            ax.text(0.98, 0.92, f'par/ss = {rp/rs:.2f}' if rs else '',
                    transform=ax.transAxes, ha='right', va='top',
                    fontsize=7.5, color=INK)
            ax.axvline(0, color=INK, lw=0.8, ls=(0, (3, 3)), zorder=1)
            if r == 0:
                ax.set_title(BAND_LAB[bd], fontsize=10, color=INK)
            if c == 0:
                ax.set_ylabel(f'{rowlab}\n{lab}', fontsize=8.5)
            if r == len(dirs) - 1:
                ax.set_xlabel('window centre (ms)', fontsize=9)
            if r == 0 and c == 0:
                ax.legend(fontsize=7, frameon=False, loc='upper left')
            sns.despine(ax=ax); ax.grid(axis='y', color=MUTED, alpha=0.2, lw=0.6)
            ax.set_axisbelow(True)
    fig.suptitle(f'{a.replace("-lh","")}→{b.replace("-lh","")} across the '
                 f'{rowlab} sweep — {task} / {stim} / {method}, 20 subjects.\n'
                 'Each panel is the full time course; par/ss is the divergence '
                 'at that setting.', fontsize=11, y=1.005)
    fig.tight_layout()
    p = os.path.join(OUT, f'{fname}_{task}_{stim}_{method}.png')
    fig.savefig(p, dpi=170, bbox_inches='tight', facecolor='white')
    plt.close(fig); print('  ', os.path.basename(p))


# ══════════════════════════════════════════════════════════════════════
# 9. FULL COVERAGE — one multi-page PDF per sweep, one page per directed
#    edge, all bands, both arms, with every subject's trace behind the
#    group mean. Configs are loaded ONCE and reused across all 30 edges.
# ══════════════════════════════════════════════════════════════════════
def pdf_full_coverage(task, stim, method, tag, subdirs, labels, rowlab, fname,
                      bands_wanted=None, measure='m1'):
    from matplotlib.backends.backend_pdf import PdfPages
    bands_wanted = bands_wanted or ALL_BANDS
    loaded = []
    for lab, sd in zip(labels, subdirs):
        d = cfg(task, method, tag, sd, stim)
        if os.path.isdir(d):
            loaded.append((lab,) + load_dir(d))
    if not loaded:
        return
    _, subs0, bands0, _, store0 = loaded[0]
    edges = sorted({k.split('__', 1)[1] for k in store0
                    if k.startswith(f'{measure}_par__')})
    bcols = [b for b in bands_wanted if b in bands0]
    path = os.path.join(OUT, f'{fname}_{task}_{stim}_{method}.pdf')
    with PdfPages(path) as pdf:
        for edge in edges:
            kp, ks = f'{measure}_par__{edge}', f'{measure}_ss__{edge}'
            if kp not in store0:
                continue
            a, b = edge.split('__')
            fig, axes = plt.subplots(len(loaded), len(bcols),
                                     figsize=(3.1 * len(bcols),
                                              1.75 * len(loaded)),
                                     sharex=True, squeeze=False)
            for r, (lab, subs, bands, win, store) in enumerate(loaded):
                for c, bd in enumerate(bcols):
                    ax = axes[r][c]
                    bi = band_idx(bands, bd)
                    if bi is None or kp not in store:
                        ax.axis('off'); continue
                    for k, col in ((kp, PAR), (ks, SS)):
                        arr = store[k][:, bi]                 # (n_subj, n_win)
                        for row in arr:                       # every subject
                            ax.plot(win, row, color=col, lw=0.35, alpha=0.16,
                                    zorder=2)
                        m, s = gm(arr)
                        ax.plot(win, m, color=col, lw=1.6, zorder=4)
                        ax.fill_between(win, m - s, m + s, color=col,
                                        alpha=0.22, lw=0, zorder=3)
                    rp = np.nanmean(store[kp][:, bi])
                    rs = np.nanmean(store[ks][:, bi])
                    ax.text(0.98, 0.94, f'{rp/rs:.2f}' if rs else '',
                            transform=ax.transAxes, ha='right', va='top',
                            fontsize=7, color=INK)
                    ax.axvline(0, color=INK, lw=0.8, ls=(0, (3, 3)), zorder=1)
                    if r == 0:
                        ax.set_title(BAND_LAB[bd], fontsize=9, color=INK)
                    if c == 0:
                        ax.set_ylabel(f'{rowlab}\n{lab}', fontsize=8)
                    if r == len(loaded) - 1:
                        ax.set_xlabel('window centre (ms)', fontsize=8)
                    ax.tick_params(labelsize=7)
                    sns.despine(ax=ax)
                    ax.grid(axis='y', color=MUTED, alpha=0.18, lw=0.5)
                    ax.set_axisbelow(True)
            fig.suptitle(
                f'{a.replace("-lh","")} → {b.replace("-lh","")}   '
                f'({measure.upper()})   {task} / {stim} / {method}   '
                f'n={len(subs0)}\n'
                'thin = individual subjects, bold = group mean ± SEM, '
                'number = parametric ÷ state-space', fontsize=10, y=1.004)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches='tight', facecolor='white')
            plt.close(fig)
    print(f'   {os.path.basename(path)}  ({len(edges)} edges)')
