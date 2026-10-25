#!/usr/bin/env python3
"""
Report for exploratory/resolution_sweep.py: which inverse settings give the
most resolvable ROIs, at what noise cost.  Reads the per-subject CSVs only
(no estimation; fine to run locally once the sweep directory is synced).

Figures / tables (RESOLUTION_ANALYSIS/{task}_{stim}/sweep/report/):

    factors_{atlas}.png      one-knob-at-a-time curves around the production
                             settings: mean pairwise |cos| (lower = better
                             separation), mean se_own (higher = better),
                             median noise_gain (lower = better), mean
                             k_pc1_frac — per family / knob, lines per method
    pareto_{atlas}.png       every configuration as a point: noise_gain vs
                             mean |cos|; production settings marked; the
                             recommended configurations labelled
    matrices_{atlas}.png     pattern-overlap matrices, production vs best
                             configuration of each family
    ranking_{atlas}.csv      all configurations, group means, sorted
    deltas.csv               per-knob marginal effect vs the production
                             settings (group mean differences)
    summary.txt              headline numbers + recommendation
    ctf_maps_{atlas}_{config}.png   (--brain-maps) group |CTF| of the best
                             configuration per family, next to production

Recommendation rule (``--max-noise-ratio``, default 2.0): among the
configurations whose median noise_gain is at most that multiple of the
production dSPM value, pick the lowest mean pairwise |cos| among the speech
ROIs; report se_own and k_pc1_frac alongside.  The rule is a default, not a
verdict — the plan document says how to read the figures.

Usage
-----
    python exploratory/resolution_sweep_report.py --task overtProd --stim-class prodDiff
    python exploratory/resolution_sweep_report.py --task perception --stim-class percDiff --brain-maps
"""
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from resolution_analysis import OUT_ROOT, _heat, _matrix
from resolution_sweep import (CFG_COLS, MN_METHODS, baseline_config_names,
                              MN_BASE, LCMV_BASE)

CONTROL_ROIS = {'Visual', 'owfa-lh', 'vwfa-lh'}
FAMILY_COLORS = {'mn': '#2a78d6', 'lcmv': '#eda100', 'deflect': '#1baf7a'}
METHOD_COLORS = {'dSPM': '#2a78d6', 'eLORETA': '#1baf7a', 'LCMV': '#eda100',
                 'DeFleCT': '#e87ba4'}
KNOBS = {'mn': ['snr', 'depth', 'loose', 'noise_cov'],
         'lcmv': ['reg', 'weight_norm', 'pick_ori', 'data_cov', 'noise_cov'],
         'deflect': ['snr', 'null_comp', 'nulls']}


# ─────────────────────────────────────────────────────────────────────
def load(sweep_dir):
    ps = sweep_dir / 'per_subject'
    roi = pd.concat([pd.read_csv(f) for f in sorted(ps.glob('*.roi.csv'))],
                    ignore_index=True)
    pair = pd.concat([pd.read_csv(f) for f in sorted(ps.glob('*.pair.csv'))],
                     ignore_index=True)
    for df in (roi, pair):
        df['weight_norm'] = df['weight_norm'].fillna('none')
        for c in CFG_COLS:
            if c not in df:
                df[c] = np.nan
    return roi, pair


def per_config(roi, pair, atlas):
    """Group-mean table, one row per configuration, for one atlas."""
    r = roi[roi.atlas == atlas]
    p = pair[pair.atlas == atlas]
    speech = ~(p.roi_i.isin(CONTROL_ROIS) | p.roi_j.isin(CONTROL_ROIS))
    g = p.groupby('config').agg(mean_cos=('pattern_cos', 'mean'),
                                max_cos=('pattern_cos', 'max'),
                                mean_xtalk=('ctf_xtalk', 'mean'),
                                n_subj=('subject', 'nunique'))
    g['speech_cos'] = p[speech].groupby('config').pattern_cos.mean()
    g['max_pair'] = p.groupby('config').apply(
        lambda d: (d.groupby(['roi_i', 'roi_j']).pattern_cos.mean().idxmax()))
    g['max_pair'] = g['max_pair'].map(lambda t: f'{t[0]}<>{t[1]}'.replace('-lh', ''))
    gr = r.groupby('config').agg(se_own=('se_own', 'mean'),
                                 sd_mm=('sd_mm', 'mean'),
                                 peak_in_roi=('peak_in_roi', 'mean'),
                                 noise_gain=('noise_gain', 'median'),
                                 k_pc1_frac=('k_pc1_frac', 'mean'))
    g = g.join(gr)
    meta = r.drop_duplicates('config').set_index('config')[
        [c for c in CFG_COLS if c != 'config']]
    g = g.join(meta)
    return g.reset_index()


def recommend(tab, max_noise_ratio, base_names):
    base = tab[tab.config == base_names['dSPM']]
    ref_noise = float(base.noise_gain.iloc[0]) if not base.empty else np.nan
    ok = tab[tab.noise_gain <= max_noise_ratio * ref_noise] if np.isfinite(ref_noise) else tab
    best = {}
    for fam, d in ok.groupby('family'):
        best[fam] = d.sort_values('speech_cos').iloc[0]
    overall = ok.sort_values('speech_cos').iloc[0] if not ok.empty else None
    return ref_noise, best, overall


# ─────────────────────────────────────────────────────────────────────
def plot_factors(tab, atlas, out_dir, task, stim):
    metrics = [('speech_cos', 'mean |cos|, speech ROI pairs'),
               ('se_own', 'mean se_own'),
               ('noise_gain', 'median noise_gain'),
               ('k_pc1_frac', 'mean PC1 share (1 = collapsed)')]
    panels = [(fam, knob) for fam in ('mn', 'lcmv', 'deflect')
              for knob in KNOBS[fam] if fam in set(tab.family)]
    if not panels:
        return
    fig, axes = plt.subplots(len(metrics), len(panels),
                             figsize=(2.6 * len(panels) + 1, 2.3 * len(metrics) + 1),
                             squeeze=False)
    for ci, (fam, knob) in enumerate(panels):
        d = tab[tab.family == fam].copy()
        # one-knob-at-a-time slice: every other knob at its base value
        base = MN_BASE if fam == 'mn' else LCMV_BASE if fam == 'lcmv' else {}
        for k, v in base.items():
            if k != knob and k in d:
                d = d[(d[k] == v) | d[k].isna()]
        if fam == 'deflect':
            hue = 'nulls' if knob != 'nulls' else 'null_comp'
        else:
            hue = 'method'
        levels = sorted(d[knob].dropna().unique(), key=lambda x: (str(type(x)), x))
        xs = np.arange(len(levels))
        for ri, (m, lab) in enumerate(metrics):
            ax = axes[ri, ci]
            for h, dh in d.groupby(hue):
                y = [dh[dh[knob] == lv][m].mean() if (dh[knob] == lv).any()
                     else np.nan for lv in levels]
                col = METHOD_COLORS.get(h, None) if hue == 'method' else None
                ax.plot(xs, y, 'o-', ms=4, lw=1.5, color=col, label=str(h))
            ax.set_xticks(xs)
            ax.set_xticklabels([f'{lv:g}' if isinstance(lv, float) else str(lv)
                                for lv in levels], rotation=30, ha='right',
                               fontsize=7)
            if m == 'noise_gain':
                ax.set_yscale('log')
            if ci == 0:
                ax.set_ylabel(lab, fontsize=8)
            if ri == 0:
                ax.set_title(f'{fam} · {knob}', fontsize=9)
                ax.legend(frameon=False, fontsize=6)
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(axis='y', color='#eeeeee', lw=0.6)
            ax.tick_params(labelsize=7)
    fig.suptitle(f'One knob at a time around the production settings — '
                 f'{atlas}, {task}/{stim}, group means', fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / f'factors_{atlas}.png', dpi=160)
    plt.close(fig)


def plot_pareto(tab, atlas, out_dir, task, stim, base_names, best, ref_noise,
                max_noise_ratio):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for fam, d in tab.groupby('family'):
        ax.scatter(d.noise_gain, d.speech_cos, s=26, alpha=0.75,
                   color=FAMILY_COLORS.get(fam, '#888'), label=fam,
                   edgecolor='white', lw=0.5)
    for k, (m, name) in enumerate(base_names.items()):
        b = tab[tab.config == name]
        if not b.empty:
            ax.scatter(b.noise_gain, b.speech_cos, s=140, marker='*',
                       color='#0b0b0b', zorder=5)
            ax.annotate(f'production {m}', (b.noise_gain.iloc[0], b.speech_cos.iloc[0]),
                        xytext=(6, 6 + 9 * k), textcoords='offset points', fontsize=7)
    for fam, b in best.items():
        ax.annotate(b.config, (b.noise_gain, b.speech_cos), xytext=(6, -10),
                    textcoords='offset points', fontsize=7,
                    color=FAMILY_COLORS.get(fam, '#444'))
    if np.isfinite(ref_noise):
        ax.axvline(max_noise_ratio * ref_noise, color='#c3c2b7', lw=0.8, ls='--')
        ax.text(max_noise_ratio * ref_noise, ax.get_ylim()[1], f' {max_noise_ratio:g}x production noise',
                fontsize=7, va='top', color='#52514e')
    ax.set_xscale('log')
    ax.set_xlabel('noise_gain (sensor noise SD per unit own-source gain; lower = better)')
    ax.set_ylabel('mean |cos| between speech ROI CTFs (lower = better)')
    ax.set_title(f'Resolution vs noise, every configuration — {atlas}, {task}/{stim}',
                 fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(color='#eeeeee', lw=0.6)
    fig.tight_layout()
    fig.savefig(out_dir / f'pareto_{atlas}.png', dpi=160)
    plt.close(fig)


def plot_matrices(pair, atlas, out_dir, task, stim, base_names, best):
    cols = [('production dSPM', base_names['dSPM']),
            ('production LCMV', base_names['LCMV'])]
    cols += [(f'best {fam}', b.config) for fam, b in best.items()]
    p = pair[pair.atlas == atlas]
    fig, axes = plt.subplots(1, len(cols), figsize=(3.9 * len(cols) + 1, 4.4),
                             squeeze=False)
    im = None
    for ax, (title, cfg) in zip(axes[0], cols):
        d = p[p.config == cfg]
        if d.empty:
            ax.set_axis_off()
            continue
        g = d.groupby(['roi_i', 'roi_j']).pattern_cos.mean().reset_index()
        names = list(dict.fromkeys(g.roi_i))
        M = _matrix(g, names, 'pattern_cos')
        im = _heat(ax, M, names, 1.0, lambda v: f'{v:.2f}', 'Blues', title)
        # config name below the panel, wrapped at the underscores
        parts, lines_, cur = cfg.split('_'), [], ''
        for part in parts:
            if len(cur) + len(part) + 1 > 26 and cur:
                lines_.append(cur)
                cur = part
            else:
                cur = f'{cur}_{part}' if cur else part
        lines_.append(cur)
        ax.set_xlabel('\n'.join(lines_), fontsize=6.5, color='#52514e', labelpad=6)
    fig.subplots_adjust(wspace=0.6, top=0.82, bottom=0.3)
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, pad=0.02)
    fig.suptitle(f'|cos(CTF_i, CTF_j)| — {atlas}, {task}/{stim}', fontsize=11)
    fig.savefig(out_dir / f'matrices_{atlas}.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


def deltas(tabs, base_names):
    """Marginal effect of each knob level vs the production settings."""
    rows = []
    for atlas, tab in tabs.items():
        for fam in ('mn', 'lcmv'):
            base = MN_BASE if fam == 'mn' else LCMV_BASE
            for _, r in tab[tab.family == fam].iterrows():
                ref_name = base_names.get(r.method)
                ref = tab[tab.config == ref_name]
                if ref.empty or r.config == ref_name:
                    continue
                changed = [k for k in KNOBS[fam]
                           if k in base and str(r[k]) != str(base[k])]
                if len(changed) != 1:
                    continue
                k = changed[0]
                rows.append(dict(atlas=atlas, family=fam, method=r.method, knob=k,
                                 level=r[k],
                                 d_speech_cos=r.speech_cos - ref.speech_cos.iloc[0],
                                 d_se_own=r.se_own - ref.se_own.iloc[0],
                                 noise_ratio=r.noise_gain / ref.noise_gain.iloc[0],
                                 d_k_pc1=r.k_pc1_frac - ref.k_pc1_frac.iloc[0]))
    return pd.DataFrame(rows)


def brain_maps(sweep_dir, out_dir, atlas, configs, titles):
    from forward_model import setup_fsaverage
    from resolution_analysis import render_ctf_maps
    _, _, src, _ = setup_fsaverage()
    files = sorted((sweep_dir / 'per_subject').glob('*_ctf.npz'))
    for cfg, title in zip(configs, titles):
        acc, names, n_lh = None, None, None
        for f in files:
            d = np.load(f, allow_pickle=True)
            key = f'{atlas}__{cfg}'
            if key not in d:
                continue
            c = np.abs(d[key].astype(float))
            c = c / np.maximum(c.max(axis=1, keepdims=True), 1e-30)
            acc = c if acc is None else acc + c
            names = list(d[f'names__{atlas}'])
            n_lh = int(d['n_lh'])
        if acc is None:
            continue
        acc /= len(files)
        render_ctf_maps(acc, names, n_lh, src, f'{title} — {atlas}\n{cfg}',
                        out_dir / f'ctf_maps_{atlas}_{cfg}.png')
        print(f'  brain maps: {atlas} / {cfg}')


# ─────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--task', required=True, choices=['perception', 'overtProd'])
    p.add_argument('--stim-class', default='prodDiff', choices=['prodDiff', 'percDiff'])
    p.add_argument('--sweep-dir', default=None)
    p.add_argument('--max-noise-ratio', type=float, default=2.0,
                   help='noise_gain budget relative to production dSPM (default 2)')
    p.add_argument('--brain-maps', action='store_true', default=False)
    return p.parse_args()


def main():
    args = parse_args()
    sweep_dir = Path(args.sweep_dir) if args.sweep_dir else \
        OUT_ROOT / f'{args.task}_{args.stim_class}' / 'sweep'
    out_dir = sweep_dir / 'report'
    out_dir.mkdir(parents=True, exist_ok=True)
    roi, pair = load(sweep_dir)
    base_names = baseline_config_names()
    atlases = list(dict.fromkeys(roi.atlas))
    n_subj = roi.subject.nunique()
    lines = [f'Resolution sweep report — {args.task}/{args.stim_class}, '
             f'{n_subj} subjects, {roi.config.nunique()} configurations',
             f'noise budget: {args.max_noise_ratio:g} x production dSPM noise_gain', '']
    tabs = {}
    for atlas in atlases:
        tab = per_config(roi, pair, atlas)
        tabs[atlas] = tab
        ref_noise, best, overall = recommend(tab, args.max_noise_ratio, base_names)
        tab.sort_values('speech_cos').to_csv(out_dir / f'ranking_{atlas}.csv', index=False)
        plot_factors(tab, atlas, out_dir, args.task, args.stim_class)
        plot_pareto(tab, atlas, out_dir, args.task, args.stim_class, base_names,
                    best, ref_noise, args.max_noise_ratio)
        plot_matrices(pair, atlas, out_dir, args.task, args.stim_class, base_names, best)
        lines.append(f'== {atlas}')
        hdr = (f'  {"config":52s} {"speech|cos|":>11s} {"max|cos|":>8s} {"se_own":>6s} '
               f'{"noise":>7s} {"PC1":>5s} {"peakIn":>6s}')
        lines.append(hdr)

        def row(label, r):
            return (f'  {label:52s} {r.speech_cos:11.3f} {r.max_cos:8.2f} {r.se_own:6.2f} '
                    f'{r.noise_gain:7.2e} {r.k_pc1_frac if np.isfinite(r.k_pc1_frac) else float("nan"):5.2f} '
                    f'{100 * r.peak_in_roi:5.0f}%')
        for m, name in base_names.items():
            b = tab[tab.config == name]
            if not b.empty:
                lines.append(row(f'production {m}', b.iloc[0]))
        lines.append('  --- best within noise budget, per family:')
        for fam, b in best.items():
            lines.append(row(b.config, b))
        if overall is not None:
            lines.append(f'  --> recommended: {overall.config}')
        lines.append('  --- top 8 overall (ignoring the noise budget):')
        for _, r in tab.sort_values('speech_cos').head(8).iterrows():
            lines.append(row(r.config, r))
        lines.append('')
        if args.brain_maps:
            cfgs = [base_names['dSPM']] + [b.config for b in best.values()]
            titles = ['production dSPM'] + [f'best {f}' for f in best]
            brain_maps(sweep_dir, out_dir, atlas, cfgs, titles)
    d = deltas(tabs, base_names)
    if not d.empty:
        d.to_csv(out_dir / 'deltas.csv', index=False)
        lines.append('Marginal effects vs production (mean over atlases), knob levels '
                     'sorted by change in speech |cos| (negative = better):')
        dd = d.groupby(['family', 'method', 'knob', 'level']).agg(
            d_speech_cos=('d_speech_cos', 'mean'), d_se_own=('d_se_own', 'mean'),
            noise_ratio=('noise_ratio', 'mean')).reset_index().sort_values('d_speech_cos')
        for _, r in dd.iterrows():
            lines.append(f'  {r.family:5s} {r.method:8s} {r.knob:12s}={str(r.level):16s} '
                         f'd|cos| {r.d_speech_cos:+.3f}  d se_own {r.d_se_own:+.3f}  '
                         f'noise x{r.noise_ratio:.2f}')
    lines += ['', 'Reading guide: speech|cos| = expected zero-lag correlation between the',
              'speech ROI estimates under white sources (controls excluded); noise = sensor',
              'noise SD reaching the ROI time course per unit own-source gain (same units for',
              'every configuration); PC1 = share of ROI variance in one component (1.0 = the',
              'beamformer collapse); peakIn = share of ROIs whose CTF peak lies inside the ROI.']
    text = '\n'.join(lines)
    print(text)
    (out_dir / 'summary.txt').write_text(text + '\n')
    print(f'\nWrote {out_dir}')


if __name__ == '__main__':
    main()
