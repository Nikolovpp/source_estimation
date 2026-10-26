#!/usr/bin/env python
"""
Per-subject QC of the inverse operators from the resolution-sweep outputs.

Why this can flag subjects at all: the forward model (fsaverage ico-5, template
biosemi64 positions) is identical for every subject, so the ONLY subject-specific
ingredients of the resolution metrics are (a) the baseline noise covariance
(whitening of the MN kernel, noise_gain), (b) the task-window data covariance
(LCMV weights, pc1 ROI weights, k_pc1_frac) and (c) the epoch count / rank.
A subject whose CTF metrics sit far from the group therefore has an unusual
covariance - residual artifacts, bad channels, low rank, few trials - which is
exactly what preprocessing QC is meant to catch, seen from the other side.

Reads  RESOLUTION_ANALYSIS/{task}_{stim}/sweep/per_subject/*.roi.csv|*.pair.csv
(20 subjects x 56 configs x 2 atlases) and writes to
RESOLUTION_ANALYSIS/{task}_{stim}/subject_qc/:

  qc_flags_{atlas}.csv           subject x metric raw values + robust z (sign:
                                 + = worse than the group) for production
                                 dSPM / eLORETA / LCMV; flags are one-sided
                                 (worse only), n_better counts the other tail
  qc_heatmap_{atlas}.png         the z-score matrix, subjects sorted worst-first,
                                 |z| >= Z_FLAG cells outlined
  qc_noise_gain_{atlas}.png      noise_gain per ROI per subject (log), group
                                 median and x0.5 / x2 band - the noise-cov check
  qc_lcmv_collapse_{atlas}.png   LCMV PC1 share per ROI per subject (1 = one
                                 global time course) - the data-cov check
  qc_matrices_{atlas}_{cfg}.png  every subject's |cos| matrix for production
                                 dSPM and LCMV - eyeball a deviant subject
  qc_config_agreement_{atlas}.png  does each subject rank the 56 configurations
                                 like the rest of the group? (Spearman rho of
                                 the subject's speech-|cos| ranking vs the
                                 leave-one-out group ranking) + mean |cos| vs
                                 noise_gain scatter labelled by subject
  qc_summary.txt                 flagged subjects with reasons

and, when more than one task dir is given, RESOLUTION_ANALYSIS/subject_qc/
overview.png + overview.csv: flag counts per subject across tasks / atlases.

Usage (local is fine - CSV in, PNG out):
  python exploratory/resolution_subject_qc.py                        # both tasks
  python exploratory/resolution_subject_qc.py --task overtProd --stim-class prodDiff
  python exploratory/resolution_subject_qc.py --z-flag 2.5
"""
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import spearmanr

from resolution_analysis import OUT_ROOT
from resolution_sweep import baseline_config_names
from resolution_sweep_report import load, CONTROL_ROIS

Z_FLAG = 3.0      # robust z (MAD-scaled) beyond which a cell is flagged
Z_WATCH = 2.0     # drawn lighter: worth a look, not a flag

# metric -> (label, higher-is-worse?)  All from the roi/pair csvs.
METRICS = {
    'log_noise_gain': ('log10 noise gain', True),
    'se_own':         ('own-ROI energy share', False),
    'speech_cos':     ('mean speech |cos|', True),
    'max_cos':        ('max |cos|', True),
    'k_pc1_frac':     ('PC1 share', True),
    'sd_mm':          ('CTF spread (mm)', True),
    'peak_in_roi':    ('CTF peak in ROI', False),
}
PROD = ['dSPM', 'eLORETA', 'LCMV']


def robust_z(x):
    x = np.asarray(x, float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) * 1.4826
    if not np.isfinite(mad) or mad == 0:
        mad = np.nanstd(x) or 1.0
    return (x - med) / mad


def subject_table(roi, pair, atlas, cfg):
    """One row per subject: the QC metrics for one configuration."""
    r = roi[(roi.atlas == atlas) & (roi.config == cfg)]
    p = pair[(pair.atlas == atlas) & (pair.config == cfg)].copy()
    p['abscos'] = p.pattern_cos.abs()
    speech = ~(p.roi_i.isin(CONTROL_ROIS) | p.roi_j.isin(CONTROL_ROIS))
    t = r.groupby('subject').agg(noise_gain=('noise_gain', 'median'),
                                 se_own=('se_own', 'mean'),
                                 k_pc1_frac=('k_pc1_frac', 'mean'),
                                 sd_mm=('sd_mm', 'mean'),
                                 peak_in_roi=('peak_in_roi', 'mean'))
    t['log_noise_gain'] = np.log10(t.noise_gain)
    t['speech_cos'] = p[speech].groupby('subject').abscos.mean()
    t['max_cos'] = p.groupby('subject').abscos.max()
    return t


def flag_table(roi, pair, atlas, base):
    """subject x (config, metric) raw values + robust z + flag count."""
    raw, z = {}, {}
    for m in PROD:
        cfg = base[m]
        t = subject_table(roi, pair, atlas, cfg)
        for k, (lab, worse_high) in METRICS.items():
            if k == 'k_pc1_frac' and m != 'LCMV':
                continue            # only meaningful as a collapse detector
            col = f'{m}:{k}'
            raw[col] = t[k]
            zz = robust_z(t[k])
            z[col] = zz if worse_high else -zz   # positive = worse
    raw = pd.DataFrame(raw)
    z = pd.DataFrame(z, index=raw.index)
    out = raw.add_suffix('|raw').join(z.add_suffix('|z'))
    # one-sided: only "worse than the group" counts. "Better than the group"
    # (e.g. an LCMV that collapses less than the others) is reported, not flagged.
    out['n_flag'] = (z >= Z_FLAG).sum(axis=1)
    out['n_watch'] = ((z >= Z_WATCH) & (z < Z_FLAG)).sum(axis=1)
    out['n_better'] = (z <= -Z_FLAG).sum(axis=1)
    out['worst_z'] = z.max(axis=1)
    order = out.sort_values(['n_flag', 'worst_z'], ascending=False).index
    return raw.loc[order], z.loc[order], out.loc[order]


def plot_heatmap(z, raw, atlas, out_dir, title):
    cols = list(z.columns)
    fig_w = max(9, 0.62 * len(cols) + 3)
    fig, ax = plt.subplots(figsize=(fig_w, 0.38 * len(z) + 2.2))
    v = z.values
    im = ax.imshow(np.clip(v, -4, 4), cmap='RdBu_r', vmin=-4, vmax=4,
                   aspect='auto')
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            val = raw.iloc[i, j]
            s = f'{val:.2g}' if abs(val) < 100 else f'{val:.0f}'
            ax.text(j, i, s, ha='center', va='center', fontsize=6.5,
                    color='white' if abs(v[i, j]) > 2.5 else 'black')
            if v[i, j] >= Z_FLAG:
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                       ec='k', lw=2))
            elif v[i, j] >= Z_WATCH:
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                       ec='k', lw=0.8, ls='--'))
    ax.set_yticks(range(len(z)))
    ax.set_yticklabels(z.index, fontsize=8)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f'{c.split(":")[0]}\n{METRICS[c.split(":")[1]][0]}'
                        for c in cols], fontsize=7, rotation=90)
    # separators between configurations
    starts = [i for i in range(1, len(cols))
              if cols[i].split(':')[0] != cols[i - 1].split(':')[0]]
    for s in starts:
        ax.axvline(s - .5, color='k', lw=1.2)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label('robust z (sign: + = worse than group)')
    ax.set_title(f'{title}\nsubject QC, {atlas}: cell text = raw value; solid box '
                 f'z >= {Z_FLAG:g} (worse than group), dashed z >= {Z_WATCH:g}; '
                 'blue = better than group (not flagged); rows sorted worst-first',
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / f'qc_heatmap_{atlas}.png', dpi=150)
    plt.close(fig)


def plot_noise_gain(roi, atlas, base, out_dir, title, subj_order):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=False)
    for ax, m in zip(axes, ['dSPM', 'eLORETA']):
        r = roi[(roi.atlas == atlas) & (roi.config == base[m])]
        rois = sorted(r.roi.unique())
        x = {s: i for i, s in enumerate(subj_order)}
        cmap = plt.get_cmap('tab10')
        for k, name in enumerate(rois):
            d = r[r.roi == name]
            ax.scatter([x[s] for s in d.subject], d.noise_gain, s=22,
                       color=cmap(k % 10), label=name.replace('-lh', ''),
                       zorder=3)
        med = r.groupby('subject').noise_gain.median()
        gm = med.median()
        ax.axhline(gm, color='k', lw=1)
        ax.axhspan(gm / 2, gm * 2, color='k', alpha=0.07)
        ax.plot([x[s] for s in med.index], med.values, 'k_', ms=14, mew=2,
                label='subject median')
        ax.set_yscale('log')
        ax.set_xticks(range(len(subj_order)))
        ax.set_xticklabels([s.replace('EEGPROD', '') for s in subj_order],
                           rotation=90, fontsize=8)
        ax.set_title(f'{m} (production): noise gain per ROI\n'
                     f'band = x0.5 .. x2 of the group median', fontsize=10)
        ax.set_ylabel('noise_gain (sensor noise SD per unit own-source gain)')
        ax.grid(alpha=.3, which='both')
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle(f'{title} - {atlas}: the noise-covariance check '
                 '(forward model is shared, so spread here is the subject\'s C_noise)',
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / f'qc_noise_gain_{atlas}.png', dpi=150)
    plt.close(fig)


def plot_lcmv_collapse(roi, atlas, base, out_dir, title, subj_order):
    r = roi[(roi.atlas == atlas) & (roi.config == base['LCMV'])]
    rois = sorted(r.roi.unique())
    mat = r.pivot_table(index='subject', columns='roi', values='k_pc1_frac')
    mat = mat.reindex(subj_order)[rois]
    fig, ax = plt.subplots(figsize=(2 + 0.75 * len(rois), 0.33 * len(mat) + 2))
    im = ax.imshow(mat.values, cmap='YlOrRd', vmin=0.3, vmax=1.0, aspect='auto')
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f'{mat.values[i, j]:.2f}', ha='center', va='center',
                    fontsize=7, color='white' if mat.values[i, j] > 0.85 else 'k')
    ax.set_xticks(range(len(rois)))
    ax.set_xticklabels([x.replace('-lh', '') for x in rois], rotation=45,
                       ha='right', fontsize=8)
    ax.set_yticks(range(len(mat)))
    ax.set_yticklabels(mat.index, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02,
                 label='PC1 share of ROI variance (1 = beamformer collapse)')
    ax.set_title(f'{title} - {atlas}\nLCMV (production) collapse detector: '
                 'share of ROI variance in one component', fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / f'qc_lcmv_collapse_{atlas}.png', dpi=150)
    plt.close(fig)


def plot_subject_matrices(pair, atlas, cfg, tag, out_dir, title, subj_order):
    p = pair[(pair.atlas == atlas) & (pair.config == cfg)].copy()
    p['abscos'] = p.pattern_cos.abs()
    names = sorted(set(p.roi_i) | set(p.roi_j))
    short = [n.replace('-lh', '').replace('Inferior_', 'Inf.')
             .replace('Superior_', 'Sup.') for n in names]
    n = len(subj_order)
    ncol = 5
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.5 * nrow))
    axes = np.atleast_2d(axes)
    for k, s in enumerate(subj_order):
        ax = axes.flat[k]
        d = p[p.subject == s]
        M = d.pivot_table(index='roi_i', columns='roi_j', values='abscos')
        M = M.reindex(index=names, columns=names)
        ax.imshow(M.values, cmap='Blues', vmin=0, vmax=1)
        ax.set_title(s.replace('EEGPROD', ''), fontsize=9)
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        if k // ncol == nrow - 1:
            ax.set_xticklabels(short, rotation=90, fontsize=6)
        else:
            ax.set_xticklabels([])
        if k % ncol == 0:
            ax.set_yticklabels(short, fontsize=6)
        else:
            ax.set_yticklabels([])
        mean_off = np.nanmean(M.values[~np.eye(len(names), dtype=bool)])
        ax.text(0.98, 0.02, f'{mean_off:.2f}', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=8,
                bbox=dict(fc='w', ec='none', alpha=.8))
    for ax in axes.flat[n:]:
        ax.axis('off')
    fig.suptitle(f'{title} - {atlas} - {cfg}\n|cos(CTF_i, CTF_j)| per subject '
                 '(corner = mean off-diagonal); same color scale 0..1', fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / f'qc_matrices_{atlas}_{tag}.png', dpi=140)
    plt.close(fig)


def config_agreement(pair, atlas):
    """Spearman rho between a subject's ranking of all configs (speech |cos|)
    and the leave-one-out group ranking."""
    p = pair[pair.atlas == atlas].copy()
    p['abscos'] = p.pattern_cos.abs()
    speech = ~(p.roi_i.isin(CONTROL_ROIS) | p.roi_j.isin(CONTROL_ROIS))
    sc = p[speech].groupby(['subject', 'config']).abscos.mean().unstack('config')
    rho = {}
    for s in sc.index:
        others = sc.drop(index=s).mean(axis=0)
        rho[s] = spearmanr(sc.loc[s].values, others.values).correlation
    return pd.Series(rho), sc


def plot_agreement(pair, roi, atlas, base, out_dir, title, subj_order):
    rho, sc = config_agreement(pair, atlas)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    ax = axes[0]
    vals = rho.reindex(subj_order)
    colors = ['#c0392b' if v < 0.8 else '#2a78d6' for v in vals]
    ax.bar(range(len(vals)), vals.values, color=colors)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels([s.replace('EEGPROD', '') for s in vals.index],
                       rotation=90, fontsize=8)
    ax.set_ylim(0, 1.02)
    ax.axhline(0.8, color='k', ls='--', lw=0.8)
    ax.set_ylabel('Spearman rho vs leave-one-out group')
    ax.set_title('Does the subject rank the 56 configurations like the group?\n'
                 '(red < 0.8)', fontsize=10)
    ax.grid(axis='y', alpha=.3)
    ax = axes[1]
    t = subject_table(roi, pair, atlas, base['dSPM'])
    ax.scatter(t.noise_gain, t.speech_cos, s=30, color='#2a78d6', zorder=3)
    for s, row in t.iterrows():
        ax.annotate(s.replace('EEGPROD', ''), (row.noise_gain, row.speech_cos),
                    fontsize=7, xytext=(3, 3), textcoords='offset points')
    ax.set_xscale('log')
    ax.set_xlabel('noise_gain (production dSPM, median over ROIs)')
    ax.set_ylabel('mean speech |cos| (production dSPM)')
    ax.set_title('Per-subject resolution vs noise at the production dSPM',
                 fontsize=10)
    ax.grid(alpha=.3, which='both')
    fig.suptitle(f'{title} - {atlas}', fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / f'qc_config_agreement_{atlas}.png', dpi=150)
    plt.close(fig)
    return rho


def run_task(task_dir, atlases, z_flag):
    global Z_FLAG
    Z_FLAG = z_flag
    sweep_dir = task_dir / 'sweep'
    if not sweep_dir.exists():
        print(f'[skip] no sweep dir under {task_dir}')
        return None
    out_dir = task_dir / 'subject_qc'
    out_dir.mkdir(exist_ok=True)
    title = task_dir.name.replace('_', '/')
    roi, pair = load(sweep_dir)
    base = baseline_config_names()
    lines = [f'Subject QC - {title}, {roi.subject.nunique()} subjects, '
             f'flag z >= {Z_FLAG:g} worse than the group (robust z, MAD-scaled), '
             f'watch z >= {Z_WATCH:g}; better-than-group deviations are not flagged',
             'Configurations: production dSPM / eLORETA / LCMV; metrics per subject '
             'are medians (noise_gain) or means over ROIs / ROI pairs.', '']
    overview = {}
    for atlas in atlases:
        if atlas not in set(roi.atlas):
            continue
        raw, z, out = flag_table(roi, pair, atlas, base)
        subj_order = list(out.index)                 # worst first
        rho = plot_agreement(pair, roi, atlas, base, out_dir, title, subj_order)
        out['config_rank_rho'] = rho.reindex(out.index)
        out.to_csv(out_dir / f'qc_flags_{atlas}.csv')
        plot_heatmap(z, raw, atlas, out_dir, title)
        plot_noise_gain(roi, atlas, base, out_dir, title, subj_order)
        plot_lcmv_collapse(roi, atlas, base, out_dir, title, subj_order)
        plot_subject_matrices(pair, atlas, base['dSPM'], 'dSPM', out_dir, title,
                              subj_order)
        plot_subject_matrices(pair, atlas, base['LCMV'], 'LCMV', out_dir, title,
                              subj_order)
        lines.append(f'== {atlas}')
        flagged = out[out.n_flag > 0]
        if flagged.empty:
            lines.append('  no subject beyond the flag threshold')
        for s, row in flagged.iterrows():
            reasons = [f'{c}={raw.loc[s, c]:.3g} (z={z.loc[s, c]:+.1f})'
                       for c in z.columns if z.loc[s, c] >= Z_FLAG]
            lines.append(f'  {s}: {row.n_flag} flag(s), {row.n_watch} watch; '
                         + '; '.join(reasons))
        watch = out[(out.n_flag == 0) & (out.n_watch > 0)]
        if not watch.empty:
            lines.append('  watch only: ' + ', '.join(
                f'{s} ({int(r.n_watch)})' for s, r in watch.iterrows()))
        low = rho[rho < 0.8]
        lines.append('  config-ranking agreement (rho vs group): median '
                     f'{rho.median():.2f}, min {rho.min():.2f} ({rho.idxmin()})'
                     + (f'; below 0.8: {", ".join(low.index)}' if len(low) else ''))
        lines.append('')
        overview[atlas] = out[['n_flag', 'n_watch', 'n_better', 'worst_z',
                               'config_rank_rho']]
    txt = '\n'.join(lines)
    (out_dir / 'qc_summary.txt').write_text(txt)
    print(txt)
    print(f'-> {out_dir}')
    return overview


def plot_overview(all_ov, root):
    """Flag counts per subject across (task, atlas)."""
    cols, data = [], {}
    for task, ov in all_ov.items():
        for atlas, df in ov.items():
            cols.append(f'{task}\n{atlas}')
            data[cols[-1]] = df.n_flag + 0.5 * df.n_watch
    M = pd.DataFrame(data).fillna(0)
    M = M.loc[M.sum(axis=1).sort_values(ascending=False).index]
    out_dir = root / 'subject_qc'
    out_dir.mkdir(exist_ok=True)
    M.to_csv(out_dir / 'overview.csv')
    fig, ax = plt.subplots(figsize=(2 + 1.3 * len(cols), 0.33 * len(M) + 2))
    im = ax.imshow(M.values, cmap='Reds', vmin=0, vmax=max(3, M.values.max()),
                   aspect='auto')
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f'{M.values[i, j]:g}', ha='center', va='center',
                    fontsize=8)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c.replace('_', '/') for c in cols], fontsize=8)
    ax.set_yticks(range(len(M)))
    ax.set_yticklabels(M.index, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02,
                 label='flags + 0.5 x watch (production dSPM/eLORETA/LCMV metrics)')
    ax.set_title('Subject QC overview across tasks and ROI sets\n'
                 'a subject that is red in every column has an unusual covariance '
                 'in both recordings', fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / 'overview.png', dpi=150)
    plt.close(fig)
    print(f'-> {out_dir / "overview.png"}')


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--task', choices=['overtProd', 'perception'])
    ap.add_argument('--stim-class', choices=['prodDiff', 'percDiff'])
    ap.add_argument('--atlases', nargs='+', default=['sensorEquiv', 'custom'])
    ap.add_argument('--z-flag', type=float, default=Z_FLAG)
    ap.add_argument('--root', default=str(OUT_ROOT),
                    help='RESOLUTION_ANALYSIS root (default from config)')
    return ap.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)
    if args.task:
        stim = args.stim_class or ('prodDiff' if args.task == 'overtProd' else 'percDiff')
        task_dirs = [root / f'{args.task}_{stim}']
    else:
        task_dirs = sorted(d for d in root.iterdir()
                           if d.is_dir() and (d / 'sweep').exists())
    all_ov = {}
    for td in task_dirs:
        ov = run_task(td, args.atlases, args.z_flag)
        if ov:
            all_ov[td.name] = ov
    if len(all_ov) > 1:
        plot_overview(all_ov, root)


if __name__ == '__main__':
    main()
