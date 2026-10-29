#!/usr/bin/env python
"""Review figures for the translation arm: vertex dSPM vs DeFleCT, decoding + GC.

Reads only finished result files (per-subject decoding CSVs, the group stats
CSVs written by ``source_stats_viz.py`` and ``granger_stats.py``, the onset
latency tables) and draws one set of easy-to-read overview figures per task,
with the two arms side by side:

    vertex   dSPM (SNR 3), sensorEquiv, vertex_selectkbest  — "does the
             discriminating signal reach the ROI" (upper bound)
    deflect  DeFleCT k3 / null2 / other_rois at SNR 10     — "is it the
             ROI's own" (lower bound; the other ROIs' leading topographies
             are nulled in every virtual channel)

Decoding (TRANSLATION_ARM_REVIEW/decoding/)
    decoding_curves_{task}.png        ROI x contrast grid, both arms overlaid
                                      (mean +- SEM); above the traces, per arm:
                                      asterisks = pointwise t vs chance (p < .05,
                                      uncorrected), bar = cluster
    decoding_rois_{task}.png          all ROIs in one panel per arm x contrast
                                      (regional selectivity at a glance)
    decoding_deflect_minus_vertex_{task}.png
                                      paired per-subject difference, asterisks
                                      = paired t (p < .05), bar = sign-flip cluster
    decoding_summary.csv / .md        peak, cluster spans, jackknife onset,
                                      fixed-window accuracy vs the Visual
                                      control, per task x contrast x arm x ROI
    decoding_arm_comparison.csv       per ROI: r(time), peak per arm, paired
                                      difference at the vertex peak, p

GC (TRANSLATION_ARM_REVIEW/gc/)
    gc_matrix_{task}_{stim}.png       significance overview: one 5 x 5 edge
                                      matrix per (arm, stats pass, band) for
                                      the directed GC (row -> column, task >
                                      baseline, right-tailed) and the net
                                      Diff-TRGC (upper triangle, signed, two-
                                      tailed). One leakage control per arm:
                                      a vertex arm reports both (TRGC is its
                                      control), a DeFleCT arm reports the
                                      directed GC only (its nulls are the
                                      control); color = -log10 cluster p,
                                      '*' p < .05 within edge, '**' survives
                                      FDR over the 120 edge x band family
    gc_trgc_{task}_{stim}.png         net TRGC time courses, 10 pairs x 3
                                      bands, both arms, baseline dashed,
                                      windows of interest shaded
    gc_directed_{task}_{stim}_{arm}.png
                                      directed GC both ways per pair, same grid

Usage (local, result files only):
    python exploratory/review_translation_arm.py                 # everything
    python exploratory/review_translation_arm.py --stage decoding
    python exploratory/review_translation_arm.py --stage gc --tasks perception
"""
import os
import sys
import re
import glob
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from mne.stats import permutation_cluster_1samp_test

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D

from config import DECODE_OUTPUT_ROOT, GC_WINDOWS_OF_INTEREST, canonical_feature_mode
from run_granger import GC_OUTPUT_ROOT, gc_tag

ROIS = ['Temporal', 'Inferior_Frontal', 'Superior_Frontal', 'Superior_Parietal', 'Visual']
ROI_SHORT = {'Temporal': 'Temporal', 'Inferior_Frontal': 'Inf. Frontal',
             'Superior_Frontal': 'Sup. Frontal', 'Superior_Parietal': 'Sup. Parietal',
             'Visual': 'Visual (control)'}
ROI_ABBR = {'Temporal': 'Tmp', 'Inferior_Frontal': 'IFG', 'Superior_Frontal': 'SFG',
            'Superior_Parietal': 'SPL', 'Visual': 'Vis'}
ROI_COLORS = {'Temporal': '#1b9e77', 'Inferior_Frontal': '#d95f02',
              'Superior_Frontal': '#7570b3', 'Superior_Parietal': '#e7298a',
              'Visual': '#555555'}
ARM_COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd']
BANDS = ['theta', 'low_beta', 'high_beta']
BAND_LABEL = {'theta': 'theta 4-8 Hz', 'low_beta': 'low beta 12-18 Hz',
              'high_beta': 'high beta 18-30 Hz', 'beta': 'beta 12-30 Hz'}
STIMS = ['prodDiff', 'percDiff']
TASK_LABEL = {'overtProd': 'overt production (0 = speech onset)',
              'perception': 'perception (0 = stimulus onset)'}
PAIRS = [(a, b) for i, a in enumerate(ROIS) for b in ROIS[i + 1:]]

REVIEW_ROOT = Path(DECODE_OUTPUT_ROOT).parent / 'TRANSLATION_ARM_REVIEW'


# ─────────────────────────────────────────────────────────────────────
# arms
# ─────────────────────────────────────────────────────────────────────
def parse_arm(spec):
    """label=method:feature_mode[:n_pcs[:normalize[:measures]]]

    measures = which GC quantity the arm reports, 'gc', 'dtrgc' or
    'gc+dtrgc' (default: a DeFleCT arm reports the directed GC — its nulls are
    the leakage control — a vertex arm reports both, TRGC being its control)."""
    label, rest = spec.split('=', 1)
    parts = rest.split(':')
    method, feat = parts[0], canonical_feature_mode(parts[1])
    n_pcs = int(parts[2]) if len(parts) > 2 and parts[2] else 1
    normalize = parts[3] if len(parts) > 3 and parts[3] else 'none'
    if len(parts) > 4 and parts[4]:
        measures = parts[4].split('+')
    else:
        measures = ['gc'] if feat.startswith('deflect') else ['gc', 'dtrgc']
    return dict(label=label, method=method, feat=feat, n_pcs=n_pcs, normalize=normalize,
                measures=measures, gc_tag=gc_tag(10, 40, 500, normalize, n_pcs=n_pcs))


DEFAULT_ARMS = ['vertex=dSPM:vertex_selectkbest:1:none:gc+dtrgc',
                'deflect=dSPM_snr10:deflect:3:none:gc']


def decode_dir(arm, task, atlas='sensorEquiv'):
    return (Path(DECODE_OUTPUT_ROOT) / task / arm['method'] / atlas / arm['feat']
            / 'raw' / 'no_pseudo' / '40_5' / 'logistic_tuned')


def gc_dir(arm, task, stim, atlas='sensorEquiv'):
    return (Path(GC_OUTPUT_ROOT) / task / arm['method'] / atlas / arm['feat']
            / 'raw' / arm['gc_tag'] / 'all_rois' / stim)


# ─────────────────────────────────────────────────────────────────────
# small helpers
# ─────────────────────────────────────────────────────────────────────
def spans(mask):
    """[(start_idx, end_idx)] of contiguous True runs."""
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if not m and start is not None:
            out.append((start, i - 1)); start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out


def span_str(t, mask):
    return '; '.join(f'{t[a]:.0f}..{t[b]:.0f}' for a, b in spans(mask)) or '-'


def mark_rows(ax, t, rows, y0, dy, star_ms=3.5, bar_lw=2.5):
    """Significance marks ABOVE the traces.  rows = [(point_mask, cluster_mask,
    color)], drawn bottom-up from y0 with spacing dy: asterisks = pointwise
    test, thin bar = cluster."""
    for i, (mp, mc, col) in enumerate(rows):
        y = y0 + i * dy
        if mp is not None and np.any(mp):
            ax.plot(t[mp], np.full(int(mp.sum()), y), ls='none', marker='*',
                    ms=star_ms, mew=0, color=col, alpha=0.9, clip_on=False)
        if mc is not None:
            for a, b in spans(mc):
                ax.plot([t[a], t[b]], [y + 0.45 * dy] * 2, color=col, lw=bar_lw,
                        solid_capstyle='butt', clip_on=False)


def md_table(df):
    """Markdown table without the tabulate dependency."""
    cols = list(df.columns)
    def fmt(v):
        if isinstance(v, float):
            return '' if not np.isfinite(v) else (f'{v:.3f}' if abs(v) < 10 else f'{v:.0f}')
        return str(v)
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    for _, r in df.iterrows():
        lines.append('| ' + ' | '.join(fmt(r[c]) for c in cols) + ' |')
    return '\n'.join(lines)


def fdr(p):
    p = np.asarray(p, float)
    ok = np.isfinite(p)
    out = np.ones_like(p)
    if ok.sum():
        out[ok] = multipletests(p[ok], method='fdr_bh')[1]
    return out


# ─────────────────────────────────────────────────────────────────────
# decoding: loading
# ─────────────────────────────────────────────────────────────────────
def load_decoding(arm, task, stim):
    """Returns dict(t, X {roi: (n_subj, n_t)}, subjects, stats DataFrame or None,
    latencies, windows)."""
    d = decode_dir(arm, task) / stim
    subj, mats = [], {}
    for f in sorted(glob.glob(str(d / 'EEGPROD4*.csv'))):
        s = re.search(r'(EEGPROD\d+)', os.path.basename(f)).group(1)
        tab = pd.read_csv(f, usecols=['key', 'ms', 'decode_acc'])
        mats[s] = tab.pivot_table(index='ms', columns='key', values='decode_acc')
        subj.append(s)
    if not subj:
        return None
    t = mats[subj[0]].index.values
    for s in subj:
        mats[s] = mats[s].reindex(t)
    rois = [r for r in ROIS if r in mats[subj[0]].columns]
    X = {r: np.stack([mats[s][r].values for s in subj]) for r in rois}
    st = sorted(glob.glob(str(d / f'{task}_{stim}_40_5_*subjAvg_stats.csv')))
    stats_df = pd.read_csv(st[-1]) if st else None
    lat_dir = decode_dir(arm, task) / '_onset_latency'
    lat = pd.read_csv(lat_dir / f'{stim}_latencies.csv') if (lat_dir / f'{stim}_latencies.csv').exists() else None
    win = pd.read_csv(lat_dir / f'{stim}_windows.csv') if (lat_dir / f'{stim}_windows.csv').exists() else None
    return dict(t=t, X=X, subjects=subj, stats=stats_df, lat=lat, win=win, rois=rois)


def stat_masks(dec, roi):
    """(pointwise FDR mask, cluster mask) on dec['t'] from the stats CSV."""
    if dec['stats'] is None:
        return None, None
    s = dec['stats'][dec['stats']['key'] == roi].set_index('ms').reindex(dec['t'])
    mp = (s['p_value'] < 0.05).fillna(False).astype(bool).values
    mc = s['sig_cluster'].fillna(False).astype(bool).values
    return mp, mc


def fdr_mask(dec, roi):
    s = dec['stats'][dec['stats']['key'] == roi].set_index('ms').reindex(dec['t'])
    return s['sig_fdr'].fillna(False).astype(bool).values


# ─────────────────────────────────────────────────────────────────────
# decoding: figures
# ─────────────────────────────────────────────────────────────────────
def fig_decoding_curves(arms, data, task, out):
    stims = [s for s in STIMS if any(data[a['label']].get(s) for a in arms)]
    n_r = len(ROIS)
    fig, axes = plt.subplots(len(stims), n_r, figsize=(3.3 * n_r, 3.1 * len(stims)),
                             sharex=True, layout='constrained', squeeze=False)
    for i, stim in enumerate(stims):
        # common y range per contrast row
        lo, hi = 0.46, 0.56
        for a in arms:
            dec = data[a['label']].get(stim)
            if dec:
                for r in dec['rois']:
                    g = dec['X'][r].mean(0); se = dec['X'][r].std(0, ddof=1) / np.sqrt(len(dec['subjects']))
                    hi = max(hi, float((g + se).max()) + 0.005)
                    lo = min(lo, float((g - se).min()) - 0.005)
        dy = (hi - lo) * 0.07
        for k, r in enumerate(ROIS):
            ax = axes[i, k]
            rows = []
            for j, a in enumerate(arms):
                dec = data[a['label']].get(stim)
                if not dec or r not in dec['X']:
                    continue
                g = dec['X'][r].mean(0); se = dec['X'][r].std(0, ddof=1) / np.sqrt(len(dec['subjects']))
                ax.plot(dec['t'], g, color=ARM_COLORS[j], lw=1.3, label=f"{a['label']} (n={len(dec['subjects'])})")
                ax.fill_between(dec['t'], g - se, g + se, color=ARM_COLORS[j], alpha=0.18, lw=0)
                mp, mc = stat_masks(dec, r)
                rows.append((mp, mc, ARM_COLORS[j]))
                t = dec['t']
            mark_rows(ax, t, rows, hi + 0.4 * dy, dy)
            ax.axhline(0.5, color='k', lw=0.6, ls=':')
            ax.axvline(0, color='k', lw=0.6, ls='--')
            ax.set_ylim(lo, hi + dy * (len(rows) + 0.8))
            ax.spines[['top', 'right']].set_visible(False)
            if i == 0:
                ax.set_title(ROI_SHORT[r], fontsize=11)
            if k == 0:
                ax.set_ylabel(f'{stim}\naccuracy (mean ± SEM)')
                if i == 0:
                    ax.legend(fontsize=8, frameon=False, loc='upper left')
            if i == len(stims) - 1:
                ax.set_xlabel('ms')
    arm_desc = ' | '.join(f"{a['label']} = {a['method']} / {a['feat']}" for a in arms)
    fig.suptitle(f'Decoding, {TASK_LABEL[task]} — sensorEquiv ROIs, logistic (tuned), 40 ms / 5 ms\n'
                 f'arms: {arm_desc}\n'
                 'marks above the traces, one row per arm: * = pointwise t vs chance (p < .05, uncorrected), '
                 'bar = cluster permutation (p < .05)', fontsize=10)
    fig.savefig(out / f'decoding_curves_{task}.png', dpi=150)
    plt.close(fig)


def fig_decoding_rois(arms, data, task, out):
    stims = [s for s in STIMS if any(data[a['label']].get(s) for a in arms)]
    fig, axes = plt.subplots(len(arms), len(stims), figsize=(5.6 * len(stims), 3.6 * len(arms)),
                             sharex=True, layout='constrained', squeeze=False)
    for j, a in enumerate(arms):
        for i, stim in enumerate(stims):
            ax = axes[j, i]
            dec = data[a['label']].get(stim)
            if not dec:
                ax.set_axis_off(); continue
            lo, hi = 0.47, 0.55
            rows = []
            for r in dec['rois']:
                g = dec['X'][r].mean(0); se = dec['X'][r].std(0, ddof=1) / np.sqrt(len(dec['subjects']))
                ax.plot(dec['t'], g, color=ROI_COLORS[r], lw=1.4, label=ROI_SHORT[r],
                        ls='--' if r == 'Visual' else '-')
                ax.fill_between(dec['t'], g - se, g + se, color=ROI_COLORS[r], alpha=0.10, lw=0)
                hi = max(hi, float((g + se).max()) + 0.004); lo = min(lo, float((g - se).min()) - 0.004)
                mp, mc = stat_masks(dec, r)
                rows.append((mp, mc, ROI_COLORS[r]))
            dy = (hi - lo) * 0.055
            mark_rows(ax, dec['t'], rows, hi + 0.4 * dy, dy, star_ms=3)
            ax.axhline(0.5, color='k', lw=0.6, ls=':'); ax.axvline(0, color='k', lw=0.6, ls='--')
            ax.set_ylim(lo, hi + dy * (len(rows) + 0.8))
            ax.spines[['top', 'right']].set_visible(False)
            ax.set_title(f"{a['label']} arm ({a['method']} / {a['feat']}) — {stim}, n = {len(dec['subjects'])}", fontsize=10)
            if i == 0:
                ax.set_ylabel('accuracy (mean ± SEM)')
            if j == len(arms) - 1:
                ax.set_xlabel('ms')
            if j == 0 and i == 0:
                ax.legend(fontsize=8, frameon=False, ncol=2, loc='upper left')
    fig.suptitle(f'Decoding, {TASK_LABEL[task]} — all ROIs per arm\n'
                 'marks above the traces, one row per ROI (bottom = Temporal … top = Visual): '
                 '* = pointwise t vs chance (p < .05, uncorrected), bar = cluster', fontsize=10)
    fig.savefig(out / f'decoding_rois_{task}.png', dpi=150)
    plt.close(fig)


def arm_difference(arms, data, task, out, rng=42):
    """DeFleCT minus vertex (second arm minus first), per subject."""
    a0, a1 = arms[0], arms[1]
    stims = [s for s in STIMS if data[a0['label']].get(s) and data[a1['label']].get(s)]
    fig, axes = plt.subplots(len(stims), len(ROIS), figsize=(3.3 * len(ROIS), 3.0 * len(stims)),
                             sharex=True, layout='constrained', squeeze=False)
    rows_out = []
    for i, stim in enumerate(stims):
        d0, d1 = data[a0['label']][stim], data[a1['label']][stim]
        common = sorted(set(d0['subjects']) & set(d1['subjects']))
        i0 = [d0['subjects'].index(s) for s in common]; i1 = [d1['subjects'].index(s) for s in common]
        t = d0['t']
        assert np.allclose(t, d1['t'])
        for k, r in enumerate(ROIS):
            ax = axes[i, k]
            D = d1['X'][r][i1] - d0['X'][r][i0]                  # (n, n_t)
            g = D.mean(0); se = D.std(0, ddof=1) / np.sqrt(len(common))
            p = stats.ttest_1samp(D, 0.0, axis=0).pvalue
            mp = p < 0.05
            mp_fdr = fdr(p) < 0.05
            _, clusters, cp, _ = permutation_cluster_1samp_test(
                D, n_permutations=1024, tail=0, seed=rng, out_type='mask', verbose=False)
            mc = np.zeros(len(t), bool)
            for c, pc in zip(clusters, cp):
                if pc < 0.05:
                    mc[c] = True
            ax.plot(t, g, color='k', lw=1.2)
            ax.fill_between(t, g - se, g + se, color='k', alpha=0.15, lw=0)
            hi = float((g + se).max()) + 0.004; lo = float((g - se).min()) - 0.004
            dy = (hi - lo) * 0.08
            mark_rows(ax, t, [(mp, mc, 'k')], hi + 0.3 * dy, dy)
            ax.set_ylim(lo, hi + 2 * dy)
            ax.axhline(0, color='k', lw=0.6); ax.axvline(0, color='k', lw=0.6, ls='--')
            ax.spines[['top', 'right']].set_visible(False)
            if i == 0:
                ax.set_title(ROI_SHORT[r], fontsize=11)
            if k == 0:
                ax.set_ylabel(f'{stim}\n{a1["label"]} − {a0["label"]} (mean ± SEM)')
            if i == len(stims) - 1:
                ax.set_xlabel('ms')
            # table
            g0 = d0['X'][r][i0].mean(0); g1 = d1['X'][r][i1].mean(0)
            ipk = int(np.nanargmax(g0)); ipk1 = int(np.nanargmax(g1))
            dpk = D[:, ipk]
            rows_out.append(dict(
                task=task, stim=stim, roi=r, n=len(common),
                r_time=float(np.corrcoef(g0, g1)[0, 1]),
                peak_vertex=float(g0[ipk]), t_peak_vertex=float(t[ipk]),
                peak_deflect=float(g1[ipk1]), t_peak_deflect=float(t[ipk1]),
                d_at_vertex_peak=float(dpk.mean()), d_sem=float(dpk.std(ddof=1) / np.sqrt(len(dpk))),
                p_at_vertex_peak=float(stats.ttest_1samp(dpk, 0).pvalue),
                mean_abs_d=float(np.abs(D).mean()),
                n_windows_fdr=int(mp_fdr.sum()), cluster_spans=span_str(t, mc)))
    fig.suptitle(f'Decoding, {TASK_LABEL[task]}: {a1["label"]} ({a1["method"]} / {a1["feat"]}) minus '
                 f'{a0["label"]} ({a0["method"]} / {a0["feat"]}), paired per subject\n'
                 '* = paired t (p < .05, uncorrected), bar = sign-flip cluster permutation (two-tailed, p < .05)',
                 fontsize=10)
    fig.savefig(out / f'decoding_deflect_minus_vertex_{task}.png', dpi=150)
    plt.close(fig)
    return rows_out


def _r0(v):
    v = float(v)
    return round(v) if np.isfinite(v) else np.nan


def decoding_summary_rows(arms, data, task):
    rows = []
    for a in arms:
        for stim in STIMS:
            dec = data[a['label']].get(stim)
            if not dec:
                continue
            t = dec['t']
            for r in dec['rois']:
                g = dec['X'][r].mean(0)
                ipk = int(np.nanargmax(g))
                mp, mc = stat_masks(dec, r)
                row = dict(task=task, stim=stim, arm=a['label'], roi=r, n=len(dec['subjects']),
                           peak_acc=round(float(g[ipk]), 3), peak_ms=round(float(t[ipk])),
                           cluster_spans_ms=span_str(t, mc) if mc is not None else 'n/a',
                           n_windows_fdr=int(fdr_mask(dec, r).sum()) if mp is not None else -1,
                           first_cluster_onset_ms=(round(float(t[spans(mc)[0][0]])) if mc is not None and mc.any() else np.nan))
                if dec['lat'] is not None:
                    l = dec['lat'][dec['lat'].roi == r]
                    if len(l):
                        row['onset_ms'] = _r0(l.onset_ms.iloc[0]); row['onset_se'] = _r0(l.onset_se.iloc[0])
                if dec['win'] is not None:
                    for wn, w in dec['win'][dec['win'].roi == r].groupby('window'):
                        row[f'{wn}_acc'] = round(float(w.acc.iloc[0]), 3)
                        row[f'{wn}_minus_control'] = round(float(w['diff'].iloc[0]), 3)
                        row[f'{wn}_p_vs_control'] = round(float(w.p_paired.iloc[0]), 3)
                rows.append(row)
    return rows


# ─────────────────────────────────────────────────────────────────────
# GC
# ─────────────────────────────────────────────────────────────────────
def gc_passes(task):
    return ['group_stats'] + [f'group_stats_win_{w}' for w in GC_WINDOWS_OF_INTEREST[task]]


def pass_label(p):
    return 'full span' if p == 'group_stats' else p.replace('group_stats_win_', 'window: ')


def load_gc(arm, task, stim, pass_name):
    f = gc_dir(arm, task, stim) / pass_name / 'gc_task_vs_baseline_stats_ttest.csv'
    if not f.exists():
        return None
    df = pd.read_csv(f)
    return df


def edge_table(df):
    """One row per (measure, src, tgt, band): min cluster p, FDR flag, signed
    direction for dtrgc, n_subj."""
    rows = []
    for (m, s, tg, b), g in df.groupby(['measure', 'src', 'tgt', 'band'], sort=False):
        task_rows = g[g.pval.notna()]
        p = g.p_cluster_min.dropna()
        p = float(p.iloc[0]) if len(p) else 1.0
        fam = bool(g.sig_p_cluster_fam_fdr.fillna(False).astype(bool).any())
        sig_w = g[g.sig_cluster.fillna(False).astype(bool)]
        ref = sig_w if len(sig_w) else task_rows
        delta = float((ref.gc_mean - ref.baseline_mean).mean()) if len(ref) else 0.0
        rows.append(dict(measure=m, src=s, tgt=tg, band=b, p_cluster=p, fam_fdr=fam,
                         delta=delta, n_subj=int(g.n_subj_perm.max()),
                         sig_spans=span_str(g.window_ms.values, g.sig_cluster.fillna(False).astype(bool).values)))
    return pd.DataFrame(rows)


def fig_gc_matrix(arms, task, stim, out):
    passes = gc_passes(task)
    bands = BANDS + ['beta']
    rows = [(a, p) for a in arms for p in passes]
    cols = [('gc', b) for b in bands] + [('dtrgc', b) for b in bands]
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(1.75 * len(cols) + 1.5, 1.75 * len(rows) + 1.6),
                             layout='constrained', squeeze=False)
    norm_gc = Normalize(0, 3); norm_tr = TwoSlopeNorm(vcenter=0, vmin=-3, vmax=3)
    n_subj_note = {}
    any_data = False
    for i, (a, p) in enumerate(rows):
        df = load_gc(a, task, stim, p)
        et = edge_table(df) if df is not None else None
        if et is not None:
            n_subj_note[a['label']] = int(et.n_subj.max())
            any_data = True
        for j, (m, b) in enumerate(cols):
            ax = axes[i, j]
            M = np.full((5, 5), np.nan)
            ann = {}
            if et is not None:
                sub = et[(et.measure == m) & (et.band == b)]
                for _, r in sub.iterrows():
                    si, ti = ROIS.index(r.src), ROIS.index(r.tgt)
                    v = -np.log10(max(r.p_cluster, 1e-3))
                    if m == 'dtrgc':
                        v *= np.sign(r.delta) if r.delta != 0 else 1
                    M[si, ti] = v
                    if r.p_cluster < 0.05:
                        ann[(si, ti)] = '**' if r.fam_fdr else '*'
            if m not in a['measures']:
                ax.set_axis_off()
                if j == 0:
                    ax.set_axis_on(); ax.set_xticks([]); ax.set_yticks([])
                    ax.set_ylabel(f"{a['label']}\n{pass_label(p)}", fontsize=9)
                    ax.set_frame_on(False)
                continue
            cmap = 'Reds' if m == 'gc' else 'RdBu_r'
            norm = norm_gc if m == 'gc' else norm_tr
            Mplot = np.ma.masked_invalid(M)
            ax.imshow(Mplot, cmap=cmap, norm=norm, aspect='equal')
            ax.set_facecolor('#eeeeee')
            for (si, ti), s in ann.items():
                ax.text(ti, si, s, ha='center', va='center', fontsize=9, fontweight='bold',
                        color='k' if abs(M[si, ti]) < 2 else 'w')
            ax.set_xticks(range(5)); ax.set_yticks(range(5))
            ax.set_xticklabels([ROI_ABBR[r] for r in ROIS], fontsize=7, rotation=90)
            ax.set_yticklabels([ROI_ABBR[r] for r in ROIS], fontsize=7)
            ax.tick_params(length=0)
            if i == 0:
                ax.set_title(('GC row→col\n' if m == 'gc' else 'net TRGC row⇄col\n') + BAND_LABEL[b], fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{a['label']}\n{pass_label(p)}", fontsize=9)
            if et is None:
                ax.text(2, 2, 'missing', ha='center', va='center', fontsize=8)
    if not any_data:
        plt.close(fig); return None
    sm1 = plt.cm.ScalarMappable(cmap='Reds', norm=norm_gc)
    sm2 = plt.cm.ScalarMappable(cmap='RdBu_r', norm=norm_tr)
    cb1 = fig.colorbar(sm1, ax=axes[:, :len(bands)].ravel().tolist(), shrink=0.5, pad=0.01, location='bottom')
    cb1.set_label('directed GC: −log10 cluster p (task > baseline, right-tailed)', fontsize=8)
    cb2 = fig.colorbar(sm2, ax=axes[:, len(bands):].ravel().tolist(), shrink=0.5, pad=0.01, location='bottom')
    cb2.set_label('net Diff-TRGC: sign(task − baseline) × −log10 cluster p (two-tailed); red = more row→col in task', fontsize=8)
    n_note = ', '.join(f'{k}: n = {v}' for k, v in n_subj_note.items())
    arm_desc = '; '.join(f"{a['label']}: {a['n_pcs']} PC, normalize {a['normalize']}, reports {'+'.join(a['measures'])}" for a in arms)
    fig.suptitle(f'GC task vs baseline, {task} / {stim} — order 10, 40 ms @ 500 Hz; {arm_desc}\n'
                 f'cell = smallest cluster p of that edge × band within the stats pass; * p < .05 within edge, ** survives FDR over the 120-test family; '
                 f'gray = no cluster. {n_note}', fontsize=9)
    f = out / f'gc_matrix_{task}_{stim}.png'
    fig.savefig(f, dpi=150); plt.close(fig)
    return f


def _gc_series(df, measure, src, tgt, band):
    g = df[(df.measure == measure) & (df.src == src) & (df.tgt == tgt) & (df.band == band)].sort_values('window_ms')
    return g


def fig_gc_trgc(arms, task, stim, out):
    arms = [a for a in arms if 'dtrgc' in a['measures']]
    dfs = {a['label']: load_gc(a, task, stim, 'group_stats') for a in arms}
    if not arms or all(v is None for v in dfs.values()):
        return None
    fig, axes = plt.subplots(len(PAIRS), len(BANDS), figsize=(4.2 * len(BANDS), 1.9 * len(PAIRS)),
                             sharex=True, layout='constrained', squeeze=False)
    wins = GC_WINDOWS_OF_INTEREST[task]
    for i, (a_, b_) in enumerate(PAIRS):
        for j, band in enumerate(BANDS):
            ax = axes[i, j]
            rows = []; hi = -np.inf; lo = np.inf
            for k, a in enumerate(arms):
                df = dfs[a['label']]
                if df is None:
                    continue
                g = _gc_series(df, 'dtrgc', a_, b_, band)
                if not len(g):
                    continue
                t = g.window_ms.values; m = g.gc_mean.values; se = g.gc_sem.values
                ax.plot(t, m, color=ARM_COLORS[k], lw=1.1, label=f"{a['label']} (n={int(g.n_subj_perm.max())})")
                ax.fill_between(t, m - se, m + se, color=ARM_COLORS[k], alpha=0.15, lw=0)
                ax.axhline(float(g.baseline_mean.iloc[0]), color=ARM_COLORS[k], lw=0.7, ls='--')
                hi = max(hi, float(np.nanmax(m + se))); lo = min(lo, float(np.nanmin(m - se)))
                rows.append((g.sig.fillna(False).astype(bool).values, g.sig_cluster.fillna(False).astype(bool).values, ARM_COLORS[k]))
            if not np.isfinite(hi):
                ax.set_axis_off(); continue
            dy = (hi - lo) * 0.09
            mark_rows(ax, t, rows, hi + 0.4 * dy, dy, star_ms=2.5, bar_lw=2)
            for wn, (wlo, whi) in wins.items():
                ax.axvspan(wlo * 1000, whi * 1000, color='0.9', zorder=0)
            ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5, ls=':')
            ax.set_ylim(lo - dy, hi + dy * (len(rows) + 1))
            ax.spines[['top', 'right']].set_visible(False)
            ax.tick_params(labelsize=7)
            if j == 0:
                ax.set_ylabel(f'{ROI_ABBR[a_]} ⇄ {ROI_ABBR[b_]}\nnet TRGC', fontsize=8)
            if i == 0:
                ax.set_title(BAND_LABEL[band], fontsize=10)
                if j == 0:
                    ax.legend(fontsize=7, frameon=False, loc='upper left')
            if i == len(PAIRS) - 1:
                ax.set_xlabel('window (ms)')
    fig.suptitle(f'Net Diff-TRGC time courses, {task} / {stim}: positive = net first→second ROI of the pair label\n'
                 'mean ± SEM; dashed = each arm\'s baseline level; gray = a-priori windows of interest; '
                 'marks above, one row per arm: * pointwise t vs baseline (uncorrected), bar = cluster (full-span pass)', fontsize=9)
    f = out / f'gc_trgc_{task}_{stim}.png'
    fig.savefig(f, dpi=140); plt.close(fig)
    return f


def fig_gc_directed(arm, task, stim, out, color_idx=0):
    df = load_gc(arm, task, stim, 'group_stats')
    if df is None:
        return None
    fig, axes = plt.subplots(len(PAIRS), len(BANDS), figsize=(4.2 * len(BANDS), 1.9 * len(PAIRS)),
                             sharex=True, layout='constrained', squeeze=False)
    wins = GC_WINDOWS_OF_INTEREST[task]
    c_f, c_r = '#1f77b4', '#ff7f0e'
    for i, (a_, b_) in enumerate(PAIRS):
        for j, band in enumerate(BANDS):
            ax = axes[i, j]
            rows = []; hi = -np.inf; lo = np.inf
            for (s, tg, col, lab) in [(a_, b_, c_f, f'{ROI_ABBR[a_]}→{ROI_ABBR[b_]}'), (b_, a_, c_r, f'{ROI_ABBR[b_]}→{ROI_ABBR[a_]}')]:
                g = _gc_series(df, 'gc', s, tg, band)
                if not len(g):
                    continue
                t = g.window_ms.values; m = g.gc_mean.values; se = g.gc_sem.values
                ax.plot(t, m, color=col, lw=1.1, label=lab)
                ax.fill_between(t, m - se, m + se, color=col, alpha=0.15, lw=0)
                ax.axhline(float(g.baseline_mean.iloc[0]), color=col, lw=0.7, ls='--')
                hi = max(hi, float(np.nanmax(m + se))); lo = min(lo, float(np.nanmin(m - se)))
                rows.append((g.sig.fillna(False).astype(bool).values, g.sig_cluster.fillna(False).astype(bool).values, col))
            if not np.isfinite(hi):
                ax.set_axis_off(); continue
            dy = (hi - lo) * 0.09
            mark_rows(ax, t, rows, hi + 0.4 * dy, dy, star_ms=2.5, bar_lw=2)
            for wn, (wlo, whi) in wins.items():
                ax.axvspan(wlo * 1000, whi * 1000, color='0.9', zorder=0)
            ax.axvline(0, color='k', lw=0.5, ls=':')
            ax.set_ylim(max(0, lo - dy), hi + dy * (len(rows) + 1))
            ax.spines[['top', 'right']].set_visible(False)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=6, frameon=False, loc='upper left', ncol=2)
            if j == 0:
                ax.set_ylabel(f'{ROI_ABBR[a_]} / {ROI_ABBR[b_]}\nGC', fontsize=8)
            if i == 0:
                ax.set_title(BAND_LABEL[band], fontsize=10)
            if i == len(PAIRS) - 1:
                ax.set_xlabel('window (ms)')
    n = int(df.n_subj_perm.max())
    fig.suptitle(f"Directed GC time courses, {task} / {stim}, {arm['label']} arm ({arm['method']} / {arm['feat']}), n = {n}\n"
                 'mean ± SEM, both directions of each pair; dashed = baseline level; gray = windows of interest; '
                 'marks above (one row per direction): * pointwise t task > baseline (uncorrected), bar = cluster', fontsize=9)
    f = out / f"gc_directed_{task}_{stim}_{arm['label']}.png"
    fig.savefig(f, dpi=140); plt.close(fig)
    return f


def gc_summary_rows(arms, task, stim):
    rows = []
    for a in arms:
        for p in gc_passes(task):
            df = load_gc(a, task, stim, p)
            if df is None:
                continue
            et = edge_table(df)
            et = et[et.measure.isin(a['measures'])].copy()
            et.insert(0, 'pass', pass_label(p)); et.insert(0, 'arm', a['label'])
            et.insert(0, 'stim', stim); et.insert(0, 'task', task)
            rows.append(et)
    return rows


# ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--arms', nargs='+', default=DEFAULT_ARMS,
                    help='label=method:feature_mode[:n_pcs[:normalize[:gc|dtrgc|gc+dtrgc]]]; first = reference')
    ap.add_argument('--tasks', nargs='+', default=['overtProd', 'perception'])
    ap.add_argument('--stage', choices=['decoding', 'gc', 'all'], default='all')
    ap.add_argument('--out-dir', default=None)
    args = ap.parse_args()
    arms = [parse_arm(s) for s in args.arms]
    root = Path(args.out_dir) if args.out_dir else REVIEW_ROOT
    (root / 'decoding').mkdir(parents=True, exist_ok=True)
    (root / 'gc').mkdir(parents=True, exist_ok=True)
    print(f'review -> {root}')
    for a in arms:
        print(f"  arm {a['label']}: {a['method']} / {a['feat']} / GC tag {a['gc_tag']}")

    if args.stage in ('decoding', 'all'):
        out = root / 'decoding'
        summary, comparison = [], []
        for task in args.tasks:
            data = {a['label']: {} for a in arms}
            for a in arms:
                for stim in STIMS:
                    dec = load_decoding(a, task, stim)
                    if dec:
                        data[a['label']][stim] = dec
                        print(f"  {task}/{stim} {a['label']}: {len(dec['subjects'])} subjects, "
                              f"stats {'yes' if dec['stats'] is not None else 'NO'}, latency {'yes' if dec['lat'] is not None else 'NO'}")
                    else:
                        print(f"  {task}/{stim} {a['label']}: no results")
            fig_decoding_curves(arms, data, task, out)
            fig_decoding_rois(arms, data, task, out)
            if len(arms) >= 2:
                comparison += arm_difference(arms, data, task, out)
            summary += decoding_summary_rows(arms, data, task)
        s = pd.DataFrame(summary)
        s.to_csv(out / 'decoding_summary.csv', index=False)
        with open(out / 'decoding_summary.md', 'w') as fh:
            fh.write('# Decoding summary, translation arm (sensorEquiv, logistic tuned, 40/5 ms)\n\n')
            fh.write('peak = group-mean accuracy; clusters = cluster permutation vs chance (p < .05); '
                     'onset = jackknife onset latency; window columns = window-mean accuracy, difference '
                     'to the Visual control and paired p.\n\n')
            for (task, stim), g in s.groupby(['task', 'stim'], sort=False):
                fh.write(f'## {task} / {stim}\n\n')
                cols = [c for c in g.columns if c not in ('task', 'stim')]
                fh.write(md_table(g[cols]) + '\n\n')
        if comparison:
            c = pd.DataFrame(comparison)
            c.to_csv(out / 'decoding_arm_comparison.csv', index=False)
            pd.set_option('display.width', 250)
            print('\nDeFleCT minus vertex, paired at the vertex peak:')
            print(c[['task', 'stim', 'roi', 'r_time', 'peak_vertex', 'peak_deflect', 'd_at_vertex_peak',
                     'd_sem', 'p_at_vertex_peak', 'cluster_spans']].round(3).to_string(index=False))
        print(f'\n-> {out}')

    if args.stage in ('gc', 'all'):
        out = root / 'gc'
        all_rows = []
        for task in args.tasks:
            for stim in STIMS:
                f = fig_gc_matrix(arms, task, stim, out)
                print(f'  {task}/{stim}: matrix {f.name if f else "no GC results"}')
                if f is None:
                    continue
                fig_gc_trgc(arms, task, stim, out)
                for a in arms:
                    if 'gc' in a['measures']:
                        fig_gc_directed(a, task, stim, out)
                all_rows += gc_summary_rows(arms, task, stim)
        if all_rows:
            g = pd.concat(all_rows, ignore_index=True)
            g.to_csv(out / 'gc_edge_summary.csv', index=False)
            sig = g[g.p_cluster < 0.05].copy()
            with open(out / 'gc_significant_edges.md', 'w') as fh:
                fh.write('# GC edges with a cluster p < .05 (task vs baseline), by arm and stats pass\n\n')
                fh.write('measure gc = directed GC (right-tailed, task > baseline); dtrgc = net Diff-TRGC '
                         '(two-tailed; delta > 0 = more src→tgt in the task window). fam_fdr = survives FDR '
                         'over the 120 edge × band tests of that pass.\n\n')
                for (task, stim, arm, p), gg in sig.groupby(['task', 'stim', 'arm', 'pass'], sort=False):
                    fh.write(f'## {task} / {stim} — {arm} — {p}  (n = {int(gg.n_subj.max())})\n\n')
                    fh.write(gg[['measure', 'src', 'tgt', 'band', 'p_cluster', 'fam_fdr', 'delta', 'sig_spans']]
                             .sort_values('p_cluster').round(4).pipe(md_table) + '\n\n')
            print('\nsignificant edges per cell (cluster p < .05 / FDR):')
            print(sig.groupby(['task', 'stim', 'arm', 'pass', 'measure']).agg(
                n_sig=('p_cluster', 'size'), n_fdr=('fam_fdr', 'sum')).to_string())
        print(f'\n-> {out}')


if __name__ == '__main__':
    main()
