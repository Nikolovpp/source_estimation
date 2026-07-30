#!/usr/bin/env python3
"""Compare two MNE state-space GC parameter configs, per ROI pair, across subjects.

Reads the per-subject ``.npz`` written by ``run_granger_mne.py`` for two
configurations (default the A/B presets: MO25/SW250/fs200 vs MO15/SW60/fs500)
and produces subplot figures — one subplot per ROI pair:

  1. *net-GC time courses*  — group-mean net GC (i→j minus j→i) through the
     epoch, config A vs config B, ±SEM. The headline "does the config change
     the answer" view.
  2. *directional bars*     — task-window-mean GC for both directions and both
     configs, with the Diff-TRGC value annotated. The compact summary.

Light (reads saved npz), so run it after the two heavy ``run_granger_mne.py``
jobs finish. Figures land in ``GC_source_space_mne/_config_comparison/``.

Usage
-----
    python compare_gc_mne_configs.py --task overtProd --stim-class prodDiff \\
        --method LCMV --atlas custom --feature-mode vertex_selectkbest \\
        --leakage-correction \\
        --pairs awfa-lh:ifc-lh tpc-lh:ifc-lh tpc-lh:pmc-lh ifc-lh:pmc-lh \\
        --band low_beta
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import SUBJECT_IDS, DECODE_OUTPUT_ROOT
from run_granger_mne import gc_tag_mne, pairs_tag, PRESETS
from run_granger import roiset_tag

GC_ROOT = DECODE_OUTPUT_ROOT.parent / 'GC_source_space_mne'

# Okabe-Ito (colour-blind safe): direction = colour, config = line style / hatch.
PAL = {'fwd': '#0072B2', 'rev': '#D55E00', 'A': '#0072B2', 'B': '#D55E00',
       'neutral': '#7F7F7F'}


def config_dir(root, task, method, atlas, feat, leakage, tag, subdir, stim):
    leak = 'leakage_corrected' if leakage else 'raw'
    return root / task / method / atlas / feat / leak / tag / subdir / stim


def load_config(cdir, task, stim, subjects, band):
    """Stack one config's subjects. Returns dict or None if nothing found."""
    pairs, fxy, fyx, dtr, found, times, roi = None, [], [], [], [], None, None
    ur = []
    for s in subjects:
        f = cdir / f'{s}_{task}_{stim}.npz'
        if not f.exists():
            continue
        d = np.load(f, allow_pickle=True)
        roi = list(map(str, d['roi_names']))
        pi, pj = d['pair_i'], d['pair_j']
        p = [(roi[a], roi[b]) for a, b in zip(pi, pj)]
        if pairs is None:
            pairs, times = p, d['window_ms']
            ur = list(map(str, d['under_resolved'])) if 'under_resolved' in d.files else []
        elif p != pairs:
            # same shape + different pair order => silently averaging different
            # edges across subjects (see granger_stats.load_gc_group)
            raise ValueError(
                f'{s} has a different pair set/order than {found[0]} in {cdir}:\n'
                f'  {found[0]}: {pairs}\n  {s}: {p}')
        fxy.append(d[f'fxy_{band}']); fyx.append(d[f'fyx_{band}'])
        if f'dtrgc_{band}' in d.files:
            dtr.append(d[f'dtrgc_{band}'])
        found.append(s)
    if not found:
        return None
    out = dict(pairs=pairs, times=np.asarray(times) / 1000.0, found=found,
               fxy=np.asarray(fxy), fyx=np.asarray(fyx),
               dtr=np.asarray(dtr) if dtr else None, under_resolved=ur)
    return out


def align_pairs(A, B):
    """Common (name_i, name_j) pairs in A, with each config's row index."""
    ib = {p: k for k, p in enumerate(B['pairs'])}
    common = [(p, ka, ib[p]) for ka, p in enumerate(A['pairs']) if p in ib]
    return common


def grid(n):
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    return nrow, ncol


def sem(x, axis=0):
    return np.nanstd(x, axis=axis) / max(1, np.sqrt(np.sum(~np.isnan(x[:, 0]))))


def fig_timecourses(A, B, common, band, labels, onset, out):
    nrow, ncol = grid(len(common))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.4 * nrow),
                             squeeze=False)
    for ax_i, (p, ka, kb) in enumerate(common):
        ax = axes[ax_i // ncol][ax_i % ncol]
        for cfg, kk, col, ls, lab in [(A, ka, PAL['A'], '-', labels[0]),
                                      (B, kb, PAL['B'], '--', labels[1])]:
            net = cfg['fxy'][:, kk, :] - cfg['fyx'][:, kk, :]   # (nsubj, ntime)
            m = np.nanmean(net, axis=0); e = sem(net)
            ax.plot(cfg['times'], m, color=col, ls=ls, lw=1.8, label=lab)
            ax.fill_between(cfg['times'], m - e, m + e, color=col, alpha=0.15)
        ax.axhline(0, color='k', lw=0.6)
        ax.axvline(onset, color='k', lw=0.8, ls=':')
        ax.set_title(f'{p[0]} → {p[1]}', fontsize=10)
        ax.set_xlabel('time (s)'); ax.set_ylabel(f'net GC ({band})')
        if ax_i == 0:
            ax.legend(fontsize=8, frameon=False)
    for j in range(len(common), nrow * ncol):
        axes[j // ncol][j % ncol].axis('off')
    fig.suptitle(f'Net MNE state-space GC (cwt), {band} — {labels[0]} vs {labels[1]} '
                 f'(group mean ±SEM, n={len(A["found"])}/{len(B["found"])})',
                 fontsize=11, y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return out


def fig_bars(A, B, common, band, labels, out):
    nrow, ncol = grid(len(common))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.4 * nrow),
                             squeeze=False)
    for ax_i, (p, ka, kb) in enumerate(common):
        ax = axes[ax_i // ncol][ax_i % ncol]
        vals, errs, cols, hatch = [], [], [], []
        for cfg, kk, hh in [(A, ka, ''), (B, kb, '//')]:
            fwd = np.nanmean(cfg['fxy'][:, kk, :], axis=1)   # per-subj task mean
            rev = np.nanmean(cfg['fyx'][:, kk, :], axis=1)
            vals += [np.nanmean(fwd), np.nanmean(rev)]
            errs += [np.nanstd(fwd) / np.sqrt(len(fwd)),
                     np.nanstd(rev) / np.sqrt(len(rev))]
            cols += [PAL['fwd'], PAL['rev']]; hatch += [hh, hh]
        x = np.arange(4)
        bars = ax.bar(x, vals, yerr=errs, color=cols, capsize=3,
                      edgecolor='white')
        for b, h in zip(bars, hatch):
            b.set_hatch(h)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{labels[0]}\ni→j', f'{labels[0]}\nj→i',
                            f'{labels[1]}\ni→j', f'{labels[1]}\nj→i'], fontsize=7)
        ax.set_title(f'{p[0]} → {p[1]}', fontsize=10)
        ax.set_ylabel(f'task-mean GC ({band})')
        # annotate Diff-TRGC if present
        txt = []
        for cfg, kk, lab in [(A, ka, labels[0]), (B, kb, labels[1])]:
            if cfg['dtr'] is not None:
                tv = np.nanmean(cfg['dtr'][:, kk, :])
                txt.append(f'TRGC {lab}={tv:+.4f}')
        if txt:
            ax.text(0.5, 0.97, '\n'.join(txt), transform=ax.transAxes,
                    fontsize=7, va='top', ha='center', color=PAL['neutral'])
    for j in range(len(common), nrow * ncol):
        axes[j // ncol][j % ncol].axis('off')
    fig.suptitle(f'Task-window directional MNE GC (cwt), {band} (i→j blue, j→i '
                 f'orange; config {labels[1]} hatched)', fontsize=11, y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return out


def parse_args():
    p = argparse.ArgumentParser(description='Compare two MNE GC configs by ROI pair')
    p.add_argument('--task', required=True, choices=['perception', 'overtProd'])
    p.add_argument('--stim-class', required=True, choices=['prodDiff', 'percDiff'])
    p.add_argument('--method', required=True, choices=['dSPM', 'LCMV'])
    p.add_argument('--atlas', default='custom')
    p.add_argument('--feature-mode', default='vertex_selectkbest')
    p.add_argument('--leakage-correction', action='store_true', default=False)
    p.add_argument('--pairs', nargs='+', default=None, metavar='ROIa:ROIb',
                   help='the same --pairs passed to run_granger_mne (locates the dir)')
    p.add_argument('--roi-subset', nargs='+', default=None,
                   help='used only if --pairs not given (all-pairs run)')
    p.add_argument('--preset-a', default='A'); p.add_argument('--preset-b', default='B')
    p.add_argument('--band', default='low_beta',
                   choices=['theta', 'alpha', 'low_beta', 'high_beta'])
    p.add_argument('--onset', type=float, default=0.0,
                   help='event-onset time (s) for the reference line')
    p.add_argument('--subjects', nargs='+', default=None)
    p.add_argument('--root', default=str(GC_ROOT))
    p.add_argument('--out-dir', default=None)
    return p.parse_args()


def main():
    a = parse_args()
    subjects = a.subjects if a.subjects else SUBJECT_IDS
    root = __import__('pathlib').Path(a.root)
    subdir = pairs_tag([tuple(s.replace('->', ':').split(':')) for s in a.pairs]) \
        if a.pairs else roiset_tag(a.roi_subset)
    tagA = gc_tag_mne(**PRESETS[a.preset_a])
    tagB = gc_tag_mne(**PRESETS[a.preset_b])
    labels = (a.preset_a, a.preset_b)

    dA = config_dir(root, a.task, a.method, a.atlas, a.feature_mode,
                    a.leakage_correction, tagA, subdir, a.stim_class)
    dB = config_dir(root, a.task, a.method, a.atlas, a.feature_mode,
                    a.leakage_correction, tagB, subdir, a.stim_class)
    print(f'config {labels[0]} <- {dA}')
    print(f'config {labels[1]} <- {dB}')
    A = load_config(dA, a.task, a.stim_class, subjects, a.band)
    B = load_config(dB, a.task, a.stim_class, subjects, a.band)
    if A is None or B is None:
        print('ERROR: no subject files found for one/both configs. Run '
              'run_granger_mne.py --preset A and --preset B first.')
        return
    common = align_pairs(A, B)
    print(f'{len(common)} common pairs; n subjects A={len(A["found"])} B={len(B["found"])}')
    if A['under_resolved'] or B['under_resolved']:
        print(f'  under-resolved bands — {labels[0]}: {A["under_resolved"]}, '
              f'{labels[1]}: {B["under_resolved"]}')
    if a.band in A['under_resolved'] or a.band in B['under_resolved']:
        print(f'  NOTE: band "{a.band}" is under-resolved in a config; '
              'the comparison for it is estimator-limited, not neural.')

    out_dir = __import__('pathlib').Path(a.out_dir) if a.out_dir else \
        root / '_config_comparison'
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f'{a.task}_{a.stim_class}_{labels[0]}_vs_{labels[1]}_{a.band}'
    f1 = fig_timecourses(A, B, common, a.band, labels, a.onset,
                         out_dir / f'{stem}_netgc_timecourse.png')
    f2 = fig_bars(A, B, common, a.band, labels, out_dir / f'{stem}_directional_bars.png')

    # printed summary
    print('\nTask-window means (i→j / j→i / net / TRGC):')
    for p, ka, kb in common:
        for cfg, kk, lab in [(A, ka, labels[0]), (B, kb, labels[1])]:
            fwd = np.nanmean(cfg['fxy'][:, kk, :]); rev = np.nanmean(cfg['fyx'][:, kk, :])
            tr = np.nanmean(cfg['dtr'][:, kk, :]) if cfg['dtr'] is not None else np.nan
            print(f'  {p[0]:>8s}->{p[1]:<8s} [{lab}] '
                  f'{fwd:.4f} / {rev:.4f} / {fwd-rev:+.4f} / {tr:+.4f}')
    print(f'\nFigures:\n  {f1}\n  {f2}')


if __name__ == '__main__':
    main()
