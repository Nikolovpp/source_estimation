#!/usr/bin/env python3
"""Compare the BSMART-parametric AR GC against MNE state-space (cwt) GC, per pair.

Head-to-head of the two estimators on the *same* data and config:
  * BSMART   : parametric MVAR, ``run_granger.py``  (GC_source_space/)
  * MNE (cwt): state-space, ``run_granger_mne.py``  (GC_source_space_mne/)

Both write the same npz schema, so this reads a subject folder from each and,
per ROI pair, produces:

  1. *normalised net-GC time courses* — each estimator's group-mean net GC
     z-scored over time and overlaid (Pearson r annotated). Answers "do they
     agree on direction and timing?" without a dual axis, since the raw scales
     differ several-fold (BSMART is order-inflated; see the chapter §12.6).
  2. *per-subject scatter* — BSMART net GC vs MNE net GC across subjects, with
     Spearman rho and the fitted slope (the scale ratio). Answers "do subjects
     rank the same, and how big is the scale gap?"

BSMART's ``window_ms`` is the window *start*; pass ``--bsmart-win-ms`` to
recentre it (+win/2) onto MNE's per-sample (centred) time axis before aligning.

Usage
-----
    python compare_bsmart_vs_mne.py \\
        --bsmart-dir  .../GC_source_space/overtProd/LCMV/custom/vertex_selectkbest/leakage_corrected/order25_win250ms_fs200/all_rois/prodDiff \\
        --mne-dir     .../GC_source_space_mne/overtProd/LCMV/custom/vertex_selectkbest/leakage_corrected/ssgc_cwt_mo25_sw250ms_fs200/pairs_.../prodDiff \\
        --bsmart-win-ms 250 --band low_beta \\
        --pairs awfa-lh:ifc-lh tpc-lh:ifc-lh
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import SUBJECT_IDS, DECODE_OUTPUT_ROOT
from compare_gc_mne_configs import load_config, grid, PAL, config_dir
from run_granger import gc_tag as bsmart_gc_tag, roiset_tag
from run_granger_mne import gc_tag_mne, pairs_tag

GC_ROOT = DECODE_OUTPUT_ROOT.parent / 'GC_source_space'
GC_ROOT_MNE = DECODE_OUTPUT_ROOT.parent / 'GC_source_space_mne'


def build_dirs(a):
    """Construct the BSMART and MNE subject folders from descriptors, unless
    explicit --bsmart-dir/--mne-dir were given."""
    if a.bsmart_dir and a.mne_dir:
        return Path(a.bsmart_dir), Path(a.mne_dir)
    pair_tuples = [tuple(p.replace('->', ':').split(':')) for p in a.pairs] \
        if a.pairs else None
    bs_subset = a.bsmart_roi_subset or (
        sorted({r for pr in pair_tuples for r in pr}) if pair_tuples else None)
    bs = config_dir(GC_ROOT, a.task, a.method, a.atlas, a.feature_mode,
                    a.leakage_correction,
                    bsmart_gc_tag(a.order, a.win_ms, a.target_fs, 'none'),
                    roiset_tag(bs_subset), a.stim_class)
    # The MNE side may use a DIFFERENT config (it cannot use BSMART's short window
    # — it is bound by ~1/T). --mne-* override; default to the shared config.
    mne_lags = a.mne_gc_n_lags if a.mne_gc_n_lags is not None else a.order
    mne_win = a.mne_win_ms if a.mne_win_ms is not None else a.win_ms
    mne_fs = a.mne_target_fs if a.mne_target_fs is not None else a.target_fs
    mne = config_dir(GC_ROOT_MNE, a.task, a.method, a.atlas, a.feature_mode,
                     a.leakage_correction,
                     gc_tag_mne(mne_lags, mne_win, mne_fs),
                     pairs_tag(pair_tuples) if pair_tuples
                     else roiset_tag(a.bsmart_roi_subset), a.stim_class)
    return bs, mne


def net_arr(cfg, k, flip):
    """Net GC (fwd − rev) per subject, oriented to the reference direction.

    ``run_granger.py`` (BSMART, roi-subset) stores pairs in ROI-index order
    (i<j); ``run_granger_mne.py`` (--pairs) stores them in the order given. So
    the same pair can be stored (a,b) in one and (b,a) in the other. ``flip``
    swaps fwd/rev so both configs measure a→b, not whatever order they stored.
    """
    fwd, rev = cfg['fxy'][:, k, :], cfg['fyx'][:, k, :]
    if flip:
        fwd, rev = rev, fwd
    return fwd - rev                              # (n_subj, n_time)


def zscore(x):
    x = np.asarray(x, float)
    s = np.nanstd(x)
    return (x - np.nanmean(x)) / (s if s > 0 else 1.0)


def align_pairs_named(A, B, wanted):
    """Match pairs across configs, orientation-insensitive.

    Returns (ref_pair, ka, flipA, kb, flipB) so each config can be re-oriented
    to the reference a→b direction.
    """
    def find(pairs, a, b):
        for k, (x, y) in enumerate(pairs):
            if (x, y) == (a, b):
                return k, False
            if (x, y) == (b, a):
                return k, True
        return None

    refs = ([tuple(w.replace('->', ':').split(':')) for w in wanted]
            if wanted else list(A['pairs']))
    out = []
    for (a, b) in refs:
        ra, rb = find(A['pairs'], a, b), find(B['pairs'], a, b)
        if ra and rb:
            out.append(((a, b), ra[0], ra[1], rb[0], rb[1]))
    return out


def fig_timecourses(A, B, common, band, onset, out):
    nrow, ncol = grid(len(common))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.7 * ncol, 3.4 * nrow),
                             squeeze=False)
    for ax_i, (p, ka, fa, kb, fb) in enumerate(common):
        ax = axes[ax_i // ncol][ax_i % ncol]
        na, ta = np.nanmean(net_arr(A, ka, fa), axis=0), A['times']
        nb, tb = np.nanmean(net_arr(B, kb, fb), axis=0), B['times']
        # common grid over the overlapping time range for r + a shared x-axis
        lo, hi = max(ta.min(), tb.min()), min(ta.max(), tb.max())
        gt = np.linspace(lo, hi, 200)
        za = zscore(np.interp(gt, ta, na))
        zb = zscore(np.interp(gt, tb, nb))
        r = pearsonr(za, zb)[0]
        ax.plot(gt, za, color=PAL['A'], lw=1.8, label='BSMART')
        ax.plot(gt, zb, color=PAL['B'], lw=1.8, ls='--', label='MNE (cwt)')
        ax.axhline(0, color='k', lw=0.6); ax.axvline(onset, color='k', lw=0.8, ls=':')
        ax.set_title(f'{p[0]} → {p[1]}   (r={r:+.2f})', fontsize=10)
        ax.set_xlabel('time (s)'); ax.set_ylabel(f'net GC ({band}), z')
        if ax_i == 0:
            ax.legend(fontsize=8, frameon=False)
    for j in range(len(common), nrow * ncol):
        axes[j // ncol][j % ncol].axis('off')
    fig.suptitle(f'BSMART vs MNE (cwt): net-GC temporal pattern, {band} '
                 f'(z-scored; scales differ, see scatter)', fontsize=11, y=1.0)
    fig.tight_layout(); fig.savefig(out, dpi=130, bbox_inches='tight'); plt.close(fig)
    return out


def fig_scatter(A, B, common, band, out):
    # subjects present in both, matched by ID
    common_subj = [s for s in A['found'] if s in B['found']]
    ia = {s: i for i, s in enumerate(A['found'])}
    ib = {s: i for i, s in enumerate(B['found'])}
    rowsA = [ia[s] for s in common_subj]; rowsB = [ib[s] for s in common_subj]
    nrow, ncol = grid(len(common))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.6 * nrow),
                             squeeze=False)
    for ax_i, (p, ka, fa, kb, fb) in enumerate(common):
        ax = axes[ax_i // ncol][ax_i % ncol]
        xa = np.nanmean(net_arr(A, ka, fa)[rowsA], axis=1)
        yb = np.nanmean(net_arr(B, kb, fb)[rowsB], axis=1)
        m = np.isfinite(xa) & np.isfinite(yb)
        rho = spearmanr(xa[m], yb[m]).correlation if m.sum() > 2 else np.nan
        slope = np.polyfit(xa[m], yb[m], 1)[0] if m.sum() > 2 else np.nan
        ax.scatter(xa, yb, s=36, color=PAL['A'], edgecolor='white', linewidth=.6)
        ax.axhline(0, color='k', lw=.6, alpha=.5); ax.axvline(0, color='k', lw=.6, alpha=.5)
        ax.set_title(f'{p[0]} → {p[1]}   (ρ={rho:+.2f}, slope={slope:.2f})', fontsize=9)
        ax.set_xlabel('BSMART net GC'); ax.set_ylabel('MNE net GC')
    for j in range(len(common), nrow * ncol):
        axes[j // ncol][j % ncol].axis('off')
    fig.suptitle(f'BSMART vs MNE (cwt): per-subject net GC, {band} '
                 f'(n={len(common_subj)}; slope = MNE/BSMART scale ratio)',
                 fontsize=11, y=1.0)
    fig.tight_layout(); fig.savefig(out, dpi=130, bbox_inches='tight'); plt.close(fig)
    return out


def parse_args():
    p = argparse.ArgumentParser(description='Compare BSMART vs MNE state-space GC')
    # Either give explicit folders, or descriptors (built from the run configs).
    p.add_argument('--bsmart-dir', default=None,
                   help='.../GC_source_space/.../stim_class folder (run_granger.py)')
    p.add_argument('--mne-dir', default=None,
                   help='.../GC_source_space_mne/.../stim_class folder (run_granger_mne.py)')
    p.add_argument('--task', default='overtProd'); p.add_argument('--stim-class', default='prodDiff')
    p.add_argument('--method', default='LCMV'); p.add_argument('--atlas', default='custom')
    p.add_argument('--feature-mode', default='vertex_selectkbest')
    p.add_argument('--leakage-correction', action='store_true', default=False)
    p.add_argument('--order', type=int, default=25, help='BSMART order (and MNE gc_n_lags unless --mne-gc-n-lags)')
    p.add_argument('--win-ms', type=float, default=250.0, help='BSMART window (ms); MNE too unless --mne-win-ms')
    p.add_argument('--target-fs', type=float, default=200.0, help='BSMART fs (Hz); MNE too unless --mne-target-fs')
    # MNE can't share BSMART's short window (it is bound by ~1/T). These override
    # the MNE side so you can compare, e.g., BSMART@60ms vs MNE@250ms.
    p.add_argument('--mne-gc-n-lags', type=int, default=None)
    p.add_argument('--mne-win-ms', type=float, default=None)
    p.add_argument('--mne-target-fs', type=float, default=None)
    p.add_argument('--bsmart-roi-subset', nargs='+', default=None,
                   help='the --roi-subset used for the BSMART run (locates its dir); '
                        'defaults to the ROIs in --pairs')
    p.add_argument('--band', default='low_beta',
                   choices=['theta', 'alpha', 'low_beta', 'high_beta'])
    p.add_argument('--pairs', nargs='+', default=None,
                   help='ROIa:ROIb subset to plot (default: all common pairs)')
    p.add_argument('--bsmart-win-ms', type=float, default=None,
                   help='shift BSMART window-START times by +win/2 to centre them '
                        '(defaults to --win-ms)')
    p.add_argument('--onset', type=float, default=0.0)
    p.add_argument('--subjects', nargs='+', default=None)
    p.add_argument('--out-dir', default=None)
    return p.parse_args()


def main():
    a = parse_args()
    subjects = a.subjects if a.subjects else SUBJECT_IDS
    if a.bsmart_win_ms is None:
        a.bsmart_win_ms = a.win_ms
    bs_dir, mne_dir = build_dirs(a)
    a.bsmart_dir, a.mne_dir = str(bs_dir), str(mne_dir)
    print(f'BSMART <- {bs_dir}')
    print(f'MNE    <- {mne_dir}')
    A = load_config(bs_dir, a.task, a.stim_class, subjects, a.band)
    B = load_config(mne_dir, a.task, a.stim_class, subjects, a.band)
    if A is None or B is None:
        print('ERROR: no subject npz found in one/both dirs.'); return
    if a.bsmart_win_ms:                       # recentre BSMART window-start times
        A['times'] = A['times'] + a.bsmart_win_ms / 2000.0
    common = align_pairs_named(A, B, a.pairs)
    if not common:
        print('ERROR: no common pairs between the two dirs.'); return
    print(f'BSMART n={len(A["found"])}  MNE n={len(B["found"])}  '
          f'common pairs={len(common)}  band={a.band}')
    if B.get('under_resolved') and a.band in B['under_resolved']:
        print(f'  NOTE: MNE band "{a.band}" is under-resolved at this window.')

    out_dir = Path(a.out_dir) if a.out_dir else GC_ROOT_MNE / '_bsmart_vs_mne'
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f'{a.task}_{a.stim_class}_bsmart_vs_mne_{a.band}'
    f1 = fig_timecourses(A, B, common, a.band, a.onset,
                         out_dir / f'{stem}_timecourse.png')
    f2 = fig_scatter(A, B, common, a.band, out_dir / f'{stem}_scatter.png')

    print('\nTask-window net GC (i→j − j→i), BSMART vs MNE:')
    cs = [s for s in A['found'] if s in B['found']]
    ia = {s: i for i, s in enumerate(A['found'])}; ib = {s: i for i, s in enumerate(B['found'])}
    rA = [ia[s] for s in cs]; rB = [ib[s] for s in cs]
    for p, ka, fa, kb, fb in common:
        xb = np.nanmean(net_arr(A, ka, fa)[rA])
        xm = np.nanmean(net_arr(B, kb, fb)[rB])
        ratio = xm / xb if xb else np.nan
        print(f'  {p[0]:>8s}->{p[1]:<8s}  BSMART {xb:+.4f}   MNE {xm:+.4f}   MNE/BSMART {ratio:.2f}')
    print(f'\nFigures:\n  {f1}\n  {f2}')


if __name__ == '__main__':
    main()
