"""
Group statistics and plotting for source/sensor Granger causality.

Aggregates the per-subject GC ``.npz`` files written by ``run_granger.py``
/ ``run_granger_sensor.py``, computes the subject-mean band-limited GC
time courses with SEM, and runs the MATLAB task-vs-baseline test at each
task time point against the subject-averaged baseline level.  Two tests
are available via ``--test``: a right-tailed one-sample Student's t-test
(``ttest``, matches ``production_pwgc_data_to_python.m`` and the v4
figures) or a right-tailed Wilcoxon signed-rank test (``signrank``,
matches the v3 figures).  Both share the identical one-sample /
right-tailed / vs-scalar-baseline design and differ only in parametric
vs non-parametric.  Then renders per-edge figures.

CLI
---
    python granger_stats.py --gc-dir <dir with subject .npz> --task overtProd
    # or point it at a source/sensor run by its parameters:
    python granger_stats.py --space source --task overtProd --stim-class prodDiff \\
        --method dSPM --atlas HCPMMP1 --feature-mode vertex_selectkbest \\
        --order 10 --win-ms 40 --target-fs 500
"""
import os
import sys
import glob
import argparse

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DECODE_OUTPUT_ROOT, BASELINE_WINDOWS, DECODE_TMIN
from granger import DEFAULT_BANDS
from run_granger import gc_tag, roiset_tag, GC_OUTPUT_ROOT

GC_SENSOR_OUTPUT_ROOT = DECODE_OUTPUT_ROOT.parent / 'GC_sensor_space'


# ─────────────────────────────────────────────────────────────────────
# Loading / aggregation
# ─────────────────────────────────────────────────────────────────────
def load_gc_group(gc_dir, bands=None):
    """Load every subject ``.npz`` in ``gc_dir`` into stacked arrays.

    Returns a dict with ``roi_names``, ``pair_i``, ``pair_j``,
    ``window_ms``, ``subjects``, and ``fxy``/``fyx`` (and ``dtrgc`` if
    present) each a dict {band: (n_subj, n_pairs, n_win)}.
    """
    if bands is None:
        bands = DEFAULT_BANDS
    band_names = list(bands)
    files = sorted(glob.glob(os.path.join(str(gc_dir), '*.npz')))
    if not files:
        raise FileNotFoundError(f'No GC .npz files in {gc_dir}')

    subjects, ref = [], None
    fxy = {b: [] for b in band_names}
    fyx = {b: [] for b in band_names}
    has_trgc = None
    dtr = {b: [] for b in band_names}
    for f in files:
        d = np.load(f, allow_pickle=True)
        subjects.append(os.path.basename(f).split('_')[0])
        if ref is None:
            ref = {
                'roi_names': list(d['roi_names']),
                'pair_i': d['pair_i'], 'pair_j': d['pair_j'],
                'window_ms': d['window_ms'],
            }
        if has_trgc is None:
            has_trgc = f'dtrgc_{band_names[0]}' in d
        for b in band_names:
            fxy[b].append(d[f'fxy_{b}'])
            fyx[b].append(d[f'fyx_{b}'])
            if has_trgc:
                dtr[b].append(d[f'dtrgc_{b}'])
        d.close()

    out = dict(ref)
    out['subjects'] = subjects
    out['fxy'] = {b: np.stack(fxy[b]) for b in band_names}
    out['fyx'] = {b: np.stack(fyx[b]) for b in band_names}
    if has_trgc:
        out['dtrgc'] = {b: np.stack(dtr[b]) for b in band_names}
    return out


# ─────────────────────────────────────────────────────────────────────
# Task-vs-baseline statistics
#   ttest    -> right-tailed one-sample Student's t (production_pwgc_data_to_python.m, v4 figs)
#   signrank -> right-tailed Wilcoxon signed-rank   (v3 figs)
# ─────────────────────────────────────────────────────────────────────
def _right_tailed_pval(x, m, test):
    """Right-tailed one-sample p-value of samples ``x`` against scalar ``m``.

    ``ttest`` uses the parametric one-sample Student's t (scipy
    ``ttest_1samp(..., alternative='greater')``).  ``signrank`` uses the
    non-parametric Wilcoxon signed-rank on ``x - m`` (scipy
    ``wilcoxon(..., alternative='greater')`` == MATLAB
    ``signrank(x, m, 'tail','right')``).  Returns NaN when the statistic
    is undefined (e.g. all differences are zero).
    """
    if test == 'ttest':
        _t, p = stats.ttest_1samp(x, m, alternative='greater')
        return p
    if test == 'signrank':
        d = np.asarray(x, dtype=float) - m
        if not np.any(d != 0.0):
            return np.nan
        try:
            _w, p = stats.wilcoxon(d, alternative='greater')
        except ValueError:
            return np.nan
        return p
    raise ValueError(f"unknown test {test!r} (expected 'ttest' or 'signrank')")


def task_vs_baseline(subj_stack, window_ms, baseline_ms, task_start_ms,
                     alpha=0.05, test='ttest'):
    """Per-pair subject mean/SEM and right-tailed task-vs-baseline test.

    subj_stack : (n_subj, n_pairs, n_win)
    baseline_ms : (lo, hi) window-start range treated as baseline.
    task_start_ms : task points are windows with start >= this.
    test : 'ttest' (parametric Student's t) or 'signrank' (non-parametric
        Wilcoxon signed-rank).  Both are right-tailed, one-sample, tested
        against the scalar subject-averaged baseline mean.

    Returns dict of (n_pairs, n_win) arrays: ``mean``, ``sem``,
    ``pval`` (NaN outside task), ``sig`` (bool), and scalar-per-pair
    ``baseline_mean`` (n_pairs,).
    """
    n_subj, n_pairs, n_win = subj_stack.shape
    subj_mean = subj_stack.mean(axis=0)
    sem = subj_stack.std(axis=0, ddof=1) / np.sqrt(n_subj)

    base_mask = (window_ms >= baseline_ms[0]) & (window_ms <= baseline_ms[1])
    if not base_mask.any():
        base_mask = np.zeros(n_win, bool); base_mask[0] = True
    baseline_mean = subj_mean[:, base_mask].mean(axis=1)      # (n_pairs,)

    task_mask = window_ms >= task_start_ms
    pval = np.full((n_pairs, n_win), np.nan)
    sig = np.zeros((n_pairs, n_win), bool)
    for pi in range(n_pairs):
        for w in np.where(task_mask)[0]:
            p = _right_tailed_pval(subj_stack[:, pi, w], baseline_mean[pi], test)
            pval[pi, w] = p
            sig[pi, w] = p < alpha
    return {'mean': subj_mean, 'sem': sem, 'pval': pval, 'sig': sig,
            'baseline_mean': baseline_mean}


# ─────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────
def plot_directed_edge(agg, stats_by_band, src_name, tgt_name, pair_idx,
                       direction, out_path, bands=None, fmt='png', test='ttest'):
    """Plot one directed edge (src->tgt) across bands with significance.

    direction : 'fxy' (pair i->j) or 'fyx' (pair j->i).
    test : which task-vs-baseline test produced ``sig`` (named in the title).
    """
    if bands is None:
        bands = DEFAULT_BANDS
    band_names = list(bands)
    window_ms = agg['window_ms']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.ravel()
    for ax, b in zip(axes, band_names):
        st = stats_by_band[b]
        m = st['mean'][pair_idx]
        se = st['sem'][pair_idx]
        ax.plot(window_ms, m, color='#2166ac', lw=2, label=f'{src_name}→{tgt_name}')
        ax.fill_between(window_ms, m - se, m + se, color='#2166ac', alpha=0.25)
        ax.axhline(st['baseline_mean'][pair_idx], color='0.5', ls='--', lw=1,
                   label='baseline')
        # significance ticks
        sig = st['sig'][pair_idx]
        if sig.any():
            ytop = np.nanmax(m + se)
            ax.plot(window_ms[sig], np.full(sig.sum(), ytop * 1.05), 's',
                    color='#b2182b', ms=3)
        ax.set_title(f'{b} ({bands[b][0]:g}–{bands[b][1]:g} Hz)', fontsize=12)
        ax.axvline(0, color='k', lw=0.8, alpha=0.5)
        ax.set_ylabel('GC')
    for ax in axes[2:]:
        ax.set_xlabel('window start (ms)')
    test_label = {'ttest': "Student's t",
                  'signrank': 'Wilcoxon signed-rank'}.get(test, test)
    fig.suptitle(f'Granger causality: {src_name} → {tgt_name}   '
                 f'(sig: right-tailed {test_label})', fontsize=15)
    axes[0].legend(fontsize=9, loc='upper left')
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, format=fmt, bbox_inches='tight')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────
def run_stats(gc_dir, task, out_dir, baseline_ms=None, task_start_ms=None,
              alpha=0.05, bands=None, fmt='png', test='ttest'):
    """Aggregate a GC group directory, run stats, write figures + CSV.

    ``test`` selects the task-vs-baseline test ('ttest' or 'signrank').
    Figure and CSV names are tagged with it, so both tests can be written
    into the same ``out_dir`` and diffed edge-by-edge.
    """
    if bands is None:
        bands = DEFAULT_BANDS
    band_names = list(bands)
    if baseline_ms is None:
        bl = BASELINE_WINDOWS.get(task, (-1.6, -1.5))
        baseline_ms = (bl[0] * 1000.0, bl[1] * 1000.0)
    if task_start_ms is None:
        task_start_ms = DECODE_TMIN.get(task, -1.5) * 1000.0

    agg = load_gc_group(gc_dir, bands)
    roi = agg['roi_names']
    n_pairs = len(agg['pair_i'])
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for direction, key in [('fxy', 'fxy'), ('fyx', 'fyx')]:
        stats_by_band = {
            b: task_vs_baseline(agg[key][b], agg['window_ms'], baseline_ms,
                                task_start_ms, alpha, test)
            for b in band_names
        }
        for pi in range(n_pairs):
            i, j = int(agg['pair_i'][pi]), int(agg['pair_j'][pi])
            if direction == 'fxy':
                src, tgt = roi[i], roi[j]
            else:
                src, tgt = roi[j], roi[i]
            fname = os.path.join(out_dir, f'GC_{src}_to_{tgt}_{test}.{fmt}')
            plot_directed_edge(agg, stats_by_band, src, tgt, pi, direction,
                               fname, bands, fmt, test)
            for b in band_names:
                st = stats_by_band[b]
                for w, wm in enumerate(agg['window_ms']):
                    rows.append({
                        'src': src, 'tgt': tgt, 'band': b, 'window_ms': wm,
                        'test': test,
                        'gc_mean': st['mean'][pi, w], 'gc_sem': st['sem'][pi, w],
                        'baseline_mean': st['baseline_mean'][pi],
                        'pval': st['pval'][pi, w], 'sig': st['sig'][pi, w],
                    })
    csv_path = os.path.join(out_dir, f'gc_task_vs_baseline_stats_{test}.csv')
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    n_edges = 2 * n_pairs
    print(f'  {len(agg["subjects"])} subjects, {n_edges} directed edges, '
          f'{len(band_names)} bands, test={test} -> {out_dir}')
    print(f'  figures: {n_edges} + stats CSV: {csv_path}')
    return csv_path


def _derive_gc_dir(args):
    root = GC_SENSOR_OUTPUT_ROOT if args.space == 'sensor' else GC_OUTPUT_ROOT
    leakage_tag = 'leakage_corrected' if args.leakage_correction else 'raw'
    return (root / args.task / args.method / args.atlas / args.feature_mode
            / leakage_tag / gc_tag(args.order, args.win_ms, args.target_fs,
                                   args.normalize, args.gc_mode)
            / roiset_tag(args.roi_subset) / args.stim_class)


def parse_args():
    p = argparse.ArgumentParser(description='Group GC stats + plots')
    p.add_argument('--gc-dir', default=None,
                   help='Directory of subject GC .npz (overrides the derived path)')
    p.add_argument('--out-dir', default=None, help='Where to write figures/CSV')
    p.add_argument('--task', required=True, choices=['perception', 'overtProd'])
    p.add_argument('--alpha', type=float, default=0.05)
    p.add_argument('--test', default='ttest', choices=['ttest', 'signrank'],
                   help="task-vs-baseline test: 'ttest' (right-tailed one-sample "
                        "Student's t; matches production_pwgc_data_to_python.m and "
                        "the v4 figures) or 'signrank' (right-tailed Wilcoxon "
                        "signed-rank; matches the v3 figures)")
    p.add_argument('--format', default='png', choices=['png', 'svg'])
    # For deriving --gc-dir from a run's parameters:
    p.add_argument('--space', default='source', choices=['source', 'sensor'])
    p.add_argument('--stim-class', default='prodDiff')
    p.add_argument('--method', default='dSPM')
    p.add_argument('--atlas', default='HCPMMP1')
    p.add_argument('--feature-mode', default='vertex_selectkbest')
    p.add_argument('--leakage-correction', action='store_true', default=False)
    p.add_argument('--order', type=int, default=10)
    p.add_argument('--win-ms', type=float, default=40.0)
    p.add_argument('--target-fs', type=float, default=500.0)
    p.add_argument('--normalize', default='none')
    p.add_argument('--gc-mode', default='pairwise',
                   choices=['pairwise', 'conditional'])
    p.add_argument('--roi-subset', nargs='+', default=None, metavar='ROI',
                   help='Same subset passed to run_granger.py, so the derived '
                        'path points at that subset run (omit for the full run)')
    return p.parse_args()


def main():
    args = parse_args()
    gc_dir = args.gc_dir if args.gc_dir else str(_derive_gc_dir(args))
    out_dir = args.out_dir if args.out_dir else os.path.join(gc_dir, 'group_stats')
    print(f'GC group stats (test={args.test})\n  gc-dir: {gc_dir}')
    run_stats(gc_dir, args.task, out_dir, alpha=args.alpha, fmt=args.format,
              test=args.test)


if __name__ == '__main__':
    main()
