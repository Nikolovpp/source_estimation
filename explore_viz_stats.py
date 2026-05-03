#!/usr/bin/env python3
"""
Visualization and group-level statistics for explore_decoding.py output.

Reads ``explore_full.csv`` written by ``explore_decoding.py``, runs a
standard cluster-based permutation test against chance (one-tailed) per
configuration, and produces comparison figures.  Decoding is never
re-run — this script operates purely on cached accuracy values so it is
cheap to re-execute whenever you want to re-plot or change the stats.

Figures written into ``<output_dir>/figures/``:

    * ``clf_comparison_sw{sw}ms_{classifiers}[_{primary}-vs-{compare}][_{suffix}].svg``
        Classifier overlay at a single sliding-window duration, with
        significant-cluster time windows shaded per classifier.  When
        ``--compare-stim-class`` is supplied the second stim class is
        overlaid as dashed lines (color still encodes the classifier).
    * ``sw_sweep_{classifier}[_tuned][_{suffix}].svg``
        Sliding-window-duration sweep for one classifier, with
        significance shading per sw_dur.  Always uses the primary
        ``--stim-class`` only.
    * ``peak_accuracy_heatmap[_{suffix}].svg``
        (classifier x sw_dur) heatmap of the mean *per-subject* peak
        accuracy — i.e. ``mean_subj(max_t acc)``.  This is biased upward
        relative to the line-plot's ``max_t(mean_subj acc)`` peak because
        each subject's peak can occur at a different latency.  Always
        uses the primary ``--stim-class`` only.

Use ``--out-suffix`` to append a custom tag to every figure (and to the
stats CSV) when you want to compare two filter selections side by side
without overwriting.

Usage:
    # Plot everything currently in explore_full.csv
    python explore_viz_stats.py --task overtProd --stim-class prodDiff \
        --method dSPM --atlas HCPMMP1 --roi Temporal

    # Restrict to a subset of classifiers / window durations
    python explore_viz_stats.py --task overtProd --stim-class prodDiff \
        --method dSPM --atlas HCPMMP1 --roi Temporal \
        --classifiers svm lda --sw-durs 40 60 --out-suffix svm-vs-lda
"""
import argparse
import os
import sys
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from mne.stats import permutation_cluster_1samp_test

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DECODE_OUTPUT_ROOT, SW_STEP_SIZE,
    PSEUDO_TRIAL_SIZE,
    explore_run_segment,
)


N_PERMUTATIONS = 1024   # matches source_stats_viz.py standard cluster test
ALPHA = 0.05

# Hand-picked high-contrast palette (ColorBrewer Set1 sans low-luminance
# yellow, plus a couple of complements). Keep ordering stable so the same
# series gets the same color across re-runs.
DISTINCT_COLORS = [
    '#e41a1c',  # red
    '#377eb8',  # blue
    '#4daf4a',  # green
    '#984ea3',  # purple
    '#ff7f00',  # orange
    '#a65628',  # brown
    '#f781bf',  # pink
    '#17becf',  # cyan
    '#bcbd22',  # olive
    '#7f7f7f',  # gray
]


def _assign_colors(keys):
    """Map an ordered list of keys to maximally distinct colors."""
    if len(keys) <= len(DISTINCT_COLORS):
        return {k: DISTINCT_COLORS[i] for i, k in enumerate(keys)}
    palette = sns.color_palette('husl', n_colors=len(keys))
    return {k: palette[i] for i, k in enumerate(keys)}


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def find_contiguous_clusters(mask):
    """Return (start, end) index pairs for contiguous True runs."""
    clusters = []
    in_cluster = False
    start = None
    for i, val in enumerate(mask):
        if val and not in_cluster:
            start = i
            in_cluster = True
        elif not val and in_cluster:
            clusters.append((start, i - 1))
            in_cluster = False
    if in_cluster:
        clusters.append((start, len(mask) - 1))
    return clusters


def _cluster_test(acc_matrix):
    """Run one-tailed cluster-based permutation test on (n_subj, n_times).

    MNE 1.10 ignores ``out_type='mask'`` and returns each cluster as a
    ``(slice(...),)`` tuple, so we index into the output arrays with the
    raw cluster object (same pattern as
    ``source_stats_viz.compute_stats``) rather than ``np.where`` — the
    latter on a tuple-of-slice silently collapses to ``array([0])`` and
    reports every cluster as a single point at time 0.

    Returns a per-time significance mask, per-time cluster p-values, and
    a list of ``(bool_mask, pv)`` pairs (normalized to boolean masks so
    callers don't have to re-parse slice tuples).
    """
    n_subj, n_times = acc_matrix.shape
    X = acc_matrix - 0.5

    T_obs_c, clusters_c, pv_c, _ = permutation_cluster_1samp_test(
        X, threshold=None, n_permutations=N_PERMUTATIONS,
        tail=1, out_type='mask', verbose=False,
    )

    cluster_mask = np.zeros(n_times, dtype=bool)
    cluster_pvals_arr = np.ones(n_times)
    bool_masks = []
    for ic, cpv in enumerate(pv_c):
        cmask_bool = np.zeros(n_times, dtype=bool)
        cmask_bool[clusters_c[ic]] = True
        cluster_pvals_arr[cmask_bool] = cpv
        if cpv < ALPHA:
            cluster_mask[cmask_bool] = True
        bool_masks.append(cmask_bool)

    return cluster_mask, cluster_pvals_arr, list(zip(bool_masks, pv_c))


def _build_accuracy_matrix(df_subset, ms_values):
    """Stack per-subject accuracy curves into (n_subj, n_times) matrix.

    Drops subjects whose coverage does not match the full ms vector.
    """
    subjects = sorted(df_subset['subject'].unique())
    n_times = len(ms_values)
    rows = []
    kept = []
    for subj in subjects:
        s_df = df_subset[df_subset['subject'] == subj].sort_values('ms')
        if len(s_df) != n_times:
            continue
        rows.append(s_df['accuracy'].values)
        kept.append(subj)
    if not rows:
        return np.zeros((0, n_times)), kept
    return np.vstack(rows), kept


# ──────────────────────────────────────────────────────────────
# Stats computation
# ──────────────────────────────────────────────────────────────
def _summarize_tuned_params_at_peak(grp, peak_ms):
    """Aggregate modal hyperparameters at the group-level peak window.

    Each subject's row at the peak window already carries its within-subject
    modal value (over 25 outer folds, from decoding).  Here we take the
    mode across subjects and report "X selected by K/N subjects" for each
    hyperparameter column present in the group.

    Returns a dict mapping parameter name to a human-readable summary,
    and a list of (param, mode_value, count, n_subjects) tuples.
    """
    peak_rows = grp[np.isclose(grp['ms'], peak_ms)]
    best_cols = [
        c for c in peak_rows.columns
        if c.startswith('best_') and not c.endswith('_freq')
    ]

    summary_parts = []
    records = []
    for col in best_cols:
        vals = peak_rows[col].dropna().tolist()
        if not vals:
            continue
        pname = col.replace('best_', '')
        mode_val, mode_count = Counter(vals).most_common(1)[0]
        summary_parts.append(
            f'{pname}={mode_val} ({mode_count}/{len(vals)} subj)'
        )
        records.append((pname, mode_val, mode_count, len(vals)))
    return '; '.join(summary_parts), records


def compute_stats(df):
    """Per-config cluster-permutation test and per-time-window stats.

    Returns
    -------
    stats_summary : DataFrame
        One row per configuration with cluster counts, earliest onset,
        peak accuracy/latency, and modal tuned hyperparameters at peak.
    sig_lookup : dict
        ``{(classifier, sw_dur, tuned): (ms_values, sig_mask, p_per_time)}``
        — used by plots to shade significant windows.
    """
    summary_rows = []
    sig_lookup = {}

    grouped = df.groupby(['classifier', 'sw_dur', 'tuned', 'stim_class'])
    print(f'Running cluster-based permutation tests '
          f'(n_permutations={N_PERMUTATIONS}) on {len(grouped)} configs...')

    for (clf, sw_dur, tuned, stim), grp in grouped:
        ms_values = np.array(sorted(grp['ms'].unique()))
        acc_matrix, kept = _build_accuracy_matrix(grp, ms_values)
        n_valid = acc_matrix.shape[0]

        tuned_str = 'Yes' if tuned else 'No'

        if n_valid < 3:
            # Too few subjects for a meaningful permutation test
            sig_mask = np.zeros(len(ms_values), dtype=bool)
            p_per_time = np.ones(len(ms_values))
            n_sig = 0
            earliest_onset = None
            mean_acc = (acc_matrix.mean(axis=0) if n_valid > 0
                        else np.full(len(ms_values), np.nan))
            peak_acc = float(np.nanmax(mean_acc)) if n_valid > 0 else np.nan
            peak_ms = (float(ms_values[np.nanargmax(mean_acc)])
                       if n_valid > 0 else np.nan)
        else:
            sig_mask, p_per_time, clusters_pv = _cluster_test(acc_matrix)
            n_sig = sum(1 for _, pv in clusters_pv if pv < ALPHA)
            earliest_onset = None
            for cmask, pv in clusters_pv:
                if pv < ALPHA:
                    onset_ms = ms_values[np.where(cmask)[0][0]]
                    if earliest_onset is None or onset_ms < earliest_onset:
                        earliest_onset = float(onset_ms)

            mean_acc = acc_matrix.mean(axis=0)
            peak_acc = float(mean_acc.max())
            peak_ms = float(ms_values[np.argmax(mean_acc)])

        sig_lookup[(clf, int(sw_dur), bool(tuned), stim)] = (
            ms_values, sig_mask, p_per_time,
        )

        # Tuned configs carry best_* columns; report the modal value at
        # the group-mean peak window for publication-ready reporting.
        if tuned and not np.isnan(peak_ms):
            best_summary, _ = _summarize_tuned_params_at_peak(grp, peak_ms)
        else:
            best_summary = ''

        summary_rows.append({
            'classifier': clf,
            'sw_dur': int(sw_dur),
            'tuned': tuned_str,
            'stim_class': stim,
            'n_subjects': int(n_valid),
            'n_sig_clusters': int(n_sig),
            'earliest_onset_ms': earliest_onset,
            'peak_acc': peak_acc,
            'peak_ms': peak_ms,
            'modal_hyperparams_at_peak': best_summary,
        })

    return pd.DataFrame(summary_rows), sig_lookup


# ──────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────
def plot_classifier_comparison(df, sw_dur, roi_name, sig_lookup,
                               primary_stim, compare_stim=None):
    """Overlay classifier accuracy curves for one sw_dur, shade sig clusters.

    When ``n_subjects > 1`` SEM bands are shown; significance shading
    uses the cluster test's per-classifier mask.  When ``compare_stim``
    is provided, both stim classes are drawn on the same axes — primary
    as solid lines, compared as dashed — with classifier color preserved
    so contrasts read cleanly.
    """
    sub = df[df['sw_dur'] == sw_dur]
    n_subjects = sub['subject'].nunique()

    # One distinct color per (classifier, tuned) pair present.
    series_keys = sorted(
        {(clf, bool(t)) for clf, t in
         sub[['classifier', 'tuned']].drop_duplicates().itertuples(index=False)},
        key=lambda x: (x[0], x[1]),
    )
    color_map = _assign_colors(series_keys)

    linestyle_map = {primary_stim: '-'}
    if compare_stim is not None:
        linestyle_map[compare_stim] = '--'

    fig, ax = plt.subplots(figsize=(12, 7))

    for (clf, tuned, stim), grp in sub.groupby(
        ['classifier', 'tuned', 'stim_class']
    ):
        ms_values = np.array(sorted(grp['ms'].unique()))
        acc_matrix, _ = _build_accuracy_matrix(grp, ms_values)
        if acc_matrix.shape[0] == 0:
            continue

        mean_acc = acc_matrix.mean(axis=0)
        sem = (acc_matrix.std(axis=0, ddof=1) /
               np.sqrt(acc_matrix.shape[0])
               if acc_matrix.shape[0] > 1 else np.zeros_like(mean_acc))

        color = color_map[(clf, bool(tuned))]
        linestyle = linestyle_map.get(stim, '-')
        stim_suffix = f' ({stim})' if compare_stim is not None else ''
        label = clf + (' [tuned]' if tuned else '') + stim_suffix

        ax.plot(ms_values, mean_acc, color=color,
                linewidth=2, linestyle=linestyle, label=label)
        if n_subjects > 1:
            ax.fill_between(ms_values, mean_acc - sem, mean_acc + sem,
                            alpha=0.18, color=color)

        # Significance shading (from cluster test on this config)
        sig = sig_lookup.get((clf, int(sw_dur), bool(tuned), stim))
        if sig is not None:
            _, sig_mask, _ = sig
            shade_alpha = 0.15 if compare_stim is not None else 0.22
            for s, e in find_contiguous_clusters(sig_mask):
                ax.axvspan(ms_values[s], ms_values[e],
                           alpha=shade_alpha, color=color, zorder=0)

    ax.axhline(0.5, color='black', linestyle='--', linewidth=0.8, label='chance')
    ax.axvline(0, color='black', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Time (ms)', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    title_stim = (f'[{primary_stim} vs {compare_stim}]'
                  if compare_stim is not None else f'[{primary_stim}]')
    ax.set_title(f'{roi_name} {title_stim} — Classifier Comparison '
                 f'(sw_dur={sw_dur}ms, n={n_subjects})', fontsize=16)
    ax.legend(loc='upper left', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)
    plt.tight_layout()
    return fig


def plot_sw_dur_comparison(df, classifier, tuned, roi_name, sw_durs,
                           sig_lookup, stim_class):
    """Overlay accuracy curves for different sw_durs (one classifier)."""
    sub = df[(df['classifier'] == classifier) & (df['tuned'] == tuned)]
    if sub.empty:
        return None

    n_subjects = sub['subject'].nunique()
    colors = _assign_colors(list(sw_durs))

    fig, ax = plt.subplots(figsize=(12, 7))

    for sw_dur in sw_durs:
        grp = sub[sub['sw_dur'] == sw_dur]
        if grp.empty:
            continue
        ms_values = np.array(sorted(grp['ms'].unique()))
        acc_matrix, _ = _build_accuracy_matrix(grp, ms_values)
        if acc_matrix.shape[0] == 0:
            continue

        mean_acc = acc_matrix.mean(axis=0)
        sem = (acc_matrix.std(axis=0, ddof=1) /
               np.sqrt(acc_matrix.shape[0])
               if acc_matrix.shape[0] > 1 else np.zeros_like(mean_acc))

        ax.plot(ms_values, mean_acc, color=colors[sw_dur],
                linewidth=2, label=f'{sw_dur}ms')
        if n_subjects > 1:
            ax.fill_between(ms_values, mean_acc - sem, mean_acc + sem,
                            alpha=0.18, color=colors[sw_dur])

        sig = sig_lookup.get(
            (classifier, int(sw_dur), bool(tuned), stim_class)
        )
        if sig is not None:
            _, sig_mask, _ = sig
            for s, e in find_contiguous_clusters(sig_mask):
                ax.axvspan(ms_values[s], ms_values[e],
                           alpha=0.22, color=colors[sw_dur], zorder=0)

    ax.axhline(0.5, color='black', linestyle='--', linewidth=0.8, label='chance')
    ax.axvline(0, color='black', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Time (ms)', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    tuned_str = ' [tuned]' if tuned else ''
    ax.set_title(f'{roi_name} [{stim_class}] — Window Duration Sweep '
                 f'({classifier}{tuned_str}, n={n_subjects})', fontsize=16)
    ax.legend(loc='upper left', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)
    plt.tight_layout()
    return fig


def plot_peak_accuracy_heatmap(df, roi_name, stim_class):
    """Heatmap of mean per-subject peak accuracy by (config x sw_dur).

    Each cell is computed as ``mean_subj(max_t(acc))`` — for every subject
    we take that subject's best time point, then average those per-subject
    peaks across subjects.  This is *not* the same as the line-plot's
    peak, which is ``max_t(mean_subj(acc))`` (peak of the group-mean
    curve).  The per-subject-peak estimator is biased upward because each
    subject's peak can occur at a different latency, so the colorbar is
    labeled accordingly.
    """
    per_subj = df.groupby(
        ['subject', 'classifier', 'sw_dur', 'tuned']
    )['accuracy'].max().reset_index()

    avg = per_subj.groupby(
        ['classifier', 'sw_dur', 'tuned']
    )['accuracy'].mean().reset_index()

    avg['config'] = avg.apply(
        lambda r: r['classifier'] + (' [tuned]' if r['tuned'] else ''),
        axis=1,
    )

    pivot = avg.pivot(index='config', columns='sw_dur', values='accuracy')
    n_subjects = df['subject'].nunique()
    classifiers_in_fig = sorted(df['classifier'].unique().tolist())

    fig, ax = plt.subplots(
        figsize=(max(8, len(pivot.columns) * 1.5),
                 max(4, len(pivot) * 0.8 + 1)),
    )
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax,
                vmin=0.48, linewidths=0.5,
                cbar_kws={'label': 'Mean per-subject peak accuracy'})
    ax.set_xlabel('Sliding Window Duration (ms)', fontsize=13)
    ax.set_ylabel('Configuration', fontsize=13)
    ax.set_title(
        f'{roi_name} [{stim_class}] — Mean per-subject peak accuracy '
        f'by Configuration  '
        f'[{", ".join(classifiers_in_fig)}] (n={n_subjects})\n'
        f'mean_subj(max_t acc) — biased above the line-plot peak '
        f'max_t(mean_subj acc)',
        fontsize=13,
    )
    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize & run stats on explore_decoding.py output',
    )
    parser.add_argument('--task', required=True,
                        choices=['perception', 'overtProd'])
    parser.add_argument('--stim-class', required=True,
                        choices=['prodDiff', 'percDiff'])
    parser.add_argument('--compare-stim-class', default=None,
                        choices=['prodDiff', 'percDiff'],
                        help='Overlay this second stim class on the '
                             'classifier-comparison plot (dashed lines). '
                             'Must differ from --stim-class. The sw_sweep '
                             'and heatmap figures still use the primary '
                             'stim class only. Output files are written '
                             'under the primary --stim-class directory.')
    parser.add_argument('--method', required=True,
                        choices=['dSPM', 'LCMV'])
    parser.add_argument('--atlas', default='aparc',
                        choices=['aparc', 'HCPMMP1', 'Schaefer200', 'custom'])
    parser.add_argument('--feature-mode', default='pca_flip',
                        choices=['pca_flip', 'vertex_pca', 'vertex_selectkbest',
                                 'vertex_selectkbest_all'])
    parser.add_argument('--roi', required=True,
                        help='ROI name (must match the directory written by '
                             'explore_decoding.py)')
    parser.add_argument('--classifiers', nargs='+', default=None,
                        choices=['svm', 'lda', 'logistic'],
                        help='Restrict plots/stats to these classifiers '
                             '(default: all present)')
    parser.add_argument('--sw-durs', nargs='+', type=int, default=None,
                        help='Restrict to these sliding-window durations')
    parser.add_argument('--combos', nargs='+', default=None,
                        metavar='CLF:SW[:tuned|:untuned]',
                        help='Plot only these specific classifier:sw_dur '
                             'combinations instead of the full cross-product. '
                             'Optional :tuned or :untuned third field selects '
                             'one variant. Examples: "svm:40 lda:60", '
                             '"svm:40:tuned logistic:80:untuned". '
                             'Overrides --classifiers/--sw-durs/--tuned.')
    parser.add_argument('--tuned', choices=['any', 'only', 'exclude'],
                        default='any',
                        help='"any"=both tuned and untuned, "only"=tuned only, '
                             '"exclude"=untuned only (default: any)')
    parser.add_argument('--out-suffix', default='',
                        help='Suffix appended to every output filename (use to '
                             'keep multiple filter selections side by side)')
    # Run-time params that affect the explore_decoding output path.
    # Must match the values used when explore_decoding.py was run, or
    # the explore_full.csv lookup will not find the data.
    parser.add_argument('--leakage-correction', action='store_true', default=False,
                        help='Match the --leakage-correction flag passed to '
                             'explore_decoding (part of the output path)')
    parser.add_argument('--pseudo-trial-size', type=int, default=PSEUDO_TRIAL_SIZE,
                        help='Match the --pseudo-trial-size passed to '
                             'explore_decoding (part of the output path)')
    parser.add_argument('--c', type=float, default=None,
                        help='Match the --c passed to explore_decoding '
                             '(part of the output path).  Omit if you ran '
                             'explore_decoding without --c (default Cdef).')
    return parser.parse_args()


def _explore_csv_path(stim_class, args, run_seg):
    return (
        DECODE_OUTPUT_ROOT / 'explore' / args.task / args.method
        / args.atlas / args.feature_mode / stim_class
        / run_seg / args.roi
    )


def main():
    args = parse_args()

    if args.compare_stim_class is not None and (
        args.compare_stim_class == args.stim_class
    ):
        raise SystemExit(
            '--compare-stim-class must differ from --stim-class'
        )

    run_seg = explore_run_segment(
        args.leakage_correction, args.pseudo_trial_size, args.c,
    )
    out_dir = _explore_csv_path(args.stim_class, args, run_seg)
    full_csv = out_dir / 'explore_full.csv'
    if not full_csv.exists():
        raise SystemExit(
            f'explore_full.csv not found at {full_csv}\n'
            f'Run explore_decoding.py first with matching arguments.'
        )

    df = pd.read_csv(full_csv)
    df['stim_class'] = args.stim_class
    print(f'Loaded {len(df)} rows from {full_csv}')

    if args.compare_stim_class is not None:
        cmp_dir = _explore_csv_path(args.compare_stim_class, args, run_seg)
        cmp_csv = cmp_dir / 'explore_full.csv'
        if not cmp_csv.exists():
            raise SystemExit(
                f'explore_full.csv for --compare-stim-class not found at '
                f'{cmp_csv}\nRun explore_decoding.py for that stim class '
                f'first with matching arguments.'
            )
        df_cmp = pd.read_csv(cmp_csv)
        df_cmp['stim_class'] = args.compare_stim_class
        print(f'Loaded {len(df_cmp)} rows from {cmp_csv}')
        df = pd.concat([df, df_cmp], ignore_index=True)

    print(f'  Subjects:    {sorted(df["subject"].unique())}')
    print(f'  Classifiers: {sorted(df["classifier"].unique())}')
    print(f'  SW durations: {sorted(df["sw_dur"].unique())}')
    print(f'  Tuned flags:  {sorted(df["tuned"].unique())}')
    print(f'  Stim classes: {sorted(df["stim_class"].unique())}')

    # ── Apply filters ───────────────────────────────────────────────
    if args.combos:
        # Specific (classifier, sw_dur[, tuned]) selections — overrides the
        # cross-product filters. Each combo: "clf:sw" or "clf:sw:tuned" /
        # "clf:sw:untuned".
        triples = []
        for combo in args.combos:
            parts = combo.split(':')
            if len(parts) not in (2, 3):
                raise SystemExit(
                    f'Invalid --combos entry "{combo}"; expected '
                    f'CLF:SW or CLF:SW:tuned/untuned'
                )
            clf = parts[0].strip()
            try:
                sw = int(parts[1])
            except ValueError:
                raise SystemExit(
                    f'Invalid sw_dur in --combos entry "{combo}"'
                )
            if len(parts) == 3:
                t_str = parts[2].strip().lower()
                if t_str not in ('tuned', 'untuned'):
                    raise SystemExit(
                        f'Invalid tuned flag in --combos entry "{combo}"; '
                        f'expected "tuned" or "untuned"'
                    )
                tuned_filter = (t_str == 'tuned')
            else:
                tuned_filter = None
            triples.append((clf, sw, tuned_filter))

        keep = pd.Series(False, index=df.index)
        for clf, sw, tuned_filter in triples:
            mask = (df['classifier'] == clf) & (df['sw_dur'] == sw)
            if tuned_filter is not None:
                mask &= (df['tuned'] == tuned_filter)
            keep |= mask
        df = df[keep]
    else:
        if args.classifiers:
            df = df[df['classifier'].isin(args.classifiers)]
        if args.sw_durs:
            df = df[df['sw_dur'].isin(args.sw_durs)]
        if args.tuned == 'only':
            df = df[df['tuned'] == True]   # noqa: E712
        elif args.tuned == 'exclude':
            df = df[df['tuned'] == False]  # noqa: E712

    if df.empty:
        raise SystemExit('No rows left after filtering; adjust CLI flags.')

    sw_durs_present = sorted(df['sw_dur'].unique().tolist())
    classifiers_present = sorted(df['classifier'].unique().tolist())
    n_subjects = df['subject'].nunique()
    print(f'\nAfter filtering: {n_subjects} subjects, '
          f'{len(classifiers_present)} classifiers, '
          f'{len(sw_durs_present)} sw_durs')

    suffix = f'_{args.out_suffix}' if args.out_suffix else ''
    comparing = args.compare_stim_class is not None

    # ── Stats ──────────────────────────────────────────────────────
    if n_subjects >= 3:
        stats_df, sig_lookup = compute_stats(df)
        stats_csv = out_dir / f'explore_stats{suffix}.csv'
        stats_df.to_csv(stats_csv, index=False)
        print(f'\nStats: {stats_csv}')

        stim_col_hdr = f'{"Stim":<10} ' if comparing else ''
        print(f'\n{"Classifier":<12} {"SW_dur":<8} {"Tuned":<7} '
              f'{stim_col_hdr}{"Sig Clusters":<14} {"Onset (ms)":<12} '
              f'{"Peak Acc":<10} {"Peak ms":<10}')
        print('-' * (75 + (10 if comparing else 0)))
        for _, row in stats_df.iterrows():
            onset_str = (f'{row["earliest_onset_ms"]:.1f}'
                         if row['earliest_onset_ms'] is not None
                         and not pd.isna(row['earliest_onset_ms'])
                         else '—')
            stim_col = f'{row["stim_class"]:<10} ' if comparing else ''
            print(f'{row["classifier"]:<12} {int(row["sw_dur"]):<8} '
                  f'{row["tuned"]:<7} '
                  f'{stim_col}'
                  f'{row["n_sig_clusters"]:<14} {onset_str:<12} '
                  f'{row["peak_acc"]:<10.3f} {row["peak_ms"]:<10.1f}')

        tuned_rows = stats_df[stats_df['modal_hyperparams_at_peak'] != '']
        if len(tuned_rows) > 0:
            print('\nModal tuned hyperparameters at peak window '
                  '(mode over 25 outer folds per subject; '
                  'mode over subjects at the group peak ms):')
            stim_col_hdr = f'{"Stim":<10} ' if comparing else ''
            print(f'{"Classifier":<12} {"SW_dur":<8} '
                  f'{stim_col_hdr}{"Peak ms":<10} {"Modal hyperparams":<50}')
            print('-' * (80 + (10 if comparing else 0)))
            for _, row in tuned_rows.iterrows():
                stim_col = f'{row["stim_class"]:<10} ' if comparing else ''
                print(f'{row["classifier"]:<12} {int(row["sw_dur"]):<8} '
                      f'{stim_col}'
                      f'{row["peak_ms"]:<10.1f} '
                      f'{row["modal_hyperparams_at_peak"]:<50}')
    else:
        print(f'\nSkipping cluster-based permutation stats '
              f'(need >= 3 subjects, have {n_subjects}); figures will '
              f'show no significance shading.')
        sig_lookup = {}

    # ── Figures ────────────────────────────────────────────────────
    fig_dir = out_dir / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    # sw_sweep + heatmap operate on the primary stim class only — they
    # have no clean way to encode a second stim class.
    df_primary = df[df['stim_class'] == args.stim_class]

    # Classifier comparison (one figure per sw_dur) — encode classifier set
    # in filename so partial runs (e.g. --classifiers svm lda) don't
    # overwrite figures from other classifier-set runs at the same sw_dur.
    # When comparing two stim classes, also tag the filename so the two
    # variants don't collide with single-stim runs.
    clf_tag = '-'.join(classifiers_present)
    stim_tag = (f'_{args.stim_class}-vs-{args.compare_stim_class}'
                if comparing else '')
    for sw_dur in sw_durs_present:
        fig = plot_classifier_comparison(
            df, sw_dur, args.roi, sig_lookup,
            args.stim_class, args.compare_stim_class,
        )
        fname = (
            fig_dir
            / f'clf_comparison_sw{sw_dur}ms_{clf_tag}{stim_tag}{suffix}.svg'
        )
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f'  Saved: {fname}')

    # SW duration sweep (one figure per classifier x tuned combo present)
    for clf_name in classifiers_present:
        tuned_flags = sorted(
            df_primary[df_primary['classifier'] == clf_name]['tuned']
            .unique().tolist()
        )
        for tuned in tuned_flags:
            fig = plot_sw_dur_comparison(
                df_primary, clf_name, bool(tuned), args.roi,
                sw_durs_present, sig_lookup, args.stim_class,
            )
            if fig is None:
                continue
            t_str = '_tuned' if tuned else ''
            fname = fig_dir / f'sw_sweep_{clf_name}{t_str}{suffix}.svg'
            fig.savefig(fname, dpi=150)
            plt.close(fig)
            print(f'  Saved: {fname}')

    # Peak accuracy heatmap — encode classifiers in filename so partial
    # runs (e.g. --classifiers svm) don't overwrite full-set heatmaps.
    fig = plot_peak_accuracy_heatmap(df_primary, args.roi, args.stim_class)
    fname = fig_dir / f'peak_accuracy_heatmap_{clf_tag}{suffix}.svg'
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f'  Saved: {fname}')

    print('\nDone.')


if __name__ == '__main__':
    main()
