#!/usr/bin/env python3
"""
Cross-ROI hyperparameter-tuning summary for explore_decoding.py output.

Goal: pick **one** defensible hyperparameter set per (classifier, sw_dur)
to apply uniformly across all ROIs in the source-space analysis, rather
than re-tuning per ROI/subject/fold (which is hard to defend without an a
priori reason to expect different optima per region).

For each (classifier, sw_dur) with ``tuned=True`` rows, this script:

    1. Loads every requested ROI's ``explore_full.csv``.
    2. Computes the group-mean accuracy curve and locates that ROI's peak
       window.
    3. At each ROI's peak window, records each subject's modal ``best_C``
       (already a within-subject mode over 25 outer folds, written by
       ``decoding.py``).
    4. Pools those modes across (ROI x subject) to produce a count matrix.

Two figures are written:

    * ``hyperparam_heatmap_{clf}_sw{N}ms[_{stim}].svg``
        ROI x C-value heatmap of subject counts.  Rows are ROIs; the last
        row pools across all ROIs.  Cells annotate count and percentage of
        subjects.  Title flags the pooled modal C.

    * ``hyperparam_summary_pooled[_{stim}].svg``
        Compact overview comparing every (classifier, sw_dur) on one
        canvas: stacked horizontal bars showing the pooled C-distribution
        per config, with the modal C labeled at the right.  Use this to
        pick a single C per config at a glance.

Both figures are written under
``{...}/explore/{task}/{method}/{atlas}/{feat_mode}/{stim_class}/{run_seg}/_hyperparam_summary/``.

Usage:
    python explore_hyperparams_summary.py --task overtProd \
        --stim-class prodDiff --method dSPM --atlas HCPMMP1 \
        --rois Temporal vSMC IFG DLPFC \
        --classifiers svm logistic --sw-durs 40 60 80

    # Or auto-discover all ROIs that have an explore_full.csv on disk
    python explore_hyperparams_summary.py --task overtProd \
        --stim-class prodDiff --method dSPM --atlas HCPMMP1 \
        --rois auto --classifiers svm logistic --sw-durs 40 60 80
"""
import argparse
import os
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DECODE_OUTPUT_ROOT, PSEUDO_TRIAL_SIZE,
    explore_run_segment,
)


POOLED_LABEL = 'ALL ROIs (pooled)'


# ──────────────────────────────────────────────────────────────
# Loaders
# ──────────────────────────────────────────────────────────────
def _stim_dir(stim_class, args, run_seg):
    return (
        DECODE_OUTPUT_ROOT / 'explore' / args.task / args.method
        / args.atlas / args.feature_mode / stim_class / run_seg
    )


def _suggest_alternatives(args, stim_class):
    """List existing (feature_mode, run_seg) combos that have ROI data.

    When the constructed stim_dir is missing, the silent culprit is
    almost always ``--feature-mode``, ``--leakage-correction``,
    ``--pseudo-trial-size``, or ``--c`` not matching what was
    written by ``explore_decoding.py``.  Walk the
    ``task/method/atlas`` parent and print every path under it that
    actually contains ROI subdirs with ``explore_full.csv`` so the
    user sees which flag combos are live.
    """
    parent = (
        DECODE_OUTPUT_ROOT / 'explore' / args.task
        / args.method / args.atlas
    )
    if not parent.exists():
        return f'  (parent {parent} does not exist either)'
    candidates = []
    for feat_dir in sorted(p for p in parent.iterdir() if p.is_dir()):
        stim_subdir = feat_dir / stim_class
        if not stim_subdir.is_dir():
            continue
        for run_dir in sorted(p for p in stim_subdir.iterdir() if p.is_dir()):
            roi_dirs = [
                p for p in run_dir.iterdir()
                if p.is_dir() and (p / 'explore_full.csv').exists()
                and not p.name.startswith('_')
            ]
            if roi_dirs:
                candidates.append(
                    f'  --feature-mode {feat_dir.name}  '
                    f'(run_seg={run_dir.name}, '
                    f'{len(roi_dirs)} ROI(s): '
                    f'{", ".join(sorted(p.name for p in roi_dirs))})'
                )
    if not candidates:
        return f'  (no ROI data under {parent} for stim_class={stim_class})'
    return '\n'.join(candidates)


def _discover_rois(stim_dir):
    """Return sorted list of ROI subdirs that have explore_full.csv."""
    if not stim_dir.exists():
        return []
    return sorted(
        p.name for p in stim_dir.iterdir()
        if p.is_dir() and (p / 'explore_full.csv').exists()
        and not p.name.startswith('_')
    )


def _peak_ms(grp_tuned):
    """Group-mean peak ms for one (classifier, sw_dur) cell.

    Mirrors compute_stats() in explore_viz_stats.py: build a per-subject
    accuracy matrix, average across subjects, take argmax.  Returns NaN
    if the group has fewer than 1 valid subject.
    """
    ms_values = np.array(sorted(grp_tuned['ms'].unique()))
    if len(ms_values) == 0:
        return np.nan
    rows = []
    for subj, s_df in grp_tuned.groupby('subject'):
        s_df = s_df.sort_values('ms')
        if len(s_df) != len(ms_values):
            continue
        rows.append(s_df['accuracy'].values)
    if not rows:
        return np.nan
    mean_acc = np.vstack(rows).mean(axis=0)
    return float(ms_values[np.nanargmax(mean_acc)])


def collect_modal_choices(rois, stim_class, args, run_seg):
    """For each (clf, sw_dur, ROI), gather subject-level modal best_C.

    Returns
    -------
    records : list of dict
        Each dict: {classifier, sw_dur, roi, subject, best_C, peak_ms}.
    missing : list of str
        ROI names whose explore_full.csv could not be loaded.
    """
    stim_dir = _stim_dir(stim_class, args, run_seg)
    records = []
    missing = []
    for roi in rois:
        csv = stim_dir / roi / 'explore_full.csv'
        if not csv.exists():
            missing.append(roi)
            continue
        df = pd.read_csv(csv)
        df = df[df['tuned'] == True]  # noqa: E712
        if df.empty:
            missing.append(roi)
            continue
        if args.classifiers:
            df = df[df['classifier'].isin(args.classifiers)]
        if args.sw_durs:
            df = df[df['sw_dur'].isin(args.sw_durs)]
        if df.empty:
            continue

        for (clf, sw), grp in df.groupby(['classifier', 'sw_dur']):
            peak_ms = _peak_ms(grp)
            if np.isnan(peak_ms):
                continue
            peak_rows = grp[np.isclose(grp['ms'], peak_ms)]
            for _, row in peak_rows.iterrows():
                bc = row.get('best_C', np.nan)
                if pd.isna(bc):
                    continue
                records.append({
                    'classifier': clf,
                    'sw_dur': int(sw),
                    'roi': roi,
                    'subject': row['subject'],
                    'best_C': float(bc),
                    'peak_ms': float(peak_ms),
                })
    return records, missing


# ──────────────────────────────────────────────────────────────
# Plot: per-config heatmap (ROI x C)
# ──────────────────────────────────────────────────────────────
def _format_c(c):
    """Display C value compactly: 0.01, 0.1, 1, 10."""
    if c == int(c):
        return f'{int(c)}'
    return f'{c:g}'


def plot_roi_x_c_heatmap(records_df, classifier, sw_dur, stim_class,
                         n_subjects_expected):
    """ROI x C heatmap of subject counts, with a pooled bottom row."""
    sub = records_df[
        (records_df['classifier'] == classifier)
        & (records_df['sw_dur'] == sw_dur)
    ]
    if sub.empty:
        return None, None

    rois = sorted(sub['roi'].unique())
    c_values = sorted(sub['best_C'].unique())

    counts = np.zeros((len(rois) + 1, len(c_values)), dtype=int)
    for i, roi in enumerate(rois):
        roi_sub = sub[sub['roi'] == roi]
        for j, c in enumerate(c_values):
            counts[i, j] = int((roi_sub['best_C'] == c).sum())
    counts[-1, :] = counts[:-1, :].sum(axis=0)

    row_labels = rois + [POOLED_LABEL]
    col_labels = [f'C={_format_c(c)}' for c in c_values]

    # Annotation: "N\n(PP%)" using each row's own total as the denominator,
    # so percentages always sum to 100% within a row.
    annot = np.empty_like(counts, dtype=object)
    for i in range(counts.shape[0]):
        row_total = counts[i].sum()
        for j in range(counts.shape[1]):
            n = counts[i, j]
            pct = (100.0 * n / row_total) if row_total > 0 else 0.0
            annot[i, j] = f'{n}\n({pct:.0f}%)' if n > 0 else ''

    pooled_modal_idx = int(np.argmax(counts[-1]))
    pooled_modal_c = c_values[pooled_modal_idx]
    pooled_count = counts[-1, pooled_modal_idx]
    pooled_total = counts[-1].sum()
    pooled_pct = (100.0 * pooled_count / pooled_total) if pooled_total else 0.0

    fig, ax = plt.subplots(
        figsize=(max(7, 1.2 * len(c_values) + 3),
                 max(4, 0.4 * len(row_labels) + 2)),
    )
    sns.heatmap(
        counts, annot=annot, fmt='', cmap='Blues',
        xticklabels=col_labels, yticklabels=row_labels,
        cbar_kws={'label': 'Subjects selecting this C'},
        linewidths=0.5, linecolor='lightgray', ax=ax,
    )
    # Visually separate the pooled row
    ax.axhline(len(rois), color='black', linewidth=2)

    title = (
        f'Modal best_C across ROIs — {classifier}, sw_dur={sw_dur}ms '
        f'[{stim_class}]\n'
        f'Pooled mode: C={_format_c(pooled_modal_c)} '
        f'({pooled_count}/{pooled_total} ROI x subject picks, '
        f'{pooled_pct:.0f}%)'
    )
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Candidate C value', fontsize=11)
    ax.set_ylabel('ROI', fontsize=11)
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    plt.tight_layout()

    return fig, {
        'classifier': classifier,
        'sw_dur': sw_dur,
        'stim_class': stim_class,
        'n_rois': len(rois),
        'pooled_modal_c': pooled_modal_c,
        'pooled_modal_count': pooled_count,
        'pooled_total': pooled_total,
        'pooled_modal_pct': pooled_pct,
        'c_values': c_values,
        'pooled_counts': counts[-1].tolist(),
    }


# ──────────────────────────────────────────────────────────────
# Plot: cross-config pooled summary (stacked bars)
# ──────────────────────────────────────────────────────────────
def plot_pooled_summary(records_df, stim_class):
    """One stacked horizontal bar per (classifier, sw_dur), pooled across
    ROIs and subjects.  Each bar shows the proportion of (ROI x subject)
    picks that landed on each C value; the modal C is annotated.
    """
    sub = records_df.copy()
    if sub.empty:
        return None

    configs = (
        sub[['classifier', 'sw_dur']]
        .drop_duplicates()
        .sort_values(['classifier', 'sw_dur'])
        .itertuples(index=False, name=None)
    )
    configs = list(configs)
    c_values = sorted(sub['best_C'].unique())

    palette = sns.color_palette('viridis', n_colors=len(c_values))
    color_map = {c: palette[i] for i, c in enumerate(c_values)}

    fig, ax = plt.subplots(
        figsize=(max(10, len(c_values) * 0.5 + 8),
                 max(3, 0.45 * len(configs) + 1.5)),
    )

    labels = []
    for row_idx, (clf, sw) in enumerate(configs):
        cell = sub[(sub['classifier'] == clf) & (sub['sw_dur'] == sw)]
        total = len(cell)
        if total == 0:
            labels.append(f'{clf} sw={sw}ms (no data)')
            continue
        counts = cell['best_C'].value_counts().reindex(c_values, fill_value=0)
        proportions = counts / total
        modal_c = c_values[int(np.argmax(counts.values))]
        modal_n = int(counts.max())
        modal_pct = 100.0 * modal_n / total

        left = 0.0
        for c in c_values:
            p = float(proportions[c])
            if p == 0:
                continue
            ax.barh(row_idx, p, left=left, color=color_map[c],
                    edgecolor='white', linewidth=0.5)
            if p >= 0.05:
                ax.text(left + p / 2, row_idx, f'{p*100:.0f}%',
                        va='center', ha='center', fontsize=9,
                        color='white' if p > 0.18 else 'black')
            left += p

        labels.append(
            f'{clf} sw={sw}ms  →  C={_format_c(modal_c)} '
            f'({modal_n}/{total}, {modal_pct:.0f}%)'
        )

    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel('Proportion of (ROI x subject) picks', fontsize=11)
    ax.set_title(
        f'Pooled C-distribution per config [{stim_class}]\n'
        f'Modal C annotated on each bar — pick this for a uniform '
        f'hyperparameter across ROIs',
        fontsize=12,
    )

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=color_map[c],
                      label=f'C={_format_c(c)}')
        for c in c_values
    ]
    ax.legend(handles=legend_handles, loc='center left',
              bbox_to_anchor=(1.01, 0.5), fontsize=10, frameon=False)
    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Cross-ROI hyperparameter-tuning summary; helps pick a single '
            'defensible C per (classifier, sw_dur) for the source-space '
            'analysis.'
        ),
    )
    parser.add_argument('--task', required=True,
                        choices=['perception', 'overtProd'])
    parser.add_argument('--stim-class', required=True,
                        choices=['prodDiff', 'percDiff'])
    parser.add_argument('--method', required=True,
                        choices=['dSPM', 'LCMV'])
    parser.add_argument('--atlas', default='aparc',
                        choices=['aparc', 'HCPMMP1', 'Schaefer200', 'custom'])
    parser.add_argument('--feature-mode', default='pca_flip',
                        choices=['pca_flip', 'vertex_pca',
                                 'vertex_selectkbest',
                                 'vertex_selectkbest_all'])
    parser.add_argument('--rois', nargs='+', required=True,
                        help='ROI names (must match explore_decoding output '
                             'subdirs); use "auto" to load every ROI with an '
                             'explore_full.csv on disk.')
    parser.add_argument('--classifiers', nargs='+', default=None,
                        choices=['svm', 'lda', 'logistic'],
                        help='Restrict to these classifiers (default: all '
                             'tunable present)')
    parser.add_argument('--sw-durs', nargs='+', type=int, default=None,
                        help='Restrict to these sliding-window durations')
    parser.add_argument('--out-suffix', default='',
                        help='Suffix appended to every output filename')
    # Run-time params that affect the output path; must match the values
    # used when explore_decoding.py was run.
    parser.add_argument('--leakage-correction', action='store_true',
                        default=False)
    parser.add_argument('--pseudo-trial-size', type=int,
                        default=PSEUDO_TRIAL_SIZE)
    parser.add_argument('--c', type=float, default=None,
                        help='Match the --c passed to explore_decoding '
                             '(part of the output path).  Omit if you ran '
                             'explore_decoding without --c (default Cdef).')
    return parser.parse_args()


def main():
    args = parse_args()
    run_seg = explore_run_segment(
        args.leakage_correction, args.pseudo_trial_size, args.c,
    )
    stim_dir = _stim_dir(args.stim_class, args, run_seg)
    if not stim_dir.exists():
        raise SystemExit(
            f'Stim directory not found:\n  {stim_dir}\n\n'
            f'You passed: --feature-mode {args.feature_mode}, '
            f'--leakage-correction={args.leakage_correction}, '
            f'--pseudo-trial-size {args.pseudo_trial_size}, '
            f'--c {args.c}\n'
            f'(these flags must match what explore_decoding.py was run '
            f'with — they are part of the output path).\n\n'
            f'Existing combos with ROI data under '
            f'{args.task}/{args.method}/{args.atlas} '
            f'for stim_class={args.stim_class}:\n'
            f'{_suggest_alternatives(args, args.stim_class)}'
        )

    if len(args.rois) == 1 and args.rois[0].lower() == 'auto':
        rois = _discover_rois(stim_dir)
        if not rois:
            raise SystemExit(
                f'No ROI subdirs with explore_full.csv under {stim_dir}'
            )
        print(f'Auto-discovered {len(rois)} ROIs: {rois}')
    else:
        rois = list(args.rois)

    records, missing = collect_modal_choices(
        rois, args.stim_class, args, run_seg,
    )
    if missing:
        print(f'Skipped {len(missing)} ROI(s) with no tuned data: {missing}')
    if not records:
        raise SystemExit(
            f'No tuned rows found across the requested ROIs under '
            f'{stim_dir}.\n'
            f'Did you run explore_decoding.py with --tune-hyperparams '
            f'and matching --feature-mode / --leakage-correction / '
            f'--pseudo-trial-size / --c?\n\n'
            f'Existing combos with ROI data under '
            f'{args.task}/{args.method}/{args.atlas} '
            f'for stim_class={args.stim_class}:\n'
            f'{_suggest_alternatives(args, args.stim_class)}'
        )

    rec_df = pd.DataFrame(records)
    print(f'\nLoaded {len(rec_df)} (ROI x subject) modal-C entries '
          f'across {rec_df["roi"].nunique()} ROIs, '
          f'{rec_df["subject"].nunique()} subjects, '
          f'{rec_df["classifier"].nunique()} classifiers, '
          f'{rec_df["sw_dur"].nunique()} sw_durs.')

    # ── Print a compact summary table ──────────────────────────────
    print(f'\nModal C selected by pooling across ROIs and subjects '
          f'[{args.stim_class}]:')
    print(f'{"Classifier":<12} {"SW_dur":<8} {"Pooled mode":<14} '
          f'{"Picks":<14} {"Distribution":<40}')
    print('-' * 90)
    summary_rows = []
    for (clf, sw), grp in rec_df.groupby(['classifier', 'sw_dur']):
        c_counts = Counter(grp['best_C'])
        total = sum(c_counts.values())
        modal_c, modal_n = c_counts.most_common(1)[0]
        dist = ', '.join(
            f'C={_format_c(c)}:{n}' for c, n in
            sorted(c_counts.items(), key=lambda kv: -kv[1])
        )
        print(f'{clf:<12} {sw:<8} C={_format_c(modal_c):<12} '
              f'{modal_n}/{total} ({100*modal_n/total:.0f}%)   {dist}')
        summary_rows.append({
            'classifier': clf,
            'sw_dur': int(sw),
            'pooled_modal_C': modal_c,
            'modal_count': modal_n,
            'total_picks': total,
            'modal_pct': 100 * modal_n / total,
            'distribution': dist,
        })

    # ── Output dir ─────────────────────────────────────────────────
    out_dir = stim_dir / '_hyperparam_summary'
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = f'_{args.out_suffix}' if args.out_suffix else ''
    pd.DataFrame(summary_rows).to_csv(
        out_dir / f'hyperparam_summary{suffix}.csv', index=False,
    )
    rec_df.to_csv(
        out_dir / f'hyperparam_records{suffix}.csv', index=False,
    )

    # ── Per-config heatmaps ────────────────────────────────────────
    n_subjects_expected = rec_df['subject'].nunique()
    for (clf, sw) in sorted(
        rec_df[['classifier', 'sw_dur']]
        .drop_duplicates().itertuples(index=False, name=None)
    ):
        fig, _ = plot_roi_x_c_heatmap(
            rec_df, clf, sw, args.stim_class, n_subjects_expected,
        )
        if fig is None:
            continue
        fname = (
            out_dir
            / f'hyperparam_heatmap_{clf}_sw{sw}ms_{args.stim_class}'
              f'{suffix}.svg'
        )
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f'  Saved: {fname}')

    # ── Cross-config pooled summary ────────────────────────────────
    fig = plot_pooled_summary(rec_df, args.stim_class)
    if fig is not None:
        fname = (
            out_dir
            / f'hyperparam_summary_pooled_{args.stim_class}{suffix}.svg'
        )
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved: {fname}')

    print(f'\nDone. Outputs under: {out_dir}')


if __name__ == '__main__':
    main()
