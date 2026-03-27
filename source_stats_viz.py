#!/usr/bin/env python3
# %% [markdown]
# # Source-Space SVM Statistics & Visualization
#
# Computes group-level statistics (cluster-based permutation + TFCE) and
# visualizes SVM decoding accuracy and source-space ERPs for each cortical ROI.
#
# Mirrors the sensor-space `CSV_stats_FWE.ipynb` workflow but adapted for
# source estimation output.
#
# **Sections:**
# 1. Compute stats (mean, SEM, t-tests, cluster perm, TFCE) per ROI
# 2. Visualize SVM accuracy with cluster significance shading
# 3. Visualize SVM accuracy with TFCE significance shading
# 4. Source-space ERP time courses per ROI (class-averaged)
# 5. Combined multi-ROI panel figures

# %% [markdown]
# ## Configuration
# %%
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests
from mne.stats import permutation_cluster_1samp_test
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                if '__file__' in dir() else os.getcwd())

from config import (
    SUBJECT_IDS, SVM_OUTPUT_ROOT, SPEECH_ROIS,
    SW_DUR, SW_STEP_SIZE, BASELINE_WINDOWS,
    ROI_TIMESERIES_ROOT,
)

# ──────────────────────────────────────────────────────────────
# USER SETTINGS — defaults for notebook use, overridden by CLI args
# ──────────────────────────────────────────────────────────────
TASK         = 'overtProd'        # 'perception' or 'overtProd'
METHOD       = 'dSPM'             # 'dSPM' or 'LCMV'
FEAT_MODE    = 'vertex_pca'         # 'pca_flip', 'vertex_pca', 'vertex_selectkbest'
ATLAS        = 'aparc'            # 'aparc', 'Schaefer200', 'HCPMMP1', 'custom'
LEAKAGE_CORRECTION = False        # True if data was run with --leakage-correction
STIM_CLASSES = ['percDiff', 'prodDiff']

# Subjects to include (default: all)
SUBJECTS = SUBJECT_IDS

# Override defaults from CLI when run as a script
if __name__ == '__main__':
    import argparse
    _parser = argparse.ArgumentParser(
        description='Source-space SVM statistics & visualization'
    )
    _parser.add_argument('--task', default=TASK,
                         choices=['perception', 'overtProd'])
    _parser.add_argument('--method', default=METHOD,
                         choices=['dSPM', 'LCMV'])
    _parser.add_argument('--feature-mode', default=FEAT_MODE,
                         choices=['pca_flip', 'vertex_pca', 'vertex_selectkbest'])
    _parser.add_argument('--atlas', default=ATLAS,
                         choices=['aparc', 'HCPMMP1', 'Schaefer200', 'custom'])
    _parser.add_argument('--leakage-correction', action='store_true',
                         default=LEAKAGE_CORRECTION)
    _parser.add_argument('--stim-classes', nargs='+', default=STIM_CLASSES,
                         choices=['percDiff', 'prodDiff'])
    _parser.add_argument('--subjects', nargs='+', default=None)
    _parser.add_argument('--skip-erp', action='store_true',
                         help='Skip source ERP computation (slow)')
    _args = _parser.parse_args()

    TASK = _args.task
    METHOD = _args.method
    FEAT_MODE = _args.feature_mode
    ATLAS = _args.atlas
    LEAKAGE_CORRECTION = _args.leakage_correction
    STIM_CLASSES = _args.stim_classes
    SUBJECTS = _args.subjects if _args.subjects else SUBJECT_IDS
    RUN_ERP = not _args.skip_erp

# Permutation test settings
N_PERMUTATIONS = 1024
TFCE_THRESHOLD = dict(start=0, step=0.2)

# Plot style
STIM_COLORS = {
    'percDiff': 'tab:blue',
    'prodDiff': 'darkorange',
}
STIM_LABELS = {
    'percDiff': 'percDiff',
    'prodDiff': 'prodDiff',
}

# ROI display names (for plot titles)
ROI_DISPLAY_NAMES = {
    'Temporal':            'Temporal',
    'Inferior_Frontal':    'Inferior Frontal',
    'Superior_Frontal':    'Superior Frontal',
    'Superior_Parietal':   'Superior Parietal',
    'Temporal_RH':         'Temporal (RH)',
    'Inferior_Frontal_RH': 'Inferior Frontal (RH)',
    'Superior_Frontal_RH': 'Superior Frontal (RH)',
    'Superior_Parietal_RH':'Superior Parietal (RH)',
}

# ──────────────────────────────────────────────────────────────
# Derived paths
# ──────────────────────────────────────────────────────────────
num_subj = len(SUBJECTS)
sw_tag = f'{SW_DUR}_{SW_STEP_SIZE}'
LEAKAGE_TAG = 'leakage_corrected' if LEAKAGE_CORRECTION else 'raw'

FIGURES_DIR = (
    SVM_OUTPUT_ROOT / TASK / METHOD / ATLAS / FEAT_MODE
    / LEAKAGE_TAG / sw_tag / 'figures'
)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# %% [markdown]
# ## Helper functions

# %%
def find_contiguous_clusters(mask):
    """Find start and end indices of contiguous True values in a boolean array."""
    clusters = []
    in_cluster = False
    start_idx = None
    for i, val in enumerate(mask):
        if val and not in_cluster:
            start_idx = i
            in_cluster = True
        elif not val and in_cluster:
            clusters.append((start_idx, i - 1))
            in_cluster = False
    if in_cluster:
        clusters.append((start_idx, len(mask) - 1))
    return clusters


def load_subject_csvs(task, method, feat_mode, stim_class, subjects):
    """Load per-subject SVM result CSVs and return list of DataFrames."""
    subj_dfs = []
    for subj in subjects:
        csv_path = (
            SVM_OUTPUT_ROOT / task / method / ATLAS / feat_mode
            / LEAKAGE_TAG / sw_tag / stim_class
            / f'{subj}_{task}_{stim_class}_{SW_DUR}_{SW_STEP_SIZE}.csv'
        )
        if not csv_path.exists():
            print(f'  WARNING: missing {csv_path.name}, skipping {subj}')
            continue
        subj_dfs.append(pd.read_csv(csv_path))
    return subj_dfs


def report_clusters(roi_name, ms_values, acc_values, clusters, label=''):
    """Print cluster onset, time range, and peak accuracy."""
    display_name = ROI_DISPLAY_NAMES.get(roi_name, roi_name)
    if not clusters:
        print(f'  {display_name}: no significant {label} clusters')
        return
    print(f'  {display_name}: {len(clusters)} significant {label} cluster(s)')
    first_onset = ms_values[clusters[0][0]]
    print(f'    First onset: {first_onset:.1f} ms')
    global_max_acc = -np.inf
    global_max_time = None
    for i, (s, e) in enumerate(clusters):
        c_accs = acc_values[s:e + 1]
        peak_idx = np.argmax(c_accs)
        peak_acc = c_accs[peak_idx]
        peak_time = ms_values[s + peak_idx]
        print(f'    Cluster {i+1}: {ms_values[s]:.1f} to {ms_values[e]:.1f} ms, '
              f'peak acc={peak_acc:.4f} at {peak_time:.1f} ms')
        if peak_acc > global_max_acc:
            global_max_acc = peak_acc
            global_max_time = peak_time
    print(f'    Global peak: {global_max_acc:.4f} at {global_max_time:.1f} ms')


# %% [markdown]
# ## Step 1: Compute group-level statistics
# %%
def compute_stats(task, method, feat_mode, stim_class, subjects):
    """
    Compute group-level stats for one task/method/stim_class combination.

    Mirrors the sensor-space CSV_stats_FWE.ipynb approach:
      - Mean and SEM across subjects
      - Pointwise one-sample t-tests (accuracy > chance=0.5)
      - Bonferroni and FDR correction per ROI
      - Standard cluster-based permutation test per ROI
      - TFCE per ROI

    Returns (mean_df, sem_df, stats_df) and saves CSVs.
    """
    subj_dfs = load_subject_csvs(task, method, feat_mode, stim_class, subjects)
    n_subj = len(subj_dfs)
    if n_subj == 0:
        print(f'  No data found for {stim_class}')
        return None, None, None

    print(f'  Loaded {n_subj} subjects for {stim_class}')

    # Reference for key/ms columns
    ref_df = subj_dfs[0][['key', 'ms']].copy()
    n_obs = len(ref_df)

    # Build accuracy matrix: (n_obs, n_subjects)
    acc_matrix = np.column_stack([s['SVM_acc'].values[:n_obs] for s in subj_dfs])

    # Mean and SEM
    mean_acc = acc_matrix.mean(axis=1)
    sem_acc = np.std(acc_matrix, axis=1, ddof=1) / np.sqrt(n_subj)

    mean_df = ref_df.copy()
    mean_df['SVM_acc'] = mean_acc

    sem_df = ref_df.copy()
    sem_df['SVM_sem'] = sem_acc

    # Initialize stats arrays
    t_stats = np.zeros(n_obs)
    p_values = np.ones(n_obs)
    bonf_pvals = np.ones(n_obs)
    fdr_pvals = np.ones(n_obs)
    bonf_reject = np.zeros(n_obs, dtype=bool)
    fdr_reject = np.zeros(n_obs, dtype=bool)
    cluster_mask = np.zeros(n_obs, dtype=bool)
    cluster_pvals_arr = np.ones(n_obs)
    tfce_scores_arr = np.zeros(n_obs)
    tfce_pvals_arr = np.ones(n_obs)
    tfce_mask_arr = np.zeros(n_obs, dtype=bool)

    # Run statistics independently per ROI (key)
    keys = ref_df['key'].unique()
    for key in keys:
        key_idx = np.where(ref_df['key'].values == key)[0]
        n_times = len(key_idx)

        # X: (n_subjects, n_times), centered at chance (0.5)
        X = acc_matrix[key_idx, :].T - 0.5

        # 1) Pointwise one-sample t-tests (> 0 = accuracy > chance)
        key_t = np.zeros(n_times)
        key_p = np.ones(n_times)
        for j in range(n_times):
            key_t[j], key_p[j] = ttest_1samp(X[:, j], 0, alternative='greater')
        t_stats[key_idx] = key_t
        p_values[key_idx] = key_p

        # 2) Bonferroni & FDR correction (per ROI)
        rej_b, pv_b, _, _ = multipletests(key_p, alpha=0.05, method='bonferroni')
        rej_f, pv_f, _, _ = multipletests(key_p, alpha=0.05, method='fdr_bh')
        bonf_pvals[key_idx] = pv_b
        fdr_pvals[key_idx] = pv_f
        bonf_reject[key_idx] = rej_b
        fdr_reject[key_idx] = rej_f

        # 3) Standard cluster-based permutation test
        T_obs_c, clusters_c, pv_c, _ = permutation_cluster_1samp_test(
            X, threshold=None, n_permutations=N_PERMUTATIONS,
            tail=1, out_type='mask', verbose=False
        )
        for ic, cpv in enumerate(pv_c):
            cluster_points = key_idx[clusters_c[ic]]
            cluster_pvals_arr[cluster_points] = cpv
            if cpv < 0.05:
                cluster_mask[cluster_points] = True

        # 4) TFCE
        T_obs_tfce, clusters_tfce, pv_tfce, _ = permutation_cluster_1samp_test(
            X, threshold=TFCE_THRESHOLD, n_permutations=N_PERMUTATIONS,
            tail=1, out_type='mask', n_jobs=-1, verbose=False
        )
        tfce_scores_arr[key_idx] = T_obs_tfce
        tfce_pvals_arr[key_idx] = pv_tfce
        tfce_mask_arr[key_idx] = pv_tfce < 0.05

        n_sig = int(np.sum(pv_tfce < 0.05))
        n_clust_sig = int(np.sum(pv_c < 0.05)) if len(pv_c) > 0 else 0
        display = ROI_DISPLAY_NAMES.get(key, key)
        print(f'    {display}: TFCE {n_sig}/{n_times} sig | '
              f'Cluster {n_clust_sig} sig clusters')

    # Assemble stats DataFrame
    stats_df = ref_df.copy()
    stats_df['T-statistic'] = t_stats
    stats_df['p_value'] = p_values
    stats_df['p_bonferroni'] = bonf_pvals
    stats_df['p_fdr'] = fdr_pvals
    stats_df['sig_bonferroni'] = bonf_reject
    stats_df['sig_fdr'] = fdr_reject
    stats_df['sig_cluster'] = cluster_mask
    stats_df['p_cluster'] = cluster_pvals_arr
    stats_df['tfce_score'] = tfce_scores_arr
    stats_df['p_tfce'] = tfce_pvals_arr
    stats_df['sig_tfce'] = tfce_mask_arr

    # Save CSVs
    out_dir = SVM_OUTPUT_ROOT / task / method / ATLAS / feat_mode / LEAKAGE_TAG / sw_tag / stim_class
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f'{task}_{stim_class}_{SW_DUR}_{SW_STEP_SIZE}_{n_subj}subjAvg'

    mean_df.to_csv(out_dir / f'{base}.csv', index=False)
    sem_df.to_csv(out_dir / f'{base}_sem.csv', index=False)
    stats_df.to_csv(out_dir / f'{base}_stats.csv', index=False)
    print(f'  Saved: {out_dir / base}_[.csv, _sem.csv, _stats.csv]')

    return mean_df, sem_df, stats_df


# Run stats for all stim classes
all_data = {}
for sc in STIM_CLASSES:
    print(f'\n{"="*60}')
    print(f'Computing stats: {TASK} / {METHOD} / {sc}')
    print(f'{"="*60}')
    mean_df, sem_df, stats_df = compute_stats(
        TASK, METHOD, FEAT_MODE, sc, SUBJECTS
    )
    if mean_df is not None:
        all_data[sc] = {
            'mean': mean_df,
            'sem': sem_df,
            'stats': stats_df,
        }


# %% [markdown]
# ## Step 2: Standard cluster-based permutation visualization (per ROI)

# %%
def plot_svm_accuracy_single_roi(roi_key, all_data, sig_column='sig_cluster',
                                  title_suffix='Cluster', ylim_top=0.60):
    """
    Plot SVM accuracy for one ROI with significance shading.

    Shows both stim_classes overlaid with SEM bands and
    significant time windows shaded.
    """
    display_name = ROI_DISPLAY_NAMES.get(roi_key, roi_key)

    fig, ax = plt.subplots(figsize=(12, 7))

    ms_all = []
    for sc in STIM_CLASSES:
        if sc not in all_data:
            continue
        d = all_data[sc]
        mask = d['mean']['key'] == roi_key
        ms = d['mean'].loc[mask, 'ms'].values
        acc = d['mean'].loc[mask, 'SVM_acc'].values
        sem = d['sem'].loc[mask, 'SVM_sem'].values
        sig = d['stats'].loc[mask, sig_column].values.astype(bool)
        ms_all.append(ms)

        color = STIM_COLORS[sc]
        label = STIM_LABELS[sc]

        # Accuracy line + SEM shading
        ax.plot(ms, acc, color=color, linewidth=2, label=label)
        ax.fill_between(ms, acc - sem, acc + sem, alpha=0.3, color=color)

        # Significance shading
        clusters = find_contiguous_clusters(sig)
        for s, e in clusters:
            ax.axvspan(ms[s], ms[e], alpha=0.25, color=color, zorder=0)

        # Report clusters
        report_clusters(roi_key, ms, acc, clusters, label=f'{label} {title_suffix}')

    ax.axhline(y=0.5, color='black', linestyle='--', label='chance')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    ax.set_ylabel('SVM Accuracy', fontsize=18)
    ax.set_xlabel('Time (ms)', fontsize=18)
    ax.set_title(f'{display_name} — Source-Space SVM ({METHOD})', fontsize=20)
    ax.set_ylim(top=ylim_top)
    if ms_all:
        ms_cat = np.concatenate(ms_all)
        ax.set_xlim(ms_cat.min(), ms_cat.max())
    ax.legend(loc='upper left', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=14)

    plt.tight_layout()
    return fig


# Plot each ROI
rois_in_data = all_data[STIM_CLASSES[0]]['mean']['key'].unique() if all_data else []
for roi_key in rois_in_data:
    fig = plot_svm_accuracy_single_roi(roi_key, all_data, sig_column='sig_cluster',
                                        title_suffix='Cluster')
    fig.savefig(
        FIGURES_DIR / f'{TASK}_{METHOD}_{roi_key}_SW{SW_DUR}_{SW_STEP_SIZE}_CLUSTER.svg',
        format='svg', bbox_inches='tight'
    )
    fig.savefig(
        FIGURES_DIR / f'{TASK}_{METHOD}_{roi_key}_SW{SW_DUR}_{SW_STEP_SIZE}_CLUSTER.png',
        dpi=300, bbox_inches='tight'
    )
    plt.show()


# %% [markdown]
# ## Step 3: TFCE visualization (per ROI, with TFCE score subplot)

# %%
def plot_svm_accuracy_tfce_single_roi(roi_key, all_data, ylim_top=0.62):
    """
    Plot SVM accuracy + TFCE scores for one ROI.

    Two subplots:
      Top: accuracy with SEM + TFCE significance shading
      Bottom: TFCE scores with significance shading
    """
    display_name = ROI_DISPLAY_NAMES.get(roi_key, roi_key)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10),
                                    height_ratios=[3, 1], sharex=True)

    ms_all = []
    for sc in STIM_CLASSES:
        if sc not in all_data:
            continue
        d = all_data[sc]
        mask = d['mean']['key'] == roi_key
        ms = d['mean'].loc[mask, 'ms'].values
        acc = d['mean'].loc[mask, 'SVM_acc'].values
        sem = d['sem'].loc[mask, 'SVM_sem'].values
        sig_tfce = d['stats'].loc[mask, 'sig_tfce'].values.astype(bool)
        tfce_score = d['stats'].loc[mask, 'tfce_score'].values
        ms_all.append(ms)

        color = STIM_COLORS[sc]
        label = STIM_LABELS[sc]

        # Top: accuracy
        ax1.plot(ms, acc, color=color, linewidth=2, label=label)
        ax1.fill_between(ms, acc - sem, acc + sem, alpha=0.3, color=color)

        # TFCE significance shading on both panels
        clusters = find_contiguous_clusters(sig_tfce)
        for s, e in clusters:
            ax1.axvspan(ms[s], ms[e], alpha=0.25, color=color, zorder=0)
            ax2.axvspan(ms[s], ms[e], alpha=0.25, color=color, zorder=0)

        # Bottom: TFCE scores
        ax2.plot(ms, tfce_score, color=color, linewidth=2,
                 label=f'TFCE score ({label})')

        report_clusters(roi_key, ms, acc, clusters, label=f'{label} TFCE')

    ax1.axhline(y=0.5, color='black', linestyle='--', label='chance')
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax1.set_ylabel('SVM Accuracy', fontsize=14)
    ax1.set_title(f'{display_name} — Source-Space SVM ({METHOD})', fontsize=16)
    ax1.set_ylim(top=ylim_top)
    if ms_all:
        ms_cat = np.concatenate(ms_all)
        ax1.set_xlim(ms_cat.min(), ms_cat.max())
    ax1.legend(loc='upper left', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=14)

    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax2.set_xlabel('Time (ms)', fontsize=14)
    ax2.set_ylabel('TFCE Score', fontsize=14)
    ax2.legend(loc='upper left', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=14)

    plt.tight_layout()
    return fig


for roi_key in rois_in_data:
    fig = plot_svm_accuracy_tfce_single_roi(roi_key, all_data)
    fig.savefig(
        FIGURES_DIR / f'{TASK}_{METHOD}_{roi_key}_SW{SW_DUR}_{SW_STEP_SIZE}_TFCE.svg',
        format='svg', bbox_inches='tight'
    )
    fig.savefig(
        FIGURES_DIR / f'{TASK}_{METHOD}_{roi_key}_SW{SW_DUR}_{SW_STEP_SIZE}_TFCE.png',
        dpi=300, bbox_inches='tight'
    )
    plt.show()


# %% [markdown]
# ## Step 4: Multi-ROI panel figure (cluster-based)
#
# All ROIs in a single figure (2 rows x 4 cols for 8 ROIs).

# %%
def plot_multi_roi_panel(all_data, sig_column='sig_cluster',
                          title_suffix='Cluster', ylim_top=0.62):
    """Create a multi-panel figure with one subplot per ROI."""
    rois = all_data[STIM_CLASSES[0]]['mean']['key'].unique()
    n_rois = len(rois)
    n_cols = 4
    n_rows = int(np.ceil(n_rois / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows),
                              squeeze=False, sharey=True)

    # Determine shared xlim from data
    ms_all = []
    for sc in STIM_CLASSES:
        if sc not in all_data:
            continue
        ms_all.append(all_data[sc]['mean']['ms'].values)
    if ms_all:
        ms_cat = np.concatenate(ms_all)
        data_xlim = (ms_cat.min(), ms_cat.max())
    else:
        data_xlim = None

    for i, roi_key in enumerate(rois):
        ax = axes[i // n_cols, i % n_cols]
        display_name = ROI_DISPLAY_NAMES.get(roi_key, roi_key)

        for sc in STIM_CLASSES:
            if sc not in all_data:
                continue
            d = all_data[sc]
            mask = d['mean']['key'] == roi_key
            ms = d['mean'].loc[mask, 'ms'].values
            acc = d['mean'].loc[mask, 'SVM_acc'].values
            sem = d['sem'].loc[mask, 'SVM_sem'].values
            sig = d['stats'].loc[mask, sig_column].values.astype(bool)

            color = STIM_COLORS[sc]
            label = STIM_LABELS[sc]

            ax.plot(ms, acc, color=color, linewidth=1.5, label=label)
            ax.fill_between(ms, acc - sem, acc + sem, alpha=0.25, color=color)

            clusters = find_contiguous_clusters(sig)
            for s, e in clusters:
                ax.axvspan(ms[s], ms[e], alpha=0.2, color=color, zorder=0)

        ax.axhline(y=0.5, color='black', linestyle='--', linewidth=0.8)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        ax.set_title(display_name, fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=11)
        ax.set_ylim(top=ylim_top)
        if data_xlim is not None:
            ax.set_xlim(*data_xlim)

        if i % n_cols == 0:
            ax.set_ylabel('SVM Accuracy', fontsize=12)
        if i // n_cols == n_rows - 1:
            ax.set_xlabel('Time (ms)', fontsize=12)

    # Legend on first axis only
    axes[0, 0].legend(loc='upper left', fontsize=10)

    # Hide empty subplots
    for j in range(i + 1, n_rows * n_cols):
        axes[j // n_cols, j % n_cols].set_visible(False)

    fig.suptitle(
        f'{TASK} | {METHOD} | {FEAT_MODE} — Source-Space SVM ({title_suffix})',
        fontsize=16
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


if all_data:
    # Cluster panel
    fig = plot_multi_roi_panel(all_data, sig_column='sig_cluster',
                                title_suffix='Cluster')
    fig.savefig(
        FIGURES_DIR / f'{TASK}_{METHOD}_allROIs_SW{SW_DUR}_{SW_STEP_SIZE}_CLUSTER.svg',
        format='svg', bbox_inches='tight'
    )
    fig.savefig(
        FIGURES_DIR / f'{TASK}_{METHOD}_allROIs_SW{SW_DUR}_{SW_STEP_SIZE}_CLUSTER.png',
        dpi=300, bbox_inches='tight'
    )
    plt.show()

    # TFCE panel
    fig = plot_multi_roi_panel(all_data, sig_column='sig_tfce',
                                title_suffix='TFCE')
    fig.savefig(
        FIGURES_DIR / f'{TASK}_{METHOD}_allROIs_SW{SW_DUR}_{SW_STEP_SIZE}_TFCE.svg',
        format='svg', bbox_inches='tight'
    )
    fig.savefig(
        FIGURES_DIR / f'{TASK}_{METHOD}_allROIs_SW{SW_DUR}_{SW_STEP_SIZE}_TFCE.png',
        dpi=300, bbox_inches='tight'
    )
    plt.show()


# %% [markdown]
# ## Step 5: Source-space ERP visualization per ROI
#
# Loads source estimates for each subject, extracts ROI time courses
# (PCA-flip), and plots class-averaged ERPs with SEM.

# %%
def _load_subject_npz(subj, task, method, stim_class, feat_mode='pca_flip',
                      atlas='aparc', leakage_correction=False):
    """
    Try to load cached .npz ROI time series for one subject.

    Returns (roi_data_dict, y, times_s) or None if not found.
    roi_data_dict maps roi_name → (n_epochs, n_times).
    """
    leakage_tag = 'leakage_corrected' if leakage_correction else 'raw'
    npz_file = (
        ROI_TIMESERIES_ROOT / task / method / atlas
        / feat_mode / leakage_tag
        / f'{subj}_{task}_{stim_class}.npz'
    )
    if not npz_file.exists():
        return None
    data = np.load(npz_file, allow_pickle=True)
    roi_names = list(data['roi_names'])
    y = data['y']
    times_s = data['times']
    roi_data = {}
    for name in roi_names:
        arr = data[name]
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[:, :, 0]  # pca_flip: (n_epochs, n_times, 1) → (n_epochs, n_times)
        elif arr.ndim == 3:
            arr = arr.mean(axis=2)  # vertex modes: average across vertices for ERP
        roi_data[name] = arr
    return roi_data, y, times_s


def _compute_one_subject_erp(subj, task, method, stim_class,
                              fwd, roi_labels, roi_names, src,
                              baseline_tmin, baseline_tmax):
    """Compute source ERPs for a single subject (no cached .npz).

    Saves the result as .npz so future runs load from cache.
    """
    from data_loader import load_subject_epochs
    from inverse_pipelines import run_dspm_lowram, run_lcmv_lowram

    try:
        epochs, y, sfreq = load_subject_epochs(subj, task, stim_class)
    except FileNotFoundError as e:
        print(f'    SKIP {subj}: {e}')
        return None

    # Use low-RAM variants: generator + immediate ROI extraction
    if method == 'dSPM':
        X_roi, stc_times = run_dspm_lowram(
            epochs, fwd, baseline_tmin, baseline_tmax,
            roi_labels, src, feature_mode='pca_flip'
        )
    elif method == 'LCMV':
        X_roi, stc_times = run_lcmv_lowram(
            epochs, fwd, baseline_tmin, baseline_tmax,
            roi_labels, src, feature_mode='pca_flip'
        )

    # X_roi shape: (n_epochs, n_rois, n_times)
    roi_data = {}
    for i, name in enumerate(roi_names):
        roi_data[name] = X_roi[:, i, :]

    # Save .npz cache for future runs (no leakage correction → 'raw')
    ts_dir = ROI_TIMESERIES_ROOT / task / method / ATLAS / 'pca_flip' / 'raw'
    ts_dir.mkdir(parents=True, exist_ok=True)
    save_dict = {
        'y': y,
        'times': stc_times,
        'sfreq': np.array(sfreq),
        'roi_names': np.array(roi_names),
    }
    for name in roi_names:
        save_dict[name] = roi_data[name][:, :, np.newaxis]  # (n_epochs, n_times, 1)
    npz_file = ts_dir / f'{subj}_{task}_{stim_class}.npz'
    np.savez_compressed(npz_file, **save_dict)
    print(f'  Cached {subj} → {npz_file.name}')

    return roi_data, y, stc_times


def compute_source_erps(task, method, stim_class, subjects):
    """
    Load data and compute source estimates for all subjects,
    then extract ROI time courses.

    Optimized to:
      1. Load cached .npz files from the SVM pipeline when available
      2. Use low-RAM generator-based inverse for uncached subjects
      3. Parallelize uncached subjects across CPU cores

    Returns
    -------
    all_X : dict
        {roi_name: np.ndarray of shape (n_total_epochs, n_times)}
    all_y : np.ndarray
        Class labels for all epochs.
    times_ms : np.ndarray
        Time axis in milliseconds.
    """
    # --- Phase 1: Try loading from cached .npz files ---
    all_roi_data = {}
    all_y = []
    times_s = None
    roi_names = None
    uncached_subjects = []

    print('Phase 1: Loading cached .npz time series...')
    for subj in subjects:
        # ERPs always use pca_flip (1 virtual sensor per ROI);
        # try pca_flip cache first, then fall back to current FEAT_MODE
        result = _load_subject_npz(subj, task, method, stim_class, 'pca_flip',
                                   atlas=ATLAS, leakage_correction=LEAKAGE_CORRECTION)
        if result is None:
            result = _load_subject_npz(subj, task, method, stim_class, FEAT_MODE,
                                       atlas=ATLAS, leakage_correction=LEAKAGE_CORRECTION)
        if result is not None:
            roi_data, y, t_s = result
            if roi_names is None:
                roi_names = list(roi_data.keys())
                all_roi_data = {name: [] for name in roi_names}
                times_s = t_s
            for name in roi_names:
                all_roi_data[name].append(roi_data[name])
            all_y.append(y)
            print(f'  {subj}: loaded from cache ({len(y)} epochs)')
        else:
            uncached_subjects.append(subj)

    # --- Phase 2: Compute uncached subjects ---
    if uncached_subjects:
        from data_loader import load_subject_epochs
        from forward_model import setup_fsaverage, make_forward, build_roi_labels
        from joblib import Parallel, delayed

        print(f'\nPhase 2: Computing inverse for {len(uncached_subjects)} '
              f'uncached subjects...')

        subjects_dir, fs_dir, src, bem = setup_fsaverage()
        if ATLAS in SPEECH_ROIS:
            roi_dict = build_roi_labels(subjects_dir, atlas=ATLAS,
                                         composite_rois=SPEECH_ROIS[ATLAS])
        else:
            roi_dict = build_roi_labels(subjects_dir, atlas=ATLAS)
        fwd_roi_labels = list(roi_dict.values())
        fwd_roi_names = list(roi_dict.keys())

        if roi_names is None:
            roi_names = fwd_roi_names
            all_roi_data = {name: [] for name in roi_names}

        baseline_tmin, baseline_tmax = BASELINE_WINDOWS[task]

        # Build forward model once
        first_epochs, _, _ = load_subject_epochs(
            uncached_subjects[0], task, stim_class
        )
        fwd = make_forward(first_epochs.info, src, bem)

        # Process uncached subjects in parallel
        n_jobs = min(len(uncached_subjects), os.cpu_count() or 4)
        print(f'  Using {n_jobs} parallel workers')

        results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(_compute_one_subject_erp)(
                subj, task, method, stim_class,
                fwd, fwd_roi_labels, fwd_roi_names, src,
                baseline_tmin, baseline_tmax
            )
            for subj in uncached_subjects
        )

        for subj, result in zip(uncached_subjects, results):
            if result is None:
                continue
            roi_data, y, stc_times = result
            if times_s is None:
                times_s = stc_times
            for name in roi_names:
                all_roi_data[name].append(roi_data[name])
            all_y.append(y)

    if not all_y:
        raise RuntimeError(f'No data found for {task}/{method}/{stim_class}')

    # Concatenate across subjects
    for name in roi_names:
        all_roi_data[name] = np.concatenate(all_roi_data[name], axis=0)
    all_y = np.concatenate(all_y)
    times_ms = times_s * 1000.0

    n_cached = len(subjects) - len(uncached_subjects) if uncached_subjects else len(subjects)
    print(f'\nDone: {len(all_y)} total epochs from {len(subjects)} subjects '
          f'({n_cached} cached), {len(roi_names)} ROIs, {len(times_ms)} time points')

    return all_roi_data, all_y, times_ms


def plot_source_erps(all_roi_data, all_y, times_ms, ylim=None):
    """
    Plot class-averaged source-space ERPs for each ROI.

    Multi-panel figure: one subplot per ROI showing class 0 vs class 1
    with SEM bands.
    """
    roi_names = list(all_roi_data.keys())
    n_rois = len(roi_names)
    n_cols = 4
    n_rows = int(np.ceil(n_rois / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4.5 * n_rows),
                              squeeze=False)

    class_colors = {0: 'tab:blue', 1: 'tab:red'}
    class_labels = {0: 'Class 0', 1: 'Class 1'}

    for i, roi_name in enumerate(roi_names):
        ax = axes[i // n_cols, i % n_cols]
        display_name = ROI_DISPLAY_NAMES.get(roi_name, roi_name)
        X = all_roi_data[roi_name]

        for cls in [0, 1]:
            cls_mask = all_y == cls
            mean_tc = X[cls_mask].mean(axis=0)
            sem_tc = X[cls_mask].std(axis=0) / np.sqrt(cls_mask.sum())

            ax.plot(times_ms, mean_tc, color=class_colors[cls],
                    linewidth=1.5, label=class_labels[cls])
            ax.fill_between(times_ms, mean_tc - sem_tc, mean_tc + sem_tc,
                            color=class_colors[cls], alpha=0.2)

        ax.axvline(0, color='black', linestyle='--', linewidth=0.8)
        ax.set_title(display_name, fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=11)

        if i % n_cols == 0:
            ax.set_ylabel('Source amplitude', fontsize=12)
        if i // n_cols == n_rows - 1:
            ax.set_xlabel('Time (ms)', fontsize=12)
        if ylim is not None:
            ax.set_ylim(ylim)

    axes[0, 0].legend(loc='upper left', fontsize=10)

    for j in range(i + 1, n_rows * n_cols):
        axes[j // n_cols, j % n_cols].set_visible(False)

    fig.suptitle(
        f'{TASK} | {METHOD} — Source-Space ERP by ROI',
        fontsize=16
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# %% [markdown]
# ### Run source ERP computation and plotting
#
# This cell is slow (loads all subjects, runs inverse). Set `RUN_ERP = True`
# to execute, or skip if you only need SVM stats/plots.

# %%
if 'RUN_ERP' not in dir():
    RUN_ERP = True  # Set to True to compute and plot source ERPs

if RUN_ERP:
    for sc in STIM_CLASSES:
        print(f'\n{"="*60}')
        print(f'Source ERPs: {TASK} / {METHOD} / {sc}')
        print(f'{"="*60}')

        all_roi_data, all_y, times_ms = compute_source_erps(
            TASK, METHOD, sc, SUBJECTS
        )

        fig = plot_source_erps(all_roi_data, all_y, times_ms)
        fig.savefig(
            FIGURES_DIR / f'{TASK}_{METHOD}_{sc}_sourceERP_allROIs.svg',
            format='svg', bbox_inches='tight'
        )
        fig.savefig(
            FIGURES_DIR / f'{TASK}_{METHOD}_{sc}_sourceERP_allROIs.png',
            dpi=300, bbox_inches='tight'
        )
        plt.show()


# %% [markdown]
# ## Step 6: Print summary of all significant clusters

# %%
if all_data:
    print(f'\n{"="*60}')
    print(f'CLUSTER SUMMARY: {TASK} / {METHOD}')
    print(f'{"="*60}')

    for roi_key in rois_in_data:
        print(f'\n--- {ROI_DISPLAY_NAMES.get(roi_key, roi_key)} ---')
        for sc in STIM_CLASSES:
            if sc not in all_data:
                continue
            d = all_data[sc]
            mask = d['mean']['key'] == roi_key
            ms = d['mean'].loc[mask, 'ms'].values
            acc = d['mean'].loc[mask, 'SVM_acc'].values

            # Standard cluster
            sig_c = d['stats'].loc[mask, 'sig_cluster'].values.astype(bool)
            clusters_c = find_contiguous_clusters(sig_c)
            report_clusters(roi_key, ms, acc, clusters_c, label=f'{sc} Cluster')

            # TFCE
            sig_t = d['stats'].loc[mask, 'sig_tfce'].values.astype(bool)
            clusters_t = find_contiguous_clusters(sig_t)
            report_clusters(roi_key, ms, acc, clusters_t, label=f'{sc} TFCE')

    print(f'\n{"="*60}')
    print(f'Figures saved to: {FIGURES_DIR}')
    print(f'{"="*60}')
