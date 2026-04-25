#!/usr/bin/env python3
# %% [markdown]
# # Source Estimation + SVM Decoding Pipeline
#
# This notebook runs the full pipeline step-by-step for a single subject,
# showing intermediate outputs at each stage. Use this to inspect and
# understand the pipeline before launching batch runs.
#
# **Pipeline steps:**
# 1. Load EEGLAB-preprocessed data → MNE Epochs
# 2. Set up fsaverage forward model
# 3. Run inverse pipeline (dSPM or LCMV) → source estimates
# 4. Extract ROI time courses
# 5. Run sliding-window SVM decoding per ROI
# 6. Plot results

# %% [markdown]
# ## Configuration

# %%
import os
import sys
import warnings

import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

# Add source_estimation directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                if '__file__' in dir() else os.getcwd())

import mne

# ──────────────────────────────────────────────────────────────
# USER SETTINGS — change these to match your run
# ──────────────────────────────────────────────────────────────
SUBJECT    = 'EEGPROD4001'
TASK       = 'perception'        # 'perception' or 'overtProd'
STIM_CLASS = 'percDiff'         # 'prodDiff' or 'percDiff'
METHOD     = 'dSPM'             # 'dSPM' or 'LCMV'
FEAT_MODE  = 'pca_flip'         # 'pca_flip', 'vertex_pca', or 'vertex_selectkbest'

# %% [markdown]
# ## Step 1: Load EEGLAB data

# %%
from data_loader import load_subject_epochs

epochs, y, sfreq = load_subject_epochs(SUBJECT, TASK, STIM_CLASS)

print(f'Epochs shape:  {epochs.get_data().shape}')
print(f'Labels:        {np.bincount(y)} (/s/, /t/)')
print(f'sfreq:         {sfreq} Hz')
print(f'Time range:    [{epochs.tmin:.3f}, {epochs.tmax:.3f}] s')
print(f'Channels:      {len(epochs.ch_names)}')
print(f'Data range:    [{epochs.get_data().min():.2e}, {epochs.get_data().max():.2e}] V')
print(f'Channel names: {epochs.ch_names}')
# %% [markdown]
# ### Quick sanity check: plot the ERP (evoked) for each class

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

for cls, ax in zip([0, 1], axes):
    evoked = epochs[y == cls].average()
    evoked.plot(axes=ax, show=False, spatial_colors=True, time_unit='ms')
    ax.set_title(f'{STIM_CLASS} Class {cls} ({np.sum(y == cls)} trials)')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Step 2: Forward model (fsaverage template)

# %%
from forward_model import setup_fsaverage, make_forward, build_roi_labels

subjects_dir, fs_dir, src, bem = setup_fsaverage()
fwd = make_forward(epochs.info, src, bem)
roi_dict = build_roi_labels(subjects_dir)

print(f'\nForward solution: {fwd["nsource"]} sources')
print(f'ROIs ({len(roi_dict)}):')
for name, label in roi_dict.items():
    print(f'  {name}: {len(label.vertices)} vertices')

# %% [markdown]
# ## Step 3: Inverse solution

# %%
from config import BASELINE_WINDOWS, DECODE_TMIN

baseline_tmin, baseline_tmax = BASELINE_WINDOWS[TASK]
decode_tmin = DECODE_TMIN[TASK]

print(f'Baseline window:  [{baseline_tmin}, {baseline_tmax}] s')
print(f'Decode starts at: {decode_tmin} s')

# %%
from inverse_pipelines import run_dspm, run_lcmv

if METHOD == 'dSPM':
    stcs = run_dspm(epochs, fwd, baseline_tmin, baseline_tmax)
elif METHOD == 'LCMV':
    stcs = run_lcmv(epochs, fwd, baseline_tmin, baseline_tmax)

stc0 = stcs[0]
print(f'\nSource estimates: {len(stcs)} epochs')
print(f'STC shape:        {stc0.data.shape} (vertices x timepoints)')
print(f'STC time range:   [{stc0.times[0]:.3f}, {stc0.times[-1]:.3f}] s')
print(f'STC value range:  [{stc0.data.min():.4f}, {stc0.data.max():.4f}]')

# %% [markdown]
# ## Step 4: Extract ROI features

# %%
from svm_decoding import extract_roi_data_pca_flip, extract_roi_data_vertices

roi_labels = list(roi_dict.values())
roi_names = list(roi_dict.keys())

if FEAT_MODE == 'pca_flip':
    X_all = extract_roi_data_pca_flip(stcs, roi_labels, src)
    print(f'PCA-flip data shape: {X_all.shape}  (epochs x ROIs x timepoints)')
else:
    # Show shape for the first ROI as example
    first_label = roi_dict[roi_names[0]]
    X_vert = extract_roi_data_vertices(stcs, first_label)
    print(f'Vertex data shape for {roi_names[0]}: {X_vert.shape}  '
          f'(epochs x vertices x timepoints)')

# %% [markdown]
# ### Visualize ROI time courses (class-averaged)

# %%
if FEAT_MODE == 'pca_flip':
    n_rois = X_all.shape[1]
    n_cols = 4
    n_rows = int(np.ceil(n_rois / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.5 * n_rows),
                             squeeze=False)

    times_ms = stc0.times * 1000.0

    for i, roi_name in enumerate(roi_names):
        ax = axes[i // n_cols, i % n_cols]
        for cls, color, label in [(0, 'blue', 'Class 0'), (1, 'red', 'Class 1')]:
            mean_tc = X_all[y == cls, i, :].mean(axis=0)
            sem_tc = X_all[y == cls, i, :].std(axis=0) / np.sqrt(np.sum(y == cls))
            ax.plot(times_ms, mean_tc, color=color, label=label)
            ax.fill_between(times_ms, mean_tc - sem_tc, mean_tc + sem_tc,
                            color=color, alpha=0.15)
        ax.axvline(0, color='k', linestyle='--', linewidth=0.8)
        ax.axvline(decode_tmin * 1000, color='gray', linestyle=':', linewidth=0.8)
        ax.set_title(roi_name, fontsize=10)
        ax.set_xlabel('Time (ms)')

    # Hide empty axes
    for j in range(i + 1, n_rows * n_cols):
        axes[j // n_cols, j % n_cols].set_visible(False)

    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f'{SUBJECT} | {TASK} | {METHOD} — ROI time courses', fontsize=12)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Step 5: SVM decoding (sliding window)

# %%
from svm_decoding import sliding_window_svm_decode
from config import SW_DUR, SW_STEP_SIZE

print(f'Sliding window: {SW_DUR} ms duration, {SW_STEP_SIZE} ms step')
print(f'Feature mode:   {FEAT_MODE}')
print(f'Decode tmin:    {decode_tmin} s')
print()

results_all_rois = {}

if FEAT_MODE == 'pca_flip':
    for i, roi_name in enumerate(roi_names):
        print(f'Decoding ROI: {roi_name}')
        X_roi = X_all[:, i, :]
        results = sliding_window_svm_decode(
            X_roi, y, sfreq, SW_DUR, SW_STEP_SIZE,
            epochs.tmin, decode_tmin, feature_mode='pca_flip',
            times=epochs.times
        )
        results_all_rois[roi_name] = results
        accs = [r['SVM_acc'] for r in results]
        print(f'  Accuracy range: [{min(accs):.3f}, {max(accs):.3f}]\n')
else:
    for roi_name, roi_label in roi_dict.items():
        print(f'Decoding ROI: {roi_name} ({FEAT_MODE})')
        X_roi = extract_roi_data_vertices(stcs, roi_label)
        results = sliding_window_svm_decode(
            X_roi, y, sfreq, SW_DUR, SW_STEP_SIZE,
            epochs.tmin, decode_tmin, feature_mode=FEAT_MODE,
            times=epochs.times
        )
        results_all_rois[roi_name] = results
        accs = [r['SVM_acc'] for r in results]
        print(f'  Accuracy range: [{min(accs):.3f}, {max(accs):.3f}]\n')

# %% [markdown]
# ## Step 6: Plot decoding results

# %%
n_rois = len(results_all_rois)
n_cols = 4
n_rows = int(np.ceil(n_rois / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.5 * n_rows),
                         squeeze=False)

for i, (roi_name, results) in enumerate(results_all_rois.items()):
    ax = axes[i // n_cols, i % n_cols]
    ms_vals = [r['ms'] for r in results]
    acc_vals = [r['SVM_acc'] for r in results]

    ax.plot(ms_vals, acc_vals, 'b-', linewidth=1)
    ax.axhline(0.5, color='k', linestyle='--', linewidth=0.8, label='Chance')
    ax.axvline(0, color='gray', linestyle=':', linewidth=0.8)
    ax.set_title(roi_name, fontsize=10)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0.35, 0.75)

    peak_idx = np.argmax(acc_vals)
    ax.annotate(f'{acc_vals[peak_idx]:.2f}',
                xy=(ms_vals[peak_idx], acc_vals[peak_idx]),
                fontsize=8, color='red', ha='center', va='bottom')

for j in range(i + 1, n_rows * n_cols):
    axes[j // n_cols, j % n_cols].set_visible(False)

fig.suptitle(f'{SUBJECT} | {TASK} | {STIM_CLASS} | {METHOD} | {FEAT_MODE}\n'
             f'Sliding-window SVM accuracy ({SW_DUR}/{SW_STEP_SIZE} ms)',
             fontsize=12)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Step 7: Save results (optional)

# %%
from config import SVM_OUTPUT_ROOT
from run_source_svm import _save_results

SAVE = False  # set to True to write CSV

if SAVE:
    _save_results(SUBJECT, TASK, STIM_CLASS, METHOD, FEAT_MODE,
                  SW_DUR, SW_STEP_SIZE, results_all_rois, SVM_OUTPUT_ROOT)
    print('Results saved.')
else:
    print('Saving skipped (set SAVE = True above to write CSV).')

# %% [markdown]
# ## Batch run (all subjects)
#
# To run all subjects from the command line:
#
# ```bash
# # Sequential
# python run_source_svm.py --task overtProd --stim-class prodDiff --method dSPM
#
# # Parallel (4 workers)
# python run_parallel.py --task overtProd --stim-class prodDiff --method dSPM --n-jobs 4
# ```
