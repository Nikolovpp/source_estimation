"""
Non-interactive figure generation for the source estimation pipeline.

All functions save figures to disk (PNG) and never call plt.show().
Designed for use in batch/parallel runners where no display is available.
"""
import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt

from config import FIGURES_ROOT


def _make_fig_dir(task_cond, method, feature_mode, subj_id,
                  atlas='aparc', leakage_correction=False):
    """Create and return the figure output directory."""
    leakage_tag = 'leakage_corrected' if leakage_correction else 'raw'
    fig_dir = (
        FIGURES_ROOT / task_cond / method / atlas / feature_mode
        / leakage_tag / 'figures' / subj_id
    )
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def save_sensor_erp(epochs, y, subj_id, task_cond, stim_class,
                    method, feature_mode, atlas='aparc',
                    leakage_correction=False):
    """
    Save sensor-space ERP butterfly plot (all 64 channels) per class.

    This shows the raw input data before source estimation.
    """
    fig_dir = _make_fig_dir(task_cond, method, feature_mode, subj_id,
                            atlas=atlas, leakage_correction=leakage_correction)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    for cls, ax in zip([0, 1], axes):
        evoked = epochs[y == cls].average()
        evoked.plot(axes=ax, show=False, spatial_colors=True, time_unit='ms')
        ax.set_title(f'{stim_class} Class {cls} ({np.sum(y == cls)} trials)')

    fig.suptitle(f'{subj_id} | {task_cond} | {stim_class} — Sensor-space ERPs',
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = fig_dir / f'{subj_id}_{stim_class}_sensor_erp.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved figure: {out_path}')


def save_source_erp(roi_data, y, times, subj_id, task_cond, stim_class,
                    method, feature_mode, decode_tmin, atlas='aparc',
                    leakage_correction=False):
    """
    Save source-space ERP plot (class-averaged time courses per ROI).

    Parameters
    ----------
    roi_data : dict
        {roi_name: X_roi} where X_roi is (n_epochs, n_times) for pca_flip
        or (n_epochs, n_vertices, n_times) for vertex modes.
    times : array-like
        Time vector in seconds.
    """
    fig_dir = _make_fig_dir(task_cond, method, feature_mode, subj_id,
                            atlas=atlas, leakage_correction=leakage_correction)

    roi_names = list(roi_data.keys())
    n_rois = len(roi_names)
    n_cols = 4
    n_rows = int(np.ceil(n_rois / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.5 * n_rows),
                             squeeze=False)

    times_ms = np.asarray(times) * 1000.0

    for i, roi_name in enumerate(roi_names):
        ax = axes[i // n_cols, i % n_cols]
        X_roi = roi_data[roi_name]

        if X_roi.ndim == 3:
            # vertex modes: (epochs, vertices, times) → average over vertices
            tc = X_roi.mean(axis=1)  # (epochs, times)
        else:
            # pca_flip: (epochs, times)
            tc = X_roi

        for cls, color, label in [(0, 'blue', 'Class 0'), (1, 'red', 'Class 1')]:
            mask = y == cls
            mean_tc = tc[mask].mean(axis=0)
            sem_tc = tc[mask].std(axis=0) / np.sqrt(np.sum(mask))
            ax.plot(times_ms, mean_tc, color=color, label=label)
            ax.fill_between(times_ms, mean_tc - sem_tc, mean_tc + sem_tc,
                            color=color, alpha=0.15)

        ax.axvline(0, color='k', linestyle='--', linewidth=0.8)
        ax.axvline(decode_tmin * 1000, color='gray', linestyle=':', linewidth=0.8)
        ax.set_title(roi_name, fontsize=10)
        ax.set_xlabel('Time (ms)')

    for j in range(i + 1, n_rows * n_cols):
        axes[j // n_cols, j % n_cols].set_visible(False)

    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f'{subj_id} | {task_cond} | {stim_class} | {method} | {feature_mode}\n'
                 f'Source-space ROI time courses', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = fig_dir / f'{subj_id}_{stim_class}_source_erp.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved figure: {out_path}')


def save_svm_results(results_all_rois, subj_id, task_cond, stim_class,
                     method, feature_mode, sw_dur, sw_step, atlas='aparc',
                     leakage_correction=False):
    """
    Save SVM decoding accuracy time course per ROI.
    """
    fig_dir = _make_fig_dir(task_cond, method, feature_mode, subj_id,
                            atlas=atlas, leakage_correction=leakage_correction)

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

    fig.suptitle(f'{subj_id} | {task_cond} | {stim_class} | {method} | {feature_mode}\n'
                 f'Sliding-window SVM accuracy ({sw_dur}/{sw_step} ms)',
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = fig_dir / f'{subj_id}_{stim_class}_svm_accuracy.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved figure: {out_path}')
