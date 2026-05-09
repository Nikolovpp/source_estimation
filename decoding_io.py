"""
Disk I/O helpers for the source-space decoding pipeline.

Holds the four functions formerly co-located inside the legacy
``run_source_svm.py`` driver:

  - ``filter_roi_dict``        — case-insensitive ROI subset selection
  - ``_load_cached_roi_data``  — read a multi-ROI .npz back into the
                                 in-memory layout used by the decoder
  - ``_save_roi_timeseries``   — persist per-subject ROI timeseries to .npz
  - ``_save_results``          — write the per-subject decoding-accuracy CSV

Extracted so active runners (``run_source_localize.py`` and
``run_decode.py``) can import them without depending on
``runners_legacy/run_source_svm.py``, which is no longer importable from
the active source tree.
"""
import sys

import numpy as np
import pandas as pd

from config import cache_feat_mode, classifier_path_segment, ROI_TIMESERIES_ROOT


def filter_roi_dict(roi_dict, roi_subset, atlas):
    """Filter roi_dict to only include requested ROI names (case-insensitive)."""
    lower_map = {k.lower(): k for k in roi_dict}
    filtered = {}
    missing = []
    for name in roi_subset:
        actual = lower_map.get(name.lower())
        if actual:
            filtered[actual] = roi_dict[actual]
        else:
            missing.append(name)
    if missing:
        print(f'ERROR: ROIs not found in {atlas} atlas: {missing}')
        print(f'Available ROIs: {sorted(roi_dict.keys())}')
        sys.exit(1)
    print(f'  ROI subset:   {list(filtered.keys())} '
          f'({len(filtered)}/{len(roi_dict)} ROIs)')
    return filtered


def _load_cached_roi_data(npz_path, feature_mode, roi_subset=None):
    """Load cached ROI timeseries from an .npz file.

    The .npz is a zip archive — ``data[name]`` only reads that array's
    bytes from the zip.  When ``roi_subset`` is provided, only those
    ROIs are decompressed (the other ROIs in the file are not touched).
    For a 7 GB / ~16-ROI cache that is ~17× less data per ROI loaded
    (matches explore_decoding._load_rois_from_cache).

    Parameters
    ----------
    npz_path : Path
    feature_mode : str
    roi_subset : iterable of str, optional
        Only materialize these ROI names (case-sensitive — must match
        the canonical names stored in ``roi_names``).  When None, every
        ROI in the file is loaded (legacy behavior).

    Returns
    -------
    roi_data, y, times, sfreq
        ``roi_data`` is a dict {roi_name: array}.  Returns
        ``(None, None, None, None)`` if any requested ROI is missing
        from the file (caller should treat as "skip subject").
    """
    data = np.load(npz_path, allow_pickle=True)
    available = set(data['roi_names'].tolist())
    if roi_subset is None:
        names_to_load = list(data['roi_names'])
    else:
        names_to_load = list(roi_subset)
        missing = [r for r in names_to_load if r not in available]
        if missing:
            data.close()
            print(f'  WARNING: ROIs missing from cache {npz_path.name}: {missing}')
            return None, None, None, None

    y = np.array(data['y'])
    times = np.array(data['times'])
    sfreq = float(data['sfreq'])
    roi_data = {}
    for name in names_to_load:
        arr = np.array(data[name])
        if feature_mode == 'pca_flip':
            # saved as (n_epochs, n_times, 1) → (n_epochs, n_times)
            if arr.ndim == 3 and arr.shape[2] == 1:
                arr = arr[:, :, 0]
        else:
            # saved as (n_epochs, n_times, n_vertices) → (n_epochs, n_vertices, n_times)
            if arr.ndim == 3:
                arr = arr.transpose(0, 2, 1)
        roi_data[name] = arr
    data.close()
    return roi_data, y, times, sfreq


def _save_roi_timeseries(subj_id, task_cond, stim_class, method,
                         feature_mode, roi_data, y, times, sfreq,
                         overwrite=False, atlas='aparc',
                         leakage_correction=False):
    """Save source-estimated ROI time series as .npz files.

    If the file already exists and *overwrite* is False, the save is
    skipped to avoid repeating this time-consuming step.

    Saved arrays use the original sensor-space pipeline's axis convention:
      - pca_flip:  (n_trials, n_timepoints, 1)    — 1 virtual sensor per ROI
      - vertex_*:  (n_trials, n_timepoints, n_vertices) — vertices as "channels"
    """
    leakage_tag = 'leakage_corrected' if leakage_correction else 'raw'
    ts_dir = (
        ROI_TIMESERIES_ROOT / task_cond / method / atlas
        / cache_feat_mode(feature_mode) / leakage_tag
    )
    ts_dir.mkdir(parents=True, exist_ok=True)

    npz_file = ts_dir / f'{subj_id}_{task_cond}_{stim_class}.npz'

    if npz_file.exists() and not overwrite:
        print(f'  ROI time series already exists, skipping: {npz_file}')
        return

    save_dict = {
        'y': y,
        'times': times,
        'sfreq': np.array(sfreq),
        'roi_names': np.array(list(roi_data.keys())),
    }

    for roi_name, X_roi in roi_data.items():
        if X_roi.ndim == 2:
            save_dict[roi_name] = X_roi[:, :, np.newaxis]
        else:
            save_dict[roi_name] = X_roi.transpose(0, 2, 1)

    np.savez_compressed(npz_file, **save_dict)
    print(f'  Saved ROI time series: {npz_file}')


def _save_results(subj_id, task_cond, stim_class, method, feature_mode,
                  sw_dur, sw_step, results_all_rois, save_dir,
                  atlas='aparc', c=1.0, leakage_correction=False,
                  pseudo_trial_size=0, classifier='svm',
                  tune_hyperparams=False):
    """Save per-subject decoding-accuracy CSV.

    Columns: key, ms, mean_list, decode_acc, best_params.
    """
    leakage_tag = 'leakage_corrected' if leakage_correction else 'raw'
    pseudo_tag = f'pseudo_{pseudo_trial_size}' if pseudo_trial_size > 0 else 'no_pseudo'
    clf_tag = classifier_path_segment(classifier, c, tune_hyperparams)
    csv_save_path = (
        save_dir / task_cond / method / atlas / feature_mode
        / leakage_tag / pseudo_tag / f'{sw_dur}_{sw_step}'
        / clf_tag / stim_class
    )
    csv_save_path.mkdir(parents=True, exist_ok=True)

    csv_file = (
        csv_save_path
        / f'{subj_id}_{task_cond}_{stim_class}_{sw_dur}_{sw_step}.csv'
    )

    if classifier == 'lda':
        best_params_str = 'lda(shrinkage=auto)'
    elif classifier == 'logistic':
        best_params_str = f'logistic(C={c}, elasticnet)'
    else:
        best_params_str = f'svm(C={c})'

    rows = []
    for roi_name, results in results_all_rois.items():
        # Sanitize ROI names: replace spaces with underscores for CSV compat
        csv_key = roi_name.replace(' ', '_')
        for r in results:
            rows.append({
                'key': csv_key,
                'ms': r['ms'],
                'mean_list': r['mean_list'],
                'decode_acc': r['decode_acc'],
                'best_params': best_params_str,
            })

    df = pd.DataFrame(
        rows,
        columns=['key', 'ms', 'mean_list', 'decode_acc', 'best_params'],
    )
    df.to_csv(csv_file, index=False)
    print(f'  Saved: {csv_file}')
