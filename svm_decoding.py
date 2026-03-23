"""
SVM decoding in source space with sliding windows.

Mirrors the existing sensor-space SVM pipeline but operates on
source-estimated vertex time courses within cortical ROIs.

Three feature-extraction strategies are supported:
  1. 'pca_flip' — single virtual sensor per ROI (extract_label_time_course)
  2. 'vertex_pca' — all vertices in ROI, PCA within sklearn pipeline
  3. 'vertex_selectkbest' — all vertices, supervised feature selection

Output CSV format matches the existing pipeline:
  Key, ms, mean_list, SVM_acc, best_params
"""
import numpy as np
import statistics
import time

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold

import mne

from config import N_CV_FOLDS, N_CV_REPEATS
from pseudo_trials import create_pseudo_trials


def _ms_to_samples(ms, sfreq):
    """
    Convert a duration in milliseconds to the nearest integer number of samples.

    Uses round() instead of int() to avoid truncation errors.
    At fs=2048 Hz, 1 sample = 0.48828125 ms (not 0.5 ms), so durations
    like 40 ms map to 81.92 samples — round() gives 82, int() would give 81.

    Parameters
    ----------
    ms : float
        Duration in milliseconds.
    sfreq : float
        Sampling frequency in Hz.

    Returns
    -------
    n_samples : int
        Number of samples (minimum 1).
    """
    return max(1, round(ms / 1000.0 * sfreq))


def extract_roi_data_vertices(stcs, roi_label):
    """
    Extract per-vertex time courses within an ROI for all epochs.

    Returns
    -------
    X : np.ndarray, shape (n_epochs, n_vertices_in_roi, n_times)
    """
    stcs_roi = [stc.in_label(roi_label) for stc in stcs]
    return np.array([stc.data for stc in stcs_roi])


def extract_roi_data_pca_flip(stcs, roi_labels, src):
    """
    Extract PCA-flipped summary time course per ROI for all epochs.

    Returns
    -------
    X : np.ndarray, shape (n_epochs, n_rois, n_times)
    """
    label_tcs = mne.extract_label_time_course(
        stcs, roi_labels, src, mode='pca_flip',
        return_generator=False,
    )
    return np.array(label_tcs)


def _apply_sliding_window_average(data_2d, sfreq, sw_dur_ms, sw_step_ms):
    """
    Apply a sliding window average to a 2D array (features × timepoints).

    Matches the existing pipeline's behavior: averages within each window,
    steps by sw_step_ms.

    Parameters
    ----------
    data_2d : np.ndarray, shape (n_features, n_timepoints)
    sfreq : float
    sw_dur_ms : float
    sw_step_ms : float

    Returns
    -------
    windowed : np.ndarray, shape (n_features, n_windows)
    """
    win_samples = _ms_to_samples(sw_dur_ms, sfreq)
    step_samples = _ms_to_samples(sw_step_ms, sfreq)

    n_features, n_times = data_2d.shape
    n_windows = (n_times - win_samples) // step_samples + 1

    windowed = np.zeros((n_features, n_windows))
    for w in range(n_windows):
        start = w * step_samples
        end = start + win_samples
        windowed[:, w] = data_2d[:, start:end].mean(axis=1)

    return windowed


def sliding_window_svm_decode(X_roi, y, sfreq, sw_dur_ms, sw_step_ms,
                              tmin, decode_tmin, feature_mode='pca_flip',
                              times=None, svm_c=1.0, pseudo_trial_size=0,
                              random_state=42):
    """
    Run SVM decoding with a sliding window across time for one ROI.

    The baseline period (before decode_tmin) is excluded from decoding.
    This replicates the existing pipeline's approach:
    - At each time window, flatten features into a vector
    - Normalize per feature (StandardScaler)
    - Run 5-fold CV × 5 repeats
    - Record mean accuracy per time window

    Parameters
    ----------
    X_roi : np.ndarray
        For vertex modes: shape (n_epochs, n_vertices, n_times)
        For pca_flip mode: shape (n_epochs, 1, n_times) or (n_epochs, n_times)
    y : np.ndarray
        Binary class labels.
    sfreq : float
        Sampling frequency in Hz.
    sw_dur_ms : float
        Sliding window duration in ms.
    sw_step_ms : float
        Sliding window step size in ms.
    tmin : float
        Start time of the full epoch in seconds (used only as fallback
        if times is not provided).
    decode_tmin : float
        Start time for SVM decoding in seconds (baseline is excluded).
    feature_mode : str
        'pca_flip', 'vertex_pca', or 'vertex_selectkbest'.
    times : np.ndarray, optional
        The exact time vector (in seconds) for the epoch, e.g. from
        epochs.times or eeg_dict['times'].  When provided, crop indices
        and window-center timestamps are derived from this vector rather
        than computed arithmetically from tmin and sfreq.
    svm_c : float
        Regularization parameter C for LinearSVC (default 1.0).
    pseudo_trial_size : int
        Number of same-class trials to average into each pseudo-trial
        within CV training folds.  0 disables pseudo-trial averaging.
    random_state : int
        Base seed for reproducible CV splits and pseudo-trial shuffling.

    Returns
    -------
    results : list of dict
        Each dict has keys: 'ms', 'mean_list', 'SVM_acc'.
        Matches the existing pipeline output format.
    """
    if X_roi.ndim == 2:
        X_roi = X_roi[:, np.newaxis, :]  # add feature dim

    n_epochs, n_features, n_times = X_roi.shape

    # Convert ms/s parameters to samples using round()
    win_samples = _ms_to_samples(sw_dur_ms, sfreq)
    step_samples = _ms_to_samples(sw_step_ms, sfreq)

    # Crop out the baseline: keep only samples from decode_tmin onward.
    # When the actual time vector is available, use it for exact alignment;
    # otherwise fall back to arithmetic from tmin and sfreq.
    if times is not None:
        times = np.asarray(times)
        crop_samples = int(np.searchsorted(times, decode_tmin))
    else:
        crop_samples = max(0, round((decode_tmin - tmin) * sfreq))

    X_roi = X_roi[:, :, crop_samples:]
    n_times_cropped = X_roi.shape[2]

    # Build the time vector for the cropped data (in seconds)
    if times is not None:
        times_cropped = times[crop_samples:]
    else:
        times_cropped = None

    # Sanity check: log the conversion so it can be verified
    sample_dur_ms = 1000.0 / sfreq
    print(f'    sfreq={sfreq} Hz, sample duration={sample_dur_ms:.4f} ms')
    print(f'    SW window: {sw_dur_ms} ms -> {win_samples} samples '
          f'(actual {win_samples * sample_dur_ms:.2f} ms)')
    print(f'    SW step:   {sw_step_ms} ms -> {step_samples} samples '
          f'(actual {step_samples * sample_dur_ms:.2f} ms)')
    print(f'    Baseline crop: {crop_samples} samples removed '
          f'(tmin={tmin}s, decode_tmin={decode_tmin}s)')
    if times_cropped is not None:
        print(f'    Decode time range: [{times_cropped[0]*1000:.2f}, '
              f'{times_cropped[-1]*1000:.2f}] ms (from time vector)')
    print(f'    Decoding {n_times_cropped} time points, '
          f'{(n_times_cropped - win_samples) // step_samples + 1} windows')

    n_windows = (n_times_cropped - win_samples) // step_samples + 1

    # Build windowed data: (n_epochs, n_features, n_windows)
    X_windowed = np.zeros((n_epochs, n_features, n_windows))
    for ep in range(n_epochs):
        X_windowed[ep] = _apply_sliding_window_average(
            X_roi[ep], sfreq, sw_dur_ms, sw_step_ms
        )

    # Build classifier pipeline
    if feature_mode == 'pca_flip':
        clf = make_pipeline(StandardScaler(), LinearSVC(C=svm_c, max_iter=5000))
    elif feature_mode == 'vertex_pca':
        clf = make_pipeline(
            StandardScaler(),
            PCA(n_components=0.95),
            LinearSVC(C=svm_c, max_iter=5000),
        )
    elif feature_mode == 'vertex_selectkbest':
        n_select = min(200, n_features)
        clf = make_pipeline(
            StandardScaler(),
            SelectKBest(f_classif, k=n_select),
            LinearSVC(C=svm_c, max_iter=5000),
        )
    else:
        raise ValueError(f'Unknown feature_mode: {feature_mode}')

    results = []
    for w in range(n_windows):
        X_win = X_windowed[:, :, w]

        # Run repeated CV (matches existing pipeline: 5 repeats × 5 folds)
        mean_list = []
        for rep in range(N_CV_REPEATS):
            seed = random_state + rep
            kf = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=seed)
            rng = np.random.default_rng(seed)

            fold_scores = []
            for train_idx, test_idx in kf.split(X_win, y):
                X_train, y_train = X_win[train_idx], y[train_idx]
                X_test, y_test = X_win[test_idx], y[test_idx]

                if pseudo_trial_size > 0:
                    X_train, y_train = create_pseudo_trials(
                        X_train, y_train,
                        group_size=pseudo_trial_size, rng=rng,
                    )

                clf.fit(X_train, y_train)
                fold_scores.append(clf.score(X_test, y_test))

            mean_list.append(np.mean(fold_scores))

        avg_acc = statistics.mean(mean_list)

        # Time in ms for the center of this window — use actual time vector
        # when available for exact sample-aligned timestamps.
        window_start_sample = w * step_samples
        window_end_sample = window_start_sample + win_samples - 1
        if times_cropped is not None:
            window_center_ms = (
                (times_cropped[window_start_sample]
                 + times_cropped[window_end_sample]) / 2.0 * 1000.0
            )
        else:
            window_center_offset_ms = sw_dur_ms / 2.0
            decode_start_ms = decode_tmin * 1000.0
            window_center_ms = (decode_start_ms
                                + (window_start_sample / sfreq) * 1000.0
                                + window_center_offset_ms)

        results.append({
            'ms': window_center_ms,
            'mean_list': mean_list,
            'SVM_acc': avg_acc,
        })

    return results
