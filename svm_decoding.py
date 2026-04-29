"""
SVM decoding in source space with sliding windows.

Mirrors the existing sensor-space SVM pipeline but operates on
source-estimated vertex time courses within cortical ROIs.

Four feature-extraction strategies are supported:
  1. 'pca_flip' — single virtual sensor per ROI (extract_label_time_course)
  2. 'vertex_pca' — all vertices in ROI, PCA within sklearn pipeline
  3. 'vertex_selectkbest' — all vertices, SelectKBest(k=min(200, n_features))
  4. 'vertex_selectkbest_all' — all vertices, no feature selection (pass-through)

Output CSV format matches the existing pipeline:
  Key, ms, mean_list, SVM_acc, best_params
"""
import numpy as np
import statistics
import time
from collections import Counter

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

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


def _build_classifier_pipeline(classifier, feature_mode, n_features, svm_c):
    """Build sklearn pipeline and optional hyperparameter grid.

    Parameters
    ----------
    classifier : str
        'svm', 'lda', or 'logistic'.
    feature_mode : str
        'pca_flip', 'vertex_pca', 'vertex_selectkbest',
        or 'vertex_selectkbest_all'.
    n_features : int
        Number of input features (for SelectKBest clamping).
    svm_c : float
        Regularization parameter C (used by svm and logistic).

    Returns
    -------
    pipeline : sklearn.pipeline.Pipeline
    param_grid : dict or None
        Grid for nested CV.  None means no tunable hyperparameters.
    """
    steps = [StandardScaler()]

    if feature_mode == 'vertex_pca':
        steps.append(PCA(n_components=0.95))
    elif feature_mode == 'vertex_selectkbest':
        steps.append(SelectKBest(f_classif, k=min(200, n_features)))
    elif feature_mode == 'vertex_selectkbest_all':
        pass  # no feature selection — all vertices go to the classifier

    if classifier == 'svm':
        steps.append(LinearSVC(C=svm_c, max_iter=5000))
        param_grid = {'linearsvc__C': [0.01, 0.1, 1.0, 10.0]}
    elif classifier == 'lda':
        steps.append(LinearDiscriminantAnalysis(
            solver='lsqr', shrinkage='auto',
        ))
        param_grid = None  # shrinkage estimated analytically (Ledoit-Wolf)
    elif classifier == 'logistic':
        # l1_ratio=0.1 is fixed: empirically the modal best across subjects
        # and (sw_dur, stim_class) configs in the explore sweep was 0.1
        # for ≥17/20 subjects in every cell. Removing it from the grid
        # cuts tuned-logistic cost by 3× (13 fits/outer iter, same as svm)
        # without measurably changing accuracy.
        steps.append(LogisticRegression(
            penalty='elasticnet', solver='saga', l1_ratio=0.1,
            C=svm_c, max_iter=5000,
        ))
        param_grid = {
            'logisticregression__C': [0.01, 0.1, 1.0, 10.0],
        }
    else:
        raise ValueError(f'Unknown classifier: {classifier}')

    return make_pipeline(*steps), param_grid


def prepare_windowed_data(X_roi, sfreq, sw_dur_ms, sw_step_ms,
                          tmin, decode_tmin, times=None, verbose=True):
    """Crop baseline, build sliding-window features, and compute window-center timestamps.

    Decoupled from the CV loop so callers (e.g. explore_decoding) can
    parallelize per-window CV across many configurations using the same
    windowed data.

    Parameters
    ----------
    X_roi : np.ndarray
        (n_epochs, n_features, n_times) or (n_epochs, n_times).
    sfreq, sw_dur_ms, sw_step_ms, tmin, decode_tmin, times :
        See sliding_window_svm_decode.
    verbose : bool
        When True, print the diagnostic conversion summary.

    Returns
    -------
    X_windowed : np.ndarray, shape (n_epochs, n_features, n_windows)
    window_center_ms : np.ndarray, shape (n_windows,)
    n_features : int
    """
    if X_roi.ndim == 2:
        X_roi = X_roi[:, np.newaxis, :]

    n_epochs, n_features, n_times = X_roi.shape
    win_samples = _ms_to_samples(sw_dur_ms, sfreq)
    step_samples = _ms_to_samples(sw_step_ms, sfreq)

    if times is not None:
        times = np.asarray(times)
        crop_samples = int(np.searchsorted(times, decode_tmin))
    else:
        crop_samples = max(0, round((decode_tmin - tmin) * sfreq))

    X_roi = X_roi[:, :, crop_samples:]
    n_times_cropped = X_roi.shape[2]
    times_cropped = times[crop_samples:] if times is not None else None

    if verbose:
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

    X_windowed = np.zeros((n_epochs, n_features, n_windows))
    for ep in range(n_epochs):
        X_windowed[ep] = _apply_sliding_window_average(
            X_roi[ep], sfreq, sw_dur_ms, sw_step_ms
        )

    window_center_ms = np.empty(n_windows)
    for w in range(n_windows):
        window_start_sample = w * step_samples
        window_end_sample = window_start_sample + win_samples - 1
        if times_cropped is not None:
            window_center_ms[w] = (
                (times_cropped[window_start_sample]
                 + times_cropped[window_end_sample]) / 2.0 * 1000.0
            )
        else:
            window_center_ms[w] = (
                decode_tmin * 1000.0
                + (window_start_sample / sfreq) * 1000.0
                + sw_dur_ms / 2.0
            )

    return X_windowed, window_center_ms, n_features


def decode_one_window(X_win, y, classifier, feature_mode, n_features,
                      svm_c=1.0, tune_hyperparams=False,
                      pseudo_trial_size=0, random_state=42):
    """Run repeated stratified CV on a single already-windowed slice.

    Returns the per-window result dict (without 'ms' — the caller
    attaches the timestamp).  Used by sliding_window_svm_decode and by
    explore_decoding's per-window parallel worker.

    Parameters
    ----------
    X_win : np.ndarray, shape (n_epochs, n_features)
    y : np.ndarray
    classifier, feature_mode, n_features, svm_c, tune_hyperparams,
    pseudo_trial_size, random_state :
        See sliding_window_svm_decode.

    Returns
    -------
    entry : dict with keys 'mean_list', 'SVM_acc', and optionally
        'best_params_mode' / 'best_params_freq' when tuning is active.
    """
    pipeline, param_grid = _build_classifier_pipeline(
        classifier, feature_mode, n_features, svm_c,
    )
    if tune_hyperparams and param_grid is not None:
        clf = GridSearchCV(
            pipeline, param_grid, cv=3, scoring='accuracy', refit=True,
        )
    else:
        clf = pipeline

    track_best_params = tune_hyperparams and param_grid is not None

    mean_list = []
    best_params_list = []
    for rep in range(N_CV_REPEATS):
        seed = random_state + rep
        kf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=seed)
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

            if track_best_params:
                best_params_list.append(dict(clf.best_params_))

        mean_list.append(np.mean(fold_scores))

    entry = {
        'mean_list': mean_list,
        'SVM_acc': statistics.mean(mean_list),
    }

    if track_best_params and best_params_list:
        mode_dict, freq_dict = {}, {}
        total = len(best_params_list)
        all_keys = set().union(*best_params_list)
        for key in all_keys:
            values = [bp[key] for bp in best_params_list if key in bp]
            mode_val, mode_count = Counter(values).most_common(1)[0]
            short_name = key.split('__')[-1]
            mode_dict[short_name] = mode_val
            freq_dict[short_name] = mode_count / total
        entry['best_params_mode'] = mode_dict
        entry['best_params_freq'] = freq_dict

    return entry


def sliding_window_svm_decode(X_roi, y, sfreq, sw_dur_ms, sw_step_ms,
                              tmin, decode_tmin, feature_mode='pca_flip',
                              times=None, classifier='svm', svm_c=1.0,
                              tune_hyperparams=False,
                              pseudo_trial_size=0, random_state=42):
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
        'pca_flip', 'vertex_pca', 'vertex_selectkbest',
        or 'vertex_selectkbest_all'.
    times : np.ndarray, optional
        The exact time vector (in seconds) for the epoch, e.g. from
        epochs.times or eeg_dict['times'].  When provided, crop indices
        and window-center timestamps are derived from this vector rather
        than computed arithmetically from tmin and sfreq.
    classifier : str
        Classifier algorithm: 'svm' (LinearSVC), 'lda' (shrinkage LDA),
        or 'logistic' (elastic-net LogisticRegression).
    svm_c : float
        Regularization parameter C for SVM/logistic (default 1.0).
        Ignored when classifier='lda'.
    tune_hyperparams : bool
        When True, use nested CV (inner 3-fold GridSearchCV) to select
        the best hyperparameters within each outer training fold.
        Has no effect for classifier='lda' (no tunable hyperparams).
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
    X_windowed, window_center_ms, n_features = prepare_windowed_data(
        X_roi, sfreq, sw_dur_ms, sw_step_ms,
        tmin, decode_tmin, times=times, verbose=True,
    )
    n_windows = X_windowed.shape[2]

    results = []
    for w in range(n_windows):
        entry = decode_one_window(
            X_windowed[:, :, w], y, classifier, feature_mode, n_features,
            svm_c=svm_c, tune_hyperparams=tune_hyperparams,
            pseudo_trial_size=pseudo_trial_size, random_state=random_state,
        )
        entry['ms'] = window_center_ms[w]
        results.append(entry)

    return results
