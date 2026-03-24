"""
Two inverse-modeling pipelines for projecting sensor-space EEG epochs
into source space:

  1. dSPM (dynamic Statistical Parametric Mapping) — minimum-norm family
  2. LCMV (Linearly Constrained Minimum Variance) beamformer

Both return lists of SourceEstimate objects with identical structure,
so downstream ROI extraction and SVM decoding code is shared.
"""
import numpy as np
import mne
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs
from mne.beamformer import make_lcmv, apply_lcmv_epochs

from config import LAMBDA2


def _compute_data_rank(epochs):
    """
    Estimate the effective rank of the EEG data.

    The EEGLAB preprocessing removes all extra channels (EOG, EMG, reference)
    before exporting, so the data arrives as exactly 64 EEG channels at
    full rank. The average reference applied in data_loader.py is a
    projection (not a physical re-referencing done during preprocessing),
    so it does not reduce the rank of the stored data.
    """
    n_channels = len(epochs.ch_names)
    return dict(eeg=n_channels)


def run_dspm(epochs, fwd, baseline_tmin, baseline_tmax):
    """
    Pipeline 1: Compute dSPM source estimates for all epochs.

    Uses fixed orientation (normal to cortex), shrinkage noise covariance
    estimated from the pre-stimulus baseline, and dSPM noise normalization.

    Parameters
    ----------
    epochs : mne.Epochs
        EEG epochs with average reference applied.
    fwd : mne.Forward
        Forward solution.
    baseline_tmin : float
        Start of baseline window in seconds (for noise covariance).
    baseline_tmax : float
        End of baseline window in seconds (for noise covariance).

    Returns
    -------
    stcs : list of mne.SourceEstimate
        One source estimate per epoch.
    """
    rank = _compute_data_rank(epochs)
    print(f'  Data rank: {rank}')
    print(f'  Noise cov baseline: [{baseline_tmin}, {baseline_tmax}] s')

    # Noise covariance from pre-stimulus baseline
    noise_cov = mne.compute_covariance(
        epochs, tmin=baseline_tmin, tmax=baseline_tmax,
        method='shrunk',
        rank=rank,
        verbose=True,
    )

    # Inverse operator with fixed orientation
    inverse_operator = make_inverse_operator(
        epochs.info,
        fwd,
        noise_cov,
        loose=0.0,
        depth=0.8,
        fixed=True,
        rank=rank,
        verbose=True,
    )

    # Apply inverse to each epoch
    stcs = apply_inverse_epochs(
        epochs,
        inverse_operator,
        lambda2=LAMBDA2,
        method='dSPM',
        pick_ori=None,
        nave=1,
        return_generator=False,
        verbose=True,
    )

    print(f'  dSPM: {len(stcs)} source estimates, '
          f'{stcs[0].data.shape[0]} vertices, '
          f'{stcs[0].data.shape[1]} time points')
    return stcs


def run_lcmv(epochs, fwd, baseline_tmin, baseline_tmax):
    """
    Pipeline 2: Compute LCMV beamformer source estimates for all epochs.

    Uses data covariance from the post-baseline window, noise covariance
    from the pre-stimulus baseline, max-power orientation, and
    unit-noise-gain normalization.

    Parameters
    ----------
    epochs : mne.Epochs
        EEG epochs with average reference applied.
    fwd : mne.Forward
        Forward solution.
    baseline_tmin : float
        Start of baseline window in seconds (for noise covariance).
    baseline_tmax : float
        End of baseline window in seconds (for noise covariance).

    Returns
    -------
    stcs : list of mne.SourceEstimate
        One source estimate per epoch.
    """
    rank = _compute_data_rank(epochs)
    print(f'  Data rank: {rank}')
    print(f'  Noise cov baseline: [{baseline_tmin}, {baseline_tmax}] s')

    # Data covariance from the post-baseline window
    data_cov = mne.compute_covariance(
        epochs, tmin=baseline_tmax, tmax=None,
        method='empirical',
        rank=rank,
        verbose=True,
    )

    # Noise covariance from baseline
    noise_cov = mne.compute_covariance(
        epochs, tmin=baseline_tmin, tmax=baseline_tmax,
        method='shrunk',
        rank=rank,
        verbose=True,
    )

    # Build spatial filter
    filters = make_lcmv(
        epochs.info,
        fwd,
        data_cov,
        reg=0.05,
        noise_cov=noise_cov,
        pick_ori='max-power',
        weight_norm='unit-noise-gain',
        reduce_rank=True,
        rank=rank,
        verbose=True,
    )

    # Apply to epochs
    stcs = apply_lcmv_epochs(
        epochs, filters,
        return_generator=False,
        verbose=True,
    )

    print(f'  LCMV: {len(stcs)} source estimates, '
          f'{stcs[0].data.shape[0]} vertices, '
          f'{stcs[0].data.shape[1]} time points')
    return stcs


# ─────────────────────────────────────────────────────────────────────
# Low-RAM variants: return ROI data directly via generators
# ─────────────────────────────────────────────────────────────────────

def _extract_rois_from_generator(stc_gen, n_epochs, roi_labels, src,
                                 feature_mode='pca_flip'):
    """
    Iterate a SourceEstimate generator, extract ROI data per-epoch,
    and discard each full STC immediately to keep RAM low.

    Parameters
    ----------
    stc_gen : generator of mne.SourceEstimate
    n_epochs : int
    roi_labels : list of mne.Label
    src : mne.SourceSpaces
    feature_mode : str
        'pca_flip' → returns (n_epochs, n_rois, n_times)
        'vertex'   → returns dict {roi_index: (n_epochs, n_verts, n_times)}

    Returns
    -------
    X_roi : ndarray or dict of ndarrays
    stc_times : ndarray
        The time vector from the source estimates.
    """
    import gc

    X_roi = None
    stc_times = None

    if feature_mode == 'pca_flip':
        for i, stc in enumerate(stc_gen):
            if stc_times is None:
                stc_times = stc.times.copy()
            tc = mne.extract_label_time_course(
                [stc], roi_labels, src, mode='pca_flip',
                return_generator=False,
            )
            # tc is a list of one array with shape (n_rois, n_times)
            tc_arr = np.asarray(tc[0])
            if X_roi is None:
                n_rois, n_times = tc_arr.shape
                X_roi = np.zeros((n_epochs, n_rois, n_times), dtype=np.float32)
            X_roi[i] = tc_arr
            del stc, tc, tc_arr
            if (i + 1) % 50 == 0:
                gc.collect()
        print(f'  Extracted pca_flip: {X_roi.shape}')
    else:
        # vertex mode: extract per-ROI vertex data
        roi_data = {j: [] for j in range(len(roi_labels))}
        for i, stc in enumerate(stc_gen):
            if stc_times is None:
                stc_times = stc.times.copy()
            for j, label in enumerate(roi_labels):
                roi_stc = stc.in_label(label)
                roi_data[j].append(roi_stc.data.astype(np.float32))
                del roi_stc
            del stc
            if (i + 1) % 50 == 0:
                gc.collect()
        X_roi = {j: np.array(v) for j, v in roi_data.items()}
        del roi_data
        gc.collect()
        sample_shape = X_roi[0].shape
        print(f'  Extracted vertex data: {len(X_roi)} ROIs, '
              f'shape per ROI e.g. {sample_shape}')

    gc.collect()
    return X_roi, stc_times


def run_dspm_lowram(epochs, fwd, baseline_tmin, baseline_tmax,
                    roi_labels, src, feature_mode='pca_flip'):
    """
    Low-RAM dSPM: uses a generator to extract ROI data per-epoch
    without ever holding all SourceEstimates in memory.

    Returns
    -------
    X_roi : ndarray or dict
        ROI feature data (see _extract_rois_from_generator).
    stc_times : ndarray
        Time vector from the source estimates.
    """
    rank = _compute_data_rank(epochs)
    print(f'  Data rank: {rank}')
    print(f'  Noise cov baseline: [{baseline_tmin}, {baseline_tmax}] s')

    noise_cov = mne.compute_covariance(
        epochs, tmin=baseline_tmin, tmax=baseline_tmax,
        method='shrunk', rank=rank, verbose=True,
    )

    inverse_operator = make_inverse_operator(
        epochs.info, fwd, noise_cov,
        loose=0.0, depth=0.8, fixed=True, rank=rank, verbose=True,
    )

    stc_gen = apply_inverse_epochs(
        epochs, inverse_operator, lambda2=LAMBDA2,
        method='dSPM', pick_ori=None, nave=1,
        return_generator=True, verbose=True,
    )

    n_epochs = len(epochs)
    print(f'  dSPM (low-RAM): processing {n_epochs} epochs via generator...')

    X_roi, stc_times = _extract_rois_from_generator(
        stc_gen, n_epochs, roi_labels, src, feature_mode
    )

    # Free the inverse operator (large object)
    del inverse_operator, noise_cov
    import gc
    gc.collect()

    return X_roi, stc_times


def run_lcmv_lowram(epochs, fwd, baseline_tmin, baseline_tmax,
                    roi_labels, src, feature_mode='pca_flip'):
    """
    Low-RAM LCMV: uses a generator to extract ROI data per-epoch
    without ever holding all SourceEstimates in memory.

    Returns
    -------
    X_roi : ndarray or dict
        ROI feature data (see _extract_rois_from_generator).
    stc_times : ndarray
        Time vector from the source estimates.
    """
    rank = _compute_data_rank(epochs)
    print(f'  Data rank: {rank}')
    print(f'  Noise cov baseline: [{baseline_tmin}, {baseline_tmax}] s')

    data_cov = mne.compute_covariance(
        epochs, tmin=baseline_tmax, tmax=None,
        method='empirical', rank=rank, verbose=True,
    )

    noise_cov = mne.compute_covariance(
        epochs, tmin=baseline_tmin, tmax=baseline_tmax,
        method='shrunk', rank=rank, verbose=True,
    )

    filters = make_lcmv(
        epochs.info, fwd, data_cov, reg=0.05,
        noise_cov=noise_cov, pick_ori='max-power',
        weight_norm='unit-noise-gain', reduce_rank=True,
        rank=rank, verbose=True,
    )

    stc_gen = apply_lcmv_epochs(
        epochs, filters,
        return_generator=True, verbose=True,
    )

    n_epochs = len(epochs)
    print(f'  LCMV (low-RAM): processing {n_epochs} epochs via generator...')

    X_roi, stc_times = _extract_rois_from_generator(
        stc_gen, n_epochs, roi_labels, src, feature_mode
    )

    del filters, data_cov, noise_cov
    import gc
    gc.collect()

    return X_roi, stc_times
