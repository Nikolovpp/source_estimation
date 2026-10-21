"""
Inverse-modeling pipelines for projecting sensor-space EEG epochs into
source space:

  1. minimum-norm family on ONE shared inverse operator (fixed orientation,
     depth 0.8, shrunk baseline noise covariance):
       - dSPM    (noise-normalized minimum norm; the original pipeline)
       - sLORETA (standardized LORETA; resolution-matrix-normalized)
       - eLORETA (exact LORETA; iterative weights, zero localization
                  error for single sources; MNE ignores the depth prior
                  for it by design — eLORETA compensates depth itself)
  2. LCMV (Linearly Constrained Minimum Variance) beamformer

All return the same structure, so downstream ROI extraction and decoding
code is shared.  ``run_inverse_lowram(method, ...)`` dispatches on the
method name; ``make_minimum_norm_inverse`` / ``make_lcmv_filters`` expose
the fitted operators so other tools (e.g. the resolution analysis in
exploratory/resolution_analysis.py) can use EXACTLY the pipeline's inverse.
"""
import numpy as np
import mne
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs
from mne.beamformer import make_lcmv, apply_lcmv_epochs

from config import LAMBDA2, MINIMUM_NORM_METHODS, INVERSE_METHODS

# eLORETA (like plain MNE) returns current density in A·m, ~1e-11 per vertex
# for single trials, whereas dSPM / sLORETA are unitless noise-normalized
# values of order 1.  The downstream code has small ABSOLUTE regularizers
# (leakage_correction reg 1e-6 / 1e-10, rank tolerances) and float32 caches,
# so eLORETA estimates are stored in nA·m (x 1e9) to keep them in a sane
# numeric range.  Scaling is a constant per run: decoding (StandardScaler),
# GC (scale-invariant) and leakage regression are unaffected in principle,
# and only the numerics are protected.
METHOD_OUTPUT_SCALE = {'eLORETA': 1e9}


def _scaled(stc_gen, scale):
    for stc in stc_gen:
        if scale != 1.0:
            stc.data *= scale
        yield stc


def _compute_data_rank(epochs):
    """
    Data-driven effective rank of the EEG data (for the covariance / inverse).

    The EEGLAB export is average-referenced — the recording reference is
    reconstructed as a zero channel, averaged over 65, then dropped — so the
    stored 64 channels are legitimately full rank 64.  But ``data_loader`` then
    applies MNE's average reference as a projector, which is MANDATORY for inverse
    modeling (``make_inverse_operator`` raises without it) and drops the effective
    rank to 63; EEGLAB's spherical interpolation of bad channels drops it further
    for some subjects.

    ``mne.compute_rank`` derives the true per-subject rank (accounting for both the
    average-reference projector and interpolated channels), so the covariance
    whitening / beamformer never inverts a near-null dimension.  A too-high rank
    (e.g. a hard-coded 64) makes the unit-noise-gain LCMV beamformer collapse to a
    single global time course for ill-conditioned subjects — every vertex gets the
    same signal.  See README "EEG Reference and Data Rank".
    """
    return mne.compute_rank(epochs, verbose='error')


def run_minimum_norm(epochs, fwd, baseline_tmin, baseline_tmax,
                     method='dSPM'):
    """Full-STC minimum-norm family (dSPM / sLORETA / eLORETA) on the
    pipeline's shared inverse operator.  Returns a list of SourceEstimate
    (one per epoch) — RAM-heavy; prefer run_inverse_lowram for full runs."""
    if method not in MINIMUM_NORM_METHODS:
        raise ValueError(f'method must be one of {MINIMUM_NORM_METHODS}, '
                         f'got {method!r}')
    inverse_operator, _, _ = make_minimum_norm_inverse(
        epochs, fwd, baseline_tmin, baseline_tmax)
    stcs = apply_inverse_epochs(
        epochs, inverse_operator, lambda2=LAMBDA2, method=method,
        pick_ori=None, nave=1, return_generator=False, verbose=True,
    )
    scale = METHOD_OUTPUT_SCALE.get(method, 1.0)
    if scale != 1.0:
        for stc in stcs:
            stc.data *= scale
    print(f'  {method}: {len(stcs)} source estimates, '
          f'{stcs[0].data.shape[0]} vertices, '
          f'{stcs[0].data.shape[1]} time points')
    return stcs


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
        # vertex mode: extract per-ROI vertex data.  The per-ROI arrays are
        # PREALLOCATED on the first epoch and filled in place: the earlier
        # list-of-epochs -> np.array() conversion held two copies of every
        # ROI at the end (~2x the cache size per worker, ~18 GB for a 2048 Hz
        # overtProd subject with the 5 sensorEquiv ROIs), which is what took
        # an 8-worker run over the workstation's memory.
        X_roi = {}
        for i, stc in enumerate(stc_gen):
            if stc_times is None:
                stc_times = stc.times.copy()
            for j, label in enumerate(roi_labels):
                roi_stc = stc.in_label(label)
                if j not in X_roi:
                    X_roi[j] = np.zeros(
                        (n_epochs, roi_stc.data.shape[0], roi_stc.data.shape[1]),
                        dtype=np.float32)
                X_roi[j][i] = roi_stc.data
                del roi_stc
            del stc
            if (i + 1) % 50 == 0:
                gc.collect()
        gc.collect()
        sample_shape = X_roi[0].shape
        tot_gb = sum(a.nbytes for a in X_roi.values()) / 1e9
        print(f'  Extracted vertex data: {len(X_roi)} ROIs, '
              f'shape per ROI e.g. {sample_shape}, {tot_gb:.1f} GB in RAM')

    gc.collect()
    return X_roi, stc_times


def make_minimum_norm_inverse(epochs, fwd, baseline_tmin, baseline_tmax):
    """Fit the pipeline's minimum-norm inverse operator (shared by dSPM,
    sLORETA and eLORETA): fixed orientation, depth 0.8, shrunk noise
    covariance from the pre-stimulus baseline, data-driven rank.

    Returns (inverse_operator, noise_cov, rank)."""
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
    return inverse_operator, noise_cov, rank


def make_lcmv_filters(epochs, fwd, baseline_tmin, baseline_tmax):
    """Fit the pipeline's LCMV spatial filter: empirical data covariance
    from the post-baseline window, shrunk baseline noise covariance,
    max-power orientation, unit-noise-gain, reg 0.05, data-driven rank.

    Returns (filters, data_cov, noise_cov, rank)."""
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
    return filters, data_cov, noise_cov, rank


def run_minimum_norm_lowram(epochs, fwd, baseline_tmin, baseline_tmax,
                            roi_labels, src, feature_mode='pca_flip',
                            method='dSPM'):
    """
    Low-RAM minimum-norm family (dSPM / sLORETA / eLORETA): one shared
    inverse operator, ``method`` selects the estimator at apply time.  Uses
    a generator to extract ROI data per-epoch without ever holding all
    SourceEstimates in memory.

    Returns
    -------
    X_roi : ndarray or dict
        ROI feature data (see _extract_rois_from_generator).
    stc_times : ndarray
        Time vector from the source estimates.
    """
    if method not in MINIMUM_NORM_METHODS:
        raise ValueError(f'method must be one of {MINIMUM_NORM_METHODS}, '
                         f'got {method!r}')
    inverse_operator, noise_cov, _ = make_minimum_norm_inverse(
        epochs, fwd, baseline_tmin, baseline_tmax)

    stc_gen = apply_inverse_epochs(
        epochs, inverse_operator, lambda2=LAMBDA2,
        method=method, pick_ori=None, nave=1,
        return_generator=True, verbose=True,
    )
    scale = METHOD_OUTPUT_SCALE.get(method, 1.0)
    if scale != 1.0:
        print(f'  {method}: output scaled by {scale:g} (A·m -> nA·m)')
        stc_gen = _scaled(stc_gen, scale)

    n_epochs = len(epochs)
    print(f'  {method} (low-RAM): processing {n_epochs} epochs via generator...')

    X_roi, stc_times = _extract_rois_from_generator(
        stc_gen, n_epochs, roi_labels, src, feature_mode
    )

    # Free the inverse operator (large object)
    del inverse_operator, noise_cov
    import gc
    gc.collect()

    return X_roi, stc_times


def run_dspm_lowram(epochs, fwd, baseline_tmin, baseline_tmax,
                    roi_labels, src, feature_mode='pca_flip'):
    """Low-RAM dSPM (kept for backward compatibility; see
    run_minimum_norm_lowram)."""
    return run_minimum_norm_lowram(epochs, fwd, baseline_tmin, baseline_tmax,
                                   roi_labels, src, feature_mode, 'dSPM')


def run_sloreta_lowram(epochs, fwd, baseline_tmin, baseline_tmax,
                       roi_labels, src, feature_mode='pca_flip'):
    """Low-RAM sLORETA on the pipeline's minimum-norm inverse operator."""
    return run_minimum_norm_lowram(epochs, fwd, baseline_tmin, baseline_tmax,
                                   roi_labels, src, feature_mode, 'sLORETA')


def run_eloreta_lowram(epochs, fwd, baseline_tmin, baseline_tmax,
                       roi_labels, src, feature_mode='pca_flip'):
    """Low-RAM eLORETA on the pipeline's minimum-norm inverse operator."""
    return run_minimum_norm_lowram(epochs, fwd, baseline_tmin, baseline_tmax,
                                   roi_labels, src, feature_mode, 'eLORETA')


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
    filters, data_cov, noise_cov, _ = make_lcmv_filters(
        epochs, fwd, baseline_tmin, baseline_tmax)

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


def run_inverse_lowram(method, epochs, fwd, baseline_tmin, baseline_tmax,
                       roi_labels, src, feature_mode='pca_flip'):
    """Dispatch on the inverse method name (config.INVERSE_METHODS)."""
    if method in MINIMUM_NORM_METHODS:
        return run_minimum_norm_lowram(
            epochs, fwd, baseline_tmin, baseline_tmax,
            roi_labels, src, feature_mode, method)
    if method == 'LCMV':
        return run_lcmv_lowram(
            epochs, fwd, baseline_tmin, baseline_tmax,
            roi_labels, src, feature_mode)
    raise ValueError(f'Unknown inverse method {method!r}; '
                     f'choose from {INVERSE_METHODS}')
