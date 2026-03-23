"""
Spatial leakage correction for source-estimated ROI time courses.

EEG inverse solutions (dSPM, LCMV) introduce zero-lag cross-talk between
ROIs due to the limited spatial resolution of the sensor array.

Two correction strategies are provided:

1. **Symmetric (Lowdin) orthogonalization** — for pca_flip mode.
   Removes zero-lag cross-correlations across all ROI summary signals
   simultaneously.  Reference: Colclough et al. (2015), NeuroImage.

2. **Regression-based vertex correction** — for vertex_pca / vertex_selectkbest.
   For each target ROI's vertices, regresses out the pca_flip summary
   signals of all *other* ROIs, removing cross-ROI leakage while
   preserving within-ROI spatial patterns.
   Reference: Hipp et al. (2012), Nature Neuroscience.
"""
import numpy as np
import scipy.linalg


# ── pca_flip mode: symmetric orthogonalization ──────────────────────

def symmetric_orthogonalize(data, reg=1e-10):
    """
    Apply symmetric (Lowdin) orthogonalization to one epoch.

    Removes zero-lag cross-talk between ROI time courses by computing
    D_orth = C^{-1/2} @ D  where  C = D @ D^T.

    Parameters
    ----------
    data : ndarray, shape (n_rois, n_timepoints)
        ROI time courses for a single epoch.
    reg : float
        Floor for eigenvalues to avoid division by zero.

    Returns
    -------
    data_orth : ndarray, shape (n_rois, n_timepoints)
    """
    C = data @ data.T                                  # (n_rois, n_rois)
    w, V = scipy.linalg.eigh(C)                        # eigendecomposition
    w = np.maximum(w, reg)                              # regularize
    C_inv_sqrt = V @ np.diag(1.0 / np.sqrt(w)) @ V.T   # C^{-1/2}
    return C_inv_sqrt @ data


def apply_leakage_correction(X_roi):
    """
    Apply per-epoch symmetric orthogonalization to all epochs.

    Parameters
    ----------
    X_roi : ndarray, shape (n_epochs, n_rois, n_timepoints)
        ROI time courses extracted via pca_flip mode.

    Returns
    -------
    X_corrected : ndarray, same shape as X_roi
    """
    n_epochs = X_roi.shape[0]
    X_corrected = np.empty_like(X_roi)
    for i in range(n_epochs):
        X_corrected[i] = symmetric_orthogonalize(X_roi[i])
    return X_corrected


# ── vertex modes: regression-based leakage correction ───────────────

def _regress_out_epoch(X_vertices, X_other_summaries, reg=1e-10):
    """
    Remove cross-ROI leakage from one epoch's vertex data by regression.

    For each vertex in the target ROI, regresses out the linear
    contribution of all other ROIs' summary signals (pca_flip) and
    returns the residual.

    Parameters
    ----------
    X_vertices : ndarray, shape (n_vertices, n_timepoints)
        Vertex time courses for the target ROI in one epoch.
    X_other_summaries : ndarray, shape (n_other_rois, n_timepoints)
        pca_flip summary time courses for all ROIs *except* the target.
    reg : float
        Tikhonov regularization for the covariance inversion.

    Returns
    -------
    X_clean : ndarray, shape (n_vertices, n_timepoints)
    """
    # beta = X_vertices @ X_other^T @ (X_other @ X_other^T + reg*I)^{-1}
    C = X_other_summaries @ X_other_summaries.T          # (n_other, n_other)
    C += reg * np.eye(C.shape[0])
    C_inv = scipy.linalg.inv(C)
    beta = X_vertices @ X_other_summaries.T @ C_inv      # (n_verts, n_other)
    return X_vertices - beta @ X_other_summaries


def compute_pca_summaries_from_vertices(roi_data_list, n_times):
    """
    Compute PCA-like summary signals from per-ROI vertex data.

    Used in the low-RAM pipeline where full stcs are not available
    for mne.extract_label_time_course().

    Parameters
    ----------
    roi_data_list : list of ndarray
        Each entry has shape (n_epochs, n_vertices, n_times).
    n_times : int
        Number of timepoints.

    Returns
    -------
    X_pca : ndarray, shape (n_epochs, n_rois, n_times)
    """
    n_rois = len(roi_data_list)
    n_epochs = roi_data_list[0].shape[0]
    X_pca = np.zeros((n_epochs, n_rois, n_times), dtype=np.float64)

    for roi_idx in range(n_rois):
        X_v = roi_data_list[roi_idx]  # (n_epochs, n_vertices, n_times)
        for ep in range(n_epochs):
            # First right singular vector scaled by singular value = first PC
            U, s, Vt = np.linalg.svd(X_v[ep], full_matrices=False)
            X_pca[ep, roi_idx, :] = s[0] * Vt[0]

    return X_pca


def apply_vertex_leakage_correction(roi_data, X_all_pca, roi_names):
    """
    Apply regression-based leakage correction to vertex-mode ROI data.

    For each target ROI, regresses out the pca_flip summary signals of
    all *other* ROIs from every vertex, removing cross-ROI cross-talk
    while preserving within-ROI spatial patterns.

    Parameters
    ----------
    roi_data : dict
        {roi_name: ndarray of shape (n_epochs, n_vertices, n_times)}.
        Modified **in-place**.
    X_all_pca : ndarray, shape (n_epochs, n_rois, n_times)
        pca_flip summary time courses for ALL ROIs (used as regressors).
    roi_names : list of str
        Ordered ROI names matching axis 1 of X_all_pca.

    Returns
    -------
    roi_data : dict
        Same dict, with vertex data corrected in-place.
    """
    n_epochs = X_all_pca.shape[0]
    n_rois = X_all_pca.shape[1]

    for target_idx, target_name in enumerate(roi_names):
        if target_name not in roi_data:
            continue
        X_target = roi_data[target_name]  # (n_epochs, n_vertices, n_times)

        # Indices of all OTHER ROIs
        other_idx = [j for j in range(n_rois) if j != target_idx]

        X_corrected = np.empty_like(X_target)
        for ep in range(n_epochs):
            X_others = X_all_pca[ep, other_idx, :]   # (n_other, n_times)
            X_corrected[ep] = _regress_out_epoch(
                X_target[ep], X_others
            )
        roi_data[target_name] = X_corrected

    return roi_data
