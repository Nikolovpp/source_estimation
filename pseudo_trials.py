"""
Pseudo-trial averaging for SNR improvement in SVM decoding.

Groups same-class trials and averages them into pseudo-trials, reducing
single-trial noise while preserving time-locked evoked/induced signals.
Applied only to training data within each CV fold to avoid data leakage.
"""
import numpy as np


def create_pseudo_trials(X, y, group_size=5, rng=None):
    """
    Create pseudo-trials by averaging groups of same-class trials.

    Parameters
    ----------
    X : ndarray, shape (n_trials, n_features)
        Training data for one time window.
    y : ndarray, shape (n_trials,)
        Binary class labels (0 or 1).
    group_size : int
        Number of trials to average per pseudo-trial.
    rng : numpy.random.Generator or None
        Random number generator for shuffling trial order.
        If None, a new default generator is created.

    Returns
    -------
    X_pseudo : ndarray, shape (n_pseudo_trials, n_features)
    y_pseudo : ndarray, shape (n_pseudo_trials,)
    """
    if rng is None:
        rng = np.random.default_rng()

    classes = np.unique(y)
    X_parts = []
    y_parts = []

    for cls in classes:
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)

        # Form complete groups, discard remainder
        n_groups = len(idx) // group_size
        if n_groups == 0:
            continue

        idx_trimmed = idx[:n_groups * group_size]
        groups = idx_trimmed.reshape(n_groups, group_size)

        # Average trials within each group
        for group in groups:
            X_parts.append(X[group].mean(axis=0))
            y_parts.append(cls)

    X_pseudo = np.array(X_parts)
    y_pseudo = np.array(y_parts)
    return X_pseudo, y_pseudo
