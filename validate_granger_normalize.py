#!/usr/bin/env python
"""Regression test: --normalize must remove the ERP, not average the ROIs.

THE BUG (commit ff2cdd6, 2026-07-29). In ``compute_subject_gc`` the stacked
virtual channels are

    V = np.stack(vcs, axis=0)          ->  (n_roi, n_epochs, n_times)

as ``V[i]`` in the pairwise branch and ``np.transpose(V, (1, 0, 2))`` in the
conditional branch both confirm.  ff2cdd6 rewrote the normalization block on
the belief that V is ``(n_trials, n_chan, n_times)`` and switched from
``V[r]`` (correct: one ROI at a time, axis-0 mean = mean over TRIALS = the
ERP) to ``V[:, c, :]`` (one EPOCH at a time, axis-0 mean = mean over ROIs).
So ``--normalize demean`` stopped removing the ERP and started subtracting a
common-average reference across the very ROIs whose directed coupling is
being measured — the exact failure mode the comment it added warns about.

WHY IT WIPES OUT PAIRWISE GC. For a 2-ROI subset the cross-ROI mean is
(a+b)/2, so the two channels become a-(a+b)/2 = (a-b)/2 and b-(a+b)/2 =
-(a-b)/2. They are the SAME SIGNAL NEGATED: the ensemble is exactly rank 1,
every covariance in the Morf recursion is singular, and ``cholesky(pb)``
raises. The NaN guard then turns each window into NaN. That is the 96-99%
``[unstable]`` rate in logs/gc_window_order_sweep/*_pairwise_*.log, flat
across every window length and model order because it has nothing to do with
sample count.

3-ROI subsets are hit just as hard: the cross-ROI mean costs one rank of
three, and the on-disk triple-wise sweep output is 93% NaN. Its surviving 7%
is not a usable remainder — on the synthetic below the correct edge drops
from 0.68 to 0.003 and the strongest edge is no longer the true one. Every
``_demean`` directory written after 2026-07-29 has to be recomputed.

    conda activate mne
    python validate_granger_normalize.py
"""
import sys
import numpy as np

from granger import moving_window_pairwise_gc, band_average
from granger_statespace import moving_window_conditional_gc
from run_granger import compute_subject_gc

FS = 200.0
N_TRIALS = 60
N_TIMES = 120
ORDER = 10
WIN_MS = 80.0
FREQS = np.arange(1.0, 31.0)


def simulate(n_roi=2, seed=0):
    """VAR(2) ensemble with a known ROI0 -> ROI1 drive at lag 1.

    Distinct per-ROI ERPs so that ERP removal has something real to remove
    and the two candidate normalizations cannot coincide by accident.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(N_TIMES) / FS
    erp = np.stack([(k + 1) * np.exp(-((t - 0.25) ** 2) / 0.002)
                    for k in range(n_roi)])            # (n_roi, n_times)

    X = np.zeros((n_roi, N_TRIALS, N_TIMES))
    e = rng.standard_normal((n_roi, N_TRIALS, N_TIMES))
    for s in range(2, N_TIMES):
        X[:, :, s] = 0.55 * X[:, :, s - 1] - 0.35 * X[:, :, s - 2] + e[:, :, s]
        X[1, :, s] += 0.60 * X[0, :, s - 1]            # ROI0 -> ROI1
    return X + erp[:, None, :]


def per_roi(V):
    """The correct normalization: axis-0 mean of V[r] is the mean over TRIALS."""
    return np.stack([V[r] - V[r].mean(axis=0, keepdims=True)
                     for r in range(V.shape[0])], axis=0)


def per_epoch(V):
    """What ff2cdd6 does: axis-0 mean of V[:, c, :] is the mean over ROIs."""
    return np.stack([V[:, c, :] - V[:, c, :].mean(axis=0, keepdims=True)
                     for c in range(V.shape[1])], axis=1)


def rank_report(name, V):
    """Channel rank and correlation of the ensemble, pooled over trials."""
    flat = V.reshape(V.shape[0], -1)                   # (n_roi, n_ep*n_t)
    r = np.linalg.matrix_rank(flat @ flat.T)
    c = np.corrcoef(flat)[0, 1]
    print(f'    {name:<28} channel rank {r}/{V.shape[0]}   corr(ch0,ch1) {c:+.6f}')
    return r


def window_gc(V):
    """Pairwise GC on the first two channels -> (n_unstable, n_windows)."""
    X = np.stack([V[0], V[1]], axis=1)                 # (n_ep, 2, n_t)
    res = moving_window_pairwise_gc(
        X, order=ORDER, freqs=FREQS, fs=FS,
        win_samples=int(round(WIN_MS / 1000.0 * FS)), step=1)
    return res['n_unstable'], res['f_xy'].shape[1], res['f_xy']


def main():
    ok = True
    V = simulate(n_roi=2)
    print(f'synthetic VAR(2), {V.shape[0]} ROIs x {N_TRIALS} trials x '
          f'{N_TIMES} samples @ {FS:g} Hz, known ROI0 -> ROI1\n')

    # ── 1. The two normalizations, at the level of the array itself ──────
    print('1. What each candidate does to the ensemble')
    rank_report('raw', V)
    r_roi = rank_report('per-ROI (correct)', per_roi(V))
    r_ep = rank_report('per-epoch (ff2cdd6)', per_epoch(V))
    if r_ep != 1:
        print('    FAIL: expected the per-epoch form to collapse to rank 1')
        ok = False
    if r_roi != 2:
        print('    FAIL: per-ROI ERP removal should preserve rank 2')
        ok = False
    print()

    # ── 2. What that does to the GC fit ─────────────────────────────────
    print(f'2. Pairwise GC, order {ORDER}, {WIN_MS:g} ms window')
    for name, W in (('raw', V), ('per-ROI (correct)', per_roi(V)),
                    ('per-epoch (ff2cdd6)', per_epoch(V))):
        n_bad, n_win, fxy = window_gc(W)
        med = np.nanmedian(fxy) if np.isfinite(fxy).any() else np.nan
        print(f'    {name:<28} {n_bad:>3}/{n_win} windows unstable   '
              f'median GC {med:.4f}')
        if name.startswith('per-epoch') and n_bad != n_win:
            print('    FAIL: expected the per-epoch form to lose every window')
            ok = False
        if name.startswith('per-ROI') and n_bad:
            print('    FAIL: correct normalization should fit every window')
            ok = False
    print()

    # ── 3. The regression guard: the runner as the sweep calls it ───────
    # This is the part that must keep passing. Sections 1-2 build the buggy
    # array by hand; here compute_subject_gc chooses for itself, so a
    # re-introduction of the axis swap shows up as a wall of NaN.
    print('3. compute_subject_gc, as run_gc_window_order_sweep.sh calls it')
    times = np.arange(N_TIMES) / FS
    roi_data = {'roiA': V[0], 'roiB': V[1]}
    for mode in ('none', 'demean'):
        res = compute_subject_gc(
            roi_data, times, FS, order=ORDER, win_ms=WIN_MS, target_fs=FS,
            freqs=FREQS, normalize=mode, gc_mode='pairwise', n_jobs=1)
        v = res['fxy']['high_beta']
        frac = 100.0 * np.isnan(v).mean()
        print(f'    --normalize {mode:<8} {frac:5.1f}% NaN   '
              f'median GC {np.nanmedian(v):.4f}')
        if frac > 5.0:
            print(f'    FAIL: --normalize {mode} is losing windows. If this is '
                  f'demean, the ff2cdd6 axis swap is back.')
            ok = False
    print()

    # ── 4. The 3-ROI arm is hit too ─────────────────────────────────────
    print('4. 3-ROI subset — same bug, same damage')
    V3 = simulate(n_roi=3, seed=1)
    rank_report('raw', V3)
    r3 = np.linalg.matrix_rank(
        per_epoch(V3).reshape(3, -1) @ per_epoch(V3).reshape(3, -1).T)
    print(f'    {"per-epoch (ff2cdd6)":<28} channel rank {r3}/3'
          f'   -> one rank lost to the cross-ROI average')
    if r3 != 2:
        print('    FAIL: expected rank 2')
        ok = False

    pairs = [(i, j) for i in range(3) for j in range(3) if i != j]
    got = {}
    for name, W in (('per-ROI (correct)', per_roi(V3)),
                    ('per-epoch (ff2cdd6)', per_epoch(V3))):
        r = moving_window_conditional_gc(
            np.transpose(W, (1, 0, 2)), order=4, freqs=FREQS, fs=FS,
            win_samples=int(round(WIN_MS / 1000.0 * FS)), step=1,
            pairs=pairs, n_jobs=1)
        got[name] = {p: np.nanmean(band_average(r['gc'][p], FREQS)['high_beta'])
                     for p in pairs}
        top = max(got[name], key=lambda p: got[name][p])
        print(f'    {name:<28} {r["n_unstable"]:>3}/{len(r["win_start"])} '
              f'unstable   strongest edge {top}  '
              f'(true edge is (0, 1))')
        if name.startswith('per-epoch') and top == (0, 1):
            print('    FAIL: expected the true edge to be lost')
            ok = False
    print()

    print('PASS — bug reproduced and localized' if ok else
          'FAIL — see the notes above')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
