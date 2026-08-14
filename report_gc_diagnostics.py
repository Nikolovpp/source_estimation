#!/usr/bin/env python
"""Trial counts and MVAR model validation for this project's GC data.

Answers three questions the manuscript needs, for BOTH task/contrast pairs:

1. **What is N?**  The real number of epochs per subject, read from the vertex
   caches on the shared drive (``config.ROI_TIMESERIES*`` / ``config.env``),
   not assumed. Every subject is listed, so a subject with an unusually thin
   ensemble is visible rather than buried in a mean.

2. **How big is the ensemble-SD jitter?**  ``--normalize zscore`` (Ding,
   Bressler, Yang & Liang 2000, preprocessing step 3) divides each trial by the
   ensemble SD at each time point. That divisor is itself estimated from N
   epochs, with relative standard error ~1/sqrt(2N). Ding et al. had 888 trials
   and could ignore it; if N here is ~90 the divisor jitters ~7.5%, and because
   that jitter is independent across time points it WHITENS the signal —
   flattening the spectrum, weakening estimated autocorrelation, and deflating
   GC. This number decides whether step 3 is safe to adopt.

3. **Are the fitted models valid?**  Per-window companion spectral radius and
   MVGC's consistency statistic across the sweep's window x order grid.
   ``rho >= 1`` means the fit is non-minimum-phase: numerically successful,
   spectrally meaningless. Nothing else in the pipeline detects it.

READING THE TWO DIAGNOSTICS. They are not interchangeable.
  rho  — a hard validity gate. >= 1 invalidates that window outright.
  cons — a soft measure, and NOT "model adequacy" in an absolute sense: it
         reduces to 1 - ||Sigma||/||Rr||, i.e. the fraction of the process's
         own variance predictable from its past. A correct model on a noisy
         process scores ~55%, so MVGC's ">0.8 is reasonable" is a rule of
         thumb for strongly autocorrelated data, not a threshold to apply
         here. Use it to COMPARE arms (preprocessing, order), not to pass or
         fail one in isolation.

    conda activate mne
    python report_gc_diagnostics.py                    # N + jitter only (fast)
    python report_gc_diagnostics.py --grid             # + the window x order sweep
    python report_gc_diagnostics.py --grid --n-subjects 5
"""
import sys
import argparse
import numpy as np

import config
from config import find_cached_npz

TASKS = [('perception', 'percDiff'), ('overtProd', 'prodDiff')]
WINDOWS = [40.0, 60.0, 80.0]
ORDERS = [2, 4, 6, 10]
PAIR = ['awfa-lh', 'ifc-lh']          # the pathway carrying the headline effect


def epoch_counts(task, stim, method, atlas, feature_mode, leakage):
    """-> [(subject, n_epochs or None)] read from the vertex caches.

    Reads ONE ROI array's leading dimension. npz stores arrays separately, so
    this decompresses a single ROI rather than the whole ~7 GB cache.
    """
    out = []
    for subj in config.SUBJECT_IDS:
        npz = find_cached_npz(task, method, atlas, feature_mode, leakage,
                              subj, stim)
        if npz is None:
            out.append((subj, None, None))
            continue
        try:
            with np.load(npz, allow_pickle=True) as z:
                names = [str(r) for r in z['roi_names']]
                key = next((k for k in z.files if k in names), None)
                if key is None:
                    key = next((k for k in z.files
                                if getattr(z[k], 'ndim', 0) == 3), None)
                n = int(z[key].shape[0]) if key else None
        except Exception as e:
            print(f'    {subj}: could not read cache ({type(e).__name__})')
            n = None
        out.append((subj, n, npz))
    return out


def report_counts(args):
    """Section 1+2: N and the ensemble-SD jitter, per task/contrast."""
    summary = {}
    for task, stim in TASKS:
        print(f'\n{"="*72}\n{task} / {stim}\n{"="*72}')
        rows = epoch_counts(task, stim, args.method, args.atlas,
                            args.feature_mode, args.leakage_correction)
        found = [(s, n) for s, n, _ in rows if n]
        if rows and rows[0][2] is not None:
            print(f'cache: {rows[0][2].parent}')
        if not found:
            print('  NO VERTEX CACHES FOUND. This report must run where the '
                  'caches live (the shared drive on the workstation); check '
                  'EEG_PROJECT_ROOT / ROI_TIMESERIES paths in config.env.')
            continue
        for s, n in found:
            print(f'    {s}  N = {n:>4} epochs   '
                  f'ensemble-SD jitter {100/np.sqrt(2*n):5.2f}%')
        missing = [s for s, n, _ in rows if not n]
        if missing:
            print(f'    no cache for: {", ".join(missing)}')

        N = np.array([n for _, n in found], float)
        print(f'\n  N over {len(N)} subjects: mean {N.mean():.0f}, '
              f'median {np.median(N):.0f}, range {N.min():.0f}-{N.max():.0f}')
        print(f'  ensemble-SD jitter 1/sqrt(2N): {100/np.sqrt(2*N.mean()):.2f}% '
              f'at the mean, {100/np.sqrt(2*N.min()):.2f}% at the thinnest '
              f'subject')
        print(f'  for comparison, Ding et al. (2000) had 888 trials -> '
              f'{100/np.sqrt(2*888):.2f}%')
        ratio = np.sqrt(888 / N.mean())
        print(f'  => the divisor --normalize zscore would apply is {ratio:.1f}x '
              f'noisier here than in the study that recommends it')
        summary[(task, stim)] = N
    return summary


def report_grid(args):
    """Section 3: rho and consistency over the sweep's window x order grid.

    Runs the REAL pipeline (compute_subject_gc) on the production frequency
    grid, so these are the numbers the sweep produces rather than an
    approximation of them.

    An earlier version cut freqs to one bin on the theory that the diagnostics
    come from the MVAR fit and so do not need the spectrum. True, but useless:
    measured, the fit dominates and the reduced grid saved 1% (1.26s vs 1.25s)
    while leaving theta with no bin, which band_average correctly rejected.
    """
    from decoding_io import _load_cached_roi_data
    from run_granger import compute_subject_gc

    for task, stim in TASKS:
        print(f'\n{"="*72}\n{task} / {stim}   ROIs {" + ".join(PAIR)}\n{"="*72}')
        subs = []
        for subj in config.SUBJECT_IDS:
            npz = find_cached_npz(task, args.method, args.atlas,
                                  args.feature_mode, args.leakage_correction,
                                  subj, stim)
            if npz is not None:
                subs.append((subj, npz))
            if len(subs) >= args.n_subjects:
                break
        if not subs:
            print('  no vertex caches found — skipping')
            continue
        print(f'  {len(subs)} subject(s): {", ".join(s for s, _ in subs)}\n')
        print(f'  {"win":>5} {"samp":>5} {"order":>6} {"rho med":>9} '
              f'{"rho p95":>9} {"rho>=1":>9} {"cons med":>9}')

        for win_ms in WINDOWS:
            m = max(2, round(win_ms / 1000.0 * args.target_fs))
            for order in ORDERS:
                if m <= order + 1:
                    print(f'  {win_ms:>5.0f} {m:>5} {order:>6} '
                          f'{"infeasible (order >= samples)":>39}')
                    continue
                rho, cons = [], []
                for subj, npz in subs:
                    roi_data, _, times, sfreq = _load_cached_roi_data(
                        npz, feature_mode=args.feature_mode, roi_subset=PAIR)
                    if roi_data is None or len(roi_data) < 2:
                        continue
                    try:
                        r = compute_subject_gc(
                            roi_data, times, sfreq, order=order,
                            win_ms=win_ms, target_fs=args.target_fs,
                            normalize=args.normalize,
                            gc_mode='pairwise', n_jobs=1, diagnostics=True)
                    except Exception as e:
                        print(f'    {subj}: {type(e).__name__}: {str(e)[:60]}')
                        continue
                    rho.append(np.asarray(r['rho'], float).ravel())
                    cons.append(np.asarray(r['consistency'], float).ravel())
                if not rho:
                    continue
                R = np.concatenate(rho)
                C = np.concatenate(cons)
                n_fin = int(np.isfinite(R).sum())
                n_bad = int((R >= 1.0).sum())
                print(f'  {win_ms:>5.0f} {m:>5} {order:>6} '
                      f'{np.nanmedian(R):9.3f} {np.nanpercentile(R, 95):9.3f} '
                      f'{n_bad:>4}/{n_fin:<4} {100*np.nanmedian(C):8.1f}%')
        print('\n  rho rising with order is the signature to watch: the fit '
              'moves toward the\n  stability boundary, which inflates spectral '
              'GC independently of any coupling.')


def self_test():
    """Check the two diagnostics against VARs whose answers are known."""
    from granger import (fit_mvar, var_spectral_radius, var_consistency,
                         var_residuals)
    ok = True

    A = np.zeros((2, 2, 1)); A[:, :, 0] = [[0.5, 0.2], [0.0, -0.9]]
    got, want = var_spectral_radius(A), 0.9
    print(f'  VAR(1) rho              {got:.6f}  want {want:.6f}')
    ok &= abs(got - want) < 1e-10

    B = np.zeros((2, 2, 1)); B[:, :, 0] = [[1.2, 0.0], [0.0, 0.4]]
    got = var_spectral_radius(B)
    print(f'  explosive VAR rho       {got:.6f}  want >= 1')
    ok &= got >= 1.0

    r = np.random.default_rng(3)
    T, R = 400, 40
    X = np.zeros((R, 2, T))
    A1 = np.array([[0.55, 0.0], [0.6, 0.5]])
    A2 = np.array([[-0.35, 0.0], [0.0, -0.3]])
    for t in range(2, T):
        X[:, :, t] = (X[:, :, t-1] @ A1.T + X[:, :, t-2] @ A2.T
                      + r.standard_normal((R, 2)))
    Ah, Sh = fit_mvar(X, 2)

    # Residuals must reproduce the fitted innovation covariance.
    E = var_residuals(X - X.mean(axis=2, keepdims=True), Ah)
    Ec = E.transpose(1, 0, 2).reshape(2, -1)
    Se = (Ec @ Ec.T) / (Ec.shape[1] - 1)
    rel = np.abs(Se - Sh).max() / np.abs(Sh).max()
    print(f'  residual cov vs Sigma   {rel:.4f} relative  want < 0.05')
    ok &= rel < 0.05

    # Consistency must rank correct order above under-fit above white noise.
    c2 = var_consistency(X, Ah)
    c1 = var_consistency(X, fit_mvar(X, 1)[0])
    W = r.standard_normal((R, 2, T))
    cw = var_consistency(W, fit_mvar(W, 2)[0])
    print(f'  consistency VAR2 {100*c2:.1f}% > VAR1 {100*c1:.1f}% > '
          f'noise {100*cw:.1f}%')
    ok &= c2 > c1 > cw

    print('\n  ' + ('PASS' if ok else 'FAIL'))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--method', default='LCMV')
    ap.add_argument('--atlas', default='custom')
    ap.add_argument('--feature-mode', default='vertex_selectkbest')
    ap.add_argument('--leakage-correction', action='store_true', default=True)
    ap.add_argument('--no-leakage-correction', dest='leakage_correction',
                    action='store_false')
    ap.add_argument('--target-fs', type=float, default=200.0)
    ap.add_argument('--normalize', default='demean',
                    choices=['none', 'demean', 'zscore'])
    ap.add_argument('--grid', action='store_true',
                    help='also run the window x order model-validation grid '
                         '(slower: fits every cell on --n-subjects subjects)')
    ap.add_argument('--n-subjects', type=int, default=3,
                    help='subjects for the --grid section (default 3)')
    ap.add_argument('--self-test', action='store_true',
                    help='check the diagnostics against known VARs and exit '
                         '(no data needed)')
    args = ap.parse_args()

    if args.self_test:
        print('DIAGNOSTIC SELF-TEST')
        return self_test()

    print('GC DATA REPORT — trial counts and MVAR model validation')
    print(f'  {args.method} / {args.atlas} / {args.feature_mode} / '
          f'leakage={args.leakage_correction} / normalize={args.normalize}')
    print(f'  project root:  {config.PROJECT_ROOT}')
    print(f'  vertex caches: {config.ROI_TIMESERIES_ROOT}')

    report_counts(args)
    if args.grid:
        report_grid(args)
    else:
        print('\n(--grid not set: skipped the window x order model-validation '
              'sweep)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
