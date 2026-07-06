"""
Validation for granger.py — checks the MVAR fit and spectral GC against
ground truth on simulated systems, and cross-checks against statsmodels.

Run:  python validate_granger.py
"""
import numpy as np

from granger import (
    fit_mvar, pairwise_spectral_gc, time_reversed_pairwise_gc,
    moving_window_pairwise_gc, band_average, order_criteria, DEFAULT_BANDS,
    reduce_roi_first_pc,
)

RNG = np.random.default_rng(0)
PASS = '\033[92mPASS\033[0m'
FAIL = '\033[91mFAIL\033[0m'
_results = []


def check(name, cond, detail=''):
    _results.append(bool(cond))
    print(f'  [{PASS if cond else FAIL}] {name}' + (f'  — {detail}' if detail else ''))


def simulate_unidirectional(n_trials, n_times, c=0.4, burn=200, seed=0):
    """x -> y only.  x(t)=0.5 x(t-1)+ex ; y(t)=0.5 y(t-1)+c x(t-1)+ey.

    Returns array (n_trials, 2, n_times); row 0 = x, row 1 = y.
    """
    rng = np.random.default_rng(seed)
    out = np.empty((n_trials, 2, n_times))
    for tr in range(n_trials):
        T = n_times + burn
        x = np.zeros(T)
        y = np.zeros(T)
        ex = rng.standard_normal(T)
        ey = rng.standard_normal(T)
        for t in range(1, T):
            x[t] = 0.5 * x[t - 1] + ex[t]
            y[t] = 0.5 * y[t - 1] + c * x[t - 1] + ey[t]
        out[tr, 0] = x[burn:]
        out[tr, 1] = y[burn:]
    return out


def simulate_known_ar(n_times, A1, A2, Sigma_chol, burn=500, seed=1):
    """Single realization of a bivariate AR(2) with given coefficients."""
    rng = np.random.default_rng(seed)
    T = n_times + burn
    x = np.zeros((2, T))
    e = Sigma_chol @ rng.standard_normal((2, T))
    for t in range(2, T):
        x[:, t] = A1 @ x[:, t - 1] + A2 @ x[:, t - 2] + e[:, t]
    return x[:, burn:]


def test_coefficient_recovery():
    print('\n== MVAR coefficient recovery (ground truth) ==')
    A1 = np.array([[0.5, 0.0], [0.3, 0.4]])
    A2 = np.array([[-0.2, 0.0], [0.0, -0.1]])
    Sig_chol = np.array([[1.0, 0.0], [0.2, 0.9]])
    x = simulate_known_ar(60000, A1, A2, Sig_chol)
    A, Sigma = fit_mvar(x, 2)
    err1 = np.abs(A[:, :, 0] - A1).max()
    err2 = np.abs(A[:, :, 1] - A2).max()
    check('A_1 recovered (convention x(t)=sum A_k x(t-k)+e)', err1 < 0.03,
          f'max|dA1|={err1:.4f}')
    check('A_2 recovered', err2 < 0.03, f'max|dA2|={err2:.4f}')
    Sigma_true = Sig_chol @ Sig_chol.T
    errS = np.abs(Sigma - Sigma_true).max()
    check('Sigma recovered', errS < 0.05, f'max|dSigma|={errS:.4f}')


def test_statsmodels_crosscheck():
    print('\n== Cross-check MVAR fit vs statsmodels VAR (OLS) ==')
    try:
        from statsmodels.tsa.api import VAR
    except Exception as e:
        check('statsmodels available', False, f'import failed: {e}')
        return
    A1 = np.array([[0.5, 0.0], [0.3, 0.4]])
    A2 = np.array([[-0.2, 0.0], [0.0, -0.1]])
    Sig_chol = np.array([[1.0, 0.0], [0.2, 0.9]])
    x = simulate_known_ar(40000, A1, A2, Sig_chol)
    A, Sigma = fit_mvar(x, 2)
    res = VAR(x.T).fit(2, trend='n')
    # statsmodels coefs[k] is A_{k+1}, shape (order, n, n)
    d1 = np.abs(A[:, :, 0] - res.coefs[0]).max()
    d2 = np.abs(A[:, :, 1] - res.coefs[1]).max()
    check('Morf vs OLS A_1 agree', d1 < 0.03, f'max|diff|={d1:.4f}')
    check('Morf vs OLS A_2 agree', d2 < 0.03, f'max|diff|={d2:.4f}')


def test_gc_direction():
    print('\n== Spectral GC recovers direction (x -> y) ==')
    X = simulate_unidirectional(60, 400, c=0.4)
    fs = 200.0
    freqs = np.arange(1, 90)
    f_xy, f_yx = pairwise_spectral_gc(X, order=3, freqs=freqs, fs=fs)
    m_xy = f_xy.mean()
    m_yx = f_yx.mean()
    check('mean GC x->y  >>  mean GC y->x', m_xy > 5 * max(m_yx, 1e-6),
          f'f_xy={m_xy:.4f}, f_yx={m_yx:.4f}')
    check('driven direction GC is positive', m_xy > 0.02, f'f_xy={m_xy:.4f}')
    check('spurious direction GC near zero', m_yx < 0.02, f'f_yx={m_yx:.4f}')


def test_no_coupling_gives_zero():
    print('\n== Independent signals -> ~zero GC both directions ==')
    X = simulate_unidirectional(60, 400, c=0.0)  # c=0 -> independent
    fs = 200.0
    freqs = np.arange(1, 90)
    f_xy, f_yx = pairwise_spectral_gc(X, order=3, freqs=freqs, fs=fs)
    check('GC x->y near zero for independent signals', f_xy.mean() < 0.02,
          f'f_xy={f_xy.mean():.4f}')
    check('GC y->x near zero for independent signals', f_yx.mean() < 0.02,
          f'f_yx={f_yx.mean():.4f}')


def test_trgc_sign():
    print('\n== Diff-TRGC sign for true x -> y flow ==')
    X = simulate_unidirectional(80, 400, c=0.4)
    fs = 200.0
    freqs = np.arange(1, 90)
    d_xy, d_yx = time_reversed_pairwise_gc(X, order=3, freqs=freqs, fs=fs)
    check('Diff-TRGC positive for true driver x->y', d_xy.mean() > 0,
          f'd_xy={d_xy.mean():.4f}')
    check('Diff-TRGC antisymmetric (d_xy = -d_yx)',
          np.allclose(d_xy, -d_yx), '')


def test_moving_window_and_bands():
    print('\n== Moving-window driver + band averaging ==')
    X = simulate_unidirectional(50, 300, c=0.4)
    fs = 200.0
    freqs = np.arange(1, 31)  # 1..30 Hz like the MATLAB pipeline
    res = moving_window_pairwise_gc(X, order=3, freqs=freqs, fs=fs,
                                    win_samples=40, step=5, trgc=True)
    n_win_expected = (300 - 40) // 5 + 1
    check('window count correct', res['f_xy'].shape == (30, n_win_expected),
          f"shape={res['f_xy'].shape}, expected=(30,{n_win_expected})")
    check('TRGC computed per window', res['d_xy'].shape == (30, n_win_expected))
    bands = band_average(res['f_xy'], freqs, DEFAULT_BANDS)
    check('band average keys', set(bands) == set(DEFAULT_BANDS))
    check('band average shape matches windows',
          bands['theta'].shape == (n_win_expected,))
    check('windowed GC still shows x->y dominance',
          res['f_xy'].mean() > res['f_yx'].mean(),
          f"f_xy={res['f_xy'].mean():.4f}, f_yx={res['f_yx'].mean():.4f}")


def test_order_selection():
    print('\n== AIC/BIC order selection prefers the true order ==')
    A1 = np.array([[0.5, 0.0], [0.3, 0.4]])
    A2 = np.array([[-0.2, 0.0], [0.0, -0.1]])
    Sig_chol = np.array([[1.0, 0.0], [0.2, 0.9]])
    x = simulate_known_ar(8000, A1, A2, Sig_chol)
    orders, aic, bic = order_criteria(x, max_order=8)
    best_bic = orders[np.argmin(bic)]
    check('BIC selects true order (2)', best_bic == 2, f'BIC order={best_bic}')


def test_multitrial_short_window():
    print('\n== Multi-trial fit is estimable on short windows ==')
    # order-6 model on 20-sample windows: infeasible per-trial, feasible
    # across the trial ensemble (the BSMART regime).
    X = simulate_unidirectional(120, 20, c=0.4, burn=300)
    fs = 500.0
    freqs = np.arange(1, 31)
    try:
        f_xy, f_yx = pairwise_spectral_gc(X, order=6, freqs=freqs, fs=fs)
        ok = np.all(np.isfinite(f_xy)) and np.all(np.isfinite(f_yx))
        check('order-6 GC on 20-sample x 120-trial ensemble is finite', ok,
              f'f_xy={f_xy.mean():.4f}, f_yx={f_yx.mean():.4f}')
        check('direction preserved on short multi-trial window',
              f_xy.mean() > f_yx.mean())
    except Exception as e:
        check('order-6 GC on short multi-trial window', False, str(e))


def _bsmart_pwcausal_literal(X, order, freqs, fs):
    """Literal transcription of BSMART spectrum.m + pwcausal.m (bivariate).

    Reproduces the MATLAB math verbatim, including the /fs factors, to
    prove granger.pairwise_spectral_gc is a faithful port.  Uses the same
    fit_mvar so only the spectral-GC arithmetic is under test.
    """
    A, Z = fit_mvar(X, order)          # A = physical coeffs (= -bsmart coeff)
    n = 2
    Fx2y = np.empty(len(freqs))
    Fy2x = np.empty(len(freqs))
    for fi, f in enumerate(freqs):
        # spectrum.m: H = I + sum coeff_bsmart * exp(-i m 2pi f/fs), with
        # coeff_bsmart = -A  ->  H = I - sum A exp(...)
        Hden = np.eye(n, dtype=complex)
        for m in range(1, order + 1):
            Hden = Hden - A[:, :, m - 1] * np.exp(-1j * m * 2 * np.pi * f / fs)
        H = np.linalg.inv(Hden)
        S = H @ Z.astype(complex) @ H.conj().T / fs      # spectrum.m /fs
        eyx = Z[1, 1] - Z[0, 1] ** 2 / Z[0, 0]
        exy = Z[0, 0] - Z[1, 0] ** 2 / Z[1, 1]
        Fy2x[fi] = np.log(abs(S[0, 0]) /
                          abs(S[0, 0] - (H[0, 1] * eyx * np.conj(H[0, 1])) / fs))
        Fx2y[fi] = np.log(abs(S[1, 1]) /
                          abs(S[1, 1] - (H[1, 0] * exy * np.conj(H[1, 0])) / fs))
    return Fx2y, Fy2x


def test_matches_bsmart_pwcausal():
    print('\n== granger.py == BSMART pwcausal.m (literal transcription) ==')
    X = simulate_unidirectional(60, 400, c=0.4, seed=11)
    fs = 200.0
    freqs = np.arange(1, 90)
    f_xy, f_yx = pairwise_spectral_gc(X, order=4, freqs=freqs, fs=fs)
    b_xy, b_yx = _bsmart_pwcausal_literal(X, order=4, freqs=freqs, fs=fs)
    check('Fx2y matches BSMART to machine precision',
          np.allclose(f_xy, b_xy, atol=1e-10),
          f'max|diff|={np.abs(f_xy - b_xy).max():.2e}')
    check('Fy2x matches BSMART to machine precision',
          np.allclose(f_yx, b_yx, atol=1e-10),
          f'max|diff|={np.abs(f_yx - b_yx).max():.2e}')


def test_roi_reduction_recovers_mode():
    print('\n== PC1 ROI reduction recovers the dominant spatial mode ==')
    rng = np.random.default_rng(3)
    n_ep, n_v, n_t = 40, 25, 300
    latent = rng.standard_normal((n_ep, n_t))          # shared temporal mode
    loadings = rng.standard_normal(n_v) + 1.5          # per-vertex weights
    vdata = (loadings[None, :, None] * latent[:, None, :]
             + 0.3 * rng.standard_normal((n_ep, n_v, n_t)))
    vc, w = reduce_roi_first_pc(vdata, return_filter=True)
    # virtual channel should track the latent mode (up to sign) per epoch
    corr = np.mean([abs(np.corrcoef(vc[e], latent[e])[0, 1])
                    for e in range(n_ep)])
    check('virtual channel correlates with latent mode', corr > 0.95,
          f'mean|r|={corr:.3f}')
    check('filter is unit norm', abs(np.linalg.norm(w) - 1.0) < 1e-8)
    check('output shape (n_epochs, n_times)', vc.shape == (n_ep, n_t))


def test_reduction_preserves_gc_direction():
    print('\n== GC direction survives vertex reduction (ROI x -> ROI y) ==')
    rng = np.random.default_rng(4)
    base = simulate_unidirectional(50, 400, c=0.4, seed=7)  # (tr, 2, T)
    n_tr, _, T = base.shape
    # expand each latent ROI signal across a patch of vertices + noise
    nvx, nvy = 20, 15
    lx = rng.standard_normal(nvx) + 1.0
    ly = rng.standard_normal(nvy) + 1.0
    Vx = lx[None, :, None] * base[:, 0][:, None, :] + 0.3 * rng.standard_normal((n_tr, nvx, T))
    Vy = ly[None, :, None] * base[:, 1][:, None, :] + 0.3 * rng.standard_normal((n_tr, nvy, T))
    cx = reduce_roi_first_pc(Vx)
    cy = reduce_roi_first_pc(Vy)
    X = np.stack([cx, cy], axis=1)                     # (tr, 2, T)
    f_xy, f_yx = pairwise_spectral_gc(X, order=3, freqs=np.arange(1, 90), fs=200.0)
    check('reduced-ROI GC recovers x->y dominance',
          f_xy.mean() > 5 * max(f_yx.mean(), 1e-6),
          f'f_xy={f_xy.mean():.4f}, f_yx={f_yx.mean():.4f}')


if __name__ == '__main__':
    test_coefficient_recovery()
    test_statsmodels_crosscheck()
    test_gc_direction()
    test_no_coupling_gives_zero()
    test_trgc_sign()
    test_moving_window_and_bands()
    test_order_selection()
    test_multitrial_short_window()
    test_matches_bsmart_pwcausal()
    test_roi_reduction_recovers_mode()
    test_reduction_preserves_gc_direction()

    n_pass = sum(_results)
    n_tot = len(_results)
    print(f'\n{"="*50}\n{n_pass}/{n_tot} checks passed')
    raise SystemExit(0 if n_pass == n_tot else 1)
