"""Nonparametric Granger causality via Wilson spectral factorisation.

The third estimator in the project, and the one that was missing. Where
``granger.py`` fits a VAR and *constructs* the transfer function from its
coefficients (parametric), and ``granger_statespace.py`` derives the reduced model
from a fitted VAR by solving a DARE (state-space), this module never fits a model at
all:

    data  ->  cross-spectral density S(w)  ->  factorise  ->  (H, Sigma)  ->  Geweke

There is **no model order anywhere**. That is the defining property of the
route and the reason it is worth having in the comparison: an order sweep moves
the parametric and state-space estimators and cannot move this one.

What it costs instead is a frequency grid fine enough that the *circular* lag
domain does not alias. Working on ``n_bins`` frequencies means working with a
circular autocovariance of length ``n_bins``; if the true autocovariance has not
decayed by lag ``n_bins/2`` it wraps onto itself and you factorise the spectrum
of a different process. The rule of thumb (chapter section 22a) is

    n_bins  >~  1 + ln(eps) / (2 ln rho)

with ``rho`` the dominant pole modulus: ~135 bins for rho=0.95, ~690 for
rho=0.99. Sharp narrowband rhythms are the adversarial case. Every routine here
therefore returns the **Wilson residual**, which tracks the discretisation error
almost perfectly and is free — a run whose residual stalls near 1e-3 instead of
falling to ~1e-13 in a handful of iterations had too coarse a grid, and the
numbers should not be trusted.

Verified against a known VAR(2) whose exact spectrum is factorised back: the
recovered ``Sigma`` and ``H`` agree with the truth to ~4e-16 and ~3e-15, and the
spectral GC to ~1e-15 (``verify_against_var`` below reproduces this).

Blocks are supported throughout, so a FIXPC-k ROI reduction (k components per
ROI) works by passing index lists rather than scalar channel numbers.
"""
from __future__ import annotations

import numpy as np

from granger import band_average, DEFAULT_BANDS   # noqa: F401  (re-exported)

__all__ = [
    'cross_spectral_density', 'wilson_factorise', 'geweke_spectral_gc',
    'pairwise_spectral_gc_np', 'moving_window_gc_np', 'verify_against_var',
]


# ─────────────────────────────────────────────────────────────────────
# Cross-spectral density
# ─────────────────────────────────────────────────────────────────────
def cross_spectral_density(X, n_bins=None, taper='hann', detrend=True):
    """Estimate the CSD on a full-circle frequency grid.

    Parameters
    ----------
    X : (n_trials, n_signals, n_times)
        The trial ensemble for ONE window.
    n_bins : int or None
        Length of the FFT, i.e. the number of frequency bins around the full
        circle. Zero-pads when longer than ``n_times``. This is the knob that
        controls wrap-around aliasing; None uses ``n_times`` (usually too few).
    taper : {'hann', 'none'}
        A taper suppresses the spectral leakage a rectangular window causes.
    detrend : bool
        Remove each trial's own mean within the window. This is *not* the ERP
        removal (that happens upstream, across trials); it only stops a DC
        offset from dominating the lowest bins.

    Returns
    -------
    S : (n_bins, n_signals, n_signals) complex
        Hermitian at every bin, normalised so that the inverse FFT of ``S``
        along frequency gives the (circular) autocovariance sequence.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 3:
        raise ValueError('X must be (n_trials, n_signals, n_times)')
    n_tr, n_sig, n_t = X.shape
    if n_bins is None:
        n_bins = n_t
    if n_bins < n_t:
        raise ValueError(f'n_bins ({n_bins}) < window length ({n_t})')

    Z = X - X.mean(axis=2, keepdims=True) if detrend else X.copy()
    if taper == 'hann':
        w = np.hanning(n_t + 2)[1:-1]
        # preserve power: normalise the taper to unit mean square
        w = w / np.sqrt(np.mean(w ** 2))
        Z = Z * w[None, None, :]
    elif taper != 'none':
        raise ValueError(f'unknown taper: {taper}')

    F = np.fft.fft(Z, n=n_bins, axis=2)                 # (tr, sig, bins)
    F = np.moveaxis(F, 2, 0)                            # (bins, tr, sig)
    S = np.einsum('fti,ftj->fij', F, np.conj(F)) / (n_tr * n_t)
    # Hermitian symmetry can drift by round-off; impose it.
    return 0.5 * (S + np.conj(np.transpose(S, (0, 2, 1))))


# ─────────────────────────────────────────────────────────────────────
# Wilson's algorithm
# ─────────────────────────────────────────────────────────────────────
def _causal_projection(g):
    """The ``[.]_+`` operator: causal part, with the zero-lag term split.

    Keep the non-negative-lag Fourier coefficients, and replace the (Hermitian)
    zero-lag term by ``tril(g0, -1) + diag(g0)/2`` — a matrix ``U0`` satisfying
    ``U0 + U0^H = g0`` exactly. Halving the diagonal is what makes ``g = 2I`` a
    fixed point; the triangular part is a gauge choice that forces psi_0
    triangular and so makes the factorisation canonical.
    """
    n_f = g.shape[0]
    gam = np.fft.ifft(g, axis=0)
    gam[n_f // 2 + 1:] = 0.0                       # drop the negative lags
    g0 = gam[0]
    gam[0] = np.tril(g0, -1) + 0.5 * np.diag(np.diag(g0))
    return np.fft.fft(gam, axis=0)


def wilson_factorise(S, max_iter=500, tol=1e-12, ridge=0.0):
    """Factorise S(w) = H Sigma H^H with H causal, minimum-phase, H(0-lag)=I.

    Parameters
    ----------
    S : (n_bins, n, n) complex
        CSD on a full-circle grid, Hermitian positive-definite at every bin.
    max_iter, tol : int, float
        Iteration budget and convergence threshold on ``max|psi_{k+1}-psi_k|``.
    ridge : float
        Optional relative diagonal loading applied to S before factorising, as
        a rescue for near-singular CSDs (see chapter section 22b). Reported in
        the result so it can never be silent. 0.0 disables it.

    Returns
    -------
    dict with keys ``H`` (n_bins, n, n), ``Sigma`` (n, n), ``residual``,
    ``n_iter``, ``converged``, ``min_eig``, ``cond``.

    Notes
    -----
    ``min_eig`` and ``cond`` are the worst-over-frequency smallest eigenvalue
    and condition number of the (possibly ridged) CSD. They are the diagnostic
    for the rank-deficiency failure mode: a CSD that is singular at some
    frequency has no factorisation at all, and the iteration will either fail
    to converge or converge to something meaningless.
    """
    S = np.asarray(S, dtype=complex)
    n_f, n, _ = S.shape

    eig = np.linalg.eigvalsh(0.5 * (S + np.conj(np.transpose(S, (0, 2, 1)))))
    min_eig = float(eig.min())
    with np.errstate(divide='ignore', invalid='ignore'):
        cond = float(np.nanmax(eig[:, -1] / np.clip(eig[:, 0], 1e-300, None)))

    if ridge:
        scale = float(np.mean(np.real(np.trace(S, axis1=1, axis2=2))) / n)
        S = S + ridge * scale * np.eye(n)[None]

    gam0 = np.real(np.fft.ifft(S, axis=0)[0])
    gam0 = 0.5 * (gam0 + gam0.T)
    try:
        psi0 = np.linalg.cholesky(gam0)
    except np.linalg.LinAlgError:
        # Zero-lag covariance not positive definite: nothing to factorise.
        return dict(H=None, Sigma=None, residual=np.inf, n_iter=0,
                    converged=False, min_eig=min_eig, cond=cond)

    psi = np.tile(psi0.astype(complex), (n_f, 1, 1))
    I = np.eye(n)
    residual, n_iter = np.inf, 0
    for n_iter in range(1, max_iter + 1):
        try:
            psi_inv = np.linalg.inv(psi)
        except np.linalg.LinAlgError:
            return dict(H=None, Sigma=None, residual=np.inf, n_iter=n_iter,
                        converged=False, min_eig=min_eig, cond=cond)
        g = psi_inv @ S @ np.conj(np.transpose(psi_inv, (0, 2, 1))) + I
        psi_new = psi @ _causal_projection(g)
        residual = float(np.max(np.abs(psi_new - psi)))
        psi = psi_new
        if not np.all(np.isfinite(psi)):
            return dict(H=None, Sigma=None, residual=np.inf, n_iter=n_iter,
                        converged=False, min_eig=min_eig, cond=cond)
        if residual < tol:
            break

    psi_0 = np.real(np.fft.ifft(psi, axis=0)[0])
    Sigma = psi_0 @ psi_0.T
    H = psi @ np.linalg.inv(psi_0)
    return dict(H=H, Sigma=Sigma, residual=residual, n_iter=n_iter,
                converged=bool(residual < tol), min_eig=min_eig, cond=cond)


# ─────────────────────────────────────────────────────────────────────
# Geweke's formula, block form
# ─────────────────────────────────────────────────────────────────────
def geweke_spectral_gc(H, S, Sigma, src, tgt):
    """Spectral GC from the source block to the target block, f_{src->tgt}.

    Implements the general (block) Geweke measure in the chapter's convention
    (Part 0, written source-first):

        F_{x->y}(w) = ln |S_yy| / |S_yy - H_yx Sbar_{xx|y} H_yx^H|

    with ``Sbar_{xx|y} = Sigma_xx - Sigma_xy Sigma_yy^-1 Sigma_yx`` the partial
    covariance that removes instantaneous causality. Note the index convention:
    the arrow is source-first, the spectrum block is the TARGET's own power, and
    the transfer block is destination-first (``H[tgt, src]``).

    Setting both blocks to size 1 reproduces the bivariate formula exactly.
    """
    src = np.atleast_1d(np.asarray(src, dtype=int))
    tgt = np.atleast_1d(np.asarray(tgt, dtype=int))

    S_tt = S[np.ix_(range(S.shape[0]), tgt, tgt)]
    H_ts = H[np.ix_(range(H.shape[0]), tgt, src)]
    Sig_ss = Sigma[np.ix_(src, src)]
    Sig_st = Sigma[np.ix_(src, tgt)]
    Sig_tt = Sigma[np.ix_(tgt, tgt)]
    partial = Sig_ss - Sig_st @ np.linalg.solve(Sig_tt, Sig_st.T)

    inner = S_tt - H_ts @ partial @ np.conj(np.transpose(H_ts, (0, 2, 1)))
    num = np.real(np.linalg.det(S_tt))
    den = np.real(np.linalg.det(inner))
    with np.errstate(divide='ignore', invalid='ignore'):
        f = np.log(num / den)
    return np.where(np.isfinite(f), np.maximum(f, 0.0), 0.0)


# ─────────────────────────────────────────────────────────────────────
# Front ends matching granger.py's schema
# ─────────────────────────────────────────────────────────────────────
def pairwise_spectral_gc_np(X, freqs, fs, n_bins=512, blocks=None,
                            taper='hann', ridge=0.0, max_iter=500, tol=1e-12):
    """Nonparametric f_xy, f_yx for one window, on the requested frequencies.

    ``blocks`` is ``(src_idx, tgt_idx)``; None means channel 0 vs channel 1.
    Returns ``(f_xy, f_yx, info)`` where ``info`` carries the Wilson diagnostics
    so a caller can record convergence per window instead of assuming it.
    """
    X = np.asarray(X, dtype=float)
    n_sig = X.shape[1]
    if blocks is None:
        if n_sig != 2:
            raise ValueError('blocks=None requires exactly 2 signals')
        src, tgt = [0], [1]
    else:
        src, tgt = [list(np.atleast_1d(b)) for b in blocks]

    S = cross_spectral_density(X, n_bins=n_bins, taper=taper)
    res = wilson_factorise(S, max_iter=max_iter, tol=tol, ridge=ridge)
    info = {k: res[k] for k in ('residual', 'n_iter', 'converged',
                                'min_eig', 'cond')}
    freqs = np.asarray(freqs, dtype=float)
    if res['H'] is None:
        nan = np.full(freqs.shape, np.nan)
        return nan, nan.copy(), info

    f_full = np.fft.fftfreq(S.shape[0], d=1.0 / fs)
    gc_xy = geweke_spectral_gc(res['H'], S, res['Sigma'], src, tgt)
    gc_yx = geweke_spectral_gc(res['H'], S, res['Sigma'], tgt, src)

    # interpolate the full-circle grid onto the requested frequencies
    keep = f_full >= 0
    order = np.argsort(f_full[keep])
    fk = f_full[keep][order]
    out = (np.interp(freqs, fk, gc_xy[keep][order]),
           np.interp(freqs, fk, gc_yx[keep][order]))
    return out[0], out[1], info


def moving_window_gc_np(X, freqs, fs, win_samples, step=1, n_bins=512,
                        blocks=None, **kw):
    """Sliding-window nonparametric GC, mirroring granger.moving_window_pairwise_gc.

    Returns a dict with ``f_xy``, ``f_yx`` of shape (n_freqs, n_windows),
    ``win_start``, and per-window diagnostic arrays ``residual``, ``n_iter``,
    ``converged``, ``min_eig``, ``cond``.
    """
    X = np.asarray(X, dtype=float)
    n_t = X.shape[2]
    starts = np.arange(0, n_t - win_samples + 1, step)
    freqs = np.asarray(freqs, dtype=float)
    n_w, n_f = starts.size, freqs.size

    f_xy = np.empty((n_f, n_w)); f_yx = np.empty((n_f, n_w))
    diag = {k: np.empty(n_w) for k in ('residual', 'n_iter', 'min_eig', 'cond')}
    diag['converged'] = np.zeros(n_w, dtype=bool)

    for w, s in enumerate(starts):
        seg = X[:, :, s:s + win_samples]
        fxy, fyx, info = pairwise_spectral_gc_np(
            seg, freqs, fs, n_bins=n_bins, blocks=blocks, **kw)
        f_xy[:, w] = fxy; f_yx[:, w] = fyx
        for k in ('residual', 'n_iter', 'min_eig', 'cond'):
            diag[k][w] = info[k]
        diag['converged'][w] = info['converged']

    return dict(f_xy=f_xy, f_yx=f_yx, win_start=starts, **diag)


# ─────────────────────────────────────────────────────────────────────
# Verification — nonparametric against a process whose truth is known
# ─────────────────────────────────────────────────────────────────────
def verify_against_var(n_bins=512, fs=200.0, verbose=True):
    """Factorise the EXACT spectrum of a known VAR(2) and compare.

    If the two routes are genuinely inverse operations, the recovered
    (H, Sigma) and the resulting spectral GC must agree with the generating
    model to machine precision. Returns a dict of the discrepancies.
    """
    from granger import spectral_transfer

    A = np.zeros((2, 2, 2))
    A[:, :, 0] = [[0.5, 0.0], [0.4, 0.5]]
    A[:, :, 1] = [[-0.3, 0.0], [0.0, -0.3]]
    Sigma = np.array([[1.0, 0.2], [0.2, 1.0]])

    f_full = np.arange(n_bins) * fs / n_bins
    H_true, S_true = spectral_transfer(A, Sigma, f_full, fs)
    res = wilson_factorise(S_true)

    d_sigma = float(np.max(np.abs(res['Sigma'] - Sigma)))
    d_H = float(np.max(np.abs(res['H'] - H_true)))
    gc_true = geweke_spectral_gc(H_true, S_true, Sigma, [0], [1])
    gc_rec = geweke_spectral_gc(res['H'], S_true, res['Sigma'], [0], [1])
    d_gc = float(np.max(np.abs(gc_true - gc_rec)))

    out = dict(d_sigma=d_sigma, d_H=d_H, d_gc=d_gc,
               n_iter=res['n_iter'], residual=res['residual'],
               converged=res['converged'])
    if verbose:
        print(f'Wilson vs known VAR(2) on {n_bins} bins: '
              f'{res["n_iter"]} iters, residual {res["residual"]:.1e}')
        print(f'  max|dSigma| = {d_sigma:.2e}')
        print(f'  max|dH|     = {d_H:.2e}')
        print(f'  max|dGC|    = {d_gc:.2e}')
    return out


if __name__ == '__main__':
    verify_against_var()
