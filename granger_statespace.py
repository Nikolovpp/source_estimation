"""
State-space multivariate/conditional Granger causality.

Faithful port of the MVGC toolbox (Barnett & Seth) state-space GC method:

    L. Barnett & A. K. Seth, "Granger causality for state-space models",
    Phys. Rev. E 91(4) Rapid Communication, 2015.

Why state-space over the classic AR conditional GC (Chen, Bressler & Ding
2006, implemented in ``granger.conditional_spectral_gc``)?  The classic
method fits a *separate reduced VAR* omitting the source, but the reduced
sub-process of a VAR is a VARMA — so a finite reduced VAR is
mis-specified and biased.  The state-space method instead derives the
reduced model's residual covariance *exactly* from the full VAR by
solving a discrete algebraic Riccati equation (DARE), giving more
accurate GC (Barnett & Seth 2015).

Both time-domain and frequency-resolved (spectral) conditional GC are
provided, ported from MVGC's ``var_to_mvgc.m`` / ``var_to_smvgc.m`` and
their helpers ``var2riss.m``, ``ss2iss.m``, ``ss2itrfun.m``, ``parcov.m``.

Conventions match ``granger.py``: ``A`` has shape ``(n, n, order)`` with
``x(t) = sum_k A_k x(t-k) + e(t)`` and residual covariance ``Sigma``.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_discrete_are, cholesky

from granger import fit_mvar, ar_transfer


def _logdet(M):
    M = np.asarray(M)
    sign, ld = np.linalg.slogdet(M)
    return float(np.real(ld))


def parcov(V, x, y):
    """Partial covariance of block ``x`` given block ``y`` (MVGC ``parcov``).

    ``P = V(x,x) - V(x,y) V(y,y)^{-1} V(y,x)``.
    """
    x = list(x); y = list(y)
    Vyy = V[np.ix_(y, y)]
    Vyx = V[np.ix_(y, x)]
    L = cholesky(Vyy, lower=True)
    U = np.linalg.solve(L, Vyx)                      # L^{-1} V(y,x)
    return V[np.ix_(x, x)] - U.T @ U


def _companion(A):
    """Full VAR companion matrix ``[[A_1..A_p],[I,0]]`` (pn x pn)."""
    n, _, p = A.shape
    top = np.concatenate([A[:, :, k] for k in range(p)], axis=1)  # (n, pn)
    pn = p * n
    if pn - n > 0:
        bottom = np.concatenate(
            [np.eye(pn - n), np.zeros((pn - n, n))], axis=1)
        return np.vstack([top, bottom])
    return top


def var2riss(A, SIG, y, r):
    """VAR -> reduced innovations form via DARE (MVGC ``var2riss``/``ss2iss``).

    Returns ``(KT, VR)``: the reduced-model Kalman gain ``KT`` (p*ny, nr)
    and innovations (residual) covariance ``VR`` (nr, nr) for the block
    ``r``, with the source block ``y`` marginalized out — computed exactly
    from the full VAR, no reduced-model refit.
    """
    n, _, p = A.shape
    y = list(y); r = list(r)
    ny = len(y); nr = len(r)
    pny = p * ny
    pny1 = pny - ny

    # State-space (general form) for the y-subsystem dynamics.
    top = np.concatenate([A[np.ix_(y, y)][:, :, k] for k in range(p)], axis=1)
    if pny1 > 0:
        A_ss = np.vstack([top, np.concatenate(
            [np.eye(pny1), np.zeros((pny1, ny))], axis=1)])
    else:
        A_ss = top
    C = np.concatenate([A[np.ix_(r, y)][:, :, k] for k in range(p)], axis=1)  # (nr, pny)
    Q = np.zeros((pny, pny)); Q[:ny, :ny] = SIG[np.ix_(y, y)]
    S = np.zeros((pny, nr)); S[:ny, :] = SIG[np.ix_(y, r)]
    R = SIG[np.ix_(r, r)]

    # DARE (ss2iss eqns): P = A P A' - (A P C'+S)(C P C'+R)^{-1}(A P C'+S)' + Q
    # scipy solves a^H X a - X - (a^H X b + s)(r + b^H X b)^{-1}(...) + q = 0
    # with a = A_ss.T, b = C.T  ->  a^H X a = A_ss X A_ss', a^H X b = A_ss X C'.
    P = solve_discrete_are(A_ss.T, C.T, Q, R, s=S)
    V = R + C @ P @ C.T
    KT = (A_ss @ P @ C.T + S) @ np.linalg.inv(V)     # (pny, nr)
    return KT, V


def _ss_inv_transfer(A_ss, C, K, freqs, fs):
    """Inverse transfer ``J(f) = I - C (e^{i w} I - (A-KC))^{-1} K``.

    Port of MVGC ``ss2itrfun`` (uses ``e^{+i w}``); w = 2*pi*f/fs.
    """
    n, r = C.shape
    B = A_ss - K @ C
    Ir = np.eye(r); In = np.eye(n)
    J = np.empty((len(freqs), n, n), dtype=complex)
    for k, f in enumerate(freqs):
        wz = np.exp(1j * 2.0 * np.pi * f / fs)
        J[k] = In - C @ np.linalg.solve(wz * Ir - B, K)
    return J


def ss_conditional_gc(A, SIG, x, y, freqs=None, fs=None):
    """State-space conditional GC ``F_{y->x | rest}``.

    Parameters
    ----------
    A : (n, n, order) VAR coefficients; SIG : (n, n) residual covariance.
    x, y : int or list — target and source variable indices.
    freqs, fs : if both given, also return the spectral GC.

    Returns
    -------
    F_time : float
        Time-domain conditional GC.
    f_spec : np.ndarray, shape (n_freqs,) — only if freqs/fs supplied.
    """
    n = A.shape[0]
    x = [x] if np.isscalar(x) else list(x)
    y = [y] if np.isscalar(y) else list(y)
    z = [v for v in range(n) if v not in x + y]
    r = x + z                                        # reduced-model block
    nx = len(x)
    xr = list(range(nx))                             # x within r

    KT, SIGR = var2riss(A, SIG, y, r)
    F_time = _logdet(SIGR[np.ix_(xr, xr)]) - _logdet(SIG[np.ix_(x, x)])

    if freqs is None or fs is None:
        return F_time

    freqs = np.asarray(freqs, dtype=float)
    w = y + z
    p = A.shape[2]
    H = ar_transfer(A, freqs, fs)                    # full transfer (e^{-iw})
    PSIGL = cholesky(parcov(SIG, w, x), lower=True)

    if not z:  # unconditional (2-block) — no reduced SS needed
        SIGL = cholesky(SIG, lower=True)
        f_spec = np.empty(freqs.size)
        for k in range(freqs.size):
            HSIGL = H[k][np.ix_(x, range(n))] @ SIGL
            SR = HSIGL @ HSIGL.conj().T
            HR = H[k][np.ix_(x, y)] @ PSIGL
            f_spec[k] = _logdet(SR) - _logdet(SR - HR @ HR.conj().T)
        return F_time, np.where(np.isfinite(f_spec), np.maximum(f_spec, 0), 0)

    # Conditional: build reduced SS (AR, CR, KR) from the DARE Kalman gain.
    ny = len(y); nr = len(r)
    pn = p * n
    AR = _companion(A)                               # (pn, pn)
    CR = np.concatenate([A[np.ix_(r, range(n))][:, :, k] for k in range(p)],
                        axis=1)                       # (nr, pn)
    KR = np.zeros((pn, nr))
    KR[r, :] = np.eye(nr)
    qn = 0
    for qy in range(0, p * ny - ny + 1, ny):
        KR[[qn + yy for yy in y], :] = KT[qy:qy + ny, :]
        qn += n
    BR = _ss_inv_transfer(AR, CR, KR, freqs, fs)     # (nf, nr, nr)

    SR = SIGR[np.ix_(xr, xr)]
    LDSR = _logdet(SR)
    f_spec = np.empty(freqs.size)
    for k in range(freqs.size):
        HR = BR[k][np.ix_(xr, range(nr))] @ H[k][np.ix_(r, w)] @ PSIGL
        f_spec[k] = LDSR - _logdet(SR - HR @ HR.conj().T)
    return F_time, np.where(np.isfinite(f_spec), np.maximum(f_spec, 0.0), 0.0)


def moving_window_conditional_gc(X, order, freqs, fs, win_samples, step=1,
                                 pairs=None, n_jobs=1):
    """Sliding-window state-space conditional spectral GC.

    Fits the full multivariate VAR once per window (all signals jointly)
    and derives each requested directed pair's conditional GC from it.
    Parallelizes over windows (each window is one joint fit + DARE solves).

    Returns
    -------
    dict
        ``gc`` : {(src, tgt): (n_freqs, n_windows)}.
        ``win_start`` : (n_windows,).
    """
    from joblib import Parallel, delayed

    X = np.asarray(X, dtype=float)
    n = X.shape[1]
    n_times = X.shape[2]
    freqs = np.asarray(freqs, dtype=float)
    starts = np.arange(0, n_times - win_samples + 1, step)
    if pairs is None:
        pairs = [(s, t) for s in range(n) for t in range(n) if s != t]

    def _win(s):
        seg = X[:, :, s:s + win_samples]
        res = statespace_conditional_gc(seg, order, freqs, fs, pairs=pairs)
        return {p: res[p][1] for p in pairs}      # spectral part only

    outs = Parallel(n_jobs=n_jobs, prefer='processes')(
        delayed(_win)(int(s)) for s in starts)

    gc = {p: np.empty((freqs.size, starts.size)) for p in pairs}
    for w, o in enumerate(outs):
        for p in pairs:
            gc[p][:, w] = o[p]
    return {'gc': gc, 'win_start': starts}


def statespace_conditional_gc(X, order, freqs=None, fs=None, pairs=None):
    """Fit the full VAR once and compute state-space conditional GC.

    Parameters
    ----------
    X : (n_trials, n_signals, n_times)
    order : int
    freqs, fs : optional — if given, spectral GC is returned per pair.
    pairs : list of (src, tgt), optional (default all ordered pairs).

    Returns
    -------
    dict {(src, tgt): value}
        ``value`` is the time-domain float, or (F_time, f_spec) if
        freqs/fs supplied.
    """
    n = X.shape[1]
    A, SIG = fit_mvar(X, order)
    if pairs is None:
        pairs = [(s, t) for s in range(n) for t in range(n) if s != t]
    out = {}
    for src, tgt in pairs:
        out[(src, tgt)] = ss_conditional_gc(A, SIG, tgt, src, freqs, fs)
    return out
