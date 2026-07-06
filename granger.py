"""
Parametric MVAR spectral Granger causality in source space.

Python reimplementation of the MATLAB BSMART ``mov_bi_ga`` pipeline used
for the sensor-space analysis, lifted here to operate on the
source-estimated ROI time courses produced by the decoding pipeline
(each ROI's vertices reduced to one virtual channel via a fixed first-PC
spatial filter — see ``reduce_roi_first_pc``).

Design goals
------------
* **Parametric MVAR** — fit a vector autoregressive model per window and
  derive the spectral Granger causality from its transfer function and
  noise covariance (Geweke 1982; Ding, Chen & Bressler 2006).  This is
  the BSMART approach, NOT non-parametric spectral factorization.
* **Frequency-resolved** — GC is returned per frequency so it can be
  averaged into the study's bands (theta 4-7, alpha 8-12, low-beta
  13-20, high-beta 21-30 Hz), matching the MATLAB output.
* **Multi-trial fitting** — the AR model is fit on the trial ensemble
  (all epochs treated as realizations of the same process), exactly as
  BSMART's ``armorf(x, Nr, Nl, p)`` does.  This is what makes an
  order-p model estimable on a short (e.g. 40 ms) window.
* **Robustness** — Diff-TRGC (time-reversed Granger causality;
  Haufe et al. 2013, Winkler et al. 2016) is provided as the primary
  guard against volume-conduction / SNR-asymmetry artefacts that make
  plain GC unreliable on leaky source signals.

The core AR estimator is a faithful port of BSMART's ``armorf.m``
(Morf's recursive multichannel LWR / maximum-entropy method; Morf et al.
1978), located in FieldTrip at ``external/bsmart/armorf.m``.

Conventions
-----------
* Model:  ``x(t) = sum_{k=1..p} A_k x(t-k) + e(t)``  with
  ``cov(e) = Sigma``.  AR coefficient arrays are returned with shape
  ``(n_signals, n_signals, order)`` (``A[:, :, k-1]`` is ``A_k``).
* Transfer function: ``A(f) = I - sum_k A_k exp(-i 2*pi*f*k/fs)``,
  ``H(f) = A(f)^-1``, spectral matrix ``S(f) = H(f) Sigma H(f)^H``.
  (An overall ``1/fs`` scaling of ``S`` cancels in every GC log-ratio,
  so it is omitted.)
* Directional naming for a signal pair ``[x, y]`` (row 0 = x, row 1 = y):
  ``F_xy`` is x->y, ``F_yx`` is y->x — matching BSMART ``mov_bi_ga``,
  where the seed is row 0 and the target is row 1.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cholesky, inv


# ─────────────────────────────────────────────────────────────────────
# MVAR estimation — faithful port of BSMART armorf.m
# ─────────────────────────────────────────────────────────────────────
def _mct(M):
    """MATLAB ``chol(M)'`` : lower-triangular transpose of the upper
    Cholesky factor.  ``scipy.linalg.cholesky(M, lower=False)`` returns
    the same upper factor ``R`` (``R^H R = M``) as MATLAB ``chol``; the
    transpose reproduces MATLAB's ``chol(M)'`` exactly.
    """
    return cholesky(M, lower=False).T


def fit_mvar(X, order):
    """Fit a multichannel AR model via Morf's method (BSMART ``armorf``).

    Parameters
    ----------
    X : np.ndarray
        ``(n_trials, n_signals, n_times)`` — the trial ensemble, or
        ``(n_signals, n_times)`` for a single realization.  All trials
        are treated as independent realizations of the same AR process
        (``Nr = n_trials``, ``Nl = n_times``).
    order : int
        AR model order ``p``.

    Returns
    -------
    A : np.ndarray, shape (n_signals, n_signals, order)
        AR coefficient matrices; ``A[:, :, k]`` is ``A_{k+1}`` in
        ``x(t) = sum_k A_k x(t-k) + e(t)``.
    Sigma : np.ndarray, shape (n_signals, n_signals)
        Noise (residual) covariance of the fitted model.

    Notes
    -----
    Direct translation of ``external/bsmart/armorf.m`` (Yonghong Chen,
    2002), preserving its normalization so results match BSMART.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 2:
        X = X[np.newaxis]
    Nr, L, Nl = X.shape
    p = int(order)
    N = Nr * Nl

    En = np.zeros((L, L))
    ap0 = np.zeros((L, L))
    bp0 = np.zeros((L, L))
    for i in range(Nr):
        seg = X[i]
        En += seg @ seg.T
        ap0 += seg[:, 1:] @ seg[:, 1:].T       # samples 2..Nl
        bp0 += seg[:, :-1] @ seg[:, :-1].T     # samples 1..Nl-1

    # armorf.m lines 40-41: inv((chol(ap/Nr*(Nl-1)))')  — precedence is
    # (ap0/Nr)*(Nl-1), ported verbatim.
    ap = [inv(_mct(ap0 / Nr * (Nl - 1)))]
    bp = [inv(_mct(bp0 / Nr * (Nl - 1)))]

    pf = np.zeros((L, L))
    pb = np.zeros((L, L))
    pfb = np.zeros((L, L))
    for i in range(Nr):
        seg = X[i]
        efp = ap[0] @ seg[:, 1:]
        ebp = bp[0] @ seg[:, :-1]
        pf += efp @ efp.T
        pb += ebp @ ebp.T
        pfb += efp @ ebp.T

    En = _mct(En / N)                          # Cholesky factor of noise cov

    I_L = np.eye(L)
    a = ap
    b = bp
    for m in range(1, p + 1):
        # Reflection (parcor) coefficient for this order.
        ck = inv(_mct(pf)) @ pfb @ inv(cholesky(pb, lower=False))
        ef = I_L - ck @ ck.T
        eb = I_L - ck.T @ ck

        En = En @ _mct(ef)

        # armorf.m lines 68-69: extend the coefficient arrays with a zero
        # matrix at the new order (ap(:,:,m+1)=zeros) before recursing.
        a = a + [np.zeros((L, L))]
        b = b + [np.zeros((L, L))]

        inv_ef = inv(_mct(ef))
        inv_eb = inv(_mct(eb))
        a_new = [None] * (m + 1)
        b_new = [None] * (m + 1)
        for i in range(1, m + 2):              # i = 1..m+1  (1-indexed)
            a_new[i - 1] = inv_ef @ (a[i - 1] - ck @ b[m + 1 - i])
            b_new[i - 1] = inv_eb @ (b[i - 1] - ck.T @ a[m + 1 - i])

        pf = np.zeros((L, L))
        pb = np.zeros((L, L))
        pfb = np.zeros((L, L))
        for k in range(Nr):
            seg = X[k]
            width = Nl - m - 1
            efp = np.zeros((L, width))
            ebp = np.zeros((L, width))
            for i in range(1, m + 2):
                s0 = m + 2 - i                 # python start (forward)
                s1 = Nl - i + 1                # python end   (exclusive)
                efp += a_new[i - 1] @ seg[:, s0:s1]
                ebp += b_new[m + 1 - i] @ seg[:, s0 - 1:s1 - 1]
            pf += efp @ efp.T
            pb += ebp @ ebp.T
            pfb += efp @ ebp.T

        a = a_new
        b = b_new

    a0_inv = inv(a[0])
    A = np.zeros((L, L, p))
    for j in range(1, p + 1):
        # armorf returns coefficients in the A(L)x=e convention (the
        # returned matrices are -A_k); negate to the physical convention
        # x(t) = sum_k A_k x(t-k) + e used by spectral_transfer.
        A[:, :, j - 1] = -a0_inv @ a[j]
    Sigma = En @ En.T
    return A, Sigma


# ─────────────────────────────────────────────────────────────────────
# Model-order selection
# ─────────────────────────────────────────────────────────────────────
def order_criteria(X, max_order, min_order=1):
    """AIC and BIC across candidate AR orders for the trial ensemble.

    Uses the multivariate Akaike / Bayesian information criteria on the
    residual covariance determinant (Lutkepohl 2005):

        AIC(p) = ln|Sigma_p| + 2 p n^2 / M
        BIC(p) = ln|Sigma_p| + ln(M) p n^2 / M

    where ``n`` is the number of signals and ``M`` the effective number
    of observations (``Nr * (Nl - p)``).

    Returns
    -------
    orders : np.ndarray
    aic, bic : np.ndarray
        Criterion value per candidate order (lower is better).
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 2:
        X = X[np.newaxis]
    Nr, n, Nl = X.shape
    orders = np.arange(min_order, max_order + 1)
    aic = np.empty(orders.size)
    bic = np.empty(orders.size)
    for idx, p in enumerate(orders):
        _, Sigma = fit_mvar(X, p)
        sign, logdet = np.linalg.slogdet(Sigma)
        M = Nr * (Nl - p)
        k = p * n * n
        aic[idx] = logdet + 2.0 * k / M
        bic[idx] = logdet + np.log(M) * k / M
    return orders, aic, bic


# ─────────────────────────────────────────────────────────────────────
# Spectral transfer function
# ─────────────────────────────────────────────────────────────────────
def ar_transfer(A, freqs, fs):
    """Transfer function ``H(f) = (I - sum_k A_k e^{-i2pi f k/fs})^{-1}``.

    Returns
    -------
    H : np.ndarray, shape (n_freqs, n, n)  (complex)
    """
    A = np.asarray(A)
    n = A.shape[0]
    p = A.shape[2]
    freqs = np.asarray(freqs, dtype=float)
    ks = np.arange(1, p + 1)
    phase = np.exp(-1j * 2.0 * np.pi * np.outer(freqs, ks) / fs)  # (nf, p)
    Af = np.tile(np.eye(n, dtype=complex), (freqs.size, 1, 1))
    for kk in range(p):
        Af -= A[:, :, kk][None, :, :] * phase[:, kk][:, None, None]
    return np.linalg.inv(Af)


def spectral_transfer(A, Sigma, freqs, fs):
    """Transfer function ``H(f)`` and spectral matrix ``S(f) = H Sigma H^H``.

    Returns
    -------
    H : np.ndarray, shape (n_freqs, n, n)  (complex)
    S : np.ndarray, shape (n_freqs, n, n)  (complex; Hermitian per freq)
    """
    H = ar_transfer(A, freqs, fs)
    S = H @ Sigma.astype(complex) @ np.conj(np.transpose(H, (0, 2, 1)))
    return H, S


# ─────────────────────────────────────────────────────────────────────
# Bivariate (pairwise) spectral Granger causality
# ─────────────────────────────────────────────────────────────────────
def pairwise_spectral_gc(X, order, freqs, fs):
    """Bivariate Geweke spectral Granger causality for a signal pair.

    Parameters
    ----------
    X : np.ndarray
        ``(n_trials, 2, n_times)`` or ``(2, n_times)``.  Row 0 = x
        (seed), row 1 = y (target).
    order : int
    freqs : array_like  (Hz)
    fs : float  (Hz)

    Returns
    -------
    f_xy : np.ndarray, shape (n_freqs,)
        Spectral GC x -> y (seed -> target).
    f_yx : np.ndarray, shape (n_freqs,)
        Spectral GC y -> x (target -> seed).

    Notes
    -----
    This is a verified byte-faithful port of BSMART ``pwcausal.m`` +
    ``spectrum.m``: the ``1/fs`` scaling BSMART applies to ``S`` (and to
    the subtracted term) cancels in the log-ratio, so the values are
    identical to machine precision (see ``validate_granger.py``,
    ``test_matches_bsmart_pwcausal``).

    Geweke (1982) / Ding, Chen & Bressler (2006), Eqs. for the bivariate
    case.  For ``Sigma = [[s_xx, s_xy], [s_xy, s_yy]]``:

        f_{y->x}(w) = ln( S_xx / (S_xx - (s_yy - s_xy^2/s_xx) |H_xy|^2) )
        f_{x->y}(w) = ln( S_yy / (S_yy - (s_xx - s_xy^2/s_yy) |H_yx|^2) )

    The subtracted term removes the part of the target's power that is
    attributable to instantaneous (zero-lag) noise covariance, isolating
    the directional (lagged) influence.
    """
    A, Sigma = fit_mvar(X, order)
    H, S = spectral_transfer(A, Sigma, freqs, fs)

    s_xx = Sigma[0, 0]
    s_yy = Sigma[1, 1]
    s_xy = Sigma[0, 1]

    S_xx = np.real(S[:, 0, 0])
    S_yy = np.real(S[:, 1, 1])
    H_xy = H[:, 0, 1]
    H_yx = H[:, 1, 0]

    # y -> x
    term_yx = (s_yy - s_xy ** 2 / s_xx) * np.abs(H_xy) ** 2
    f_yx = np.log(S_xx / (S_xx - term_yx))
    # x -> y
    term_xy = (s_xx - s_xy ** 2 / s_yy) * np.abs(H_yx) ** 2
    f_xy = np.log(S_yy / (S_yy - term_xy))

    # Numerical guard: tiny negative arguments from round-off -> 0.
    f_xy = np.where(np.isfinite(f_xy), np.maximum(f_xy, 0.0), 0.0)
    f_yx = np.where(np.isfinite(f_yx), np.maximum(f_yx, 0.0), 0.0)
    return f_xy, f_yx


def time_reversed_pairwise_gc(X, order, freqs, fs):
    """Diff-TRGC for a signal pair (Haufe 2013; Winkler 2016).

    Computes net GC on the data and on its time-reversed copy, and
    returns the difference ``D = net(forward) - net(reversed)`` per
    direction.  A robust x->y influence should give ``D_xy > 0`` and
    ``D_yx < 0`` (and vice versa); artefacts from symmetric mixing /
    SNR asymmetry cancel because they are (to first order) invariant to
    time reversal.

    Returns
    -------
    d_xy, d_yx : np.ndarray, shape (n_freqs,)
        Diff-TRGC scores per direction.  ``d_xy = net_fwd_xy -
        net_rev_xy`` where ``net = f_xy - f_yx``; by construction
        ``d_xy = -d_yx``.
    """
    f_xy, f_yx = pairwise_spectral_gc(X, order, freqs, fs)
    Xr = np.flip(np.asarray(X, dtype=float), axis=-1)
    r_xy, r_yx = pairwise_spectral_gc(Xr, order, freqs, fs)
    net_fwd = f_xy - f_yx
    net_rev = r_xy - r_yx
    d_xy = net_fwd - net_rev
    return d_xy, -d_xy


# ─────────────────────────────────────────────────────────────────────
# Conditional (multivariate) Granger causality
# ─────────────────────────────────────────────────────────────────────
def time_domain_conditional_gc(X, order, pairs=None):
    """Time-domain conditional GC ``F_{src->tgt | rest}`` (Geweke 1984).

    ``F = ln( Sigma^reduced_{tgt,tgt} / Sigma^full_{tgt,tgt} )`` — the log
    ratio of the target's one-step prediction-error variance from the
    reduced model (all signals except ``src``) to the full model (all
    signals).  Exact and assumption-light; used as the anchor/validation
    for the spectral version.

    Parameters
    ----------
    X : np.ndarray, shape (n_trials, n_signals, n_times)
    order : int
    pairs : list of (src, tgt), optional
        Directed pairs to compute (default all ordered pairs).

    Returns
    -------
    dict {(src, tgt): float}
    """
    n = X.shape[1]
    _, S_full = fit_mvar(X, order)
    if pairs is None:
        pairs = [(s, t) for s in range(n) for t in range(n) if s != t]
    out = {}
    red_cache = {}
    for src, tgt in pairs:
        if src not in red_cache:
            keep = [v for v in range(n) if v != src]
            _, Sr = fit_mvar(X[:, keep, :], order)
            red_cache[src] = (keep, Sr)
        keep, Sr = red_cache[src]
        tr = keep.index(tgt)
        out[(src, tgt)] = float(np.log(Sr[tr, tr] / S_full[tgt, tgt]))
    return out


def _cond_spectral_pair(tgt, src, n, S_full, H_full, keep, S_red, G_red,
                        freqs):
    """Chen, Bressler & Ding (2006) spectral conditional GC for one pair.

    ``F_{src -> tgt | rest}(f)``, using the precomputed full model
    (``S_full``, ``H_full``) and the reduced model that omits ``src``
    (``keep``, ``S_red``, ``G_red``).  See conditional_spectral_gc.
    """
    z = [v for v in range(n) if v not in (tgt, src)]
    perm = [tgt] + z + [src]                    # [X, Z..., Y]
    nc = 1 + len(z)                             # size of X+Z block; Y is last
    Sf = S_full[np.ix_(perm, perm)]
    Scc = Sf[:nc, :nc]
    Syc = Sf[nc:, :nc]
    # Transform P removes the Y-innovation's instantaneous correlation
    # with the (X,Z) innovations -> Sigma_hat is block-diagonal.
    B = -Syc @ inv(Scc)
    P = np.eye(n)
    P[nc:, :nc] = B
    Pinv = inv(P)
    Shat = P @ Sf @ P.T                         # block-diag [Scc, Schur(Y)]

    # Reduced transfer/noise reordered to [X, Z...] (drop Y).
    red_order = [tgt] + z
    pos = {v: i for i, v in enumerate(keep)}
    rperm = [pos[v] for v in red_order]
    Gamma = S_red[np.ix_(rperm, rperm)]
    Gr = G_red[:, rperm][:, :, rperm]           # (nf, nc, nc)

    Hf = H_full[:, perm][:, :, perm]            # (nf, n, n)
    Scc_c = Scc.astype(complex)
    Shat_c = Shat.astype(complex)
    eye1 = np.eye(n - nc)
    f_arr = np.empty(freqs.size)
    for fi in range(freqs.size):
        Hhat = Hf[fi] @ Pinv
        Ghat = np.zeros((n, n), dtype=complex)
        Ghat[:nc, :nc] = Gr[fi]
        Ghat[nc:, nc:] = eye1
        Q = inv(Ghat) @ Hhat
        q0 = Q[0]                               # X row
        total = np.real(q0 @ Shat_c @ q0.conj())
        intrinsic = np.real(q0[:nc] @ Scc_c @ q0[:nc].conj())
        f_arr[fi] = np.log(total / intrinsic)
    return np.where(np.isfinite(f_arr), np.maximum(f_arr, 0.0), 0.0)


def conditional_spectral_gc(X, order, freqs, fs, pairs=None):
    """Frequency-resolved conditional GC ``F_{src->tgt | rest}(f)``.

    Multivariate (conditional) Geweke spectral Granger causality: the
    influence of ``src`` on ``tgt`` after accounting for every other
    signal.  This removes common-input and mediated/indirect pathways
    that inflate pairwise GC — the "direct influence" measure.

    Method: Chen, Bressler & Ding (2006), *J. Neurosci. Methods* 150:228
    ("Frequency decomposition of conditional Granger causality").  Fits
    the full MVAR once and one reduced MVAR per source; each pair's GC is
    obtained by the partition-matrix normalization (no per-pair refit of
    reduced models beyond the per-source one).

    Parameters
    ----------
    X : np.ndarray, shape (n_trials, n_signals, n_times)
    order : int
    freqs : array_like  (Hz)
    fs : float  (Hz)
    pairs : list of (src, tgt), optional
        Directed pairs (default all ordered pairs).

    Returns
    -------
    dict {(src, tgt): np.ndarray of shape (n_freqs,)}
    """
    n = X.shape[1]
    freqs = np.asarray(freqs, dtype=float)
    A_full, S_full = fit_mvar(X, order)
    H_full = ar_transfer(A_full, freqs, fs)
    if pairs is None:
        pairs = [(s, t) for s in range(n) for t in range(n) if s != t]

    red_cache = {}
    out = {}
    for src, tgt in pairs:
        if src not in red_cache:
            keep = [v for v in range(n) if v != src]
            A_red, S_red = fit_mvar(X[:, keep, :], order)
            G_red = ar_transfer(A_red, freqs, fs)
            red_cache[src] = (keep, S_red, G_red)
        keep, S_red, G_red = red_cache[src]
        out[(src, tgt)] = _cond_spectral_pair(
            tgt, src, n, S_full, H_full, keep, S_red, G_red, freqs,
        )
    return out


def moving_window_conditional_gc(X, order, freqs, fs, win_samples, step=1,
                                 pairs=None):
    """Sliding-window conditional spectral GC for the requested pairs.

    Returns
    -------
    result : dict
        ``gc`` : dict {(src, tgt): (n_freqs, n_windows)}.
        ``win_start`` : (n_windows,) window start sample indices.
    """
    X = np.asarray(X, dtype=float)
    n_times = X.shape[2]
    freqs = np.asarray(freqs, dtype=float)
    starts = np.arange(0, n_times - win_samples + 1, step)
    n = X.shape[1]
    if pairs is None:
        pairs = [(s, t) for s in range(n) for t in range(n) if s != t]

    gc = {p: np.empty((freqs.size, starts.size)) for p in pairs}
    for w, s in enumerate(starts):
        seg = X[:, :, s:s + win_samples]
        res = conditional_spectral_gc(seg, order, freqs, fs, pairs=pairs)
        for p in pairs:
            gc[p][:, w] = res[p]
    return {'gc': gc, 'win_start': starts}


# ─────────────────────────────────────────────────────────────────────
# Moving-window driver (matches BSMART mov_bi_ga)
# ─────────────────────────────────────────────────────────────────────
def moving_window_pairwise_gc(X, order, freqs, fs, win_samples, step=1,
                              trgc=False):
    """Sliding-window bivariate spectral GC across time.

    For each window position the AR model is fit on the trial ensemble
    restricted to that window, and spectral GC is evaluated at ``freqs``.
    This mirrors BSMART ``mov_bi_ga(dat, start, end, win, order, fs, f)``,
    whose window slides one sample at a time over the trial ensemble.

    Parameters
    ----------
    X : np.ndarray, shape (n_trials, 2, n_times)
        Row 0 = x (seed), row 1 = y (target).
    order : int
    freqs : array_like  (Hz)
    fs : float  (Hz)
    win_samples : int
        Window length in samples (e.g. 20 for 40 ms at 500 Hz).
    step : int
        Window step in samples (BSMART uses 1).
    trgc : bool
        If True, also return Diff-TRGC per window.

    Returns
    -------
    result : dict
        ``f_xy``, ``f_yx`` : (n_freqs, n_windows) spectral GC.
        ``win_start`` : (n_windows,) window start sample indices.
        If ``trgc``: ``d_xy`` : (n_freqs, n_windows) Diff-TRGC (x->y).
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 3 or X.shape[1] != 2:
        raise ValueError('X must be (n_trials, 2, n_times) for pairwise GC')
    n_times = X.shape[2]
    freqs = np.asarray(freqs, dtype=float)
    starts = np.arange(0, n_times - win_samples + 1, step)
    n_win = starts.size
    n_f = freqs.size

    f_xy = np.empty((n_f, n_win))
    f_yx = np.empty((n_f, n_win))
    d_xy = np.empty((n_f, n_win)) if trgc else None

    for w, s in enumerate(starts):
        seg = X[:, :, s:s + win_samples]
        fxy, fyx = pairwise_spectral_gc(seg, order, freqs, fs)
        f_xy[:, w] = fxy
        f_yx[:, w] = fyx
        if trgc:
            dxy, _ = time_reversed_pairwise_gc(seg, order, freqs, fs)
            d_xy[:, w] = dxy

    result = {'f_xy': f_xy, 'f_yx': f_yx, 'win_start': starts}
    if trgc:
        result['d_xy'] = d_xy
    return result


# ─────────────────────────────────────────────────────────────────────
# Band averaging  (matches production_pwgc_data_to_python.m)
# ─────────────────────────────────────────────────────────────────────
DEFAULT_BANDS = {
    'theta': (4.0, 7.0),
    'alpha': (8.0, 12.0),
    'low_beta': (13.0, 20.0),
    'high_beta': (21.0, 30.0),
}


# ─────────────────────────────────────────────────────────────────────
# ROI vertex reduction — one virtual channel per ROI via a fixed filter
# ─────────────────────────────────────────────────────────────────────
def reduce_roi_first_pc(vertex_data, return_filter=False):
    """Collapse an ROI's vertex time courses to one virtual channel.

    The spatial filter is the ROI's first principal component, estimated
    **once** from the whole trial ensemble (vertices x concatenated
    trials/time), then applied to every trial.  Because the filter is
    fixed across trials, all epochs become realizations of the *same*
    virtual channel — the consistency the multi-trial AR fit requires
    (unlike per-epoch ``pca_flip``).  It is the vertex-weighted analogue
    of the sensor pseudo-channels used in the MATLAB pipeline, and the
    connectivity-literature standard (PCA aggregation; Pellegrini et al.
    2023).

    Parameters
    ----------
    vertex_data : np.ndarray, shape (n_epochs, n_vertices, n_times)
    return_filter : bool
        Also return the spatial filter weights.

    Returns
    -------
    vc : np.ndarray, shape (n_epochs, n_times)
        Virtual-channel time course.  (GC is invariant to per-channel
        scaling, so the filter is left unit-norm and unscaled.)
    w : np.ndarray, shape (n_vertices,)   (only if return_filter)

    Notes
    -----
    Vertices are mean-centered over the ensemble before the SVD (standard
    PCA).  The projected virtual channel is returned without re-centering;
    the GC front-end demeans each window/trial ensemble as needed.
    """
    X = np.asarray(vertex_data, dtype=float)
    if X.ndim != 3:
        raise ValueError('vertex_data must be (n_epochs, n_vertices, n_times)')
    n_ep, n_v, n_t = X.shape
    if n_v == 1:
        vc = X[:, 0, :]
        return (vc, np.ones(1)) if return_filter else vc

    # (n_vertices, n_epochs * n_times), mean-centered per vertex.
    M = np.transpose(X, (1, 0, 2)).reshape(n_v, n_ep * n_t)
    M = M - M.mean(axis=1, keepdims=True)
    U, _, _ = np.linalg.svd(M, full_matrices=False)
    w = U[:, 0]
    # Deterministic sign: largest-magnitude loading positive.
    if w[np.argmax(np.abs(w))] < 0:
        w = -w
    vc = np.einsum('v,evt->et', w, X)
    return (vc, w) if return_filter else vc


def band_average(gc, freqs, bands=None):
    """Average a frequency-resolved GC array into named bands.

    Parameters
    ----------
    gc : np.ndarray, shape (n_freqs, ...)
        GC with frequency on axis 0.
    freqs : array_like, shape (n_freqs,)
    bands : dict {name: (fmin, fmax)}, optional
        Inclusive frequency bounds (defaults to the study's
        theta/alpha/low-beta/high-beta scheme).

    Returns
    -------
    dict {name: np.ndarray of shape gc.shape[1:]}
    """
    if bands is None:
        bands = DEFAULT_BANDS
    freqs = np.asarray(freqs, dtype=float)
    out = {}
    for name, (fmin, fmax) in bands.items():
        mask = (freqs >= fmin) & (freqs <= fmax)
        if not mask.any():
            raise ValueError(f'No frequencies fall in band {name} '
                             f'[{fmin}, {fmax}] Hz')
        out[name] = gc[mask].mean(axis=0)
    return out
