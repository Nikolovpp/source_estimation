"""
Validation for state-space GC (granger_statespace.py) — Barnett & Seth 2015.

Checks the DARE-based conditional GC against ground-truth networks, its
agreement with the classic Chen (2006) method, the degenerate reduction
to pairwise GC, and the Geweke spectral-integral identity.

Run:  python validate_granger_statespace.py
"""
import numpy as np

from granger import pairwise_spectral_gc, conditional_spectral_gc
from granger_statespace import ss_conditional_gc, statespace_conditional_gc

PASS = '\033[92mPASS\033[0m'
FAIL = '\033[91mFAIL\033[0m'
_results = []


def check(name, cond, detail=''):
    _results.append(bool(cond))
    print(f'  [{PASS if cond else FAIL}] {name}' + (f'  — {detail}' if detail else ''))


def simulate(coupling, n_tr, T, burn=400, seed=0, a=0.5):
    n = 1 + max(max(s, t) for s, t, _ in coupling)
    rng = np.random.default_rng(seed)
    out = np.empty((n_tr, n, T))
    for tr in range(n_tr):
        TT = T + burn
        v = np.zeros((n, TT))
        e = rng.standard_normal((n, TT))
        for t in range(1, TT):
            v[:, t] = a * v[:, t - 1] + e[:, t]
            for s, tg, w in coupling:
                v[tg, t] += w * v[s, t - 1]
        out[tr] = v[:, burn:]
    return out


FS = 200.0
FREQS = np.arange(1, 90)
ORDER = 4


def test_chain_mediation():
    print('\n== SS conditional kills indirect (chain X->Z->Y) ==')
    X = simulate([(0, 1, 0.5), (1, 2, 0.5)], 80, 500, seed=1)
    A, SIG = _fit(X)
    Ft, fs_ = ss_conditional_gc(A, SIG, x=2, y=0, freqs=FREQS, fs=FS)  # X->Y|Z
    check('time-domain X->Y|Z ~0', Ft < 0.02, f'F_time={Ft:.4f}')
    check('spectral X->Y|Z mean ~0', fs_.mean() < 0.02, f'mean f={fs_.mean():.4f}')


def test_direct_survives():
    print('\n== SS conditional preserves direct X->Y ==')
    X = simulate([(0, 1, 0.5), (1, 2, 0.5), (0, 2, 0.5)], 80, 500, seed=3)
    A, SIG = _fit(X)
    Ft, fs_ = ss_conditional_gc(A, SIG, x=2, y=0, freqs=FREQS, fs=FS)
    check('time-domain X->Y|Z clearly > 0', Ft > 0.05, f'F_time={Ft:.4f}')
    check('spectral X->Y|Z clearly > 0', fs_.mean() > 0.03, f'mean f={fs_.mean():.4f}')


def test_agreement_with_chen():
    print('\n== SS spectral agrees with Chen (2006) on well-specified VAR ==')
    X = simulate([(0, 1, 0.5), (1, 2, 0.5), (0, 2, 0.5)], 120, 800, seed=5)
    A, SIG = _fit(X)
    _, ss_spec = ss_conditional_gc(A, SIG, x=2, y=0, freqs=FREQS, fs=FS)
    chen = conditional_spectral_gc(X, ORDER, FREQS, FS, pairs=[(0, 2)])[(0, 2)]
    rel = abs(ss_spec.mean() - chen.mean()) / max(chen.mean(), 1e-6)
    check('SS and Chen conditional GC agree within 15%', rel < 0.15,
          f'SS={ss_spec.mean():.4f}, Chen={chen.mean():.4f}, rel={rel:.3f}')


def test_degenerate_pairwise():
    print('\n== SS unconditional (2-var) == pairwise Geweke GC ==')
    X = simulate([(0, 1, 0.4)], 100, 600, seed=4)
    A, SIG = _fit(X)
    _, ss_spec = ss_conditional_gc(A, SIG, x=1, y=0, freqs=FREQS, fs=FS)  # X->Y, no Z
    f_xy, _ = pairwise_spectral_gc(X, ORDER, FREQS, FS)                    # X->Y
    md = np.abs(ss_spec - f_xy).max()
    check('SS unconditional spectral == pairwise to ~1e-8', md < 1e-8,
          f'max|diff|={md:.2e}')


def test_geweke_integral():
    print('\n== Geweke identity: mean spectral GC over [0, fs/2] ~ time-domain ==')
    X = simulate([(0, 1, 0.5), (1, 2, 0.5), (0, 2, 0.5)], 120, 800, seed=6)
    A, SIG = _fit(X)
    freqs = np.linspace(0, FS / 2, 257)          # uniform over [0, pi]
    Ft, fs_ = ss_conditional_gc(A, SIG, x=2, y=0, freqs=freqs, fs=FS)
    rel = abs(fs_.mean() - Ft) / max(Ft, 1e-6)
    check('spectral mean ~ time-domain GC (within 12%)', rel < 0.12,
          f'spec_mean={fs_.mean():.4f}, F_time={Ft:.4f}, rel={rel:.3f}')


def _fit(X):
    from granger import fit_mvar
    return fit_mvar(X, ORDER)


def test_driver_wrapper():
    print('\n== statespace_conditional_gc driver (all pairs) ==')
    X = simulate([(0, 1, 0.5), (1, 2, 0.5)], 80, 500, seed=1)
    res = statespace_conditional_gc(X, ORDER, freqs=FREQS, fs=FS,
                                    pairs=[(0, 2), (2, 0)])
    check('driver returns (F_time, spec) per pair',
          isinstance(res[(0, 2)], tuple) and res[(0, 2)][1].shape == FREQS.shape)


if __name__ == '__main__':
    test_chain_mediation()
    test_direct_survives()
    test_agreement_with_chen()
    test_degenerate_pairwise()
    test_geweke_integral()
    test_driver_wrapper()

    n_pass = sum(_results)
    n_tot = len(_results)
    print(f'\n{"="*50}\n{n_pass}/{n_tot} state-space GC checks passed')
    raise SystemExit(0 if n_pass == n_tot else 1)
