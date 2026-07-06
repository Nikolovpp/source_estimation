"""
Validation for conditional (multivariate) Granger causality in granger.py.

Ground-truth network tests: conditional GC must remove indirect
(mediated) and common-input pathways that inflate pairwise GC, while
preserving genuine direct influence.

Run:  python validate_granger_conditional.py
"""
import numpy as np

from granger import (
    pairwise_spectral_gc, conditional_spectral_gc,
    time_domain_conditional_gc,
)

PASS = '\033[92mPASS\033[0m'
FAIL = '\033[91mFAIL\033[0m'
_results = []


def check(name, cond, detail=''):
    _results.append(bool(cond))
    print(f'  [{PASS if cond else FAIL}] {name}' + (f'  — {detail}' if detail else ''))


def simulate(coupling, n_tr, T, burn=400, seed=0, a=0.5):
    """Simulate an n-variable AR(1)-style network.

    coupling : list of (src, tgt, weight) added as weight*var[src](t-1).
    Returns (n_tr, n, T); each var also has AR(1) self-term a and unit noise.
    """
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


def _mean_cond(X, src, tgt):
    d = conditional_spectral_gc(X, order=4, freqs=FREQS, fs=FS,
                                pairs=[(src, tgt)])
    return d[(src, tgt)].mean()


def _mean_pair(X, i, j):
    # pairwise GC i->j from the 2-variable submodel
    f_xy, f_yx = pairwise_spectral_gc(X[:, [i, j], :], order=4,
                                      freqs=FREQS, fs=FS)
    return f_xy.mean()  # i->j


def test_chain_mediation():
    print('\n== Mediation chain X->Z->Y: conditional kills indirect X->Y ==')
    # 0=X, 1=Z, 2=Y ;  X->Z->Y (no direct X->Y, no Y->X)
    X = simulate([(0, 1, 0.5), (1, 2, 0.5)], n_tr=80, T=500, seed=1)
    pair_xy = _mean_pair(X, 0, 2)
    cond_xy = _mean_cond(X, 0, 2)
    check('pairwise X->Y is clearly nonzero (indirect)', pair_xy > 0.02,
          f'pairwise={pair_xy:.4f}')
    check('conditional X->Y|Z collapses to ~0', cond_xy < 0.2 * pair_xy,
          f'cond={cond_xy:.4f} vs pairwise={pair_xy:.4f}')
    check('conditional Y->X|Z ~0 (no true Y->X)', _mean_cond(X, 2, 0) < 0.01,
          f'cond Y->X={_mean_cond(X, 2, 0):.4f}')


def test_common_driver():
    print('\n== Common driver (asymmetric lag): conditional kills spurious X->Y ==')
    # 0=X, 1=Y, 2=Z, 3=M ; Z->X (lag1) and Z->M->Y so Y sees Z one lag later.
    # X thus leads Y (both driven by Z at different lags) -> spurious pairwise
    # X->Y, which conditioning on {Z, M} must remove.
    X = simulate([(2, 0, 0.6), (2, 3, 0.6), (3, 1, 0.6)], n_tr=80, T=500, seed=2)
    pair_xy = _mean_pair(X, 0, 1)
    cond_xy = _mean_cond(X, 0, 1)      # X->Y | {Z, M}
    check('pairwise X->Y spuriously nonzero', pair_xy > 0.02,
          f'pairwise={pair_xy:.4f}')
    check('conditional X->Y|rest ~0 (common input removed)',
          cond_xy < 0.2 * pair_xy, f'cond={cond_xy:.4f} vs pairwise={pair_xy:.4f}')
    check('conditional Y->X|rest ~0', _mean_cond(X, 1, 0) < 0.02,
          f'cond Y->X={_mean_cond(X, 1, 0):.4f}')


def test_direct_survives():
    print('\n== Direct X->Y (plus indirect via Z): conditional preserved ==')
    # 0=X,1=Z,2=Y ; X->Z->Y AND direct X->Y
    X = simulate([(0, 1, 0.5), (1, 2, 0.5), (0, 2, 0.5)], n_tr=80, T=500, seed=3)
    cond_xy = _mean_cond(X, 0, 2)
    cond_yx = _mean_cond(X, 2, 0)
    check('conditional X->Y|Z stays clearly positive (direct path)',
          cond_xy > 0.05, f'cond X->Y={cond_xy:.4f}')
    check('conditional Y->X|Z ~0 (no Y->X)', cond_yx < 0.02,
          f'cond Y->X={cond_yx:.4f}')


def test_degenerate_equals_pairwise():
    print('\n== 2-variable case: conditional == pairwise ==')
    X = simulate([(0, 1, 0.4)], n_tr=80, T=500, seed=4)  # only X,Y
    cond_xy = _mean_cond(X, 0, 1)
    pair_xy = _mean_pair(X, 0, 1)
    check('conditional X->Y == pairwise X->Y (no conditioning set)',
          abs(cond_xy - pair_xy) < 1e-6, f'cond={cond_xy:.4f}, pair={pair_xy:.4f}')


def test_time_domain_anchor():
    print('\n== Time-domain conditional GC anchor (chain) ==')
    X = simulate([(0, 1, 0.5), (1, 2, 0.5)], n_tr=80, T=500, seed=1)
    td = time_domain_conditional_gc(X, order=4, pairs=[(0, 2), (2, 0)])
    check('time-domain conditional X->Y|Z ~0 (indirect removed)',
          td[(0, 2)] < 0.02, f'td X->Y={td[(0, 2)]:.4f}')
    check('time-domain conditional is non-negative',
          td[(0, 2)] >= -1e-9 and td[(2, 0)] >= -1e-9)


if __name__ == '__main__':
    test_chain_mediation()
    test_common_driver()
    test_direct_survives()
    test_degenerate_equals_pairwise()
    test_time_domain_anchor()

    n_pass = sum(_results)
    n_tot = len(_results)
    print(f'\n{"="*50}\n{n_pass}/{n_tot} conditional-GC checks passed')
    raise SystemExit(0 if n_pass == n_tot else 1)
