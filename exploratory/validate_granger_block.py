"""
Validation for the FIXPC-k block path in granger.py / run_granger.py.

Checks that (1) the block formula collapses to the verified bivariate BSMART
formula for 1x1 blocks, (2) ``reduce_roi_top_pcs`` reproduces
``reduce_roi_first_pc`` for k=1 and caps at the ROI rank, (3) block GC
recovers a simulated block-to-block direction with the right TRGC sign,
(4) block GC is invariant to a nonsingular transform within a block (so the
PC rotation/sign convention cannot matter), (5) ``compute_subject_gc``
with ``n_pcs=1`` is byte-identical to the pre-FIXPC path, and (6) a FIXPC4
result saved by ``save_subject_gc`` lands on the ``_pc4`` path and loads
through ``granger_stats.load_gc_group``.

Run:  python exploratory/validate_granger_block.py
"""
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np

from granger import (
    fit_mvar, spectral_transfer, _block_gc_direction,
    pairwise_spectral_gc, time_reversed_pairwise_gc, block_spectral_gc,
    time_reversed_block_gc, moving_window_pairwise_gc,
    reduce_roi_first_pc, reduce_roi_top_pcs,
)
from run_granger import compute_subject_gc, gc_tag

PASS = '\033[92mPASS\033[0m'
FAIL = '\033[91mFAIL\033[0m'
_results = []


def check(name, cond, detail=''):
    _results.append(bool(cond))
    print(f'  [{PASS if cond else FAIL}] {name}' + (f'  — {detail}' if detail else ''))


def simulate_block_var(n_trials, n_times, c=0.4, burn=200, seed=0):
    """Two 2-channel blocks, x-block -> y-block only (VAR(1))."""
    rng = np.random.default_rng(seed)
    A = np.zeros((4, 4))
    A[0, 0] = A[1, 1] = A[2, 2] = A[3, 3] = 0.5
    A[0, 1] = 0.2                       # within-x coupling
    A[3, 2] = -0.2                      # within-y coupling
    A[2, 0] = c                         # x0 -> y0
    A[3, 1] = c                         # x1 -> y1
    out = np.empty((n_trials, 4, n_times))
    for tr in range(n_trials):
        T = n_times + burn
        z = np.zeros((4, T))
        e = rng.standard_normal((4, T))
        for t in range(1, T):
            z[:, t] = A @ z[:, t - 1] + e[:, t]
        out[tr] = z[:, burn:]
    return out


def test_scalar_blocks_match_pairwise():
    print('\n[1] 1x1 blocks reproduce the bivariate formula')
    rng = np.random.default_rng(1)
    X = simulate_block_var(30, 300, seed=3)[:, [0, 2], :]
    freqs = np.arange(1, 31)
    fs = 200.0
    p_xy, p_yx = pairwise_spectral_gc(X, 4, freqs, fs)
    b_xy, b_yx = block_spectral_gc(X, 4, freqs, fs, ([0], [1]))
    err = max(np.max(np.abs(p_xy - b_xy)), np.max(np.abs(p_yx - b_yx)))
    check('block == pairwise for scalar blocks', err < 1e-12, f'max |diff| {err:.2e}')
    pd, _ = time_reversed_pairwise_gc(X, 4, freqs, fs)
    bd, _ = time_reversed_block_gc(X, 4, freqs, fs, ([0], [1]))
    err = np.max(np.abs(pd - bd))
    check('block TRGC == pairwise TRGC for scalar blocks', err < 1e-12, f'max |diff| {err:.2e}')


def test_reduction():
    print('\n[2] reduce_roi_top_pcs')
    rng = np.random.default_rng(2)
    n_ep, n_v, n_t = 20, 40, 150
    src = rng.standard_normal((n_ep, 3, n_t))
    W = rng.standard_normal((n_v, 3))
    vdata = np.einsum('vk,ekt->evt', W, src) + 0.05 * rng.standard_normal((n_ep, n_v, n_t))
    vc1 = reduce_roi_first_pc(vdata)
    top1 = reduce_roi_top_pcs(vdata, 1)
    check('k=1 equals reduce_roi_first_pc', top1.shape == (n_ep, 1, n_t)
          and np.array_equal(top1[:, 0, :], vc1))
    top4, Wt = reduce_roi_top_pcs(vdata, 4, return_filter=True)
    check('k=4 gives 4 orthonormal filters', top4.shape == (n_ep, 4, n_t)
          and np.allclose(Wt.T @ Wt, np.eye(4), atol=1e-10))
    check('first component of k=4 equals k=1', np.allclose(top4[:, 0, :], vc1))
    # rank-1 ROI (collapsed beamformer): one global time course on every vertex
    g = rng.standard_normal((n_ep, 1, n_t))
    collapsed = np.einsum('v,ekt->evt', rng.standard_normal(n_v), g)
    capped = reduce_roi_top_pcs(collapsed, 4)
    check('rank-1 ROI is capped to 1 component', capped.shape[1] == 1,
          f'shape {capped.shape}')
    one_v = reduce_roi_top_pcs(vdata[:, :1, :], 3)
    check('single-vertex ROI passes through', np.array_equal(one_v[:, 0, :], vdata[:, 0, :]))


def test_block_direction_and_trgc():
    print('\n[3] block GC recovers x-block -> y-block')
    X = simulate_block_var(60, 400, c=0.4, seed=5)
    freqs = np.arange(1, 31)
    fs = 200.0
    blk = ([0, 1], [2, 3])
    f_xy, f_yx = block_spectral_gc(X, 2, freqs, fs, blk)
    check('mean f_xy >> mean f_yx', f_xy.mean() > 5 * max(f_yx.mean(), 1e-6),
          f'{f_xy.mean():.3f} vs {f_yx.mean():.3f}')
    check('all values finite and >= 0', np.all(np.isfinite(f_xy)) and np.all(f_xy >= 0))
    d_xy, d_yx = time_reversed_block_gc(X, 2, freqs, fs, blk)
    check('TRGC d_xy > 0 and d_yx = -d_xy', d_xy.mean() > 0 and np.allclose(d_xy, -d_yx),
          f'mean d_xy {d_xy.mean():.3f}')
    X0 = simulate_block_var(60, 400, c=0.0, seed=6)
    f0_xy, f0_yx = block_spectral_gc(X0, 2, freqs, fs, blk)
    check('no coupling -> both directions small', max(f0_xy.mean(), f0_yx.mean()) < 0.05,
          f'{f0_xy.mean():.4f}, {f0_yx.mean():.4f}')


def test_within_block_invariance():
    print('\n[4] invariance to a nonsingular transform within a block')
    rng = np.random.default_rng(7)
    X = simulate_block_var(40, 300, seed=8)
    freqs = np.arange(1, 31)
    fs = 200.0
    blk = ([0, 1], [2, 3])
    T = np.eye(4)
    T[:2, :2] = rng.standard_normal((2, 2)) + 2 * np.eye(2)   # mix x block
    T[2:, 2:] = [[0, -1], [1, 0]]                              # rotate y block
    Ti = np.linalg.inv(T)

    # (a) The FORMULA is exactly invariant: transform one fitted VAR
    #     analytically (A_k -> T A_k T^-1, Sigma -> T Sigma T^T).
    A, S = fit_mvar(X, 3)
    At = np.stack([T @ A[:, :, k] @ Ti for k in range(A.shape[2])], axis=2)
    St = T @ S @ T.T
    H, Sp = spectral_transfer(A, S, freqs, fs)
    Ht, Spt = spectral_transfer(At, St, freqs, fs)
    f1 = _block_gc_direction(H, Sp, S, blk[0], blk[1])
    f2 = _block_gc_direction(Ht, Spt, St, blk[0], blk[1])
    err = np.max(np.abs(f1 - f2))
    check('formula exactly invariant on a transformed model', err < 1e-12,
          f'max |diff| {err:.2e}')

    # (b) End to end the residual is the Morf recursion itself: armorf is
    #     only approximately equivariant under channel mixing (its A differs
    #     by ~1e-5 after refitting T X), which propagates to ~1e-6 in GC.
    #     The verified bivariate path shows the same under a rotation, so
    #     this is a property of the shared fit, not of the block formula.
    f_xy, f_yx = block_spectral_gc(X, 3, freqs, fs, blk)
    Xt = np.einsum('ij,ejt->eit', T, X)
    g_xy, g_yx = block_spectral_gc(Xt, 3, freqs, fs, blk)
    err = max(np.max(np.abs(f_xy - g_xy)), np.max(np.abs(f_yx - g_yx)))
    check('end-to-end invariant to fit round-off', err < 1e-4,
          f'max |diff| {err:.2e} (GC scale {f_xy.mean():.2f})')


def test_moving_window_blocks():
    print('\n[5] moving_window_pairwise_gc with blocks')
    X = simulate_block_var(40, 200, seed=9)
    freqs = np.arange(1, 31)
    res = moving_window_pairwise_gc(X, order=2, freqs=freqs, fs=200.0,
                                    win_samples=40, step=10, trgc=True,
                                    diagnostics=True, blocks=([0, 1], [2, 3]))
    n_win = res['win_start'].size
    check('shapes', res['f_xy'].shape == (30, n_win) and res['d_xy'].shape == (30, n_win)
          and res['rho'].shape == (n_win,), f'{n_win} windows')
    check('direction per window', np.nanmean(res['f_xy']) > np.nanmean(res['f_yx']))
    check('diagnostics finite', np.all(np.isfinite(res['rho'])) and np.all(np.isfinite(res['cons'])))
    try:
        moving_window_pairwise_gc(X, 2, freqs, 200.0, 40, blocks=([0, 1], [2]))
        check('bad blocks rejected', False)
    except ValueError:
        check('bad blocks rejected', True)


def test_compute_subject_gc():
    print('\n[6] compute_subject_gc n_pcs plumbing')
    rng = np.random.default_rng(10)
    n_ep, n_t, fs = 30, 400, 200.0
    src = simulate_block_var(n_ep, n_t, c=0.5, seed=11)           # (ep, 4, t)
    times = np.arange(n_t) / fs - 0.5
    roi = {}
    for name, cols in (('A', [0, 1]), ('B', [2, 3])):
        W = rng.standard_normal((25, 2))
        roi[name] = (np.einsum('vk,ekt->evt', W, src[:, cols, :])
                     + 0.05 * rng.standard_normal((n_ep, 25, n_t)))
    kw = dict(order=2, win_ms=200, target_fs=fs, step=10, normalize='none',
              trgc=True, n_jobs=1)
    r_default = compute_subject_gc(roi, times, fs, **kw)
    r_pc1 = compute_subject_gc(roi, times, fs, n_pcs=1, **kw)
    same = all(np.array_equal(r_default['fxy'][b], r_pc1['fxy'][b])
               and np.array_equal(r_default['dtrgc'][b], r_pc1['dtrgc'][b])
               for b in r_default['fxy'])
    check('n_pcs=1 identical to default path', same)
    check('n_pcs/n_comp recorded', r_pc1['n_pcs'] == 1 and list(r_pc1['n_comp']) == [1, 1])
    r_pc2 = compute_subject_gc(roi, times, fs, n_pcs=2, **kw)
    check('n_pcs=2 shapes', all(r_pc2['fxy'][b].shape == r_pc1['fxy'][b].shape
                                for b in r_pc2['fxy']) and list(r_pc2['n_comp']) == [2, 2])
    hb1 = np.nanmean(r_pc1['dtrgc']['high_beta'])
    hb2 = np.nanmean(r_pc2['dtrgc']['high_beta'])
    check('A->B TRGC positive at PC1 and PC2', hb1 > 0 and hb2 > 0,
          f'pc1 {hb1:.3f}, pc2 {hb2:.3f}')
    try:
        compute_subject_gc(roi, times, fs, n_pcs=2, gc_mode='conditional', **kw)
        check('conditional + n_pcs>1 rejected', False)
    except ValueError:
        check('conditional + n_pcs>1 rejected', True)
    check('gc_tag unchanged for pc1, suffixed for pc4',
          gc_tag(10, 80, 200, 'none') == 'order10_win80ms_fs200'
          and gc_tag(10, 80, 200, 'none', n_pcs=4) == 'order10_win80ms_fs200_pc4'
          and gc_tag(10, 80, 200, 'zscore', n_pcs=4) == 'order10_win80ms_fs200_zscore_pc4')


def test_io_roundtrip():
    print('\n[7] save_subject_gc at n_pcs=4 -> granger_stats.load_gc_group')
    import tempfile
    from pathlib import Path
    from run_granger import save_subject_gc, subject_out_path
    from granger_stats import load_gc_group
    rng = np.random.default_rng(12)
    n_ep, n_t, fs = 24, 300, 200.0
    src = simulate_block_var(n_ep, n_t, c=0.5, seed=13)
    times = np.arange(n_t) / fs - 0.5
    roi = {}
    for name, cols in (('awfa-lh', [0, 1]), ('tpc-lh', [2, 3])):
        W = rng.standard_normal((30, 2))
        roi[name] = (np.einsum('vk,ekt->evt', W, src[:, cols, :])
                     + 0.05 * rng.standard_normal((n_ep, 30, n_t)))
    res = compute_subject_gc(roi, times, fs, order=2, win_ms=200, target_fs=fs,
                             step=10, normalize='none', trgc=True, n_jobs=1,
                             n_pcs=4)
    check('4 components kept per full-rank ROI', list(res['n_comp']) == [4, 4])
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        args = ('EEGPROD4001', 'overtProd', 'prodDiff', 'LCMV', 'custom',
                'vertex_selectkbest', True, 2, 200.0, fs, 'none')
        out = save_subject_gc(res, *args, output_root=root,
                              roi_subset=['awfa-lh', 'tpc-lh'], n_pcs=4)
        expect = subject_out_path(*args, output_root=root,
                                  roi_subset=['awfa-lh', 'tpc-lh'], n_pcs=4)
        check('written where subject_out_path says', out == expect
              and 'order2_win200ms_fs200_pc4' in str(out), str(out.relative_to(root)))
        d = np.load(out, allow_pickle=True)
        check('n_pcs / n_comp stored', int(d['n_pcs']) == 4 and list(d['n_comp']) == [4, 4])
        g = load_gc_group(out.parent)
        check('load_gc_group reads it (dtrgc present)', 'dtrgc' in g
              and g['dtrgc']['high_beta'].shape == (1,) + res['dtrgc']['high_beta'].shape
              and np.allclose(g['dtrgc']['high_beta'][0], res['dtrgc']['high_beta'],
                              equal_nan=True))


if __name__ == '__main__':
    test_scalar_blocks_match_pairwise()
    test_reduction()
    test_block_direction_and_trgc()
    test_within_block_invariance()
    test_moving_window_blocks()
    test_compute_subject_gc()
    test_io_roundtrip()
    n_ok = sum(_results)
    print(f'\n{n_ok}/{len(_results)} checks passed')
    _sys.exit(0 if n_ok == len(_results) else 1)
