"""End-to-end DIRECTION test on synthetic data.

Simulates two 'ROIs' with a known unidirectional coupling (A drives B with a
lag), runs them through the production path exactly as run_gc_final.sh /
run_gc_stats_figs.sh / plot_gc_window_sweep.py do, and checks at every stage
that the labelled direction is the simulated one:

  compute_subject_gc  -> fxy (i->j) >> fyx (j->i), dtrgc > 0  for roi order (A, B)
                      -> fxy << fyx, dtrgc < 0                for roi order (B, A)
  save/load npz       -> load_gc_group reproduces it
  granger_stats CSV   -> rows with src=A,tgt=B carry the big GC; dtrgc src=A
  plot_gc_window_sweep summarize() -> the per-run deltas keep the sign

Also exercises the FIXPC-2 block path and the state-space conditional path.
Everything is tiny synthetic data in the scratchpad; nothing touches a cache.
"""
import os, sys, shutil, tempfile
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'exploratory'))

from granger import (pairwise_spectral_gc, time_reversed_pairwise_gc,
                     block_spectral_gc, time_reversed_block_gc,
                     conditional_spectral_gc)
from granger_statespace import statespace_conditional_gc
from granger_wilson import pairwise_spectral_gc_np
from run_granger import compute_subject_gc, save_subject_gc, roiset_tag, gc_tag
from granger_stats import load_gc_group, run_stats, REPORT_BANDS
import pandas as pd

FAILS = []
def check(name, ok, info=''):
    print(f'  [{"PASS" if ok else "FAIL"}] {name}  {info}')
    if not ok:
        FAILS.append(name)


def simulate(n_trials, n_times, fs=500.0, c=0.6, seed=0, burn=300):
    """x drives y: y(t) = 0.5 y(t-1) - 0.3 y(t-2) + c x(t-2) + e_y; x is AR(2)."""
    rng = np.random.default_rng(seed)
    T = n_times + burn
    X = np.zeros((n_trials, 2, T))
    e = rng.standard_normal((n_trials, 2, T))
    for t in range(2, T):
        X[:, 0, t] = 0.55 * X[:, 0, t-1] - 0.4 * X[:, 0, t-2] + e[:, 0, t]
        X[:, 1, t] = 0.5 * X[:, 1, t-1] - 0.3 * X[:, 1, t-2] + c * X[:, 0, t-2] + e[:, 1, t]
    return X[:, :, burn:]


fs = 500.0
freqs = np.arange(1, 31, dtype=float)
X = simulate(80, 400, fs)

print('\n== 1. estimator-level direction (x = row 0 drives y = row 1) ==')
fxy, fyx = pairwise_spectral_gc(X, 6, freqs, fs)
check('pairwise f_xy >> f_yx', fxy.mean() > 5 * fyx.mean(), f'{fxy.mean():.3f} vs {fyx.mean():.3f}')
dxy, dyx = time_reversed_pairwise_gc(X, 6, freqs, fs)
check('TRGC d_xy > 0 and d_yx = -d_xy', dxy.mean() > 0 and np.allclose(dyx, -dxy), f'd_xy={dxy.mean():.3f}')
# swapped rows must swap the answer
fxy_s, fyx_s = pairwise_spectral_gc(X[:, ::-1, :], 6, freqs, fs)
# The Morf/armorf recursion is not exactly permutation-equivariant (its
# forward/backward normalisation uses Cholesky factors, so channel order
# enters at round-off-plus level: ~1e-3 of the GC on 400-sample windows, up
# to a few % on 20-30-sample windows; the same holds in BSMART).  So compare
# with a tolerance: the two directions must swap, not be bit-identical.
tol = 0.05 * fxy.mean()
check('row swap swaps directions (within 5% of GC scale)',
      np.abs(fxy_s - fyx).max() < tol and np.abs(fyx_s - fxy).max() < tol,
      f'max|diff| {max(np.abs(fxy_s - fyx).max(), np.abs(fyx_s - fxy).max()):.1e}')
# block path with 2 comps per ROI (second comp = scaled+noise copy)
rng = np.random.default_rng(1)
Xb = np.concatenate([X[:, :1], X[:, :1] * 0.5 + 0.3 * rng.standard_normal(X[:, :1].shape),
                     X[:, 1:], X[:, 1:] * 0.7 + 0.3 * rng.standard_normal(X[:, 1:].shape)], axis=1)
bxy, byx = block_spectral_gc(Xb, 6, freqs, fs, ([0, 1], [2, 3]))
check('block f_xy >> f_yx', bxy.mean() > 5 * byx.mean(), f'{bxy.mean():.3f} vs {byx.mean():.3f}')
bd, _ = time_reversed_block_gc(Xb, 6, freqs, fs, ([0, 1], [2, 3]))
check('block TRGC > 0', bd.mean() > 0, f'{bd.mean():.3f}')
bxy2, byx2 = block_spectral_gc(Xb, 6, freqs, fs, ([2, 3], [0, 1]))
check('block swap swaps directions', np.allclose(bxy2, byx) and np.allclose(byx2, bxy))
# conditional paths with a 3rd independent signal appended
Z = rng.standard_normal((X.shape[0], 1, X.shape[2]))
X3 = np.concatenate([X, Z], axis=1)
ss = statespace_conditional_gc(X3, 6, freqs, fs)
check('state-space (0->1) >> (1->0)', ss[(0, 1)][1].mean() > 5 * ss[(1, 0)][1].mean(),
      f'{ss[(0,1)][1].mean():.3f} vs {ss[(1,0)][1].mean():.3f}')
ch = conditional_spectral_gc(X3, 6, freqs, fs)
check('Chen conditional (0->1) >> (1->0)', ch[(0, 1)].mean() > 5 * ch[(1, 0)].mean(),
      f'{ch[(0,1)].mean():.3f} vs {ch[(1,0)].mean():.3f}')
nxy, nyx, info = pairwise_spectral_gc_np(X, freqs, fs, n_bins=512)
check('Wilson nonparametric f_xy >> f_yx', nxy.mean() > 5 * nyx.mean(),
      f'{nxy.mean():.3f} vs {nyx.mean():.3f} (converged={info["converged"]})')

print('\n== 2. compute_subject_gc: direction follows roi_names order ==')
# Build 'vertex' data: each ROI = 3 vertices = the virtual signal with different
# gains + noise, at a 'cache' rate of 1000 Hz so the resampler runs (to 500 Hz).
def to_cache(sig, gains, seed):
    r = np.random.default_rng(seed)
    v = np.stack([g * sig for g in gains], axis=1)          # (ep, 3, t)
    return v + 0.2 * r.standard_normal(v.shape)
Xc = simulate(60, 1200, fs=1000.0, seed=5)                   # 1.2 s epochs
times = -0.5 + np.arange(1200) / 1000.0
roi_AB = {'A': to_cache(Xc[:, 0], [1.0, 0.8, 0.6], 2), 'B': to_cache(Xc[:, 1], [0.9, 0.7, 0.5], 3)}
roi_BA = {'B': roi_AB['B'], 'A': roi_AB['A']}
common = dict(order=6, win_ms=80.0, target_fs=500.0, step=4, freqs=freqs,
              normalize='none', trgc=True, n_jobs=1, diagnostics=True)
res_AB = compute_subject_gc(roi_AB, times, 1000.0, **common)
res_BA = compute_subject_gc(roi_BA, times, 1000.0, **common)
for res, order in ((res_AB, 'A,B'), (res_BA, 'B,A')):
    i, j = res['roi_names'][int(res['pair_i'][0])], res['roi_names'][int(res['pair_j'][0])]
    m_xy = np.nanmean(res['fxy']['theta'] + res['fxy']['low_beta'])
    m_yx = np.nanmean(res['fyx']['theta'] + res['fyx']['low_beta'])
    d = np.nanmean(res['dtrgc']['low_beta'])
    print(f'    roi order {order}: pair i={i}, j={j}; fxy={m_xy:.3f} fyx={m_yx:.3f} dtrgc={d:+.3f}')
    if i == 'A':
        check(f'[{order}] fxy (A->B) >> fyx and dtrgc>0', m_xy > 3 * m_yx and d > 0)
    else:
        check(f'[{order}] fyx (A->B) >> fxy and dtrgc<0', m_yx > 3 * m_xy and d < 0)
check('window_ms starts at epoch start', np.isclose(res_AB['window_ms'][0], -500.0),
      f'{res_AB["window_ms"][0]:.1f}')
check('n windows = (n_t - win + 1)/step', res_AB['window_ms'].size == len(range(0, 600 - 40 + 1, 4)))
# FIXPC-2 block path through the runner
res_pc2 = compute_subject_gc(roi_AB, times, 1000.0, n_pcs=2, **common)
check('FIXPC-2: n_comp == [2,2]', list(res_pc2['n_comp']) == [2, 2], str(res_pc2['n_comp']))
check('FIXPC-2: fxy (A->B) >> fyx, dtrgc>0',
      np.nanmean(res_pc2['fxy']['low_beta']) > 3 * np.nanmean(res_pc2['fyx']['low_beta'])
      and np.nanmean(res_pc2['dtrgc']['low_beta']) > 0)
# conditional (state-space) path through the runner, 3 ROIs
roi_ABC = dict(roi_AB); roi_ABC['C'] = to_cache(np.random.default_rng(9).standard_normal(Xc[:, 0].shape), [1, .8, .6], 4)
res_c = compute_subject_gc(roi_ABC, times, 1000.0, gc_mode='conditional',
                           **{k: v for k, v in common.items() if k != 'trgc'})
k = [p for p, (i, j) in enumerate(zip(res_c['pair_i'], res_c['pair_j']))
     if res_c['roi_names'][i] == 'A' and res_c['roi_names'][j] == 'B'][0]
check('conditional runner: fxy[A->B] >> fyx[B->A]',
      np.nanmean(res_c['fxy']['low_beta'][k]) > 3 * np.nanmean(res_c['fyx']['low_beta'][k]),
      f'{np.nanmean(res_c["fxy"]["low_beta"][k]):.3f} vs {np.nanmean(res_c["fyx"]["low_beta"][k]):.3f}')

print('\n== 3. npz round trip + granger_stats CSV labelling ==')
root = tempfile.mkdtemp(prefix='gc_e2e_')
from pathlib import Path
root = Path(root)
# fake 5 'subjects' by re-simulating with different seeds, in BOTH roi orders
# (two different pair directories, both alphabetical tag 'rois_a-b')
def subj_result(seed, order):
    Xs = simulate(60, 1200, fs=1000.0, seed=seed)
    d = {'A': to_cache(Xs[:, 0], [1.0, 0.8, 0.6], seed), 'B': to_cache(Xs[:, 1], [0.9, 0.7, 0.5], seed + 1)}
    if order == 'BA':
        d = {'B': d['B'], 'A': d['A']}
    return compute_subject_gc(d, times, 1000.0, **common)
for order in ('AB', 'BA'):
    for s in range(5):
        res = subj_result(100 + s, order)
        out = save_subject_gc(res, f'SUBJ{s:02d}', 'perception', 'percDiff', 'LCMV', 'custom',
                              'vertex_selectkbest', True, 6, 80.0, 500.0, 'none',
                              output_root=root / order, roi_subset=list(res['roi_names']))
    gc_dir = out.parent
    agg = load_gc_group(str(gc_dir), REPORT_BANDS)
    i, j = int(agg['pair_i'][0]), int(agg['pair_j'][0])
    names = agg['roi_names']
    print(f'    [{order}] loaded {len(agg["subjects"])} subj, roi_names={names}, pair=({names[i]},{names[j]})')
    csv = run_stats(str(gc_dir), 'perception', str(gc_dir / 'group_stats_bl'),
                    baseline_ms=(-80.0, 0.0), permutation=True, n_permutations=64, tfce=False)
    df = pd.read_csv(csv)
    g = df[df.measure == 'gc'].groupby(['src', 'tgt'])['gc_mean'].mean()
    d = df[df.measure == 'dtrgc'].groupby(['src', 'tgt'])['gc_mean'].mean()
    print(f'    [{order}] CSV gc_mean by (src,tgt): {g.to_dict()}')
    print(f'    [{order}] CSV dtrgc rows: {d.to_dict()}')
    check(f'[{order}] CSV: gc A->B >> B->A', g[('A', 'B')] > 3 * g[('B', 'A')])
    dk = list(d.index)[0]
    expect_sign = 1 if dk == ('A', 'B') else -1
    check(f'[{order}] CSV: dtrgc src={dk[0]} tgt={dk[1]} sign matches (expect {expect_sign:+d})',
          np.sign(d[dk]) == expect_sign, f'{d[dk]:+.3f}')
    # task windows: perception task starts at 0 -> pval only there
    tested = df[(df.measure == 'gc') & df.pval.notna()]
    check(f'[{order}] pointwise tests only on windows with start >= 0', (tested.window_ms >= 0).all())

print('\n== 4. plot_gc_window_sweep.summarize keeps the direction ==')
import plot_gc_window_sweep as pw
import argparse
# the AB tree, re-rooted so pair_dir resolves inside the scratch tree
pw.GC_OUTPUT_ROOT = root / 'AB'
args = argparse.Namespace(runs=[(6, 80.0)], sweep='window', target_fs=500.0, normalize='none',
                          n_pcs=1, n_pcs_roi=None, method='LCMV', atlas='custom',
                          feature_mode='vertex_selectkbest', stats_subdir='group_stats_bl',
                          edge_guard=30.0, baseline_dur=100.0, baseline_overtprod=None,
                          baseline_perception=None, overview_range=[100.0, 400.0])
series = pw.load_series(args, 'perception', 'percDiff', 'A,B')
rows = pd.DataFrame(pw.summarize(args, 'perception', 'percDiff', 'A,B', series))
print(rows[['measure', 'src', 'tgt', 'band', 'baseline_mean_80', 'n_sig_pointwise_80', 'range_mean_delta_80']].to_string(index=False))
ab = rows[(rows.measure == 'gc') & (rows.src == 'A')]['baseline_mean_80'].mean()
ba = rows[(rows.measure == 'gc') & (rows.src == 'B')]['baseline_mean_80'].mean()
check('sweep summary: A->B baseline GC >> B->A', ab > 3 * ba, f'{ab:.3f} vs {ba:.3f}')
check('sweep summary has the dtrgc row with src=A', ((rows.measure == 'dtrgc') & (rows.src == 'A')).any())
# and with the pair string reversed, the dtrgc row is NOT found (documents the silent skip)
rows_rev = pd.DataFrame(pw.summarize(args, 'perception', 'percDiff', 'B,A',
                                     pw.load_series(args, 'perception', 'percDiff', 'B,A')))
has_d = ((rows_rev.measure == 'dtrgc') & rows_rev.filter(like='baseline_mean').notna().any(axis=1)).any()
print(f'    pair given as B,A: dtrgc row populated = {has_d}')

shutil.rmtree(root, ignore_errors=True)
print('\n' + ('ALL PASSED' if not FAILS else f'FAILED: {FAILS}'))
sys.exit(1 if FAILS else 0)
