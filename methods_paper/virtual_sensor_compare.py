#!/usr/bin/env python3
"""Compare ROI virtual-sensor (aggregation) methods and their impact on GC.

Motivation (methods paper).  The source-space GC reduces each ROI's vertices to
ONE virtual channel via a fixed whole-ensemble first principal component
(``granger.reduce_roi_first_pc`` = FIXPC1), then runs bivariate spectral MVAR
GC.  This matches the sensor BSMART pipeline (one mean-of-3 pseudo-channel per
ROI + plain ``mov_bi_ga`` GC) for a like-for-like sensor<->source contrast.

Pellegrini et al. 2023 (NeuroImage; "Identifying good practices for detecting
inter-regional linear functional connectivity from EEG") benchmarked source-FC
pipelines and recommend LCMV -> PCA aggregation with a FIXED number of
components (FIXPC3/4 beat FIXPC1) -> TIME-REVERSED GC (TRGC), because spurious
mixing/leakage coupling is instantaneous and cancels under time reversal while
real, delayed coupling survives.  Our FIXPC1 + plain GC therefore deviates on
two axes that could themselves manufacture the observed baseline washout.

This script screens, LOCALLY, on one/two custom-atlas ROI pairs and both tasks,
how the virtual-sensor choice changes (a) the ROI time series and (b) the GC:

  aggregation (1 channel / ROI): FIXPC1 (current) | mean | mean_flip | maxpower
  GC estimator               : plain pairwise GC AND Diff-TRGC (both from the
                               SAME engine, moving_window_pairwise_gc trgc=True)

Reported per method:
  TIME SERIES  PC1 variance-explained, virtual-channel similarity to FIXPC1,
               within-ROI task/baseline power ratio, inter-ROI coupling
               (corr + low_beta coherence) at baseline vs task.
  GC           low_beta baseline / task / (task-baseline) for both directions,
               plain GC and TRGC.

Headline question: does any aggregation, or TRGC, restore a genuine
baseline->task modulation in source (the thing the sensor GC shows)?

The FIXPC3/4 arm needs block/multivariate spectral GC (a small engine addition)
and is handled in a follow-up; this covers the 1-channel aggregations + the
plain-vs-TRGC axis with NO new engine code.

Outputs (durable, under GC_sensor_vs_source_baseline_check/virtual_sensor_compare/):
  {task}_vsensor_gc.png          GC task-baseline contrast by method x estimator
  {task}_vsensor_ts.png          TS-impact panels by method
  {task}_vsensor_metrics.csv     per (subj, pair, method) raw metrics
  {task}_vsensor_group.csv       group summary

Usage (local):
    conda activate mne
    python methods_paper/virtual_sensor_compare.py --task overtProd  --stim-class prodDiff
    python methods_paper/virtual_sensor_compare.py --task perception --stim-class percDiff
"""
import os, sys, glob, argparse, warnings
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from config import DECODE_OUTPUT_ROOT
from decoding_io import _load_cached_roi_data
from granger import moving_window_pairwise_gc, band_average, DEFAULT_BANDS
from run_granger import resample_channels

OUT = DECODE_OUTPUT_ROOT.parent / 'GC_sensor_vs_source_baseline_check' / 'virtual_sensor_compare'
OUT.mkdir(parents=True, exist_ok=True)

# Vertex-TS path is resolved through the project's standard resolver
# (config.find_cached_npz -> ROI_TIMESERIES_ROOT + ROI_TIMESERIES_EXTERNAL from
# config.env), so the same code finds the data on the workstation (config.env
# points ROI_TIMESERIES_EXTERNAL at the workstation drives) without any flags.
# --src-root is only a fallback template for machines whose config.env does not
# yet list the data root (e.g. this box, where the TS live on /mnt/s).  {leak}
# is filled with leakage_corrected|raw so the same template serves both.
DEFAULT_SRC = ('/mnt/s/Research/SpeechProduction/DECODE_source_space_timeseries/'
               '{task}/LCMV/custom/vertex/{leak}/{subj}_{task}_{stim}.npz')


def resolve_src(subj, task, stim, leakage, src_tmpl):
    """Path to a subject's vertex-TS npz: config resolver first, template fallback."""
    p = config.find_cached_npz(task, 'LCMV', 'custom', 'vertex_selectkbest',
                               leakage, subj, stim)
    if p is not None:
        return str(p)
    leak = 'leakage_corrected' if leakage else 'raw'
    return src_tmpl.format(task=task, subj=subj, stim=stim, leak=leak)

FS = 500.0
FREQS = np.arange(1, 31)
BAND = 'low_beta'
GC_ORDER = 10
GC_WIN_MS = 40
GC_STEP = 5            # moving-window step (samples); 5 = screening (means over
                      # window ranges are stable; production uses step=1)
METHODS = ['fixpc1', 'mean', 'mean_flip', 'maxpower']
MCOL = {'fixpc1': '#b2182b', 'mean': '#7fbf7b', 'mean_flip': '#1b7837', 'maxpower': '#762a83'}

TASK_CFG = {
    'overtProd':  dict(base=(-1.50, -1.40), task=(-0.30, 0.30), onset='production onset'),
    'perception': dict(base=(-0.20, -0.10), task=(0.0, 0.30),  onset='auditory onset'),
}
DEFAULT_PAIRS = ['awfa-lh:ifc-lh', 'tpc-lh:ifc-lh']
# a representative subject spread (clean low-floor + ill-conditioned high-floor)
DEFAULT_SUBS = ['EEGPROD4001', 'EEGPROD4002', 'EEGPROD4003', 'EEGPROD4005',
                'EEGPROD4004', 'EEGPROD4007', 'EEGPROD4008', 'EEGPROD4011']


# ── aggregation: all methods from ONE ensemble eigendecomposition per ROI ──
def aggregate(X, kmax=4):
    """X: (n_ep, n_vtx, n_t) raw vertex TS -> dict of virtual channels + diag.

    FIXPC1 matches granger.reduce_roi_first_pc exactly (same mean-center,
    same sign convention: largest-|loading| positive), but the spatial
    modes come from the vertex covariance eigendecomposition (cheap: V x V,
    not an SVD of the tall V x (ep*t) matrix).
    """
    n_ep, n_v, n_t = X.shape
    if n_v == 1:
        vc = X[:, 0, :]
        return dict(fixpc1=vc, mean=vc, mean_flip=vc, maxpower=vc,
                    pcs=vc[:, None, :], var_exp=np.array([1.0]))
    M = np.transpose(X, (1, 0, 2)).reshape(n_v, n_ep * n_t)       # (V, ep*t)
    M = M - M.mean(axis=1, keepdims=True)
    C = M @ M.T                                                   # (V, V) scatter
    evals, evecs = np.linalg.eigh(C)                              # ascending
    order = np.argsort(evals)[::-1]
    evals = evals[order]; evecs = evecs[:, order]
    evals = np.clip(evals, 0, None)
    var_exp = evals / (evals.sum() + 1e-30)
    # sign-fix each kept mode: largest-magnitude loading positive
    k = min(kmax, n_v)
    W = evecs[:, :k].copy()
    for j in range(k):
        if W[np.argmax(np.abs(W[:, j])), j] < 0:
            W[:, j] = -W[:, j]
    w1 = W[:, 0]
    fixpc1 = np.einsum('v,evt->et', w1, X)                        # project RAW X
    pcs = np.einsum('vk,evt->ekt', W, X)                          # (n_ep, k, n_t)
    mean = X.mean(axis=1)
    signs = np.sign(w1); signs[signs == 0] = 1.0
    mean_flip = np.einsum('v,evt->et', signs / n_v, X)            # flipped average
    vpow = np.diag(C)                                             # ensemble variance/vertex
    maxpower = X[:, int(np.argmax(vpow)), :]
    return dict(fixpc1=fixpc1, mean=mean, mean_flip=mean_flip, maxpower=maxpower,
                pcs=pcs, var_exp=var_exp)


def win(t, lo, hi):
    return (t >= lo) & (t <= hi)


def coh_lowbeta(a, b, nfft=128):
    """Ensemble low_beta (13-20 Hz) magnitude-squared coherence over trials."""
    n = a.shape[1]; w = np.hanning(n)
    A = np.fft.rfft((a - a.mean(1, keepdims=True)) * w, n=nfft, axis=1)
    B = np.fft.rfft((b - b.mean(1, keepdims=True)) * w, n=nfft, axis=1)
    f = np.fft.rfftfreq(nfft, 1.0 / FS)
    Sxy = (A * np.conj(B)).mean(0); Sxx = (A * np.conj(A)).mean(0).real
    Syy = (B * np.conj(B)).mean(0).real
    coh = (np.abs(Sxy) ** 2) / (Sxx * Syy + 1e-30)
    m = (f >= 13) & (f <= 20)
    return float(coh[m].mean())


def gc_metrics(a, b, tvec, base, task):
    """Plain GC (both dirs) + Diff-TRGC on 2 virtual channels -> band means."""
    ws = int(round(GC_WIN_MS / 1000.0 * FS))
    res = moving_window_pairwise_gc(np.stack([a, b], 1), order=GC_ORDER, freqs=FREQS,
                                    fs=FS, win_samples=ws, step=GC_STEP, trgc=True)
    starts = np.arange(0, a.shape[1] - ws + 1, GC_STEP)
    wm = tvec[starts] * 1000.0
    fxy = band_average(res['f_xy'], FREQS, DEFAULT_BANDS)[BAND]   # a->b
    fyx = band_average(res['f_yx'], FREQS, DEFAULT_BANDS)[BAND]   # b->a
    dxy = band_average(res['d_xy'], FREQS, DEFAULT_BANDS)[BAND]   # net TRGC a->b
    mb = win(wm, base[0] * 1000, base[1] * 1000)
    mt = win(wm, task[0] * 1000, task[1] * 1000)
    out = {}
    for tag, g in (('fwd', fxy), ('rev', fyx), ('trgc', dxy)):
        out[f'{tag}_base'] = float(g[mb].mean())
        out[f'{tag}_task'] = float(g[mt].mean())
        out[f'{tag}_contrast'] = float(g[mt].mean() - g[mb].mean())
    return out


def process(subj, task, stim, pairs, leakage, src_tmpl, base, task_win):
    path = resolve_src(subj, task, stim, leakage, src_tmpl)
    if not os.path.exists(path):
        return []
    rois = sorted({r for p in pairs for r in p.split(':')})
    from pathlib import Path
    roi_data, _y, times, sfreq = _load_cached_roi_data(Path(path), 'vertex_selectkbest',
                                                       roi_subset=rois)
    if roi_data is None:
        return []
    # aggregate each ROI once; resample each virtual channel to 500 Hz
    agg = {}
    for r in rois:
        A = aggregate(np.asarray(roi_data[r], float))
        vc = {m: A[m] for m in METHODS}
        # resample all methods' 1-ch channels together (n_meth, n_ep, n_t)
        stacked = np.stack([vc[m] for m in METHODS], 0)
        stacked, fs = resample_channels(stacked, sfreq, FS)
        agg[r] = dict(vc={m: stacked[i] for i, m in enumerate(METHODS)},
                      var_exp=A['var_exp'])
    del roi_data
    tvec = times[0] + np.arange(next(iter(agg.values()))['vc']['fixpc1'].shape[1]) / FS

    rows = []
    for p in pairs:
        ra, rb = p.split(':')
        for m in METHODS:
            a = agg[ra]['vc'][m]; b = agg[rb]['vc'][m]
            n = min(a.shape[0], b.shape[0]); a, b = a[:n], b[:n]
            row = dict(subj=subj, task=task, pair=p, method=m,
                       pc1_var_a=float(agg[ra]['var_exp'][0]),
                       pc1_var_b=float(agg[rb]['var_exp'][0]))
            # similarity of this method's channel to FIXPC1 (|corr|, per ROI)
            for tag, r in (('a', ra), ('b', rb)):
                f1 = agg[r]['vc']['fixpc1'].ravel(); vv = agg[r]['vc'][m].ravel()
                row[f'sim_fixpc1_{tag}'] = float(abs(np.corrcoef(f1, vv)[0, 1]))
            # within-ROI task/baseline power ratio
            mb = win(tvec, *base); mt = win(tvec, *task_win)
            for tag, x in (('a', a), ('b', b)):
                pw = (x ** 2).mean(0)
                row[f'pwr_ratio_{tag}'] = float(pw[mt].mean() / (pw[mb].mean() + 1e-30))
            # inter-ROI coupling, baseline vs task
            def r_ab(mask):
                aa = a[:, mask].ravel(); bb = b[:, mask].ravel()
                return float(np.corrcoef(aa, bb)[0, 1])
            row['corr_base'] = r_ab(mb); row['corr_task'] = r_ab(mt)
            row['coh_base'] = coh_lowbeta(a[:, mb], b[:, mb])
            row['coh_task'] = coh_lowbeta(a[:, mt], b[:, mt])
            # GC
            row.update(gc_metrics(a, b, tvec, base, task_win))
            rows.append(row)
    return rows


def group_summary(df):
    recs = []
    metrics = ['pc1_var_a', 'pc1_var_b', 'sim_fixpc1_a', 'sim_fixpc1_b',
               'pwr_ratio_a', 'pwr_ratio_b', 'corr_base', 'corr_task',
               'coh_base', 'coh_task',
               'fwd_base', 'fwd_task', 'fwd_contrast', 'rev_base', 'rev_task',
               'rev_contrast', 'trgc_base', 'trgc_task', 'trgc_contrast']
    for (pair, method), g in df.groupby(['pair', 'method']):
        rec = dict(pair=pair, method=method, n=len(g))
        for mt in metrics:
            v = g[mt].values
            rec[f'{mt}_mean'] = float(np.mean(v))
            rec[f'{mt}_sem'] = float(np.std(v) / np.sqrt(max(len(v), 1)))
        for tag in ('fwd', 'rev', 'trgc'):
            c = g[f'{tag}_contrast'].values
            rec[f'{tag}_npos'] = int((c > 0).sum())
            t, pv = stats.ttest_1samp(c, 0.0) if len(c) > 1 else (np.nan, np.nan)
            rec[f'{tag}_t'] = float(t); rec[f'{tag}_p'] = float(pv)
        recs.append(rec)
    return pd.DataFrame(recs)


def fig_gc(summ, task, pairs, sfx=''):
    ests = [('fwd', 'plain GC  ROI-A->B'), ('rev', 'plain GC  ROI-B->A'),
            ('trgc', 'TRGC (net A->B)')]
    fig, axes = plt.subplots(len(pairs), len(ests),
                             figsize=(4.2 * len(ests), 3.6 * len(pairs)),
                             squeeze=False)
    x = np.arange(len(METHODS))
    for i, p in enumerate(pairs):
        for j, (tag, title) in enumerate(ests):
            ax = axes[i][j]
            sub = summ[summ.pair == p].set_index('method')
            cont = [sub.loc[m, f'{tag}_contrast_mean'] for m in METHODS]
            sem = [sub.loc[m, f'{tag}_contrast_sem'] for m in METHODS]
            npos = [int(sub.loc[m, f'{tag}_npos']) for m in METHODS]
            n = int(sub.iloc[0]['n'])
            bars = ax.bar(x, cont, yerr=sem, color=[MCOL[m] for m in METHODS],
                          capsize=3, alpha=0.9)
            ax.axhline(0, color='k', lw=0.7)
            for xi, b, np_ in zip(x, bars, npos):
                ax.annotate(f'{np_}/{n}', (xi, b.get_height()),
                            ha='center', va='bottom' if b.get_height() >= 0 else 'top',
                            fontsize=7)
            ax.set_xticks(x); ax.set_xticklabels(METHODS, rotation=30, fontsize=8)
            if j == 0:
                ax.set_ylabel(f'{p}\ntask - baseline GC (low_beta)', fontsize=9)
            if i == 0:
                ax.set_title(title, fontsize=10)
            ax.grid(axis='y', alpha=0.25)
    fig.suptitle(f'{task}: does virtual-sensor choice or TRGC restore a source '
                 f'baseline->task GC rise?\n(bars = group mean +/- sem; '
                 f'annotation = # subjects with task>baseline)', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    f = OUT / f'{task}{sfx}_vsensor_gc.png'; fig.savefig(f, dpi=140); plt.close(fig)
    return f


def fig_ts(summ, task, pairs, sfx=''):
    panels = [('pc1_var_a', 'PC1 var. explained (ROI-A)'),
              ('sim_fixpc1_b', '|corr| of channel to FIXPC1 (ROI-B)'),
              ('pwr_ratio_a', 'within-ROI task/baseline power (ROI-A)'),
              ('coh_base', 'low_beta coherence @ baseline'),
              ('coh_task', 'low_beta coherence @ task'),
              ('corr_task', 'inter-ROI corr @ task')]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2))
    x = np.arange(len(METHODS))
    for ax, (key, title) in zip(axes.ravel(), panels):
        for oi, p in enumerate(pairs):
            sub = summ[summ.pair == p].set_index('method')
            vals = [sub.loc[m, f'{key}_mean'] for m in METHODS]
            sem = [sub.loc[m, f'{key}_sem'] for m in METHODS]
            ax.errorbar(x + 0.12 * oi, vals, yerr=sem, marker='o', ms=5, lw=1.4,
                        capsize=2, label=p)
        ax.set_xticks(x); ax.set_xticklabels(METHODS, rotation=30, fontsize=8)
        ax.set_title(title, fontsize=9); ax.grid(alpha=0.25)
    axes.ravel()[0].legend(fontsize=7)
    fig.suptitle(f'{task}: virtual-sensor method impact on the ROI time series',
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    f = OUT / f'{task}{sfx}_vsensor_ts.png'; fig.savefig(f, dpi=140); plt.close(fig)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', default='overtProd', choices=['overtProd', 'perception'])
    ap.add_argument('--stim-class', default='prodDiff')
    ap.add_argument('--pairs', nargs='+', default=DEFAULT_PAIRS,
                    help='ROI pairs "roiA:roiB" (custom-atlas names)')
    ap.add_argument('--subjects', nargs='+', default=DEFAULT_SUBS)
    ap.add_argument('--src-root', default=DEFAULT_SRC,
                    help='fallback vertex-TS npz template with {task}/{subj}/{stim}/{leak}; '
                         'only used when config.find_cached_npz does not resolve')
    ap.add_argument('--no-leakage', dest='leakage', action='store_false',
                    help='use the raw (non-leakage-corrected) vertex TS instead')
    ap.set_defaults(leakage=True)
    ap.add_argument('--n-jobs', type=int, default=8)
    args = ap.parse_args()
    cfg = TASK_CFG[args.task]
    leak_tag = 'leakage_corrected' if args.leakage else 'raw'
    print(f'{args.task}/{args.stim_class} [{leak_tag}]: {len(args.subjects)} subj x '
          f'{len(args.pairs)} pairs x {len(METHODS)} methods | '
          f'base {cfg["base"]}s task {cfg["task"]}s ({cfg["onset"]})')

    out = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(process)(s, args.task, args.stim_class, args.pairs, args.leakage,
                         args.src_root, cfg['base'], cfg['task'])
        for s in args.subjects)
    rows = [r for sub in out for r in sub]
    if not rows:
        print('No data — check --src-root path / mounts.'); return
    df = pd.DataFrame(rows)
    sfx = '' if args.leakage else '_raw'
    df.to_csv(OUT / f'{args.task}{sfx}_vsensor_metrics.csv', index=False)
    summ = group_summary(df)
    summ.to_csv(OUT / f'{args.task}{sfx}_vsensor_group.csv', index=False)

    # console: headline pair, forward GC + TRGC contrast per method
    hp = args.pairs[0]
    print(f'\n=== {args.task}  {hp}  low_beta task-minus-baseline (n={df.subj.nunique()}) ===')
    print(f'{"method":10} {"GC_fwd":>18} {"GC_rev":>18} {"TRGC":>18}')
    s = summ[summ.pair == hp].set_index('method')
    for m in METHODS:
        cells = []
        for tag in ('fwd', 'rev', 'trgc'):
            cells.append(f'{s.loc[m, f"{tag}_contrast_mean"]:+.4f} '
                         f'({int(s.loc[m, f"{tag}_npos"])}/{int(s.loc[m,"n"])})')
        print(f'{m:10} {cells[0]:>18} {cells[1]:>18} {cells[2]:>18}')
    print('\n  PC1 var-explained / |sim to FIXPC1| by method (ROI-A):')
    for m in METHODS:
        print(f'    {m:10} pc1var={s.loc[m,"pc1_var_a_mean"]:.2f}  '
              f'sim={s.loc[m,"sim_fixpc1_a_mean"]:.2f}  '
              f'pwr_ratio={s.loc[m,"pwr_ratio_a_mean"]:.2f}')

    f1 = fig_gc(summ, args.task, args.pairs, sfx)
    f2 = fig_ts(summ, args.task, args.pairs, sfx)
    print(f'\nwrote {OUT / f"{args.task}{sfx}_vsensor_metrics.csv"}')
    print(f'wrote {OUT / f"{args.task}{sfx}_vsensor_group.csv"}')
    print(f'wrote {f1}\nwrote {f2}')


if __name__ == '__main__':
    main()
