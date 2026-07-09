#!/usr/bin/env python3
"""Diagnose why sensor GC has a low, task-modulated baseline but LCMV source ROIs do not.

Reads the small reduced-TS caches written by ``baseline_ts_extract.py`` and,
for every subject, matched channel pair, and space (sensor / source), compares
the BASELINE window against the TASK window on the time series themselves and
on the resulting Granger causality.  The point is to locate the difference in a
*time-series property*, not just in the GC output.

Metrics (baseline vs task), per subject x pair x space:
  - var ratio         task/baseline signal variance per channel (is baseline
                      genuinely quiet in sensors but not in source?)
  - r0                zero-lag inter-channel correlation
  - CCF               ensemble cross-correlation over lags +-20 samples; its
                      shape encodes the *directed lagged* coupling that GC reads.
  - ccf_shape_sim     corr(CCF_baseline, CCF_task): if ~1 the lag structure is
                      the SAME at rest and task => a fixed (filter-imposed),
                      time-invariant coupling => GC cannot dip at baseline.
  - coh_lowbeta       13-20 Hz magnitude-squared coherence
  - gc_lowbeta        production-style moving-window Geweke GC (win 40 ms, order
                      10), averaged over baseline vs task windows, both directions
  - shared_frac       (per subject x space) top-eigenvalue fraction of the
                      4-channel covariance in baseline = how collinear the ROIs
                      are (a degeneracy / shared-leakage proxy)

Outputs (durable, under derivatives/.../GC_sensor_vs_source_baseline_check/):
  - ts_baseline_metrics.csv          tidy per subject/pair/space
  - ts_shared_component.csv          per subject/space
  - fig_group_summary.png            the group answer
  - fig_ccf_examples.png             CCF baseline vs task, sensor vs source

Usage
-----
    conda activate mne
    python methods_paper/baseline_ts_analyze.py --task overtProd --stim-class prodDiff
"""
import os, sys, glob, argparse, warnings
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
warnings.filterwarnings('ignore')
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DECODE_OUTPUT_ROOT
from granger import moving_window_pairwise_gc, band_average, DEFAULT_BANDS

RED_ROOT = (DECODE_OUTPUT_ROOT.parent / 'GC_sensor_vs_source_baseline_check' / 'reduced_ts')
OUT_ROOT = (DECODE_OUTPUT_ROOT.parent / 'GC_sensor_vs_source_baseline_check')
FS = 500.0
BASE = (-1.50, -1.35)      # leading/baseline window (matches the sensor-GC reference region)
TASK = (-0.30, 0.30)       # around production onset
MAXLAG = 20                # +-40 ms at 500 Hz
FREQS = np.arange(1, 31)
BAND = 'low_beta'
# example subjects for the CCF panel: 2 low-floor (clean), 2 high-floor
EXAMPLES = ['EEGPROD4014', 'EEGPROD4003', 'EEGPROD4019', 'EEGPROD4021']


def win(t, lo, hi):
    return (t >= lo) & (t <= hi)


def ccf_ensemble(a, b, maxlag):
    """Ensemble cross-correlation r_ab(l)=corr(a(t), b(t+l)), avg over trials.

    a, b : (n_ep, n_t) already restricted to a window.  Positive lag => a leads b.
    """
    a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
    a = a / (a.std(1, keepdims=True) + 1e-20); b = b / (b.std(1, keepdims=True) + 1e-20)
    n = a.shape[1]; lags = np.arange(-maxlag, maxlag + 1); out = np.zeros(lags.size)
    for i, l in enumerate(lags):
        if l >= 0:
            out[i] = (a[:, :n - l] * b[:, l:]).mean(1).mean() if n - l > 0 else np.nan
        else:
            out[i] = (a[:, -l:] * b[:, :n + l]).mean(1).mean() if n + l > 0 else np.nan
    return lags, out


def coherence_lowbeta(a, b):
    """Ensemble 13-20 Hz magnitude-squared coherence (avg cross-/auto-spectra).

    Fast FFT version: Hann-tapered rFFT per trial, average Sxy/Sxx/Syy over
    trials, coh = |Sxy|^2 / (Sxx*Syy).  a, b : (n_ep, n_win).
    """
    n = a.shape[1]
    if n < 8:
        return np.nan
    w = np.hanning(n)
    A = np.fft.rfft((a - a.mean(1, keepdims=True)) * w, axis=1)
    B = np.fft.rfft((b - b.mean(1, keepdims=True)) * w, axis=1)
    f = np.fft.rfftfreq(n, 1.0 / FS)
    Sxy = (A * np.conj(B)).mean(0)
    Sxx = (A * np.conj(A)).mean(0).real
    Syy = (B * np.conj(B)).mean(0).real
    coh = (np.abs(Sxy) ** 2) / (Sxx * Syy + 1e-30)
    m = (f >= 13) & (f <= 20)
    return float(coh[m].mean()) if m.any() else np.nan


def gc_base_task(a, b, tvec):
    """Moving-window Geweke GC (win 40 ms, order 10) averaged within the
    baseline and task crops.  Both crops use the SAME 20-sample window so the
    finite-sample GC bias is identical between conditions (a single big-window
    MVAR would bias the shorter baseline upward).  Returns
    (gc_xy_base, gc_xy_task, gc_yx_base, gc_yx_task) in low_beta.
    """
    n_ep = min(a.shape[0], b.shape[0])
    ws = int(round(0.04 * FS))                       # 20 samples
    a = a[:n_ep]; b = b[:n_ep]

    def crop_gc(lo, hi):
        m = win(tvec, lo, hi)
        X = np.stack([a[:, m], b[:, m]], 1)          # (n_ep, 2, n_win)
        res = moving_window_pairwise_gc(X, order=10, freqs=FREQS, fs=FS,
                                        win_samples=ws, step=1)
        fxy = band_average(res['f_xy'], FREQS, DEFAULT_BANDS)[BAND]
        fyx = band_average(res['f_yx'], FREQS, DEFAULT_BANDS)[BAND]
        return fxy.mean(), fyx.mean()

    gxb, gyb = crop_gc(*BASE)
    gxt, gyt = crop_gc(*TASK)
    return gxb, gxt, gyb, gyt


# primary pair whose CCF is stored for the example figure
_PRIMARY = ({'Temporal', 'Inferior_Frontal'}, {'awfa-lh', 'ifc-lh'})


def _combo_metrics(subj, sp, ni, nj, a, b, tvec):
    """All baseline-vs-task metrics for one (subject, space, channel pair).
    Module-level so it can run in a joblib process pool (GC dominates cost).
    Returns (row_dict, ccf_key_or_None, ccf_tuple_or_None).
    """
    mb = win(tvec, *BASE); mt = win(tvec, *TASK)
    lb, ob = ccf_ensemble(a[:, mb], b[:, mb], MAXLAG)
    _lt, ot = ccf_ensemble(a[:, mt], b[:, mt], MAXLAG)
    vb = np.array([a[:, mb].var(1).mean(), b[:, mb].var(1).mean()])
    vt = np.array([a[:, mt].var(1).mean(), b[:, mt].var(1).mean()])
    cohb = coherence_lowbeta(a[:, mb], b[:, mb])
    coht = coherence_lowbeta(a[:, mt], b[:, mt])
    gxb, gxt, gyb, gyt = gc_base_task(a, b, tvec)
    row = dict(
        subject=subj, space=sp, pair=f'{ni}~{nj}',
        var_ratio=float((vt / vb).mean()),
        r0_base=float(ob[MAXLAG]), r0_task=float(ot[MAXLAG]),
        ccf_peak_lag_base=int(lb[np.nanargmax(np.abs(ob))]),
        ccf_peak_base=float(ob[np.nanargmax(np.abs(ob))]),
        ccf_shape_sim=float(np.corrcoef(ob, ot)[0, 1]),
        coh_lb_base=cohb, coh_lb_task=coht,
        gc_base=float((gxb + gyb) / 2), gc_task=float((gxt + gyt) / 2),
    )
    ccf = ((subj, sp), (lb, ob, ot)) if {ni, nj} in _PRIMARY else (None, None)
    return row, ccf[0], ccf[1]


def shared_fraction(chans, t, window):
    """Top-eigenvalue fraction of the multi-channel covariance in `window`.

    chans: list of (n_ep, n_t) arrays.  Concatenate trials, covariance across
    channels, return lambda_max / sum(lambda).  High => channels collinear.
    """
    m = win(t, *window)
    M = np.stack([c[:, m].reshape(-1) for c in chans], 0)   # (n_chan, n_ep*n_win)
    C = np.cov(M)
    ev = np.linalg.eigvalsh(C)
    return float(ev[-1] / ev.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', default='overtProd')
    ap.add_argument('--stim-class', default='prodDiff')
    ap.add_argument('--n-jobs', type=int, default=10)
    args = ap.parse_args()

    files = sorted(glob.glob(str(RED_ROOT / f'*_{args.task}_{args.stim_class}.npz')))
    if not files:
        print(f'No reduced-TS caches in {RED_ROOT} — run baseline_ts_extract.py first.'); return
    print(f'{len(files)} subjects; baseline={BASE} task={TASK}; n_jobs={args.n_jobs}')

    # Build the flat task list (one per subject x pair x space); GC dominates
    # cost, so fan the combos out across processes.
    shared_rows, tasks = [], []
    for f in files:
        subj = os.path.basename(f).split('_')[0]
        try:                                    # skip a cache still being written
            d = np.load(f, allow_pickle=True)
        except Exception as e:
            print(f'  {subj}: unreadable ({type(e).__name__}) — skip (still writing?)'); continue
        tvec = d['times']
        src_rois = list(d['src_rois']); sen_names = list(d['sen_names'])
        spaces = {'sensor': {n: d[f'sen__{n}'].astype(float) for n in sen_names},
                  'source': {r: d[f'src__{r}'].astype(float) for r in src_rois}}
        names = {'sensor': sen_names, 'source': src_rois}
        for sp in ('sensor', 'source'):
            sf = shared_fraction([spaces[sp][n] for n in names[sp]], tvec, BASE)
            shared_rows.append(dict(subject=subj, space=sp, shared_frac_base=sf))
        for (i, j) in combinations(range(len(src_rois)), 2):
            for sp in ('sensor', 'source'):
                nm = names[sp]
                tasks.append((subj, sp, nm[i], nm[j],
                              spaces[sp][nm[i]], spaces[sp][nm[j]], tvec))
        d.close()
    print(f'  {len(tasks)} (subject x pair x space) combos -> computing GC...')

    out = Parallel(n_jobs=args.n_jobs, prefer='processes')(
        delayed(_combo_metrics)(*t) for t in tasks)
    rows = [r for r, _k, _v in out]
    ccf_store = {k: v for _r, k, v in out if k is not None}

    met = pd.DataFrame(rows); shr = pd.DataFrame(shared_rows)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    met.to_csv(OUT_ROOT / 'ts_baseline_metrics.csv', index=False)
    shr.to_csv(OUT_ROOT / 'ts_shared_component.csv', index=False)
    print(f'\nwrote {OUT_ROOT}/ts_baseline_metrics.csv ({len(met)} rows), ts_shared_component.csv')

    _figures(met, shr, ccf_store)
    _print_summary(met, shr)


def _figures(met, shr, ccf_store):
    C = {'sensor': '#2166ac', 'source': '#b2182b'}
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))

    def paired(axx, col_base, col_task, title, ylab):
        for sp in ('sensor', 'source'):
            sub = met[met.space == sp]
            g = sub.groupby('subject')[[col_base, col_task]].mean()
            x = {'sensor': 0, 'source': 1}[sp]
            for _, r in g.iterrows():
                axx.plot([x - .15, x + .15], [r[col_base], r[col_task]], '-', color=C[sp], alpha=.35, lw=.8)
            axx.plot(np.full(len(g), x - .15), g[col_base], 'o', color=C[sp], ms=4)
            axx.plot(np.full(len(g), x + .15), g[col_task], 's', color=C[sp], ms=4, mfc='white')
        axx.set_xticks([0, 1]); axx.set_xticklabels(['SENSOR', 'SOURCE'])
        axx.set_title(title); axx.set_ylabel(ylab)
        axx.plot([], [], 'o', color='k', label='baseline'); axx.plot([], [], 's', color='k', mfc='white', label='task')
        axx.legend(fontsize=8)

    # 1. variance ratio task/base
    for sp in ('sensor', 'source'):
        v = met[met.space == sp].groupby('subject')['var_ratio'].mean()
        x = {'sensor': 0, 'source': 1}[sp]
        ax[0, 0].scatter(np.full(len(v), x) + np.random.RandomState(0).uniform(-.08, .08, len(v)), v, color=C[sp], s=35, edgecolor='k', lw=.3)
        ax[0, 0].hlines(v.mean(), x - .2, x + .2, color=C[sp], lw=2.5)
    ax[0, 0].axhline(1, ls='--', color='.5'); ax[0, 0].set_xticks([0, 1]); ax[0, 0].set_xticklabels(['SENSOR', 'SOURCE'])
    ax[0, 0].set_title('A. Signal variance task/baseline ratio\n(>1 = quiet baseline that rises)'); ax[0, 0].set_ylabel('task/baseline variance')

    # 2. GC base vs task
    paired(ax[0, 1], 'gc_base', 'gc_task', 'B. GC (low_beta) baseline vs task\nsensor rises; source flat', 'GC')

    # 3. CCF shape similarity base-vs-task
    for sp in ('sensor', 'source'):
        v = met[met.space == sp].groupby('subject')['ccf_shape_sim'].mean()
        x = {'sensor': 0, 'source': 1}[sp]
        ax[0, 2].scatter(np.full(len(v), x) + np.random.RandomState(1).uniform(-.08, .08, len(v)), v, color=C[sp], s=35, edgecolor='k', lw=.3)
        ax[0, 2].hlines(v.mean(), x - .2, x + .2, color=C[sp], lw=2.5)
    ax[0, 2].set_xticks([0, 1]); ax[0, 2].set_xticklabels(['SENSOR', 'SOURCE'])
    ax[0, 2].set_title('C. CCF shape similarity baseline vs task\n(~1 = same lag coupling at rest & task = fixed/leaked)'); ax[0, 2].set_ylabel('corr(CCF_base, CCF_task)')

    # 4. coherence base vs task
    paired(ax[1, 0], 'coh_lb_base', 'coh_lb_task', 'D. Low-beta coherence baseline vs task', 'coherence')

    # 5. shared-component fraction baseline
    for sp in ('sensor', 'source'):
        v = shr[shr.space == sp]['shared_frac_base']
        x = {'sensor': 0, 'source': 1}[sp]
        ax[1, 1].scatter(np.full(len(v), x) + np.random.RandomState(2).uniform(-.08, .08, len(v)), v, color=C[sp], s=35, edgecolor='k', lw=.3)
        ax[1, 1].hlines(v.mean(), x - .2, x + .2, color=C[sp], lw=2.5)
    ax[1, 1].set_xticks([0, 1]); ax[1, 1].set_xticklabels(['SENSOR', 'SOURCE'])
    ax[1, 1].set_title('E. Baseline shared-component fraction\n(top eigenvalue / total; high = collinear ROIs)'); ax[1, 1].set_ylabel('lambda_max fraction')

    # 6. GC baseline floor vs shared fraction (source)
    g = met[met.space == 'source'].groupby('subject')['gc_base'].mean()
    s = shr[shr.space == 'source'].set_index('subject')['shared_frac_base']
    idx = g.index.intersection(s.index)
    ax[1, 2].scatter(s[idx], g[idx], color=C['source'], s=45, edgecolor='k', lw=.4)
    for sj in idx:
        ax[1, 2].annotate(sj[-2:], (s[sj], g[sj]), fontsize=7)
    ax[1, 2].set_title('F. SOURCE baseline GC floor vs shared-component\n(does collinearity drive the floor?)')
    ax[1, 2].set_xlabel('baseline shared-component fraction'); ax[1, 2].set_ylabel('baseline GC (low_beta)')

    fig.suptitle('Sensor vs LCMV-source time series: what makes the GC baseline low (or not)', fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_ROOT / 'fig_group_summary.png', dpi=130, bbox_inches='tight')
    plt.close(fig)

    # CCF example figure
    ex = [s for s in EXAMPLES if (s, 'sensor') in ccf_store]
    if ex:
        fig, ax = plt.subplots(len(ex), 2, figsize=(11, 3 * len(ex)), squeeze=False)
        for k, subj in enumerate(ex):
            for c, sp in enumerate(('sensor', 'source')):
                if (subj, sp) not in ccf_store:
                    continue
                lag, ob, ot = ccf_store[(subj, sp)]
                ax[k, c].plot(lag * 2, ob, color='#1a1a1a', lw=1.6, label='baseline')
                ax[k, c].plot(lag * 2, ot, color='#d6604d', lw=1.4, ls='--', label='task')
                ax[k, c].axvline(0, color='.7', lw=.6); ax[k, c].axhline(0, color='.7', lw=.6)
                ax[k, c].set_title(f'{subj}  {sp}  (Temporal~IFG)', fontsize=10)
                if k == len(ex) - 1: ax[k, c].set_xlabel('lag (ms)  [+ = left leads right]')
                if c == 0: ax[k, c].set_ylabel('cross-corr')
        ax[0, 0].legend(fontsize=9)
        fig.suptitle('Cross-correlation lag structure: baseline vs task (the coupling GC reads)', fontsize=13)
        fig.tight_layout()
        fig.savefig(OUT_ROOT / 'fig_ccf_examples.png', dpi=130, bbox_inches='tight')
        plt.close(fig)
    print(f'wrote {OUT_ROOT}/fig_group_summary.png , fig_ccf_examples.png')


def _print_summary(met, shr):
    print('\n=== SUMMARY (subject-mean over pairs) ===')
    for sp in ('sensor', 'source'):
        s = met[met.space == sp].groupby('subject').mean(numeric_only=True)
        sf = shr[shr.space == sp]['shared_frac_base']
        print(f'  {sp.upper():6s} var_ratio={s.var_ratio.mean():.2f}  '
              f'gc task/base={s.gc_task.mean()/s.gc_base.mean():.2f}  '
              f'ccf_shape_sim={s.ccf_shape_sim.mean():.2f}  '
              f'coh base/task={s.coh_lb_base.mean():.2f}/{s.coh_lb_task.mean():.2f}  '
              f'shared_frac={sf.mean():.2f}')


if __name__ == '__main__':
    main()
