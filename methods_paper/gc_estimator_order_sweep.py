#!/usr/bin/env python3
"""Plain GC vs TRGC across MVAR model orders — full source-space sweep.

Workstation run for the methods paper.  Two screen results motivate it:
  * plain source GC is dominated by a task-invariant MIXING pedestal, and
    Diff-TRGC (Haufe 2013; Winkler 2016) strips it and recovers a directional,
    task-modulated, pathway-specific signal that plain GC washes out;
  * the single-pair plain-GC task-vs-baseline contrast is FRAGILE to MVAR order
    (it can flip sign between adjacent orders).

This computes, for all custom-atlas ROI pairs, both tasks, all 20 subjects:
  - plain spectral GC (both directions) AND Diff-TRGC (net), from ONE engine
    call per (subject, pair, order) with trgc=True,
  - across a sweep of MVAR model orders,
  - with the ROI virtual channel fixed to FIXPC1 (the pinned aggregation:
    the ROIs are ~rank-1, so FIXPC1 ~= mean ~= mean_flip; see the vsensor screen),
  - reporting the group task-vs-baseline contrast per (pair, order, band) for
    each estimator, plus an all-pairs aggregate and a BIC-optimal-order
    diagnostic (a principled-order recommendation).

Design mirrors the pipeline's two-stage split:
  Stage A (reduce, RAM-heavy, subject-parallel): resolve the vertex TS through
    config.find_cached_npz (config.env ROI_TIMESERIES_EXTERNAL), FIXPC1-reduce
    every ROI to a virtual channel, resample to --target-fs, cache the tiny
    reduced npz (reused on re-runs so the big /media reads happen once).
  Stage B (GC sweep, light, (subject x pair x order)-parallel): read the reduced
    cache, run moving_window_pairwise_gc(trgc=True), band-average, contrast.

Path resolution is config.env-aware, so on the workstation NO path flags are
needed (set EEG_PROJECT_ROOT + ROI_TIMESERIES_EXTERNAL in config.env).  Use
--no-leakage later for the raw arm (needs raw TS generated first).

Outputs (under GC_sensor_vs_source_baseline_check/estimator_order_sweep/); every
file carries {task}_{stim}{sfx} so no run overwrites another:
  {task}_{stim}{sfx}_estimator_order.csv        per (subj, pair, order, band) raw
  {task}_{stim}{sfx}_estimator_order_group.csv  group summary
  {task}_{stim}{sfx}_estimator_order.png        low_beta contrast vs order, per estimator
  {task}_{stim}{sfx}_estimator_order_allband.png all-pairs-mean contrast vs order, all bands
  {task}_{stim}{sfx}_bic_order.csv              BIC-optimal order distribution

Usage (workstation; config.env set):
    conda activate mne
    # both stim-classes for a task in ONE call (each writes its own reports)
    python methods_paper/gc_estimator_order_sweep.py --task overtProd \
        --stim-class prodDiff percDiff --reduce-jobs 16 --gc-jobs 60
    python methods_paper/gc_estimator_order_sweep.py --task perception \
        --stim-class percDiff --reduce-jobs 16 --gc-jobs 60
Local smoke test (small):
    python methods_paper/gc_estimator_order_sweep.py --task overtProd --stim-class prodDiff \
        --subjects EEGPROD4003 EEGPROD4005 --pairs awfa-lh:ifc-lh --orders 8 10 \
        --reduce-jobs 2 --gc-jobs 4
"""
import os, sys, glob, argparse, warnings, itertools
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from config import DECODE_OUTPUT_ROOT, SUBJECT_IDS
from decoding_io import _load_cached_roi_data
from granger import (reduce_roi_first_pc, moving_window_pairwise_gc, band_average,
                     DEFAULT_BANDS, fit_mvar)
from run_granger import resample_channels

OUT = DECODE_OUTPUT_ROOT.parent / 'GC_sensor_vs_source_baseline_check' / 'estimator_order_sweep'
OUT.mkdir(parents=True, exist_ok=True)
RED = OUT / 'reduced_fixpc1'                       # reduced virtual-channel cache
RED.mkdir(parents=True, exist_ok=True)

# Fallback template (only used if config.find_cached_npz does not resolve).
DEFAULT_SRC = ('/mnt/s/Research/SpeechProduction/DECODE_source_space_timeseries/'
               '{task}/{method}/{atlas}/vertex/{leak}/{subj}_{task}_{stim}.npz')

FREQS = np.arange(1, 31)
BANDS = list(DEFAULT_BANDS)                         # theta, alpha, low_beta, high_beta
DEFAULT_ORDERS = [5, 8, 10, 12, 14, 16]
TASK_CFG = {
    'overtProd':  dict(base=(-1.50, -1.40), task=(-0.30, 0.30), onset='production onset'),
    'perception': dict(base=(-0.20, -0.10), task=(0.0, 0.30),  onset='auditory onset'),
}
EST = [('fwd', 'plain GC A->B'), ('rev', 'plain GC B->A'), ('trgc', 'TRGC net A->B')]


# ── Stage A: resolve + FIXPC1-reduce + resample + cache ──
def resolve_src(subj, task, stim, method, atlas, leakage, src_tmpl):
    p = config.find_cached_npz(task, method, atlas, 'vertex_selectkbest',
                               leakage, subj, stim)
    if p is not None:
        return str(p)
    leak = 'leakage_corrected' if leakage else 'raw'
    return src_tmpl.format(task=task, subj=subj, stim=stim, method=method,
                           atlas=atlas, leak=leak)


def reduce_subject(subj, task, stim, method, atlas, leakage, target_fs, src_tmpl,
                   overwrite):
    leak = 'leakage_corrected' if leakage else 'raw'
    cache = RED / f'{task}_{atlas}_{leak}' / f'{subj}_{task}_{stim}.npz'
    if cache.exists() and not overwrite:
        return ('cached', subj, str(cache))
    path = resolve_src(subj, task, stim, method, atlas, leakage, src_tmpl)
    if not os.path.exists(path):
        return ('missing', subj, path)
    roi_data, _y, times, sfreq = _load_cached_roi_data(Path(path), 'vertex_selectkbest')
    if roi_data is None:
        return ('missing', subj, path)
    rois = list(roi_data.keys())
    vcs = {r: reduce_roi_first_pc(np.asarray(roi_data[r], float)) for r in rois}
    del roi_data
    V = np.stack([vcs[r] for r in rois], 0)                    # (n_roi, n_ep, n_t)
    V, fs = resample_channels(V, sfreq, target_fs)
    t = times[0] + np.arange(V.shape[2]) / fs
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, times=t, sfreq=fs, rois=np.array(rois),
                        **{f'vc__{r}': V[i] for i, r in enumerate(rois)})
    os.replace(tmp, cache)
    return ('reduced', subj, str(cache))


def load_reduced(subj, task, stim, atlas, leakage):
    leak = 'leakage_corrected' if leakage else 'raw'
    cache = RED / f'{task}_{atlas}_{leak}' / f'{subj}_{task}_{stim}.npz'
    if not cache.exists():
        return None
    d = np.load(cache, allow_pickle=True)
    rois = [str(r) for r in d['rois']]
    return {r: d[f'vc__{r}'].astype(float) for r in rois}, d['times'], float(d['sfreq'])


# ── Stage B: one (subj, pair, order) GC + TRGC ──
def win(t, lo, hi):
    return (t >= lo) & (t <= hi)


def gc_cell(subj, task, stim, atlas, leakage, pair, order, gc_win_ms, gc_step,
            base, task_win):
    got = load_reduced(subj, task, stim, atlas, leakage)
    if got is None:
        return None
    vcs, tvec, fs = got
    ra, rb = pair.split(':')
    if ra not in vcs or rb not in vcs:
        return None
    a, b = vcs[ra], vcs[rb]
    n = min(a.shape[0], b.shape[0]); a, b = a[:n], b[:n]
    ws = int(round(gc_win_ms / 1000.0 * fs))
    if ws <= order + 1 or ws > a.shape[1]:
        return None
    res = moving_window_pairwise_gc(np.stack([a, b], 1), order=order, freqs=FREQS,
                                    fs=fs, win_samples=ws, step=gc_step, trgc=True)
    starts = np.arange(0, a.shape[1] - ws + 1, gc_step)
    wm = tvec[starts] * 1000.0
    mb = win(wm, base[0] * 1000, base[1] * 1000)
    mt = win(wm, task_win[0] * 1000, task_win[1] * 1000)
    if not (mb.any() and mt.any()):
        return None
    rows = []
    ba_fxy = band_average(res['f_xy'], FREQS, DEFAULT_BANDS)
    ba_fyx = band_average(res['f_yx'], FREQS, DEFAULT_BANDS)
    ba_dxy = band_average(res['d_xy'], FREQS, DEFAULT_BANDS)
    for band in BANDS:
        row = dict(subj=subj, pair=pair, order=order, band=band)
        for tag, g in (('fwd', ba_fxy[band]), ('rev', ba_fyx[band]), ('trgc', ba_dxy[band])):
            row[f'{tag}_base'] = float(g[mb].mean())
            row[f'{tag}_task'] = float(g[mt].mean())
            row[f'{tag}_contrast'] = float(g[mt].mean() - g[mb].mean())
        rows.append(row)
    return rows


# ── BIC-optimal order diagnostic (whole-epoch pair MVAR) ──
def bic_order(subj, task, stim, atlas, leakage, pair, orders, base, task_win):
    got = load_reduced(subj, task, stim, atlas, leakage)
    if got is None:
        return None
    vcs, tvec, fs = got
    ra, rb = pair.split(':')
    if ra not in vcs or rb not in vcs:
        return None
    a, b = vcs[ra], vcs[rb]
    n = min(a.shape[0], b.shape[0])
    seg = np.stack([a[:n], b[:n]], 1)                          # full epoch
    N = seg.shape[0] * seg.shape[2]                            # data points
    best, best_bic = None, np.inf
    for p in orders:
        try:
            _A, S = fit_mvar(seg, p)
            ll = N * np.log(np.linalg.det(S) + 1e-30)
            bic = ll + (2 ** 2) * p * np.log(N)                # k = n_ch^2 * p params
            if bic < best_bic:
                best_bic, best = bic, p
        except Exception:
            continue
    return dict(subj=subj, pair=pair, bic_order=best)


def default_pairs(rois):
    return [f'{a}:{b}' for a, b in itertools.combinations(sorted(rois), 2)]


def group_summary(df):
    recs = []
    for (pair, order, band), g in df.groupby(['pair', 'order', 'band']):
        rec = dict(pair=pair, order=order, band=band, n=len(g))
        for tag in ('fwd', 'rev', 'trgc'):
            c = g[f'{tag}_contrast'].values
            rec[f'{tag}_base'] = float(g[f'{tag}_base'].mean())
            rec[f'{tag}_task'] = float(g[f'{tag}_task'].mean())
            rec[f'{tag}_contrast'] = float(np.mean(c))
            rec[f'{tag}_sem'] = float(np.std(c) / np.sqrt(max(len(c), 1)))
            rec[f'{tag}_npos'] = int((c > 0).sum())
            t, p = stats.ttest_1samp(c, 0.0) if len(c) > 1 else (np.nan, np.nan)
            rec[f'{tag}_t'] = float(t); rec[f'{tag}_p'] = float(p)
        recs.append(rec)
    return pd.DataFrame(recs)


def allpairs_curve(df, band):
    """Per-subject mean over pairs, then group -> all-pairs contrast vs order."""
    d = df[df.band == band]
    out = {}
    for tag in ('fwd', 'rev', 'trgc'):
        rows = []
        for order, g in d.groupby('order'):
            per_subj = g.groupby('subj')[f'{tag}_contrast'].mean().values
            t, p = stats.ttest_1samp(per_subj, 0.0) if len(per_subj) > 1 else (np.nan, np.nan)
            rows.append(dict(order=order, contrast=float(per_subj.mean()),
                             sem=float(per_subj.std() / np.sqrt(max(len(per_subj), 1))),
                             npos=int((per_subj > 0).sum()), n=len(per_subj),
                             t=float(t), p=float(p)))
        out[tag] = pd.DataFrame(rows).sort_values('order')
    return out


def fig_lowbeta(summ, df, task, pairs, orders, stem, outdir):
    band = 'low_beta'
    ap = allpairs_curve(df, band)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True)
    for ax, (tag, title) in zip(axes, EST):
        for p in pairs:
            sub = summ[(summ.pair == p) & (summ.band == band)].sort_values('order')
            if sub.empty:
                continue
            ax.plot(sub.order, sub[f'{tag}_contrast'], '-', lw=0.8, alpha=0.45,
                    marker='.', ms=4)
        a = ap[tag]
        ax.errorbar(a.order, a.contrast, yerr=a['sem'], color='k', lw=2.4, marker='o',
                    ms=6, capsize=3, label='all-pairs mean', zorder=5)
        for _, r in a.iterrows():
            if r.p < 0.05:
                ax.annotate('*', (r.order, r.contrast), color='crimson', fontsize=14,
                            ha='center', va='bottom', zorder=6)
        ax.axhline(0, color='k', lw=0.7)
        ax.set_title(title, fontsize=11); ax.set_xlabel('MVAR order')
        ax.grid(alpha=0.25)
    axes[0].set_ylabel('group task - baseline GC (low_beta)')
    axes[0].legend(fontsize=8, loc='best')
    fig.suptitle(f'{task}: plain GC vs TRGC task-baseline contrast across MVAR order '
                 f'(thin = per pair, bold = all-pairs mean, * = p<0.05)\n'
                 f'stable/consistent estimator = flat across order & consistent sign',
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    f = outdir / f'{stem}_estimator_order.png'; fig.savefig(f, dpi=140); plt.close(fig)
    return f


def fig_allband(df, task, stem, outdir):
    fig, axes = plt.subplots(1, len(BANDS), figsize=(4.0 * len(BANDS), 4.2), sharex=True)
    for ax, band in zip(np.atleast_1d(axes), BANDS):
        ap = allpairs_curve(df, band)
        for tag, title in EST:
            a = ap[tag]
            ax.errorbar(a.order, a.contrast, yerr=a['sem'], marker='o', ms=4, lw=1.6,
                        capsize=2, label=title)
        ax.axhline(0, color='k', lw=0.7); ax.set_title(band); ax.set_xlabel('order')
        ax.grid(alpha=0.25)
    np.atleast_1d(axes)[0].set_ylabel('all-pairs mean task - baseline GC')
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.suptitle(f'{task}: all-pairs-mean contrast vs order, per band + estimator',
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    f = outdir / f'{stem}_estimator_order_allband.png'; fig.savefig(f, dpi=140); plt.close(fig)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', default='overtProd', choices=['overtProd', 'perception'])
    ap.add_argument('--stim-class', nargs='+', default=['prodDiff'],
                    help='one or more stim-classes; each runs the full pipeline in turn '
                         'and writes its own task_stimclass-named reports (Stage A caches '
                         'are per stim-class, so nothing is recomputed across them)')
    ap.add_argument('--method', default='LCMV')
    ap.add_argument('--atlas', default='custom')
    ap.add_argument('--subjects', nargs='+', default=list(SUBJECT_IDS))
    ap.add_argument('--pairs', nargs='+', default=None,
                    help='ROI pairs "a:b"; default = all pairs among cache ROIs')
    ap.add_argument('--rois', nargs='+', default=None,
                    help='restrict to these ROIs when auto-building pairs')
    ap.add_argument('--orders', nargs='+', type=int, default=DEFAULT_ORDERS)
    ap.add_argument('--target-fs', type=float, default=500.0)
    ap.add_argument('--gc-win-ms', type=float, default=40.0)
    ap.add_argument('--gc-step', type=int, default=1)
    ap.add_argument('--no-leakage', dest='leakage', action='store_false')
    ap.set_defaults(leakage=True)
    ap.add_argument('--src-root', default=DEFAULT_SRC)
    ap.add_argument('--reduce-jobs', type=int, default=16, help='Stage A (RAM-heavy)')
    ap.add_argument('--gc-jobs', type=int, default=60, help='Stage B (light)')
    ap.add_argument('--overwrite-reduced', action='store_true')
    ap.add_argument('--fresh', action='store_true',
                    help='delete any existing Stage-B checkpoint CSV and recompute from scratch '
                         '(default resumes, skipping subjects already in it)')
    args = ap.parse_args()
    print(f'stim-classes to run: {args.stim_class}')
    for i, stim in enumerate(args.stim_class):
        if len(args.stim_class) > 1:
            print('\n' + '#' * 70
                  + f'\n# stim-class {i + 1}/{len(args.stim_class)}: {args.task}/{stim}\n'
                  + '#' * 70)
        run_one(args, stim)


def run_one(args, stim):
    """Full Stage A + Stage B + reports for ONE stim-class."""
    cfg = TASK_CFG[args.task]
    sfx = '' if args.leakage else '_raw'
    leak = 'leakage_corrected' if args.leakage else 'raw'
    # Every report filename carries task_stimclass{_raw}, so distinct runs never
    # overwrite each other (overtProd_prodDiff_... vs overtProd_percDiff_...).
    stem = f'{args.task}_{stim}{sfx}'
    rep = OUT                               # flat report dir; identity lives in the filename
    rep.mkdir(parents=True, exist_ok=True)
    print(f'{args.task}/{stim} [{leak}] atlas={args.atlas} method={args.method}')
    print(f'  {len(args.subjects)} subj | orders {args.orders} | '
          f'base {cfg["base"]}s task {cfg["task"]}s | win {args.gc_win_ms}ms step {args.gc_step}')
    # Echo the exact target files UP FRONT (before the multi-hour compute) so a
    # stale-code / wrong-flag run is caught in seconds, not after it overwrites.
    outs = [f'{stem}_estimator_order.csv', f'{stem}_estimator_order_group.csv',
            f'{stem}_bic_order.csv', f'{stem}_estimator_order.png',
            f'{stem}_estimator_order_allband.png']
    print(f'  reports -> {rep}/')
    for o in outs:
        print(f'    {o}')
    print('  ^ every filename carries task_stimclass; Ctrl-C now if these look wrong.')

    # ── Stage A ──
    print(f'\n[Stage A] reduce + cache (reduce-jobs={args.reduce_jobs}) ...')
    red = Parallel(n_jobs=args.reduce_jobs, verbose=5)(
        delayed(reduce_subject)(s, args.task, stim, args.method, args.atlas,
                                args.leakage, args.target_fs, args.src_root,
                                args.overwrite_reduced)
        for s in args.subjects)
    ok = [r for r in red if r[0] in ('cached', 'reduced')]
    miss = [r for r in red if r[0] == 'missing']
    subs_ok = [r[1] for r in ok]
    print(f'  reduced/cached: {len(ok)} | missing: {len(miss)}'
          + (f' -> {[m[1] for m in miss]}' if miss else ''))
    if not subs_ok:
        print('  No subjects reduced — check config.env / --src-root.'); return

    # ROI list + pairs from the first available reduced cache
    got = load_reduced(subs_ok[0], args.task, stim, args.atlas, args.leakage)
    rois = list(got[0].keys())
    if args.rois:
        rois = [r for r in rois if r in set(args.rois)]
    pairs = args.pairs if args.pairs else default_pairs(rois)
    print(f'  ROIs: {rois}\n  {len(pairs)} pairs')

    # ── Stage B (resumable: the raw per-cell CSV is the checkpoint, appended
    #    one subject at a time; a re-run skips subjects already in it) ──
    raw_csv = rep / f'{stem}_estimator_order.csv'
    if args.fresh and raw_csv.exists():
        raw_csv.unlink()
    done = set()
    if raw_csv.exists():
        try:
            done = set(pd.read_csv(raw_csv, usecols=['subj'])['subj'].unique())
        except Exception:
            done = set()
    todo = [s for s in subs_ok if s not in done]
    print(f'\n[Stage B] GC + TRGC sweep (gc-jobs={args.gc_jobs}) — '
          f'{len(done)} subj done, {len(todo)} to go'
          + ('  [resuming]' if done else ''))
    for i, s in enumerate(todo):
        cells = Parallel(n_jobs=args.gc_jobs)(
            delayed(gc_cell)(s, args.task, stim, args.atlas, args.leakage,
                             p, o, args.gc_win_ms, args.gc_step, cfg['base'], cfg['task'])
            for p in pairs for o in args.orders)
        rows = [r for cell in cells if cell for r in cell]
        if rows:                                # append this subject; header only if new file
            pd.DataFrame(rows).to_csv(raw_csv, mode='a', header=not raw_csv.exists(),
                                      index=False)
        print(f'  [{i + 1}/{len(todo)}] {s}: {len(rows)} rows -> {raw_csv.name}', flush=True)
    if not raw_csv.exists():
        print('  No GC rows produced — check reduced caches.'); return
    df = pd.read_csv(raw_csv)
    summ = group_summary(df)
    summ.to_csv(rep / f'{stem}_estimator_order_group.csv', index=False)

    # ── BIC diagnostic ──
    bic = Parallel(n_jobs=args.gc_jobs)(
        delayed(bic_order)(s, args.task, stim, args.atlas, args.leakage,
                           p, args.orders, cfg['base'], cfg['task'])
        for s in subs_ok for p in pairs)
    bic_df = pd.DataFrame([b for b in bic if b])
    bic_df.to_csv(rep / f'{stem}_bic_order.csv', index=False)

    # ── console: low_beta all-pairs curve ──
    apc = allpairs_curve(df, 'low_beta')
    print(f'\n=== {args.task} low_beta all-pairs task-minus-baseline vs order ===')
    print(f'{"order":>5} | ' + ' | '.join(f'{t:>22}' for t, _ in EST))
    for o in sorted(df.order.unique()):
        cells = []
        for tag, _ in EST:
            r = apc[tag][apc[tag].order == o]
            if r.empty:
                cells.append(' ' * 22); continue
            r = r.iloc[0]
            star = '*' if r.p < 0.05 else ' '
            cells.append(f'{r.contrast:+.4f} ({int(r.npos)}/{int(r.n)}) p={r.p:.3f}{star}')
        print(f'{int(o):>5} | ' + ' | '.join(cells))
    if not bic_df.empty:
        vc = bic_df.bic_order.value_counts().sort_index()
        print(f'\nBIC-optimal order (pairs x subj): '
              + ', '.join(f'p{int(k)}:{int(v)}' for k, v in vc.items())
              + f'  | median={int(bic_df.bic_order.median())}')

    f1 = fig_lowbeta(summ, df, args.task, pairs, args.orders, stem, rep)
    f2 = fig_allband(df, args.task, stem, rep)
    print(f'\nwrote {rep / f"{stem}_estimator_order.csv"}')
    print(f'wrote {rep / f"{stem}_estimator_order_group.csv"}')
    print(f'wrote {rep / f"{stem}_bic_order.csv"}')
    print(f'wrote {f1}\nwrote {f2}')


if __name__ == '__main__':
    main()
