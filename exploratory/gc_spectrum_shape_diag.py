#!/usr/bin/env python
"""Full 1-30 Hz GC spectrum through the PRODUCTION path, several estimator
configs side by side — the diagnostic for "why is low beta flat?".

Band-averaged result files cannot show the spectral shape of the estimator,
so this recomputes GC with 1 Hz bins (``bands = {f: [f, f+1)}``) via
``run_granger.compute_subject_gc`` — vertex cache -> fixed-filter PCs ->
block GC, exactly as run_granger.py — and averages the spectrum over the
windows of an interior stretch of the epoch, then over pairs, directions and
subjects.  Configs compared (edit CONFIGS):

    PC1          500 Hz / 40 ms / order 10      bivariate, 20 ms memory
    PC2 (pmc 3)  500 Hz / 40 ms / order 10      the production fs500 arm
    PC2 (pmc 3)  same, --normalize demean       ERP removed: is the theta
                                                excess a shared evoked offset?
    PC2 (pmc 3)  same, no per-trial demean      raw BSMART-style input
    PC2 (pmc 3)  500 Hz / 80 ms / order 25      same fs, 50 ms memory
    PC2 (pmc 3)  200 Hz / 80 ms / order 10      the old 80 ms arm

What to read off: if low beta sits in a trough between a theta lobe and a
rise toward 30 Hz only in the 20 ms-memory arms, it is model memory; if the
theta lobe collapses when the ERP is removed, it is the within-window offset
(cf. GC_qc/demean_ab); if PC1 is flat and every PC2 arm has the trough, it
is the block aggregation.

HEAVY: one MVAR fit per window per pair per config per subject on the
vertex caches.  RUN ON THE WORKSTATION, not locally.

    conda activate mne
    python exploratory/gc_spectrum_shape_diag.py --task perception \\
        --stim-class percDiff --n-jobs 8
    python exploratory/gc_spectrum_shape_diag.py --task overtProd \\
        --stim-class prodDiff --n-jobs 8

Output (GC_source_space/_figures_final_review/):
    gc_spectrum_shape_production_{task}_{stim}.png
    gc_spectrum_shape_production_{task}_{stim}.npz    per-subject spectra
    gc_spectrum_shape_production_{task}_{stim}.csv    band means per config
"""
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import run_granger as rg                                       # noqa: E402
from config import SUBJECT_IDS                                 # noqa: E402

ROIS = ['awfa-lh', 'ifc-lh', 'pmc-lh', 'tpc-lh']
FREQS = np.arange(1, 31.0)
BINS = {f'f{int(f):02d}': (f, f + 1) for f in FREQS}          # 1 Hz, half-open
BANDS = {'theta': (4, 8), 'alpha': (8, 12), 'low_beta': (12, 18),
         'high_beta': (18, 31)}
# interior stretch (s): clear of both epoch edges and of the leading-edge
# inflation; perception = post-onset, overtProd = pre-event plateau
STRETCH = {'perception': (0.10, 0.40), 'overtProd': (-1.25, -0.65)}
CONFIGS = {
    'PC1 | 500 Hz / 40 ms / order 10 (memory 20 ms)':
        dict(n_pcs=1, n_pcs_roi=None, target_fs=500, win_ms=40, order=10),
    'PC2, pmc 3 | 500 Hz / 40 ms / order 10 (memory 20 ms)  PRODUCTION':
        dict(n_pcs=2, n_pcs_roi=['pmc-lh=3'], target_fs=500, win_ms=40, order=10),
    'PC2, pmc 3 | 500 Hz / 40 ms / order 10, ERP removed (normalize demean)':
        dict(n_pcs=2, n_pcs_roi=['pmc-lh=3'], target_fs=500, win_ms=40, order=10,
             normalize='demean'),
    'PC2, pmc 3 | 500 Hz / 40 ms / order 10, no per-trial demean':
        dict(n_pcs=2, n_pcs_roi=['pmc-lh=3'], target_fs=500, win_ms=40, order=10,
             demean_trials=False),
    'PC2, pmc 3 | 500 Hz / 80 ms / order 25 (memory 50 ms)':
        dict(n_pcs=2, n_pcs_roi=['pmc-lh=3'], target_fs=500, win_ms=80, order=25),
    'PC2, pmc 3 | 200 Hz / 80 ms / order 10 (memory 50 ms)  OLD':
        dict(n_pcs=2, n_pcs_roi=['pmc-lh=3'], target_fs=200, win_ms=80, order=10),
}


def _spectrum(d):
    """{bin: (n_pairs, n_win)} -> (n_pairs, n_freqs) window-mean spectrum."""
    if isinstance(d, dict):
        return np.stack([np.nanmean(d[b], axis=-1) for b in BINS], -1)
    return np.nanmean(d, axis=-1)


def one_subject(subj, args):
    npz = (os.path.join(args.cache_root, f'{subj}_{args.task}_{args.stim_class}.npz')
           if args.cache_root else
           rg.find_cached_npz(args.task, args.method, args.atlas, args.feature_mode,
                              True, subj, args.stim_class))
    if npz is None or not os.path.exists(str(npz)):
        print(f'  {subj}: no vertex cache — skipped', flush=True)
        return subj, None
    roi_data, y, times, sfreq = rg._load_cached_roi_data(
        npz, feature_mode=args.feature_mode, roi_subset=ROIS)
    if roi_data is None or len(roi_data) < 2:
        print(f'  {subj}: ROIs missing — skipped', flush=True)
        return subj, None
    tmin, tmax = STRETCH[args.task]
    out = {}
    t0 = time.time()
    for name, c in CONFIGS.items():
        win = int(round(c['win_ms'] / 1000.0 * c['target_fs']))
        r = rg.compute_subject_gc(
            roi_data, times, sfreq, order=c['order'], win_ms=c['win_ms'],
            target_fs=c['target_fs'], step=max(1, win // args.step_div),
            freqs=FREQS, bands=BINS, normalize=c.get('normalize', 'none'),
            demean_trials=c.get('demean_trials', True), trgc=False,
            tmin=tmin, tmax=tmax, gc_mode='pairwise', n_jobs=1,
            diagnostics=False, n_pcs=c['n_pcs'], n_pcs_roi=c['n_pcs_roi'])
        out[name] = np.concatenate([_spectrum(r['fxy']), _spectrum(r['fyx'])], 0)
        out[name + '__n_comp'] = np.asarray(r['n_comp'])
    print(f'  {subj}: done in {(time.time() - t0) / 60:.1f} min', flush=True)
    return subj, out


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--task', required=True, choices=['perception', 'overtProd'])
    p.add_argument('--stim-class', required=True)
    p.add_argument('--method', default='LCMV')
    p.add_argument('--atlas', default='custom')
    p.add_argument('--feature-mode', default='vertex_selectkbest')
    p.add_argument('--subjects', nargs='+', default=None)
    p.add_argument('--cache-root', default=None,
                   help='directory holding {subj}_{task}_{stim}.npz vertex '
                        'caches (default: config lookup via find_cached_npz)')
    p.add_argument('--step-div', type=int, default=2,
                   help='window step = window / step-div samples (2 = half-'
                        'overlapping windows; 1 = non-overlapping)')
    p.add_argument('--n-jobs', type=int, default=8, help='subjects in parallel')
    p.add_argument('--out-dir', default=os.path.join(
        str(rg.GC_OUTPUT_ROOT), '_figures_final_review'))
    args = p.parse_args()

    subs = args.subjects or list(SUBJECT_IDS)
    print(f'{args.task}/{args.stim_class}: {len(subs)} subjects, '
          f'{len(CONFIGS)} configs, stretch {STRETCH[args.task]} s, '
          f'{args.n_jobs} in parallel')
    res = Parallel(n_jobs=args.n_jobs)(delayed(one_subject)(s, args) for s in subs)
    res = [(s, o) for s, o in res if o is not None]
    if not res:
        raise SystemExit('nothing computed')

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.join(args.out_dir,
                        f'gc_spectrum_shape_production_{args.task}_{args.stim_class}')
    np.savez_compressed(stem + '.npz', freqs=FREQS, subjects=[s for s, _ in res],
                        configs=list(CONFIGS),
                        **{f'spec_{i}': np.stack([o[k] for _, o in res])
                           for i, k in enumerate(CONFIGS)})

    rows = []
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for k in CONFIGS:
        S = np.concatenate([o[k] for _, o in res], 0)            # edges x freqs
        mu, se = S.mean(0), S.std(0, ddof=1) / np.sqrt(len(S))
        ax.plot(FREQS, mu, lw=2, label=k)
        ax.fill_between(FREQS, mu - se, mu + se, alpha=.15, lw=0)
        bm = {b: float(mu[(FREQS >= lo) & (FREQS < hi)].mean())
              for b, (lo, hi) in BANDS.items()}
        rows.append(dict(config=k, n_edges=len(S), **bm,
                         low_beta_over_theta=bm['low_beta'] / bm['theta'],
                         n_comp=str(res[0][1][k + '__n_comp'].tolist())))
    for lo, hi, c in ((4, 8, '#0072B2'), (12, 18, '#009E73'), (18, 30, '#E69F00')):
        ax.axvspan(lo, hi, color=c, alpha=.08, lw=0)
    ax.set(xlabel='frequency (Hz)', ylabel='GC',
           title=f'Spectral GC through the production path '
                 f'(vertex cache → fixed-filter PCs → block GC)\n'
                 f'{args.task}/{args.stim_class}, {len(res)} subj × 6 pairs × 2 dirs, '
                 f'windows {STRETCH[args.task][0]:+.2f}..{STRETCH[args.task][1]:+.2f} s')
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(stem + '.png', dpi=150)
    df = pd.DataFrame(rows)
    df.to_csv(stem + '.csv', index=False)
    print(df.round(3).to_string(index=False))
    print('wrote', stem + '.{png,npz,csv}')


if __name__ == '__main__':
    main()
