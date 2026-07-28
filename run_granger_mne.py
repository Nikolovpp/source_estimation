#!/usr/bin/env python3
"""Source-space Granger causality runner — MNE-Connectivity state-space GC.

The MNE counterpart of ``run_granger.py``. Same inputs (the vertex ROI
timeseries caches written by ``run_source_localize.py``), same output schema
(so ``granger_stats.py`` / plotting are reusable), but the estimator is
``mne_connectivity.spectral_connectivity_epochs(method=["gc","gc_tr"])`` in
continuous-wavelet mode instead of the BSMART-parametric AR — the citable,
order-stable route (see ``GC_fundamentals/mne_vs_bsmart_findings.md``).

Per subject:
  1. load the requested ROIs' vertex timeseries from the .npz cache
  2. reduce each ROI to one virtual channel (fixed first-PC filter)
  3. resample to ``--target-fs``, crop to the task window, per-trial demean
  4. time-resolved state-space GC (cwt) on a fixed 4-30 Hz grid, for the
     requested ROI pairs, both directions, + Diff-TRGC
  5. band-average into theta/alpha/low-beta/high-beta

Parallelism: subject-parallel (joblib), BLAS pinned to 1 thread per worker.
Each subject is a couple of MNE calls, so the natural grain is one subject
per worker.

Config presets (this is what the config comparison uses)
--------------------------------------------------------
    --preset A   ->  MO 25 / SW 250 ms / fs 200 Hz   (canonical)
    --preset B   ->  MO 15 / SW  60 ms / fs 500 Hz   (fast/short window)

Explicit ``--gc-n-lags`` / ``--win-ms`` / ``--target-fs`` override a preset.

Usage
-----
    # both configs, all 20 subjects, the speech-network pairs
    python run_granger_mne.py --task overtProd --stim-class prodDiff \\
        --method LCMV --atlas custom --feature-mode vertex_selectkbest \\
        --leakage-correction --preset A --trgc --n-jobs 8
    python run_granger_mne.py --task overtProd --stim-class prodDiff \\
        --method LCMV --atlas custom --feature-mode vertex_selectkbest \\
        --leakage-correction --preset B --trgc --n-jobs 8

    # restrict to specific ROI pairs
        ... --pairs awfa-lh:ifc-lh tpc-lh:ifc-lh tpc-lh:pmc-lh ifc-lh:pmc-lh
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('BLIS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import argparse
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from joblib import Parallel, delayed

from config import (
    SUBJECT_IDS, SPEECH_ROIS, DECODE_OUTPUT_ROOT,
    GC_TASK_START, GC_TASK_END, find_cached_npz,
)
from decoding_io import _load_cached_roi_data
from granger_mne import compute_subject_gc_mne, DEFAULT_CWT_FREQS
from run_granger import roiset_tag

GC_OUTPUT_ROOT = DECODE_OUTPUT_ROOT.parent / 'GC_source_space_mne'

PRESETS = {
    'A': dict(gc_n_lags=20, win_ms=250.0, target_fs=200.0),
    'B': dict(gc_n_lags=15, win_ms=60.0, target_fs=500.0),
}


# ─────────────────────────────────────────────────────────────────────
# IO
# ─────────────────────────────────────────────────────────────────────
def gc_tag_mne(gc_n_lags, win_ms, target_fs, n_pcs=1):
    # 'ssgc_cwt' = state-space GC, continuous-wavelet (Morlet) mode — the only
    # MNE mode that runs at these window sizes. pc{k} = FIXPC-k ROI aggregation
    # (k PCs per ROI; k>1 = multivariate block GC).
    return f'ssgc_cwt_pc{n_pcs}_mo{gc_n_lags}_sw{win_ms:g}ms_fs{target_fs:g}'


def pairs_tag(pair_names):
    """Directory segment for an explicit set of ROI pairs (order-stable)."""
    safe = '-'.join(f'{a}_{b}' for a, b in pair_names).lower().replace(' ', '')
    if len(safe) <= 60:
        return f'pairs_{safe}'
    import hashlib
    return f'pairs_{len(pair_names)}x_' + hashlib.sha1(safe.encode()).hexdigest()[:8]


def save_subject(result, subj, task, stim_class, method, atlas, feature_mode,
                 leakage_correction, gc_n_lags, win_ms, target_fs,
                 roi_subset, pairs_tag, output_root=GC_OUTPUT_ROOT):
    leakage_tag = 'leakage_corrected' if leakage_correction else 'raw'
    n_pcs = int(result.get('n_pcs', 1))
    out_dir = (
        output_root / task / method / atlas / feature_mode / leakage_tag
        / gc_tag_mne(gc_n_lags, win_ms, target_fs, n_pcs)
        / (pairs_tag or roiset_tag(roi_subset)) / stim_class
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f'{subj}_{task}_{stim_class}.npz'
    save = {
        'roi_names': np.array(result['roi_names']),
        'pair_i': result['pair_i'], 'pair_j': result['pair_j'],
        'window_ms': result['window_ms'], 'freqs': result['freqs'],
        'fs': np.array(result['fs']),
        'mode': np.array(result.get('mode', 'cwt_morlet')),
        'estimator': np.array('mne_statespace_cwt'),
        'n_pcs': np.array(n_pcs),
        'under_resolved': np.array(result.get('under_resolved', [])),
    }
    for b, arr in result['fxy'].items():
        save[f'fxy_{b}'] = arr
    for b, arr in result['fyx'].items():
        save[f'fyx_{b}'] = arr
    if 'dtrgc' in result:
        for b, arr in result['dtrgc'].items():
            save[f'dtrgc_{b}'] = arr
    np.savez_compressed(out_file, **save)
    return out_file


# ─────────────────────────────────────────────────────────────────────
# ROI / pair resolution
# ─────────────────────────────────────────────────────────────────────
def peek_cache_roi_names(task, method, atlas, feature_mode, leakage, subjects,
                         stim_class):
    for s in subjects:
        npz = find_cached_npz(task, method, atlas, feature_mode, leakage, s,
                              stim_class)
        if npz is not None:
            with np.load(npz, allow_pickle=True) as d:
                return list(map(str, d['roi_names']))
    return None


def resolve_pairs(pair_args, roi_names):
    """Map 'ROIa:ROIb' strings to (i, j) index pairs (case-insensitive).

    Returns (pairs, subset_names, missing). ``subset_names`` is the set of ROI
    names that must be loaded from the cache (only those in the pairs).
    """
    lut = {n.lower(): n for n in roi_names}
    idx = {n: i for i, n in enumerate(roi_names)}
    pairs, used, missing = [], [], []
    for spec in pair_args:
        parts = spec.replace('->', ':').split(':')
        if len(parts) != 2:
            missing.append(spec); continue
        a, b = parts[0].strip().lower(), parts[1].strip().lower()
        if a not in lut or b not in lut:
            missing.append(spec); continue
        na, nb = lut[a], lut[b]
        pairs.append((idx[na], idx[nb]))
        used += [na, nb]
    return pairs, sorted(set(used)), missing


# ─────────────────────────────────────────────────────────────────────
# Per-subject worker
# ─────────────────────────────────────────────────────────────────────
def process_subject(subj, args, subset, pair_names, cwt_freqs):
    npz = find_cached_npz(args.task, args.method, args.atlas, args.feature_mode,
                          args.leakage_correction, subj, args.stim_class)
    if npz is None:
        return subj, None, 'no cache'

    roi_data, y, times, sfreq = _load_cached_roi_data(
        npz, feature_mode=args.feature_mode, roi_subset=subset)
    if roi_data is None or len(roi_data) < 2:
        return subj, None, 'ROIs missing / <2'

    roi_names = list(roi_data)
    # Re-resolve pair indices against THIS subject's loaded ROI order.
    if pair_names:
        idx = {n.lower(): i for i, n in enumerate(roi_names)}
        pairs = [(idx[a.lower()], idx[b.lower()]) for a, b in pair_names
                 if a.lower() in idx and b.lower() in idx]
    else:
        pairs = None

    tmin = args.tmin if args.tmin is not None else GC_TASK_START.get(args.task)
    tmax = args.tmax if args.tmax is not None else GC_TASK_END.get(args.task)

    t0 = time.time()
    result = compute_subject_gc_mne(
        roi_data, times, sfreq, gc_n_lags=args.gc_n_lags, win_ms=args.win_ms,
        target_fs=args.target_fs, cwt_freqs=cwt_freqs, pairs=pairs,
        trgc=args.trgc, tmin=tmin, tmax=tmax, ncycle_floor=args.ncycle_floor,
        n_pcs=args.n_pcs)
    out_file = save_subject(
        result, subj, args.task, args.stim_class, args.method, args.atlas,
        args.feature_mode, args.leakage_correction, args.gc_n_lags,
        args.win_ms, args.target_fs, args.roi_subset, args.pairs_tag)
    return subj, out_file, f'{result["pair_i"].size} pairs, {time.time()-t0:.1f}s'


# ─────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description='Source-space state-space Granger causality (MNE-Connectivity)')
    p.add_argument('--task', required=True, choices=['perception', 'overtProd'])
    p.add_argument('--stim-class', required=True, choices=['prodDiff', 'percDiff'])
    p.add_argument('--method', required=True, choices=['dSPM', 'LCMV'])
    p.add_argument('--atlas', default='custom',
                   choices=['aparc', 'HCPMMP1', 'Schaefer200', 'custom'])
    p.add_argument('--feature-mode', default='vertex_selectkbest',
                   choices=['vertex_pca', 'vertex_selectkbest',
                            'vertex_selectkbest_all'])
    p.add_argument('--leakage-correction', action='store_true', default=False)
    p.add_argument('--preset', choices=['A', 'B'], default=None,
                   help='A = MO25/SW250/fs200 (canonical); B = MO15/SW60/fs500. '
                        'Explicit --gc-n-lags/--win-ms/--target-fs override it.')
    p.add_argument('--gc-n-lags', type=int, default=None,
                   help='state-space autocovariance lags (the MO knob)')
    p.add_argument('--win-ms', type=float, default=None,
                   help='effective analysis window in ms (the SW knob; sets '
                        'cwt cycles = f*win/1000)')
    p.add_argument('--target-fs', type=float, default=None,
                   help='resample virtual channels to this rate (the fs knob)')
    p.add_argument('--pairs', nargs='+', default=None, metavar='ROIa:ROIb',
                   help='specific directed-agnostic ROI pairs (GC computed both '
                        'ways). Default: all pairs among --roi-subset / speech ROIs.')
    p.add_argument('--roi-subset', nargs='+', default=None, metavar='ROI',
                   help='ROIs to include when --pairs is not given.')
    p.add_argument('--n-pcs', type=int, default=1,
                   help='PCs kept per ROI (FIXPC-k). 1 = one virtual channel + '
                        'bivariate GC; 3 or 4 = multivariate block GC (Pellegrini '
                        '2023). k>1 REQUIRES the full multi-vertex cache (all '
                        'vertices per ROI) — a pre-reduced/single-channel cache '
                        'can only do k=1.')
    p.add_argument('--ncycle-floor', type=float, default=1.0,
                   help='minimum Morlet cycles (protects short-window low freqs)')
    p.add_argument('--fmin', type=float, default=4.0)
    p.add_argument('--fmax', type=float, default=30.0)
    p.add_argument('--fstep', type=float, default=1.0)
    p.add_argument('--tmin', type=float, default=None,
                   help='GC window start (s); default config.GC_TASK_START[task]')
    p.add_argument('--tmax', type=float, default=None,
                   help='GC window end (s); default config.GC_TASK_END[task]')
    p.add_argument('--trgc', action='store_true', default=False,
                   help='also compute Diff-TRGC (method="gc_tr")')
    p.add_argument('--subjects', nargs='+', default=None)
    p.add_argument('--n-jobs', type=int, default=8,
                   help='subject-parallel workers (BLAS pinned to 1 each)')
    p.add_argument('--overwrite', action='store_true', default=False)
    return p.parse_args()


def main():
    args = parse_args()
    # Apply preset, then let explicit flags override.
    cfg = dict(PRESETS.get(args.preset, {}))
    if args.gc_n_lags is None:
        args.gc_n_lags = cfg.get('gc_n_lags', 25)
    if args.win_ms is None:
        args.win_ms = cfg.get('win_ms', 250.0)
    if args.target_fs is None:
        args.target_fs = cfg.get('target_fs', 200.0)

    subjects = args.subjects if args.subjects else SUBJECT_IDS
    cwt_freqs = np.arange(args.fmin, args.fmax + args.fstep / 2.0, args.fstep)

    # Resolve ROI subset / pairs against a real cache.
    cache_names = peek_cache_roi_names(
        args.task, args.method, args.atlas, args.feature_mode,
        args.leakage_correction, subjects, args.stim_class)
    if cache_names is None:
        print('  No vertex cache found for any subject — run '
              'run_source_localize.py first (check ROI_TIMESERIES_* in config.env).')
        return

    pair_names = None
    args.pairs_tag = None
    if args.pairs:
        pairs, subset, missing = resolve_pairs(args.pairs, cache_names)
        if missing:
            print(f'ERROR: unresolved --pairs {missing}')
            print(f'Available ROIs: {sorted(cache_names)}')
            return
        pair_names = [(cache_names[i], cache_names[j]) for i, j in pairs]
        args.roi_subset = subset
        args.pairs_tag = pairs_tag(pair_names)
    elif args.roi_subset is None and args.atlas in SPEECH_ROIS \
            and SPEECH_ROIS[args.atlas]:
        args.roi_subset = list(SPEECH_ROIS[args.atlas].keys())

    print('Source-space state-space Granger causality (MNE-Connectivity)')
    print(f'  Task/class:   {args.task} / {args.stim_class}')
    print(f'  Method/atlas: {args.method} / {args.atlas} ({args.feature_mode})')
    print(f'  Leakage corr: {args.leakage_correction}')
    print(f'  Estimator:    MNE state-space GC, mode=cwt_morlet (CWT)')
    print(f'  Config:       preset {args.preset}  ->  MO(gc_n_lags)={args.gc_n_lags}'
          f'  SW(win_ms)={args.win_ms:g}  fs={args.target_fs:g}')
    print(f'  cwt grid:     {cwt_freqs[0]:g}-{cwt_freqs[-1]:g} Hz '
          f'({cwt_freqs.size} bins), ncycle_floor={args.ncycle_floor:g}')
    from granger_mne import min_fft_window_samples
    nfft = min_fft_window_samples(args.gc_n_lags, args.target_fs,
                                  args.fmin, args.fmax)
    win_samp = max(2, round(args.win_ms / 1000.0 * args.target_fs))
    print(f'  (FFT-mode GC would need >={nfft} samp for MO={args.gc_n_lags}; '
          f'this window is {win_samp} samp -> cwt required)')
    agg = ('FIXPC1 (single channel, bivariate GC)' if args.n_pcs == 1
           else f'FIXPC{args.n_pcs} (multivariate block GC — needs full vertex cache)')
    print(f'  ROI aggreg.:  {agg}')
    if pair_names:
        print(f'  ROI pairs:    {["->".join(p) for p in pair_names]}')
    else:
        print(f'  ROI subset:   {args.roi_subset} (all pairs)')
    print(f'  TRGC:         {args.trgc}   Subjects: {len(subjects)}   '
          f'n_jobs: {args.n_jobs}')
    from granger_mne import under_resolved_bands
    ur = under_resolved_bands(args.win_ms)
    if ur:
        print(f'  WARNING: SW={args.win_ms:g}ms under-resolves {ur} '
              f'(< 2 wavelet cycles); read those bands with caution.')
    print()

    t_all = time.time()
    results = Parallel(n_jobs=args.n_jobs, prefer='processes')(
        delayed(process_subject)(s, args, args.roi_subset, pair_names, cwt_freqs)
        for s in subjects)
    ok, failed = 0, []
    for subj, out_file, msg in results:
        if out_file is None:
            print(f'  {subj}: SKIP ({msg})'); failed.append(subj)
        else:
            print(f'  {subj}: {msg} -> {out_file.name}'); ok += 1
    print(f'\n{ok}/{len(subjects)} subjects in {(time.time()-t_all)/60:.1f} min')
    if failed:
        print(f'FAILED/SKIPPED: {", ".join(failed)}')


if __name__ == '__main__':
    main()
