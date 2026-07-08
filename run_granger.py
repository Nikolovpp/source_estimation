#!/usr/bin/env python3
"""
Source-space Granger causality runner (BSMART ``mov_bi_ga`` analogue).

Lifts the sensor-space BSMART pairwise spectral GC analysis into source
space, between the speech-network ROIs.  Reuses the **vertex** ROI
timeseries caches written by ``run_source_localize.py`` (the same caches
the decoding used) — no pca_flip cache and no re-running the inverse.

Per subject:
  1. load the requested ROIs' vertex timeseries from the .npz cache
  2. reduce each ROI to one virtual channel via a fixed first-PC spatial
     filter (``granger.reduce_roi_first_pc``)
  3. downsample the virtual channels to ``--target-fs`` (default 500 Hz,
     matching the MATLAB GC; avoids ill-conditioned AR on ~2 kHz data)
  4. for every ROI pair, moving-window bivariate Geweke spectral GC
     (``granger.moving_window_pairwise_gc``), 1-sample step, raw signals
     — byte-faithful to BSMART ``mov_bi_ga`` (verified to 2.8e-16)
  5. average the frequency-resolved GC into theta/alpha/low-beta/high-beta
  6. optionally also compute Diff-TRGC (time-reversed GC) per band

Parallelism: subject-sequential outer loop; ROI pairs run in a joblib
pool (BLAS pinned to 1 thread per worker).

Usage
-----
    python run_granger.py --task overtProd --stim-class prodDiff \\
        --method dSPM --atlas HCPMMP1 --feature-mode vertex_selectkbest \\
        --order 10 --win-ms 40 --target-fs 500 --trgc --n-jobs 64
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
from itertools import combinations

import numpy as np
from scipy.signal import resample_poly
from fractions import Fraction

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from joblib import Parallel, delayed

from config import (
    SUBJECT_IDS, SPEECH_ROIS, DECODE_OUTPUT_ROOT,
    find_cached_npz, cache_feat_mode,
)
from decoding_io import _load_cached_roi_data, filter_roi_dict
from granger import (
    reduce_roi_first_pc, moving_window_pairwise_gc, band_average,
    DEFAULT_BANDS,
)

GC_OUTPUT_ROOT = DECODE_OUTPUT_ROOT.parent / 'GC_source_space'


# ─────────────────────────────────────────────────────────────────────
# Signal prep
# ─────────────────────────────────────────────────────────────────────
def resample_channels(x, sfreq, target_fs, pad_mode='reflect'):
    """Anti-aliased resample along the last axis from sfreq to target_fs.

    Each signal's edges are padded before the polyphase FIR and the pad is
    cropped off afterward, so the anti-alias filter never sees the zero
    padding that ``resample_poly`` assumes by default — that zero padding
    otherwise injects a ringing transient at both ends of every (epoched)
    signal, which for high-order moving-window MVAR/Granger causality shows
    up as spurious sharp GC at the first/last windows.  This mirrors the
    edge handling EEGLAB ``pop_resample`` applies (see ``myresample``).

    ``pad_mode`` is any ``numpy.pad`` mode: 'reflect' (default) preserves
    the edge slope and is the standard DSP choice; 'edge' reproduces
    EEGLAB's DC-hold.  It falls back to 'edge' when the signal is shorter
    than the required pad length (a constraint of 'reflect').

    The output has exactly the same length as an un-padded
    ``resample_poly`` call, so downstream window/time-axis bookkeeping is
    unchanged.  Returns (x_resampled, new_sfreq); if the rates already
    match (within 1 Hz) the input is returned unchanged.
    """
    if abs(sfreq - target_fs) < 1.0:
        return x, float(sfreq)
    frac = Fraction(int(round(target_fs)), int(round(sfreq))).limit_denominator(1000)
    up, down = frac.numerator, frac.denominator

    # resample_poly's default anti-alias FIR has half-length 10*max(up,down)
    # in the up-sampled domain → ceil(10*max(up,down)/up) input samples.
    # Pad by that, rounded up to a whole multiple of `down`, so the post-
    # resample crop (npad*up/down) is an exact integer number of samples.
    half_in = int(np.ceil(10 * max(up, down) / up))
    npad = int(np.ceil(half_in / down) * down)
    n_time = x.shape[-1]
    mode = 'edge' if (pad_mode == 'reflect' and npad >= n_time) else pad_mode

    pad_width = [(0, 0)] * (x.ndim - 1) + [(npad, npad)]
    xp = np.pad(x, pad_width, mode=mode)
    xr = resample_poly(xp, up, down, axis=-1)

    crop = npad * up // down
    # Length of an un-padded resample, taken from scipy directly so the
    # slice matches exactly regardless of its internal rounding.
    n_ref = resample_poly(np.zeros(n_time, dtype=xr.dtype), up, down).shape[-1]
    xr = xr[..., crop:crop + n_ref]
    return xr, float(sfreq) * up / down


def normalize_ensemble(x, mode):
    """Optional across-trial normalization per time point.

    ``x`` is (n_trials, n_times).  BSMART itself does NO normalization
    (``mode='none'`` reproduces it exactly); the other modes follow the
    Ding & Bressler event-related recommendation and are exploratory.
    """
    if mode == 'none':
        return x
    mu = x.mean(axis=0, keepdims=True)
    if mode == 'demean':
        return x - mu
    if mode == 'zscore':
        sd = x.std(axis=0, keepdims=True)
        sd[sd == 0] = 1.0
        return (x - mu) / sd
    raise ValueError(f'Unknown normalize mode: {mode}')


# ─────────────────────────────────────────────────────────────────────
# Per-subject compute core (cache-independent — unit-testable)
# ─────────────────────────────────────────────────────────────────────
def compute_subject_gc(roi_data, times, sfreq, *, order=10, win_ms=40.0,
                       target_fs=500.0, step=1, freqs=None, bands=None,
                       normalize='none', trgc=False, tmin=None, tmax=None,
                       gc_mode='pairwise', n_jobs=1):
    """Moving-window Geweke GC for one subject, all ROI pairs.

    ``gc_mode='pairwise'`` runs bivariate BSMART GC per pair.
    ``gc_mode='conditional'`` runs state-space conditional GC (Barnett &
    Seth 2015): each directed edge conditioned on all other ROIs (fits the
    joint MVAR once per window).  In both cases the output uses the same
    ``fxy``/``fyx`` schema (``fxy`` = i->j, ``fyx`` = j->i), so the stats
    layer is identical.  ``trgc`` applies to pairwise mode only.

    Parameters
    ----------
    roi_data : dict {roi_name: np.ndarray}
        Either vertex data ``(n_epochs, n_vertices, n_times)`` (reduced to
        one virtual channel here) or already-reduced ``(n_epochs, n_times)``.
    times : np.ndarray
        Time vector in seconds for the cached series (pre-resample).
    sfreq : float
        Sampling rate of the cached series (Hz).
    order, win_ms, target_fs, step, normalize, trgc :
        GC parameters.  ``win_ms`` at ``target_fs`` sets the window in
        samples (40 ms @ 500 Hz = 20, matching BSMART).
    freqs : np.ndarray, optional
        Frequencies in Hz (default 1..30, 1 Hz steps).
    bands : dict, optional
        Band definitions (default theta/alpha/low-beta/high-beta).
    tmin, tmax : float, optional
        Restrict GC to this time window (seconds); default full epoch.

    Returns
    -------
    result : dict
        ``roi_names`` (list), ``pair_i``/``pair_j`` (index arrays),
        ``window_ms`` (n_windows,), ``freqs``, ``bands`` (dict),
        ``fxy[band]``/``fyx[band]`` : (n_pairs, n_windows) directed GC
        (i->j / j->i), and if ``trgc`` ``dtrgc[band]`` : (n_pairs,
        n_windows) Diff-TRGC (i->j).
    """
    if freqs is None:
        freqs = np.arange(1, 31)
    if bands is None:
        bands = DEFAULT_BANDS
    freqs = np.asarray(freqs, dtype=float)
    band_names = list(bands)

    roi_names = list(roi_data.keys())
    # Reduce each ROI to a single virtual channel.
    vcs = []
    for r in roi_names:
        arr = np.asarray(roi_data[r], dtype=float)
        vc = reduce_roi_first_pc(arr) if arr.ndim == 3 else arr
        vcs.append(vc)                                 # (n_epochs, n_times)
    V = np.stack(vcs, axis=0)                           # (n_rois, n_ep, n_t)

    # Downsample to target_fs.
    V, fs = resample_channels(V, sfreq, target_fs)
    n_t = V.shape[2]
    new_times = times[0] + np.arange(n_t) / fs

    # Crop to [tmin, tmax].
    lo = 0 if tmin is None else int(np.searchsorted(new_times, tmin))
    hi = n_t if tmax is None else int(np.searchsorted(new_times, tmax, 'right'))
    V = V[:, :, lo:hi]
    win_times = new_times[lo:hi]

    # Optional ensemble normalization (default none = BSMART).
    if normalize != 'none':
        V = np.stack([normalize_ensemble(V[r], normalize)
                      for r in range(V.shape[0])], axis=0)

    win_samples = max(2, round(win_ms / 1000.0 * fs))
    starts = np.arange(0, V.shape[2] - win_samples + 1, step)
    # Window START time in ms (matches production_pwgc_data_to_python.m).
    window_ms = win_times[starts] * 1000.0
    n_win = starts.size

    n_roi = len(roi_names)
    pairs = list(combinations(range(n_roi), 2))
    n_pairs = len(pairs)
    pair_i = np.array([p[0] for p in pairs])
    pair_j = np.array([p[1] for p in pairs])
    fxy = {b: np.full((n_pairs, n_win), np.nan) for b in band_names}
    fyx = {b: np.full((n_pairs, n_win), np.nan) for b in band_names}
    use_trgc = trgc and gc_mode == 'pairwise'
    dtr = {b: np.full((n_pairs, n_win), np.nan) for b in band_names} if use_trgc else None

    if gc_mode == 'conditional':
        from granger_statespace import moving_window_conditional_gc
        Xmv = np.transpose(V, (1, 0, 2))               # (n_ep, n_roi, n_t)
        directed = [(i, j) for i in range(n_roi) for j in range(n_roi) if i != j]
        res = moving_window_conditional_gc(
            Xmv, order=order, freqs=freqs, fs=fs,
            win_samples=win_samples, step=step, pairs=directed, n_jobs=n_jobs,
        )
        band_ed = {p: band_average(res['gc'][p], freqs, bands) for p in directed}
        for k, (i, j) in enumerate(pairs):
            for b in band_names:
                fxy[b][k] = band_ed[(i, j)][b]         # i->j | rest
                fyx[b][k] = band_ed[(j, i)][b]         # j->i | rest
    else:  # pairwise (BSMART)
        def _pair_gc(i, j):
            X = np.stack([V[i], V[j]], axis=1)         # (n_ep, 2, n_t)
            res = moving_window_pairwise_gc(
                X, order=order, freqs=freqs, fs=fs,
                win_samples=win_samples, step=step, trgc=use_trgc,
            )
            b_xy = band_average(res['f_xy'], freqs, bands)
            b_yx = band_average(res['f_yx'], freqs, bands)
            b_d = band_average(res['d_xy'], freqs, bands) if use_trgc else None
            return i, j, b_xy, b_yx, b_d

        out = Parallel(n_jobs=n_jobs, prefer='processes')(
            delayed(_pair_gc)(i, j) for i, j in pairs
        )
        pos = {(i, j): k for k, (i, j) in enumerate(pairs)}
        for i, j, b_xy, b_yx, b_d in out:
            k = pos[(i, j)]
            for b in band_names:
                fxy[b][k] = b_xy[b]
                fyx[b][k] = b_yx[b]
                if use_trgc:
                    dtr[b][k] = b_d[b]

    result = {
        'roi_names': roi_names,
        'pair_i': pair_i,
        'pair_j': pair_j,
        'window_ms': window_ms,
        'freqs': freqs,
        'bands': dict(bands),
        'fxy': fxy,
        'fyx': fyx,
        'fs': fs,
    }
    if use_trgc:
        result['dtrgc'] = dtr
    return result


# ─────────────────────────────────────────────────────────────────────
# IO
# ─────────────────────────────────────────────────────────────────────
def gc_tag(order, win_ms, target_fs, normalize, gc_mode='pairwise'):
    t = f'order{order}_win{win_ms:g}ms_fs{target_fs:g}'
    if normalize != 'none':
        t += f'_{normalize}'
    if gc_mode != 'pairwise':
        t += f'_{gc_mode}'
    return t


def roiset_tag(roi_subset):
    """Directory segment identifying the ROI subset, so a ``--roi-subset``
    run writes to its own folder and never overwrites the full run (or a
    different subset).  Order- and case-insensitive; full runs -> 'all_rois'.
    Long subsets fall back to a short hash so paths stay sane.
    """
    if not roi_subset:
        return 'all_rois'
    names = sorted(n.strip().lower().replace(' ', '_') for n in roi_subset)
    safe = '-'.join(names)
    if len(safe) <= 60:
        return f'rois_{safe}'
    import hashlib
    h = hashlib.sha1('|'.join(names).encode()).hexdigest()[:8]
    return f'rois_{len(names)}x_{h}'


def save_subject_gc(result, subj, task, stim_class, method, atlas,
                    feature_mode, leakage_correction, order, win_ms,
                    target_fs, normalize, output_root=GC_OUTPUT_ROOT,
                    gc_mode='pairwise', roi_subset=None):
    leakage_tag = 'leakage_corrected' if leakage_correction else 'raw'
    out_dir = (
        output_root / task / method / atlas / feature_mode / leakage_tag
        / gc_tag(order, win_ms, target_fs, normalize, gc_mode)
        / roiset_tag(roi_subset) / stim_class
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f'{subj}_{task}_{stim_class}.npz'

    save = {
        'roi_names': np.array(result['roi_names']),
        'pair_i': result['pair_i'],
        'pair_j': result['pair_j'],
        'window_ms': result['window_ms'],
        'freqs': result['freqs'],
        'fs': np.array(result['fs']),
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
# Orchestration
# ─────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description='Source-space pairwise Granger causality (BSMART port)')
    p.add_argument('--task', required=True, choices=['perception', 'overtProd'])
    p.add_argument('--stim-class', required=True, choices=['prodDiff', 'percDiff'])
    p.add_argument('--method', required=True, choices=['dSPM', 'LCMV'])
    p.add_argument('--atlas', default='HCPMMP1',
                   choices=['aparc', 'HCPMMP1', 'Schaefer200', 'custom'])
    p.add_argument('--feature-mode', default='vertex_selectkbest',
                   choices=['vertex_pca', 'vertex_selectkbest',
                            'vertex_selectkbest_all'],
                   help='Locates the shared vertex cache (all vertex modes '
                        'share it). pca_flip is NOT used for GC.')
    p.add_argument('--leakage-correction', action='store_true', default=False)
    p.add_argument('--gc-mode', default='pairwise',
                   choices=['pairwise', 'conditional'],
                   help="'pairwise' = bivariate BSMART GC; 'conditional' = "
                        "state-space conditional GC (Barnett & Seth 2015), "
                        "each edge conditioned on all other ROIs")
    p.add_argument('--roi-subset', nargs='+', default=None, metavar='ROI',
                   help='ROIs to include (default all speech ROIs); GC is '
                        'computed for every pair among them.')
    p.add_argument('--order', type=int, default=10, help='AR model order')
    p.add_argument('--win-ms', type=float, default=40.0,
                   help='Moving-window length in ms (40 ms @ 500 Hz = 20 samples)')
    p.add_argument('--target-fs', type=float, default=500.0,
                   help='Resample virtual channels to this rate before GC')
    p.add_argument('--step', type=int, default=1, help='Window step in samples (BSMART=1)')
    p.add_argument('--fmin', type=float, default=1.0)
    p.add_argument('--fmax', type=float, default=30.0)
    p.add_argument('--fstep', type=float, default=1.0)
    p.add_argument('--tmin', type=float, default=None, help='GC window start (s); default full epoch')
    p.add_argument('--tmax', type=float, default=None, help='GC window end (s); default full epoch')
    p.add_argument('--normalize', default='none', choices=['none', 'demean', 'zscore'],
                   help='Ensemble normalization (none = BSMART-faithful)')
    p.add_argument('--trgc', action='store_true', default=False,
                   help='Also compute Diff-TRGC (time-reversed GC robustness control)')
    p.add_argument('--subjects', nargs='+', default=None)
    p.add_argument('--n-jobs', type=int, default=64)
    p.add_argument('--overwrite', action='store_true', default=False)
    return p.parse_args()


def main():
    args = parse_args()
    subjects = args.subjects if args.subjects else SUBJECT_IDS
    freqs = np.arange(args.fmin, args.fmax + args.fstep / 2.0, args.fstep)

    print('Source-space pairwise Granger causality (BSMART port)')
    print(f'  Task/class:   {args.task} / {args.stim_class}')
    print(f'  Method/atlas: {args.method} / {args.atlas}')
    print(f'  Feature mode: {args.feature_mode} (vertex cache -> first-PC reduction)')
    print(f'  Leakage corr: {args.leakage_correction}')
    print(f'  Order:        {args.order}')
    print(f'  Window:       {args.win_ms} ms @ {args.target_fs} Hz, step {args.step} sample(s)')
    print(f'  Freqs:        {freqs[0]:g}-{freqs[-1]:g} Hz ({freqs.size} bins)')
    print(f'  Normalize:    {args.normalize}   TRGC: {args.trgc}')
    print(f'  GC mode:      {args.gc_mode}')
    print(f'  Subjects:     {len(subjects)}   n_jobs: {args.n_jobs}')
    print()

    if args.atlas in SPEECH_ROIS:
        roi_universe = list(SPEECH_ROIS[args.atlas].keys())
    else:
        roi_universe = None  # e.g. custom atlas — resolve against the cache

    # Resolve/validate --roi-subset against the ACTUAL ROI names stored in the
    # cache (peek at the first available subject).  Works for every atlas —
    # including 'custom', whose ROI names (e.g. awfa-lh, ifc-lh) differ from
    # the speech-network names — and on mismatch lists what IS available.
    # Case-insensitive.  GC on a 2-ROI subset gives just that seed<->target pair.
    if args.roi_subset:
        cache_names = None
        for s in subjects:
            npz0 = find_cached_npz(args.task, args.method, args.atlas,
                                   args.feature_mode, args.leakage_correction,
                                   s, args.stim_class)
            if npz0 is not None:
                with np.load(npz0, allow_pickle=True) as d0:
                    cache_names = list(d0['roi_names'])
                break
        if cache_names is None:
            print('  No vertex cache found for any subject — run '
                  'run_source_localize.py first, or check the '
                  'ROI_TIMESERIES_EXTERNAL paths in config.env.')
            return
        lut = {n.lower(): n for n in cache_names}
        subset, missing = [], []
        for name in args.roi_subset:
            (subset.append(lut[name.lower()]) if name.lower() in lut
             else missing.append(name))
        if missing:
            print(f'ERROR: requested ROIs not in the {args.atlas} cache: {missing}')
            print(f'Available ROIs in cache: {sorted(cache_names)}')
            return
        print(f'  ROI subset:   {subset} ({len(subset)} ROIs, '
              f'{len(subset)*(len(subset)-1)//2} pair(s))')
    else:
        subset = roi_universe

    total_start = time.time()
    ok, failed = 0, []
    for subj in subjects:
        npz = find_cached_npz(args.task, args.method, args.atlas,
                              args.feature_mode, args.leakage_correction,
                              subj, args.stim_class)
        if npz is None:
            print(f'  {subj}: no vertex cache found — run run_source_localize.py first. SKIP')
            failed.append(subj)
            continue

        # ROIs to load (resolved above).
        roi_data, y, times, sfreq = _load_cached_roi_data(
            npz, feature_mode=args.feature_mode, roi_subset=subset,
        )
        if roi_data is None:
            print(f'  {subj}: requested ROIs missing from cache. SKIP')
            failed.append(subj)
            continue
        if len(roi_data) < 2:
            print(f'  {subj}: need >=2 ROIs for pairwise GC. SKIP')
            failed.append(subj)
            continue

        t0 = time.time()
        result = compute_subject_gc(
            roi_data, times, sfreq,
            order=args.order, win_ms=args.win_ms, target_fs=args.target_fs,
            step=args.step, freqs=freqs, normalize=args.normalize,
            trgc=args.trgc, tmin=args.tmin, tmax=args.tmax,
            gc_mode=args.gc_mode, n_jobs=args.n_jobs,
        )
        out_file = save_subject_gc(
            result, subj, args.task, args.stim_class, args.method,
            args.atlas, args.feature_mode, args.leakage_correction,
            args.order, args.win_ms, args.target_fs, args.normalize,
            gc_mode=args.gc_mode, roi_subset=args.roi_subset,
        )
        n_pairs = result['pair_i'].size
        print(f'  {subj}: {len(roi_data)} ROIs, {n_pairs} pairs, '
              f'{result["window_ms"].size} windows in '
              f'{(time.time()-t0)/60:.1f} min -> {out_file.name}')
        ok += 1

    print(f'\n{ok}/{len(subjects)} subjects done in '
          f'{(time.time()-total_start)/60:.1f} min')
    if failed:
        print(f'FAILED/SKIPPED: {", ".join(failed)}')


if __name__ == '__main__':
    main()
