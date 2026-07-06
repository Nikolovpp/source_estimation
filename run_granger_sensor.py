#!/usr/bin/env python3
"""
Sensor-space Granger causality runner (reproduces the MATLAB BSMART PWGC).

Rebuilds the original sensor-space pairwise spectral GC analysis in
Python, so previously-computed MATLAB results can be reproduced/checked.
It forms "pseudo-channels" (spatial averages of sensor sets — the same
seed/target groups used in the MATLAB scripts) and runs the identical
moving-window bivariate Geweke GC engine used for the source-space
analysis (``granger.py``, verified equal to BSMART ``pwcausal`` to 2.8e-16).

Difference from the source runner: signals come from EEG **sensors**
(via ``data_loader``) rather than source-ROI virtual channels.  Data are
resampled to ``--target-fs`` (default 500 Hz — the rate the MATLAB GC
used) before fitting.

Usage
-----
    python run_granger_sensor.py --task overtProd --stim-class all \\
        --order 10 --win-ms 40 --trgc --n-jobs 8
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import argparse
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import SUBJECT_IDS, DECODE_OUTPUT_ROOT
from data_loader import load_subject_epochs
from run_granger import compute_subject_gc, save_subject_gc

GC_SENSOR_OUTPUT_ROOT = DECODE_OUTPUT_ROOT.parent / 'GC_sensor_space'

# Sensor pseudo-channels — the seed/target groups from the MATLAB scripts
# (task_overtProd_bivariate_GC_*).  Each is a spatial average of its
# sensors, forming one virtual channel.
SENSOR_PSEUDOCHANNELS = {
    'Temporal':          ['FT7', 'T7', 'TP7'],
    'Inferior_Frontal':  ['F5', 'FC5', 'FC3'],
    'Superior_Frontal':  ['FCz', 'FC1', 'F1'],
    'Superior_Parietal': ['CPz', 'CP1', 'P1'],
}


def load_epochs_for_gc(subj, task, stim_class):
    """Load sensor epochs; ``stim_class='all'`` concatenates both contrasts.

    Returns (data (n_epochs, n_channels, n_times), ch_names, times, sfreq).
    """
    if stim_class == 'all':
        contrasts = ['prodDiff', 'percDiff']
    else:
        contrasts = [stim_class]
    datas, ch_names, times, sfreq = [], None, None, None
    for sc in contrasts:
        try:
            epochs, _y, sf = load_subject_epochs(subj, task, sc)
        except Exception as e:
            print(f'    ({sc}) load failed: {e}')
            continue
        datas.append(epochs.get_data(copy=False))
        ch_names = list(epochs.ch_names)
        times = np.asarray(epochs.times)
        sfreq = float(sf)
    if not datas:
        return None, None, None, None
    data = np.concatenate(datas, axis=0)
    return data, ch_names, times, sfreq


def build_pseudochannels(data, ch_names, pseudochan_defs):
    """Average sensor sets into pseudo-channels.

    Returns dict {name: (n_epochs, n_times)}.  Sensor names matched
    case-insensitively; sets with no matching sensors are skipped.
    """
    lut = {c.lower(): i for i, c in enumerate(ch_names)}
    out = {}
    for name, sensors in pseudochan_defs.items():
        idx = [lut[s.lower()] for s in sensors if s.lower() in lut]
        missing = [s for s in sensors if s.lower() not in lut]
        if missing:
            print(f'    {name}: missing sensors {missing}')
        if not idx:
            print(f'    {name}: no sensors found — skipping')
            continue
        out[name] = data[:, idx, :].mean(axis=1)      # (n_epochs, n_times)
    return out


def parse_args():
    p = argparse.ArgumentParser(description='Sensor-space pairwise Granger causality (BSMART reproduction)')
    p.add_argument('--task', required=True, choices=['perception', 'overtProd'])
    p.add_argument('--stim-class', required=True,
                   choices=['prodDiff', 'percDiff', 'all'],
                   help="'all' concatenates both contrasts (matches the MATLAB "
                        "all-trials-combined analysis)")
    p.add_argument('--order', type=int, default=10)
    p.add_argument('--win-ms', type=float, default=40.0)
    p.add_argument('--target-fs', type=float, default=500.0)
    p.add_argument('--step', type=int, default=1)
    p.add_argument('--fmin', type=float, default=1.0)
    p.add_argument('--fmax', type=float, default=30.0)
    p.add_argument('--fstep', type=float, default=1.0)
    p.add_argument('--tmin', type=float, default=None)
    p.add_argument('--tmax', type=float, default=None)
    p.add_argument('--normalize', default='none', choices=['none', 'demean', 'zscore'])
    p.add_argument('--trgc', action='store_true', default=False)
    p.add_argument('--subjects', nargs='+', default=None)
    p.add_argument('--n-jobs', type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()
    subjects = args.subjects if args.subjects else SUBJECT_IDS
    freqs = np.arange(args.fmin, args.fmax + args.fstep / 2.0, args.fstep)

    print('Sensor-space pairwise Granger causality (BSMART reproduction)')
    print(f'  Task/class:   {args.task} / {args.stim_class}')
    print(f'  Pseudo-chans: {list(SENSOR_PSEUDOCHANNELS)}')
    print(f'  Order/window: {args.order} / {args.win_ms} ms @ {args.target_fs} Hz')
    print(f'  Freqs:        {freqs[0]:g}-{freqs[-1]:g} Hz    TRGC: {args.trgc}')
    print(f'  Subjects:     {len(subjects)}\n')

    t_start = time.time()
    ok, failed = 0, []
    for subj in subjects:
        data, ch_names, times, sfreq = load_epochs_for_gc(
            subj, args.task, args.stim_class)
        if data is None:
            print(f'  {subj}: no data. SKIP')
            failed.append(subj)
            continue
        roi_data = build_pseudochannels(data, ch_names, SENSOR_PSEUDOCHANNELS)
        if len(roi_data) < 2:
            print(f'  {subj}: <2 pseudo-channels. SKIP')
            failed.append(subj)
            continue

        t0 = time.time()
        result = compute_subject_gc(
            roi_data, times, sfreq,
            order=args.order, win_ms=args.win_ms, target_fs=args.target_fs,
            step=args.step, freqs=freqs, normalize=args.normalize,
            trgc=args.trgc, tmin=args.tmin, tmax=args.tmax, n_jobs=args.n_jobs,
        )
        out_file = save_subject_gc(
            result, subj, args.task, args.stim_class, method='sensor',
            atlas='sensor', feature_mode='pseudochan', leakage_correction=False,
            order=args.order, win_ms=args.win_ms, target_fs=args.target_fs,
            normalize=args.normalize, output_root=GC_SENSOR_OUTPUT_ROOT,
        )
        print(f'  {subj}: {data.shape[0]} trials, {result["pair_i"].size} pairs, '
              f'{result["window_ms"].size} windows in '
              f'{(time.time()-t0)/60:.1f} min -> {out_file.name}')
        ok += 1

    print(f'\n{ok}/{len(subjects)} subjects done in {(time.time()-t_start)/60:.1f} min')
    if failed:
        print(f'FAILED/SKIPPED: {", ".join(failed)}')


if __name__ == '__main__':
    main()
