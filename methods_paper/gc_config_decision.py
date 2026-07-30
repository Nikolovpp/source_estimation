#!/usr/bin/env python3
"""Decide the MNE GC config BEFORE committing to a full re-run.

Answers one question: **after the fixes, which pair x band values change sign,
and which are stable?** It does that by crossing the two switchable choices —

    normalize : 'none' (legacy)  vs  'demean' (new default, ERP removal)
    cwt grid  : 4-30 Hz (shipped) vs  ~DC-Nyquist (2 Hz .. 0.49*fs)

— over all subjects, on the real vertex caches (so the ERP is removed WITHIN
each level of ``y``, which is the correct form and which a reduced_ts cache
cannot do). The disjoint-band fix is already in the code, so it is held
constant across all four cells and its effect is not re-measured here.

Reports, per (pair, band): the group-mean net GC in each of the four cells, and
whether the sign agrees with the legacy cell. Writes a CSV + npz alongside.

Not covered (needs a code change, not a flag): the crop-before-CWT baseline
bias. Decide that separately.

Usage
-----
    conda activate mne
    python methods_paper/gc_config_decision.py \
        --task overtProd --stim-class prodDiff \
        --method LCMV --atlas custom --feature-mode vertex_selectkbest \
        --leakage-correction \
        --pairs awfa-lh:ifc-lh tpc-lh:ifc-lh tpc-lh:pmc-lh ifc-lh:pmc-lh \
        --gc-n-lags 15 --win-ms 250 --target-fs 200 --n-jobs 20

Runtime: the full-band cells dominate (~4x the bins). With --n-jobs 20 expect
a few minutes; add --quick to use the first 6 subjects while you sanity-check.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import argparse
import sys
import warnings

import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from joblib import Parallel, delayed

from config import SUBJECT_IDS, GC_TASK_START, GC_TASK_END, find_cached_npz
from decoding_io import _load_cached_roi_data
from granger import DEFAULT_BANDS
from granger_mne import compute_subject_gc_mne
from run_granger_mne import resolve_pairs, peek_cache_roi_names

BANDS = list(DEFAULT_BANDS)


def one_subject(subj, a, subset, pair_names, grids):
    """Return {(normalize, grid_name): {band: (n_pairs, n_times)}} of net GC."""
    npz = find_cached_npz(a.task, a.method, a.atlas, a.feature_mode,
                          a.leakage_correction, subj, a.stim_class)
    if npz is None:
        return subj, None, 'no cache'
    roi_data, y, times, sfreq = _load_cached_roi_data(
        npz, feature_mode=a.feature_mode, roi_subset=subset)
    if roi_data is None or len(roi_data) < 2:
        return subj, None, 'ROIs missing / <2'

    roi_names = list(roi_data)
    idx = {n.lower(): i for i, n in enumerate(roi_names)}
    pairs = [(idx[p.lower()], idx[q.lower()]) for p, q in pair_names
             if p.lower() in idx and q.lower() in idx]
    if len(pairs) != len(pair_names):
        return subj, None, 'pair not resolvable'

    tmin = a.tmin if a.tmin is not None else GC_TASK_START.get(a.task)
    tmax = a.tmax if a.tmax is not None else GC_TASK_END.get(a.task)

    out = {}
    for gname, fr in grids.items():
        for norm in ['none', 'demean']:
            r = compute_subject_gc_mne(
                roi_data, times, sfreq, gc_n_lags=a.gc_n_lags,
                win_ms=a.win_ms, target_fs=a.target_fs, cwt_freqs=fr,
                pairs=pairs, trgc=False, tmin=tmin, tmax=tmax,
                ncycle_floor=a.ncycle_floor, n_pcs=a.n_pcs,
                normalize=norm, labels=y)
            out[(norm, gname)] = ({b: r['fxy'][b] - r['fyx'][b] for b in BANDS},
                                  int(np.asarray(r['freqs']).size))
    return subj, out, f'{len(pairs)} pairs, y levels={len(np.unique(y))}'


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--task', default='overtProd')
    p.add_argument('--stim-class', default='prodDiff')
    p.add_argument('--method', default='LCMV')
    p.add_argument('--atlas', default='custom')
    p.add_argument('--feature-mode', default='vertex_selectkbest')
    p.add_argument('--leakage-correction', action='store_true', default=False)
    p.add_argument('--pairs', nargs='+', required=True,
                   help='ROIa:ROIb ... (the pairs you intend to report)')
    p.add_argument('--gc-n-lags', type=int, default=15)
    p.add_argument('--win-ms', type=float, default=250.0,
                   help='Morlet cycle count is f*win_ms/1000 (floor 1), so '
                        'sigma_f = 500/win_ms Hz at every frequency. 250 ms -> '
                        '2 Hz, half the narrowest reported band.')
    p.add_argument('--target-fs', type=float, default=200.0)
    p.add_argument('--ncycle-floor', type=float, default=1.0)
    p.add_argument('--n-pcs', type=int, default=1)
    p.add_argument('--tmin', type=float, default=None)
    p.add_argument('--tmax', type=float, default=None)
    p.add_argument('--n-jobs', type=int, default=8)
    p.add_argument('--quick', action='store_true',
                   help='first 6 subjects only')
    p.add_argument('--out-dir', default=None)
    return p.parse_args()


def main():
    a = parse_args()
    subjects = SUBJECT_IDS[:6] if a.quick else list(SUBJECT_IDS)

    cache_names = peek_cache_roi_names(a.task, a.method, a.atlas, a.feature_mode,
                                       a.leakage_correction, subjects,
                                       a.stim_class)
    if cache_names is None:
        print('ERROR: no vertex cache found for any subject '
              '(check ROI_TIMESERIES_* in config.env).')
        return 1
    pr, subset, missing = resolve_pairs(a.pairs, cache_names)
    if missing:
        print(f'ERROR: unresolved --pairs {missing}')
        print(f'Available ROIs: {sorted(cache_names)}')
        return 1
    pair_names = [(cache_names[i], cache_names[j]) for i, j in pr]

    nyq = 0.49 * a.target_fs
    grids = {'shipped_4-30Hz': np.arange(4.0, 30.0 + 1e-9, 1.0),
             'full_2-Nyq': np.arange(2.0, nyq + 1e-9, 1.0)}

    print('MNE GC config decision sweep')
    print(f'  {a.task} / {a.stim_class} / {a.method} / {a.atlas} '
          f'/ leakage={a.leakage_correction}')
    print(f'  MO={a.gc_n_lags}  SW={a.win_ms:g} ms  fs={a.target_fs:g}  '
          f'n_pcs={a.n_pcs}')
    print(f'  pairs: {[f"{x}->{y}" for x, y in pair_names]}')
    print(f'  grids: ' + ', '.join(f'{k} ({v.size} bins)'
                                   for k, v in grids.items()))
    print(f'  normalize: none (legacy) vs demean (ERP removed within each y level)')
    print(f'  subjects: {len(subjects)}   n_jobs: {a.n_jobs}\n', flush=True)

    res = Parallel(n_jobs=a.n_jobs, prefer='processes', verbose=5)(
        delayed(one_subject)(s, a, subset, pair_names, grids) for s in subjects)

    ok = {}
    nfreq = {}
    for subj, out, msg in res:
        if out is None:
            print(f'  SKIP {subj}: {msg}')
            continue
        for key, (bands, nf) in out.items():
            ok.setdefault(key, []).append(bands)
            nfreq[key] = nf
    if not ok:
        print('\nERROR: no subject produced results.')
        return 1
    n_used = len(next(iter(ok.values())))
    print(f'\n{n_used} subjects contributed.')
    for key, nf in sorted(nfreq.items()):
        print(f'  {key[0]:7s} / {key[1]:15s}: MNE returned {nf} frequency bins')

    cells = [('none', 'shipped_4-30Hz'), ('demean', 'shipped_4-30Hz'),
             ('none', 'full_2-Nyq'), ('demean', 'full_2-Nyq')]
    hdr = (f'\n{"pair":20s} {"band":10s} ' +
           ' '.join(f'{c[0][:4]}/{c[1][:8]:>9s}' for c in cells) +
           '   agree-with-legacy')
    print(hdr)
    print('-' * (len(hdr) + 6))
    rows = []
    npair = len(pair_names)
    for k, (pi, pj) in enumerate(pair_names):
        nm = f'{pi}->{pj}'
        for b in BANDS:
            vals = {}
            for c in cells:
                arr = np.stack([d[b][k] for d in ok[c]])   # (n_subj, n_times)
                vals[c] = float(np.nanmean(arr))
            leg = vals[cells[0]]
            agree = ''.join('.' if np.sign(vals[c]) == np.sign(leg) else 'X'
                            for c in cells[1:])
            print(f'{nm:20s} {b:10s} ' +
                  ' '.join(f'{vals[c]:+14.6f}' for c in cells) +
                  f'   {agree}')
            rows.append([nm, b] + [vals[c] for c in cells] + [agree])

    nflip = sum(r[-1].count('X') for r in rows)
    print(f'\nsign changes vs legacy (none/4-30 Hz): {nflip} of {len(rows)*3} '
          f'comparisons   ("X" = sign differs, "." = same)')
    print('column order: none/4-30(legacy), demean/4-30, none/full, demean/full')

    out_dir = a.out_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'methods_paper')
    os.makedirs(out_dir, exist_ok=True)
    stem = f'gc_config_decision_{a.task}_{a.stim_class}_mo{a.gc_n_lags}_sw{a.win_ms:g}'
    csv = os.path.join(out_dir, stem + '.csv')
    with open(csv, 'w') as fh:
        fh.write('pair,band,' + ','.join(f'{c[0]}_{c[1]}' for c in cells) +
                 ',agree_demean430_nonefull_demeanfull\n')
        for r in rows:
            fh.write(','.join(str(x) for x in r) + '\n')
    print(f'\nwrote {csv}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
