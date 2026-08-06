#!/usr/bin/env python3
"""Parametric vs state-space Granger causality on identical data.

Two estimators, same windows, same frequency grid, same ROI reduction, so the
estimator is the only thing that varies:

    parametric   granger.py            fit MVAR, construct H; the reduced model
                                       for the GC ratio is fitted by a SECOND
                                       REGRESSION (Chen/Bressler/Ding 2006)
    state-space  granger_statespace.py fit the same MVAR, then DERIVE the reduced
                                       model in closed form via the DARE
                                       (Barnett & Seth 2015)

The nonparametric (Wilson) route is implemented and verified in
``granger_wilson.py`` but is deliberately NOT run here: its wrap-around limit
makes theta unrecoverable from the short windows this experiment requires
(section 22a of the theory chapter). That exclusion is a measured result, not an
omission.

Measures — chosen because they are the ones on which the two estimators can
differ at all:

  M0  pairwise spectral      IDENTICAL by construction (Geweke's bivariate
                             spectral formula never touches the reduced model).
                             Computed once per run as a pipeline self-check: if
                             this is not ~0 to machine precision, something is
                             mis-wired.
  M1  conditional spectral   6-ROI joint model, F_{a->b | all others}.
  M2  pairwise time-domain   where the dual-regression bias lives.
  M3  block GC               FIXPC-k ROI blocks (k>1); the reduced block model
                             is ARMA, so the two estimators should diverge most.
  F   triple-wise conditional  F_{a->c | b} on a 3-variable model, for EVERY
                             ordered triple. This is what isolates a specific
                             mediator, and it is a different quantity from M1.

Direction convention throughout: ``fxy`` is x -> y with x the SOURCE, matching
the theory chapter.

Examples
--------
    # A - canonical (Damera): order 6 / 60 ms / fs 200
    python run_granger_routes.py --task overtProd --stim-class prodDiff \
        --method dSPM --win-ms 60 --order 6 --config-tag A_canonical

    # B - window sweep
    python run_granger_routes.py --task overtProd --stim-class prodDiff \
        --method dSPM --win-ms 40 60 80 120 --order 6 --config-tag B_winsweep

    # C - order sweep at two windows
    python run_granger_routes.py ... --win-ms 60  --order 2 4 6 8
    python run_granger_routes.py ... --win-ms 120 --order 4 8 12 16 20

    # E - FIXPC1 vs FIXPC4
    python run_granger_routes.py ... --win-ms 60 --order 6 --n-pcs 1 4

    # F - exhaustive triple-wise conditional GC
    python run_granger_routes.py ... --win-ms 60 --order 6 --triples exhaustive
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
import time

import numpy as np
from scipy.signal import resample_poly
from joblib import Parallel, delayed

import config
from config import find_cached_npz
from granger import (fit_mvar, pairwise_spectral_gc, conditional_spectral_gc,
                     time_domain_conditional_gc, band_average)
from granger_statespace import ss_conditional_gc

# ─────────────────────────────────────────────────────────────────────
# Reporting grid. The data are low-passed at ~38 Hz (measured: -12.9 dB at
# 30 Hz, -67 dB at 40 Hz), so 2-38 Hz is the supported band. 4-30 truncates
# high beta at its own edge, which is why the old grid understated band GC.
# ─────────────────────────────────────────────────────────────────────
FREQS = np.arange(2.0, 38.5, 1.0)
BANDS = {'theta': (4.0, 8.0), 'alpha': (8.0, 12.0), 'low_beta': (12.0, 18.0),
         'high_beta': (18.0, 30.0), 'upper': (30.0, 38.0)}

# Your pre-registered mediation hypotheses. Every ordered triple is computed;
# these are flagged ``primary`` in the output so the rest stay exploratory.
PRIMARY_TRIPLES = [
    ('awfa-lh', 'tpc-lh', 'pmc-lh'),    # 1 dorsal stream: Spt as the interface
    ('awfa-lh', 'pmc-lh', 'tpc-lh'),    # 2 reverse: frontal first
    ('awfa-lh', 'tpc-lh', 'ifc-lh'),    # 3 parietal mediates auditory->Broca
    ('awfa-lh', 'ifc-lh', 'tpc-lh'),    # 4 Broca mediates auditory->parietal
    ('awfa-lh', 'ifc-lh', 'pmc-lh'),    # 5 ventral-to-motor
    ('awfa-lh', 'pmc-lh', 'ifc-lh'),    # 6 premotor before Broca
    ('tpc-lh', 'ifc-lh', 'pmc-lh'),     # 7 within-frontal ordering
    ('tpc-lh', 'pmc-lh', 'ifc-lh'),     # 8
    ('pmc-lh', 'tpc-lh', 'awfa-lh'),    # 9 efference copy (production)
    ('ifc-lh', 'pmc-lh', 'tpc-lh'),     # 10
    ('pmc-lh', 'awfa-lh', 'tpc-lh'),    # 11
    ('owfa-lh', 'vwfa-lh', 'awfa-lh'),  # 12 reading hierarchy
    ('vwfa-lh', 'tpc-lh', 'ifc-lh'),    # 13 dorsal reading
    ('vwfa-lh', 'awfa-lh', 'ifc-lh'),   # 14 ventral reading
]


# ─────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────
def load_roi_vertices(path, rois):
    """Return {roi: (n_trials, n_vertices, n_times)}, plus the time axis and fs.

    NOTE on axis order. ``decoding_io.save_roi_timeseries`` stores each ROI as
    ``X_roi.transpose(0, 2, 1)``, i.e. **(n_trials, n_times, n_vertices)** —
    time on axis 1, vertices last. Everything downstream here wants vertices on
    axis 1 and time last, so 3-D arrays are transposed back on load. Getting
    this wrong resamples the vertex axis and yields ROIs of differing length.
    """
    z = np.load(path, allow_pickle=True)
    out, times = {}, z['times']
    n_t = len(times)
    for r in rois:
        for key in (r, f'vertex__{r}', f'vc__{r}'):
            if key in z.files:
                v = z[key]
                if v.ndim == 3:
                    v = v.transpose(0, 2, 1)          # -> (trials, vertices, time)
                if v.shape[-1] != n_t:
                    raise ValueError(
                        f"{os.path.basename(path)}: ROI '{r}' has {v.shape[-1]} "
                        f"samples on its last axis but times has {n_t}; "
                        f"stored shape was {z[key].shape}. Axis order changed?")
                out[r] = v
                break
    return out, times, float(z['sfreq']) if 'sfreq' in z.files else None


def reduce_fixpc(v, k, rank_tol=1e-10):
    """FIXPC-k: first k principal components of an ROI's vertex data.

    Returns ``(comp, k_used, rank)``. ``k`` is CAPPED at the numerical rank so a
    requested component that the data does not actually contain cannot be
    manufactured out of round-off — the cap is reported, never silent.
    """
    if v.ndim == 2:                       # already reduced upstream
        return v[:, None, :], 1, 1
    n_tr, n_v, n_t = v.shape
    X = np.moveaxis(v, 1, 0).reshape(n_v, -1)         # (vertices, trials*time)
    X = X - X.mean(axis=1, keepdims=True)
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    rank = int(np.sum(S > (S[0] * rank_tol))) if S.size else 0
    k_used = int(min(k, max(rank, 1)))
    W = U[:, :k_used]                                  # (vertices, k)
    comp = np.einsum('vk,tvn->tkn', W, v)
    return comp, k_used, rank


def lcmv_collapse_score(v):
    """Fraction of an ROI's vertex variance in its first PC.

    A degenerate beamformer solution puts ~all vertices on one global time
    course, so this goes to 1.0. Reported per ROI so collapsed subjects are
    visible rather than silently averaged in.
    """
    if v.ndim != 3:
        return np.nan
    n_tr, n_v, n_t = v.shape
    X = np.moveaxis(v, 1, 0).reshape(n_v, -1)
    X = X - X.mean(axis=1, keepdims=True)
    s = np.linalg.svd(X, compute_uv=False)
    return float(s[0] ** 2 / np.sum(s ** 2)) if s.size else np.nan


# ─────────────────────────────────────────────────────────────────────
# The two estimators, on one window
# ─────────────────────────────────────────────────────────────────────
def _both_conditional(X, order, freqs, fs, src, tgt):
    """F_{src->tgt | rest} from both estimators on the SAME fitted MVAR.

    Returns (parametric_spec, statespace_spec, parametric_time, statespace_time).
    Fitting once and sharing it is what makes this a comparison of the reduced
    model rather than of two independent fits.
    """
    A, Sig = fit_mvar(X, order)
    par_s = conditional_spectral_gc(X, order, freqs, fs, pairs=[(src, tgt)])[(src, tgt)]
    td = time_domain_conditional_gc(X, order, pairs=[(src, tgt)])
    par_t = float(list(td.values())[0])
    ss_t, ss_s = ss_conditional_gc(A, Sig, x=[tgt], y=[src], freqs=freqs, fs=fs)
    return par_s, ss_s, par_t, float(ss_t)


def analyse_window(seg, order, freqs, fs, roi_names, blocks, triples,
                   want_m0=False):
    """All measures for ONE window of one subject.

    ``seg``   : (n_trials, n_channels, n_win) with channels grouped by ROI
    ``blocks``: {roi: [channel indices]} — length 1 for FIXPC1, k for FIXPC-k
    """
    out = {}
    n_roi = len(roi_names)
    scalar = all(len(b) == 1 for b in blocks.values())
    idx = {r: blocks[r][0] for r in roi_names} if scalar else None

    # ---- M0: the identity check (scalar ROIs only) ----
    if want_m0 and scalar and n_roi >= 2:
        a, b = roi_names[0], roi_names[1]
        two = seg[:, [idx[a], idx[b]], :]
        p_xy, _ = pairwise_spectral_gc(two, order, freqs, fs)
        A2, S2 = fit_mvar(two, order)
        _, s_xy = ss_conditional_gc(A2, S2, x=[1], y=[0], freqs=freqs, fs=fs)
        out['m0_max_abs_diff'] = float(np.max(np.abs(p_xy - s_xy)))

    # ---- M1 / M2: fully conditional + time domain, on the joint model ----
    if scalar:
        m1_par, m1_ss, m2_par, m2_ss = {}, {}, {}, {}
        for a, b in itertools.permutations(roi_names, 2):
            ps, ss, pt, st = _both_conditional(seg, order, freqs, fs,
                                               idx[a], idx[b])
            m1_par[(a, b)] = ps; m1_ss[(a, b)] = ss
            m2_par[(a, b)] = pt; m2_ss[(a, b)] = st
        out['m1_par'], out['m1_ss'] = m1_par, m1_ss
        out['m2_par'], out['m2_ss'] = m2_par, m2_ss

    # ---- M3: block GC when the ROIs carry >1 component ----
    else:
        m3_par, m3_ss = {}, {}
        A, Sig = fit_mvar(seg, order)
        for a, b in itertools.permutations(roi_names, 2):
            # parametric block GC: reduced model fitted by dropping the source
            keep = [c for r in roi_names if r != a for c in blocks[r]]
            sub = seg[:, keep, :]
            pos = {r: [keep.index(c) for c in blocks[r]]
                   for r in roi_names if r != a}
            A_r, S_r = fit_mvar(sub, order)
            det_full = np.linalg.det(Sig[np.ix_(blocks[b], blocks[b])])
            det_red = np.linalg.det(S_r[np.ix_(pos[b], pos[b])])
            m3_par[(a, b)] = float(np.log(max(det_red, 1e-300) /
                                          max(det_full, 1e-300)))
            # returns a bare float when freqs/fs are omitted
            st = ss_conditional_gc(A, Sig, x=blocks[b], y=blocks[a])
            m3_ss[(a, b)] = float(st)
        out['m3_par'], out['m3_ss'] = m3_par, m3_ss

    # ---- F: triple-wise conditional, F_{a->c | b} on 3 variables ----
    if triples and scalar:
        f_par, f_ss, f_pair = {}, {}, {}
        pair_cache = {}
        for a, b, c in triples:
            three = seg[:, [idx[a], idx[b], idx[c]], :]
            ps, ss, _, _ = _both_conditional(three, order, freqs, fs, 0, 2)
            f_par[(a, b, c)] = ps
            f_ss[(a, b, c)] = ss
            if (a, c) not in pair_cache:            # the unconditional baseline
                two = seg[:, [idx[a], idx[c]], :]
                pair_cache[(a, c)] = pairwise_spectral_gc(two, order, freqs, fs)[0]
            f_pair[(a, c)] = pair_cache[(a, c)]
        out['f_par'], out['f_ss'], out['f_pair'] = f_par, f_ss, f_pair
    return out


# ─────────────────────────────────────────────────────────────────────
# One subject, one configuration
# ─────────────────────────────────────────────────────────────────────
def run_subject(path, subj, win_ms, order, target_fs, n_pcs, rois, triples,
                step_ms=5.0, normalize='demean'):
    t_start = time.time()
    vert, times, fs_in = load_roi_vertices(path, rois)
    missing = [r for r in rois if r not in vert]
    if missing:
        return dict(subject=subj, error=f'missing ROIs: {missing}')
    fs_in = fs_in or 500.0

    collapse = {r: lcmv_collapse_score(vert[r]) for r in rois}

    # resample, then reduce each ROI to n_pcs components
    up, down = int(round(target_fs)), int(round(fs_in))
    g = np.gcd(up, down)
    chans, blocks, k_caps = [], {}, {}
    for r in rois:
        v = vert[r]
        v = resample_poly(v, up // g, down // g, axis=-1)
        comp, k_used, rank = reduce_fixpc(v, n_pcs)
        k_caps[r] = dict(requested=n_pcs, used=k_used, rank=rank)
        blocks[r] = list(range(len(chans), len(chans) + k_used))
        chans.extend(comp[:, i, :] for i in range(k_used))
    X = np.stack(chans, axis=1)                      # (trials, channels, times)
    t_axis = np.linspace(times[0], times[-1], X.shape[2]) * 1000.0

    if normalize == 'demean':                        # ERP removal
        X = X - X.mean(axis=0, keepdims=True)

    win = int(round(win_ms * target_fs / 1000.0))
    step = max(1, int(round(step_ms * target_fs / 1000.0)))
    if win <= order + 1:
        return dict(subject=subj,
                    error=f'window {win} samples cannot hold order {order}')
    starts = np.arange(0, X.shape[2] - win + 1, step)

    per_window = []
    for wi, s in enumerate(starts):
        per_window.append(analyse_window(
            X[:, :, s:s + win], order, FREQS, target_fs, rois, blocks,
            triples, want_m0=(wi == 0)))

    return dict(subject=subj, window_ms=t_axis[(starts + win // 2).astype(int)],
                per_window=per_window, collapse=collapse, k_caps=k_caps,
                n_trials=int(X.shape[0]), elapsed=time.time() - t_start)


def _stack(per_window, key, freqs):
    """{edge: (n_bands, n_windows)} band-averaged, from a list of per-window dicts."""
    have = [w for w in per_window if key in w]
    if not have:
        return {}
    edges = list(have[0][key].keys())
    band_names = list(BANDS)
    out = {}
    for e in edges:
        arr = np.full((len(band_names), len(per_window)), np.nan)
        for wi, w in enumerate(per_window):
            if key not in w:
                continue
            v = w[key][e]
            if np.isscalar(v) or np.ndim(v) == 0:
                arr[:, wi] = float(v)
            else:
                ba = band_average(np.asarray(v)[:, None], freqs, BANDS,
                                  on_empty='nan')
                arr[:, wi] = [np.squeeze(ba[b]) for b in band_names]
        out[e] = arr
    return out


def save_subject(res, out_dir, subj, task, stim):
    os.makedirs(out_dir, exist_ok=True)
    pw = res['per_window']
    payload = dict(window_ms=res['window_ms'], bands=np.array(list(BANDS)),
                   freqs=FREQS, n_trials=res['n_trials'],
                   collapse=np.array([[k, v] for k, v in res['collapse'].items()],
                                     dtype=object),
                   k_caps=np.array([[k, v['requested'], v['used'], v['rank']]
                                    for k, v in res['k_caps'].items()], dtype=object))
    m0 = [w['m0_max_abs_diff'] for w in pw if 'm0_max_abs_diff' in w]
    if m0:
        payload['m0_max_abs_diff'] = np.array(m0)
    for key in ('m1_par', 'm1_ss', 'm2_par', 'm2_ss', 'm3_par', 'm3_ss'):
        for edge, arr in _stack(pw, key, FREQS).items():
            payload[f'{key}__{edge[0]}__{edge[1]}'] = arr
    for key in ('f_par', 'f_ss'):
        for tri, arr in _stack(pw, key, FREQS).items():
            payload[f'{key}__{tri[0]}__{tri[1]}__{tri[2]}'] = arr
    for pair, arr in _stack(pw, 'f_pair', FREQS).items():
        payload[f'f_pair__{pair[0]}__{pair[1]}'] = arr
    path = os.path.join(out_dir, f'{subj}_{task}_{stim}.npz')
    np.savez_compressed(path, **payload)
    return path


# ─────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--task', required=True, choices=['perception', 'overtProd'])
    p.add_argument('--stim-class', required=True, choices=['prodDiff', 'percDiff'])
    p.add_argument('--method', required=True, choices=['dSPM', 'LCMV'])
    p.add_argument('--atlas', default='custom')
    p.add_argument('--feature-mode', default='vertex_selectkbest',
                   help='which cache directory to read (see config.cache_feat_mode)')
    p.add_argument('--leakage-correction', action='store_true', default=True)
    p.add_argument('--rois', nargs='+',
                   default=['awfa-lh', 'ifc-lh', 'owfa-lh', 'pmc-lh',
                            'tpc-lh', 'vwfa-lh'])
    p.add_argument('--win-ms', type=float, nargs='+', default=[60.0])
    p.add_argument('--order', type=int, nargs='+', default=[6])
    p.add_argument('--n-pcs', type=int, nargs='+', default=[1])
    p.add_argument('--target-fs', type=float, default=200.0)
    p.add_argument('--step-ms', type=float, default=5.0)
    p.add_argument('--triples', choices=['none', 'primary', 'exhaustive'],
                   default='none')
    p.add_argument('--normalize', default='demean', choices=['none', 'demean'])
    p.add_argument('--subjects', nargs='+', default=None)
    p.add_argument('--config-tag', default='run')
    p.add_argument('--out-root', default=None)
    p.add_argument('--n-jobs', type=int, default=64)
    p.add_argument('--self-test', action='store_true',
                   help='verify the parametric/state-space identity and exit')
    p.add_argument('--check', action='store_true',
                   help='resolve every subject cache, report, and exit without computing')
    if '--self-test' in sys.argv:
        for a in ('--task', '--stim-class', '--method'):
            for act in p._actions:
                if a in act.option_strings:
                    act.required = False
    args = p.parse_args()

    if args.self_test:
        rng = np.random.default_rng(0)
        X = rng.normal(size=(150, 2, 60))
        X[:, 1, 4:] += 0.4 * X[:, 0, :-4]
        f = np.arange(2.0, 38.5, 1.0)
        pxy, _ = pairwise_spectral_gc(X, 6, f, 200.0)
        A, S = fit_mvar(X, 6)
        _, sxy = ss_conditional_gc(A, S, x=[1], y=[0], freqs=f, fs=200.0)
        d = float(np.max(np.abs(pxy - sxy)))
        print(f'M0 identity check: max|parametric - state-space| = {d:.2e}')
        print('PASS' if d < 1e-10 else 'FAIL — the two arms are mis-wired')
        return

    triples = []
    if args.triples == 'primary':
        triples = [t for t in PRIMARY_TRIPLES if set(t) <= set(args.rois)]
    elif args.triples == 'exhaustive':
        triples = list(itertools.permutations(args.rois, 3))

    subjects = args.subjects or list(config.SUBJECT_IDS)
    if args.check:
        found = 0
        for s_ in subjects:
            pth = find_cached_npz(args.task, args.method, args.atlas,
                                  args.feature_mode, args.leakage_correction,
                                  s_, args.stim_class)
            print(f'  {s_}: {pth if pth else "MISSING"}')
            found += pth is not None
        print(f'{found}/{len(subjects)} caches resolved for '
              f'{args.task}/{args.method}/{args.atlas}/{args.feature_mode}')
        sys.exit(0 if found else 2)
    out_root = args.out_root or os.path.join(
        os.path.dirname(config.DECODE_OUTPUT_ROOT), 'GC_routes')

    print(f'task={args.task} stim={args.stim_class} method={args.method}')
    print(f'ROIs: {args.rois}')
    print(f'windows={args.win_ms} orders={args.order} n_pcs={args.n_pcs}')
    print(f'triples: {args.triples} ({len(triples)} of '
          f'{len(list(itertools.permutations(args.rois, 3)))} possible; '
          f'{len([t for t in PRIMARY_TRIPLES if set(t) <= set(args.rois)])} primary)')
    print(f'grid: {FREQS[0]:.0f}-{FREQS[-1]:.0f} Hz, bands {list(BANDS)}')

    for win_ms, order, n_pcs in itertools.product(args.win_ms, args.order,
                                                  args.n_pcs):
        tag = (f'{args.config_tag}/win{win_ms:g}ms_order{order}_'
               f'fs{args.target_fs:g}_pc{n_pcs}')
        out_dir = os.path.join(out_root, args.task, args.method, args.atlas,
                               tag, args.stim_class)
        jobs = []
        for s in subjects:
            path = find_cached_npz(args.task, args.method, args.atlas,
                                   args.feature_mode, args.leakage_correction,
                                   s, args.stim_class)
            if path:
                jobs.append((str(path), s))
            else:
                print(f'  [skip] no cache for {s}')
        if not jobs:
            print(f'ERROR: no cached ROI timeseries found for {tag}.', file=sys.stderr)
            print(f'  looked for task={args.task} method={args.method} '
                  f'atlas={args.atlas} feature_mode={args.feature_mode} '
                  f'leakage={args.leakage_correction}', file=sys.stderr)
            print('  run run_source_localize.py first, or check '
                  'ROI_TIMESERIES_EXTERNAL in config.env.', file=sys.stderr)
            sys.exit(2)
        print(f'\n=== {tag}: {len(jobs)} subjects ===')
        t0 = time.time()
        results = Parallel(n_jobs=args.n_jobs, verbose=5)(
            delayed(run_subject)(pth, s, win_ms, order, args.target_fs, n_pcs,
                                 args.rois, triples, args.step_ms,
                                 args.normalize)
            for pth, s in jobs)
        n_ok = 0
        for res in results:
            if res.get('error'):
                print(f"  [error] {res['subject']}: {res['error']}")
                continue
            save_subject(res, out_dir, res['subject'], args.task, args.stim_class)
            n_ok += 1
            caps = [f"{r}:{v['used']}/{v['requested']}(rank {v['rank']})"
                    for r, v in res['k_caps'].items()
                    if v['used'] < v['requested']]
            if caps:
                print(f"  [PC cap] {res['subject']}: {', '.join(caps)}")
            bad = [f'{r}={c:.2f}' for r, c in res['collapse'].items()
                   if np.isfinite(c) and c > 0.90]
            if bad:
                print(f"  [collapse?] {res['subject']}: {', '.join(bad)}")
        print(f'  {n_ok}/{len(jobs)} written in {time.time()-t0:.0f}s -> {out_dir}')


if __name__ == '__main__':
    main()
