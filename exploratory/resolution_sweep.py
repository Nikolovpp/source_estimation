#!/usr/bin/env python3
"""
Inverse-model optimization sweep: ROI resolution metrics across inverse
settings, so the inverse is chosen on resolvability BEFORE the expensive
decoding / GC runs (exploratory/INVERSE_OPTIMIZATION_PLAN.md).

Every configuration is scored with the same ROI-level cross-talk machinery
as exploratory/resolution_analysis.py (CTF = w_i^T K_i P G), plus two
metrics that make configurations comparable on the noise axis:

    noise_gain   sqrt(f_i^T C_ref f_i) / mean |CTF_i| over ROI i's own
                 vertices, with f_i the ROI's sensor-space filter and C_ref
                 the (fixed) shrunk baseline noise covariance: the sensor
                 noise standard deviation that reaches the ROI time course
                 per unit of own-source gain.  Same units for every
                 configuration, lower is better; the price paid for sharper
                 cross-talk functions.
    k_pc1_frac   share of the ROI's projected data covariance captured by
                 its first PC (top eigenvalue of K_i C_data K_i^T / trace):
                 1.0 = every vertex carries one time course (the LCMV
                 collapse signature), lower = real spatial structure.

Families and knobs (the five levers of the plan):

  mn       dSPM / eLORETA on the minimum-norm operator
             snr        regularization, lambda2 = 1/snr^2       (lever 2)
             depth      depth weighting exponent                 (lever 2)
             loose      orientation constraint; > 0 is applied with
                        pick_ori='normal' to stay linear         (lever 5)
             noise_cov  inverse_pipelines.make_noise_cov method  (lever 3)
  lcmv     beamformer: reg, weight_norm, pick_ori, data_cov window,
           noise_cov; k_pc1_frac flags the spatial collapse       (lever 4)
  deflect  DeFleCT ROI filters designed from the leadfield (Hauk &
           Stenroos 2014): unit gain on the ROI's dominant patch
           topography, hard nulls on the other ROIs' patch topographies
           (``nulls=other_rois``, ``null_comp`` SVD components each),
           minimum response elsewhere + noise penalty (snr)      (lever 1)

``--grid star`` (default) varies one knob at a time around the production
settings (dSPM/eLORETA: snr 3, depth 0.8, fixed, shrunk; LCMV: reg 0.05,
unit-noise-gain, max-power, task-window data cov, shrunk); ``--grid full``
adds the full snr x depth x loose factorial for the mn family.

Per subject the work is: one inverse operator per (depth, loose, noise)
cell, one kernel per configuration, then the ROI algebra.  Light enough to
run alongside the decode queue; ~10 min per subject with the star grid.
Resumes: subjects with a finished CSV are skipped.

Usage
-----
    conda activate mne
    python exploratory/resolution_sweep.py --task overtProd --stim-class prodDiff \\
        --atlases custom sensorEquiv --n-jobs 8
    python exploratory/resolution_sweep.py --task perception --stim-class percDiff --n-jobs 8
    # afterwards (reads CSVs only, fine locally):
    python exploratory/resolution_sweep_report.py --task overtProd --stim-class prodDiff

Output
------
    derivatives/source_estimation/RESOLUTION_ANALYSIS/{task}_{stim}/sweep/
        configs.csv                        the grid that was run
        per_subject/{subj}.roi.csv         per ROI x config metrics
        per_subject/{subj}.pair.csv        per ROI-pair x config metrics
        per_subject/{subj}_ctf.npz         CTF vectors (float32) per atlas x config
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import argparse
import itertools
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import mne
from mne.minimum_norm import apply_inverse
from mne.beamformer import apply_lcmv

from config import SUBJECT_IDS, resolve_noise_baseline
from resolution_analysis import (OUT_ROOT, build_roi_sets, label_rows,
                                 leadfield_fixed, sensor_covariance,
                                 roi_weights, ctf_metrics)

MN_METHODS = ['dSPM', 'eLORETA']          # sLORETA == dSPM up to a per-vertex scale
NOISE_OPTS = ['shrunk', 'auto', 'ledoit_wolf', 'adhoc', 'fullepoch', 'identity']
SNR_LEVELS = [1.0, 2.0, 3.0, 5.0, 10.0]
DEPTH_LEVELS = [0.0, 0.4, 0.8]
LOOSE_LEVELS = [0.0, 0.2, 0.4, 0.6]
LCMV_REG = [0.01, 0.05, 0.1, 0.3]
LCMV_WN = ['unit-noise-gain', 'nai', None]
LCMV_ORI = ['max-power', 'normal']
LCMV_DCOV = ['task', 'full']
DEFLECT_SNR = [1.0, 3.0, 10.0]

CFG_COLS = ['config', 'family', 'method', 'snr', 'depth', 'loose', 'noise_cov',
            'reg', 'weight_norm', 'pick_ori', 'data_cov', 'nulls', 'null_comp']

MN_BASE = dict(snr=3.0, depth=0.8, loose=0.0, noise_cov='shrunk')
LCMV_BASE = dict(reg=0.05, weight_norm='unit-noise-gain', pick_ori='max-power',
                 data_cov='task', noise_cov='shrunk')


# ─────────────────────────────────────────────────────────────────────
# Configuration grid
# ─────────────────────────────────────────────────────────────────────
def _mn(method, **kw):
    p = {**MN_BASE, **kw}
    return dict(config=f"mn_{method}_snr{p['snr']:g}_d{p['depth']:g}"
                       f"_l{p['loose']:g}_{p['noise_cov']}",
                family='mn', method=method, **p)


def _lcmv(**kw):
    p = {**LCMV_BASE, **kw}
    return dict(config=f"lcmv_reg{p['reg']:g}_{p['weight_norm'] or 'none'}"
                       f"_{p['pick_ori']}_{p['data_cov']}_{p['noise_cov']}",
                family='lcmv', method='LCMV', **p)


def _deflect(snr, null_comp, nulls):
    return dict(config=f'deflect_snr{snr:g}_null{null_comp}_{nulls}',
                family='deflect', method='DeFleCT', snr=snr,
                null_comp=null_comp, nulls=nulls, noise_cov='shrunk')


def baseline_config_names():
    return {m: _mn(m)['config'] for m in MN_METHODS} | {'LCMV': _lcmv()['config']}


def build_configs(grid, families):
    cfgs, seen = [], set()

    def add(c):
        if c['config'] not in seen:
            seen.add(c['config'])
            cfgs.append(c)

    if 'mn' in families:
        for m in MN_METHODS:
            for s in SNR_LEVELS:
                add(_mn(m, snr=s))
            for d in DEPTH_LEVELS:
                add(_mn(m, depth=d))
            for l in LOOSE_LEVELS:
                add(_mn(m, loose=l))
            for n in NOISE_OPTS:
                add(_mn(m, noise_cov=n))
            if grid == 'full':
                for s, d, l in itertools.product(SNR_LEVELS, DEPTH_LEVELS,
                                                 LOOSE_LEVELS):
                    add(_mn(m, snr=s, depth=d, loose=l))
    if 'lcmv' in families:
        for r in LCMV_REG:
            add(_lcmv(reg=r))
        for w in LCMV_WN:
            add(_lcmv(weight_norm=w))
        for o in LCMV_ORI:
            add(_lcmv(pick_ori=o))
        for dc in LCMV_DCOV:
            add(_lcmv(data_cov=dc))
        for n in NOISE_OPTS + ['none']:
            add(_lcmv(noise_cov=n))
        if grid == 'full':
            for r, w, o in itertools.product(LCMV_REG, LCMV_WN, LCMV_ORI):
                add(_lcmv(reg=r, weight_norm=w, pick_ori=o))
    if 'deflect' in families:
        for s, nc, nl in itertools.product(DEFLECT_SNR, [1, 2],
                                           ['other_rois', 'none']):
            add(_deflect(s, nc, nl))
    for c in cfgs:                       # uniform columns
        for k in CFG_COLS:
            c.setdefault(k, None)
    return cfgs


# ─────────────────────────────────────────────────────────────────────
# DeFleCT filters
# ─────────────────────────────────────────────────────────────────────
def patch_topographies(G, rows, k):
    """First k left singular vectors of the ROI patch leadfield (dominant
    field patterns of the patch), sign-aligned with the mean column."""
    U, _, _ = np.linalg.svd(G[:, rows], full_matrices=False)
    U = U[:, :k].copy()
    if U[:, 0] @ G[:, rows].mean(axis=1) < 0:
        U[:, 0] *= -1
    return U


def deflect_filter(G, Minv, rows_pass, rows_null, null_comp):
    """Hauk & Stenroos (2014) DeFleCT: sensor filter w with w^T u_pass = 1,
    w^T u_null = 0 for every null topography, and minimum
    w^T (G G^T + lambda C) w elsewhere.  ``Minv`` = pinv(G G^T + lambda C)."""
    F = [patch_topographies(G, rows_pass, 1)]
    F += [patch_topographies(G, r, null_comp) for r in rows_null]
    F = np.hstack(F)
    d = np.zeros(F.shape[1])
    d[0] = 1.0
    A = F.T @ Minv @ F
    A = A + 1e-10 * np.trace(A) / len(A) * np.eye(len(A))
    return Minv @ F @ np.linalg.solve(A, d)


def deflect_Minv(G, C_ref, snr):
    GG = G @ G.T
    Cs = C_ref * (np.trace(GG) / np.trace(C_ref))
    return np.linalg.pinv(GG + Cs / snr ** 2, rcond=1e-12, hermitian=True)


# ─────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────
def pair_metrics_sweep(ctfs, psfs, rows):
    out = []
    names = list(ctfs)
    for a in names:
        ca = ctfs[a]
        own_c = np.mean(ca[rows[a]] ** 2)
        for b in names:
            if a == b:
                continue
            cb = ctfs[b]
            cos = abs(float(ca @ cb) / (np.linalg.norm(ca) * np.linalg.norm(cb)))
            xt = float(np.mean(ca[rows[b]] ** 2) / own_c)
            if psfs is not None:
                pb = psfs[b]
                ps = float(np.mean(pb[rows[a]] ** 2) / np.mean(pb[rows[b]] ** 2))
            else:
                ps = np.nan
            out.append(dict(roi_i=a, roi_j=b, pattern_cos=cos,
                            ctf_xtalk=xt, psf_xtalk=ps))
    return out


def k_pc1_share(K_i, C_data):
    ev = np.linalg.eigvalsh(K_i @ C_data @ K_i.T)
    ev = np.clip(ev, 0, None)
    return float(ev[-1] / ev.sum()) if ev.sum() > 0 else np.nan


# ─────────────────────────────────────────────────────────────────────
# Per-subject worker
# ─────────────────────────────────────────────────────────────────────
_G = {}


def _init(fwd, src, roi_sets, task, stim, configs, out_dir):
    _G.update(fwd=fwd, src=src, roi_sets=roi_sets, task=task, stim=stim,
              configs=configs, out_dir=out_dir)


def _subject(subj):
    from data_loader import load_subject_epochs
    from inverse_pipelines import (make_minimum_norm_inverse, make_lcmv_filters,
                                   make_noise_cov, _compute_data_rank)
    t0 = time.time()
    try:
        epochs, _, _ = load_subject_epochs(subj, _G['task'], _G['stim'])
    except FileNotFoundError as e:
        print(f'  {subj}: {e} — SKIP')
        return None
    fwd = _G['fwd']
    bl_lo, bl_hi, _ = resolve_noise_baseline(
        _G['task'], float(epochs.times[0]), float(epochs.times[-1]))
    G, verts, rr = leadfield_fixed(fwd, epochs.info)
    n_lh = len(verts[0])
    n_ch = len(epochs.ch_names)
    C_data = sensor_covariance(epochs)
    rank = _compute_data_rank(epochs)
    ev = mne.EvokedArray(np.eye(n_ch), epochs.info, tmin=0.0, nave=1,
                         verbose=False)

    rows_by_atlas = {
        atlas: {r: label_rows(lab, verts[0], verts[1])
                for r, lab in labs.items()}
        for atlas, labs in _G['roi_sets'].items()}

    noise_cache = {}

    def noise(method):
        if method not in noise_cache:
            noise_cache[method] = (None if method == 'none' else
                                   make_noise_cov(epochs, bl_lo, bl_hi, rank,
                                                  method))
        return noise_cache[method]

    C_ref = noise('shrunk').data                 # fixed reference for noise_gain
    inv_cache, minv_cache = {}, {}
    roi_rec, pair_rec, ctf_store = [], [], {}

    for cfg in _G['configs']:
        tc = time.time()
        fam = cfg['family']
        K = None
        try:
            if fam == 'mn':
                key = (cfg['depth'], cfg['loose'], cfg['noise_cov'])
                if key not in inv_cache:
                    inv_cache[key] = make_minimum_norm_inverse(
                        epochs, fwd, bl_lo, bl_hi, depth=cfg['depth'],
                        loose=cfg['loose'], noise_cov=noise(cfg['noise_cov']))[0]
                stc = apply_inverse(
                    ev, inv_cache[key], lambda2=1.0 / cfg['snr'] ** 2,
                    method=cfg['method'],
                    pick_ori='normal' if cfg['loose'] > 0 else None,
                    verbose=False)
                K = np.asarray(stc.data, float)
            elif fam == 'lcmv':
                filters = make_lcmv_filters(
                    epochs, fwd, bl_lo, bl_hi, reg=cfg['reg'],
                    weight_norm=cfg['weight_norm'], pick_ori=cfg['pick_ori'],
                    data_cov_window=cfg['data_cov'],
                    noise_cov_method=cfg['noise_cov'],
                    noise_cov=noise(cfg['noise_cov']))[0]
                K = np.asarray(apply_lcmv(ev, filters, verbose=False).data, float)
            elif fam == 'deflect':
                if cfg['snr'] not in minv_cache:
                    minv_cache[cfg['snr']] = deflect_Minv(G, C_ref, cfg['snr'])
                Minv = minv_cache[cfg['snr']]
        except Exception as e:
            print(f'  {subj}: {cfg["config"]} FAILED: {e}', flush=True)
            continue
        if K is not None and K.shape[0] != G.shape[1]:
            print(f'  {subj}: {cfg["config"]}: kernel has {K.shape[0]} rows, '
                  f'leadfield {G.shape[1]} — skipped', flush=True)
            continue

        meta = {k: cfg[k] for k in CFG_COLS}
        for atlas, labs in _G['roi_sets'].items():
            rows = rows_by_atlas[atlas]
            names = list(labs)
            ctfs, psfs = {}, ({} if K is not None else None)
            for r, lab in labs.items():
                if fam == 'deflect':
                    nulls = ([rows[o] for o in names if o != r]
                             if cfg['nulls'] == 'other_rois' else [])
                    f = deflect_filter(G, Minv, rows[r], nulls, cfg['null_comp'])
                    kpc = np.nan
                else:
                    K_i = K[rows[r]]
                    w = roi_weights(K_i, C_data, lab, _G['src'], cfg['method'],
                                    'pc1')
                    f = w @ K_i
                    kpc = k_pc1_share(K_i, C_data)
                    psfs[r] = K @ (G[:, rows[r]].mean(axis=1))
                ctf = f @ G
                ctfs[r] = ctf
                m = ctf_metrics(ctf, rows[r], rr, n_lh)
                own_gain = float(np.mean(np.abs(ctf[rows[r]])))
                ng = float(np.sqrt(max(f @ C_ref @ f, 0.0)) / own_gain) \
                    if own_gain > 0 else np.nan
                roi_rec.append(dict(subject=subj, atlas=atlas, roi=r,
                                    n_vertices=len(rows[r]), noise_gain=ng,
                                    k_pc1_frac=kpc, **m, **meta))
            for d in pair_metrics_sweep(ctfs, psfs, rows):
                pair_rec.append(dict(subject=subj, atlas=atlas, **d, **meta))
            ctf_store[f'{atlas}__{cfg["config"]}'] = np.stack(
                [ctfs[r] for r in names]).astype(np.float32)
        print(f'  {subj}: {cfg["config"]} ({time.time() - tc:.0f}s)', flush=True)
        del K

    ps = _G['out_dir'] / 'per_subject'
    pd.DataFrame(pair_rec).to_csv(ps / f'{subj}.pair.csv', index=False)
    np.savez_compressed(
        ps / f'{subj}_ctf.npz', n_lh=n_lh,
        **{f'names__{a}': np.array(list(l)) for a, l in _G['roi_sets'].items()},
        **ctf_store)
    pd.DataFrame(roi_rec).to_csv(ps / f'{subj}.roi.csv', index=False)  # last: marks done
    print(f'  {subj}: done in {(time.time() - t0) / 60:.1f} min', flush=True)
    return subj


# ─────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--task', required=True, choices=['perception', 'overtProd'])
    p.add_argument('--stim-class', default='prodDiff',
                   choices=['prodDiff', 'percDiff'])
    p.add_argument('--atlases', nargs='+', default=['custom', 'sensorEquiv'],
                   choices=['custom', 'sensorEquiv', 'HCPMMP1', 'aparc',
                            'Schaefer200'])
    p.add_argument('--families', nargs='+', default=['mn', 'lcmv', 'deflect'],
                   choices=['mn', 'lcmv', 'deflect'])
    p.add_argument('--grid', default='star', choices=['star', 'full'])
    p.add_argument('--subjects', nargs='+', default=None)
    p.add_argument('--n-jobs', type=int, default=4)
    p.add_argument('--out-dir', default=None)
    p.add_argument('--list-configs', action='store_true', default=False,
                   help='print the grid and exit')
    return p.parse_args()


def main():
    args = parse_args()
    configs = build_configs(args.grid, args.families)
    if args.list_configs:
        print(pd.DataFrame(configs)[CFG_COLS].to_string(index=False))
        print(f'\n{len(configs)} configurations')
        return
    subjects = args.subjects or SUBJECT_IDS
    out_dir = Path(args.out_dir) if args.out_dir else \
        OUT_ROOT / f'{args.task}_{args.stim_class}' / 'sweep'
    (out_dir / 'per_subject').mkdir(parents=True, exist_ok=True)
    pd.DataFrame(configs)[CFG_COLS].to_csv(out_dir / 'configs.csv', index=False)
    print(f'Resolution sweep: {args.task}/{args.stim_class}  grid={args.grid}')
    print(f'  {len(configs)} configs ({", ".join(args.families)}); '
          f'atlases {args.atlases}; subjects {len(subjects)}; n_jobs {args.n_jobs}')
    print(f'  out {out_dir}\n')

    from forward_model import setup_fsaverage, make_forward
    subjects_dir, _, src, bem = setup_fsaverage()
    roi_sets = build_roi_sets(subjects_dir, args.atlases)
    for atlas, labs in roi_sets.items():
        print(f'  {atlas}: {list(labs)}')

    todo = [s for s in subjects
            if not (out_dir / 'per_subject' / f'{s}.roi.csv').exists()]
    print(f'\n  {len(todo)} subject(s) to compute '
          f'({len(subjects) - len(todo)} cached)')
    if todo:
        from data_loader import load_subject_epochs
        ep0, _, _ = load_subject_epochs(todo[0], args.task, args.stim_class)
        fwd = make_forward(ep0.info, src, bem)
        del ep0
        with Pool(args.n_jobs, initializer=_init,
                  initargs=(fwd, src, roi_sets, args.task, args.stim_class,
                            configs, out_dir)) as pool:
            pool.map(_subject, todo)
    print(f'\nWrote {out_dir}\nNext: python exploratory/resolution_sweep_report.py '
          f'--task {args.task} --stim-class {args.stim_class}')


if __name__ == '__main__':
    main()
