#!/usr/bin/env python3
"""Robust characterization of the per-epoch RESAMPLE edge artifact in
source-space Granger causality, and its removal by fix #1.

METHODS-PAPER ANALYSIS — not part of the current manuscript. Outputs live
under ``derivatives/source_estimation/methods_paper_analyses/``.

What it does
------------
Real LCMV source ROI time courses (fs 2048) are pushed through the EXACT
production GC code (``run_granger.compute_subject_gc``) TWICE, changing ONLY
the resample step:

    NAIVE : scipy ``resample_poly`` default zero-padding      (= pre-fix #1)
    FIX   : ``run_granger.resample_channels`` (edge-pad+crop)  (= fix #1)

Everything else — source TCs, MVAR order, window, band averaging — is
byte-identical between the two arms, so any difference in the GC time course
is attributable to the resample edge handling alone.  Characterized across:
    - all subjects (SUBJECT_IDS; unstable-covariance subjects auto-skip)
    - all 4 bands (theta / alpha / low_beta / high_beta)
    - all directed ROI edges of the chosen atlas

Two distinct edge effects (this script isolates only #1):
    1. resample zero-pad transient      — NAIVE-only, removed by fix #1  <-- HERE
    2. moving-window MVAR boundary ramp — BOTH arms; handled separately by the
       GC_TASK_START/END crop at stats time (granger_stats.py), not here.

Parallelism (built for a many-core workstation)
-----------------------------------------------
The long pole is LCMV per subject (largely single-threaded MNE).  Two modes,
one level of parallelism at a time (never nested):

  --subject-jobs N (>1)  : N subjects in parallel (multiprocessing.Pool);
                           GC runs sequentially inside each worker.
                           BEST for a big box.  RAM is the limit (~a few GB
                           per concurrent subject with run_lcmv_lowram).
  --subject-jobs 1       : subjects sequential; GC parallelizes over ROI
                           pairs with --gc-jobs.  Good for a modest box.

Resumable + cache-cheap
-----------------------
  source_tc_cache/{subj}.npz  : LCMV output, so re-analysis never re-runs the
                                inverse (band/edge/order/window changes reuse it
                                for GC — only order/window need a GC recompute).
  per_subject/{subj}.npz      : that subject's naive+fix GC (all bands/edges).
                                Present -> subject is skipped (unless --overwrite).
Aggregation + all figures rebuild from the per-subject caches; ``--plot-only``
rebuilds figures with no compute.

Remote usage
------------
    conda activate mne
    # many-core box, all 20 subjects in parallel, custom atlas, overtProd:
    python resample_edge_artifact.py --atlas custom --task overtProd \\
        --stim-class prodDiff --subject-jobs 20

    # cross-atlas robustness (heavier — more ROIs/edges):
    python resample_edge_artifact.py --atlas HCPMMP1 --task overtProd \\
        --stim-class prodDiff --subject-jobs 16

    # perception task:
    python resample_edge_artifact.py --atlas custom --task perception \\
        --stim-class percDiff --subject-jobs 20

    # just rebuild figures from caches:
    python resample_edge_artifact.py --atlas custom --task overtProd \\
        --stim-class prodDiff --plot-only
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('BLIS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import sys
import time
import argparse
import warnings
import multiprocessing as mp
from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly

warnings.filterwarnings('ignore')
# methods-paper script lives one level down; add the source_estimation root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import (SUBJECT_IDS, SPEECH_ROIS, BASELINE_WINDOWS,
                    GC_BASELINE_WINDOWS, GC_TASK_START, GC_TASK_END, PROJECT_ROOT)
from data_loader import load_subject_epochs
from forward_model import setup_fsaverage, make_forward, build_roi_labels
from inverse_pipelines import run_lcmv_lowram
from decoding_io import filter_roi_dict
from granger import DEFAULT_BANDS
import run_granger as rg

FIX = rg.resample_channels
BANDS = list(DEFAULT_BANDS)                 # theta, alpha, low_beta, high_beta
K_EDGE = 3                                   # first/last K windows = "edge"
HEATMAP_MAX_EDGES = 40                       # cap heatmap rows for big atlases
C_NAI, C_FIX = '#d1495b', '#2e6f95'          # red = naive(artifact), blue = fix#1


# ─────────────────────────────────────────────────────────────────────
# Paths (all derived from config.env's EEG_PROJECT_ROOT -> portable)
# ─────────────────────────────────────────────────────────────────────
def out_paths(atlas, task, stim):
    base = (PROJECT_ROOT / 'derivatives' / 'source_estimation'
            / 'methods_paper_analyses' / 'resample_edge_artifact'
            / atlas / f'{task}_{stim}')
    return dict(base=base, tc=base / 'source_tc_cache',
                per=base / 'per_subject', npz=base / 'edge_artifact_curves.npz')


# ─────────────────────────────────────────────────────────────────────
# Signal + GC helpers
# ─────────────────────────────────────────────────────────────────────
def resample_naive(x, sfreq, target_fs, *a, **k):
    """Pre-fix #1 behaviour: plain resample_poly, default zero-padding."""
    if abs(sfreq - target_fs) < 1.0:
        return x, float(sfreq)
    frac = Fraction(int(round(target_fs)), int(round(sfreq))).limit_denominator(1000)
    up, down = frac.numerator, frac.denominator
    xr = resample_poly(x, up, down, axis=-1)
    return xr, float(sfreq) * up / down


def lcmv_tcs(subj, task, stim, fwd, src, labels, names, bl, tc_dir):
    """LCMV -> {roi:(n_ep,n_t)}; cached per subject."""
    tc_dir.mkdir(parents=True, exist_ok=True)
    cache = tc_dir / f'{subj}.npz'
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        nm = list(d['names']); X = d['X_roi']
        return {nm[i]: X[:, i, :] for i in range(len(nm))}, np.asarray(d['times']), float(d['sf'])
    epochs, _y, sf = load_subject_epochs(subj, task, stim, fs=2000)
    X_roi, times = run_lcmv_lowram(epochs, fwd, bl[0], bl[1], labels, src,
                                   feature_mode='pca_flip')
    np.savez_compressed(cache, X_roi=X_roi, times=np.asarray(times), sf=sf,
                        names=np.array(names))
    return {names[i]: X_roi[:, i, :] for i in range(len(names))}, np.asarray(times), float(sf)


def gc_both(roi, times, sf, order, win_ms, target_fs, n_jobs):
    rg.resample_channels = FIX
    res_fix = rg.compute_subject_gc(roi, times, sf, order=order, win_ms=win_ms,
                                    target_fs=target_fs, n_jobs=n_jobs)
    rg.resample_channels = resample_naive
    res_nai = rg.compute_subject_gc(roi, times, sf, order=order, win_ms=win_ms,
                                    target_fs=target_fs, n_jobs=n_jobs)
    rg.resample_channels = FIX
    return res_fix, res_nai


def directed_stack(res):
    """(labels, curves (n_bands, n_edges, n_win)) for all directed edges."""
    names = list(res['roi_names'])
    pi, pj = res['pair_i'], res['pair_j']
    labels = []
    cur = np.zeros((len(BANDS), 2 * len(pi), res['window_ms'].size))
    e = 0
    for p in range(len(pi)):
        i, j = int(pi[p]), int(pj[p])
        labels.append(f'{names[i]}->{names[j]}')
        for bi, b in enumerate(BANDS):
            cur[bi, e] = res['fxy'][b][p]
        e += 1
        labels.append(f'{names[j]}->{names[i]}')
        for bi, b in enumerate(BANDS):
            cur[bi, e] = res['fyx'][b][p]
        e += 1
    return labels, cur


# ─────────────────────────────────────────────────────────────────────
# Per-subject worker (writes its own cache -> resumable + parallel-safe)
# ─────────────────────────────────────────────────────────────────────
_S = {}   # shared read-only context for pool workers (set via initializer)


def _init(shared):
    _S.update(shared)


def process_subject(subj):
    P = _S
    paths = P['paths']
    per = paths['per'] / f'{subj}.npz'
    if per.exists() and not P['overwrite']:
        return (subj, 'cached')
    t0 = time.time()
    try:
        roi, times, sf = lcmv_tcs(subj, P['task'], P['stim'], P['fwd'], P['src'],
                                  P['labels'], P['names'], P['bl'], paths['tc'])
    except Exception as e:
        return (subj, f'LCMV-FAIL {e!r}')
    # mechanism arrays (time-domain resample difference) for this subject
    V = np.stack([roi[n] for n in P['names']], axis=0)
    Vfix, fs = FIX(V, sf, P['target_fs'])
    Vnai, _ = resample_naive(V, sf, P['target_fs'])
    nt = min(Vfix.shape[2], Vnai.shape[2])
    tvec = (times[0] + np.arange(nt) / fs) * 1000.0
    absdiff = np.abs(Vnai[:, :, :nt] - Vfix[:, :, :nt]).mean(axis=1)
    si = P['names'].index(P['seed'][0])
    try:
        res_fix, res_nai = gc_both(roi, times, sf, P['order'], P['win_ms'],
                                   P['target_fs'], P['gc_jobs'])
    except Exception as e:
        return (subj, f'GC-FAIL {e!r}')
    el, cf = directed_stack(res_fix)
    _,  cn = directed_stack(res_nai)
    paths['per'].mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        per, cf=cf, cn=cn, wref=res_fix['window_ms'], edge_labels=np.array(el),
        mech_t=tvec, mech_absdiff=absdiff, mech_names=np.array(P['names']),
        mech_ex_fix=Vfix[si, 0, :nt], mech_ex_nai=Vnai[si, 0, :nt])
    return (subj, f'ok {(time.time()-t0)/60:.1f}min')


# ─────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────
def compute(args):
    paths = out_paths(args.atlas, args.task, args.stim_class)
    for p in (paths['base'], paths['tc'], paths['per']):
        p.mkdir(parents=True, exist_ok=True)
    subjects = args.subjects or SUBJECT_IDS
    task, stim = args.task, args.stim_class
    bl = BASELINE_WINDOWS[task]

    print(f'Resample edge artifact | {args.atlas} | {task}/{stim} | {len(subjects)} subj '
          f'| bands={BANDS}')
    print('Setting up fsaverage + ROI labels...')
    subjects_dir, _fs, src, bem = setup_fsaverage()
    if args.atlas in SPEECH_ROIS:
        roi_dict = build_roi_labels(subjects_dir, atlas=args.atlas,
                                    composite_rois=SPEECH_ROIS[args.atlas])
    else:
        roi_dict = build_roi_labels(subjects_dir, atlas=args.atlas)
    if args.roi_subset:
        roi_dict = filter_roi_dict(roi_dict, args.roi_subset, args.atlas)
    names = list(roi_dict); labels = list(roi_dict.values())
    n_edges = len(names) * (len(names) - 1)
    print(f'  ROIs ({len(names)}): {names}')
    print(f'  directed edges: {n_edges}')

    seed = tuple(args.seed) if args.seed else ('awfa-lh', 'ifc-lh')
    if seed[0] not in names or seed[1] not in names:
        seed = (names[0], names[1])
        print(f'  seed pair not in atlas; using {seed}')

    print('Building forward model...')
    info_ep, _, _ = load_subject_epochs(subjects[0], task, stim, fs=2000)
    fwd = make_forward(info_ep.info, src, bem); del info_ep, bem

    # ONE level of parallelism only: parallel subjects -> GC sequential inside.
    gc_jobs = 1 if args.subject_jobs > 1 else args.gc_jobs
    shared = dict(paths=paths, task=task, stim=stim, fwd=fwd, src=src,
                  labels=labels, names=names, bl=bl, seed=seed,
                  order=args.order, win_ms=args.win_ms, target_fs=args.target_fs,
                  gc_jobs=gc_jobs, overwrite=args.overwrite)

    t0 = time.time()
    if args.subject_jobs > 1:
        print(f'  parallel: {args.subject_jobs} subjects at a time '
              f'(GC sequential per subject)\n')
        ctx = mp.get_context('fork')  # inherit fwd/src cheaply; Linux workstation
        with ctx.Pool(args.subject_jobs, initializer=_init, initargs=(shared,)) as pool:
            for subj, status in pool.imap_unordered(process_subject, subjects):
                print(f'  {subj}: {status}', flush=True)
    else:
        print(f'  sequential subjects; GC over pairs with {gc_jobs} jobs\n')
        _init(shared)
        for subj in subjects:
            print(f'  {subj}: {process_subject(subj)[1]}', flush=True)
    print(f'\ncompute wall-clock: {(time.time()-t0)/60:.1f} min')

    aggregate(paths, task)
    return paths


def aggregate(paths, task):
    """Stack all per-subject caches into one npz for the figures."""
    files = sorted(paths['per'].glob('*.npz'))
    if not files:
        print('  no per-subject caches to aggregate'); return None
    fix_list, nai_list, done, wref, el, mech = [], [], [], None, None, None
    for f in files:
        d = np.load(f, allow_pickle=True)
        fix_list.append(d['cf']); nai_list.append(d['cn'])
        done.append(f.stem); wref = d['wref']; el = list(d['edge_labels'])
        if mech is None:
            mech = {k: d[k] for k in ('mech_t', 'mech_absdiff', 'mech_names',
                                      'mech_ex_fix', 'mech_ex_nai')}
            mech['subj'] = f.stem
    nw = min(a.shape[-1] for a in fix_list)
    fix_all = np.stack([a[..., :nw] for a in fix_list])
    nai_all = np.stack([a[..., :nw] for a in nai_list])
    np.savez(paths['npz'], wref=wref[:nw], bands=np.array(BANDS),
             edge_labels=np.array(el), subjects=np.array(done),
             fix_all=fix_all, nai_all=nai_all,
             gc_bl=np.array(GC_BASELINE_WINDOWS[task]),
             tstart=GC_TASK_START[task], tend=GC_TASK_END[task], **{
                 'mech_t': mech['mech_t'], 'mech_absdiff': mech['mech_absdiff'],
                 'mech_names': mech['mech_names'], 'mech_ex_fix': mech['mech_ex_fix'],
                 'mech_ex_nai': mech['mech_ex_nai'], 'mech_subj': mech['subj']})
    print(f'  aggregated {len(done)} subjects -> {paths["npz"]}')
    return paths['npz']


# ─────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────
def figures(paths):
    npz = paths['npz']
    if not npz.exists():
        print(f'  no aggregate npz at {npz} — run compute first'); return
    d = np.load(npz, allow_pickle=True)
    w = d['wref']; bands = [str(b) for b in d['bands']]; el = [str(x) for x in d['edge_labels']]
    fix_all, nai_all = d['fix_all'], d['nai_all']
    nsub, nb, ne, nw = fix_all.shape
    gc_bl = d['gc_bl']; ts, te = float(d['tstart'])*1000, float(d['tend'])*1000
    bl_ms = (gc_bl[0]*1000, gc_bl[1]*1000)
    subj0 = str(d['mech_subj'])
    lb = bands.index('low_beta') if 'low_beta' in bands else 0
    seed_fwd = next((x for x in el if x.startswith('awfa') and 'ifc' in x), el[0])
    fig_dir = paths['base']

    def shade(ax):
        ax.axvspan(*bl_ms, color='0.85', alpha=0.7)
        ax.axvline(ts, color='0.4', ls='--', lw=0.9)
        ax.axvline(te, color='0.4', ls='--', lw=0.9)

    # 1. mechanism -------------------------------------------------------
    t = d['mech_t']; mnames = [str(x) for x in d['mech_names']]; absdiff = d['mech_absdiff']
    s0 = seed_fwd.split('->')[0]; s1 = seed_fwd.split('->')[1]
    ai = mnames.index(s0); ii = mnames.index(s1) if s1 in mnames else (1 if len(mnames) > 1 else 0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    ax.plot(t, absdiff[ai], color='#4c72b0', lw=1.4, label=s0)
    ax.plot(t, absdiff[ii], color='#dd8452', lw=1.4, label=mnames[ii])
    ax.set_title('Where naïve and fixed resample differ\n'
                 '(ensemble-mean |naïve − fix #1|, per time point)')
    ax.set_xlabel('time (ms)'); ax.set_ylabel('|Δ virtual channel| (a.u.)')
    ax.legend(frameon=False, fontsize=9)
    ax.axvline(t[0], color='0.6', ls=':', lw=0.8); ax.axvline(t[-1], color='0.6', ls=':', lw=0.8)
    ax = axes[1]; nE = int(0.12 * t.size)
    ax.plot(t[:nE], d['mech_ex_nai'][:nE], color=C_NAI, lw=1.8, marker='.', ms=4, label='naïve (zero-pad)')
    ax.plot(t[:nE], d['mech_ex_fix'][:nE], color=C_FIX, lw=1.8, label='fix #1 (edge-pad)')
    ax.set_title(f'Leading-edge zoom, one epoch ({s0})')
    ax.set_xlabel('time (ms)'); ax.set_ylabel('virtual channel (a.u.)')
    ax.legend(frameon=False, fontsize=9)
    fig.suptitle(f'Resample edge artifact — mechanism  ({subj0})', fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(fig_dir / 'gc_edge_artifact__mechanism.png', dpi=150); plt.close(fig)

    # 2. before/after seed edge, all bands (subject 0) -------------------
    si = el.index(seed_fwd)
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True)
    for bi, b in enumerate(bands):
        ax = axes.flat[bi]; shade(ax)
        ax.plot(w, nai_all[0, bi, si], color=C_NAI, lw=1.4, label='naïve (pre-fix #1)')
        ax.plot(w, fix_all[0, bi, si], color=C_FIX, lw=1.4, label='fix #1')
        ax.set_title(b); ax.set_ylabel('GC')
        if bi == 0:
            ax.legend(frameon=False, fontsize=8, loc='upper right')
        if bi >= 2:
            ax.set_xlabel('window-start time (ms)')
    fig.suptitle(f'{seed_fwd.replace("->", " → ")} GC, all bands ({subj0}) — '
                 f'naïve edge spikes, fix #1 clean; interiors identical',
                 fontweight='bold', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(fig_dir / 'gc_edge_artifact__gc_beforeafter.png', dpi=150); plt.close(fig)

    # 3. systematic |naïve − fix|(time) per band ------------------------
    delta = np.abs(nai_all - fix_all)
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True)
    for bi, b in enumerate(bands):
        ax = axes.flat[bi]; db = delta[:, bi].reshape(-1, nw); shade(ax)
        ax.fill_between(w, 0, db.max(0), color=C_NAI, alpha=0.22, label='max across edges×subj')
        ax.plot(w, db.mean(0), color=C_NAI, lw=1.6, label='mean across edges×subj')
        ax.set_title(b); ax.set_ylabel('| naïve − fix #1 | GC')
        if bi == 0:
            ax.legend(frameon=False, fontsize=8, loc='upper center')
        if bi >= 2:
            ax.set_xlabel('window-start time (ms)')
    fig.suptitle('The naïve−fix GC difference is confined to the epoch-edge windows '
                 f'in every band\npooled over {ne} directed edges × {nsub} subjects; '
                 'interior windows are bit-identical', fontweight='bold', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(fig_dir / 'gc_edge_artifact__systematic_bands.png', dpi=150); plt.close(fig)

    # 4. peak edge |Δ| heatmap: edges × bands ---------------------------
    edge_win = np.concatenate([np.arange(K_EDGE), np.arange(nw - K_EDGE, nw)])
    peak_med = np.median(delta[..., edge_win].max(-1), axis=0)      # (nb, ne)
    order = np.argsort(-peak_med[lb])
    if ne > HEATMAP_MAX_EDGES:
        order = order[:HEATMAP_MAX_EDGES]
    M = peak_med[:, order].T
    fig, ax = plt.subplots(figsize=(7.2, min(9, 1.5 + 0.24 * len(order))))
    im = ax.imshow(M, aspect='auto', cmap='magma')
    ax.set_xticks(range(nb)); ax.set_xticklabels(bands, rotation=30, ha='right')
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([el[o].replace('->', '→') for o in order], fontsize=6)
    cap = f' (top {HEATMAP_MAX_EDGES})' if ne > HEATMAP_MAX_EDGES else ''
    ax.set_title(f'Peak edge-window |naïve − fix #1| GC{cap}\n'
                 f'median over {nsub} subjects — every edge, every band',
                 fontsize=11, fontweight='bold')
    fig.colorbar(im, ax=ax, shrink=0.6, label='peak |Δ| GC at edge windows')
    fig.tight_layout()
    fig.savefig(fig_dir / 'gc_edge_artifact__edge_band_heatmap.png', dpi=150); plt.close(fig)

    # 5. impact: fraction of cases whose global-max GC is on an edge -----
    def frac_edge_argmax(arr):
        am = arr.argmax(-1)
        on = (am < K_EDGE) | (am >= nw - K_EDGE)
        return on.mean(axis=(0, 2)) * 100
    fn, ff = frac_edge_argmax(nai_all), frac_edge_argmax(fix_all)
    x = np.arange(nb); bw = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x - bw/2, fn, bw, color=C_NAI, label='naïve (pre-fix #1)')
    ax.bar(x + bw/2, ff, bw, color=C_FIX, label='fix #1')
    for xi, (a, b_) in enumerate(zip(fn, ff)):
        ax.text(xi - bw/2, a + 0.5, f'{a:.0f}%', ha='center', fontsize=8)
        ax.text(xi + bw/2, b_ + 0.5, f'{b_:.0f}%', ha='center', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(bands)
    ax.set_ylabel(f'% of (edge × subject) cases with\nglobal-max GC in first/last {K_EDGE} windows')
    ax.set_title(f'Practical impact: naïve resample forces the peak GC onto an epoch edge\n'
                 f'{ne} edges × {nsub} subjects per band', fontsize=11, fontweight='bold')
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / 'gc_edge_artifact__impact.png', dpi=150); plt.close(fig)
    print(f'  figures ({nsub} subj, {nb} bands, {ne} edges) -> {fig_dir}')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--task', default='overtProd', choices=['overtProd', 'perception'])
    p.add_argument('--stim-class', default='prodDiff', choices=['prodDiff', 'percDiff'])
    p.add_argument('--atlas', default='custom',
                   choices=['custom', 'HCPMMP1', 'Schaefer200', 'aparc'])
    p.add_argument('--subjects', nargs='+', default=None)
    p.add_argument('--roi-subset', nargs='+', default=None)
    p.add_argument('--seed', nargs=2, default=None, metavar='ROI',
                   help='seed edge for the example figure (default awfa-lh ifc-lh)')
    p.add_argument('--order', type=int, default=10)
    p.add_argument('--win-ms', type=float, default=40.0)
    p.add_argument('--target-fs', type=float, default=500.0)
    p.add_argument('--subject-jobs', type=int, default=1,
                   help='subjects in parallel (>1 => GC sequential per subject)')
    p.add_argument('--gc-jobs', type=int, default=8,
                   help='GC pair-parallelism when --subject-jobs 1')
    p.add_argument('--overwrite', action='store_true',
                   help='recompute subjects even if per-subject cache exists')
    p.add_argument('--plot-only', action='store_true',
                   help='rebuild figures from caches (aggregate + plot, no compute)')
    return p.parse_args()


def main():
    args = parse_args()
    paths = out_paths(args.atlas, args.task, args.stim_class)
    if args.plot_only:
        aggregate(paths, args.task)
    else:
        compute(args)
    figures(paths)


if __name__ == '__main__':
    main()
