#!/usr/bin/env python3
"""
Resolution analysis: can the 64-channel fsaverage inverse separate the ROIs?

Computes ROI-level cross-talk functions (CTF) and point-spread functions
(PSF) for the pipeline's OWN inverse operators (Hauk, Stenroos & Treder
2019/2022; Farahibozorg et al. 2018 for ROI CTFs), without ever forming
the full (n_src x n_src) resolution matrix:

    R = K P G       K  inverse kernel  (n_src x n_ch), from the pipeline's
                                        inverse operator / LCMV filter
                    P  average-reference projector
                    G  fixed-orientation leadfield (n_ch x n_src)

    CTF of ROI i  = w_i^T K_i P G       what ROI i's summary time course
                                        actually picks up, cortex-wide
    PSF of ROI j  = K P G_j 1/|V_j|     where a patch source in ROI j
                                        shows up after the inverse

ROI weights ``w_i`` mirror the pipeline: ``pc1`` = first PC of the ROI's
vertex time courses, i.e. the top eigenvector of K_i C_data K_i^T with
C_data the subject's sensor covariance (exactly what granger.reduce_roi_
first_pc computes on real data); ``mean_flip`` = sign-flipped mean
(MNE's mean_flip; mean without flips for LCMV max-power).

Per ROI (CSV ``roi_metrics``):
    se_own        fraction of CTF energy inside the ROI itself
    peak_in_roi   whether the CTF peak vertex lies inside the ROI
    peak_dist_mm  distance from the CTF peak to the nearest ROI vertex
    sd_mm         spatial dispersion: sqrt(sum d^2 ctf^2 / sum ctf^2), d =
                  distance of each vertex to the ROI (Molins 2008; Hauk 2019)
    rh_frac       fraction of CTF energy in the right hemisphere
Per ROI pair (CSV ``pair_metrics``):
    pattern_cos   |cos(CTF_i, CTF_j)| = expected zero-lag correlation of
                  the two ROI estimates under independent, spatially white
                  cortical sources of equal variance (the "fixed coupling"
                  that TRGC / leakage correction have to remove)
    ctf_xtalk     mean CTF_i^2 over ROI j's vertices / mean over ROI i's
                  own vertices (per-vertex source power in j that ROI i
                  reports, relative to its own)
    psf_xtalk     mean PSF_j^2 over ROI i's vertices / mean over ROI j's
                  own vertices (how much of a patch source in j lands in i)

The SENSOR pseudo-channels of the manuscript analysis (3-electrode means)
are analysed on the same footing: their "CTF" is the mean of the three
leadfield rows, so their pairwise pattern overlap is directly comparable
with the source ROIs' — this is the sensor-vs-source resolvability figure.

Runs per subject (noise/data covariance differ), then aggregates.  Light:
K and G are (n_src x 64); nothing larger is held.  Decoding-free, GC-free
(no estimator runs), so it can run locally or on the workstation.

Usage
-----
    conda activate mne
    # custom (GC) ROIs + the translation-arm ROIs, all four inverses, 20 subjects
    python exploratory/resolution_analysis.py --task overtProd --stim-class prodDiff \\
        --methods dSPM sLORETA eLORETA LCMV --atlases custom sensorEquiv --n-jobs 4

    # quick local look: 3 subjects, no brain maps
    python exploratory/resolution_analysis.py --task perception --stim-class percDiff \\
        --subjects EEGPROD4001 EEGPROD4003 EEGPROD4005 --methods dSPM eLORETA --n-jobs 3

    # re-plot from the per-subject caches
    python exploratory/resolution_analysis.py --task overtProd --stim-class prodDiff --plot-only

Output
------
    derivatives/source_estimation/RESOLUTION_ANALYSIS/{task}_{stim}/
        per_subject/{subj}_{method}_{atlas}.npz     CTF/PSF vectors + metrics
        roi_metrics.csv, pair_metrics.csv            long format, all subjects
        roi_metrics_group.csv, pair_metrics_group.csv  mean/SD over subjects
        pattern_overlap.png    |cos| matrices: ROI sets x methods (+ sensor)
        crosstalk.png          CTF cross-talk matrices (%)
        roi_metrics.png        se_own and sd_mm per ROI per method
        ctf_maps_{method}_{atlas}.png   group-mean |CTF| on the brain (--brain-maps)
        summary.txt            the headline numbers
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import argparse
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import mne
from mne.minimum_norm import apply_inverse
from mne.beamformer import apply_lcmv

from config import (SUBJECT_IDS, SPEECH_ROIS, VISUAL_CONTROL_ROI,
                    INVERSE_METHODS, MINIMUM_NORM_METHODS, LAMBDA2,
                    PROJECT_ROOT, resolve_noise_baseline)

OUT_ROOT = PROJECT_ROOT / 'derivatives' / 'source_estimation' / 'RESOLUTION_ANALYSIS'

# The manuscript's sensor-space pseudo-channels (exploratory/run_granger_sensor.py)
SENSOR_PSEUDOCHANNELS = {
    'Temporal':          ['FT7', 'T7', 'TP7'],
    'Inferior_Frontal':  ['F5', 'FC5', 'FC3'],
    'Superior_Frontal':  ['FCz', 'FC1', 'F1'],
    'Superior_Parietal': ['CPz', 'CP1', 'P1'],
}

# ROI sets per atlas: which labels to analyse (lh only for the custom atlas)
CUSTOM_GC_ROIS = ['awfa-lh', 'ifc-lh', 'pmc-lh', 'tpc-lh']
CUSTOM_CONTROL_ROIS = ['owfa-lh', 'vwfa-lh']

# Fixed categorical order for the inverse methods (never cycled)
METHOD_COLORS = {'dSPM': '#2a78d6', 'sLORETA': '#eb6834',
                 'eLORETA': '#1baf7a', 'LCMV': '#eda100', 'sensor': '#52514e'}


# ─────────────────────────────────────────────────────────────────────
# ROI sets
# ─────────────────────────────────────────────────────────────────────
def build_roi_sets(subjects_dir, atlases):
    """{atlas: {roi_name: Label}} for the requested atlases."""
    from forward_model import build_roi_labels
    sets = {}
    for atlas in atlases:
        if atlas == 'custom':
            d = build_roi_labels(subjects_dir, atlas='custom')
            keep = [r for r in CUSTOM_GC_ROIS + CUSTOM_CONTROL_ROIS if r in d]
            sets[atlas] = {r: d[r] for r in keep}
        else:
            comp = dict(SPEECH_ROIS[atlas])
            if 'Visual' not in comp and atlas in VISUAL_CONTROL_ROI:
                comp['Visual'] = VISUAL_CONTROL_ROI[atlas]
            sets[atlas] = build_roi_labels(subjects_dir, atlas=atlas,
                                           composite_rois=comp)
    return sets


def label_rows(label, vertices_lh, vertices_rh):
    """Row indices (into the stacked lh+rh source list) of a label's vertices."""
    if label.hemi == 'lh':
        idx = np.where(np.isin(vertices_lh, label.vertices))[0]
    else:
        idx = len(vertices_lh) + np.where(np.isin(vertices_rh, label.vertices))[0]
    return idx


# ─────────────────────────────────────────────────────────────────────
# Kernels
# ─────────────────────────────────────────────────────────────────────
def inverse_kernel(method, epochs, fwd, bl_tmin, bl_tmax):
    """Effective inverse kernel K (n_src x n_ch) of the pipeline's inverse for
    ``method``: the inverse applied to an identity 'evoked' (the same trick
    mne.minimum_norm.resolution_matrix uses), so whitening, projector, noise
    normalization (dSPM/sLORETA), eLORETA weights and LCMV orientation are all
    included exactly as in the production runs.  Returns (K, vertices)."""
    from inverse_pipelines import make_minimum_norm_inverse, make_lcmv_filters
    n_ch = len(epochs.ch_names)
    ev = mne.EvokedArray(np.eye(n_ch), epochs.info, tmin=0.0, nave=1,
                         verbose=False)
    if method in MINIMUM_NORM_METHODS:
        inv, _, _ = make_minimum_norm_inverse(epochs, fwd, bl_tmin, bl_tmax)
        stc = apply_inverse(ev, inv, lambda2=LAMBDA2, method=method,
                            pick_ori=None, verbose=False)
    elif method == 'LCMV':
        filters, _, _, _ = make_lcmv_filters(epochs, fwd, bl_tmin, bl_tmax)
        stc = apply_lcmv(ev, filters, verbose=False)
    else:
        raise ValueError(method)
    return np.asarray(stc.data, float), [v.copy() for v in stc.vertices]


def leadfield_fixed(fwd, info):
    """Fixed-orientation leadfield (n_ch x n_src) in ``info`` channel order,
    with the average-reference projector applied; plus vertices and
    coordinates (mm, both hemispheres stacked)."""
    fwd_f = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True,
                                         use_cps=True, verbose=False)
    fwd_f = mne.pick_channels_forward(fwd_f, include=info['ch_names'],
                                      ordered=True, verbose=False)
    G = np.asarray(fwd_f['sol']['data'], float)
    try:                                   # MNE >= 1.6
        from mne._fiff.proj import make_projector
    except ImportError:                    # older MNE
        from mne.io.proj import make_projector
    P, _, _ = make_projector(info['projs'], info['ch_names'])
    G = P @ G
    verts = [s['vertno'].copy() for s in fwd_f['src']]
    rr = np.vstack([s['rr'][s['vertno']] for s in fwd_f['src']]) * 1000.0
    return G, verts, rr


def sensor_covariance(epochs):
    """Sensor covariance over the whole ensemble (all trials, all times),
    the covariance behind the pipeline's fixed first-PC ROI filter."""
    x = epochs.get_data(copy=False)
    x = x.transpose(1, 0, 2).reshape(x.shape[1], -1)
    x = x - x.mean(axis=1, keepdims=True)
    return (x @ x.T) / x.shape[1]


def roi_weights(K_i, C_data, label, src, method, mode):
    """Aggregation weights over ROI i's vertices (unit norm)."""
    if mode == 'pc1':
        M = K_i @ C_data @ K_i.T
        w = np.linalg.eigh(M)[1][:, -1]
    elif mode == 'mean_flip':
        if method in MINIMUM_NORM_METHODS:
            w = mne.label.label_sign_flip(label, src).astype(float)
        else:
            w = np.ones(K_i.shape[0])
        w = w / np.linalg.norm(w)
    else:
        raise ValueError(mode)
    return w


# ─────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────
def roi_distance_mm(rr, roi_rows):
    """Distance (mm) of every source to the nearest vertex of the ROI."""
    from scipy.spatial import cKDTree
    return cKDTree(rr[roi_rows]).query(rr)[0]


def ctf_metrics(ctf, roi_rows, rr, n_lh):
    e = ctf ** 2
    tot = e.sum()
    d = roi_distance_mm(rr, roi_rows)
    peak = int(np.argmax(np.abs(ctf)))
    return dict(
        se_own=float(e[roi_rows].sum() / tot),
        peak_in_roi=bool(np.isin(peak, roi_rows)),
        peak_dist_mm=float(d[peak]),
        sd_mm=float(np.sqrt((d ** 2 * e).sum() / tot)),
        rh_frac=float(e[n_lh:].sum() / tot),
    )


def pair_metrics(ctfs, psfs, rows):
    """ctfs/psfs: {roi: vector}; rows: {roi: index array}.  Returns list of
    dict rows for every ordered pair i != j."""
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
            pb = psfs[b]
            ps = float(np.mean(pb[rows[a]] ** 2) / np.mean(pb[rows[b]] ** 2))
            out.append(dict(roi_i=a, roi_j=b, pattern_cos=cos,
                            ctf_xtalk=xt, psf_xtalk=ps))
    return out


# ─────────────────────────────────────────────────────────────────────
# Per-subject worker
# ─────────────────────────────────────────────────────────────────────
_G = {}


def _init(fwd, src, roi_sets, task, stim, methods, weights, out_dir):
    _G.update(fwd=fwd, src=src, roi_sets=roi_sets, task=task, stim=stim,
              methods=methods, weights=weights, out_dir=out_dir)


def _subject(subj):
    from data_loader import load_subject_epochs
    t0 = time.time()
    try:
        epochs, _, _ = load_subject_epochs(subj, _G['task'], _G['stim'])
    except FileNotFoundError as e:
        print(f'  {subj}: {e} — SKIP')
        return None
    bl_lo, bl_hi, _ = resolve_noise_baseline(
        _G['task'], float(epochs.times[0]), float(epochs.times[-1]))
    G, verts, rr = leadfield_fixed(_G['fwd'], epochs.info)
    n_lh = len(verts[0])
    C_data = sensor_covariance(epochs)
    ch = epochs.ch_names

    rows_by_atlas = {
        atlas: {r: label_rows(lab, verts[0], verts[1])
                for r, lab in labs.items()}
        for atlas, labs in _G['roi_sets'].items()}

    done = []
    # sensor pseudo-channels: "CTF" = mean leadfield row of the electrodes
    sens = {}
    for name, elec in SENSOR_PSEUDOCHANNELS.items():
        idx = [ch.index(e) for e in elec if e in ch]
        if len(idx) < len(elec):
            print(f'  {subj}: {name} missing electrodes '
                  f'{set(elec) - set(ch)}; using {len(idx)}')
        if idx:
            sens[name] = G[idx].mean(axis=0)
    f = _G['out_dir'] / 'per_subject' / f'{subj}_sensor.npz'
    np.savez_compressed(f, names=np.array(list(sens)),
                        ctf=np.stack(list(sens.values())), n_lh=n_lh)
    done.append(f)

    for method in _G['methods']:
        K, kverts = inverse_kernel(method, epochs, _G['fwd'], bl_lo, bl_hi)
        assert all(np.array_equal(a, b) for a, b in zip(kverts, verts)), \
            'inverse and leadfield source spaces differ'
        KG_cache = None
        for atlas, labs in _G['roi_sets'].items():
            rows = rows_by_atlas[atlas]
            save = dict(roi_names=np.array(list(labs)), n_lh=n_lh,
                        roi_rows=np.array([rows[r] for r in labs], dtype=object))
            roi_rec, pair_rec = [], []
            # PSF per ROI (weights-independent): K @ mean leadfield column
            psfs = {r: K @ (G[:, rows[r]].mean(axis=1)) for r in labs}
            save['psf'] = np.stack([psfs[r] for r in labs])
            for wmode in _G['weights']:
                ctfs = {}
                for r, lab in labs.items():
                    K_i = K[rows[r]]
                    w = roi_weights(K_i, C_data, lab, _G['src'], method, wmode)
                    ctfs[r] = (w @ K_i) @ G
                    m = ctf_metrics(ctfs[r], rows[r], rr, n_lh)
                    roi_rec.append(dict(subject=subj, method=method,
                                        atlas=atlas, weights=wmode, roi=r,
                                        n_vertices=len(rows[r]), **m))
                for d in pair_metrics(ctfs, psfs, rows):
                    pair_rec.append(dict(subject=subj, method=method,
                                         atlas=atlas, weights=wmode, **d))
                save[f'ctf_{wmode}'] = np.stack([ctfs[r] for r in labs])
            f = _G['out_dir'] / 'per_subject' / f'{subj}_{method}_{atlas}.npz'
            np.savez_compressed(f, **save)
            pd.DataFrame(roi_rec).to_csv(f.with_suffix('.roi.csv'), index=False)
            pd.DataFrame(pair_rec).to_csv(f.with_suffix('.pair.csv'), index=False)
            done.append(f)
        del K
    print(f'  {subj}: done in {(time.time() - t0) / 60:.1f} min', flush=True)
    return subj


# ─────────────────────────────────────────────────────────────────────
# Aggregation + figures
# ─────────────────────────────────────────────────────────────────────
def aggregate(out_dir):
    ps = out_dir / 'per_subject'
    roi = pd.concat([pd.read_csv(f) for f in sorted(ps.glob('*.roi.csv'))],
                    ignore_index=True)
    pair = pd.concat([pd.read_csv(f) for f in sorted(ps.glob('*.pair.csv'))],
                     ignore_index=True)
    roi.to_csv(out_dir / 'roi_metrics.csv', index=False)
    pair.to_csv(out_dir / 'pair_metrics.csv', index=False)
    keys = ['method', 'atlas', 'weights', 'roi']
    roi_g = roi.groupby(keys).agg(
        n=('subject', 'nunique'), n_vertices=('n_vertices', 'first'),
        se_own=('se_own', 'mean'), se_own_sd=('se_own', 'std'),
        peak_in_roi=('peak_in_roi', 'mean'),
        peak_dist_mm=('peak_dist_mm', 'mean'),
        sd_mm=('sd_mm', 'mean'), sd_mm_sd=('sd_mm', 'std'),
        rh_frac=('rh_frac', 'mean')).reset_index()
    keys = ['method', 'atlas', 'weights', 'roi_i', 'roi_j']
    pair_g = pair.groupby(keys).agg(
        n=('subject', 'nunique'),
        pattern_cos=('pattern_cos', 'mean'), pattern_cos_sd=('pattern_cos', 'std'),
        ctf_xtalk=('ctf_xtalk', 'mean'), psf_xtalk=('psf_xtalk', 'mean')
    ).reset_index()
    roi_g.to_csv(out_dir / 'roi_metrics_group.csv', index=False)
    pair_g.to_csv(out_dir / 'pair_metrics_group.csv', index=False)
    return roi, pair, roi_g, pair_g


def sensor_pairs(out_dir):
    """Group-mean |cos| matrix of the sensor pseudo-channel leadfield rows."""
    files = sorted((out_dir / 'per_subject').glob('*_sensor.npz'))
    if not files:
        return None, None
    mats, names = [], None
    for f in files:
        d = np.load(f, allow_pickle=True)
        c = d['ctf']
        names = list(d['names'])
        n = c / np.linalg.norm(c, axis=1, keepdims=True)
        mats.append(np.abs(n @ n.T))
    return names, np.mean(mats, axis=0)


def _matrix(df, names, col):
    M = np.full((len(names), len(names)), np.nan)
    for _, r in df.iterrows():
        M[names.index(r.roi_i), names.index(r.roi_j)] = r[col]
    return M


def _heat(ax, M, names, vmax, fmt, cmap, title):
    im = ax.imshow(M, vmin=0, vmax=vmax, cmap=cmap)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    short = [n.replace('-lh', '').replace('_', ' ').replace('Superior', 'Sup.')
             .replace('Inferior', 'Inf.') for n in names]
    ax.set_xticklabels(short, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(short, fontsize=8)
    ax.set_title(title, fontsize=10)
    for i in range(len(names)):
        for j in range(len(names)):
            if i == j or not np.isfinite(M[i, j]):
                continue
            v = M[i, j]
            ax.text(j, i, fmt(v), ha='center', va='center', fontsize=7,
                    color='white' if v > 0.6 * vmax else '#0b0b0b')
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    return im


def plot_matrices(out_dir, pair_g, methods, atlases, weights, task, stim):
    sens_names, sens_M = sensor_pairs(out_dir)
    for col, vmax, fmt, cmap, fname, label in [
        ('pattern_cos', 1.0, lambda v: f'{v:.2f}', 'Blues', 'pattern_overlap',
         '|cos(CTF_i, CTF_j)| = expected zero-lag correlation under white sources'),
        ('ctf_xtalk', 1.0, lambda v: f'{100 * v:.0f}', 'Oranges', 'crosstalk',
         'CTF cross-talk: source power in ROI j reported by ROI i, % of its own'),
    ]:
        n_rows = len(atlases) + (1 if sens_M is not None else 0)
        n_cols = len(methods)
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(3.6 * n_cols + 1.2, 3.2 * n_rows + 0.8),
                                 squeeze=False)
        im = None
        for ai, atlas in enumerate(atlases):
            for mi, method in enumerate(methods):
                d = pair_g[(pair_g.method == method) & (pair_g.atlas == atlas)
                           & (pair_g.weights == weights)]
                ax = axes[ai, mi]
                if d.empty:
                    ax.set_axis_off()
                    continue
                names = list(dict.fromkeys(d.roi_i))
                M = _matrix(d, names, col)
                im = _heat(ax, M, names, vmax, fmt, cmap,
                           f'{atlas} · {method}' if ai == 0 else atlas)
        if sens_M is not None:
            ax = axes[-1, 0]
            im = _heat(ax, np.minimum(sens_M, vmax), sens_names, vmax, fmt, cmap,
                       'sensor pseudo-channels (3-electrode means)')
            for mi in range(1, n_cols):
                axes[-1, mi].set_axis_off()
        fig.suptitle(f'{label}\n{task} / {stim}, group mean over subjects, '
                     f'ROI weights = {weights}', fontsize=11)
        if im is not None:
            fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.5, pad=0.02)
        fig.savefig(out_dir / f'{fname}.png', dpi=170, bbox_inches='tight')
        plt.close(fig)


def plot_roi_metrics(out_dir, roi_g, methods, atlases, weights, task, stim):
    fig, axes = plt.subplots(2, len(atlases), figsize=(5.2 * len(atlases), 6.4),
                             squeeze=False)
    for ai, atlas in enumerate(atlases):
        d = roi_g[(roi_g.atlas == atlas) & (roi_g.weights == weights)]
        if d.empty:
            axes[0, ai].set_axis_off()
            axes[1, ai].set_axis_off()
            continue
        names = list(dict.fromkeys(d.roi))
        x = np.arange(len(names))
        wbar = 0.8 / len(methods)
        for row, (col, sdcol, ylab) in enumerate([
                ('se_own', 'se_own_sd', 'CTF energy inside own ROI'),
                ('sd_mm', 'sd_mm_sd', 'CTF spatial dispersion (mm)')]):
            ax = axes[row, ai]
            for mi, method in enumerate(methods):
                dm = d[d.method == method].set_index('roi').reindex(names)
                ax.bar(x + (mi - (len(methods) - 1) / 2) * wbar, dm[col],
                       width=wbar * 0.9, yerr=dm[sdcol], capsize=2,
                       color=METHOD_COLORS.get(method, '#888'), label=method,
                       error_kw=dict(lw=0.8, color='#52514e'))
            ax.set_xticks(x)
            ax.set_xticklabels([n.replace('-lh', '').replace('_', ' ')
                                for n in names], rotation=30, ha='right',
                               fontsize=8)
            ax.set_ylabel(ylab, fontsize=9)
            if row == 0:
                ax.set_title(atlas, fontsize=10)
                ax.set_ylim(0, 1)
                ax.axhline(1.0 / max(len(names), 1), color='#c3c2b7', lw=0.8,
                           ls='--')
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(axis='y', color='#eeeeee', lw=0.6)
            ax.set_axisbelow(True)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(f'ROI cross-talk metrics — {task} / {stim}, mean ± SD over '
                 f'subjects, weights = {weights}', fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / 'roi_metrics.png', dpi=170)
    plt.close(fig)


def plot_brain_maps(out_dir, src, methods, atlases, weights):
    """Group-mean |CTF| of every ROI on the fsaverage surface (lateral lh)."""
    try:
        from visualize_rois import _load_surface, _render_brain_views
    except Exception as e:  # pragma: no cover - rendering is optional
        print(f'  brain maps skipped (visualize_rois import failed: {e})')
        return
    from mne.datasets import fetch_fsaverage
    subjects_dir = fetch_fsaverage(verbose=False).parent
    coords, faces = _load_surface(subjects_dir, 'lh', surf='pial', smooth=0.25)
    ps = out_dir / 'per_subject'
    for method in methods:
        for atlas in atlases:
            files = sorted(ps.glob(f'*_{method}_{atlas}.npz'))
            if not files:
                continue
            acc, names, n_lh = None, None, None
            for f in files:
                d = np.load(f, allow_pickle=True)
                c = np.abs(d[f'ctf_{weights}'])
                c = c / c.max(axis=1, keepdims=True)
                acc = c if acc is None else acc + c
                names = list(d['roi_names'])
                n_lh = int(d['n_lh'])
            acc /= len(files)
            vert_lh = src[0]['vertno']
            n = len(names)
            fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.4))
            axes = np.atleast_1d(axes)
            for k, name in enumerate(names):
                stc = mne.SourceEstimate(
                    acc[k][:n_lh, None], vertices=[vert_lh, np.array([], int)],
                    tmin=0, tstep=1, subject='fsaverage')
                morph = mne.compute_source_morph(
                    stc, 'fsaverage', 'fsaverage', spacing=None, smooth=5,
                    subjects_dir=subjects_dir, verbose=False)
                full = morph.apply(stc).data[:, 0][:len(coords)]   # lh only
                scal = np.where(full > 0.05, full, np.nan).astype(float)
                img = _render_brain_views(coords, faces, ['lateral'], 'lh',
                                          size=(900, 700), vertex_scalars=scal,
                                          cmap='hot', clim=(0, 1))['lateral']
                axes[k].imshow(img)
                axes[k].set_axis_off()
                axes[k].set_title(name.replace('-lh', ''), fontsize=10)
            fig.suptitle(f'group-mean |CTF| (each ROI scaled to its max) — '
                         f'{method} / {atlas} / {weights}', fontsize=11)
            fig.tight_layout()
            fig.savefig(out_dir / f'ctf_maps_{method}_{atlas}.png', dpi=150)
            plt.close(fig)
            print(f'  brain maps: {method} / {atlas}')


def write_summary(out_dir, roi_g, pair_g, methods, atlases, weights, task, stim):
    sens_names, sens_M = sensor_pairs(out_dir)
    lines = [f'Resolution analysis — {task} / {stim}  (weights = {weights})', '']
    if sens_M is not None:
        off = sens_M[~np.eye(len(sens_names), dtype=bool)]
        lines.append(f'sensor pseudo-channels: mean |cos| {off.mean():.2f}, '
                     f'max {off.max():.2f}')
    lines.append('')
    lines.append(f'{"atlas":12s} {"method":8s} {"mean|cos|":>9s} {"max|cos|":>8s}'
                 f' {"max pair":>28s} {"mean se_own":>11s} {"mean sd_mm":>10s}'
                 f' {"peak in ROI":>11s}')
    for atlas in atlases:
        for method in methods:
            p = pair_g[(pair_g.method == method) & (pair_g.atlas == atlas)
                       & (pair_g.weights == weights)]
            r = roi_g[(roi_g.method == method) & (roi_g.atlas == atlas)
                      & (roi_g.weights == weights)]
            if p.empty:
                continue
            top = p.loc[p.pattern_cos.idxmax()]
            lines.append(
                f'{atlas:12s} {method:8s} {p.pattern_cos.mean():9.2f} '
                f'{p.pattern_cos.max():8.2f} '
                f'{(top.roi_i + "<>" + top.roi_j).replace("-lh", ""):>28s} '
                f'{r.se_own.mean():11.2f} {r.sd_mm.mean():10.1f} '
                f'{100 * r.peak_in_roi.mean():10.0f}%')
    lines += ['', 'Reading guide:',
              '  |cos| is the zero-lag correlation two ROI estimates would show',
              '  with independent white sources; > 0.5 means the pair largely',
              '  reports the same cortex.  se_own is the share of an ROI\'s CTF',
              '  energy that comes from its own vertices (chance = ROI size /',
              '  cortex).  sd_mm is how far the CTF spreads from the ROI.']
    text = '\n'.join(lines)
    print('\n' + text)
    (out_dir / 'summary.txt').write_text(text + '\n')


# ─────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--task', required=True, choices=['perception', 'overtProd'])
    p.add_argument('--stim-class', default='prodDiff',
                   choices=['prodDiff', 'percDiff'])
    p.add_argument('--methods', nargs='+', default=INVERSE_METHODS,
                   choices=INVERSE_METHODS)
    p.add_argument('--atlases', nargs='+', default=['custom', 'sensorEquiv'],
                   choices=['custom', 'sensorEquiv', 'HCPMMP1', 'aparc',
                            'Schaefer200'])
    p.add_argument('--weights', nargs='+', default=['pc1', 'mean_flip'],
                   choices=['pc1', 'mean_flip'])
    p.add_argument('--plot-weights', default='pc1', choices=['pc1', 'mean_flip'],
                   help='which weights the figures/summary show (default pc1 '
                        '= the pipeline\'s fixed first-PC virtual channel)')
    p.add_argument('--subjects', nargs='+', default=None)
    p.add_argument('--n-jobs', type=int, default=2)
    p.add_argument('--brain-maps', action='store_true', default=False,
                   help='also render group-mean |CTF| brain maps (PyVista)')
    p.add_argument('--plot-only', action='store_true', default=False,
                   help='skip computation; aggregate + plot the per-subject caches')
    p.add_argument('--out-dir', default=None)
    return p.parse_args()


def main():
    args = parse_args()
    subjects = args.subjects or SUBJECT_IDS
    out_dir = Path(args.out_dir) if args.out_dir else \
        OUT_ROOT / f'{args.task}_{args.stim_class}'
    (out_dir / 'per_subject').mkdir(parents=True, exist_ok=True)
    print(f'Resolution analysis: {args.task}/{args.stim_class}')
    print(f'  methods {args.methods}  atlases {args.atlases}  weights {args.weights}')
    print(f'  subjects {len(subjects)}  n_jobs {args.n_jobs}  out {out_dir}\n')

    from forward_model import setup_fsaverage, make_forward
    subjects_dir, _, src, bem = setup_fsaverage()
    roi_sets = build_roi_sets(subjects_dir, args.atlases)
    for atlas, labs in roi_sets.items():
        print(f'  {atlas}: {list(labs)}')

    if not args.plot_only:
        from data_loader import load_subject_epochs
        todo = [s for s in subjects if not all(
            (out_dir / 'per_subject' / f'{s}_{m}_{a}.npz').exists()
            for m in args.methods for a in args.atlases)]
        print(f'\n  {len(todo)} subject(s) to compute '
              f'({len(subjects) - len(todo)} cached)')
        if todo:
            ep0, _, _ = load_subject_epochs(todo[0], args.task, args.stim_class)
            fwd = make_forward(ep0.info, src, bem)
            del ep0
            with Pool(args.n_jobs, initializer=_init,
                      initargs=(fwd, src, roi_sets, args.task, args.stim_class,
                                args.methods, args.weights, out_dir)) as pool:
                pool.map(_subject, todo)

    roi, pair, roi_g, pair_g = aggregate(out_dir)
    methods = [m for m in args.methods if m in set(pair_g.method)]
    plot_matrices(out_dir, pair_g, methods, args.atlases, args.plot_weights,
                  args.task, args.stim_class)
    plot_roi_metrics(out_dir, roi_g, methods, args.atlases, args.plot_weights,
                     args.task, args.stim_class)
    if args.brain_maps:
        plot_brain_maps(out_dir, src, methods, args.atlases, args.plot_weights)
    write_summary(out_dir, roi_g, pair_g, methods, args.atlases,
                  args.plot_weights, args.task, args.stim_class)
    print(f'\nWrote {out_dir}')


if __name__ == '__main__':
    main()
