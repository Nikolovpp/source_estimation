#!/usr/bin/env python3
"""
A/B check: does downsampling ORDER change LCMV source-space Granger causality?

Two pipelines, both ending in 500 Hz ROI time courses and identical GC:

  Pipeline A  (downsample BEFORE source estimation)
      load the 500 Hz continuous-resampled epochs (fs_500) -> LCMV ->
      ROI time courses.  No per-epoch resample happens (data already 500 Hz).

  Pipeline B  (source estimation THEN downsample -- the current run_granger path)
      load the 2048 Hz epochs (fs_2000) -> LCMV -> ROI time courses at 2048 Hz
      -> resample_channels() to 500 Hz per epoch (now edge-padded).

Everything else is shared: the same fsaverage forward model, ROI labels, LCMV
settings (pick_ori='max-power', unit-noise-gain, reg=0.05, shrunk noise cov),
and GC parameters.  The ONLY difference is the sampling rate at which the
beamformer DATA covariance is estimated -- exactly the quantity the literature
says makes beamformers (unlike linear dSPM) order-dependent.

Because LCMV's max-power orientation is chosen from the data covariance, a
covariance that differs between 2048 and 500 Hz can flip a source's orientation,
which would show up as a low (or sign-flipped) correlation between the A and B
ROI time courses and as a divergence in the directed GC.

Outputs (under derivatives/source_estimation/GC_downsample_order_check/):
  - roi_tc_comparison.csv   per (subject, ROI): A-vs-B time-course agreement
  - gc_comparison.csv       per (subject, pair, direction, band): A-vs-B GC agreement
  - <subject>_roi_tc.png    trial-mean A vs B ROI time courses
  - <subject>_gc_lowbeta.png overlaid A vs B GC for every directed edge (low_beta)

Usage
-----
    conda activate mne
    python compare_downsample_order.py --task overtProd --stim-class prodDiff \\
        --atlas HCPMMP1 --roi-subset Temporal IFG vSMC DLPFC \\
        --subjects EEGPROD4001 EEGPROD4002 EEGPROD4003
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
import pandas as pd

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
# methods-paper script lives one level down; add the source_estimation root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import (SUBJECT_IDS, SPEECH_ROIS, BASELINE_WINDOWS,
                    DECODE_OUTPUT_ROOT)
from data_loader import load_subject_epochs
from forward_model import setup_fsaverage, make_forward, build_roi_labels
from inverse_pipelines import run_lcmv_lowram
from decoding_io import filter_roi_dict
from run_granger import resample_channels, compute_subject_gc
from granger import DEFAULT_BANDS

OUT_ROOT = DECODE_OUTPUT_ROOT.parent / 'GC_downsample_order_check'


# ─────────────────────────────────────────────────────────────────────
# Per-pipeline ROI time courses (shared LCMV, only the sfreq differs)
# ─────────────────────────────────────────────────────────────────────
def _lcmv_roi_tcs(subj, task, stim_class, fs, fwd, src, roi_labels, roi_names, bl):
    """LCMV -> {roi_name: (n_epochs, n_times)} at the file's native rate."""
    epochs, _y, sf = load_subject_epochs(subj, task, stim_class, fs=fs)
    X_roi, times = run_lcmv_lowram(
        epochs, fwd, bl[0], bl[1], roi_labels, src, feature_mode='pca_flip')
    # pca_flip -> X_roi is (n_epochs, n_rois, n_times)
    roi = {roi_names[i]: X_roi[:, i, :] for i in range(len(roi_names))}
    return roi, np.asarray(times), float(sf)


def pipeline_A(subj, task, stim_class, fwd, src, roi_labels, roi_names, bl):
    """Downsample-before: source-estimate directly on the 500 Hz data."""
    return _lcmv_roi_tcs(subj, task, stim_class, 500, fwd, src,
                         roi_labels, roi_names, bl)


def pipeline_B(subj, task, stim_class, fwd, src, roi_labels, roi_names, bl,
               target_fs=500.0):
    """Source-then-downsample: LCMV at 2048 Hz, resample ROI TCs to 500 Hz."""
    roi2k, times2k, sf = _lcmv_roi_tcs(subj, task, stim_class, 2000, fwd, src,
                                       roi_labels, roi_names, bl)
    V = np.stack([roi2k[n] for n in roi_names], axis=0)   # (n_roi, n_ep, n_t)
    V, fs = resample_channels(V, sf, target_fs)           # edge-padded resample
    new_times = times2k[0] + np.arange(V.shape[2]) / fs
    roi = {roi_names[i]: V[i] for i in range(len(roi_names))}
    return roi, new_times, fs


# ─────────────────────────────────────────────────────────────────────
# Alignment + comparison metrics
# ─────────────────────────────────────────────────────────────────────
def align_time(times_a, times_b):
    """Common index ranges so times_a[ia0:ia0+n] ≈ times_b[ib0:ib0+n]."""
    t0 = max(times_a[0], times_b[0])
    ia0 = int(np.argmin(np.abs(times_a - t0)))
    ib0 = int(np.argmin(np.abs(times_b - t0)))
    n = min(times_a.size - ia0, times_b.size - ib0)
    off = np.max(np.abs(times_a[ia0:ia0 + n] - times_b[ib0:ib0 + n]))
    return ia0, ib0, n, off


def compare_tc(a, b):
    """Per-epoch A-vs-B agreement for one ROI. a, b: (n_epochs, n_times)."""
    rs = []
    n_ep = min(a.shape[0], b.shape[0])   # guard trial-count mismatch
    for ep in range(n_ep):
        x, y = a[ep], b[ep]
        if x.std() == 0 or y.std() == 0:
            continue
        rs.append(np.corrcoef(x, y)[0, 1])
    rs = np.asarray(rs)
    if rs.size == 0:
        return dict(mean_r=np.nan, mean_absr=np.nan, frac_signflip=np.nan,
                    frac_diverge=np.nan, n_epochs=0)
    return dict(
        mean_r=float(rs.mean()),
        mean_absr=float(np.abs(rs).mean()),
        frac_signflip=float((rs < 0).mean()),        # |r|~1 but negative
        frac_diverge=float((np.abs(rs) < 0.9).mean()),  # genuinely different TC
        n_epochs=int(rs.size),
    )


def perturb_matched(roi, r_by_name, seed=0):
    """Add white noise to each ROI TC so its per-epoch correlation to the
    original matches ``r_by_name[name]`` — the same magnitude of difference
    that pipeline B has from A.  This is the GC-sensitivity control: if this
    matched *white* perturbation scrambles GC as much as the A-vs-B
    difference does, the divergence is generic GC fragility rather than a
    structured effect of the downsampling order.
    """
    rng = np.random.RandomState(seed)
    out = {}
    for name, x in roi.items():                       # x: (n_epochs, n_times)
        r0 = float(np.clip(r_by_name.get(name, 0.996), 0.5, 0.99999))
        sd = x.std(axis=1, keepdims=True)             # per-epoch signal std
        nstd = sd * np.sqrt(1.0 / r0 ** 2 - 1.0)      # noise std for target r0
        out[name] = x + rng.standard_normal(x.shape) * nstd
    return out


def compare_gc(resA, resB, roi_names, band_names):
    """Per (pair, direction, band) A-vs-B GC-time-course agreement."""
    rows = []
    pi_i, pi_j = resA['pair_i'], resA['pair_j']
    for b in band_names:
        for key, direction in [('fxy', 'i->j'), ('fyx', 'j->i')]:
            A = resA[key][b]
            B = resB[key][b]
            nw = min(A.shape[1], B.shape[1])
            for p in range(A.shape[0]):
                a = A[p, :nw]
                bb = B[p, :nw]
                i, j = int(pi_i[p]), int(pi_j[p])
                src_name, tgt_name = ((roi_names[i], roi_names[j])
                                      if key == 'fxy' else
                                      (roi_names[j], roi_names[i]))
                denom = a.std() + 1e-12
                r = (np.corrcoef(a, bb)[0, 1]
                     if a.std() > 0 and bb.std() > 0 else np.nan)
                rows.append(dict(
                    src=src_name, tgt=tgt_name, band=b,
                    gc_r=r,
                    nrmse=float(np.sqrt(np.mean((a - bb) ** 2)) / denom),
                    mean_gc_A=float(a.mean()), mean_gc_B=float(bb.mean()),
                    mean_abs_diff=float(np.mean(np.abs(a - bb))),
                ))
    return rows


# ─────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────
def plot_roi_tc(roiA, roiB, times, roi_names, subj, out_path):
    n = len(roi_names)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3 * nrow),
                             squeeze=False)
    for k, name in enumerate(roi_names):
        ax = axes[k // ncol][k % ncol]
        ma = roiA[name].mean(axis=0)
        mb = roiB[name].mean(axis=0)
        # sign-match B to A for display (GC is sign-invariant anyway)
        if np.corrcoef(ma, mb)[0, 1] < 0:
            mb = -mb
        ax.plot(times * 1000, ma, lw=1.4, label='A: 500 Hz source', color='#2166ac')
        ax.plot(times * 1000, mb, lw=1.1, label='B: 2048→500', color='#b2182b', alpha=0.8)
        ax.axvline(0, color='k', lw=0.6, alpha=0.4)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel('time (ms)')
    axes[0][0].legend(fontsize=8, loc='upper left')
    fig.suptitle(f'{subj}: LCMV ROI time courses, pipeline A vs B '
                 f'(trial mean, B sign-matched)', fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


def plot_gc(resA, resB, roi_names, subj, band, out_path):
    pi_i, pi_j = resA['pair_i'], resA['pair_j']
    wmA, wmB = resA['window_ms'], resB['window_ms']
    n = pi_i.size
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3 * nrow),
                             squeeze=False)
    for p in range(n):
        ax = axes[p // ncol][p % ncol]
        i, j = int(pi_i[p]), int(pi_j[p])
        ax.plot(wmA, resA['fxy'][band][p], lw=1.4, color='#2166ac',
                label='A i→j')
        ax.plot(wmB, resB['fxy'][band][p], lw=1.0, color='#66a3d2', ls='--',
                label='B i→j')
        ax.plot(wmA, resA['fyx'][band][p], lw=1.4, color='#b2182b',
                label='A j→i')
        ax.plot(wmB, resB['fyx'][band][p], lw=1.0, color='#e08a86', ls='--',
                label='B j→i')
        ax.set_title(f'{roi_names[i]} ↔ {roi_names[j]}', fontsize=9)
        ax.set_xlabel('window start (ms)')
    axes[0][0].legend(fontsize=7, loc='upper left')
    fig.suptitle(f'{subj}: GC ({band}) pipeline A (solid) vs B (dashed)',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description='A/B downsampling-order check for LCMV source GC')
    p.add_argument('--task', default='overtProd', choices=['perception', 'overtProd'])
    p.add_argument('--stim-class', default='prodDiff', choices=['prodDiff', 'percDiff'])
    p.add_argument('--atlas', default='HCPMMP1',
                   choices=['aparc', 'HCPMMP1', 'Schaefer200'])
    p.add_argument('--roi-subset', nargs='+', default=None, metavar='ROI',
                   help='Subset of ROI names (default: all speech ROIs for the atlas)')
    p.add_argument('--subjects', nargs='+', default=None,
                   help='Subjects to test (default: first 3)')
    p.add_argument('--order', type=int, default=10)
    p.add_argument('--win-ms', type=float, default=40.0)
    p.add_argument('--target-fs', type=float, default=500.0)
    p.add_argument('--tmin', type=float, default=None)
    p.add_argument('--tmax', type=float, default=None)
    p.add_argument('--n-jobs', type=int, default=8)
    p.add_argument('--no-figs', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    subjects = args.subjects if args.subjects else SUBJECT_IDS[:3]
    bl = BASELINE_WINDOWS[args.task]
    freqs = np.arange(1, 31)
    band_names = list(DEFAULT_BANDS)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print('Downsampling-order A/B check (LCMV source-space GC)')
    print(f'  Task/class: {args.task} / {args.stim_class}   atlas={args.atlas}')
    print(f'  Subjects:   {subjects}')
    print(f'  GC:         order {args.order}, win {args.win_ms} ms @ {args.target_fs} Hz')
    print(f'  Out:        {OUT_ROOT}\n')

    print('Setting up fsaverage source space + ROI labels...')
    subjects_dir, _fs_dir, src, bem = setup_fsaverage()
    if args.atlas in SPEECH_ROIS:
        roi_dict = build_roi_labels(subjects_dir, atlas=args.atlas,
                                    composite_rois=SPEECH_ROIS[args.atlas])
    else:
        roi_dict = build_roi_labels(subjects_dir, atlas=args.atlas)
    if args.roi_subset:
        roi_dict = filter_roi_dict(roi_dict, args.roi_subset, args.atlas)
    roi_names = list(roi_dict.keys())
    roi_labels = list(roi_dict.values())
    print(f'  ROIs ({len(roi_names)}): {roi_names}\n')

    # One forward model (2048 Hz info; leadfield is sfreq-independent).
    print('Building forward model...')
    info_epochs, _, _ = load_subject_epochs(subjects[0], args.task,
                                            args.stim_class, fs=2000)
    fwd = make_forward(info_epochs.info, src, bem)
    del info_epochs, bem

    tc_rows, gc_rows = [], []
    for subj in subjects:
        t0 = time.time()
        print(f'\n{"="*60}\n{subj}\n{"="*60}')
        try:
            roiA, tA, fsA = pipeline_A(subj, args.task, args.stim_class,
                                       fwd, src, roi_labels, roi_names, bl)
            roiB, tB, fsB = pipeline_B(subj, args.task, args.stim_class,
                                       fwd, src, roi_labels, roi_names, bl,
                                       target_fs=args.target_fs)

            ia0, ib0, n, off = align_time(tA, tB)
            # Match trial counts: the two preprocessing rates can reject
            # slightly different trials.  Truncate to the common count and
            # flag mismatches (per-trial TC agreement is unreliable then).
            nA = next(iter(roiA.values())).shape[0]
            nB = next(iter(roiB.values())).shape[0]
            n_ep = min(nA, nB)
            trials_matched = (nA == nB)
            print(f'  aligned {n} samples, max time offset {off*1000:.3f} ms '
                  f'(fsA={fsA:.1f}, fsB={fsB:.1f}); trials A={nA} B={nB}'
                  f'{"" if trials_matched else "  MISMATCH"}')
            roiA = {k: v[:n_ep, ia0:ia0 + n] for k, v in roiA.items()}
            roiB = {k: v[:n_ep, ib0:ib0 + n] for k, v in roiB.items()}
            t_common = tA[ia0:ia0 + n]

            absr_by_name = {}
            for name in roi_names:
                m = compare_tc(roiA[name], roiB[name])
                m.update(subject=subj, roi=name, trials_matched=trials_matched)
                tc_rows.append(m)
                absr_by_name[name] = m['mean_absr']
                print(f'    {name:18s} |r|={m["mean_absr"]:.3f}  '
                      f'signflip={m["frac_signflip"]:.2f}  diverge={m["frac_diverge"]:.2f}')

            # GC on both pipelines (already 500 Hz -> resample is a no-op inside)
            gc_kw = dict(order=args.order, win_ms=args.win_ms,
                         target_fs=args.target_fs, freqs=freqs,
                         tmin=args.tmin, tmax=args.tmax, n_jobs=args.n_jobs)
            resA = compute_subject_gc(roiA, t_common, fsA, **gc_kw)
            resB = compute_subject_gc(roiB, t_common, fsB, **gc_kw)
            # Control: GC on A perturbed by white noise matched to the A-vs-B
            # TC difference, to separate structured order effects from generic
            # GC fragility to any sub-percent perturbation.
            roiA_noisy = perturb_matched(roiA, absr_by_name, seed=0)
            resAn = compute_subject_gc(roiA_noisy, t_common, fsA, **gc_kw)

            ab = compare_gc(resA, resB, roi_names, band_names)
            noise = compare_gc(resA, resAn, roi_names, band_names)
            for row, nrow in zip(ab, noise):
                row['subject'] = subj
                row['trials_matched'] = trials_matched
                row['gc_r_noise'] = nrow['gc_r']
                row['nrmse_noise'] = nrow['nrmse']
                gc_rows.append(row)

            if not args.no_figs:
                sdir = OUT_ROOT / subj
                sdir.mkdir(parents=True, exist_ok=True)
                plot_roi_tc(roiA, roiB, t_common, roi_names, subj,
                            sdir / f'{subj}_roi_tc.png')
                band_fig = 'low_beta' if 'low_beta' in band_names else band_names[0]
                plot_gc(resA, resB, roi_names, subj, band_fig,
                        sdir / f'{subj}_gc_{band_fig}.png')

            gcdf_subj = pd.DataFrame([r for r in gc_rows if r['subject'] == subj])
            print(f'  GC A-vs-B median r={gcdf_subj["gc_r"].median():.3f} '
                  f'(matched-noise baseline r={gcdf_subj["gc_r_noise"].median():.3f}), '
                  f'median nRMSE={gcdf_subj["nrmse"].median():.3f}  '
                  f'({(time.time()-t0)/60:.1f} min)')
        except Exception as e:
            print(f'  SKIP {subj}: {type(e).__name__}: {e}')
            continue

    # Write tables + overall summary
    tc = pd.DataFrame(tc_rows)
    gc = pd.DataFrame(gc_rows)
    if tc.empty:
        print('\nNo subjects processed (no fs_500 data?).')
        return
    tc.to_csv(OUT_ROOT / 'roi_tc_comparison.csv', index=False)
    gc.to_csv(OUT_ROOT / 'gc_comparison.csv', index=False)

    print(f'\n{"="*60}\nSUMMARY across {tc["subject"].nunique()} subject(s)\n{"="*60}')
    if not tc.empty:
        print(f'ROI time courses:  median |r|={tc["mean_absr"].median():.3f}, '
              f'mean frac sign-flip={tc["frac_signflip"].mean():.3f}, '
              f'mean frac diverge (|r|<0.9)={tc["frac_diverge"].mean():.3f}')
    if not gc.empty:
        r_ab = gc["gc_r"].median()
        r_noise = gc["gc_r_noise"].median()
        print(f'GC A-vs-B         (all edges/bands): median r={r_ab:.3f}, '
              f'median nRMSE={gc["nrmse"].median():.3f}')
        print(f'GC matched-noise baseline          : median r={r_noise:.3f}, '
              f'median nRMSE={gc["nrmse_noise"].median():.3f}')
        verdict = ('order effect ≈ generic GC fragility (structured effect not '
                   'separable from matched noise)') if r_ab >= r_noise - 0.05 else (
                   'order effect EXCEEDS matched white noise → structured, not just '
                   'fragility')
        print(f'  → {verdict}')
    print(f'\nTables: {OUT_ROOT}/roi_tc_comparison.csv , gc_comparison.csv')
    print('Interpretation: source TCs |r|→1 but GC r well below 1 ⇒ GC amplifies '
          'sub-percent source differences. Compare A-vs-B r to the matched-noise '
          'baseline: if A-vs-B is as bad as / worse than matched noise, GC cannot '
          'distinguish downsample order from noise at this window/order — a '
          'cautionary result worth reporting.')


if __name__ == '__main__':
    main()
