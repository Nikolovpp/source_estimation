#!/usr/bin/env python3
"""
Decoding TIMING per ROI, against a null (visual-cortex) ROI.

Under heavy inverse cross-talk every ROI inherits a copy of the dominant
signal, so differences in decoding AMPLITUDE between ROIs are not
interpretable.  Differences in WHEN information appears can still be, and
an ROI that carries nothing task-specific (early visual cortex in an
auditory / articulation task) gives the null against which regional
specificity is tested.  For one decoding run (task, atlas, method, ...):

  1. group accuracy curves per ROI (± SEM), smoothed with a short boxcar;
  2. jackknife onset and peak latencies per ROI (Miller, Patterson &
     Ulrich 1998): onset = first of N consecutive windows at which the
     leave-one-subject-out group curve exceeds chance + frac x (peak -
     chance); SE from the jackknife; latency differences between every
     ROI and the control (and between ROIs) with the jackknife t test
     (Ulrich & Miller 2001);
  3. regional specificity over time: ROI minus control accuracy per
     subject, sign-flip cluster permutation over time (mne.stats.
     permutation_cluster_1samp_test, one-tailed ROI > control), the same
     estimator the decoding stats use;
  4. ROI x time interaction: one-way repeated-measures F across all ROIs
     at every window, cluster permutation over time (f_mway_rm);
  5. the a-priori fixed windows (config.GC_WINDOWS_OF_INTEREST): window-mean
     accuracy per ROI vs control, paired t, two-tailed.

Reads the per-subject decoding CSVs written by run_decode.py; computes
nothing in source space.  The control ROI defaults to 'Visual' (present in
the sensorEquiv atlas); for the custom atlas use ``--control-roi vwfa-lh``.

Usage
-----
    conda activate mne
    # translation arm
    python exploratory/decode_onset_latency.py --task overtProd --method eLORETA \\
        --atlas sensorEquiv --classifier logistic --tune-hyperparams
    # existing custom-ROI run, visual word form area as the control
    python exploratory/decode_onset_latency.py --task overtProd --method LCMV \\
        --atlas custom --leakage-correction --classifier logistic --tune-hyperparams \\
        --control-roi vwfa-lh

Output
------
    DECODE_source_space/{task}/{method}/{atlas}/{feature_mode}/{leak}/{pseudo}/{sw}/{clf}/
        _onset_latency/{stim}_onset_latency.png     curves + stats above the traces
                       {stim}_latencies.csv          per-ROI onset/peak (jackknife SE)
                       {stim}_latency_contrasts.csv  ROI-vs-ROI jackknife tests
                       {stim}_windows.csv            fixed-window accuracy vs control
                       {stim}_summary.txt
"""
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import (SUBJECT_IDS, DECODE_OUTPUT_ROOT, DEFAULT_C, SW_DUR,
                    SW_STEP_SIZE, INVERSE_METHODS, ATLAS_CHOICES,
                    GC_WINDOWS_OF_INTEREST, classifier_path_segment)

CHANCE = 0.5
N_PERM = 1024
SEED = 42
# fixed categorical order (never cycled); the control ROI is always grey
ROI_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300',
              '#6250d6', '#e34948']
CONTROL_COLOR = '#52514e'


# ─────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────
def result_dir(args, stim):
    leakage_tag = 'leakage_corrected' if args.leakage_correction else 'raw'
    pseudo_tag = (f'pseudo_{args.pseudo_trial_size}' if args.pseudo_trial_size > 0
                  else 'no_pseudo')
    c = args.c if args.c is not None else DEFAULT_C.get(args.classifier, 1.0)
    clf_tag = classifier_path_segment(args.classifier, c, args.tune_hyperparams)
    return (DECODE_OUTPUT_ROOT / args.task / args.method / args.atlas
            / args.feature_mode / leakage_tag / pseudo_tag
            / f'{args.sw_dur}_{args.sw_step}' / clf_tag / stim)


def load_run(d, task, stim, sw_dur, sw_step, subjects):
    """{roi: (n_subj, n_times)}, times_ms, subjects_used."""
    per_roi, times, used = {}, None, []
    for s in subjects:
        f = d / f'{s}_{task}_{stim}_{sw_dur}_{sw_step}.csv'
        if not f.exists():
            continue
        df = pd.read_csv(f)
        piv = df.pivot_table(index='ms', columns='key', values='decode_acc').sort_index()
        if times is None:
            times = piv.index.values.astype(float)
        elif len(piv.index) != len(times) or not np.allclose(piv.index.values, times):
            print(f'  {s}: time axis differs — skipped')
            continue
        for roi in piv.columns:
            per_roi.setdefault(roi, []).append(piv[roi].values)
        used.append(s)
    if not used:
        raise FileNotFoundError(f'no CSVs in {d}')
    per_roi = {r: np.vstack(v) for r, v in per_roi.items() if len(v) == len(used)}
    return per_roi, times, used


def smooth(x, times, smooth_ms):
    """Boxcar along the last axis, width = smooth_ms (NaN-safe)."""
    if smooth_ms <= 0:
        return x
    step = float(np.median(np.diff(times)))
    n = max(int(round(smooth_ms / step)), 1)
    if n < 2:
        return x
    k = np.ones(n) / n
    pad = n // 2
    xp = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(pad, n - 1 - pad)], mode='edge')
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode='valid'), -1, xp)


# ─────────────────────────────────────────────────────────────────────
# Latencies
# ─────────────────────────────────────────────────────────────────────
def onset_of(curve, times, mask, frac, n_consec):
    """Onset (ms) of a group curve: first run of n_consec windows inside
    ``mask`` at or above chance + frac x (peak - chance); NaN if never."""
    c = np.where(mask, curve, np.nan)
    if not np.isfinite(c).any():
        return np.nan, np.nan
    peak = np.nanmax(c)
    if peak <= CHANCE:
        return np.nan, float(times[np.nanargmax(c)])
    thr = CHANCE + frac * (peak - CHANCE)
    above = np.isfinite(c) & (c >= thr)
    run = 0
    for i, a in enumerate(above):
        run = run + 1 if a else 0
        if run >= n_consec:
            return float(times[i - n_consec + 1]), float(times[np.nanargmax(c)])
    return np.nan, float(times[np.nanargmax(c)])


def jackknife_latencies(X, times, mask, frac, n_consec):
    """X: (n_subj, n_times).  Returns dict with grand onset/peak, the n
    leave-one-out estimates and jackknife SEs."""
    n = X.shape[0]
    grand = X.mean(axis=0)
    on, pk = onset_of(grand, times, mask, frac, n_consec)
    loo_on, loo_pk = [], []
    for i in range(n):
        g = np.delete(X, i, axis=0).mean(axis=0)
        o, p = onset_of(g, times, mask, frac, n_consec)
        loo_on.append(o)
        loo_pk.append(p)
    loo_on, loo_pk = np.array(loo_on), np.array(loo_pk)

    def jse(v):
        if not np.isfinite(v).all():
            return np.nan
        return float(np.sqrt((n - 1) / n * np.sum((v - v.mean()) ** 2)))
    return dict(onset=on, peak=pk, onset_se=jse(loo_on), peak_se=jse(loo_pk),
                loo_onset=loo_on, loo_peak=loo_pk, n=n)


def jackknife_contrast(a, b, key):
    """Jackknife t test of a latency difference (Ulrich & Miller 2001):
    the difference of the leave-one-out estimates, SE scaled by (n-1)/n."""
    da = a[f'loo_{key}']
    db = b[f'loo_{key}']
    if not (np.isfinite(da).all() and np.isfinite(db).all()):
        return np.nan, np.nan, np.nan
    d = da - db
    n = d.size
    se = np.sqrt((n - 1) / n * np.sum((d - d.mean()) ** 2))
    diff = a[key] - b[key]
    if se == 0:
        return float(diff), np.nan, np.nan
    t = diff / se
    p = 2 * stats.t.sf(abs(t), n - 1)
    return float(diff), float(t), float(p)


# ─────────────────────────────────────────────────────────────────────
# Cluster tests
# ─────────────────────────────────────────────────────────────────────
def _to_mask(c, n):
    """Cluster -> boolean mask of length n (MNE returns bool arrays, index
    arrays or (slice,) tuples depending on version/dimensionality)."""
    if isinstance(c, tuple):
        c = c[0]
    c = np.asarray(c) if not isinstance(c, slice) else c
    if isinstance(c, np.ndarray) and c.dtype == bool:
        return c
    m = np.zeros(n, bool)
    m[c] = True
    return m


def cluster_vs_control(X_roi, X_ctrl, mask, alpha):
    from mne.stats import permutation_cluster_1samp_test
    d = (X_roi - X_ctrl)[:, mask]
    n = d.shape[0]
    thr = stats.t.ppf(1 - alpha, n - 1)
    T, clusters, pv, _ = permutation_cluster_1samp_test(
        d, threshold=thr, tail=1, n_permutations=N_PERM, seed=SEED,
        out_type='mask', verbose=False)
    sig = np.zeros(mask.sum(), bool)
    spans = []
    for c, p in zip(clusters, pv):
        c = _to_mask(c, mask.sum())
        if p < alpha:
            sig |= c
            spans.append((c, float(p)))
    pointwise = stats.ttest_1samp(d, 0.0, axis=0).pvalue / 2.0   # one-tailed
    pointwise = np.where(d.mean(axis=0) > 0, pointwise, 1.0)
    return sig, spans, pointwise


def cluster_interaction(Xs, mask, alpha):
    """One-way RM ANOVA (ROI) at each window + cluster permutation over time."""
    from mne.stats import f_mway_rm, f_threshold_mway_rm, permutation_cluster_test
    n_rois = len(Xs)
    n = Xs[0].shape[0]
    factor_levels = [n_rois]

    def stat_fun(*arrs):
        data = np.stack(arrs, axis=1)   # (n_subj, n_rois [conditions], n_times)
        return f_mway_rm(data, factor_levels, effects='A',
                         return_pvals=False)[0]
    thr = f_threshold_mway_rm(n, factor_levels, effects='A', pvalue=alpha)
    arrs = [x[:, mask] for x in Xs]
    F, clusters, pv, _ = permutation_cluster_test(
        arrs, threshold=thr, stat_fun=stat_fun, tail=1, n_permutations=N_PERM,
        seed=SEED, out_type='mask', verbose=False)
    sig = np.zeros(mask.sum(), bool)
    spans = []
    for c, p in zip(clusters, pv):
        c = _to_mask(c, mask.sum())
        if p < alpha:
            sig |= c
            spans.append((c, float(p)))
    return F, sig, spans


# ─────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--task', required=True, choices=['perception', 'overtProd'])
    p.add_argument('--stim-classes', nargs='+', default=['prodDiff', 'percDiff'])
    p.add_argument('--method', required=True, choices=INVERSE_METHODS)
    p.add_argument('--atlas', required=True, choices=ATLAS_CHOICES)
    p.add_argument('--feature-mode', default='vertex_selectkbest')
    p.add_argument('--leakage-correction', action='store_true', default=False)
    p.add_argument('--pseudo-trial-size', type=int, default=0)
    p.add_argument('--classifier', default='svm', choices=['svm', 'lda', 'logistic'])
    p.add_argument('--c', type=float, default=None)
    p.add_argument('--tune-hyperparams', action='store_true', default=False)
    p.add_argument('--sw-dur', type=int, default=SW_DUR)
    p.add_argument('--sw-step', type=int, default=SW_STEP_SIZE)
    p.add_argument('--subjects', nargs='+', default=SUBJECT_IDS)
    p.add_argument('--control-roi', default='Visual',
                   help="null ROI (default 'Visual'; custom atlas: vwfa-lh / owfa-lh)")
    p.add_argument('--rois', nargs='+', default=None,
                   help='ROIs to analyse (default: all in the CSVs)')
    p.add_argument('--smooth-ms', type=float, default=20.0)
    p.add_argument('--onset-frac', type=float, default=0.5,
                   help='onset threshold as a fraction of (peak - chance)')
    p.add_argument('--n-consecutive', type=int, default=5)
    p.add_argument('--tmin', type=float, default=None,
                   help='analysis window start (ms); default = first window')
    p.add_argument('--tmax', type=float, default=None)
    p.add_argument('--alpha', type=float, default=0.05)
    p.add_argument('--out-dir', default=None)
    args = p.parse_args()

    for stim in args.stim_classes:
        d = result_dir(args, stim)
        if not d.is_dir():
            print(f'{stim}: no results at {d} — skipped')
            continue
        out = Path(args.out_dir) if args.out_dir else d.parent / '_onset_latency'
        out.mkdir(parents=True, exist_ok=True)
        per_roi, times, used = load_run(d, args.task, stim, args.sw_dur,
                                        args.sw_step, args.subjects)
        rois = args.rois or list(per_roi)
        lut = {r.lower(): r for r in per_roi}
        rois = [lut[r.lower()] for r in rois if r.lower() in lut]
        ctrl = lut.get(args.control_roi.lower())
        if ctrl is None:
            raise SystemExit(f'control ROI {args.control_roi!r} not in '
                             f'{sorted(per_roi)}')
        test_rois = [r for r in rois if r != ctrl]
        mask = np.ones(times.size, bool)
        if args.tmin is not None:
            mask &= times >= args.tmin
        if args.tmax is not None:
            mask &= times <= args.tmax
        print(f'{args.task}/{stim}: {len(used)} subjects, ROIs {test_rois}, '
              f'control {ctrl}')

        Xs = {r: smooth(per_roi[r], times, args.smooth_ms) for r in rois}
        # ── latencies ───────────────────────────────────────────────
        lat = {r: jackknife_latencies(Xs[r], times, mask, args.onset_frac,
                                      args.n_consecutive) for r in rois}
        lat_rows = [dict(roi=r, is_control=(r == ctrl), n=lat[r]['n'],
                         onset_ms=lat[r]['onset'], onset_se=lat[r]['onset_se'],
                         peak_ms=lat[r]['peak'], peak_se=lat[r]['peak_se'],
                         peak_acc=float(np.nanmax(np.where(mask, Xs[r].mean(0), np.nan))))
                    for r in rois]
        pd.DataFrame(lat_rows).to_csv(out / f'{stim}_latencies.csv', index=False)
        con_rows = []
        for i, a in enumerate(rois):
            for b in rois[i + 1:]:
                for key in ('onset', 'peak'):
                    diff, t, pv = jackknife_contrast(lat[a], lat[b], key)
                    con_rows.append(dict(roi_a=a, roi_b=b, measure=key,
                                         diff_ms=diff, t_jack=t, p=pv))
        pd.DataFrame(con_rows).to_csv(out / f'{stim}_latency_contrasts.csv',
                                      index=False)
        # ── ROI > control clusters, interaction ─────────────────────
        vs = {r: cluster_vs_control(Xs[r], Xs[ctrl], mask, args.alpha)
              for r in test_rois}
        F, inter_sig, inter_spans = cluster_interaction(
            [Xs[r] for r in rois], mask, args.alpha)
        # ── fixed windows ───────────────────────────────────────────
        win_rows = []
        for wname, (lo, hi) in GC_WINDOWS_OF_INTEREST[args.task].items():
            m = mask & (times >= lo * 1000) & (times <= hi * 1000)
            for r in test_rois:
                a = Xs[r][:, m].mean(axis=1)
                c = Xs[ctrl][:, m].mean(axis=1)
                t, pv = stats.ttest_rel(a, c)
                t0, p0 = stats.ttest_1samp(a, CHANCE)
                win_rows.append(dict(window=wname, lo_ms=lo * 1000, hi_ms=hi * 1000,
                                     roi=r, acc=a.mean(), acc_control=c.mean(),
                                     diff=(a - c).mean(), t_paired=t, p_paired=pv,
                                     p_vs_chance=p0))
        win_df = pd.DataFrame(win_rows)
        win_df.to_csv(out / f'{stim}_windows.csv', index=False)

        # ── figure ──────────────────────────────────────────────────
        t_m = times[mask]
        fig, axes = plt.subplots(3, 1, figsize=(10, 9.5), sharex=True,
                                 gridspec_kw=dict(height_ratios=[3.2, 1.3, 1.0]))
        ax = axes[0]
        colors = {r: ROI_COLORS[i % len(ROI_COLORS)] for i, r in enumerate(test_rois)}
        colors[ctrl] = CONTROL_COLOR
        ymax = 0
        for r in rois:
            mu = Xs[r].mean(0)[mask]
            se = Xs[r].std(0, ddof=1)[mask] / np.sqrt(Xs[r].shape[0])
            ymax = max(ymax, np.nanmax(mu + se))
            ax.fill_between(t_m, mu - se, mu + se, color=colors[r], alpha=0.15, lw=0)
            ax.plot(t_m, mu, color=colors[r], lw=1.6 if r != ctrl else 1.4,
                    ls='-' if r != ctrl else '--',
                    label=f'{r.replace("-lh", "")}{" (control)" if r == ctrl else ""}')
            if np.isfinite(lat[r]['onset']):
                ax.plot(lat[r]['onset'], CHANCE - 0.004, marker='^', color=colors[r],
                        ms=7, clip_on=False)
        for lo, hi in GC_WINDOWS_OF_INTEREST[args.task].values():
            ax.axvspan(lo * 1000, hi * 1000, color='#eeeeee', zorder=0)
        ax.axhline(CHANCE, color='#c3c2b7', lw=0.8)
        ax.axvline(0, color='#c3c2b7', lw=0.8, ls=':')
        # stats ABOVE the traces: per ROI a row — asterisks = pointwise, bar = cluster
        span = max(ymax - CHANCE, 0.02)
        for i, r in enumerate(test_rois):
            sig, spans, pw = vs[r]
            y = ymax + (0.08 + 0.09 * i) * span
            pts = pw < args.alpha
            if pts.any():
                ax.plot(t_m[pts], np.full(pts.sum(), y), ls='none', marker='*',
                        ms=3, color=colors[r], alpha=0.8)
            if sig.any():
                ax.plot(t_m[sig], np.full(sig.sum(), y + 0.035 * span), ls='none',
                        marker='_', ms=5, mew=2, color=colors[r])
        ax.set_ylim(CHANCE - 0.02, ymax + (0.2 + 0.09 * len(test_rois)) * span)
        ax.set_ylabel('decoding accuracy')
        ax.legend(fontsize=8, frameon=False, ncol=len(rois), loc='lower left',
                  bbox_to_anchor=(0, 1.0))
        fig.suptitle(f'{args.task} / {stim} — {args.method} / {args.atlas} / '
                     f'{args.classifier}{" tuned" if args.tune_hyperparams else ""} '
                     f'(n = {len(used)}); triangles = jackknife onset; above the traces: '
                     f'* pointwise ROI > control p<.05, bar = cluster', fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)

        ax = axes[1]
        yv = np.arange(len(rois))[::-1]
        for yy, r in zip(yv, rois):
            o, se = lat[r]['onset'], lat[r]['onset_se']
            if np.isfinite(o):
                ax.errorbar(o, yy, xerr=se if np.isfinite(se) else 0, fmt='o',
                            color=colors[r], capsize=3, ms=6)
            else:
                ax.text(t_m[0], yy, 'no onset', fontsize=8, va='center',
                        color=colors[r])
            pk = lat[r]['peak']
            if np.isfinite(pk):
                ax.plot(pk, yy, marker='D', color=colors[r], ms=4, mfc='white')
        ax.set_yticks(yv)
        ax.set_yticklabels([r.replace('-lh', '') for r in rois], fontsize=8)
        ax.set_ylabel('onset ± jackknife SE\n(diamond = peak)', fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='x', color='#eeeeee', lw=0.6)

        ax = axes[2]
        ax.plot(t_m, F, color='#0b0b0b', lw=1.2)
        if inter_sig.any():
            ax.plot(t_m[inter_sig], np.full(inter_sig.sum(), np.nanmax(F) * 1.08),
                    ls='none', marker='_', ms=5, mew=2, color='#e34948')
        ax.set_ylabel('ROI x time\nRM-ANOVA F', fontsize=8)
        ax.set_xlabel('time (ms, window center)')
        ax.spines[['top', 'right']].set_visible(False)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(out / f'{stim}_onset_latency.png', dpi=160)
        plt.close(fig)

        # ── summary ─────────────────────────────────────────────────
        L = [f'{args.task}/{stim} — {args.method}/{args.atlas}, n={len(used)}, '
             f'control = {ctrl}', '',
             f'{"ROI":20s} {"onset ms":>9s} {"±SE":>6s} {"peak ms":>8s} {"±SE":>6s} '
             f'{"peak acc":>8s}  {"ROI>control clusters":s}']
        for r in rois:
            l_ = lat[r]
            cl = ''
            if r in vs:
                cl = '; '.join(f'{t_m[c][0]:.0f}..{t_m[c][-1]:.0f} ms (p={pv:.3f})'
                               for c, pv in vs[r][1]) or 'none'
            L.append(f'{r.replace("-lh", ""):20s} {l_["onset"]:9.0f} {l_["onset_se"]:6.0f} '
                     f'{l_["peak"]:8.0f} {l_["peak_se"]:6.0f} '
                     f'{np.nanmax(np.where(mask, Xs[r].mean(0), np.nan)):8.3f}  {cl}')
        L.append('')
        L.append('Jackknife latency contrasts vs control (onset):')
        for row in con_rows:
            if row['measure'] == 'onset' and ctrl in (row['roi_a'], row['roi_b']):
                L.append(f'  {row["roi_a"]} - {row["roi_b"]}: {row["diff_ms"]:+.0f} ms, '
                         f't_jack = {row["t_jack"]:.2f}, p = {row["p"]:.3f}')
        L.append('')
        L.append('ROI x time interaction clusters: ' + (
            '; '.join(f'{t_m[c][0]:.0f}..{t_m[c][-1]:.0f} ms (p={pv:.3f})'
                      for c, pv in inter_spans) or 'none'))
        L.append('')
        L.append('Fixed windows (window-mean accuracy, ROI vs control, paired t):')
        for _, r in win_df.iterrows():
            L.append(f'  {r.window:9s} {r.roi.replace("-lh", ""):18s} acc {r.acc:.3f} '
                     f'ctrl {r.acc_control:.3f} diff {r["diff"]:+.3f} p {r.p_paired:.3f} '
                     f'(vs chance p {r.p_vs_chance:.3g})')
        text = '\n'.join(L)
        print('\n' + text)
        (out / f'{stim}_summary.txt').write_text(text + '\n')
        print(f'\nWrote {out}')


if __name__ == '__main__':
    main()
