#!/usr/bin/env python3
"""Compare TWO GC result directories, subject-paired, edge by edge and band by band.

Point it at two directories of per-subject GC ``.npz`` — anything written by
``run_granger.py`` (BSMART/parametric) or ``run_granger_mne.py`` (MNE
state-space) — and it answers: *does the answer change?*  The intended use is
an A/B on one knob at a time, e.g.::

    .../leakage_corrected/order6_win60ms_fs200         (A)
    .../leakage_corrected/order6_win60ms_fs200_demean  (B)

but it is deliberately agnostic about WHAT differs: it aligns on subjects,
directed edges and window times, and compares whatever is left.  Comparing two
estimators (BSMART vs MNE) works too; see ``--on-window-mismatch`` for how the
different moving-window axes are reconciled.

What it computes, per directed edge x band
------------------------------------------
* **span mean, per subject, in each condition** -> paired t-test and
  Wilcoxon signed-rank (two-tailed: this asks *whether* the conditions differ,
  not whether either exceeds baseline — that is ``granger_stats.py``'s job)
* **effect size** — Cohen's dz on the paired difference, plus the ratio of
  group means (how many-fold A over-/under-states B)
* **across-subject Pearson & Spearman r** — does the condition preserve the
  ranking of subjects?  A high r with a large mean shift means a rescaling; a
  low r means the two conditions are measuring genuinely different things.
* **sign agreement of net GC** (i->j minus j->i) — the directional conclusion
* **cluster-based permutation on the paired difference across the span**
  (two-tailed sign-flip, ``mne.stats.permutation_cluster_1samp_test``), so
  "they differ in a sustained time window" is FWER-controlled over windows.
  P-values are then corrected across the (edge x band) family (Bonferroni + BH).

Outputs (in ``--out-dir``, default ``<dir-a>/../_condition_comparison/<stem>``)
------------------------------------------------------------------------------
* ``gc_condition_comparison.csv``  — one row per edge x band (the summary above)
* ``gc_condition_comparison_windows.csv`` — one row per edge x band x window
  (group means, paired difference, per-window p, cluster p)
* ``<edge>_<label_a>_vs_<label_b>.png`` — per edge, 4 band panels: both
  conditions' group-mean time courses +/-SEM with the significant-difference
  clusters shaded
* ``scatter_<band>.png`` — per band, subject-level A vs B task means, one panel
  per edge, with the identity line
* ``summary.log`` — the printed table

Usage
-----
    python compare_gc_conditions.py \\
        --dir-a  .../leakage_corrected/order6_win60ms_fs200/rois_.../prodDiff \\
        --dir-b  .../leakage_corrected/order6_win60ms_fs200_demean/rois_.../prodDiff \\
        --label-a raw --label-b demean

    # labels default to the config directory name, so this is usually enough:
    python compare_gc_conditions.py --dir-a <A> --dir-b <B>
"""
import argparse
import os
import re
import sys
import glob

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import GC_TASK_END
from granger import DEFAULT_BANDS
from granger_stats import (N_PERMUTATIONS, TFCE_THRESHOLD, bh_fdr,
                           cluster_to_mask, infer_task_from_path,
                           _contiguous_spans)

# Okabe-Ito, colour-blind safe
PAL_A, PAL_B, PAL_SIG = '#0072B2', '#D55E00', '#b2182b'


# ─────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────
def load_condition(gc_dir, bands):
    """Load every subject .npz in ``gc_dir`` keyed by subject and edge.

    Returns dict with ``subjects`` (list), ``window_ms`` (n_win,), ``edges``
    (list of (src, tgt) directed-edge names, 2 per stored pair) and ``gc``
    {band: (n_subj, n_edge, n_win)}.

    Both directions are unrolled into explicit directed edges here, so two
    directories that stored the SAME edge with i/j swapped still align.
    """
    files = sorted(glob.glob(os.path.join(str(gc_dir), '*.npz')))
    if not files:
        raise FileNotFoundError(f'No GC .npz files in {gc_dir}')
    band_names = list(bands)
    subjects, window_ms, edges = [], None, None
    per_subj = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        subj = os.path.basename(f).split('_')[0]
        roi = list(map(str, d['roi_names']))
        pi, pj = d['pair_i'], d['pair_j']
        # directed edges: fxy row p is roi[pi]->roi[pj]; fyx row p is the reverse
        e = ([(roi[int(a)], roi[int(b)]) for a, b in zip(pi, pj)]
             + [(roi[int(b)], roi[int(a)]) for a, b in zip(pi, pj)])
        vals = {}
        for b in band_names:
            vals[b] = np.concatenate([d[f'fxy_{b}'], d[f'fyx_{b}']], axis=0)
        wm = np.asarray(d['window_ms'], float)
        d.close()
        if window_ms is None:
            window_ms, edges = wm, e
        elif not np.array_equal(wm, window_ms):
            raise ValueError(
                f'{subj} in {gc_dir} has a different moving-window axis than '
                f'{subjects[0]} ({wm.size} vs {window_ms.size} windows). '
                'These were produced by different runs — do not mix them.')
        subjects.append(subj)
        per_subj.append((e, vals))

    # per-subject edge order can differ (pairs are re-resolved per subject in
    # run_granger_mne.process_subject); index by NAME rather than by position.
    common = [e for e in edges if all(e in es for es, _ in per_subj)]
    dropped = [e for e in edges if e not in common]
    gc = {b: np.stack([
            np.stack([v[b][es.index(e)] for e in common])
            for es, v in per_subj]) for b in band_names}
    return {'subjects': subjects, 'window_ms': window_ms, 'edges': common,
            'gc': gc, 'dropped_edges': dropped, 'dir': str(gc_dir)}


def align(A, B, on_window_mismatch='intersect'):
    """Reduce A and B to their common subjects, edges and windows (in place).

    ``on_window_mismatch``:
      ``intersect``   keep only window times present in both (default; exact
                      match on the 5 ms grid, which is what an A/B on one knob
                      gives you)
      ``interp``      linearly resample B onto A's window axis — use when
                      comparing estimators whose axes genuinely differ
      ``error``       refuse
    """
    subs = [s for s in A['subjects'] if s in set(B['subjects'])]
    if not subs:
        raise ValueError('no subjects in common between the two directories')
    edges = [e for e in A['edges'] if e in set(B['edges'])]
    if not edges:
        raise ValueError('no directed edges in common between the two directories')

    def take(C, want_edges, want_subs):
        si = [C['subjects'].index(s) for s in want_subs]
        ei = [C['edges'].index(e) for e in want_edges]
        C['gc'] = {b: v[np.ix_(si, ei)] for b, v in C['gc'].items()}
        C['subjects'] = list(want_subs)
        C['edges'] = list(want_edges)

    take(A, edges, subs)
    take(B, edges, subs)

    wa, wb = A['window_ms'], B['window_ms']
    if np.array_equal(wa, wb):
        return A, B, wa, 'identical'
    if on_window_mismatch == 'error':
        raise ValueError(f'window axes differ ({wa.size} vs {wb.size}); pass '
                         '--on-window-mismatch intersect|interp')
    if on_window_mismatch == 'interp':
        lo, hi = max(wa[0], wb[0]), min(wa[-1], wb[-1])
        keep = (wa >= lo) & (wa <= hi)
        w = wa[keep]
        A['gc'] = {b: v[:, :, keep] for b, v in A['gc'].items()}
        B['gc'] = {b: np.stack([[np.interp(w, wb, row) for row in subj]
                                for subj in v]) for b, v in B['gc'].items()}
        return A, B, w, f'B interpolated onto A ({w.size} windows, {lo:g}..{hi:g} ms)'
    w, ia, ib = np.intersect1d(wa, wb, return_indices=True)
    if w.size < 2:
        raise ValueError(
            f'window axes share only {w.size} time point(s) '
            f'(A {wa[0]:g}..{wa[-1]:g}, B {wb[0]:g}..{wb[-1]:g}). '
            'Pass --on-window-mismatch interp to resample instead.')
    A['gc'] = {b: v[:, :, ia] for b, v in A['gc'].items()}
    B['gc'] = {b: v[:, :, ib] for b, v in B['gc'].items()}
    return A, B, w, (f'intersected to {w.size} shared windows '
                     f'({w[0]:g}..{w[-1]:g} ms; A had {wa.size}, B had {wb.size})')


# ─────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────
def paired_cluster(diff, n_permutations=N_PERMUTATIONS, seed=42, tail=0,
                   tfce=True, alpha=0.05):
    """Two-tailed sign-flip permutation on a (n_subj, n_win) paired difference.

    Runs cluster-mass and (unless ``tfce=False``) TFCE.  ``tail=0`` because the
    question is whether the conditions differ, in either direction.

    Read this alongside the paired t on the task MEAN, not instead of it.  The
    two answer different questions and have very different power profiles:

      * a *uniform offset* across the whole span — e.g. what ERP removal
        does — puts a modest t at every window.  If that per-window t sits below
        the cluster-forming threshold, cluster-mass finds only short accidental
        runs and reports a null.  Verified on this data: an offset with a
        task-mean paired t of -4.58 (p=2e-4) produced a largest cluster of 7
        windows out of 344 and a cluster p of 0.075.  TFCE recovers some of it
        (p=0.042) because it integrates across thresholds.  The task-mean t is
        the right test for that shape of effect.
      * a *time-localised* divergence — the conditions agreeing early and
        parting company at some latency — is exactly what cluster-mass is for,
        and the task-mean t will dilute it towards nothing.

    Returns (p_per_window, sig_mask, p_cluster_min, p_tfce_min).
    """
    from mne.stats import permutation_cluster_1samp_test
    n_win = diff.shape[1]
    p_win = np.full(n_win, np.nan)
    sig = np.zeros(n_win, bool)
    if diff.shape[0] < 3 or n_win < 2 or not np.isfinite(diff).all() \
            or np.allclose(diff, 0):
        return p_win, sig, np.nan, np.nan
    _T, clusters, cl_p, _ = permutation_cluster_1samp_test(
        diff, threshold=None, n_permutations=n_permutations, tail=tail,
        out_type='mask', seed=seed, verbose=False)
    for cl, p in zip(clusters, cl_p):
        m = cluster_to_mask(cl, n_win)
        p_win[m] = np.where(np.isnan(p_win[m]), p, np.minimum(p_win[m], p))
        if p < alpha:
            sig[m] = True
    p_min = float(np.min(cl_p)) if len(cl_p) else np.nan
    p_tfce_min = np.nan
    if tfce:
        _T2, _c2, p_tfce, _ = permutation_cluster_1samp_test(
            diff, threshold=TFCE_THRESHOLD, n_permutations=n_permutations,
            tail=tail, out_type='mask', seed=seed, verbose=False)
        p_tfce_min = float(np.min(p_tfce))
    return p_win, sig, p_min, p_tfce_min


def compare(A, B, window_ms, bands, task_start_ms, task_end_ms,
            n_permutations=N_PERMUTATIONS, seed=42, alpha=0.05, tfce=True):
    """Per edge x band summary rows + per-window rows."""
    band_names = list(bands)
    task = window_ms >= task_start_ms
    if task_end_ms is not None:
        task &= window_ms <= task_end_ms
    if not task.any():
        raise ValueError(
            f'no windows in the compared span [{task_start_ms:g}, '
            f'{task_end_ms if task_end_ms is not None else "end"}] ms; the '
            f'shared window axis runs {window_ms[0]:g}..{window_ms[-1]:g} ms')

    summary, per_win = [], []
    for ei, (src, tgt) in enumerate(A['edges']):
        for b in band_names:
            a = A['gc'][b][:, ei, :]                 # (n_subj, n_win)
            bb = B['gc'][b][:, ei, :]
            am, bm = a.mean(0), bb.mean(0)
            n = a.shape[0]
            row = {'src': src, 'tgt': tgt, 'edge': f'{src}->{tgt}', 'band': b,
                   'n_subj': n}
            # per-subject span mean
            ta, tb = a[:, task].mean(1), bb[:, task].mean(1)
            allnan = not (np.isfinite(ta).any() and np.isfinite(tb).any())
            row.update({'mean_a': np.nanmean(ta), 'mean_b': np.nanmean(tb)})
            if allnan:
                row.update({k: np.nan for k in
                            ['diff', 'ratio_a_over_b', 't', 'p_ttest',
                             'p_wilcoxon', 'dz', 'pearson_r', 'spearman_r',
                             'p_cluster_min', 'p_tfce_min', 'n_sig_windows']})
                row['note'] = 'all-NaN in at least one condition'
                summary.append(row)
                continue
            d = ta - tb
            row['diff'] = float(np.nanmean(d))
            row['ratio_a_over_b'] = (float(np.nanmean(ta) / np.nanmean(tb))
                                     if np.nanmean(tb) != 0 else np.nan)
            t, p = stats.ttest_rel(ta, tb, nan_policy='omit')
            row['t'], row['p_ttest'] = float(t), float(p)
            try:
                row['p_wilcoxon'] = float(stats.wilcoxon(d)[1])
            except ValueError:
                row['p_wilcoxon'] = np.nan
            sd = np.nanstd(d, ddof=1)
            row['dz'] = float(np.nanmean(d) / sd) if sd > 0 else np.nan
            row['pearson_r'] = float(stats.pearsonr(ta, tb)[0]) if n > 2 else np.nan
            row['spearman_r'] = float(stats.spearmanr(ta, tb)[0]) if n > 2 else np.nan

            p_win, sig, p_min, p_tfce_min = paired_cluster(
                (a - bb)[:, task], n_permutations, seed, tfce=tfce, alpha=alpha)
            row['p_cluster_min'] = p_min
            row['p_tfce_min'] = p_tfce_min
            row['n_sig_windows'] = int(sig.sum())
            row['n_task_windows'] = int(task.sum())
            row['note'] = ''
            summary.append(row)

            pw_p = np.full(window_ms.size, np.nan); pw_p[task] = p_win
            pw_s = np.zeros(window_ms.size, bool);  pw_s[task] = sig
            for w in range(window_ms.size):
                per_win.append({
                    'src': src, 'tgt': tgt, 'edge': f'{src}->{tgt}', 'band': b,
                    'window_ms': window_ms[w],
                    'mean_a': am[w], 'mean_b': bm[w],
                    'sem_a': a[:, w].std(ddof=1) / np.sqrt(n),
                    'sem_b': bb[:, w].std(ddof=1) / np.sqrt(n),
                    'diff': am[w] - bm[w],
                    'p_cluster': pw_p[w], 'sig_cluster': pw_s[w],
                    'in_task': bool(task[w]),
                })

    df = pd.DataFrame(summary)
    n_fam = len(df)
    for col in ['p_ttest', 'p_cluster_min', 'p_tfce_min']:
        df[col + '_fam_bonf'] = np.minimum(df[col] * n_fam, 1.0)
        df[col + '_fam_fdr'] = bh_fdr(df[col].to_numpy())
    df['sig_cluster_fam_fdr'] = df['p_cluster_min_fam_fdr'] < alpha
    df['sig_tfce_fam_fdr'] = df['p_tfce_min_fam_fdr'] < alpha
    df['sig_ttest_fam_fdr'] = df['p_ttest_fam_fdr'] < alpha
    return df, pd.DataFrame(per_win), task


# ─────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────
def fig_edge(win_df, src, tgt, bands, labels, out_path, task_start_ms,
             task_end_ms, fmt='png'):
    band_names = list(bands)
    sub = win_df[(win_df.src == src) & (win_df.tgt == tgt)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for ax, b in zip(axes.ravel(), band_names):
        s = sub[sub.band == b].sort_values('window_ms')
        w = s.window_ms.to_numpy()
        for col, mcol, scol, lab in [(PAL_A, 'mean_a', 'sem_a', labels[0]),
                                     (PAL_B, 'mean_b', 'sem_b', labels[1])]:
            m, e = s[mcol].to_numpy(), s[scol].to_numpy()
            ax.plot(w, m, color=col, lw=1.8, label=lab)
            ax.fill_between(w, m - e, m + e, color=col, alpha=0.2, lw=0)
        for k, (i0, i1) in enumerate(_contiguous_spans(s.sig_cluster.to_numpy())):
            ax.axvspan(w[i0], w[i1], color=PAL_SIG, alpha=0.13, lw=0,
                       label='A≠B (cluster p<.05)' if k == 0 else None)
        ax.axvline(0, color='k', lw=0.8, alpha=0.5)
        if task_start_ms is not None:
            ax.axvline(task_start_ms, color='0.4', ls=':', lw=1)
        ax.set_title(f'{b} ({bands[b][0]:g}–{bands[b][1]:g} Hz)', fontsize=11)
        ax.set_ylabel('GC')
    for ax in axes[1]:
        ax.set_xlabel('window start (ms)')
    axes[0][0].legend(fontsize=9, frameon=False, loc='upper left')
    fig.suptitle(f'{src} → {tgt}:  {labels[0]}  vs  {labels[1]}   '
                 f'(group mean ±SEM)', fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, format=fmt, bbox_inches='tight')
    plt.close(fig)


def fig_scatter(A, B, band, task, labels, out_path, fmt='png'):
    edges = A['edges']
    ncol = min(4, len(edges)); nrow = int(np.ceil(len(edges) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 3.2 * nrow),
                             squeeze=False)
    for k, (src, tgt) in enumerate(edges):
        ax = axes[k // ncol][k % ncol]
        x = A['gc'][band][:, k, :][:, task].mean(1)
        y = B['gc'][band][:, k, :][:, task].mean(1)
        ax.scatter(x, y, s=22, color=PAL_A, alpha=0.8, edgecolor='white', lw=0.5)
        finite = np.isfinite(x) & np.isfinite(y)
        if finite.sum() > 1:
            lim = [min(x[finite].min(), y[finite].min()),
                   max(x[finite].max(), y[finite].max())]
            ax.plot(lim, lim, color='0.5', ls='--', lw=1)
            r = stats.pearsonr(x[finite], y[finite])[0] if finite.sum() > 2 else np.nan
            ax.set_title(f'{src}→{tgt}\nr={r:.2f}', fontsize=9)
        else:
            ax.set_title(f'{src}→{tgt}\n(no data)', fontsize=9)
        ax.set_xlabel(labels[0], fontsize=8); ax.set_ylabel(labels[1], fontsize=8)
        ax.tick_params(labelsize=7)
    for k in range(len(edges), nrow * ncol):
        axes[k // ncol][k % ncol].axis('off')
    fig.suptitle(f'Span-mean GC per subject — {band}', fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, format=fmt, bbox_inches='tight')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────
def default_label(gc_dir):
    """The config directory name — the part that actually differs in an A/B."""
    parts = [p for p in re.split(r'[\\/]+', str(gc_dir).rstrip('/\\')) if p]
    for p in reversed(parts):
        if re.match(r'^(order\d|ssgc|gc_)', p):
            return p
    return parts[-3] if len(parts) >= 3 else parts[-1]


def parse_args():
    p = argparse.ArgumentParser(
        description='Compare two GC result directories, subject-paired',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dir-a', required=True, help='condition A: dir of subject .npz')
    p.add_argument('--dir-b', required=True, help='condition B: dir of subject .npz')
    p.add_argument('--label-a', default=None,
                   help='name for A in figures/CSV (default: its config dir name)')
    p.add_argument('--label-b', default=None)
    p.add_argument('--out-dir', default=None,
                   help='default: <dir-a>/../../_condition_comparison/<A>_vs_<B>')
    p.add_argument('--task', default=None, choices=['perception', 'overtProd'],
                   help='sets the default task-end crop; inferred from --dir-a')
    p.add_argument('--span-start', type=float, default=None,
                   help='compared span starts here (s). Default: the first '
                        'window. Unlike granger_stats.py this does NOT exclude '
                        'the baseline period — whether two configs agree during '
                        'baseline is part of the question. Pass e.g. 0 to '
                        'restrict the comparison to the post-stimulus span.')
    p.add_argument('--span-end', type=float, default=None,
                   help='compared span ends here (s); default config.GC_TASK_END '
                        'for the task, which drops the trailing MVAR-boundary '
                        'windows. Pass a large value to disable the crop.')
    p.add_argument('--on-window-mismatch', default='intersect',
                   choices=['intersect', 'interp', 'error'],
                   help='how to reconcile different moving-window axes '
                        '(default intersect)')
    p.add_argument('--n-permutations', type=int, default=N_PERMUTATIONS)
    p.add_argument('--no-tfce', dest='tfce', action='store_false', default=True,
                   help='skip the TFCE pass (~2x faster). Keep it for diffuse '
                        'offsets — cluster-mass is underpowered against those')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--alpha', type=float, default=0.05)
    p.add_argument('--bands', nargs='+', default=None,
                   help='subset of theta alpha low_beta high_beta')
    p.add_argument('--format', default='png', choices=['png', 'svg'])
    p.add_argument('--no-figures', dest='figures', action='store_false',
                   default=True)
    return p.parse_args()


def main():
    a = parse_args()
    bands = DEFAULT_BANDS if a.bands is None else \
        {k: DEFAULT_BANDS[k] for k in a.bands}
    la = a.label_a or default_label(a.dir_a)
    lb = a.label_b or default_label(a.dir_b)
    if la == lb:
        la, lb = la + ' (A)', lb + ' (B)'
    task = a.task or infer_task_from_path(a.dir_a)

    print(f'A [{la}]  {a.dir_a}\nB [{lb}]  {a.dir_b}')
    A = load_condition(a.dir_a, bands)
    B = load_condition(a.dir_b, bands)
    n_a, n_b = len(A['subjects']), len(B['subjects'])
    A, B, window_ms, how = align(A, B, a.on_window_mismatch)
    print(f'  subjects: {n_a} / {n_b} -> {len(A["subjects"])} paired')
    print(f'  edges:    {len(A["edges"])} common directed edges')
    print(f'  windows:  {how}')
    for C, lab in [(A, la), (B, lb)]:
        if C['dropped_edges']:
            print(f'  NOTE [{lab}] dropped edges missing in some subjects: '
                  f'{C["dropped_edges"]}')

    task_start_ms = (a.span_start * 1000.0 if a.span_start is not None
                     else float(window_ms[0]))
    if a.span_end is not None:
        task_end_ms = a.span_end * 1000.0
    elif task in GC_TASK_END:
        task_end_ms = GC_TASK_END[task] * 1000.0
    else:
        task_end_ms = None
    end_str = f'{task_end_ms:g}' if task_end_ms is not None else 'end'
    print(f'  compared span: [{task_start_ms:g}, {end_str}] ms  (task={task})')

    summary, per_win, task_mask = compare(
        A, B, window_ms, bands, task_start_ms, task_end_ms,
        a.n_permutations, a.seed, a.alpha, a.tfce)

    out_dir = a.out_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(a.dir_a.rstrip('/\\')))),
        '_condition_comparison', f'{la}_vs_{lb}'.replace(' ', '').replace('/', '-'))
    os.makedirs(out_dir, exist_ok=True)
    summary.insert(0, 'label_b', lb); summary.insert(0, 'label_a', la)
    summary.to_csv(os.path.join(out_dir, 'gc_condition_comparison.csv'), index=False)
    per_win.to_csv(os.path.join(out_dir, 'gc_condition_comparison_windows.csv'),
                   index=False)

    if a.figures:
        for src, tgt in A['edges']:
            fig_edge(per_win, src, tgt, bands, (la, lb),
                     os.path.join(out_dir, f'{src}_to_{tgt}_{la}_vs_{lb}'
                                           f'.{a.format}'.replace(' ', '')),
                     task_start_ms, task_end_ms, a.format)
        for b in bands:
            fig_scatter(A, B, b, task_mask, (la, lb),
                        os.path.join(out_dir, f'scatter_{b}.{a.format}'), a.format)

    # ── printed summary ────────────────────────────────────────────────
    lines = [f'GC condition comparison:  A={la}   B={lb}',
             f'  A: {a.dir_a}', f'  B: {a.dir_b}',
             f'  {len(A["subjects"])} paired subjects, {len(A["edges"])} edges, '
             f'{len(bands)} bands, {len(summary)} tests',
             f'  windows: {how}',
             f'  compared span [{task_start_ms:g}, {end_str}] ms '
             f'({int(task_mask.sum())} windows)', '',
             'Sorted by the paired t on the span mean — the sensitive '
             'test for a uniform shift.',
             'p_clust / p_tfce localise WHEN they differ and are '
             'underpowered against a diffuse offset (see paired_cluster).', '',
             f'{"edge":>22s} {"band":>10s} {"mean_A":>10s} {"mean_B":>10s} '
             f'{"A/B":>7s} {"dz":>7s} {"p_pair":>9s} {"pair_fdr":>9s} '
             f'{"r":>6s} {"p_clust":>9s} {"p_tfce":>9s} {"nsig":>6s}']
    s = summary.sort_values('p_ttest')
    for _, r in s.iterrows():
        if r.get('note'):
            lines.append(f'{r.edge:>22s} {r.band:>10s}   {r.note}')
            continue
        flag = ' **' if r.sig_ttest_fam_fdr else (
            ' *' if (np.isfinite(r.p_ttest) and r.p_ttest < a.alpha) else '')
        lines.append(
            f'{r.edge:>22s} {r.band:>10s} {r.mean_a:10.5f} {r.mean_b:10.5f} '
            f'{r.ratio_a_over_b:7.2f} {r.dz:7.2f} {r.p_ttest:9.4f} '
            f'{r.p_ttest_fam_fdr:9.4f} {r.pearson_r:6.2f} '
            f'{r.p_cluster_min:9.4f} {r.p_tfce_min:9.4f} '
            f'{int(r.n_sig_windows):6d}{flag}')
    ok = summary[summary.note == '']
    lines += ['',
              f'  {int((ok.p_ttest < a.alpha).sum())}/{len(ok)} edge x band '
              f'differ on the task mean (paired t, uncorrected); '
              f'{int(ok.sig_ttest_fam_fdr.sum())} after FDR',
              f'  {int(ok.sig_cluster_fam_fdr.sum())}/{len(ok)} differ by '
              f'cluster-mass after FDR; '
              f'{int(ok.sig_tfce_fam_fdr.sum())}/{len(ok)} by TFCE',
              f'  median |A/B| ratio {np.nanmedian(np.abs(ok.ratio_a_over_b)):.3f}, '
              f'median across-subject r {np.nanmedian(ok.pearson_r):.3f}',
              '  (* p<alpha uncorrected, ** survives FDR across the family)']
    text = '\n'.join(lines)
    print('\n' + text)
    with open(os.path.join(out_dir, 'summary.log'), 'w') as fh:
        fh.write(text + '\n')
    print(f'\n-> {out_dir}')


if __name__ == '__main__':
    main()
