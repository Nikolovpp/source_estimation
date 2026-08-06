"""
Group statistics and plotting for source/sensor Granger causality.

Aggregates the per-subject GC ``.npz`` files written by ``run_granger.py``,
``run_granger_mne.py`` or ``run_granger_sensor.py``, computes the
subject-mean band-limited GC time courses with SEM, and tests task
against baseline.  TWO FAMILIES of test are run and both are written to
the same CSV, so they can be read side by side:

**Pointwise (per window)** — the MATLAB task-vs-baseline design.  At each
task window, a right-tailed one-sample test of the subjects' GC against
the *scalar* group-averaged baseline level.  ``--test ttest`` is the
parametric Student's t (matches ``production_pwgc_data_to_python.m`` and
the v4 figures); ``--test signrank`` is the non-parametric Wilcoxon
signed-rank (matches the v3 figures).  No correction across windows —
these are the raw per-window p-values.

**Cluster-based permutation (over the whole task span)** — the same test
used for the decoding results (``source_stats_viz.py``): a sign-flip
one-sample permutation test over the task-window axis via
``mne.stats.permutation_cluster_1samp_test``, run both in cluster-mass
mode and in TFCE mode.  This one CONTROLS the family-wise error rate
across windows, which the pointwise test does not, so it is the test to
report.  Disable with ``--no-permutation``.

Important difference between the two: the pointwise test compares each
subject to *the group's* baseline scalar (a one-sample test against a
constant that was itself estimated from the same subjects).  The
permutation test uses each subject's OWN baseline mean, i.e. it tests the
within-subject contrast ``GC(task window) - GC(baseline)``, which is what
makes the sign-flip null valid.  The permutation result is therefore the
better-founded of the two; the pointwise one is retained for continuity
with the MATLAB figures.

On top of the within-edge correction the permutation p-values are also
corrected across the whole (edge x band) family — Bonferroni and
Benjamini-Hochberg FDR — in the ``*_fam_bonf`` / ``*_fam_fdr`` columns.

CLI
---
    # point it at ONE directory of subject .npz — everything else is inferred
    python granger_stats.py --gc-dir <dir with subject .npz>

    # or derive the directory from a run's parameters
    python granger_stats.py --space source --task overtProd --stim-class prodDiff \\
        --method dSPM --atlas HCPMMP1 --feature-mode vertex_selectkbest \\
        --order 10 --win-ms 40 --target-fs 500
"""
import os
import re
import sys
import glob
import argparse

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DECODE_OUTPUT_ROOT, GC_TASK_END
from granger import DEFAULT_BANDS
from run_granger import gc_tag, roiset_tag, GC_OUTPUT_ROOT

GC_SENSOR_OUTPUT_ROOT = DECODE_OUTPUT_ROOT.parent / 'GC_sensor_space'

# Matches source_stats_viz.py so GC and decoding are tested identically.
N_PERMUTATIONS = 1024
TFCE_THRESHOLD = dict(start=0, step=0.2)


def infer_task_from_path(gc_dir):
    """Return 'overtProd'/'perception' if either appears in ``gc_dir``, else None.

    The task only affects the default task-end crop (``config.GC_TASK_END``),
    but getting it wrong silently changes which windows are tested, so it is
    read off the output path rather than left to the caller to remember.
    """
    parts = re.split(r'[\\/]+', str(gc_dir))
    for t in ('overtProd', 'perception'):
        if t in parts:
            return t
    return None


# ─────────────────────────────────────────────────────────────────────
# Loading / aggregation
# ─────────────────────────────────────────────────────────────────────
def load_gc_group(gc_dir, bands=None):
    """Load every subject ``.npz`` in ``gc_dir`` into stacked arrays.

    Returns a dict with ``roi_names``, ``pair_i``, ``pair_j``,
    ``window_ms``, ``subjects``, and ``fxy``/``fyx`` (and ``dtrgc`` if
    present) each a dict {band: (n_subj, n_pairs, n_win)}.
    """
    if bands is None:
        bands = DEFAULT_BANDS
    band_names = list(bands)
    files = sorted(glob.glob(os.path.join(str(gc_dir), '*.npz')))
    if not files:
        raise FileNotFoundError(f'No GC .npz files in {gc_dir}')

    subjects, ref = [], None
    fxy = {b: [] for b in band_names}
    fyx = {b: [] for b in band_names}
    has_trgc = None
    dtr = {b: [] for b in band_names}
    for f in files:
        d = np.load(f, allow_pickle=True)
        subj = os.path.basename(f).split('_')[0]
        subjects.append(subj)
        if ref is None:
            ref = {
                'roi_names': list(d['roi_names']),
                'pair_i': d['pair_i'], 'pair_j': d['pair_j'],
                'window_ms': d['window_ms'],
            }
        else:
            # Every subject must agree on WHICH pair each row is, or the stack
            # silently averages different edges together. Pairs are re-resolved
            # per subject in run_granger_mne.process_subject and any a subject
            # lacks are dropped, so two subjects can end up the same SHAPE with
            # different CONTENT — np.stack would accept that without complaint.
            ref_pairs = [(str(ref['roi_names'][i]), str(ref['roi_names'][j]))
                         for i, j in zip(ref['pair_i'], ref['pair_j'])]
            this_names = list(map(str, d['roi_names']))
            this_pairs = [(this_names[i], this_names[j])
                          for i, j in zip(d['pair_i'], d['pair_j'])]
            if this_pairs != ref_pairs:
                raise ValueError(
                    f'{subj} has a different pair set/order than '
                    f'{subjects[0]} in {gc_dir}.\n'
                    f'  {subjects[0]}: {ref_pairs}\n'
                    f'  {subj}: {this_pairs}\n'
                    'Stacking these would average different directed edges '
                    'across subjects. Re-run the affected subject(s), or drop '
                    'them via --subjects.')
            if d['window_ms'].shape != ref['window_ms'].shape:
                raise ValueError(
                    f'{subj} has {d["window_ms"].size} windows but '
                    f'{subjects[0]} has {ref["window_ms"].size} in {gc_dir}.')
        if has_trgc is None:
            has_trgc = f'dtrgc_{band_names[0]}' in d
        for b in band_names:
            fxy[b].append(d[f'fxy_{b}'])
            fyx[b].append(d[f'fyx_{b}'])
            if has_trgc:
                dtr[b].append(d[f'dtrgc_{b}'])
        d.close()

    out = dict(ref)
    out['subjects'] = subjects
    out['fxy'] = {b: np.stack(fxy[b]) for b in band_names}
    out['fyx'] = {b: np.stack(fyx[b]) for b in band_names}
    if has_trgc:
        out['dtrgc'] = {b: np.stack(dtr[b]) for b in band_names}
    return out


# ─────────────────────────────────────────────────────────────────────
# Task-vs-baseline statistics
#   ttest    -> right-tailed one-sample Student's t (production_pwgc_data_to_python.m, v4 figs)
#   signrank -> right-tailed Wilcoxon signed-rank   (v3 figs)
# ─────────────────────────────────────────────────────────────────────
def _right_tailed_pval(x, m, test):
    """Right-tailed one-sample p-value of samples ``x`` against scalar ``m``.

    ``ttest`` uses the parametric one-sample Student's t (scipy
    ``ttest_1samp(..., alternative='greater')``).  ``signrank`` uses the
    non-parametric Wilcoxon signed-rank on ``x - m`` (scipy
    ``wilcoxon(..., alternative='greater')`` == MATLAB
    ``signrank(x, m, 'tail','right')``).  Returns NaN when the statistic
    is undefined (e.g. all differences are zero).
    """
    if test == 'ttest':
        _t, p = stats.ttest_1samp(x, m, alternative='greater')
        return p
    if test == 'signrank':
        d = np.asarray(x, dtype=float) - m
        if not np.any(d != 0.0):
            return np.nan
        try:
            _w, p = stats.wilcoxon(d, alternative='greater')
        except ValueError:
            return np.nan
        return p
    raise ValueError(f"unknown test {test!r} (expected 'ttest' or 'signrank')")


def task_vs_baseline(subj_stack, window_ms, baseline_ms, task_start_ms,
                     alpha=0.05, test='ttest', task_end_ms=None):
    """Per-pair subject mean/SEM and right-tailed task-vs-baseline test.

    subj_stack : (n_subj, n_pairs, n_win)
    baseline_ms : (lo, hi) window-start range treated as baseline.
    task_start_ms : task points are windows with start >= this.
    task_end_ms : if given, task points are also capped at start <= this
        (drops the trailing edge windows); None = no upper cap.
    test : 'ttest' (parametric Student's t) or 'signrank' (non-parametric
        Wilcoxon signed-rank).  Both are right-tailed, one-sample, tested
        against the scalar subject-averaged baseline mean.

    Returns dict of (n_pairs, n_win) arrays: ``mean``, ``sem``,
    ``pval`` (NaN outside task), ``sig`` (bool), and scalar-per-pair
    ``baseline_mean`` (n_pairs,).
    """
    n_subj, n_pairs, n_win = subj_stack.shape
    subj_mean = subj_stack.mean(axis=0)
    sem = subj_stack.std(axis=0, ddof=1) / np.sqrt(n_subj)

    base_mask = (window_ms >= baseline_ms[0]) & (window_ms <= baseline_ms[1])
    if not base_mask.any():
        base_mask = np.zeros(n_win, bool); base_mask[0] = True
    baseline_mean = subj_mean[:, base_mask].mean(axis=1)      # (n_pairs,)

    task_mask = window_ms >= task_start_ms
    if task_end_ms is not None:
        task_mask &= window_ms <= task_end_ms
    pval = np.full((n_pairs, n_win), np.nan)
    sig = np.zeros((n_pairs, n_win), bool)
    for pi in range(n_pairs):
        for w in np.where(task_mask)[0]:
            p = _right_tailed_pval(subj_stack[:, pi, w], baseline_mean[pi], test)
            pval[pi, w] = p
            sig[pi, w] = p < alpha
    return {'mean': subj_mean, 'sem': sem, 'pval': pval, 'sig': sig,
            'baseline_mean': baseline_mean}


# ─────────────────────────────────────────────────────────────────────
# Cluster-based permutation task-vs-baseline
#   Same estimator as source_stats_viz.py uses for the decoding curves:
#   mne.stats.permutation_cluster_1samp_test, sign-flip null, tail=1.
# ─────────────────────────────────────────────────────────────────────
def permutation_task_vs_baseline(subj_stack, window_ms, baseline_ms,
                                 task_start_ms, alpha=0.05, task_end_ms=None,
                                 n_permutations=N_PERMUTATIONS, tfce=True,
                                 seed=42, n_jobs=1):
    """Sign-flip cluster permutation of task-vs-baseline, per pair.

    subj_stack : (n_subj, n_pairs, n_win)

    Unlike ``task_vs_baseline`` (which tests every subject against ONE
    group-level baseline scalar), each subject is here referenced to its
    OWN baseline mean::

        X[s, w] = GC[s, pair, w] - mean_over_baseline_windows(GC[s, pair, :])

    That makes the contrast a within-subject difference, which is exactly
    the exchangeability the sign-flip null assumes.  ``tail=1`` keeps the
    right-tailed "task > baseline" direction of the pointwise test.

    Runs twice: cluster-mass (``threshold=None``, i.e. MNE's default
    parametric cluster-forming threshold) and TFCE.  Cluster p-values are
    broadcast onto every window belonging to that cluster, so the returned
    arrays line up window-for-window with ``task_vs_baseline`` output.

    Returns dict of (n_pairs, n_win) arrays — ``p_cluster``, ``sig_cluster``,
    ``cluster_id`` (1-based, 0 = not in any cluster), ``tfce_score``,
    ``p_tfce``, ``sig_tfce`` — plus (n_pairs,) ``p_cluster_min`` /
    ``p_tfce_min``, the smallest p-value found for that pair (the natural
    per-edge summary).  Windows outside the task span are NaN / False / 0.
    """
    from mne.stats import permutation_cluster_1samp_test

    n_subj, n_pairs, n_win = subj_stack.shape
    base_mask = (window_ms >= baseline_ms[0]) & (window_ms <= baseline_ms[1])
    if not base_mask.any():
        base_mask = np.zeros(n_win, bool); base_mask[0] = True
    task_mask = window_ms >= task_start_ms
    if task_end_ms is not None:
        task_mask &= window_ms <= task_end_ms
    task_idx = np.where(task_mask)[0]

    out = {
        'p_cluster': np.full((n_pairs, n_win), np.nan),
        'sig_cluster': np.zeros((n_pairs, n_win), bool),
        'cluster_id': np.zeros((n_pairs, n_win), int),
        'tfce_score': np.full((n_pairs, n_win), np.nan),
        'p_tfce': np.full((n_pairs, n_win), np.nan),
        'sig_tfce': np.zeros((n_pairs, n_win), bool),
        'p_cluster_min': np.full(n_pairs, np.nan),
        'p_tfce_min': np.full(n_pairs, np.nan),
    }
    if task_idx.size < 2:
        return out

    for pi in range(n_pairs):
        # within-subject contrast over the task span
        base = subj_stack[:, pi, base_mask].mean(axis=1)          # (n_subj,)
        X = subj_stack[:, pi, task_idx] - base[:, None]           # (n_subj, n_task)
        if not np.isfinite(X).all() or np.allclose(X, 0):
            # an all-NaN band (e.g. theta dropped by MNE's frequency floor)
            # or a degenerate constant — leave this pair as NaN
            continue

        _T, clusters, cl_p, _ = permutation_cluster_1samp_test(
            X, threshold=None, n_permutations=n_permutations, tail=1,
            out_type='mask', seed=seed, verbose=False)
        for ci, (cl, p) in enumerate(zip(clusters, cl_p), start=1):
            pts = task_idx[cluster_to_mask(cl, task_idx.size)]
            out['p_cluster'][pi, pts] = p
            out['cluster_id'][pi, pts] = ci
            if p < alpha:
                out['sig_cluster'][pi, pts] = True
        if len(cl_p):
            out['p_cluster_min'][pi] = float(np.min(cl_p))

        if tfce:
            T_tfce, _cl, p_tfce, _ = permutation_cluster_1samp_test(
                X, threshold=TFCE_THRESHOLD, n_permutations=n_permutations,
                tail=1, out_type='mask', seed=seed, n_jobs=n_jobs, verbose=False)
            out['tfce_score'][pi, task_idx] = T_tfce
            out['p_tfce'][pi, task_idx] = p_tfce
            out['sig_tfce'][pi, task_idx] = p_tfce < alpha
            out['p_tfce_min'][pi] = float(np.min(p_tfce))
    return out


def cluster_to_mask(cluster, n):
    """Normalise one entry of ``permutation_cluster_1samp_test``'s cluster list.

    With ``out_type='mask'`` and 1-D data MNE hands back a ``(slice,)`` tuple
    rather than a boolean array.  Indexing happens to work either way for 1-D,
    but ``.sum()``/``~`` do not, so convert once, here.
    """
    m = np.zeros(n, bool)
    m[cluster] = True
    return m


def bh_fdr(pvals):
    """Benjamini-Hochberg adjusted p-values; NaNs pass through as NaN."""
    p = np.asarray(pvals, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return out
    v = p[ok]
    order = np.argsort(v)
    n = v.size
    adj = v[order] * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]     # enforce monotonicity
    res = np.empty(n)
    res[order] = np.minimum(adj, 1.0)
    out[ok] = res
    return out


# ─────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────
def _contiguous_spans(mask):
    """[(start_idx, end_idx), ...] of each run of True in a 1-D bool array."""
    mask = np.asarray(mask, bool)
    if not mask.any():
        return []
    d = np.diff(mask.astype(int))
    starts = list(np.where(d == 1)[0] + 1)
    ends = list(np.where(d == -1)[0])
    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        ends.append(mask.size - 1)
    return list(zip(starts, ends))


def plot_directed_edge(agg, stats_by_band, src_name, tgt_name, pair_idx,
                       direction, out_path, bands=None, fmt='png', test='ttest',
                       baseline_ms=None, task_start_ms=None, task_end_ms=None,
                       perm_by_band=None):
    """Plot one directed edge (src->tgt) across bands with significance.

    direction : 'fxy' (pair i->j) or 'fyx' (pair j->i).
    test : which task-vs-baseline test produced ``sig`` (named in the title).
    baseline_ms, task_start_ms, task_end_ms : shade the GC baseline window and
        mark the task-window start.  The plot is also RESTRICTED to the analysed
        span — from the baseline start through the task-end crop — so the leading
        pre-baseline segment and the trailing cropped windows (both excluded from
        the stats) are not drawn either.
    perm_by_band : optional {band: permutation_task_vs_baseline(...) dict}.  When
        given, windows inside a significant permutation cluster are shaded, so
        the FWER-controlled result is visually separable from the uncorrected
        per-window ticks.
    """
    if bands is None:
        bands = DEFAULT_BANDS
    band_names = list(bands)
    window_ms = agg['window_ms']
    # Only show what the stats use: baseline start .. task end.  Everything
    # outside (leading pre-baseline windows, trailing MVAR-boundary windows
    # dropped by the task-end crop) is hidden.
    lo = baseline_ms[0] if baseline_ms is not None else float(window_ms[0])
    hi = task_end_ms if task_end_ms is not None else float(window_ms[-1])
    keep = (window_ms >= lo) & (window_ms <= hi)
    wm = window_ms[keep]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.ravel()
    for ax, b in zip(axes, band_names):
        st = stats_by_band[b]
        m = st['mean'][pair_idx][keep]
        se = st['sem'][pair_idx][keep]
        ax.plot(wm, m, color='#2166ac', lw=2, label=f'{src_name}→{tgt_name}')
        ax.fill_between(wm, m - se, m + se, color='#2166ac', alpha=0.25)
        ax.axhline(st['baseline_mean'][pair_idx], color='0.5', ls='--', lw=1,
                   label='baseline')
        if baseline_ms is not None:
            ax.axvspan(baseline_ms[0], baseline_ms[1], color='0.6', alpha=0.15,
                       lw=0, label='baseline window')
        if task_start_ms is not None:
            ax.axvline(task_start_ms, color='0.4', ls=':', lw=1)
        # FWER-controlled permutation clusters, drawn UNDER the curve
        title_extra = ''
        if perm_by_band is not None and b in perm_by_band:
            pm = perm_by_band[b]
            spans = _contiguous_spans(pm['sig_cluster'][pair_idx][keep])
            for k, (s, e) in enumerate(spans):
                ax.axvspan(wm[s], wm[e], color='#b2182b', alpha=0.12, lw=0,
                           label='sig. cluster' if k == 0 else None)
            pmin = pm['p_cluster_min'][pair_idx]
            if np.isfinite(pmin):
                title_extra = f'  [cluster p={pmin:.3g}]'
        # significance ticks (task windows only; already within the kept span)
        sig = st['sig'][pair_idx][keep]
        if sig.any():
            ytop = np.nanmax(m + se)
            ax.plot(wm[sig], np.full(int(sig.sum()), ytop * 1.05), 's',
                    color='#b2182b', ms=3)
        ax.set_title(f'{b} ({bands[b][0]:g}–{bands[b][1]:g} Hz){title_extra}',
                     fontsize=11)
        ax.axvline(0, color='k', lw=0.8, alpha=0.5)
        ax.set_xlim(lo, hi)
        ax.set_ylabel('GC')
    for ax in axes[2:]:
        ax.set_xlabel('window start (ms)')
    test_label = {'ttest': "Student's t",
                  'signrank': 'Wilcoxon signed-rank'}.get(test, test)
    sig_note = f'ticks: right-tailed {test_label} (uncorrected)'
    if perm_by_band is not None:
        sig_note += '   |   shading: permutation cluster p<0.05 (FWER)'
    fig.suptitle(f'Granger causality: {src_name} → {tgt_name}\n{sig_note}',
                 fontsize=13)
    axes[0].legend(fontsize=9, loc='upper left')
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, format=fmt, bbox_inches='tight')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────
def run_stats(gc_dir, task, out_dir, baseline_ms=None, task_start_ms=None,
              alpha=0.05, bands=None, fmt='png', test='ttest', task_end_ms=None,
              baseline_dur_ms=100.0, edge_guard_ms=0.0, permutation=True,
              n_permutations=N_PERMUTATIONS, tfce=True, seed=42, n_jobs=1):
    """Aggregate a GC group directory, run stats, write figures + CSV.

    ``test`` selects the POINTWISE task-vs-baseline test ('ttest' or
    'signrank').  Figure and CSV names are tagged with it, so both tests
    can be written into the same ``out_dir`` and diffed edge-by-edge.

    ``permutation`` additionally runs the sign-flip cluster permutation
    test (cluster-mass + TFCE) over the task span, per edge and band, and
    corrects the resulting p-values across the whole (edge x band) family.
    That is the FWER-controlled result; the pointwise columns are not
    corrected across windows at all.

    The baseline is the epoch's ACTUAL pre-stimulus baseline period — the
    leading ``baseline_dur_ms`` (100 ms) of the moving-window axis — derived
    from the data itself, so it is correct whatever the epoch length: the -1.6 s
    sensor run -> [-1600, -1500] ms, the -1.5 s source run -> [-1500, -1400] ms.
    A LOW baseline is expected and is the signal (a rest / silent period has low
    directed connectivity), so the baseline is shown in full and NOT trimmed by
    default (earlier this was a hardcoded interior window shifted 50 ms off the
    epoch start; that was wrong — it cut into the genuine baseline).

    ``edge_guard_ms`` (default 0) optionally drops the very first moving
    window(s): the single first window is a computational edge point (an
    isolated one-window drop at the epoch start — verified at -1500 ms in
    overtProd AND -200 ms in perception, jumping straight back to the interior
    level, i.e. not the sustained baseline low), but it is one window in a
    100 ms average (negligible), so it is kept by default.  Set e.g.
    ``--edge-guard 4`` to drop it.  The trailing MVAR-boundary windows (a sharp
    spike in the last ~2 windows) are dropped separately by the task-end crop
    (config.GC_TASK_END).  Override any of this with --baseline-start/
    --baseline-end / --task-start / --task-end / --edge-guard.
    """
    if bands is None:
        bands = DEFAULT_BANDS
    band_names = list(bands)
    agg = load_gc_group(gc_dir, bands)
    window_ms = agg['window_ms']
    if baseline_ms is None:
        # leading baseline = [epoch start + edge guard, epoch start + duration]:
        # the true baseline period minus the contaminated boundary window(s).
        baseline_ms = (float(window_ms[0]) + edge_guard_ms,
                       float(window_ms[0]) + baseline_dur_ms)
    if task_start_ms is None:
        task_start_ms = baseline_ms[1]           # task begins where baseline ends
    if task_end_ms is None and task in GC_TASK_END:
        task_end_ms = GC_TASK_END[task] * 1000.0
    end_str = f'{task_end_ms:g}' if task_end_ms is not None else 'end'
    print(f'  GC baseline window: [{baseline_ms[0]:g}, {baseline_ms[1]:g}] ms '
          f'(epoch leading {baseline_dur_ms:g} ms); '
          f'task windows [{task_start_ms:g}, {end_str}] ms')
    # The default baseline is the LEADING part of the moving-window axis, which
    # is only a real pre-stimulus baseline if that axis starts before 0.  The
    # MNE cwt runs on the short perception crop start at about -45 ms, so the
    # default "baseline" is almost entirely POST-stimulus and every
    # task-vs-baseline p-value below would be meaningless.
    if baseline_ms[1] > 0:
        frac_post = ((min(baseline_ms[1], float(window_ms[-1])) - max(baseline_ms[0], 0.0))
                     / (baseline_ms[1] - baseline_ms[0]))
        print(f'  *** WARNING: the baseline window extends past t=0 '
              f'({100 * max(0.0, frac_post):.0f}% of it is post-stimulus). '
              f'The moving-window axis starts at {window_ms[0]:g} ms, so this '
              f'run has no usable pre-stimulus baseline.\n'
              f'  *** Task-vs-baseline results here compare the task to ITSELF. '
              f'Either re-run with a longer crop, or pass an explicit '
              f'--baseline-start/--baseline-end, or use compare_gc_conditions.py '
              f'to contrast two conditions instead of testing against baseline.')

    roi = agg['roi_names']
    n_pairs = len(agg['pair_i'])
    os.makedirs(out_dir, exist_ok=True)

    if permutation:
        print(f'  permutation: {n_permutations} sign-flips, cluster-mass'
              f'{" + TFCE" if tfce else ""}, per edge x band '
              f'({2 * n_pairs * len(band_names)} tests)')

    rows = []
    for direction, key in [('fxy', 'fxy'), ('fyx', 'fyx')]:
        stats_by_band = {
            b: task_vs_baseline(agg[key][b], agg['window_ms'], baseline_ms,
                                task_start_ms, alpha, test, task_end_ms)
            for b in band_names
        }
        perm_by_band = None
        if permutation:
            perm_by_band = {
                b: permutation_task_vs_baseline(
                    agg[key][b], agg['window_ms'], baseline_ms, task_start_ms,
                    alpha, task_end_ms, n_permutations, tfce, seed, n_jobs)
                for b in band_names
            }
        for pi in range(n_pairs):
            i, j = int(agg['pair_i'][pi]), int(agg['pair_j'][pi])
            if direction == 'fxy':
                src, tgt = roi[i], roi[j]
            else:
                src, tgt = roi[j], roi[i]
            fname = os.path.join(out_dir, f'GC_{src}_to_{tgt}_{test}.{fmt}')
            plot_directed_edge(agg, stats_by_band, src, tgt, pi, direction,
                               fname, bands, fmt, test, baseline_ms,
                               task_start_ms, task_end_ms, perm_by_band)
            for b in band_names:
                st = stats_by_band[b]
                pm = perm_by_band[b] if perm_by_band is not None else None
                for w, wm in enumerate(agg['window_ms']):
                    row = {
                        'src': src, 'tgt': tgt, 'band': b, 'window_ms': wm,
                        'test': test,
                        'gc_mean': st['mean'][pi, w], 'gc_sem': st['sem'][pi, w],
                        'baseline_mean': st['baseline_mean'][pi],
                        'pval': st['pval'][pi, w], 'sig': st['sig'][pi, w],
                    }
                    if pm is not None:
                        row.update({
                            'p_cluster': pm['p_cluster'][pi, w],
                            'sig_cluster': pm['sig_cluster'][pi, w],
                            'cluster_id': pm['cluster_id'][pi, w],
                            'tfce_score': pm['tfce_score'][pi, w],
                            'p_tfce': pm['p_tfce'][pi, w],
                            'sig_tfce': pm['sig_tfce'][pi, w],
                            'p_cluster_min': pm['p_cluster_min'][pi],
                            'p_tfce_min': pm['p_tfce_min'][pi],
                        })
                    rows.append(row)
    df = pd.DataFrame(rows)

    # ── correction across the (edge x band) family ──────────────────────
    # The cluster/TFCE p-values already control FWER across WINDOWS within
    # one edge x band; nothing yet controls the fact that we ran that test
    # once per edge and band.  Correct the per-edge minimum p-value over
    # that family and broadcast it back onto the edge's rows.
    if permutation:
        n_fam = 2 * n_pairs * len(band_names)
        keys = ['src', 'tgt', 'band']
        for col in ['p_cluster_min', 'p_tfce_min']:
            fam = df.drop_duplicates(keys)[keys + [col]].copy()
            fam[col + '_fam_bonf'] = np.minimum(fam[col] * n_fam, 1.0)
            fam[col + '_fam_fdr'] = bh_fdr(fam[col].to_numpy())
            df = df.merge(fam.drop(columns=[col]), on=keys, how='left')
        for base in ['p_cluster_min', 'p_tfce_min']:
            for kind in ['bonf', 'fdr']:
                df[f'sig_{base[:-4]}_fam_{kind}'] = \
                    df[f'{base}_fam_{kind}'] < alpha

    csv_path = os.path.join(out_dir, f'gc_task_vs_baseline_stats_{test}.csv')
    df.to_csv(csv_path, index=False)

    n_edges = 2 * n_pairs
    print(f'  {len(agg["subjects"])} subjects, {n_edges} directed edges, '
          f'{len(band_names)} bands, test={test} -> {out_dir}')
    print(f'  figures: {n_edges} + stats CSV: {csv_path}')
    if permutation:
        _print_perm_summary(df, alpha, out_dir, test)
    return csv_path


def _print_perm_summary(df, alpha, out_dir, test):
    """One line per surviving edge x band, and the same to a .log file."""
    keys = ['src', 'tgt', 'band']
    edge = df.drop_duplicates(keys)[keys + [
        'p_cluster_min', 'p_cluster_min_fam_bonf', 'p_cluster_min_fam_fdr',
        'p_tfce_min', 'p_tfce_min_fam_fdr']].copy()
    edge = edge.sort_values('p_cluster_min')
    lines = ['Permutation task-vs-baseline (right-tailed, sign-flip)',
             f'  {len(edge)} edge x band tests; alpha={alpha}',
             '',
             f'{"edge":>22s} {"band":>10s} {"p_clust":>9s} {"bonf":>9s} '
             f'{"fdr":>9s} {"p_tfce":>9s} {"tfce_fdr":>9s}']
    n_raw = n_fdr = 0
    for _, r in edge.iterrows():
        if not np.isfinite(r.p_cluster_min):
            continue
        n_raw += int(r.p_cluster_min < alpha)
        n_fdr += int(r.p_cluster_min_fam_fdr < alpha)
        flag = '  **' if r.p_cluster_min_fam_fdr < alpha else \
               ('  *' if r.p_cluster_min < alpha else '')
        lines.append(f'{r.src + "->" + r.tgt:>22s} {r.band:>10s} '
                     f'{r.p_cluster_min:9.4f} {r.p_cluster_min_fam_bonf:9.4f} '
                     f'{r.p_cluster_min_fam_fdr:9.4f} {r.p_tfce_min:9.4f} '
                     f'{r.p_tfce_min_fam_fdr:9.4f}{flag}')
    lines += ['', f'  {n_raw} edge x band significant uncorrected, '
                  f'{n_fdr} after FDR across the family',
              '  (* uncorrected, ** survives FDR)']
    text = '\n'.join(lines)
    print('\n' + text)
    with open(os.path.join(out_dir, f'gc_permutation_summary_{test}.log'), 'w') as fh:
        fh.write(text + '\n')


def _derive_gc_dir(args):
    root = GC_SENSOR_OUTPUT_ROOT if args.space == 'sensor' else GC_OUTPUT_ROOT
    leakage_tag = 'leakage_corrected' if args.leakage_correction else 'raw'
    return (root / args.task / args.method / args.atlas / args.feature_mode
            / leakage_tag / gc_tag(args.order, args.win_ms, args.target_fs,
                                   args.normalize, args.gc_mode)
            / roiset_tag(args.roi_subset) / args.stim_class)


def parse_args():
    p = argparse.ArgumentParser(description='Group GC stats + plots')
    p.add_argument('--gc-dir', default=None,
                   help='Directory of subject GC .npz (overrides the derived '
                        'path). This is the recommended way to call the script: '
                        'point it at ONE results directory and --task is read '
                        'off the path.')
    p.add_argument('--out-dir', default=None,
                   help='Where to write figures/CSV (default: <gc-dir>/group_stats)')
    p.add_argument('--task', default=None, choices=['perception', 'overtProd'],
                   help='Only sets the default task-end crop '
                        '(config.GC_TASK_END). Inferred from --gc-dir when it '
                        'contains the task name; required otherwise.')
    p.add_argument('--alpha', type=float, default=0.05)
    p.add_argument('--baseline-start', type=float, default=None,
                   help="GC baseline window start (s); default is the epoch's "
                        'leading 100 ms (the true baseline period, derived from '
                        'the data). Pass with --baseline-end to override.')
    p.add_argument('--baseline-end', type=float, default=None,
                   help='GC baseline window end (s)')
    p.add_argument('--task-start', type=float, default=None,
                   help='GC task windows begin here (s); default is the end of '
                        'the baseline window')
    p.add_argument('--task-end', type=float, default=None,
                   help='GC task windows end here (s), dropping the trailing '
                        'edge; default from config.GC_TASK_END[task]. Pass a '
                        'value beyond the last window to disable the crop.')
    p.add_argument('--edge-guard', type=float, default=0.0,
                   help='ms trimmed off the leading baseline to drop the single '
                        'first moving window (an isolated computational edge '
                        'point). Default 0 (keep the full, genuine baseline); '
                        'set e.g. 4 to drop the first window.')
    p.add_argument('--test', default='ttest', choices=['ttest', 'signrank'],
                   help="task-vs-baseline test: 'ttest' (right-tailed one-sample "
                        "Student's t; matches production_pwgc_data_to_python.m and "
                        "the v4 figures) or 'signrank' (right-tailed Wilcoxon "
                        "signed-rank; matches the v3 figures). The permutation "
                        'test below runs alongside it regardless.')
    p.add_argument('--no-permutation', dest='permutation', action='store_false',
                   default=True,
                   help='Skip the sign-flip cluster permutation test (it is on '
                        'by default and is the FWER-controlled result)')
    p.add_argument('--n-permutations', type=int, default=N_PERMUTATIONS,
                   help=f'sign-flips per edge x band (default {N_PERMUTATIONS}, '
                        'same as the decoding stats)')
    p.add_argument('--no-tfce', dest='tfce', action='store_false', default=True,
                   help='Skip the TFCE pass (keeps cluster-mass only; ~2x faster)')
    p.add_argument('--seed', type=int, default=42,
                   help='permutation RNG seed, so reruns are reproducible')
    p.add_argument('--n-jobs', type=int, default=1,
                   help='parallel jobs inside the TFCE permutation')
    p.add_argument('--format', default='png', choices=['png', 'svg'])
    # For deriving --gc-dir from a run's parameters:
    p.add_argument('--space', default='source', choices=['source', 'sensor'])
    p.add_argument('--stim-class', default='prodDiff')
    p.add_argument('--method', default='dSPM')
    p.add_argument('--atlas', default='HCPMMP1')
    p.add_argument('--feature-mode', default='vertex_selectkbest')
    p.add_argument('--leakage-correction', action='store_true', default=False)
    p.add_argument('--order', type=int, default=10)
    p.add_argument('--win-ms', type=float, default=40.0)
    p.add_argument('--target-fs', type=float, default=500.0)
    p.add_argument('--normalize', default='demean',
                   help='matches run_granger.py --normalize (part of the path)')
    p.add_argument('--gc-mode', default='pairwise',
                   choices=['pairwise', 'conditional'])
    p.add_argument('--roi-subset', nargs='+', default=None, metavar='ROI',
                   help='Same subset passed to run_granger.py, so the derived '
                        'path points at that subset run (omit for the full run)')
    return p.parse_args()


def main():
    args = parse_args()
    if args.gc_dir:
        gc_dir = args.gc_dir
        task = args.task or infer_task_from_path(gc_dir)
        if task is None:
            raise SystemExit(
                f'Could not infer --task from {gc_dir!r} (no "overtProd" or '
                '"perception" path component). Pass --task explicitly.')
        if args.task and args.task != infer_task_from_path(gc_dir) \
                and infer_task_from_path(gc_dir) is not None:
            print(f'WARNING: --task {args.task} but the path says '
                  f'{infer_task_from_path(gc_dir)}; using {args.task}.')
    else:
        if args.task is None:
            raise SystemExit('--task is required when --gc-dir is not given.')
        gc_dir, task = str(_derive_gc_dir(args)), args.task
    out_dir = args.out_dir if args.out_dir else os.path.join(gc_dir, 'group_stats')
    baseline_ms = None
    if args.baseline_start is not None and args.baseline_end is not None:
        baseline_ms = (args.baseline_start * 1000.0, args.baseline_end * 1000.0)
    task_start_ms = args.task_start * 1000.0 if args.task_start is not None else None
    task_end_ms = args.task_end * 1000.0 if args.task_end is not None else None
    print(f'GC group stats (pointwise test={args.test}, '
          f'permutation={"on" if args.permutation else "off"})\n'
          f'  gc-dir: {gc_dir}\n  task:   {task}')
    run_stats(gc_dir, task, out_dir, baseline_ms=baseline_ms,
              task_start_ms=task_start_ms, alpha=args.alpha, fmt=args.format,
              test=args.test, task_end_ms=task_end_ms,
              edge_guard_ms=args.edge_guard, permutation=args.permutation,
              n_permutations=args.n_permutations, tfce=args.tfce,
              seed=args.seed, n_jobs=args.n_jobs)


if __name__ == '__main__':
    main()
