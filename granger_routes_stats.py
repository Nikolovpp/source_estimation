#!/usr/bin/env python3
"""Group statistics for the parametric-vs-state-space GC comparison.

Reads what ``run_granger_routes.py`` wrote and produces three things:

  1. an ARM CONTRAST   parametric vs state-space, paired across subjects, per
                       directed edge x band, on the conditional-spectral (M1),
                       time-domain (M2) and block (M3) measures
  2. a MEDIATION table M(a->c | b) = 1 - F_{a->c|b} / F_{a->c} per ordered
                       triple x band, with primary and exploratory hypotheses
                       corrected as separate families
  3. a MISSINGNESS report (see below), printed first and never optional

=============================================================================
WHY THE MISSINGNESS REPORT COMES FIRST
=============================================================================
This pipeline normally produces no NaN at all, so a NaN here is a signal, not
noise, and it is emphatically NOT missing at random.

``run_granger_routes.py`` marks a (window, edge) cell NaN when the estimator
cannot support the fit: a singular MVAR, an ill-conditioned DARE (scipy's
"generalized Schur form" ValueError), or a failed reduced-model regression.
Those failures concentrate exactly where samples-per-parameter is worst — high
order, short window, low trial count. So the surviving windows at a high order
are a BIASED subsample: the ones that happened to be well conditioned.

Averaging over survivors would therefore make high orders look better behaved
than they are, and could invert the order-sweep conclusion. Every statistic
below reports ``n_valid`` and ``nan_frac`` alongside the estimate, cells above
``--max-nan-frac`` (default 0.20) are excluded from group tests and listed, and
the arm contrast uses only windows where BOTH arms are finite — otherwise the
two arms would be compared on different subsets of the data.

If the report says 0% everywhere, which is the usual case, nothing here bites
and the numbers are the plain ones. Read it anyway; that is the point.
=============================================================================

    conda activate mne
    python granger_routes_stats.py --gc-dir <.../GC_routes/overtProd/dSPM/custom/A_canonical/win60ms_order6_fs200_pc1/prodDiff>
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

MEASURES = {'m1': 'conditional spectral', 'm2': 'pairwise time-domain',
            'm3': 'block (FIXPC-k)'}


# ─────────────────────────────────────────────────────────────────────
def bh_fdr(p):
    """Benjamini-Hochberg, monotonicity enforced; NaNs pass through as NaN."""
    p = np.asarray(p, float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return out
    q = p[ok]
    order = np.argsort(q)
    ranked = q[order] * q.size / np.arange(1, q.size + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    res = np.empty_like(ranked)
    res[order] = np.clip(ranked, 0, 1)
    out[ok] = res
    return out


def load_dir(gc_dir):
    """Return (subjects, bands, window_ms, {key: (n_subj, n_bands, n_win)})."""
    files = sorted(glob.glob(os.path.join(gc_dir, '*.npz')))
    if not files:
        raise SystemExit(f'no .npz under {gc_dir}')
    subjects, store, bands, win = [], defaultdict(list), None, None
    for f in files:
        z = np.load(f, allow_pickle=True)
        subjects.append(os.path.basename(f)[:11])
        if bands is None:
            bands = [str(b) for b in z['bands']]
            win = np.asarray(z['window_ms'], float)
        for k in z.files:
            if k.split('__')[0] in ('m1_par', 'm1_ss', 'm2_par', 'm2_ss',
                                    'm3_par', 'm3_ss', 'f_par', 'f_ss',
                                    'f_pair'):
                store[k].append(np.asarray(z[k], float))
    n_win = min(min(a.shape[-1] for a in v) for v in store.values())
    out = {k: np.stack([a[..., :n_win] for a in v]) for k, v in store.items()
           if len(v) == len(subjects)}
    dropped = [k for k, v in store.items() if len(v) != len(subjects)]
    if dropped:
        print(f'  note: {len(dropped)} keys absent from some subjects, skipped')
    return subjects, bands, win[:n_win], out


# ─────────────────────────────────────────────────────────────────────
def missingness(store, subjects, bands, span):
    """Per-cell NaN fraction. Printed before anything else, always."""
    rows = []
    for key, arr in store.items():
        pre, *rest = key.split('__')
        a = arr[:, :, span]
        for bi, b in enumerate(bands):
            cell = a[:, bi]
            rows.append(dict(key=key, measure=pre, edge='__'.join(rest),
                             band=b, n_subj=len(subjects),
                             nan_frac=float(np.isnan(cell).mean()),
                             subj_all_nan=int((np.isnan(cell).all(1)).sum())))
    return pd.DataFrame(rows)


def arm_contrast(store, bands, span, max_nan):
    """Parametric vs state-space, paired across subjects, per edge x band.

    Both arms are reduced to the SAME finite windows before averaging, so the
    contrast is never between two different subsets of the data.
    """
    rows = []
    for key in sorted(k for k in store if k.startswith(('m1_par', 'm2_par',
                                                        'm3_par'))):
        ss_key = key.replace('_par', '_ss')
        if ss_key not in store:
            continue
        par, ss = store[key][:, :, span], store[ss_key][:, :, span]
        meas, *rest = key.split('__')
        edge = '__'.join(rest)
        for bi, b in enumerate(bands):
            p_, s_ = par[:, bi], ss[:, bi]
            both = np.isfinite(p_) & np.isfinite(s_)   # pair the windows
            nan_frac = 1.0 - both.mean()
            with np.errstate(invalid='ignore'):
                pm = np.where(both.any(1), np.nanmean(np.where(both, p_, np.nan), 1), np.nan)
                sm = np.where(both.any(1), np.nanmean(np.where(both, s_, np.nan), 1), np.nan)
            ok = np.isfinite(pm) & np.isfinite(sm)
            n = int(ok.sum())
            row = dict(measure=meas.split('_')[0], edge=edge, band=b,
                       n_valid=n, nan_frac=nan_frac,
                       mean_par=float(np.nanmean(pm)) if n else np.nan,
                       mean_ss=float(np.nanmean(sm)) if n else np.nan)
            row['ratio_par_over_ss'] = (row['mean_par'] / row['mean_ss']
                                        if row['mean_ss'] else np.nan)
            if n >= 3 and nan_frac <= max_nan:
                d = pm[ok] - sm[ok]
                row['t'] = float(ttest_rel(pm[ok], sm[ok]).statistic)
                row['p'] = float(ttest_rel(pm[ok], sm[ok]).pvalue)
                row['dz'] = float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) else np.nan
                try:
                    row['p_wilcoxon'] = float(wilcoxon(pm[ok], sm[ok]).pvalue)
                except ValueError:
                    row['p_wilcoxon'] = np.nan
            else:
                row.update(t=np.nan, p=np.nan, dz=np.nan, p_wilcoxon=np.nan)
            row['excluded_high_nan'] = bool(nan_frac > max_nan)
            rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        for meas, sub in df.groupby('measure'):
            df.loc[sub.index, 'p_fdr'] = bh_fdr(sub['p'].to_numpy())
    return df


def mediation(store, bands, span, primaries, max_nan):
    """M(a->c | b) = 1 - F_{a->c|b} / F_{a->c}, per triple x band, per arm.

    M ~ 1 means b explains the influence away. Mediation and common-drive both
    do that, so a serial chain a->b->c additionally requires both legs to carry
    influence; ``chain_ok`` records whether the unconditional a->b and b->c are
    themselves above the median edge strength, and it is a necessary condition,
    not a test.
    """
    rows = []
    for arm in ('par', 'ss'):
        for key in sorted(k for k in store if k.startswith(f'f_{arm}__')):
            _, a, b, c = key.split('__')
            pair_key = f'f_pair__{a}__{c}'
            if pair_key not in store:
                continue
            cond = store[key][:, :, span]
            uncond = store[pair_key][:, :, span]
            for bi, bd in enumerate(bands):
                cd, un = cond[:, bi], uncond[:, bi]
                both = np.isfinite(cd) & np.isfinite(un) & (un > 0)
                nan_frac = 1.0 - both.mean()
                with np.errstate(invalid='ignore', divide='ignore'):
                    cm = np.where(both.any(1), np.nanmean(np.where(both, cd, np.nan), 1), np.nan)
                    um = np.where(both.any(1), np.nanmean(np.where(both, un, np.nan), 1), np.nan)
                m = 1.0 - cm / um
                ok = np.isfinite(m)
                n = int(ok.sum())
                row = dict(arm=arm, source=a, mediator=b, target=c, band=bd,
                           n_valid=n, nan_frac=nan_frac,
                           mean_uncond=float(np.nanmean(um)) if n else np.nan,
                           mean_cond=float(np.nanmean(cm)) if n else np.nan,
                           M=float(np.nanmean(m)) if n else np.nan,
                           primary=(a, b, c) in primaries)
                if n >= 3 and nan_frac <= max_nan:
                    # is the conditioned influence reliably below the raw one?
                    row['t_vs_zero'] = float(ttest_rel(um[ok], cm[ok]).statistic)
                    row['p'] = float(ttest_rel(um[ok], cm[ok]).pvalue)
                else:
                    row.update(t_vs_zero=np.nan, p=np.nan)
                row['excluded_high_nan'] = bool(nan_frac > max_nan)
                rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # primary and exploratory are corrected as SEPARATE families
    for arm in df['arm'].unique():
        for fam in (True, False):
            sel = (df['arm'] == arm) & (df['primary'] == fam)
            if sel.any():
                df.loc[sel, 'p_fdr'] = bh_fdr(df.loc[sel, 'p'].to_numpy())
    df['family'] = np.where(df['primary'], 'primary', 'exploratory')
    return df


# ─────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--gc-dir', required=True)
    p.add_argument('--out-dir', default=None)
    p.add_argument('--span-ms', type=float, nargs=2, default=None,
                   help='restrict to a window-centre range, e.g. --span-ms -500 280')
    p.add_argument('--max-nan-frac', type=float, default=0.20,
                   help='cells with a larger NaN fraction are excluded from tests')
    p.add_argument('--alpha', type=float, default=0.05)
    args = p.parse_args()

    out_dir = args.out_dir or args.gc_dir
    os.makedirs(out_dir, exist_ok=True)
    subjects, bands, win, store = load_dir(args.gc_dir)
    span = slice(None)
    if args.span_ms:
        idx = np.where((win >= args.span_ms[0]) & (win <= args.span_ms[1]))[0]
        span = slice(int(idx[0]), int(idx[-1]) + 1)

    print(f'{len(subjects)} subjects, {len(bands)} bands, {len(win)} windows '
          f'({win[0]:.0f}..{win[-1]:.0f} ms)')

    # ---- 1. missingness, always first ----
    miss = missingness(store, subjects, bands, span)
    worst = miss.sort_values('nan_frac', ascending=False)
    overall = float(miss['nan_frac'].mean())
    print('\n=== MISSINGNESS ===')
    print(f'overall NaN fraction: {overall:.4%}')
    if overall == 0:
        print('  no NaN cells — estimates below are the plain ones')
    else:
        print(f'  cells above --max-nan-frac ({args.max_nan_frac:.0%}): '
              f'{int((miss.nan_frac > args.max_nan_frac).sum())} of {len(miss)}')
        print('  NaN is NOT missing at random: it concentrates where '
              'samples-per-parameter is worst.')
        print('  worst cells:')
        for _, r in worst.head(8).iterrows():
            print(f'    {r.nan_frac:6.1%}  {r.measure:7s} {r.edge:28s} {r.band}')
    miss.to_csv(os.path.join(out_dir, 'gc_routes_missingness.csv'), index=False)

    # ---- 2. arm contrast ----
    arms = arm_contrast(store, bands, span, args.max_nan_frac)
    if not arms.empty:
        arms.to_csv(os.path.join(out_dir, 'gc_routes_arm_contrast.csv'), index=False)
        print('\n=== PARAMETRIC vs STATE-SPACE ===')
        for meas, sub in arms.groupby('measure'):
            sig = sub[sub.p_fdr < args.alpha]
            print(f'  {MEASURES.get(meas, meas):24s} '
                  f'{len(sig)}/{len(sub)} cells differ at FDR<{args.alpha}; '
                  f'median ratio par/ss = {sub.ratio_par_over_ss.median():.3f}'
                  + (f'; {int(sub.excluded_high_nan.sum())} excluded for NaN'
                     if sub.excluded_high_nan.any() else ''))

    # ---- 3. mediation ----
    try:
        from run_granger_routes import PRIMARY_TRIPLES
        primaries = set(PRIMARY_TRIPLES)
    except Exception:
        primaries = set()
    med = mediation(store, bands, span, primaries, args.max_nan_frac)
    if not med.empty:
        med.to_csv(os.path.join(out_dir, 'gc_routes_mediation.csv'), index=False)
        print('\n=== MEDIATION  M = 1 - F(a->c|b)/F(a->c) ===')
        for arm in sorted(med.arm.unique()):
            for fam in ('primary', 'exploratory'):
                sub = med[(med.arm == arm) & (med.family == fam)]
                if sub.empty:
                    continue
                sig = sub[(sub.p_fdr < args.alpha) & (sub.M > 0.5)]
                print(f'  [{arm}] {fam:12s} {len(sig)}/{len(sub)} cells with '
                      f'M>0.5 at FDR<{args.alpha}')
                for _, r in sig.sort_values('M', ascending=False).head(5).iterrows():
                    print(f'     M={r.M:5.2f}  {r.source} -> {r.target} '
                          f'| {r.mediator}   ({r.band}, p_fdr={r.p_fdr:.4f}, '
                          f'n={r.n_valid})')
    print(f'\nwrote CSVs to {out_dir}')


if __name__ == '__main__':
    main()
