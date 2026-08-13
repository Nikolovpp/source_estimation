#!/usr/bin/env python
"""Order-dependence of conditional spectral GC — LCMV, GC_routes config C.

WHAT THIS IS. Config C sweeps MVAR order at two fixed window lengths (60 ms
orders 2/4/6/8; 120 ms orders 4/8/12). The measure is ``m1``: conditional
spectral GC on 6 ROIs, so every directed edge is conditioned on the OTHER FOUR.
That is not bivariate GC and not triple-wise A->B|C — those scopes come from
run_gc_window_order_sweep.sh, which writes to GC_source_space.

WHY 60 AND 120 ARE PLOTTED SEPARATELY. They are two different experiments with
different order ranges. Sharing one x-axis implies a crossed grid that was
never run, which is exactly the confound the new sweep exists to remove.

    conda activate mne
    python plot_lcmv_order_sweep.py
"""
import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = ('/mnt/r/phd_thesis/Research/SpeechProduction/EEG/derivatives/'
     'source_estimation/GC_routes')
OUT = f'{R}/_figures'
os.makedirs(OUT, exist_ok=True)

BANDS = ['delta', 'theta', 'alpha', 'low_beta', 'high_beta']
SHOW = ['theta', 'low_beta', 'high_beta']        # the bands of interest
COL = {'theta': '#1b7837', 'low_beta': '#2166ac', 'high_beta': '#b2182b'}

SWEEPS = {
    60:  ('C_ordersweep_win60',  [2, 4, 6, 8]),
    120: ('C_ordersweep_win120', [4, 8, 12]),
}


def sh(r):
    return r.replace('-lh', '')


def load(cfg_dir, order, win):
    """-> (edges, {edge: (n_subj, n_bands, n_win)}) for both estimators."""
    d = (f'{R}/overtProd/LCMV/custom/{cfg_dir}/'
         f'win{win}ms_order{order}_fs200_pc1/prodDiff')
    files = sorted(glob.glob(f'{d}/*.npz'))
    if not files:
        return None, None, None
    par, ss = {}, {}
    bands = None
    for f in files:
        z = np.load(f, allow_pickle=True)
        if bands is None:
            bands = [str(b) for b in z['bands']]
        for k in z.files:
            if k.startswith('m1_par__'):
                e = tuple(k.split('__')[1:])
                par.setdefault(e, []).append(z[k])
            elif k.startswith('m1_ss__'):
                e = tuple(k.split('__')[1:])
                ss.setdefault(e, []).append(z[k])
    par = {e: np.stack(v) for e, v in par.items()}
    ss = {e: np.stack(v) for e, v in ss.items()}
    return par, ss, bands


def collect():
    """{win: {order: (par, ss, bands)}} — loaded once, reused by both figures."""
    out = {}
    for win, (cfg, orders) in SWEEPS.items():
        out[win] = {}
        for o in orders:
            par, ss, bands = load(cfg, o, win)
            if par:
                out[win][o] = (par, ss, bands)
                n = len(next(iter(par.values())))
                print(f'  win{win} order{o}: {len(par)} edges, {n} subjects')
    return out


def fig_matrix(data, win, estimator='ss'):
    """6x6 edge matrix: band-mean GC vs model order, one line per band.

    Mean over windows THEN over subjects, with the SEM across subjects. NaN
    windows (ill-conditioned fits) are skipped by nanmean rather than
    propagating, and the count is reported so silent loss is visible.
    """
    orders = sorted(data[win])
    if not orders:
        return
    edges = sorted(next(iter(data[win].values()))[0])
    rois = sorted({r for e in edges for r in e})
    n = len(rois)

    fig, axes = plt.subplots(n, n, figsize=(2.05 * n, 1.85 * n),
                             sharex=True, squeeze=False)
    ymax = 0.0
    n_nan = 0
    for i, src in enumerate(rois):
        for j, tgt in enumerate(rois):
            ax = axes[i][j]
            if src == tgt:
                ax.set_facecolor('#f0f0f0')
                ax.set_xticks([]); ax.set_yticks([])
                continue
            for bd in SHOW:
                bi = BANDS.index(bd)
                m, s = [], []
                for o in orders:
                    par, ss, _ = data[win][o]
                    A = (ss if estimator == 'ss' else par).get((src, tgt))
                    if A is None:
                        m.append(np.nan); s.append(np.nan); continue
                    n_nan += int(np.isnan(A[:, bi, :]).sum())
                    per_subj = np.nanmean(A[:, bi, :], axis=1)
                    m.append(np.nanmean(per_subj))
                    s.append(np.nanstd(per_subj) / np.sqrt(len(per_subj)))
                m, s = np.array(m), np.array(s)
                ax.errorbar(orders, m, yerr=s, color=COL[bd], lw=1.6,
                            marker='o', ms=3.5, capsize=2, label=bd)
                if np.isfinite(m).any():
                    ymax = max(ymax, np.nanmax(m + s))
            ax.axhline(0, color='0.6', lw=0.6, zorder=0)
            if i == 0:
                ax.set_title(sh(tgt), fontsize=9, weight='bold')
            if j == 0:
                ax.set_ylabel(sh(src), fontsize=9, weight='bold')
            if i == n - 1:
                ax.set_xlabel('order', fontsize=8)
            ax.tick_params(labelsize=7)

    for row in axes:
        for ax in row:
            if ax.get_facecolor()[:3] != (0.0, 0.0, 0.0):
                ax.set_ylim(0, ymax * 1.08 if ymax > 0 else 1)

    axes[0][1].legend(fontsize=7, loc='upper right', frameon=False)
    name = {'ss': 'state-space', 'par': 'parametric'}[estimator]
    fig.suptitle(
        f'Conditional spectral GC vs model order — {win} ms window, LCMV, '
        f'{name}\nrow = source, column = target; each edge conditioned on the '
        f'other 4 ROIs; mean±SEM over 20 subjects',
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    p = f'{OUT}/lcmv_ordersweep_win{win}_{estimator}_matrix.png'
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f'wrote {os.path.basename(p)}  (NaN cells encountered: {n_nan})')


def fig_summary(data):
    """Two panels: how much order moves the answer, and whether the two
    estimators track each other as order changes."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    # (a) band-mean GC vs order, averaged over all 30 edges, both windows.
    ax = axes[0]
    for win, ls in ((60, '-'), (120, '--')):
        orders = sorted(data[win])
        for bd in SHOW:
            bi = BANDS.index(bd)
            m = []
            for o in orders:
                _, ss, _ = data[win][o]
                vals = [np.nanmean(A[:, bi, :]) for A in ss.values()]
                m.append(np.nanmean(vals))
            ax.plot(orders, m, ls, color=COL[bd], marker='o', ms=4,
                    label=f'{bd}, {win} ms')
    ax.set_xlabel('MVAR model order')
    ax.set_ylabel('conditional GC, mean over 30 edges')
    # Do NOT soften this title. At 60 ms the ranking inverts: theta leads at
    # order 2 (0.020 vs high-beta 0.001), high beta leads at order 6 (0.040 vs
    # theta 0.012). Which band you would call dominant is set by the order.
    ax.set_title('(a) The dominant band INVERTS with order (60 ms)')
    ax.legend(fontsize=8, frameon=False, ncol=2)

    # (b) parametric vs state-space agreement. These SHOULD diverge: the
    # reduced model here is ARMA, so the parametric route fits it at an order
    # known to be wrong (see GC_fundamentals/reduced_model_explained.md).
    ax = axes[1]
    for win, ls in ((60, '-'), (120, '--')):
        orders = sorted(data[win])
        rel = []
        for o in orders:
            par, ss, _ = data[win][o]
            num, den = [], []
            for e in ss:
                if e in par:
                    num.append(np.nanmean(np.abs(ss[e] - par[e])))
                    den.append(np.nanmean(np.abs(ss[e])))
            rel.append(100 * np.nansum(num) / np.nansum(den))
        ax.plot(orders, rel, ls, color='0.2', marker='s', ms=4,
                label=f'{win} ms window')
    ax.set_xlabel('MVAR model order')
    ax.set_ylabel('|state-space − parametric| / |state-space|  (%)')
    ax.set_title('(b) Estimator disagreement peaks at the canonical order 6')
    ax.legend(fontsize=9, frameon=False, loc='best')

    fig.suptitle('LCMV order sweep, conditional spectral GC (each edge '
                 'conditioned on the other 4 ROIs), overtProd/prodDiff, n=20',
                 fontsize=11, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = f'{OUT}/lcmv_ordersweep_summary.png'
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f'wrote {os.path.basename(p)}')


if __name__ == '__main__':
    print('loading LCMV config C ...')
    data = collect()
    for win in SWEEPS:
        if data.get(win):
            fig_matrix(data, win, 'ss')
            fig_matrix(data, win, 'par')
    fig_summary(data)
    print(f'\nfigures in {OUT}')
