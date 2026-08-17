#!/usr/bin/env python
"""Why the sweep runs at 200 Hz — the sampling-rate / model-order trade-off.

Model order is a DURATION, not a unitless measure of model richness: an
order-p fit spans p/fs seconds of history. Raising the sampling rate buys
samples and costs reach; lowering it does the reverse. This figure shows, on a
simulated interaction whose true lag is known, that reach is what matters and
that samples cannot substitute for it.

    (a) model memory p/fs against the cycle lengths of the reported bands —
        the arithmetic, with the 500 Hz and 200 Hz grids overlaid.
    (b) adjacent-sample correlation for a 0.1-30 Hz signal. At 500 Hz
        consecutive samples share ~95% of their variance, so the extra samples
        are near-duplicates while each lag still costs n^2 coefficients.
    (c) recovered directional ratio against model memory, from the simulation.
        The vertical line is the TRUE lag; every configuration to its left is
        trying to see an interaction it cannot reach. Configurations sharing a
        memory differ only in window length and land on one marker — which is
        itself the point: at fixed reach, the window barely matters.
    (d) the same runs as a bar chart, labelled by (fs, window, order), showing
        that doubling the 500 Hz window does not rescue an under-reaching model.

GROUND TRUTH. A VAR resonating at 22 Hz in x drives y 15 ms later, low-passed
at 30 Hz to match the study's passband, then decimated to each candidate rate
so both pipelines see the same underlying signal.

CAVEAT. One synthetic process with one lag, chosen to make the mechanism
visible. It demonstrates why reach matters; it does not establish the optimal
configuration for real data — that is what the window x order sweep was for.

    conda activate mne
    python plot_sampling_rate_tradeoff.py
"""
import os
import sys
import argparse
import numpy as np
from scipy.signal import butter, filtfilt, resample_poly
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from granger import (pairwise_spectral_gc, band_average, var_spectral_radius,
                     DEFAULT_BANDS)
from run_granger import GC_OUTPUT_ROOT

OUT = f'{GC_OUTPUT_ROOT}/_figures_methods'
FS_GEN, DUR, N_TRIALS = 1000, 2.0, 300
F_OSC, LAG_MS, POLE = 22.0, 15.0, 0.90
LOWPASS = 30.0                      # the study's 0.1-30 Hz passband
RATES = (500, 200)
WINDOWS = (40, 80)
ORDERS = (6, 10)
RCOL = {500: '#b5483c', 200: '#2f6f9f'}


def simulate(seed=1):
    """(n_trials, 2, n_times) at FS_GEN: x -> y at F_OSC with a LAG_MS lag."""
    rng = np.random.default_rng(seed)
    n = int(FS_GEN * DUR)
    lag = int(LAG_MS / 1000 * FS_GEN)
    c1 = 2 * POLE * np.cos(2 * np.pi * F_OSC / FS_GEN)
    c2 = -POLE ** 2
    X = np.zeros((N_TRIALS, 2, n))
    for r in range(N_TRIALS):
        d = np.zeros(n)
        e = rng.standard_normal(n)
        for s in range(2, n):
            d[s] = c1 * d[s - 1] + c2 * d[s - 2] + e[s]
        X[r, 0] = d + 1.5 * rng.standard_normal(n)
        X[r, 1] = 0.9 * np.roll(d, lag) + 1.5 * rng.standard_normal(n)
    b, a = butter(4, LOWPASS / (FS_GEN / 2), 'low')
    return filtfilt(b, a, X, axis=-1)


def run(X):
    """-> [(fs, win_ms, order, memory_ms, n_samp, xy, yx, ratio, rho)]"""
    freqs = np.arange(1.0, 31.0)
    rows = []
    for fs in RATES:
        Xr = resample_poly(X, fs, FS_GEN, axis=-1)
        for win_ms in WINDOWS:
            m = int(win_ms * fs / 1000)
            for order in ORDERS:
                mem = 1000.0 * order / fs
                if m <= order + 1:
                    rows.append((fs, win_ms, order, mem, m,
                                 np.nan, np.nan, np.nan, np.nan))
                    continue
                acc = []
                for st_ms in range(300, 1200, 60):
                    i0 = int(st_ms * fs / 1000)
                    seg = Xr[:, :, i0:i0 + m]
                    if seg.shape[2] < m:
                        continue
                    try:
                        fxy, fyx, A, _ = pairwise_spectral_gc(
                            seg, order, freqs, fs, return_model=True)
                        acc.append((band_average(fxy, freqs)['high_beta'],
                                    band_average(fyx, freqs)['high_beta'],
                                    var_spectral_radius(A)))
                    except Exception:
                        continue
                if not acc:
                    rows.append((fs, win_ms, order, mem, m,
                                 np.nan, np.nan, np.nan, np.nan))
                    continue
                a_ = np.array(acc)
                xy, yx, rho = a_[:, 0].mean(), a_[:, 1].mean(), a_[:, 2].mean()
                rows.append((fs, win_ms, order, mem, m, xy, yx,
                             xy / max(yx, 1e-9), rho))
    return rows


def empirical_redundancy(seed=0):
    """Adjacent-sample correlation under the REAL filter, not the ideal sinc."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(200000)
    b, a = butter(4, LOWPASS / (FS_GEN / 2), 'low')
    xf = filtfilt(b, a, x)
    out = {}
    for fs in (100, 150, 200, 300, 400, 500):
        xr = resample_poly(xf, fs, FS_GEN)
        out[fs] = float(np.corrcoef(xr[:-1], xr[1:])[0, 1])
    return out


def figure(rows, out_png):
    fig = plt.figure(figsize=(13.6, 9.4))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.26,
                          left=0.075, right=0.975, top=0.85, bottom=0.075)

    # ---- (a) memory vs the bands it has to span -------------------------
    ax = fig.add_subplot(gs[0, 0])
    band_col = {'theta': '#6b8e23', 'alpha': '#8a7fb5',
                'low_beta': '#4f9d9d', 'high_beta': '#c98a3a'}
    for name, (lo, hi) in DEFAULT_BANDS.items():
        ax.axhspan(1000.0 / hi, 1000.0 / lo, color=band_col[name], alpha=0.20,
                   lw=0)
        ax.text(11.35, np.sqrt((1000.0 / hi) * (1000.0 / lo)),
                name.replace('_', ' '), fontsize=8, va='center', ha='right',
                color=band_col[name], weight='bold')
    orders = np.array([2, 4, 6, 10])
    for fs in RATES:
        ax.plot(orders, 1000.0 * orders / fs, 'o-', color=RCOL[fs], lw=2,
                ms=6, label=f'{fs} Hz', zorder=3)
    # Linear x. Order is four small integers, so a log axis buys nothing and
    # leaks a stray minor tick label ("3x10^0") between the majors.
    ax.set_yscale('log')
    ax.set_xticks(orders)
    ax.set_yticks([5, 10, 25, 50, 100, 250])
    ax.set_yticklabels(['5', '10', '25', '50', '100', '250'])
    ax.set_xlim(0.7, 11.6)
    ax.set_xlabel('MVAR model order')
    ax.set_ylabel('model memory  $p/f_s$  (ms)')
    ax.set_title('(a) Order is a duration, and the duration\n'
                 'depends on the sampling rate', fontsize=11, loc='left')
    ax.legend(fontsize=9, frameon=False, loc='lower right')
    ax.grid(alpha=0.18, which='both', lw=0.5)

    # ---- (b) how redundant the extra samples are ------------------------
    ax = fig.add_subplot(gs[0, 1])
    fs_grid = np.linspace(70, 600, 400)
    ax.plot(fs_grid, np.sinc(2 * LOWPASS / fs_grid), color='0.35', lw=2,
            label='ideal brick-wall low-pass')
    # The analytic sinc assumes an ideal filter on white input; the study uses
    # a 4th-order Butterworth. Measure it rather than let the idealisation
    # stand in for the real thing — it turns out to UNDERSTATE the redundancy.
    emp = empirical_redundancy()
    ax.plot(list(emp), list(emp.values()), 's', color='0.15', ms=7, zorder=4,
            label=f'measured, {LOWPASS:g} Hz Butterworth')
    for fs in RATES:
        r = float(np.sinc(2 * LOWPASS / fs))
        ax.plot([fs], [r], 'o', color=RCOL[fs], ms=10, zorder=5)
        ax.annotate(f'{fs} Hz\nr = {r:.3f}', (fs, r),
                    textcoords='offset points', xytext=(0, -40),
                    fontsize=9, color=RCOL[fs], ha='center', weight='bold')
    ax.legend(fontsize=8, frameon=False, loc='lower right')
    ax.set_ylim(0, 1.12)
    ax.set_xlabel('sampling rate (Hz)')
    ax.set_ylabel('correlation between adjacent samples')
    ax.set_title(f'(b) A {LOWPASS:g} Hz-limited signal oversampled:\n'
                 'the extra samples are near-duplicates',
                 fontsize=11, loc='left')
    ax.grid(alpha=0.18, lw=0.5)

    ok = [r for r in rows if np.isfinite(r[7])]

    # ---- (c) recovered direction vs reach -------------------------------
    ax = fig.add_subplot(gs[1, 0])
    ax.axvspan(0, LAG_MS, color='#999', alpha=0.12, lw=0, zorder=0)
    ax.axvline(LAG_MS, color='#444', ls='--', lw=1.4, zorder=1)
    for fs in RATES:
        pts = sorted([r for r in ok if r[0] == fs], key=lambda r: r[3])
        # Configurations at the SAME memory differ only in window length, so
        # they land on the same x. Collapse them to one marker and say so in
        # the label rather than stacking unreadable annotations.
        by_mem = {}
        for pt in pts:
            by_mem.setdefault(pt[3], []).append(pt)
        xs = sorted(by_mem)
        ys = [np.mean([q[7] for q in by_mem[m_]]) for m_ in xs]
        ax.plot(xs, ys, 'o-', color=RCOL[fs], lw=2, ms=9,
                label=f'{fs} Hz', zorder=3)
        for m_, yv in zip(xs, ys):
            grp = by_mem[m_]
            wins = sorted({int(q[1]) for q in grp})
            txt = (f'order {grp[0][2]}\n'
                   + ('/'.join(str(w) for w in wins) + ' ms window'
                      if len(wins) > 1 else f'{wins[0]} ms window'))
            # Per-point placement. Keying the offset on the rate alone put
            # the 500 Hz order-10 label straight through the 200 Hz order-6
            # one; these two markers are close in both x and y.
            off, ha = {12: ((-9, -22), 'right'), 20: ((0, 14), 'center'),
                       30: ((9, -22), 'left'), 50: ((-8, 6), 'right')}.get(
                           int(round(m_)), ((9, 6), 'left'))
            ax.annotate(txt, (m_, yv), textcoords='offset points',
                        xytext=off, ha=ha, fontsize=8, color=RCOL[fs],
                        zorder=4)
    ax.set_yscale('log')
    ax.set_xlim(0, 62)
    ax.set_ylim(1.6, 1600)
    ax.text(LAG_MS - 0.8, 1000, f'true lag {LAG_MS:g} ms', fontsize=9,
            color='#444', ha='right', va='top', rotation=90)
    ax.text(LAG_MS / 2, 2.1, 'cannot\nreach it', fontsize=8.5, color='#777',
            ha='center', va='bottom', style='italic')
    ax.set_xlabel('model memory  $p/f_s$  (ms)')
    ax.set_ylabel('recovered  GC(x→y) / GC(y→x)')
    ax.set_title('(c) Reach decides the answer, and two window\n'
                 'lengths at the same reach give the same answer',
                 fontsize=11, loc='left')
    ax.legend(fontsize=9, frameon=False, loc='lower right')
    ax.grid(alpha=0.18, which='both', lw=0.5)

    # ---- (d) samples cannot substitute for reach ------------------------
    ax = fig.add_subplot(gs[1, 1])
    # Include the infeasible cell. Dropping it hides why the sweep's 40 ms
    # column stops at order 6.
    lab = [f'{r[0]}Hz  {r[1]:g}ms\norder {r[2]}  ({r[4]} samp)' for r in rows]
    y = np.arange(len(rows))
    for i, r in enumerate(rows):
        if not np.isfinite(r[7]):
            ax.barh(i, 1200, color='0.88', height=0.68, zorder=0)
            ax.text(1.35, i, 'INFEASIBLE — 8 samples cannot support order 10',
                    va='center', fontsize=8, color='#777', style='italic')
            continue
        ax.barh(i, r[7], color=RCOL[r[0]], height=0.68)
        ax.text(r[7] * 1.10, i, f'{r[7]:.0f}×   reach {r[3]:.0f} ms',
                va='center', fontsize=8.5,
                color='#333' if r[3] >= LAG_MS else '#b5483c',
                weight='normal' if r[3] >= LAG_MS else 'bold')
    ax.set_yticks(y); ax.set_yticklabels(lab, fontsize=8)
    ax.set_xscale('log'); ax.set_xlim(1, 1400)
    ax.invert_yaxis()
    ax.set_xlabel('recovered directional ratio  (log scale)')
    ax.set_title('(d) 40 samples at 12 ms reach lose to\n'
                 '16 samples at 50 ms', fontsize=11, loc='left')
    ax.grid(alpha=0.18, axis='x', which='both', lw=0.5)

    fig.suptitle(
        'Why the sweep runs at 200 Hz — sampling rate trades samples against '
        'temporal reach\n'
        f'simulated: x drives y at {F_OSC:g} Hz with a {LAG_MS:g} ms lag, '
        f'low-passed {LOWPASS:g} Hz, {N_TRIALS} trials, decimated to each rate '
        f'so both see the same signal',
        fontsize=12.5, y=0.965)
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--out', default=OUT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print(f'simulating: {F_OSC:g} Hz, {LAG_MS:g} ms lag, {N_TRIALS} trials, '
          f'low-pass {LOWPASS:g} Hz')
    rows = run(simulate(args.seed))

    print(f'\n{"fs":>5}{"win":>6}{"samp":>6}{"ord":>5}{"memory":>9}'
          f'{"x->y":>9}{"y->x":>9}{"ratio":>9}{"rho":>7}')
    for fs, win, order, mem, m, xy, yx, ratio, rho in rows:
        if not np.isfinite(ratio):
            print(f'{fs:>5}{win:>6}{m:>6}{order:>5}{mem:>7.0f}ms'
                  f'{"infeasible":>34}')
            continue
        print(f'{fs:>5}{win:>6}{m:>6}{order:>5}{mem:>7.0f}ms'
              f'{xy:>9.3f}{yx:>9.3f}{ratio:>9.1f}{rho:>7.3f}')

    p = f'{args.out}/sampling_rate_tradeoff.png'
    figure(rows, p)
    print(f'\nwrote {p}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
