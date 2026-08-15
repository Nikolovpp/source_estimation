#!/usr/bin/env python
"""Should the GC baseline be a fixed duration, or scale with the window?

A flat 100 ms baseline always holds the same NUMBER of moving windows (100/step
= 20 at a 5 ms step), but not the same amount of INDEPENDENT evidence: at a
1-sample step consecutive windows share (win-1)/win of their samples, so 100 ms
spans 2.5 independent window-lengths at 40 ms and only 1.25 at 80 ms. The
baseline is therefore a weaker estimate at the long windows exactly where GC is
least biased — a confound between the baseline and the sweep parameter.

This measures whether that matters, on the real data:

  (a) baseline LEVEL vs baseline duration, one line per window length. If the
      level drifts with duration, the baseline is picking up epoch structure
      rather than a resting level.
  (b) SEM of the baseline estimate vs duration. Where this stops falling,
      extra baseline is buying overlapping windows, not information.
  (c) significant task windows (pointwise t) vs baseline duration — the
      practical consequence of the choice.

    conda activate mne
    python plot_gc_baseline_choice.py
    python plot_gc_baseline_choice.py --order 10 --rois awfa-lh-tpc-lh
"""
import os
import sys
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import GC_TASK_END
from run_granger import GC_OUTPUT_ROOT
from granger_stats import load_gc_group, task_vs_baseline, TASK_ONSET_MS

D = str(GC_OUTPUT_ROOT)
OUT = f'{D}/_figures_baseline_choice'
BANDS = ['theta', 'low_beta', 'high_beta']
LABEL = {'theta': 'theta 4–8 Hz', 'low_beta': 'low beta 12–18 Hz',
         'high_beta': 'high beta 18–30 Hz'}
WCOL = {40: '#4575b4', 60: '#1a9850', 80: '#d73027'}
DURATIONS = [25.0, 50.0, 75.0, 100.0, 150.0, 200.0]


def sh(r):
    return str(r).replace('-lh', '')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', default='perception')
    ap.add_argument('--stim', default='percDiff')
    ap.add_argument('--rois', default='ifc-lh-tpc-lh')
    ap.add_argument('--order', type=int, default=6)
    ap.add_argument('--windows', type=float, nargs='+', default=[40, 60, 80])
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    task_end = GC_TASK_END.get(args.task)
    task_end = task_end * 1000.0 if task_end is not None else None
    onset = TASK_ONSET_MS.get(args.task)

    # {(win, band, direction): {dur: (level, sem, n_sig, n_tested, n_bwin)}}
    res = {}
    for win in args.windows:
        cfg = f'order{args.order}_win{win:g}ms_fs200_zscore'
        dirs = glob.glob(f'{D}/{args.task}/**/{cfg}/rois_{args.rois}/'
                         f'{args.stim}', recursive=True)
        if not dirs:
            print(f'  win{win:g}: no data'); continue
        agg = load_gc_group(dirs[0])
        w = np.asarray(agg['window_ms'], float)
        for band in BANDS:
            for key in ('fxy', 'fyx'):
                stack = agg[key][band]
                for dur in DURATIONS:
                    base = (float(w[0]), float(w[0]) + dur)
                    tstart = base[1] if onset is None else max(base[1], onset)
                    if tstart >= (task_end if task_end is not None else w[-1]):
                        continue
                    st = task_vs_baseline(stack, w, base, tstart,
                                          task_end_ms=task_end)
                    bmask = (w >= base[0]) & (w <= base[1])
                    # SEM of the baseline estimate ACROSS SUBJECTS, using each
                    # subject's own baseline mean — the quantity the test
                    # compares against.
                    per_subj = np.nanmean(stack[:, 0, bmask], axis=1)
                    sem = (np.nanstd(per_subj, ddof=1)
                           / np.sqrt(np.isfinite(per_subj).sum()))
                    res[(win, band, key, dur)] = (
                        float(st['baseline_mean'][0]), float(sem),
                        int(st['sig'][0].sum()),
                        int(np.isfinite(st['pval'][0]).sum()),
                        int(bmask.sum()))
        print(f'  win{win:g}: {len(agg["subjects"])} subjects, '
              f'{w.size} windows', flush=True)
    if not res:
        print('nothing to plot'); return 1

    roi = load_gc_group(dirs[0])['roi_names']
    lab = {'fxy': f'{sh(roi[0])}→{sh(roi[1])}',
           'fyx': f'{sh(roi[1])}→{sh(roi[0])}'}

    fig, axes = plt.subplots(3, len(BANDS), figsize=(4.4 * len(BANDS), 9.6),
                             squeeze=False)
    for c, band in enumerate(BANDS):
        for win in args.windows:
            ds = [d for d in DURATIONS if (win, band, 'fxy', d) in res]
            if not ds:
                continue
            col = WCOL.get(int(win), '0.3')
            lvl = [res[(win, band, 'fxy', d)][0] for d in ds]
            sem = [res[(win, band, 'fxy', d)][1] for d in ds]
            sig = [res[(win, band, 'fxy', d)][2]
                   + res[(win, band, 'fyx', d)][2] for d in ds]
            axes[0][c].errorbar(ds, lvl, yerr=sem, color=col, marker='o',
                                ms=4, lw=1.6, capsize=2, label=f'{win:g} ms')
            axes[1][c].plot(ds, sem, color=col, marker='o', ms=4, lw=1.6)
            axes[1][c].axvline(win, color=col, ls=':', lw=1, alpha=0.7)
            axes[2][c].plot(ds, sig, color=col, marker='o', ms=4, lw=1.6)
        for r_ in range(3):
            axes[r_][c].tick_params(labelsize=8)
            axes[r_][c].set_xlabel('baseline duration (ms)', fontsize=8)
        axes[0][c].set_title(LABEL[band], fontsize=10, weight='bold')
    axes[0][0].set_ylabel(f'baseline level, {lab["fxy"]}\n(mean ± SEM over '
                          f'subjects)', fontsize=9)
    axes[1][0].set_ylabel('SEM of the baseline estimate\n(dotted line = one '
                          'window length)', fontsize=9)
    axes[2][0].set_ylabel('significant task windows\n(pointwise t, both '
                          'directions)', fontsize=9)
    axes[0][-1].legend(fontsize=8, frameon=False, title='GC window',
                       title_fontsize=8)
    fig.suptitle(
        f'Baseline duration — {args.task}/{args.stim}, {sh(roi[0])}↔'
        f'{sh(roi[1])}, order {args.order}, zscore, n=20\n'
        f'a flat duration holds the same NUMBER of windows at every GC window '
        f'length, but not the same independent evidence:\n'
        f'100 ms spans 2.5 window-lengths at 40 ms and 1.25 at 80 ms',
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    p = f'{OUT}/baseline_duration_{args.task}_{args.stim}_' \
        f'{args.rois}_order{args.order}.png'
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f'\nwrote {p}')

    print(f'\n{"win":>5}{"dur":>7}{"n_bwin":>8}{"level":>10}{"SEM":>9}'
          f'{"sig(fxy)":>10}')
    for win in args.windows:
        for d in DURATIONS:
            k = (win, 'theta', 'fxy', d)
            if k in res:
                lv, se, sg, nt, nb = res[k]
                print(f'{win:>5.0f}{d:>7.0f}{nb:>8}{lv:>10.4f}{se:>9.4f}'
                      f'{str(sg)+"/"+str(nt):>10}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
