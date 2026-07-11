#!/usr/bin/env python3
"""Rule-of-thumb MVAR sweep: window length locked to 2x model order (SW = 2*MO).

Model order and GC window length are NOT independent.  The MVAR needs
win_samples > order, and the ratio order/win_samples ("saturation") drives
estimator variance and spectral smoothness.  A fixed-window order sweep
(gc_estimator_order_sweep: 40 ms window, order 5-16) and a fixed-order window
sweep (window_order_stability) each vary ONE axis but let saturation drift, so
their "isolated" main effects are still entangled with saturation.

This sweeps the DIAGONAL win_samples = 2*order -- the common "window = twice the
model order" rule of thumb -- which holds saturation CONSTANT (order/win = 0.5)
so only the overall temporal scale / spectral resolution changes.  At fs = 500 Hz
that is win_ms = 4*order (MO10 -> 40 ms, the pipeline default).  If the group
task-minus-baseline contrast is flat along this diagonal, the conclusion is
robust to the joint MO/SW choice, not just to either axis alone.

Reads the FIXPC1 reduced caches already built by gc_estimator_order_sweep
(estimator_order_sweep/reduced_fixpc1/...), so there is NO Stage A -- run
gc_estimator_order_sweep first if the caches are missing.  For each MO it runs
moving_window_pairwise_gc(order=MO, win_samples=ratio*MO, trgc=True) over all
custom pairs and subjects, band-averages, and plots the group contrast (plain GC
both directions + Diff-TRGC) vs MO, per band.

Outputs (under GC_sensor_vs_source_baseline_check/rule_of_thumb_mo_sw/); every
file carries {task}_{stim}{sfx} so distinct runs never overwrite:
  {task}_{stim}{sfx}_rule_of_thumb.csv        per (subj, pair, mo, band) raw
  {task}_{stim}{sfx}_rule_of_thumb_group.csv  group summary
  {task}_{stim}{sfx}_rule_of_thumb.png        contrast vs MO (SW=ratio*MO), per band

Usage:
    conda activate mne
    python methods_paper/rule_of_thumb_mo_sw.py --task overtProd --stim-class prodDiff percDiff
    python methods_paper/rule_of_thumb_mo_sw.py --task perception --stim-class percDiff
    # screening step (5x faster, contrast means stable): --gc-step 5
"""
import os, sys, argparse, warnings
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DECODE_OUTPUT_ROOT, SUBJECT_IDS
from granger import moving_window_pairwise_gc, band_average, DEFAULT_BANDS
from gc_estimator_order_sweep import load_reduced, default_pairs, TASK_CFG, FREQS, BANDS

OUT = DECODE_OUTPUT_ROOT.parent / 'GC_sensor_vs_source_baseline_check' / 'rule_of_thumb_mo_sw'
OUT.mkdir(parents=True, exist_ok=True)
DEFAULT_MO = [5, 8, 10, 12, 14, 16, 20]
EST = [('fwd', 'plain GC A->B'), ('rev', 'plain GC B->A'), ('trgc', 'TRGC net A->B')]
ECOL = {'fwd': '#1b7837', 'rev': '#7fbf7b', 'trgc': '#b2182b'}


def win(t, lo, hi):
    return (t >= lo) & (t <= hi)


def gc_cell(subj, task, stim, atlas, leakage, pair, mo, ratio, step, base, task_win):
    """One (subj, pair, MO) GC with win_samples = ratio*MO -> band contrasts."""
    got = load_reduced(subj, task, stim, atlas, leakage)
    if got is None:
        return None
    vcs, tvec, fs = got
    ra, rb = pair.split(':')
    if ra not in vcs or rb not in vcs:
        return None
    a, b = vcs[ra], vcs[rb]
    n = min(a.shape[0], b.shape[0]); a, b = a[:n], b[:n]
    ws = int(round(ratio * mo))                     # rule of thumb: SW = ratio*MO samples
    if ws <= mo + 1 or ws > a.shape[1]:
        return None
    res = moving_window_pairwise_gc(np.stack([a, b], 1), order=mo, freqs=FREQS, fs=fs,
                                    win_samples=ws, step=step, trgc=True)
    starts = np.arange(0, a.shape[1] - ws + 1, step)
    wm = tvec[starts] * 1000.0
    mb = win(wm, base[0] * 1000, base[1] * 1000)
    mt = win(wm, task_win[0] * 1000, task_win[1] * 1000)
    if not (mb.any() and mt.any()):
        return None
    ba = dict(fwd=band_average(res['f_xy'], FREQS, DEFAULT_BANDS),
              rev=band_average(res['f_yx'], FREQS, DEFAULT_BANDS),
              trgc=band_average(res['d_xy'], FREQS, DEFAULT_BANDS))
    rows = []
    for band in BANDS:
        row = dict(subj=subj, pair=pair, mo=mo, win_samples=ws,
                   win_ms=round(ws / fs * 1000, 1), band=band)
        for tag in ('fwd', 'rev', 'trgc'):
            g = ba[tag][band]
            row[f'{tag}_base'] = float(g[mb].mean())
            row[f'{tag}_task'] = float(g[mt].mean())
            row[f'{tag}_contrast'] = float(g[mt].mean() - g[mb].mean())
        rows.append(row)
    return rows


def curve(raw, band, tag):
    """Per-subject mean over pairs, then group t-test -> all-pairs contrast vs MO."""
    d = raw[raw.band == band]
    out = []
    for mo, g in d.groupby('mo'):
        ps = g.groupby('subj')[f'{tag}_contrast'].mean().values
        t, p = stats.ttest_1samp(ps, 0.0) if len(ps) > 1 else (np.nan, np.nan)
        out.append(dict(mo=int(mo), win_ms=float(g.win_ms.iloc[0]),
                        mean=float(ps.mean()), sem=float(ps.std() / np.sqrt(max(len(ps), 1))),
                        npos=int((ps > 0).sum()), n=len(ps), t=float(t), p=float(p)))
    return pd.DataFrame(out).sort_values('mo')


def group_summary(raw):
    recs = []
    for (mo, band), g in raw.groupby(['mo', 'band']):
        rec = dict(mo=int(mo), win_ms=float(g.win_ms.iloc[0]), band=band, n_rows=len(g))
        for tag in ('fwd', 'rev', 'trgc'):
            ps = g.groupby('subj')[f'{tag}_contrast'].mean().values
            t, p = stats.ttest_1samp(ps, 0.0) if len(ps) > 1 else (np.nan, np.nan)
            rec[f'{tag}_contrast'] = float(ps.mean())
            rec[f'{tag}_sem'] = float(ps.std() / np.sqrt(max(len(ps), 1)))
            rec[f'{tag}_npos'] = int((ps > 0).sum()); rec[f'{tag}_n'] = len(ps)
            rec[f'{tag}_t'] = float(t); rec[f'{tag}_p'] = float(p)
        recs.append(rec)
    return pd.DataFrame(recs).sort_values(['band', 'mo'])


def figure(raw, task, stem, ratio, outdir):
    fig, axes = plt.subplots(1, len(BANDS), figsize=(4.3 * len(BANDS), 4.6), sharex=True)
    for ax, band in zip(np.atleast_1d(axes), BANDS):
        for tag, title in EST:
            c = curve(raw, band, tag)
            ax.errorbar(c.mo, c['mean'], yerr=c['sem'], marker='o', ms=5, lw=1.8,
                        capsize=3, color=ECOL[tag], label=title)
            for _, r in c.iterrows():
                if r.p < 0.05:
                    ax.annotate('*', (r.mo, r['mean']), color=ECOL[tag], fontsize=13,
                                ha='center', va='bottom')
        ax.axhline(0, color='k', lw=0.7)
        ax.set_title(band, fontsize=11); ax.grid(alpha=0.25)
        c0 = curve(raw, band, 'fwd')                 # shared x tick labels: MO + SW(ms)
        ax.set_xticks(c0.mo)
        ax.set_xticklabels([f'{int(m)}\n{w:.0f}ms' for m, w in zip(c0.mo, c0.win_ms)], fontsize=8)
        ax.set_xlabel('model order (MO)\nwindow = %g*MO' % ratio)
    np.atleast_1d(axes)[0].set_ylabel('group task - baseline GC')
    np.atleast_1d(axes)[0].legend(fontsize=8, loc='best')
    fig.suptitle(f'{task}: task-baseline GC along the SW = {ratio:g}*MO rule-of-thumb diagonal '
                 f'(saturation held at 1/{ratio:g})\nflat line => conclusion robust to the joint '
                 f'MO/window choice  (* p<0.05)', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    f = outdir / f'{stem}_rule_of_thumb.png'; fig.savefig(f, dpi=140); plt.close(fig)
    return f


def run_one(args, stim):
    cfg = TASK_CFG[args.task]
    sfx = '' if args.leakage else '_raw'
    stem = f'{args.task}_{stim}{sfx}'
    print(f'{args.task}/{stim}: MO={args.mo} ratio(SW/MO)={args.ratio} '
          f'-> SW(ms)={[round(args.ratio*m/args.target_fs*1000,0) for m in args.mo]} '
          f'| base {cfg["base"]}s task {cfg["task"]}s | step {args.gc_step}')
    # pairs from the first resolvable reduced cache
    got = None; sub0 = None
    for s in args.subjects:
        got = load_reduced(s, args.task, stim, args.atlas, args.leakage)
        if got is not None:
            sub0 = s; break
    if got is None:
        print('  No reduced FIXPC1 caches found — run gc_estimator_order_sweep first.'); return
    rois = list(got[0].keys())
    if args.rois:
        rois = [r for r in rois if r in set(args.rois)]
    pairs = args.pairs if args.pairs else default_pairs(rois)
    print(f'  ROIs: {rois}\n  {len(pairs)} pairs (first cache: {sub0})')

    stem_out = OUT / f'{stem}_rule_of_thumb.csv'
    print(f'  reports -> {OUT}/  ({stem}_rule_of_thumb.csv/_group.csv/.png)')
    cells = Parallel(n_jobs=args.gc_jobs, verbose=5)(
        delayed(gc_cell)(s, args.task, stim, args.atlas, args.leakage, p, mo, args.ratio,
                         args.gc_step, cfg['base'], cfg['task'])
        for s in args.subjects for p in pairs for mo in args.mo)
    rows = [r for cell in cells if cell for r in cell]
    if not rows:
        print('  No GC rows produced — check reduced caches.'); return
    raw = pd.DataFrame(rows)
    raw.to_csv(stem_out, index=False)
    summ = group_summary(raw)
    summ.to_csv(OUT / f'{stem}_rule_of_thumb_group.csv', index=False)

    # console: low_beta + high_beta all-pairs curve
    for band in ('low_beta', 'high_beta'):
        print(f'\n=== {args.task}/{stim}  {band}  all-pairs task-minus-baseline along SW={args.ratio:g}*MO ===')
        print(f'{"MO":>3} {"SWms":>5} | ' + ' | '.join(f'{t:>20}' for t, _ in EST))
        curves = {tag: curve(raw, band, tag) for tag, _ in EST}
        for mo in args.mo:
            row = curves['fwd'][curves['fwd'].mo == mo]
            if row.empty:
                continue
            wms = row.iloc[0].win_ms
            cs = []
            for tag, _ in EST:
                r = curves[tag][curves[tag].mo == mo].iloc[0]
                star = '*' if r.p < 0.05 else ' '
                cs.append(f'{r["mean"]:+.4f} ({int(r.npos):2d}/{int(r.n)}) p={r.p:.3f}{star}')
            print(f'{mo:>3} {wms:>5.0f} | ' + ' | '.join(cs))

    f = figure(raw, args.task, stem, args.ratio, OUT)
    print(f'\nwrote {stem_out}\nwrote {OUT / f"{stem}_rule_of_thumb_group.csv"}\nwrote {f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', default='overtProd', choices=['overtProd', 'perception'])
    ap.add_argument('--stim-class', nargs='+', default=['prodDiff'],
                    help='one or more stim-classes; each writes its own task_stimclass reports')
    ap.add_argument('--atlas', default='custom')
    ap.add_argument('--subjects', nargs='+', default=list(SUBJECT_IDS))
    ap.add_argument('--pairs', nargs='+', default=None)
    ap.add_argument('--rois', nargs='+', default=None)
    ap.add_argument('--mo', nargs='+', type=int, default=DEFAULT_MO,
                    help='model orders; window = ratio*MO samples for each')
    ap.add_argument('--ratio', type=float, default=2.0,
                    help='window/order ratio (rule of thumb = 2, i.e. SW samples = 2*MO)')
    ap.add_argument('--target-fs', type=float, default=500.0,
                    help='sampling rate of the reduced caches (for the SWms print only)')
    ap.add_argument('--gc-step', type=int, default=1)
    ap.add_argument('--no-leakage', dest='leakage', action='store_false')
    ap.set_defaults(leakage=True)
    ap.add_argument('--gc-jobs', type=int, default=None,
                    help='parallel workers (default: all physical cores)')
    args = ap.parse_args()
    if args.gc_jobs is None:
        args.gc_jobs = len(os.sched_getaffinity(0))
    for i, stim in enumerate(args.stim_class):
        if len(args.stim_class) > 1:
            print('\n' + '#' * 70
                  + f'\n# stim-class {i + 1}/{len(args.stim_class)}: {args.task}/{stim}\n'
                  + '#' * 70)
        run_one(args, stim)


if __name__ == '__main__':
    main()
