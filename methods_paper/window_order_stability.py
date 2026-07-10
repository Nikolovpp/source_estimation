#!/usr/bin/env python3
"""Window-length / model-order stability of the sensor-vs-source GC washout.

Open lever #3 from the baseline-washout handoff: is the flat, task-invariant
LCMV source GC an artifact of an *under-resolved* moving-window MVAR (40 ms
window / order 10 — the window is far shorter than a theta cycle), or a genuine
property of the beamformer-imposed coupling?

This recomputes the matched Temporal<->IFG GC from the SAME local virtual
channels used by pair_baseline_diagnose.py -- sensor mastoid pseudo-channels
(Temporal, Inferior_Frontal) and LCMV source ROIs (awfa-lh, ifc-lh) -- through
the SAME engine (moving_window_pairwise_gc), sweeping a grid of moving-window
durations and MVAR model orders.  For each cell it measures the group
task-vs-baseline contrast (low_beta) in both spaces and both directions.

Reading: if the SOURCE task-minus-baseline contrast stays ~0 (few subjects >0)
across the whole grid while the SENSOR contrast stays positive/consistent, the
washout is NOT a window/order resolution artifact -- it survives better-resolved
MVARs, reinforcing the beamformer-fixed-coupling mechanism.  If the source
contrast rises toward the sensor as windows lengthen, the washout is partly an
estimation artifact.

Each space keeps its NATIVE baseline (overtProd sensor -1.6:-1.5, source
-1.5:-1.4; perception both -0.2:-0.1); task window is shared per task.

Outputs (durable, under GC_sensor_vs_source_baseline_check/):
  {task}_window_order_stability.png   grid of group contrasts vs (win, order)
  {task}_window_order_stability.csv   per-cell group stats

Usage:
    conda activate mne
    python methods_paper/window_order_stability.py --task overtProd  --stim-class prodDiff
    python methods_paper/window_order_stability.py --task perception --stim-class percDiff
"""
import os, sys, glob, argparse, warnings
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DECODE_OUTPUT_ROOT
from granger import moving_window_pairwise_gc, band_average, DEFAULT_BANDS

ROOT = DECODE_OUTPUT_ROOT.parent / 'GC_sensor_vs_source_baseline_check'
SRC_RED = ROOT / 'reduced_ts'                 # source virtual channels
SEN_MAST = ROOT / 'sensor_mastoid_ts'         # mastoid sensor pseudo-channels
FS = 500.0
FREQS = np.arange(1, 31)
BAND = 'low_beta'                             # primary band (matches the suite)
SRC = ('awfa-lh', 'ifc-lh')                   # temporal, IFG

# Grid.  win_dur in ms -> win_samples at 500 Hz {20,30,40,60,80}.  Orders span
# under- to well-resolved MVARs; guard keeps order < win_samples.
WIN_MS = [40, 60, 80, 120, 160]
ORDERS = [5, 8, 10, 14]

# Per-task native baselines (seconds) + shared task window.
TASK_CFG = {
    'overtProd': dict(base_sen=(-1.60, -1.50), base_src=(-1.50, -1.40),
                      task_win=(-0.30, 0.30), onset='production onset'),
    'perception': dict(base_sen=(-0.20, -0.10), base_src=(-0.20, -0.10),
                       task_win=(0.0, 0.30), onset='auditory onset'),
}
COL = {'sensor': '#2166ac', 'source': '#b2182b'}
DIR_LS = {'fwd': '-', 'rev': '--'}            # fwd = Temporal->IFG, rev = IFG->Temporal


def load_pair(subj, task, stim, space):
    """Return (a, b) = (Temporal-like, IFG-like) channels, (n_epochs, n_times), + times."""
    if space == 'sensor':
        f = SEN_MAST / f'{subj}_{task}_{stim}.npz'
        if not f.exists():
            return None
        d = np.load(f, allow_pickle=True)
        return d['Temporal'].astype(float), d['Inferior_Frontal'].astype(float), d['times']
    f = SRC_RED / f'{subj}_{task}_{stim}.npz'
    if not f.exists():
        return None
    d = np.load(f, allow_pickle=True)
    return d[f'src__{SRC[0]}'].astype(float), d[f'src__{SRC[1]}'].astype(float), d['times']


def one_cell(subj, task, stim, space, win_ms, order, base, task_win):
    """One (subj, space, win, order) GC run -> baseline/task low_beta means, both dirs."""
    got = load_pair(subj, task, stim, space)
    if got is None:
        return None
    a, b, tvec = got
    n = min(a.shape[0], b.shape[0]); a, b = a[:n], b[:n]
    ws = int(round(win_ms / 1000.0 * FS))
    if ws <= order + 1 or ws > a.shape[1]:
        return None
    res = moving_window_pairwise_gc(np.stack([a, b], 1), order=order, freqs=FREQS,
                                    fs=FS, win_samples=ws, step=1)
    starts = np.arange(0, a.shape[1] - ws + 1)
    wm = tvec[starts] * 1000.0                                   # window-start times (ms)
    fxy = band_average(res['f_xy'], FREQS, DEFAULT_BANDS)[BAND]  # a->b  (Temporal->IFG)
    fyx = band_average(res['f_yx'], FREQS, DEFAULT_BANDS)[BAND]  # b->a  (IFG->Temporal)
    mb = (wm >= base[0] * 1000) & (wm <= base[1] * 1000)
    mt = (wm >= task_win[0] * 1000) & (wm <= task_win[1] * 1000)
    if not (mb.any() and mt.any()):
        return None
    out = dict(subj=subj, space=space, win_ms=win_ms, order=order)
    for tag, g in (('fwd', fxy), ('rev', fyx)):
        out[f'{tag}_base'] = float(g[mb].mean())
        out[f'{tag}_task'] = float(g[mt].mean())
        out[f'{tag}_contrast'] = float(g[mt].mean() - g[mb].mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', default='overtProd', choices=['overtProd', 'perception'])
    ap.add_argument('--stim-class', default='prodDiff')
    ap.add_argument('--n-jobs', type=int, default=32)
    args = ap.parse_args()
    cfg = TASK_CFG[args.task]

    subs = sorted(os.path.basename(f).split('_')[0]
                  for f in glob.glob(str(SRC_RED / f'*_{args.task}_{args.stim_class}.npz')))
    subs = [s for s in subs if (SEN_MAST / f'{s}_{args.task}_{args.stim_class}.npz').exists()]
    print(f'{args.task}/{args.stim_class}: {len(subs)} subjects | grid '
          f'{len(WIN_MS)} win x {len(ORDERS)} order x 2 spaces')
    print(f'  sensor baseline {cfg["base_sen"]} s | source baseline {cfg["base_src"]} s | '
          f'task {cfg["task_win"]} s ({cfg["onset"]})')

    jobs = []
    for space in ('sensor', 'source'):
        base = cfg['base_sen'] if space == 'sensor' else cfg['base_src']
        for wm in WIN_MS:
            for od in ORDERS:
                for s in subs:
                    jobs.append((s, space, wm, od, base))
    rows = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(one_cell)(s, args.task, args.stim_class, space, wm, od, base, cfg['task_win'])
        for (s, space, wm, od, base) in jobs)
    df = pd.DataFrame([r for r in rows if r is not None])

    # ── group aggregation per (space, win, order, direction) ──
    recs = []
    for (space, wm, od), g in df.groupby(['space', 'win_ms', 'order']):
        for tag in ('fwd', 'rev'):
            c = g[f'{tag}_contrast'].values
            t, p = stats.ttest_1samp(c, 0.0) if c.size > 1 else (np.nan, np.nan)
            recs.append(dict(
                space=space, win_ms=wm, order=od, direction=tag, n=c.size,
                base_gc=float(g[f'{tag}_base'].mean()), task_gc=float(g[f'{tag}_task'].mean()),
                contrast=float(c.mean()), n_pos=int((c > 0).sum()),
                t=float(t), p=float(p)))
    summ = pd.DataFrame(recs).sort_values(['space', 'direction', 'win_ms', 'order'])
    rep = ROOT / 'window_order_stability' / args.stim_class   # per-stim-class report dir
    rep.mkdir(parents=True, exist_ok=True)
    csv = rep / f'{args.task}_window_order_stability.csv'
    summ.to_csv(csv, index=False)

    # console: forward direction (Temporal->IFG), the headline pair
    print(f'\n=== {args.task}  Temporal->IFG (fwd) low_beta task-minus-baseline ===')
    print(f'{"space":7} {"win":>4} {"ord":>4} {"base":>7} {"task":>7} '
          f'{"t-b":>8} {"n>0":>5} {"t":>7} {"p":>8}')
    for _, r in summ[summ.direction == 'fwd'].iterrows():
        star = '*' if r.p < 0.05 else ' '
        print(f'{r.space:7} {int(r.win_ms):>4} {int(r.order):>4} {r.base_gc:>7.4f} '
              f'{r.task_gc:>7.4f} {r.contrast:>+8.4f} {r.n_pos:>3}/{r.n:<2} '
              f'{r.t:>7.2f} {r.p:>8.4f}{star}')

    # ── figure: contrast vs window, one panel per order; sensor vs source ──
    fig, axes = plt.subplots(1, len(ORDERS), figsize=(4.1 * len(ORDERS), 4.4),
                             sharey=True)
    for ax, od in zip(np.atleast_1d(axes), ORDERS):
        for space in ('sensor', 'source'):
            for tag in ('fwd', 'rev'):
                sub = summ[(summ.space == space) & (summ.order == od)
                           & (summ.direction == tag)].sort_values('win_ms')
                if sub.empty:
                    continue
                ax.plot(sub.win_ms, sub.contrast, DIR_LS[tag], color=COL[space],
                        marker='o', ms=4, lw=1.6,
                        label=f'{space} {"T->IFG" if tag=="fwd" else "IFG->T"}')
        ax.axhline(0, color='k', lw=0.7, alpha=0.5)
        ax.set_title(f'order {od}'); ax.set_xlabel('window (ms)')
        ax.grid(alpha=0.25)
    np.atleast_1d(axes)[0].set_ylabel('group task - baseline GC (low_beta)')
    np.atleast_1d(axes)[0].legend(fontsize=7, loc='best')
    fig.suptitle(f'{args.task}: sensor-vs-source GC task-baseline contrast across '
                 f'window/order\n(source flat across the grid => washout is not a '
                 f'resolution artifact)', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    png = rep / f'{args.task}_window_order_stability.png'
    fig.savefig(png, dpi=140); plt.close(fig)

    print(f'\nwrote {csv}\nwrote {png}')


if __name__ == '__main__':
    main()
