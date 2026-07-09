#!/usr/bin/env python3
"""Single-subject matched sensor-vs-source view, per task (Temporal<->IFG).

2x2:  sensor GC (Temporal<->IFG) | source GC (awfa<->ifc)   [top]
      sensor trial-mean TCs       | source trial-mean TCs     [bottom]

Uses the mastoid sensor pseudo-channels (sensor_mastoid_ts/) and the source
virtual channels (reduced_ts/) for one subject — the same caches as
pair_baseline_diagnose.py.  Output: {task}_baseline_compare.png.

Usage:
    python methods_paper/baseline_compare.py --task perception --stim-class percDiff --subject EEGPROD4003
"""
import os, sys, argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DECODE_OUTPUT_ROOT
from granger import moving_window_pairwise_gc, band_average, DEFAULT_BANDS

ROOT = DECODE_OUTPUT_ROOT.parent / 'GC_sensor_vs_source_baseline_check'
SRC_RED = ROOT / 'reduced_ts'; SEN_MAST = ROOT / 'sensor_mastoid_ts'
FS = 500.0; FREQS = np.arange(1, 31); BAND = 'low_beta'
SRC = ('awfa-lh', 'ifc-lh')
CFG = {'overtProd': dict(base=(-1500, -1400), onset='production onset'),
       'perception': dict(base=(-200, -100), onset='auditory onset')}


def gc(a, b):
    ws = int(round(0.04 * FS))
    res = moving_window_pairwise_gc(np.stack([a, b], 1), order=10, freqs=FREQS,
                                    fs=FS, win_samples=ws, step=1)
    return (band_average(res['f_xy'], FREQS, DEFAULT_BANDS)[BAND],
            band_average(res['f_yx'], FREQS, DEFAULT_BANDS)[BAND], ws)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', default='perception', choices=['overtProd', 'perception'])
    ap.add_argument('--stim-class', default='percDiff')
    ap.add_argument('--subject', default='EEGPROD4003')
    args = ap.parse_args()
    cfg = CFG[args.task]; base = cfg['base']
    tag = f'{args.subject}_{args.task}_{args.stim_class}'
    dm = np.load(SEN_MAST / f'{tag}.npz', allow_pickle=True)
    ds = np.load(SRC_RED / f'{tag}.npz', allow_pickle=True)
    senT, senI, st = dm['Temporal'].astype(float), dm['Inferior_Frontal'].astype(float), dm['times'] * 1000
    srcA, srcB, srt = ds[f'src__{SRC[0]}'].astype(float), ds[f'src__{SRC[1]}'].astype(float), ds['times'] * 1000

    sxy, syx, ws = gc(senT, senI); wt_s = st[np.arange(senT.shape[1] - ws + 1)]
    rxy, ryx, _ = gc(srcA, srcB); wt_r = srt[np.arange(srcA.shape[1] - ws + 1)]

    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    ax[0, 0].plot(wt_s, sxy, label='Temporal→IFG', color='#2166ac')
    ax[0, 0].plot(wt_s, syx, label='IFG→Temporal', color='#b2182b')
    ax[0, 1].plot(wt_r, rxy, label='awfa→ifc', color='#2166ac')
    ax[0, 1].plot(wt_r, ryx, label='ifc→awfa', color='#b2182b')
    ymax = max(sxy.max(), syx.max(), rxy.max(), ryx.max()) * 1.05
    for j, ttl in [(0, 'SENSOR GC (low_beta)'), (1, 'SOURCE GC (low_beta)')]:
        ax[0, j].axvspan(base[0], base[1], color='0.6', alpha=.2); ax[0, j].axvline(0, color='k', lw=.6)
        ax[0, j].set_title(ttl); ax[0, j].legend(fontsize=8); ax[0, j].set_xlabel(f'window start (ms) [0={cfg["onset"]}]')
        ax[0, j].set_ylabel('GC'); ax[0, j].set_ylim(0, ymax)
    panels = [(0, st, senT.mean(0), senI.mean(0), 'Temporal', 'IFG', 'SENSOR'),
              (1, srt, srcA.mean(0), srcB.mean(0), 'awfa-lh', 'ifc-lh', 'SOURCE')]
    for j, t, mA, mB, laA, laB, sp in panels:
        ax[1, j].plot(t, mA / np.abs(mA).max(), label=laA, color='#2166ac')
        ax[1, j].plot(t, mB / np.abs(mB).max(), label=laB, color='#b2182b')
        ax[1, j].axvspan(base[0], base[1], color='0.6', alpha=.2); ax[1, j].axvline(0, color='k', lw=.6)
        ax[1, j].set_title(f'{sp} trial-mean TC (norm)'); ax[1, j].legend(fontsize=8); ax[1, j].set_xlabel('time (ms)')
    fig.suptitle(f'{args.subject} {args.task}: sensor vs source GC & time courses '
                 f'(baseline {base[0]}:{base[1]} shaded)', fontsize=13)
    fig.tight_layout(); out = ROOT / f'{args.task}_baseline_compare.png'
    fig.savefig(out, dpi=130, bbox_inches='tight'); plt.close(fig)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
