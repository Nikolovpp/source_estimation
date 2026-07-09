#!/usr/bin/env python3
"""Group-level sensor-vs-source GC overview (all speech pairs), per task.

Four panels, low_beta:
  A  sensor group-mean GC over time (all pcROI pairs) — is the baseline a low
     point that rises into the task?
  B  source group GC over time (mean and median over subjects) — flat?
  C  per-subject task-minus-baseline GC, sensor vs source (is the rise
     consistent across subjects in sensors but not source?)
  D  per-subject source GC "floor" (leading window), sorted — heavy tail?

Sensor = the BSMART .mat (all 15 seed->target pairs, both directions).
Source = the production GC_source_space run (LCMV custom, all pairs).
NB the sensor pcROI pairs and the source functional-ROI pairs are different
parcellations, so this is a group-level "does sensor rise / does source stay
flat" overview, not a pair-matched comparison (see pair_baseline_diagnose.py
for the matched Temporal<->IFG edge).

Usage:
    python methods_paper/group_washout.py --task overtProd  --stim-class prodDiff
    python methods_paper/group_washout.py --task perception --stim-class percDiff
"""
import os, sys, glob, argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DECODE_OUTPUT_ROOT

ROOT = DECODE_OUTPUT_ROOT.parent / 'GC_sensor_vs_source_baseline_check'
GC_SRC = DECODE_OUTPUT_ROOT.parent / 'GC_source_space'
BSMART = '/mnt/r/phd_thesis/Research/SpeechProduction/EEG/derivatives/BSMART'
FREQS = np.arange(1, 31); LB = (FREQS >= 13) & (FREQS <= 20)

TASK_CFG = {
    'overtProd': dict(
        mat=(f'{BSMART}/overtProd/seed_to_target_channels/pcROI_seed_to_pcROI_target/'
             'SW20_MO10_fs500_3LHSeeds_LHTargets_1600ms/'
             'overtProd_pwgc_20subj_fs500_SW20_M10_3LHSeeds_LHTargets_1600ms.mat'),
        sen_tmin=-1.6, sen_nwin=981, base=(-1600, -1500), task=(-1350, 320),
        onset='production onset'),
    'perception': dict(
        mat=(f'{BSMART}/perception/seed_to_target_channels/pcROI_seed_to_pcROI_target/'
             'SW20_MO10_fs500_3LHSeeds_LHTargets/'
             'perception_pwgc_20subj_fs500_SW20_M10_3LHSeeds_LHTargets.mat'),
        sen_tmin=-0.2, sen_nwin=381, base=(-200, -100), task=(0, 300),
        onset='auditory onset'),
}


def load_sensor_all_pairs(mat, tmin, nwin):
    """All pairs/dirs low_beta from the .mat -> (n_subj, n_pair, 2, n_win), win_ms."""
    import h5py
    f = h5py.File(mat, 'r')
    top = f['pwgc_mov_seed2target_results'][()]        # (nsubj,1,npair)
    ns, _, npair = top.shape
    G = np.full((ns, npair, 2, nwin), np.nan)
    for s in range(ns):
        for p in range(npair):
            cell = f[top[s, 0, p]][()]
            for dcol in range(2):
                arr = np.array(f[cell[dcol, 0]]).squeeze()      # (nwin,30)
                G[s, p, dcol] = arr[:, LB].mean(1)
    f.close()
    wm = (tmin + np.arange(nwin) / 500.0) * 1000.0
    return G, wm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', default='overtProd', choices=['overtProd', 'perception'])
    ap.add_argument('--stim-class', default='prodDiff')
    args = ap.parse_args()
    cfg = TASK_CFG[args.task]; base, task, onset = cfg['base'], cfg['task'], cfg['onset']

    # sensor (.mat, all pairs)
    G, swm = load_sensor_all_pairs(cfg['mat'], cfg['sen_tmin'], cfg['sen_nwin'])
    sen_tc = np.nanmean(G, (0, 1, 2))                                     # group time course
    sb = (swm >= base[0]) & (swm <= base[1]); st = (swm >= task[0]) & (swm <= task[1])
    sen_tb = np.nanmean(G[:, :, :, st], (1, 2, 3)) - np.nanmean(G[:, :, :, sb], (1, 2, 3))

    # source (GC_source_space, all pairs)
    GD = (GC_SRC / args.task / 'LCMV' / 'custom' / 'vertex_selectkbest' /
          'leakage_corrected' / 'order10_win40ms_fs500' / 'all_rois' / args.stim_class)
    files = sorted(glob.glob(str(GD / '*.npz')))
    src_tcs, src_tb, src_floor, subj = [], [], [], []
    for fp in files:
        z = np.load(fp, allow_pickle=True); wm = z['window_ms']
        g = np.concatenate([z['fxy_low_beta'], z['fyx_low_beta']], 0)      # (2*npair,nwin)
        src_tcs.append(g.mean(0))
        lm = wm <= (wm[0] + 98)                                            # leading window
        bm = (wm >= base[0]) & (wm <= base[1]); tm = (wm >= task[0]) & (wm <= task[1])
        # source epoch may start later than the sensor baseline; fall back to its own leading 100 ms
        if not bm.any():
            bm = wm <= (wm[0] + 100)
        src_tb.append(g[:, tm].mean() - g[:, bm].mean()); src_floor.append(g[:, lm].mean())
        subj.append(os.path.basename(fp).split('_')[0][-2:])
    src_tcs = np.array(src_tcs); swm2 = wm
    src_tb = np.array(src_tb); src_floor = np.array(src_floor)

    C = {'sen': '#2166ac', 'src': '#b2182b'}
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    ax[0, 0].plot(swm, sen_tc, color=C['sen'], lw=1.3)
    ax[0, 0].axvspan(base[0], base[1], color='0.6', alpha=.25, label=f'baseline {base[0]}:{base[1]}')
    ax[0, 0].axvline(0, color='k', lw=.6)
    ax[0, 0].set_title('A. SENSOR group-mean GC (low_beta, all pairs)')
    ax[0, 0].set_xlabel(f'window start (ms)  [0={onset}]'); ax[0, 0].set_ylabel('GC'); ax[0, 0].legend(loc='upper right')
    ax[0, 1].plot(swm2, src_tcs.mean(0), color=C['src'], lw=1.3, label='mean over subjects')
    ax[0, 1].plot(swm2, np.median(src_tcs, 0), color='#f4a582', lw=1.3, label='median over subjects')
    ax[0, 1].axvspan(base[0], base[1], color='0.6', alpha=.25); ax[0, 1].axvline(0, color='k', lw=.6)
    ax[0, 1].set_title('B. SOURCE group GC (low_beta, all pairs)')
    ax[0, 1].set_xlabel(f'window start (ms)  [0={onset}]'); ax[0, 1].set_ylabel('GC'); ax[0, 1].legend(loc='upper right'); ax[0, 1].set_ylim(0, None)

    def strip(axx, vals, x, color, lbl):
        axx.scatter(np.full(len(vals), x) + np.random.RandomState(0).uniform(-.08, .08, len(vals)),
                    vals, c=color, s=45, zorder=3, label=lbl, edgecolor='k', lw=.4)
        axx.hlines(vals.mean(), x - .18, x + .18, color=color, lw=2.5, zorder=4)
    ax[1, 0].axhline(0, color='0.5', lw=1, ls='--')
    strip(ax[1, 0], sen_tb, 0, C['sen'], f'sensor ({int((sen_tb>0).sum())}/{len(sen_tb)} > 0)')
    strip(ax[1, 0], src_tb, 1, C['src'], f'source ({int((src_tb>0).sum())}/{len(src_tb)} > 0)')
    ax[1, 0].set_xticks([0, 1]); ax[1, 0].set_xticklabels(['SENSOR', 'SOURCE'])
    ax[1, 0].set_title('C. Per-subject task−baseline GC (bar = group mean)')
    ax[1, 0].set_ylabel('task − baseline GC (low_beta)'); ax[1, 0].legend(loc='upper right', fontsize=9)
    order = np.argsort(src_floor)
    ax[1, 1].bar(range(len(src_floor)), src_floor[order],
                 color=['#b2182b' if v > 0.1 else '#92c5de' for v in src_floor[order]])
    ax[1, 1].axhline(0.1, color='0.4', ls=':', lw=1)
    ax[1, 1].set_xticks(range(len(src_floor))); ax[1, 1].set_xticklabels([subj[i] for i in order], rotation=90, fontsize=7)
    ax[1, 1].set_title(f'D. Per-subject SOURCE GC floor (leading window); {int((src_floor>0.1).sum())}/{len(src_floor)} > 0.1 (red)')
    ax[1, 1].set_xlabel('subject (EEGPROD40xx)'); ax[1, 1].set_ylabel('leading-window GC (low_beta)')
    fig.suptitle(f'{args.task}: sensor-vs-source GC across all speech pairs (low_beta)', fontsize=14, y=1.005)
    fig.tight_layout(); out = ROOT / f'{args.task}_group_washout.png'
    fig.savefig(out, dpi=130, bbox_inches='tight'); plt.close(fig)
    print(f'wrote {out}')
    print(f'  sensor mean task-base=+{sen_tb.mean():.4f} ({int((sen_tb>0).sum())}/{len(sen_tb)}>0); '
          f'source mean task-base={src_tb.mean():+.4f} ({int((src_tb>0).sum())}/{len(src_tb)}>0)')
    print(f'  source floor: mean={src_floor.mean():.3f} median={np.median(src_floor):.3f} >0.1: {int((src_floor>0.1).sum())}/{len(src_floor)}')


if __name__ == '__main__':
    main()
