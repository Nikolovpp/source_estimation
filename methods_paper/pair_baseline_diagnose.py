#!/usr/bin/env python3
"""Focused diagnosis of ONE connection (Temporal <-> Inferior Frontal), per task.

Sensor (mastoid_ref, mean-of-3 pseudo-channels, matching the BSMART GC .mat)
vs LCMV source ROIs awfa-lh (temporal) <-> ifc-lh (IFG).  Each space keeps its
NATIVE epoch / baseline and is labelled as such.

  overtProd : sensor GC = the -1.6 s .mat (baseline -1600:-1500); source = -1.5 s
              LCMV (baseline -1500:-1400); "task" ~ production onset (-0.3..0.3).
  perception: sensor GC = the -0.2 s .mat and source = -0.2 s LCMV — SAME epoch,
              shared baseline -200:-100 (a pure silent period); task = auditory
              onset onward (0..0.3).

Inputs:
  sensor GC : the BSMART .mat (pair index 2 = seed Temporal -> target IFG)
  sensor TS : sensor_mastoid_ts/ caches   (sensor_mastoid_extract.py, mastoid)
  source TS : reduced_ts/ caches          (baseline_ts_extract.py, source part)

Outputs (durable, under GC_sensor_vs_source_baseline_check/), task-prefixed:
  {task}_pair_gc_overlay.png   per-subject GC time courses, baselines labelled
  {task}_pair_mechanism.png    baseline-vs-task power / coherence / cross-corr
  {task}_pair_metrics.csv , {task}_pair_mechanism_metrics.csv

Usage:
    python methods_paper/pair_baseline_diagnose.py --task overtProd  --stim-class prodDiff
    python methods_paper/pair_baseline_diagnose.py --task perception --stim-class percDiff
"""
import os, sys, glob, argparse, warnings
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DECODE_OUTPUT_ROOT
from granger import moving_window_pairwise_gc, band_average, DEFAULT_BANDS

ROOT = DECODE_OUTPUT_ROOT.parent / 'GC_sensor_vs_source_baseline_check'
SRC_RED = ROOT / 'reduced_ts'                 # source virtual channels
SEN_MAST = ROOT / 'sensor_mastoid_ts'         # mastoid sensor pseudo-channels
BSMART = '/mnt/r/phd_thesis/Research/SpeechProduction/EEG/derivatives/BSMART'
FS = 500.0
FREQS = np.arange(1, 31); LB = (FREQS >= 13) & (FREQS <= 20)
BAND = 'low_beta'; MAXLAG = 20
SRC = ('awfa-lh', 'ifc-lh')                   # temporal, IFG
SENSOR_PAIR_IDX = 2                           # seed Temporal -> target IFG (both tasks)
COL = {'sensor': '#2166ac', 'source': '#b2182b'}

# Per-task config.  base_sen/base_src in seconds; task_win in seconds.
TASK_CFG = {
    'overtProd': dict(
        mat=(f'{BSMART}/overtProd/seed_to_target_channels/pcROI_seed_to_pcROI_target/'
             'SW20_MO10_fs500_3LHSeeds_LHTargets_1600ms/'
             'overtProd_pwgc_20subj_fs500_SW20_M10_3LHSeeds_LHTargets_1600ms.mat'),
        sen_tmin=-1.6, sen_nwin=981,
        base_sen=(-1.60, -1.50), base_src=(-1.50, -1.40), task_win=(-0.30, 0.30),
        onset='production onset', sen_epoch='−1.6 s .mat', src_epoch='−1.5 s LCMV',
        overlay_sub='sensor baseline (−1600:−1500) is a low point; source (−1500:−1400) is flat'),
    'perception': dict(
        mat=(f'{BSMART}/perception/seed_to_target_channels/pcROI_seed_to_pcROI_target/'
             'SW20_MO10_fs500_3LHSeeds_LHTargets/'
             'perception_pwgc_20subj_fs500_SW20_M10_3LHSeeds_LHTargets.mat'),
        sen_tmin=-0.2, sen_nwin=381,
        base_sen=(-0.20, -0.10), base_src=(-0.20, -0.10), task_win=(0.0, 0.30),
        onset='auditory onset', sen_epoch='−0.2 s .mat', src_epoch='−0.2 s LCMV',
        overlay_sub='silent baseline (−200:−100) vs auditory task (t>0): does the sensor rise where source doesn\'t?'),
}


def win(t, lo, hi):
    return (t >= lo) & (t <= hi)


def ccf_ensemble(a, b, maxlag):
    """r_ab(l)=corr(a(t),b(t+l)) averaged over trials; +lag => a leads b."""
    a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
    a = a / (a.std(1, keepdims=True) + 1e-20); b = b / (b.std(1, keepdims=True) + 1e-20)
    n = a.shape[1]; lags = np.arange(-maxlag, maxlag + 1); out = np.zeros(lags.size)
    for i, l in enumerate(lags):
        out[i] = ((a[:, :n - l] * b[:, l:]).mean() if l >= 0
                  else (a[:, -l:] * b[:, :n + l]).mean())
    return lags, out


def coh_spectrum(a, b, nfft=128):
    """Ensemble magnitude-squared coherence on a fixed 1-30 Hz grid."""
    n = a.shape[1]; w = np.hanning(n)
    A = np.fft.rfft((a - a.mean(1, keepdims=True)) * w, n=nfft, axis=1)
    B = np.fft.rfft((b - b.mean(1, keepdims=True)) * w, n=nfft, axis=1)
    f = np.fft.rfftfreq(nfft, 1.0 / FS)
    Sxy = (A * np.conj(B)).mean(0); Sxx = (A * np.conj(A)).mean(0).real
    Syy = (B * np.conj(B)).mean(0).real
    coh = (np.abs(Sxy) ** 2) / (Sxx * Syy + 1e-30)
    m = (f >= 1) & (f <= 30)
    return f[m], coh[m]


# ── sensor GC from the BSMART .mat ───────────────────────────────────
def load_sensor_mat_gc(mat, pair_idx, tmin, nwin):
    import h5py
    f = h5py.File(mat, 'r')
    top = f['pwgc_mov_seed2target_results'][()]        # (20,1,15) refs
    fxy, fyx = [], []
    for s in range(top.shape[0]):
        cell = f[top[s, 0, pair_idx]][()]               # (2,1) refs -> {Fxy,Fyx}
        Fxy = np.array(f[cell[0, 0]]).squeeze()         # Temporal->IFG
        Fyx = np.array(f[cell[1, 0]]).squeeze()         # IFG->Temporal
        fxy.append(Fxy[:, LB].mean(1)); fyx.append(Fyx[:, LB].mean(1))
    f.close()
    wm = (tmin + np.arange(nwin) / FS) * 1000.0
    return wm, np.array(fxy), np.array(fyx)


# ── source GC from the source virtual channels ───────────────────────
def source_gc(subj, task, stim):
    f = SRC_RED / f'{subj}_{task}_{stim}.npz'
    if not f.exists():
        return None
    d = np.load(f, allow_pickle=True)
    a = d[f'src__{SRC[0]}'].astype(float); b = d[f'src__{SRC[1]}'].astype(float)
    tvec = d['times']; ws = int(round(0.04 * FS))
    res = moving_window_pairwise_gc(np.stack([a, b], 1), order=10, freqs=FREQS,
                                    fs=FS, win_samples=ws, step=1)
    starts = np.arange(0, a.shape[1] - ws + 1)
    wm = tvec[starts] * 1000.0
    fxy = band_average(res['f_xy'], FREQS, DEFAULT_BANDS)[BAND]   # awfa->ifc
    fyx = band_average(res['f_yx'], FREQS, DEFAULT_BANDS)[BAND]   # ifc->awfa
    return dict(subj=subj, wm=wm, fxy=fxy, fyx=fyx)


# ── mechanism (power / coherence / CCF), each space at its own baseline ──
def mechanism(subj, task, stim, base_sen, base_src, task_win):
    out = {}
    srcf = SRC_RED / f'{subj}_{task}_{stim}.npz'
    senf = SEN_MAST / f'{subj}_{task}_{stim}.npz'
    if not (srcf.exists() and senf.exists()):
        return None
    ds = np.load(srcf, allow_pickle=True); dm = np.load(senf, allow_pickle=True)
    specs = {'sensor': (dm['Temporal'].astype(float), dm['Inferior_Frontal'].astype(float),
                        dm['times'], base_sen),
             'source': (ds[f'src__{SRC[0]}'].astype(float), ds[f'src__{SRC[1]}'].astype(float),
                        ds['times'], base_src)}
    for sp, (a, b, tv, base) in specs.items():
        n = min(a.shape[0], b.shape[0]); a, b = a[:n], b[:n]
        mb = win(tv, *base); mt = win(tv, *task_win)
        pa = (a ** 2).mean(0); pb = (b ** 2).mean(0); pwr = (pa + pb) / 2
        cf, cb = coh_spectrum(a[:, mb], b[:, mb]); _cf, ct = coh_spectrum(a[:, mt], b[:, mt])
        lags, xb = ccf_ensemble(a[:, mb], b[:, mb], MAXLAG)
        _l, xt = ccf_ensemble(a[:, mt], b[:, mt], MAXLAG)
        out[sp] = dict(
            tvec=tv, base=tuple(base), pwr=pwr, cohf=cf, coh_base=cb, coh_task=ct, lags=lags,
            ccf_base=xb, ccf_task=xt,
            pwr_ratio=float(pwr[mt].mean() / pwr[mb].mean()),
            coh_lb_base=float(cb[(cf >= 13) & (cf <= 20)].mean()),
            coh_lb_task=float(ct[(cf >= 13) & (cf <= 20)].mean()),
            ccf_shape_sim=float(np.corrcoef(xb, xt)[0, 1]))
    out['subj'] = subj
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', default='overtProd', choices=['overtProd', 'perception'])
    ap.add_argument('--stim-class', default='prodDiff')
    ap.add_argument('--n-jobs', type=int, default=8)
    args = ap.parse_args()
    cfg = TASK_CFG[args.task]
    subs = sorted(os.path.basename(f).split('_')[0]
                  for f in glob.glob(str(SRC_RED / f'*_{args.task}_{args.stim_class}.npz')))
    subs = [s for s in subs if (SEN_MAST / f'{s}_{args.task}_{args.stim_class}.npz').exists()]
    print(f'{args.task}/{args.stim_class}: sensor baseline {cfg["base_sen"]} s, '
          f'source baseline {cfg["base_src"]} s, task {cfg["task_win"]} s; '
          f'{len(subs)} subjects with both caches')

    swm, sfxy, sfyx = load_sensor_mat_gc(cfg['mat'], SENSOR_PAIR_IDX, cfg['sen_tmin'], cfg['sen_nwin'])
    src = [r for r in Parallel(n_jobs=args.n_jobs, prefer='processes')(
        delayed(source_gc)(s, args.task, args.stim_class) for s in subs) if r]
    mech = [r for r in (mechanism(s, args.task, args.stim_class,
                                  cfg['base_sen'], cfg['base_src'], cfg['task_win'])
                        for s in subs) if r]
    print(f'  sensor .mat: {sfxy.shape[0]} subj; source GC: {len(src)}; mechanism: {len(mech)}')

    ROOT.mkdir(parents=True, exist_ok=True)
    _overlay(args.task, cfg, swm, sfxy, sfyx, src)
    _mech_fig(args.task, cfg, mech)
    _csv(args.task, cfg, swm, sfxy, sfyx, src, mech)


def _blab(base):
    return f'baseline {base[0]*1000:.0f}:{base[1]*1000:.0f}'


def _overlay(task, cfg, swm, sfxy, sfyx, src):
    fig, ax = plt.subplots(2, 2, figsize=(14, 8))
    for c, (M, title) in enumerate([(sfxy, 'Temporal → IFG'), (sfyx, 'IFG → Temporal')]):
        A = ax[0, c]
        for row in M:
            A.plot(swm, row, color=COL['sensor'], alpha=.15, lw=.6)
        A.plot(swm, M.mean(0), color='k', lw=2.2, label='group mean')
        A.axvspan(cfg['base_sen'][0] * 1000, cfg['base_sen'][1] * 1000, color='.6', alpha=.3,
                  label=_blab(cfg['base_sen']))
        A.axvline(0, color='k', lw=.6)
        A.set_title(f'SENSOR ({cfg["sen_epoch"]}, mastoid): {title}  (n={M.shape[0]})')
        A.set_ylabel('GC (low_beta)'); A.legend(fontsize=8, loc='upper right')
    wm = src[0]['wm']
    for c, (key, title) in enumerate([('fxy', 'awfa → ifc'), ('fyx', 'ifc → awfa')]):
        A = ax[1, c]; M = np.vstack([r[key] for r in src])
        for row in M:
            A.plot(wm, row, color=COL['source'], alpha=.15, lw=.6)
        A.plot(wm, M.mean(0), color='k', lw=2.2, label='group mean')
        A.axvspan(cfg['base_src'][0] * 1000, cfg['base_src'][1] * 1000, color='.6', alpha=.3,
                  label=_blab(cfg['base_src']))
        A.axvline(0, color='k', lw=.6)
        A.set_title(f'SOURCE ({cfg["src_epoch"]}): {title}  (n={M.shape[0]})')
        A.set_xlabel(f'window start (ms)   [0 = {cfg["onset"]}]'); A.set_ylabel('GC (low_beta)')
        A.legend(fontsize=8, loc='upper right')
    fig.suptitle(f'{task}  Temporal↔IFG GC — {cfg["overlay_sub"]}', fontsize=13)
    fig.tight_layout(); out = ROOT / f'{task}_pair_gc_overlay.png'
    fig.savefig(out, dpi=130, bbox_inches='tight'); plt.close(fig); print(f'wrote {out}')


def _mech_fig(task, cfg, mech):
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    for sp in ('sensor', 'source'):
        tvec = mech[0][sp]['tvec'] * 1000
        P = np.vstack([m[sp]['pwr'] / m[sp]['pwr'].mean() for m in mech]).mean(0)
        base = mech[0][sp]['base']
        ax[0, 0].plot(tvec, P, color=COL[sp], lw=1.5,
                      label=f'{sp} (base {base[0]*1000:.0f}:{base[1]*1000:.0f})')
        ax[0, 0].axvspan(base[0] * 1000, base[1] * 1000, color=COL[sp], alpha=.12)
    ax[0, 0].axvline(0, color='k', lw=.6)
    ax[0, 0].set_title('A. Signal power over time (norm.), each space at its own baseline\nis baseline quiet?')
    ax[0, 0].set_xlabel(f'time (ms)   [0 = {cfg["onset"]}]'); ax[0, 0].set_ylabel('power / mean'); ax[0, 0].legend(fontsize=8)
    xpos = {'sensor': 0, 'source': 1}
    for sp in ('sensor', 'source'):
        cb = np.array([np.abs(m[sp]['ccf_base']).max() for m in mech])
        ct = np.array([np.abs(m[sp]['ccf_task']).max() for m in mech])
        x = xpos[sp]
        for i in range(len(cb)):
            ax[0, 1].plot([x - .12, x + .12], [cb[i], ct[i]], '-', color=COL[sp], alpha=.35, lw=.7)
        ax[0, 1].plot(np.full(len(cb), x - .12), cb, 'o', color=COL[sp], ms=4)
        ax[0, 1].plot(np.full(len(ct), x + .12), ct, 's', color=COL[sp], ms=4, mfc='white')
    ax[0, 1].set_xticks([0, 1]); ax[0, 1].set_xticklabels(['SENSOR', 'SOURCE'])
    ax[0, 1].plot([], [], 'ko', label='baseline'); ax[0, 1].plot([], [], 'ks', mfc='white', label='task')
    ax[0, 1].set_title('B. Inter-channel coupling strength (max |cross-corr|)\nsensor strongly coupled; leakage-corrected source ≈ decorrelated')
    ax[0, 1].set_ylabel('max |cross-correlation|'); ax[0, 1].legend(fontsize=8)
    for sp in ('sensor', 'source'):
        f = mech[0][sp]['cohf']
        cb = np.vstack([m[sp]['coh_base'] for m in mech]).mean(0)
        ct = np.vstack([m[sp]['coh_task'] for m in mech]).mean(0)
        ax[1, 0].plot(f, cb, color=COL[sp], lw=1.6, label=f'{sp} baseline')
        ax[1, 0].plot(f, ct, color=COL[sp], lw=1.3, ls='--', label=f'{sp} task')
    ax[1, 0].axvspan(13, 20, color='.85', alpha=.5, zorder=0)
    ax[1, 0].set_title('C. Coherence spectrum baseline(—) vs task(--)\n(shaded = low_beta)')
    ax[1, 0].set_xlabel('frequency (Hz)'); ax[1, 0].set_ylabel('coherence'); ax[1, 0].legend(fontsize=8)
    for sp in ('sensor', 'source'):
        lag = mech[0][sp]['lags'] * 2
        cb = np.vstack([m[sp]['ccf_base'] for m in mech]).mean(0)
        ct = np.vstack([m[sp]['ccf_task'] for m in mech]).mean(0)
        ax[1, 1].plot(lag, cb, color=COL[sp], lw=1.6, label=f'{sp} baseline')
        ax[1, 1].plot(lag, ct, color=COL[sp], lw=1.3, ls='--', label=f'{sp} task')
    ax[1, 1].axvline(0, color='.7', lw=.6); ax[1, 1].axhline(0, color='.7', lw=.6)
    ax[1, 1].set_title('D. Cross-correlation vs lag (group mean)\n(+lag: Temporal/awfa leads)')
    ax[1, 1].set_xlabel('lag (ms)'); ax[1, 1].set_ylabel('cross-corr'); ax[1, 1].legend(fontsize=8)
    fig.suptitle(f'{task}  Temporal↔IFG mechanism — sensor {_blab(cfg["base_sen"])}, source {_blab(cfg["base_src"])}: '
                 'sensor strongly coupled; leakage-corrected source ROIs nearly decorrelated', fontsize=12)
    fig.tight_layout(); out = ROOT / f'{task}_pair_mechanism.png'
    fig.savefig(out, dpi=130, bbox_inches='tight'); plt.close(fig); print(f'wrote {out}')


def _csv(task, cfg, swm, sfxy, sfyx, src, mech):
    bm = win(swm, cfg['base_sen'][0] * 1000, cfg['base_sen'][1] * 1000)
    tm = win(swm, cfg['task_win'][0] * 1000, cfg['task_win'][1] * 1000)
    rows = []
    for i in range(sfxy.shape[0]):
        for d, M in [('T→IFG', sfxy), ('IFG→T', sfyx)]:
            rows.append(dict(space=f'sensor({cfg["sen_epoch"]})', direction=d,
                             gc_base=float(M[i, bm].mean()), gc_task=float(M[i, tm].mean())))
    wm = src[0]['wm']; sb = win(wm, cfg['base_src'][0] * 1000, cfg['base_src'][1] * 1000)
    st = win(wm, cfg['task_win'][0] * 1000, cfg['task_win'][1] * 1000)
    for r in src:
        for d, key in [('awfa→ifc', 'fxy'), ('ifc→awfa', 'fyx')]:
            rows.append(dict(space=f'source({cfg["src_epoch"]})', direction=d, subject=r['subj'],
                             gc_base=float(r[key][sb].mean()), gc_task=float(r[key][st].mean())))
    pd.DataFrame(rows).to_csv(ROOT / f'{task}_pair_metrics.csv', index=False)
    mrows = [dict(subject=m['subj'], space=sp, pwr_ratio=m[sp]['pwr_ratio'],
                  coh_lb_base=m[sp]['coh_lb_base'], coh_lb_task=m[sp]['coh_lb_task'],
                  ccf_shape_sim=m[sp]['ccf_shape_sim']) for m in mech for sp in ('sensor', 'source')]
    pd.DataFrame(mrows).to_csv(ROOT / f'{task}_pair_mechanism_metrics.csv', index=False)
    print(f'wrote {ROOT}/{task}_pair_metrics.csv , {task}_pair_mechanism_metrics.csv')
    print('\n=== SUMMARY ===')
    print(f'  SENSOR GC ({cfg["sen_epoch"]}, {_blab(cfg["base_sen"])}): '
          f'Temporal→IFG base={sfxy[:,bm].mean():.4f} task={sfxy[:,tm].mean():.4f} '
          f'| IFG→Temporal base={sfyx[:,bm].mean():.4f} task={sfyx[:,tm].mean():.4f}')
    fxy_all = np.vstack([r['fxy'] for r in src]); fyx_all = np.vstack([r['fyx'] for r in src])
    print(f'  SOURCE GC ({cfg["src_epoch"]}, {_blab(cfg["base_src"])}): '
          f'awfa→ifc base={fxy_all[:,sb].mean():.4f} task={fxy_all[:,st].mean():.4f} '
          f'| ifc→awfa base={fyx_all[:,sb].mean():.4f} task={fyx_all[:,st].mean():.4f}')
    md = pd.DataFrame(mrows)
    for sp in ('sensor', 'source'):
        s = md[md.space == sp]
        print(f'  MECHANISM {sp:6s}: power task/base={s.pwr_ratio.mean():.2f}  '
              f'coh_lb base/task={s.coh_lb_base.mean():.2f}/{s.coh_lb_task.mean():.2f}  '
              f'CCF base↔task sim={s.ccf_shape_sim.mean():.2f}')


if __name__ == '__main__':
    main()
