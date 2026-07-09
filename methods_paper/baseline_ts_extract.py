#!/usr/bin/env python3
"""Extract matched sensor + LCMV-source ROI time series for the baseline-GC check.

Goal (diagnostic for a possible methods point): understand why sensor-space
Granger causality has a low, task-modulated baseline while LCMV source ROIs do
not.  This script does the *heavy IO* half — it reads each subject's big vertex
cache once, reduces every ROI to one virtual channel (the same PC1 spatial
filter ``run_granger.py`` uses), resamples to 500 Hz, and caches the tiny
reduced time series next to the matched sensor pseudo-channels.  ``baseline_ts_
analyze.py`` then runs the diagnostics on those caches (fast, re-runnable).

Matched channels (source functional ROI  <->  sensor pseudo-channel):
    awfa-lh  <-> Temporal          (FT7 T7 TP7)
    ifc-lh   <-> Inferior_Frontal  (F5 FC5 FC3)
    pmc-lh   <-> Superior_Frontal   (FCz FC1 F1)
    tpc-lh   <-> Superior_Parietal (CPz CP1 P1)

Both sides end up on the SAME grid: overtProd epoch -1.5..0.4 s @ 500 Hz.
(The original MATLAB sensor GC used -1.6..0.4; here we deliberately match the
source epoch so the only difference is sensor-vs-source, not the window.)

Resumable: skips subjects whose reduced cache already exists (unless
--overwrite).  Source caches live on a slow mount, so run this in the
background; the analyze step needs only the small local caches.

Usage
-----
    conda activate mne
    python methods_paper/baseline_ts_extract.py --task overtProd --stim-class prodDiff
"""
import os, sys, argparse, time, warnings
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
warnings.filterwarnings('ignore')
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SUBJECT_IDS, DECODE_OUTPUT_ROOT
from decoding_io import _load_cached_roi_data
from granger import reduce_roi_first_pc
from run_granger import resample_channels
from data_loader import load_subject_epochs

# transferred source-timeseries root (custom atlas, LCMV, leakage-corrected)
SRC_TS_ROOT = ('/mnt/s/Research/SpeechProduction/DECODE_source_space_timeseries'
               '/{task}/LCMV/custom/vertex/leakage_corrected/'
               '{subj}_{task}_{stim}.npz')
OUT_ROOT = (DECODE_OUTPUT_ROOT.parent / 'GC_sensor_vs_source_baseline_check'
            / 'reduced_ts')

# source ROI  ->  sensor pseudo-channel (name, electrode set)
ROI_MAP = {
    'awfa-lh': ('Temporal',          ['FT7', 'T7', 'TP7']),
    'ifc-lh':  ('Inferior_Frontal',  ['F5', 'FC5', 'FC3']),
    'pmc-lh':  ('Superior_Frontal',  ['FCz', 'FC1', 'F1']),
    'tpc-lh':  ('Superior_Parietal', ['CPz', 'CP1', 'P1']),
}
SRC_ROIS = list(ROI_MAP)
TARGET_FS = 500.0


def sensor_pseudochannels(subj, task, stim):
    ep, _y, sf = load_subject_epochs(subj, task, stim, fs=500)
    data = ep.get_data(copy=False)
    chn = [c.lower() for c in ep.ch_names]
    out = {}
    for _src, (name, elecs) in ROI_MAP.items():
        idx = [chn.index(e.lower()) for e in elecs if e.lower() in chn]
        out[name] = data[:, idx, :].mean(axis=1)          # (n_ep, n_t)
    return out, np.asarray(ep.times), float(sf)


def source_virtual_channels(npz_path):
    roi_data, _y, times, sfreq = _load_cached_roi_data(
        npz_path, 'vertex_selectkbest', roi_subset=SRC_ROIS)
    vcs = {r: reduce_roi_first_pc(np.asarray(roi_data[r], float)) for r in roi_data}
    V = np.stack([vcs[r] for r in SRC_ROIS], 0)           # (n_roi, n_ep, n_t)
    V, fs = resample_channels(V, sfreq, TARGET_FS)
    t = times[0] + np.arange(V.shape[2]) / fs
    return {SRC_ROIS[i]: V[i] for i in range(len(SRC_ROIS))}, t, fs


def align(ts_a, ts_b):
    """Common index range so t_a[ia:ia+n] ~= t_b[ib:ib+n]."""
    t0 = max(ts_a[0], ts_b[0])
    ia = int(np.argmin(np.abs(ts_a - t0))); ib = int(np.argmin(np.abs(ts_b - t0)))
    n = min(ts_a.size - ia, ts_b.size - ib)
    return ia, ib, n


def process_subject(subj, task, stim, src_template, overwrite, source_only=False):
    """Extract + cache one subject's reduced TS. Returns a status str.

    ``source_only`` skips the average-ref sensor pseudo-channels (the mechanism
    uses the mastoid sensor caches from sensor_mastoid_extract.py instead); use
    it for perception, whose sensor side comes entirely from mastoid.
    """
    out = OUT_ROOT / f'{subj}_{task}_{stim}.npz'
    if out.exists() and not overwrite:
        return f'  {subj}: cached, skip'
    src_npz = src_template.format(task=task, subj=subj, stim=stim)
    if not os.path.exists(src_npz):
        return f'  {subj}: source TS not present — skip'
    try:
        t0 = time.time()
        src, ts_s, _fs_s = source_virtual_channels(src_npz)
        save = {'sfreq': np.array(TARGET_FS), 'src_rois': np.array(SRC_ROIS),
                'sen_names': np.array([ROI_MAP[r][0] for r in SRC_ROIS])}
        if source_only:
            save['times'] = ts_s
            for r in SRC_ROIS:
                save[f'src__{r}'] = src[r].astype(np.float32)
            n = ts_s.size
        else:
            sen, ts_e, _fs_e = sensor_pseudochannels(subj, task, stim)
            ia, ib, n = align(ts_s, ts_e)
            src = {k: v[:, ia:ia + n] for k, v in src.items()}
            sen = {k: v[:, ib:ib + n] for k, v in sen.items()}
            save['times'] = ts_s[ia:ia + n]
            for r in SRC_ROIS:
                save[f'src__{r}'] = src[r].astype(np.float32)
                save[f'sen__{ROI_MAP[r][0]}'] = sen[ROI_MAP[r][0]].astype(np.float32)
        tmp = out.with_name(out.stem + '.tmp.npz')  # savez keeps a .npz name; atomic replace
        np.savez_compressed(tmp, **save); os.replace(tmp, out)
        ne = next(iter(src.values())).shape[0]
        return f'  {subj}: OK  n={n} samp  ep={ne}  ({time.time()-t0:.0f}s) -> {out.name}'
    except Exception as e:
        return f'  {subj}: FAIL {type(e).__name__}: {e}'


def main():
    import multiprocessing as mp
    p = argparse.ArgumentParser()
    p.add_argument('--task', default='overtProd', choices=['overtProd', 'perception'])
    p.add_argument('--stim-class', default='prodDiff', choices=['prodDiff', 'percDiff'])
    p.add_argument('--subjects', nargs='+', default=None)
    p.add_argument('--src-ts-root', default=os.environ.get('SRC_TS_ROOT', SRC_TS_ROOT),
                   help='Template for the source-timeseries .npz (uses {task}/{subj}/{stim}). '
                        'Override for the workstation; env SRC_TS_ROOT also works.')
    p.add_argument('--subject-jobs', type=int, default=1,
                   help='Parallelise across subjects (IO + PC reduction). On a fast '
                        'disk + many cores set this to ~8-16; on a slow network mount '
                        'keep it 1-2 (reads contend).')
    p.add_argument('--source-only', action='store_true',
                   help='Skip the average-ref sensor part (mechanism uses the '
                        'mastoid sensor caches); use for perception.')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    subjects = args.subjects or SUBJECT_IDS
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f'Extract matched reduced TS: {args.task}/{args.stim_class}  '
          f'jobs={args.subject_jobs}  source_only={args.source_only}  -> {OUT_ROOT}')

    work = [(s, args.task, args.stim_class, args.src_ts_root, args.overwrite,
             args.source_only) for s in subjects]
    if args.subject_jobs > 1:
        with mp.Pool(args.subject_jobs) as pool:
            results = pool.starmap(process_subject, work)
    else:
        results = [process_subject(*w) for w in work]
    for r in results:
        print(r)
    ok = sum('OK' in r for r in results)
    print(f'\ndone: {ok}/{len(results)} extracted (rest cached/skipped)')


if __name__ == '__main__':
    main()
