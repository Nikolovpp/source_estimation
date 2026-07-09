#!/usr/bin/env python3
"""Build mastoid-referenced sensor pseudo-channels for the Temporal~IFG check.

The sensor-space GC (``crunchtime_task_overtProd_GC.m``) used the **mastoid_ref**
epochs and a pseudo-channel = mean of 3 electrodes.  ``data_loader`` re-applies
an *average* reference, so we load the mastoid ``.mat`` directly and keep its
reference untouched, reusing only ``data_loader``'s trial-selection helpers so
the trial set matches the source cache (same good + prodDiff/percDiff trials).

Pseudo-channels: Temporal = mean(FT7,T7,TP7); Inferior_Frontal = mean(F5,FC5,FC3).
Uses the -1.6 s mastoid epochs (baseline -1600:-1500) so the sensor TIME SERIES
matches the -1.6 s sensor GC .mat exactly (--epoch 1.5 falls back to the -1.5 s
files, baseline -1500:-1400).

Saves one small npz per subject to
  derivatives/.../GC_sensor_vs_source_baseline_check/sensor_mastoid_ts/{subj}_{task}_{stim}.npz
with arrays Temporal, Inferior_Frontal (n_trials, n_times), times (s), sfreq.

Usage:
    conda activate mne
    python methods_paper/sensor_mastoid_extract.py --task overtProd --stim-class prodDiff
"""
import os, sys, glob, argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np
import mat73
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import EEGLAB_DIR, SUBJECT_IDS, DECODE_OUTPUT_ROOT
from data_loader import _extract_word_list, _get_good_trial_mask, _build_class_labels

OUT_ROOT = (DECODE_OUTPUT_ROOT.parent / 'GC_sensor_vs_source_baseline_check'
            / 'sensor_mastoid_ts')
PSEUDO = {'Temporal': ['FT7', 'T7', 'TP7'],
          'Inferior_Frontal': ['F5', 'FC5', 'FC3']}


EPOCH_FILE = {'1.6': '-1.6_0.4_-1599_-1500', '1.5': '-1.5_0.4_-1499_-1400'}


def mastoid_mat_path(subj, task, epoch='1.6'):
    # mastoid_ref, fs_500 (matches the sensor GC .mat for each task)
    if task == 'perception':
        # perception epoch -0.2..0.6 s; the GC .mat used popthresh120
        fn = (f'{subj}_task-perception_0.1_30_sep_1_1_mastoid_ref_'
              f'-0.2_0.6_500Hz_reSample_popthresh120.mat')
        return EEGLAB_DIR / task / subj / 'mastoid_ref' / 'eeglab_standard' / 'fs_500' / fn
    # overtProd: -1.6 s matches the sensor GC .mat, -1.5 s matches the source grid
    fn = (f'{subj}_task_overtProduction_0.1_30_sep_1_1_'
          f'{EPOCH_FILE[epoch]}_500Hz_reSample_ProdOnset.mat')
    return EEGLAB_DIR / task / subj / 'mastoid_ref' / 'fs_500' / fn


def build_subject(subj, task, stim, epoch='1.6'):
    p = mastoid_mat_path(subj, task, epoch)
    if not p.exists():
        return None, f'  {subj}: no mastoid file'
    d = mat73.loadmat(str(p))
    labels = list(d['chanlocs']['labels'])
    lut = {c.lower(): i for i, c in enumerate(labels)}
    data = d['data']                              # (chan, time, trial)
    data = data.swapaxes(2, 0).swapaxes(1, 2)     # (trial, chan, time)
    times = np.asarray(d['times']).ravel() / 1000.0   # ms -> s
    # same trial selection as the source pipeline (good + contrast)
    good = _get_good_trial_mask(d).ravel()
    gi = np.where(good)[0]
    words = _extract_word_list(d, task)
    words_good = [words[i] for i in gi]
    y, keep = _build_class_labels(words_good, stim)
    data_sel = data[gi][keep]                     # (n_trials, chan, time)
    out = {}
    for name, elecs in PSEUDO.items():
        idx = [lut[e.lower()] for e in elecs if e.lower() in lut]
        miss = [e for e in elecs if e.lower() not in lut]
        if miss:
            return None, f'  {subj}: missing electrodes {miss}'
        out[name] = data_sel[:, idx, :].mean(1).astype(np.float32)   # (n_trials, time)
    return (out, times, data_sel.shape[0]), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', default='overtProd')
    ap.add_argument('--stim-class', default='prodDiff')
    ap.add_argument('--subjects', nargs='+', default=None)
    ap.add_argument('--epoch', default='1.6', choices=['1.6', '1.5'],
                    help="mastoid epoch: 1.6 (baseline -1600:-1500, matches sensor GC .mat) "
                         "or 1.5 (baseline -1500:-1400, matches source grid)")
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()
    subjects = args.subjects or SUBJECT_IDS
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f'Mastoid sensor pseudo-channels: {args.task}/{args.stim_class} '
          f'(-{args.epoch}s epoch) -> {OUT_ROOT}')
    ok = 0
    for subj in subjects:
        out = OUT_ROOT / f'{subj}_{args.task}_{args.stim_class}.npz'
        if out.exists() and not args.overwrite:
            print(f'  {subj}: cached'); ok += 1; continue
        res, err = build_subject(subj, args.task, args.stim_class, args.epoch)
        if res is None:
            print(err); continue
        chans, times, ntr = res
        save = {'times': times, 'sfreq': np.array(500.0)}
        save.update(chans)
        tmp = out.with_name(out.stem + '.tmp.npz')   # savez keeps a .npz name as-is
        np.savez_compressed(tmp, **save); os.replace(tmp, out)
        print(f'  {subj}: OK  {ntr} trials, {times.size} samp -> {out.name}')
        ok += 1
    print(f'done: {ok}/{len(subjects)}')


if __name__ == '__main__':
    main()
