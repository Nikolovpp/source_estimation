#!/usr/bin/env python
"""Where does each ROI's cross-talk function take its energy from, by region?

For every sensorEquiv ROI the CTF energy (sum of CTF² over vertices) is split
into anatomical regions: the ROI itself, its right-hemisphere mirror, left
and right auditory cortex (the Temporal composite and its mirror), the left
and right sensorimotor strip (areas 4/3a/3b/1/2), and the rest of each
hemisphere.  Answers questions such as "why does Superior Parietal decode
the auditory contrast": if its filter takes a sizeable share of its energy
from right auditory cortex, the ROI reports bilateral auditory activity.

Three filters are compared:
    dSPM (SNR 3)        the production vertex arm — the ROI's PC1 CTF from the
                        per-subject baseline files of resolution_analysis.py
                        (real noise covariances, averaged over subjects)
    DeFleCT other_rois  the DeFleCT arm's bank (k 3, 2 null patterns per other
                        ROI, SNR 10); channel 1 and the energy summed over the
                        3 channels
    DeFleCT none        the same bank without nulls (what the nulls remove)
The DeFleCT banks are designed here from the fsaverage leadfield with an
identity noise covariance (no epochs locally), so their numbers are the
template-level design, not a subject's.

Outputs (RESOLUTION_ANALYSIS/{task}_{stim}/):
    ctf_region_shares_sensorEquiv.csv / .png   energy shares by region
    ctf_region_gains_sensorEquiv.csv           per-vertex sensitivity to selected regions
                                               relative to the own ROI (mean CTF² ratio, %)

Usage (local, leadfield only):
    python exploratory/ctf_region_shares.py --task perception --stim-class percDiff
    python exploratory/ctf_region_shares.py --designs null2_other_rois null1_ring null2_ring_rois
"""
import os
import re
import sys
import glob
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne

from config import SPEECH_ROIS, DEFLECT_SM_PARCELS
from resolution_analysis import OUT_ROOT
from forward_model import setup_fsaverage, make_forward, build_roi_labels
from deflect import leadfield_fixed, label_rows, design_filters, build_ring_labels, sensorimotor_label

ROIS = ['Temporal', 'Inferior_Frontal', 'Superior_Frontal', 'Superior_Parietal', 'Visual']
ABBR = {'Temporal': 'Temporal', 'Inferior_Frontal': 'Inf. Frontal', 'Superior_Frontal': 'Sup. Frontal',
        'Superior_Parietal': 'Sup. Parietal', 'Visual': 'Visual'}


def mirror(names):
    return [n.replace('L_', 'R_', 1).replace('-lh', '-rh') for n in names]


def region_defs():
    base = SPEECH_ROIS['sensorEquiv']
    regs = {r: list(base[r]) for r in ROIS}
    for r in ROIS:
        regs[f'R_{r}'] = mirror(base[r])
    regs['L_sensorimotor'] = [f'L_{p}_ROI-lh' for p in DEFLECT_SM_PARCELS]
    regs['R_sensorimotor'] = mirror(regs['L_sensorimotor'])
    return regs


def shares(ctf_row, rows, roi, n_lh):
    """Energy shares of one CTF (n_src,) by region; returns dict."""
    e = ctf_row ** 2
    tot = e.sum()
    out = {}
    used_lh = np.zeros(len(e), bool); used_rh = np.zeros(len(e), bool)
    own = rows[roi]
    out['own ROI'] = e[own].sum() / tot
    used_lh[own] = True
    mir = np.setdiff1d(rows[f'R_{roi}'], own)
    out['R mirror of ROI'] = e[mir].sum() / tot; used_rh[mir] = True
    if roi != 'Temporal':
        la = np.setdiff1d(rows['Temporal'], np.where(used_lh)[0])
        out['L auditory (Temporal ROI)'] = e[la].sum() / tot; used_lh[la] = True
        ra = np.setdiff1d(rows['R_Temporal'], np.where(used_rh)[0])
        out['R auditory (mirror)'] = e[ra].sum() / tot; used_rh[ra] = True
    else:
        out['L auditory (Temporal ROI)'] = 0.0; out['R auditory (mirror)'] = 0.0
    ls = np.setdiff1d(rows['L_sensorimotor'], np.where(used_lh)[0])
    out['L sensorimotor strip'] = e[ls].sum() / tot; used_lh[ls] = True
    rs = np.setdiff1d(rows['R_sensorimotor'], np.where(used_rh)[0])
    out['R sensorimotor strip'] = e[rs].sum() / tot; used_rh[rs] = True
    other_rois = np.concatenate([rows[o] for o in ROIS if o not in (roi, 'Temporal')])
    orl = np.setdiff1d(other_rois, np.where(used_lh)[0])
    out['other speech ROIs (L)'] = e[orl].sum() / tot; used_lh[orl] = True
    lh_idx = np.arange(n_lh); rh_idx = np.arange(n_lh, len(e))
    out['rest of L hemisphere'] = e[lh_idx[~used_lh[lh_idx]]].sum() / tot
    out['rest of R hemisphere'] = e[rh_idx[~used_rh[rh_idx]]].sum() / tot
    out['R hemisphere total'] = e[rh_idx].sum() / tot
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--task', default='perception', choices=['perception', 'overtProd'])
    ap.add_argument('--stim-class', default='percDiff', choices=['prodDiff', 'percDiff'])
    ap.add_argument('--method', default='dSPM', help='vertex-arm method whose baseline CTFs are read')
    ap.add_argument('--snr', type=float, default=10.0)
    ap.add_argument('--k', type=int, default=3)
    ap.add_argument('--null-comp', type=int, default=2, help='(ignored when --designs is given)')
    ap.add_argument('--designs', nargs='+', default=None,
                    help='DeFleCT null designs to compare, as null<N>_<nulls>, e.g. '
                         'null2_other_rois null1_ring null2_ring_rois (default: '
                         'null<--null-comp>_other_rois and _none)')
    args = ap.parse_args()

    out_dir = OUT_ROOT / f'{args.task}_{args.stim_class}'
    subjects_dir, _, src, bem = setup_fsaverage()
    regs = region_defs()
    labels = build_roi_labels(subjects_dir, atlas='HCPMMP1', composite_rois=regs)

    # ── leadfield from a template biosemi64 info (average reference) ──
    mont = mne.channels.make_standard_montage('biosemi64')
    info = mne.create_info(mont.ch_names, 500.0, 'eeg')
    info.set_montage(mont)
    raw = mne.io.RawArray(np.zeros((len(mont.ch_names), 10)), info, verbose=False)
    raw.set_eeg_reference('average', projection=True, verbose=False)
    info = raw.info
    fwd = make_forward(info, src, bem)
    G, verts, P = leadfield_fixed(fwd, info)
    n_lh = len(verts[0])
    rows = {n: label_rows(l, verts[0], verts[1]) for n, l in labels.items()}
    print(f'leadfield {G.shape}, n_lh {n_lh}; ROI sizes ' +
          ', '.join(f'{r} {len(rows[r])}' for r in ROIS))

    records = []
    # ── vertex arm: per-subject baseline CTFs ──
    files = sorted(glob.glob(str(out_dir / 'per_subject' / f'*_{args.method}_sensorEquiv.npz')))
    if files:
        acc = []
        for f in files:
            d = np.load(f, allow_pickle=True)
            ctf = d['ctf_pc1']; names = list(d['roi_names']); n_lh_f = int(d['n_lh'])
            if ctf.shape[1] != G.shape[1]:
                raise SystemExit(f'{f}: {ctf.shape[1]} sources vs leadfield {G.shape[1]}')
            acc.append({r: shares(ctf[names.index(r)], rows, r, n_lh_f) for r in ROIS})
        for r in ROIS:
            keys = acc[0][r].keys()
            rec = {k: float(np.mean([a[r][k] for a in acc])) for k in keys}
            rec.update(filter=f'{args.method} SNR 3 (PC1 CTF, n={len(acc)} subj)', roi=r)
            records.append(rec)
    else:
        print(f'  no baseline files for {args.method} in {out_dir / "per_subject"} — vertex arm skipped')

    # ── DeFleCT banks (identity noise covariance) ──
    roi_labels = [labels[r] for r in ROIS]
    roi_dict = {r: labels[r] for r in ROIS}
    C = np.eye(G.shape[0])
    designs = args.designs or [f'null{args.null_comp}_other_rois', f'null{args.null_comp}_none']
    ring_labels = sm_lbl = None
    if any('ring' in d for d in designs):
        ring_labels = build_ring_labels(subjects_dir, roi_dict)
    if any(d.endswith('sm') for d in designs):
        sm_lbl = sensorimotor_label(subjects_dir)
    designs_parsed = []
    for d in designs:
        m = re.match(r'null(\d+)_(.+)$', d)
        if not m:
            raise SystemExit(f'bad design {d!r}: expected null<N>_<nulls>')
        designs_parsed.append((int(m.group(1)), m.group(2)))
    for nc, nulls in designs_parsed:
        des = design_filters(G, verts, C, args.snr, roi_labels, ROIS, args.k, nc, nulls,
                             sm_label=sm_lbl, ring_labels=ring_labels)
        tag = f'DeFleCT k{args.k} null{nc} {nulls} SNR {args.snr:g}'
        for r in ROIS:
            W = des['W'][r]
            ctf = W @ G
            rec = shares(ctf[0], rows, r, n_lh)
            rec.update(filter=f'{tag} (ch 1)', roi=r); records.append(rec)
            e_sum = np.sqrt((ctf ** 2).sum(axis=0))
            rec = shares(e_sum, rows, r, n_lh)
            rec.update(filter=f'{tag} (3 ch energy)', roi=r); records.append(rec)

    # per-vertex sensitivity relative to the own ROI (the ctf_xtalk convention:
    # mean CTF^2 over the region / mean CTF^2 over the own vertices, in %).
    gain_regions = ['Temporal', 'R_Temporal', 'Superior_Parietal', 'R_Superior_Parietal',
                    'Superior_Frontal', 'L_sensorimotor', 'R_sensorimotor', 'Visual']

    def gains(ctf_row, roi):
        own = rows[roi]
        e_own = (ctf_row[own] ** 2).mean()
        out = {}
        for g in gain_regions:
            if g == roi:
                continue
            r = np.setdiff1d(rows[g], own)
            out[f'{g} %'] = 100 * (ctf_row[r] ** 2).mean() / e_own
        return out

    grecs = []
    if files:
        acc = []
        for f in files:
            d = np.load(f, allow_pickle=True)
            ctf = d['ctf_pc1']; names = list(d['roi_names'])
            acc.append({r: gains(ctf[names.index(r)], r) for r in ROIS})
        for r in ROIS:
            rec = {k: float(np.mean([a[r][k] for a in acc])) for k in acc[0][r]}
            rec.update(filter=f'{args.method} SNR 3 (PC1 CTF, n={len(acc)} subj)', roi=r); grecs.append(rec)
    for nc, nulls in designs_parsed:
        des = design_filters(G, verts, C, args.snr, roi_labels, ROIS, args.k, nc, nulls,
                             sm_label=sm_lbl, ring_labels=ring_labels, verbose=False)
        for r in ROIS:
            d = des['diag'][r]
            rec = gains((des['W'][r] @ G)[0], r)
            rec.update({'se_own ch1': float(d['se_own'][0]), 'noise gain ch1 (x1e-2)': 100 * float(d['noise_gain'][0]),
                        'max |cos| other ROI': d['max_xtalk_cos'], 'n_null': d['n_null']})
            rec.update(filter=f'DeFleCT k{args.k} null{nc} {nulls} SNR {args.snr:g} (ch 1)', roi=r); grecs.append(rec)
    gdf = pd.DataFrame(grecs)
    gcols = ['filter', 'roi'] + [c for c in gdf.columns if c not in ('filter', 'roi')]
    gdf = gdf[gcols]
    gdf.to_csv(out_dir / 'ctf_region_gains_sensorEquiv.csv', index=False)
    for filt, g in gdf.groupby('filter', sort=False):
        print(f'\n{filt}  — per-vertex sensitivity to other regions, % of the own ROI (mean CTF² ratio)')
        print(g.set_index('roi')[gcols[2:]].round(1).to_string())

    df = pd.DataFrame(records)
    cols = ['filter', 'roi', 'own ROI', 'R mirror of ROI', 'L auditory (Temporal ROI)', 'R auditory (mirror)',
            'L sensorimotor strip', 'R sensorimotor strip', 'other speech ROIs (L)',
            'rest of L hemisphere', 'rest of R hemisphere', 'R hemisphere total']
    df = df[cols]
    df.to_csv(out_dir / 'ctf_region_shares_sensorEquiv.csv', index=False)
    pd.set_option('display.width', 250)
    for filt, g in df.groupby('filter', sort=False):
        print(f'\n{filt}  (share of CTF energy, %)')
        print((g.set_index('roi')[cols[2:]] * 100).round(1).to_string())

    # ── figure: stacked bars, one panel per filter ──
    filts = list(dict.fromkeys(df['filter']))
    show = [f for f in filts if '3 ch energy' not in f]
    fig_w = min(4.6 * len(show), 26)
    parts = cols[2:-1]
    pal = ['#1b9e77', '#66c2a5', '#d95f02', '#fdae61', '#7570b3', '#bcbddc', '#e7298a', '#cccccc', '#888888']
    fig, axes = plt.subplots(1, len(show), figsize=(fig_w, 3.8), sharey=True, layout='constrained')
    axes = np.atleast_1d(axes)
    for ax, filt in zip(axes, show):
        g = df[df['filter'] == filt].set_index('roi').loc[ROIS]
        left = np.zeros(len(ROIS))
        for part, col in zip(parts, pal):
            v = g[part].values * 100
            ax.barh(range(len(ROIS)), v, left=left, color=col, label=part, height=0.7)
            left += v
        ax.set_yticks(range(len(ROIS))); ax.set_yticklabels([ABBR[r] for r in ROIS])
        ax.invert_yaxis(); ax.set_xlim(0, 100); ax.set_xlabel('% of CTF energy')
        ax.set_title(filt, fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
    axes[-1].legend(fontsize=7, frameon=False, loc='center left', bbox_to_anchor=(1.01, 0.5))
    fig.suptitle(f'Where each ROI\'s cross-talk energy comes from — {args.task}/{args.stim_class}, sensorEquiv\n'
                 'dSPM from the per-subject baseline CTFs; DeFleCT designed on the template leadfield (identity noise covariance)',
                 fontsize=10)
    fig.savefig(out_dir / 'ctf_region_shares_sensorEquiv.png', dpi=150)
    print(f'\n-> {out_dir / "ctf_region_shares_sensorEquiv"}.csv / .png, ctf_region_gains_sensorEquiv.csv')


if __name__ == '__main__':
    main()
