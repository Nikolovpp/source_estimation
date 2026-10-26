#!/usr/bin/env python
"""
Whole-brain leakage maps: where does each ROI's cross-talk energy come FROM?

The ROI metrics (se_own, pairwise |cos|) only compare the ROIs with each
other. About two thirds of every sensorEquiv ROI's signal energy comes from
cortex outside the ROI, and the pairwise numbers cannot say from where. This
script answers that from the CTF vectors the resolution sweep already saved
(RESOLUTION_ANALYSIS/{task}/sweep/per_subject/*_ctf.npz, one (n_roi, n_src)
array per configuration), with no new estimation:

  E_i(v)  = CTF_i(v)^2 / sum_v CTF_i(v)^2    fraction of ROI i's signal energy
                                              contributed by a unit white
                                              source at vertex v (group mean)
  L_i(v)  = E_i(v) outside ROI i's own vertices  (sums to 1 - se_own)
  total   = sum over speech ROIs of L_i        where the leaked energy lands
  common  = geometric mean over speech ROIs of L_i / sum L_i
                                               vertices that feed EVERY ROI
                                               (the shared-source candidate)

and aggregates E / L onto the 360 HCPMMP1 parcels (both hemispheres), so
the output is a ranked parcel table: which parcels supply the most leaked
energy to the ROIs, to how many ROIs at once, and how much of the leak is a
sensorimotor-strip (4, 3a, 3b, 1, 2, 6v, 6d, 6mp, 55b, 43, OP4) or
right-hemisphere problem. That table is what decides between a DeFleCT null
on a region, adding it as an ROI, or ignoring it.

Outputs, per task dir, under RESOLUTION_ANALYSIS/{task}/leakage_map/:
  parcels_{atlas}_{cfg}.csv      one row per HCPMMP1 parcel: energy fraction
                                 per ROI, total_leak, common, n_rois_top10
  parcels_{atlas}_{cfg}.png      top parcels, stacked by receiving ROI
  brain_{atlas}_{cfg}.png        per-ROI leakage maps (own ROI masked),
                                 total and common maps; lateral + medial lh
  summary.txt                    the numbers in words

Local run (npz in, csv/png out):
  python exploratory/ctf_leakage_map.py                       # both tasks
  python exploratory/ctf_leakage_map.py --task overtProd --configs mn_dSPM_snr3_d0.8_l0_shrunk
  python exploratory/ctf_leakage_map.py --no-brain             # tables only
"""
import os
import sys
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

from resolution_analysis import OUT_ROOT
from resolution_sweep_report import CONTROL_ROIS

DEFAULT_CONFIGS = ['mn_dSPM_snr3_d0.8_l0_shrunk',
                   'mn_dSPM_snr10_d0.8_l0_shrunk',
                   'deflect_snr10_null2_other_rois']
# HCPMMP1 sensorimotor strip (pre/postcentral + ventral premotor + OP4)
SENSORIMOTOR = ['4', '3a', '3b', '1', '2', '6v', '6d', '6mp', '55b', '43', 'OP4']
ROI_SHORT = {'Inferior_Frontal': 'IFG', 'Superior_Frontal': 'SFG',
             'Superior_Parietal': 'SPL', 'Temporal': 'Temp', 'Visual': 'Vis'}


def short(name):
    return ROI_SHORT.get(name, name.replace('-lh', ''))


# ─────────────────────────────────────────────────────────────────────
def parcel_rows_simple(src, subjects_dir):
    """HCPMMP1 parcel name -> source-row indices (lh rows, then rh rows + n_lh)."""
    vert = [src[0]['vertno'], src[1]['vertno']]
    n_lh = len(vert[0])
    labels = mne.read_labels_from_annot('fsaverage', parc='HCPMMP1', hemi='both',
                                        subjects_dir=subjects_dir, verbose=False)
    rows = {}
    for lab in labels:
        if lab.name.startswith('???'):
            continue
        h = 0 if lab.hemi == 'lh' else 1
        common = np.intersect1d(vert[h], lab.vertices)
        if len(common) == 0:
            continue
        idx = np.searchsorted(vert[h], common)
        rows[lab.name] = idx + (n_lh if h == 1 else 0)
    return rows


def roi_rows_from_baseline(task_dir, atlas):
    ps = task_dir / 'per_subject'
    f = sorted(ps.glob(f'*_dSPM_{atlas}.npz'))
    if not f:
        raise FileNotFoundError(f'no baseline npz for {atlas} under {ps}')
    d = np.load(f[0], allow_pickle=True)
    return {str(n): np.asarray(r, int) for n, r in zip(d['roi_names'], d['roi_rows'])}


def group_energy(sweep_dir, atlas, cfg):
    """Group-mean energy fraction maps E (n_roi, n_src) and ROI names."""
    files = sorted((sweep_dir / 'per_subject').glob('*_ctf.npz'))
    acc, names, n = None, None, 0
    key = f'{atlas}__{cfg}'
    for f in files:
        d = np.load(f, allow_pickle=True)
        if key not in d:
            continue
        c = d[key].astype(float)
        e = c ** 2
        e /= np.maximum(e.sum(axis=1, keepdims=True), 1e-300)
        acc = e if acc is None else acc + e
        names = [str(x) for x in d[f'names__{atlas}']]
        n += 1
    if acc is None:
        return None, None, 0
    return acc / n, names, n


# ─────────────────────────────────────────────────────────────────────
def parcel_table(E, names, roi_rows, parcels, speech):
    rows = []
    L = E.copy()
    for i, nm in enumerate(names):
        L[i, roi_rows[nm]] = 0.0            # own ROI masked
    # rank of each parcel as a leak source for each ROI
    per_roi_rank = {}
    for i, nm in enumerate(names):
        vals = {p: L[i, idx].sum() for p, idx in parcels.items()}
        order = sorted(vals, key=vals.get, reverse=True)
        per_roi_rank[nm] = {p: k + 1 for k, p in enumerate(order)}
    for p, idx in parcels.items():
        rec = {'parcel': p, 'hemi': p[-2:], 'n_src': len(idx)}
        leaks = []
        for i, nm in enumerate(names):
            rec[f'E_{short(nm)}'] = E[i, idx].sum()
            rec[f'L_{short(nm)}'] = L[i, idx].sum()
            if nm in speech:
                leaks.append(L[i, idx].sum())
        leaks = np.array(leaks)
        rec['total_leak'] = leaks.sum()
        rec['min_leak'] = leaks.min()
        rec['common'] = float(np.exp(np.log(np.maximum(leaks, 1e-12)).mean()))
        rec['n_rois_top10'] = sum(per_roi_rank[nm][p] <= 10 for nm in speech)
        rows.append(rec)
    df = pd.DataFrame(rows).sort_values('total_leak', ascending=False)
    return df, L


def summarize(df, E, L, names, roi_rows, speech, parcels, cfg, atlas, n_subj):
    lines = [f'== {atlas} / {cfg}  ({n_subj} subjects; speech ROIs = {", ".join(speech)})']
    for i, nm in enumerate(names):
        own = E[i, roi_rows[nm]].sum()
        rh = sum(E[i, idx].sum() for p, idx in parcels.items() if p.endswith('-rh'))
        sm = sum(L[i, idx].sum() for p, idx in parcels.items()
                 if p.startswith('L_') and p[2:-7] in SENSORIMOTOR)
        top = df.sort_values(f'L_{short(nm)}', ascending=False).head(5)
        tops = ', '.join(f'{r.parcel[2:-7]}{"(R)" if r.hemi == "rh" else ""} '
                         f'{getattr(r, f"L_{short(nm)}"):.1%}' for r in top.itertuples())
        lines.append(f'  {nm:18s} own {own:.0%} | leak: sensorimotor-strip {sm:.1%}, '
                     f'right hemi {rh:.1%} | top sources: {tops}')
    tot = df.total_leak.sum()
    sm_tot = df[df.parcel.str.startswith('L_')
                & df.parcel.str[2:-7].isin(SENSORIMOTOR)].total_leak.sum()
    rh_tot = df[df.hemi == 'rh'].total_leak.sum()
    lines.append(f'  summed over speech ROIs: leaked energy {tot:.2f} ROI-equivalents; '
                 f'{sm_tot / tot:.0%} from the sensorimotor strip, {rh_tot / tot:.0%} from the right hemisphere')
    lines.append('  top parcels by total leak (fraction of each ROI\'s energy; n = how many speech ROIs rank it top-10):')
    for r in df.head(12).itertuples():
        per = ' '.join(f'{short(nm)} {getattr(r, f"L_{short(nm)}"):.1%}' for nm in names)
        lines.append(f'    {r.parcel[2:-7]:>10s}{"(R)" if r.hemi == "rh" else "   "}  total {r.total_leak:.1%}  n={r.n_rois_top10}  [{per}]')
    lines.append('  top parcels by COMMON leak (feed every speech ROI):')
    for r in df.sort_values('common', ascending=False).head(8).itertuples():
        lines.append(f'    {r.parcel[2:-7]:>10s}{"(R)" if r.hemi == "rh" else "   "}  '
                     f'common {r.common:.1%}  min {r.min_leak:.1%}  n={r.n_rois_top10}')
    return lines


# ─────────────────────────────────────────────────────────────────────
def plot_parcels(df, names, speech, out_png, title, top=20):
    d = df.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.38 * top + 1.8))
    left = np.zeros(len(d))
    cmap = plt.get_cmap('tab10')
    for k, nm in enumerate(speech):
        v = d[f'L_{short(nm)}'].values * 100
        ax.barh(range(len(d)), v, left=left, color=cmap(k), label=nm)
        left += v
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([f'{p[2:-7]}{" (R)" if p.endswith("-rh") else ""}'
                        for p in d.parcel], fontsize=8)
    ax.set_xlabel('leaked energy, summed over speech ROIs (% of each ROI\'s energy)')
    ax.legend(fontsize=8, loc='lower right')
    ax.set_title(title, fontsize=10)
    ax.grid(axis='x', alpha=.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def render_maps(maps, titles, n_lh, src, subjects_dir, out_png, suptitle,
                views=('lateral', 'medial')):
    try:
        from visualize_rois import _load_surface, _render_brain_views
    except Exception as e:
        print(f'  brain maps skipped ({e})')
        return
    coords, faces = _load_surface(subjects_dir, 'lh', surf='pial', smooth=0.25)
    vert_lh = src[0]['vertno']
    n = len(maps)
    fig, axes = plt.subplots(len(views), n, figsize=(3.4 * n, 3.1 * len(views)))
    axes = np.atleast_2d(axes)
    for k, (m, t) in enumerate(zip(maps, titles)):
        m = np.asarray(m, float)
        m = m / max(m.max(), 1e-30)
        stc = mne.SourceEstimate(m[:n_lh, None], vertices=[vert_lh, np.array([], int)],
                                 tmin=0, tstep=1, subject='fsaverage')
        morph = mne.compute_source_morph(stc, 'fsaverage', 'fsaverage', spacing=None,
                                         smooth=5, subjects_dir=subjects_dir, verbose=False)
        full = morph.apply(stc).data[:, 0][:len(coords)]
        full = full / max(full.max(), 1e-30)
        scal = np.where(full > 0.05, full, np.nan).astype(float)
        for r, view in enumerate(views):
            img = _render_brain_views(coords, faces, [view], 'lh', size=(900, 700),
                                      vertex_scalars=scal, cmap='hot', clim=(0, 1))[view]
            axes[r, k].imshow(img)
            axes[r, k].set_axis_off()
            if r == 0:
                axes[r, k].set_title(t, fontsize=9)
            if k == 0:
                axes[r, k].text(-0.02, 0.5, view, transform=axes[r, k].transAxes,
                                rotation=90, va='center', ha='right', fontsize=9)
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
def run_task(task_dir, atlas, configs, brain, src, subjects_dir, parcels):
    sweep_dir = task_dir / 'sweep'
    out_dir = task_dir / 'leakage_map'
    out_dir.mkdir(exist_ok=True)
    title = task_dir.name.replace('_', '/')
    roi_rows = roi_rows_from_baseline(task_dir, atlas)
    lines = [f'CTF leakage maps - {title}', '']
    for cfg in configs:
        E, names, n_subj = group_energy(sweep_dir, atlas, cfg)
        if E is None:
            print(f'  [skip] {atlas}/{cfg}: not in the sweep output')
            continue
        n_lh = len(src[0]['vertno'])
        speech = [nm for nm in names if nm not in CONTROL_ROIS]
        df, L = parcel_table(E, names, roi_rows, parcels, speech)
        df.to_csv(out_dir / f'parcels_{atlas}_{cfg}.csv', index=False)
        lines += summarize(df, E, L, names, roi_rows, speech, parcels, cfg, atlas, n_subj)
        lines.append('')
        plot_parcels(df, names, speech, out_dir / f'parcels_{atlas}_{cfg}.png',
                     f'{title} - {atlas} - {cfg}\nwhere the leaked energy comes from '
                     f'(HCPMMP1 parcels, both hemispheres; own-ROI parcels excluded)')
        if brain:
            idx = [names.index(nm) for nm in speech]
            total = L[idx].sum(axis=0)
            dist = L[idx] / np.maximum(L[idx].sum(axis=1, keepdims=True), 1e-300)
            common = np.exp(np.log(np.maximum(dist, 1e-12)).mean(axis=0))
            common[total <= 0] = 0
            maps = [L[i] for i in range(len(names))] + [total, common]
            titles = [f'{short(nm)}: leak sources\n(own ROI masked)' for nm in names] + \
                     ['total leak\n(sum over speech ROIs)', 'common leak\n(feeds every speech ROI)']
            render_maps(maps, titles, n_lh, src, subjects_dir,
                        out_dir / f'brain_{atlas}_{cfg}.png',
                        f'{title} - {atlas} - {cfg}: where each ROI\'s cross-talk energy '
                        f'comes from (group mean, each map scaled to its max)')
        print(f'  {atlas}/{cfg}: done')
    txt = '\n'.join(lines)
    (out_dir / 'summary.txt').write_text(txt)
    print(txt)
    print(f'-> {out_dir}')


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--task', choices=['overtProd', 'perception'])
    ap.add_argument('--stim-class', choices=['prodDiff', 'percDiff'])
    ap.add_argument('--atlas', default='sensorEquiv')
    ap.add_argument('--configs', nargs='+', default=DEFAULT_CONFIGS)
    ap.add_argument('--no-brain', action='store_true')
    ap.add_argument('--root', default=str(OUT_ROOT))
    return ap.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)
    if args.task:
        stim = args.stim_class or ('prodDiff' if args.task == 'overtProd' else 'percDiff')
        dirs = [root / f'{args.task}_{stim}']
    else:
        dirs = sorted(d for d in root.iterdir() if d.is_dir() and (d / 'sweep').exists())
    from forward_model import setup_fsaverage
    subjects_dir, _, src, _ = setup_fsaverage()
    parcels = parcel_rows_simple(src, subjects_dir)
    print(f'{len(parcels)} HCPMMP1 parcels mapped onto {sum(len(v) for v in parcels.values())} source rows')
    for d in dirs:
        run_task(d, args.atlas, args.configs, not args.no_brain, src, subjects_dir, parcels)


if __name__ == '__main__':
    main()
