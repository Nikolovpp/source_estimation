#!/usr/bin/env python
"""Signed difference of two inverses' CTF maps: method A minus method B.

Answers "do dSPM and eLORETA really give different CTF maps?" in one
figure.  For every ROI the group-mean |CTF| of each method is scaled to its
own maximum (exactly as in ``ctf_maps_*.png``), the two normalized maps are
subtracted, and the signed difference is drawn on a diverging scale:

    red  = method A hears this vertex relatively more than method B
    blue = method B hears it more
    gray = |difference| below ``--thresh`` (default 0.02 of the per-ROI max)

Reads the baseline per-subject files written by ``resolution_analysis.py``
(``RESOLUTION_ANALYSIS/{task}_{stim}/per_subject/{subj}_{method}_{atlas}.npz``)
and prints, per ROI, the max and mean |difference| over the vertices drawn
in the original maps, so the picture has numbers next to it.

Outputs (same folder as the baseline maps):
    ctf_maps_diff_{A}-{B}_{atlas}.png

Usage (local, plotting only):
    python exploratory/ctf_method_diff_map.py --task overtProd --stim-class prodDiff
    python exploratory/ctf_method_diff_map.py --task perception --stim-class percDiff \
        --methods dSPM sLORETA --atlases sensorEquiv
"""
import os
import sys
import argparse
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne

from resolution_analysis import OUT_ROOT


def group_norm_map(per_subject, method, atlas, weights='pc1'):
    """Group-mean |CTF| per ROI, each subject's ROI row scaled to its max
    (the quantity ``ctf_maps_*.png`` shows).  Returns (acc, names, n_lh)."""
    files = sorted(per_subject.glob(f'*_{method}_{atlas}.npz'))
    if not files:
        raise FileNotFoundError(f'no {method}/{atlas} files in {per_subject}')
    acc, names, n_lh = None, None, None
    for f in files:
        d = np.load(f, allow_pickle=True)
        c = np.abs(d[f'ctf_{weights}'])
        c = c / c.max(axis=1, keepdims=True)
        acc = c if acc is None else acc + c
        names = list(d['roi_names'])
        n_lh = int(d['n_lh'])
    return acc / len(files), names, n_lh


def render_diff(diff, drawn, names, n_lh, src, subjects_dir, title, out_png,
                thresh, vlim):
    from visualize_rois import _load_surface, _render_brain_views
    coords, faces = _load_surface(subjects_dir, 'lh', surf='pial', smooth=0.25)
    vert_lh = src[0]['vertno']
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n + 0.9, 3.4),
                             layout='constrained')
    axes = np.atleast_1d(axes)
    for k, name in enumerate(names):
        stc = mne.SourceEstimate(
            diff[k][:n_lh, None], vertices=[vert_lh, np.array([], int)],
            tmin=0, tstep=1, subject='fsaverage')
        morph = mne.compute_source_morph(
            stc, 'fsaverage', 'fsaverage', spacing=None, smooth=5,
            subjects_dir=subjects_dir, verbose=False)
        full = morph.apply(stc).data[:, 0][:len(coords)]
        scal = np.where(np.abs(full) >= thresh, full, np.nan).astype(float)
        img = _render_brain_views(coords, faces, ['lateral'], 'lh',
                                  size=(900, 700), vertex_scalars=scal,
                                  cmap='RdBu_r', clim=(-vlim, vlim))['lateral']
        axes[k].imshow(img)
        axes[k].set_axis_off()
        axes[k].set_title(name.replace('-lh', ''), fontsize=10)
    fig.suptitle(title, fontsize=10)
    sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=plt.Normalize(-vlim, vlim))
    cb = fig.colorbar(sm, ax=axes.tolist(), shrink=0.75, pad=0.01)
    cb.set_label('difference of normalized |CTF|', fontsize=8)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--task', required=True, choices=['perception', 'overtProd'])
    ap.add_argument('--stim-class', default='prodDiff', choices=['prodDiff', 'percDiff'])
    ap.add_argument('--methods', nargs=2, default=['dSPM', 'eLORETA'],
                    metavar=('A', 'B'), help='plot A minus B (default dSPM eLORETA)')
    ap.add_argument('--atlases', nargs='+', default=['sensorEquiv', 'custom'])
    ap.add_argument('--weights', default='pc1', choices=['pc1', 'mean_flip'])
    ap.add_argument('--thresh', type=float, default=0.02,
                    help='|difference| below this is not drawn (default 0.02)')
    ap.add_argument('--vlim', type=float, default=None,
                    help='color limit (default: 99.5th percentile of |diff| over '
                         'all ROIs, at least 0.05)')
    ap.add_argument('--out-dir', default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else \
        OUT_ROOT / f'{args.task}_{args.stim_class}'
    ps = out_dir / 'per_subject'
    A, B = args.methods

    from forward_model import setup_fsaverage
    subjects_dir, _, src, _ = setup_fsaverage()

    for atlas in args.atlases:
        try:
            ma, names, n_lh = group_norm_map(ps, A, atlas, args.weights)
            mb, names_b, _ = group_norm_map(ps, B, atlas, args.weights)
        except FileNotFoundError as e:
            print(f'  skip {atlas}: {e}')
            continue
        assert names == names_b, (names, names_b)
        diff = ma - mb
        drawn = (ma > 0.05) | (mb > 0.05)        # what the originals show
        vlim = args.vlim or max(0.05, float(np.percentile(np.abs(diff[drawn]), 99.5)))
        print(f'\n{args.task}/{args.stim_class}  {atlas}  {A} - {B}  '
              f'(normalized |CTF|, color limit ±{vlim:.2f}, drawn |diff| >= {args.thresh})')
        print(f'  {"ROI":18s} {"max|diff|":>9s} {"mean|diff|":>10s} '
              f'{"% drawn >= thr":>14s} {"cos(A,B)":>9s}')
        for k, nm in enumerate(names):
            dk, wk = diff[k], drawn[k]
            cos = float(ma[k] @ mb[k] / np.linalg.norm(ma[k]) / np.linalg.norm(mb[k]))
            frac = 100 * float((np.abs(dk[wk]) >= args.thresh).mean())
            print(f'  {nm.replace("-lh", ""):18s} {np.abs(dk[wk]).max():9.3f} '
                  f'{np.abs(dk[wk]).mean():10.3f} {frac:13.0f}% {cos:9.3f}')
        title = (f'{A} minus {B}: group-mean |CTF|, each method scaled to its own '
                 f'per-ROI max — {args.task}/{args.stim_class} / {atlas} / {args.weights}\n'
                 f'red = {A} relatively more sensitive here, blue = {B} more; '
                 f'|diff| < {args.thresh} not drawn; colors clip at ±{vlim:.2f}')
        out_png = out_dir / f'ctf_maps_diff_{A}-{B}_{atlas}.png'
        render_diff(diff, drawn, names, n_lh, src, subjects_dir, title, out_png,
                    args.thresh, vlim)
        print(f'  -> {out_png}')


if __name__ == '__main__':
    main()
