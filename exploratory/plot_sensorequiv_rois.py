#!/usr/bin/env python
"""Where the sensorEquiv ROIs and the sensorimotor strip sit on the brain.

Figures (derivatives/source_estimation/ROI_maps/sensorEquiv/):
    sensorEquiv_overview.png        the five composite ROIs + the HCPMMP1
                                    sensorimotor strip (areas 4/3a/3b/1/2),
                                    one color each; lateral, medial, dorsal
    sensorEquiv_speech_plus_SM.png  the four speech ROIs + the strip only
    parcels_{ROI}.png               one figure per ROI: its HCPMMP1 parcels in
                                    separate colors with the parcel names
    parcels_Sensorimotor.png        the strip's five areas
Full-resolution fsaverage parcels (the source space samples these at ico-5).

Usage (local, plotting only):
    python exploratory/plot_sensorequiv_rois.py
    python exploratory/plot_sensorequiv_rois.py --views lateral dorsal --surf inflated
"""
import os
import re
import sys
import argparse
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import matplotlib
matplotlib.use('Agg')

from config import SPEECH_ROIS, DEFLECT_SM_PARCELS, DECODE_OUTPUT_ROOT
from forward_model import setup_fsaverage, build_roi_labels
from visualize_rois import plot_roi_brain, plot_single_roi

ROIS = ['Temporal', 'Inferior_Frontal', 'Superior_Frontal', 'Superior_Parietal', 'Visual']
SM_NAME = 'Sensorimotor_strip'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--views', nargs='+', default=['lateral', 'medial', 'dorsal'])
    ap.add_argument('--surf', default='pial', choices=['pial', 'inflated', 'white'])
    ap.add_argument('--smooth', type=float, default=0.25)
    ap.add_argument('--mode', default='full', choices=['full', 'ico5'],
                    help='full-resolution parcels (default) or the ico-5 source vertices only')
    ap.add_argument('--out-dir', default=None)
    args = ap.parse_args()

    out = Path(args.out_dir) if args.out_dir else Path(DECODE_OUTPUT_ROOT).parent / 'ROI_maps' / 'sensorEquiv'
    out.mkdir(parents=True, exist_ok=True)
    subjects_dir, _, _, _ = setup_fsaverage()

    comp = {r: list(SPEECH_ROIS['sensorEquiv'][r]) for r in ROIS}
    comp[SM_NAME] = [f'L_{p}_ROI-lh' for p in DEFLECT_SM_PARCELS]
    labels = build_roi_labels(subjects_dir, atlas='HCPMMP1', composite_rois=comp)

    kw = dict(subjects_dir=subjects_dir, views=args.views, hemi='lh', save=False,
              out_dir=out, mode=args.mode, surf=args.surf, smooth=args.smooth)

    # 1. everything at once
    fig = plot_roi_brain(roi_dict={k: labels[k] for k in ROIS + [SM_NAME]}, atlas='HCPMMP1', **kw)
    fig.suptitle('sensorEquiv ROIs (HCPMMP1 composites under the four sensor pseudo-channels), '
                 'the Visual control and the sensorimotor strip (areas 4, 3a, 3b, 1, 2)', fontsize=12)
    fig.savefig(out / 'sensorEquiv_overview.png', dpi=150, bbox_inches='tight')
    print(f'-> {out / "sensorEquiv_overview.png"}')

    # 2. the four speech ROIs with the strip between them
    fig = plot_roi_brain(roi_dict={k: labels[k] for k in ROIS[:4] + [SM_NAME]}, atlas='HCPMMP1', **kw)
    fig.suptitle('the four speech ROIs and the sensorimotor strip that borders Superior Frontal and Superior Parietal',
                 fontsize=12)
    fig.savefig(out / 'sensorEquiv_speech_plus_SM.png', dpi=150, bbox_inches='tight')
    print(f'-> {out / "sensorEquiv_speech_plus_SM.png"}')

    # 3. one figure per ROI, parcels in separate colors
    for name, parcels in comp.items():
        fig = plot_single_roi(name, parcel_names=parcels, atlas='HCPMMP1', **kw)
        short = [re.sub(r'^L_', '', p).replace('_ROI-lh', '') for p in parcels]
        fig.suptitle(f'{name}: {len(parcels)} HCPMMP1 parcels — {", ".join(short)}', fontsize=11)
        f = out / f'parcels_{name}.png'
        fig.savefig(f, dpi=150, bbox_inches='tight')
        print(f'-> {f}')


if __name__ == '__main__':
    main()
