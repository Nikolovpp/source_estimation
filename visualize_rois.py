"""
Visualize the actual source-space vertices used for ROI-based decoding.

Shows ROI coverage on the fsaverage cortical surface in two modes:

- **ico5** (default): the sparse ico-5 source-space vertices that the
  inverse solution actually reconstructs.
- **full**: the full-resolution parcel vertices from the atlas.

Supports three atlas parcellations via ``--atlas``:

- ``aparc`` (default): Desikan-Killiany, 8 composite ROIs
- ``Schaefer200``: Schaefer 2018, 200 parcels (17-network variant)
- ``HCPMMP1``: HCP Multi-Modal Parcellation, 360 parcels

Usage
-----
    # Basic atlas views
    python visualize_rois.py                                        # aparc, ico-5
    python visualize_rois.py --atlas HCPMMP1 --hemi lh              # HCP-MMP left hemi

    # Speech-network ROIs (16 regions from config.SPEECH_ROIS)
    python visualize_rois.py --speech-rois --atlas HCPMMP1 --save   # all 16 ROIs
    python visualize_rois.py --roi Anterior_STS --atlas HCPMMP1     # single named ROI
    python visualize_rois.py --roi vSMC --atlas Schaefer200 --save  # Chang vSMC
    python visualize_rois.py --list-rois --atlas HCPMMP1            # list ROI names

    # SVG output for publication figures
    python visualize_rois.py --speech-rois --atlas HCPMMP1 --save --format svg
    python visualize_rois.py --roi Temporal --atlas HCPMMP1 --save --format svg

    # Full atlas / compare modes
    python visualize_rois.py --atlas Schaefer200 --mode full        # full-res parcels
    python visualize_rois.py --mode compare --format svg --save     # full vs ico-5
"""
import argparse
from pathlib import Path

import numpy as np
import mne
from mne.datasets import fetch_fsaverage

from config import (SPEECH_ROIS, SPEECH_ROI_NAMES, ATLAS_PARC_MAP,
                     FIGURES_ROOT)
from forward_model import build_roi_labels

# Colorblind-friendly palette: Wong (2011) + Tol muted, 20 distinct colours.
# Covers up to 20 ROIs; for larger atlases, cycles with offset luminance.
_CB_PALETTE = [
    '#0077BB',  # blue
    '#EE7733',  # orange
    '#009988',  # teal
    '#CC3311',  # red
    '#33BBEE',  # cyan
    '#EE3377',  # magenta
    '#BBBBBB',  # grey
    '#000000',  # black
    '#AA3377',  # wine
    '#DDCC77',  # sand
    '#44BB99',  # mint
    '#882255',  # purple
    '#332288',  # indigo
    '#88CCEE',  # light cyan
    '#999933',  # olive
    '#661100',  # brown
    '#117733',  # green
    '#CC6677',  # rose
    '#6699CC',  # steel
    '#AA4499',  # plum
]


def _get_roi_colours(roi_names):
    """Return one colour per ROI name from a colorblind-friendly palette."""
    return [_CB_PALETTE[i % len(_CB_PALETTE)] for i in range(len(roi_names))]


def _restrict_labels_to_src(roi_dict, src):
    """
    Restrict ROI labels to only contain vertices present in the source
    space, returning new labels with the sparse ico-5 vertex set.
    """
    restricted = {}
    for roi_name, label in roi_dict.items():
        rlabel = label.restrict(src)
        restricted[roi_name] = rlabel
    return restricted

def build_speech_roi_labels(atlas, subjects_dir=None):
    """
    Build composite mne.Labels for each speech-network ROI.

    Returns {roi_name: mne.Label} with one merged label per speech ROI.
    """
    if subjects_dir is None:
        fs_dir = fetch_fsaverage(verbose=False)
        subjects_dir = str(fs_dir.parent)

    parc = ATLAS_PARC_MAP.get(atlas, atlas)
    labels_all = mne.read_labels_from_annot(
        'fsaverage', parc=parc, hemi='both', subjects_dir=subjects_dir
    )
    label_lookup = {l.name: l for l in labels_all}

    roi_defs = SPEECH_ROIS.get(atlas, {})
    speech_labels = {}
    for roi_name in SPEECH_ROI_NAMES:
        parcel_names = roi_defs.get(roi_name, [])
        parcels = [label_lookup[p] for p in parcel_names if p in label_lookup]
        missing = [p for p in parcel_names if p not in label_lookup]
        if missing:
            print(f'  Warning: {roi_name}: {len(missing)} parcels not found '
                  f'in {atlas}: {missing[:3]}...')
        if not parcels:
            continue
        merged = parcels[0].copy()
        for p in parcels[1:]:
            merged += p
        merged.name = roi_name
        speech_labels[roi_name] = merged
    return speech_labels


def _save_brain_views(brain, views, out_dir, name_prefix, fmt='png',
                      legend_items=None):
    """Save brain views as composed figure with optional colour legend.

    Parameters
    ----------
    legend_items : list of (name, colour) | None
        If provided, a colour legend is added to the right of the figure
        and the output is always a single composed file (even for PNG).
    """
    import matplotlib.pyplot as plt
    from matplotlib.image import imread
    from matplotlib.patches import Patch
    import tempfile

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Simple PNG-per-view when no legend is needed
    if fmt == 'png' and legend_items is None:
        for view in views:
            brain.show_view(view)
            fpath = out_dir / f'{name_prefix}_{view}.png'
            brain.save_image(str(fpath))
            print(f'Saved: {fpath}')
        brain.close()
        return

    # Composed matplotlib figure (SVG, or PNG-with-legend)
    n = len(views)
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 8), squeeze=False)
    for col, view in enumerate(views):
        brain.show_view(view)
        tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        brain.save_image(tmp.name)
        axes[0, col].imshow(imread(tmp.name))
        axes[0, col].set_title(view, fontsize=14)
        axes[0, col].axis('off')
        tmp.close()
    brain.close()

    if legend_items:
        handles = [Patch(facecolor=c, edgecolor='grey', label=n)
                   for n, c in legend_items]
        fig.legend(handles=handles, loc='center left',
                   bbox_to_anchor=(1.0, 0.5), fontsize=10, frameon=True)

    fig.suptitle(name_prefix.replace('_', ' '), fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    ext = fmt if fmt in ('png', 'svg') else 'png'
    fpath = out_dir / f'{name_prefix}.{ext}'
    fig.savefig(str(fpath), format=ext, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {fpath}')


def plot_roi_brain(roi_dict=None, subjects_dir=None, views=None,
                   hemi=None, save=False, out_dir=None,
                   mode='ico5', atlas='aparc', fmt='png',
                   brain_kwargs=None):
    """
    Plot ROI labels on the fsaverage cortical surface.

    Parameters
    ----------
    roi_dict : dict | None
        {roi_name: mne.Label}.  If None, built from *atlas*.
    subjects_dir : Path | str | None
        MNE subjects_dir containing 'fsaverage'.  Auto-detected if None.
    views : list of str | None
        Brain views to show, e.g. ['lateral', 'medial', 'ventral'].
        Defaults to ['lateral', 'medial'].
    hemi : str
        'lh', 'rh', or 'both'.  Which hemisphere(s) to render.
    save : bool
        If True, save screenshots to *out_dir* and close instead of
        showing the interactive viewer.
    out_dir : Path | str | None
        Directory for saved PNGs.  Defaults to FIGURES_ROOT / 'ROI_maps'.
    mode : str
        'ico5' to show only source-space vertices (default), or 'full'
        to show full-resolution atlas parcel vertices.
    atlas : str
        Atlas parcellation: 'aparc', 'Schaefer200', or 'HCPMMP1'.
    brain_kwargs : dict | None
        Extra keyword arguments forwarded to ``mne.viz.Brain``.

    Returns
    -------
    brain : mne.viz.Brain
        The Brain instance (useful for further interaction in a notebook).
    """
    # ── resolve subjects_dir ──────────────────────────────────────────
    if subjects_dir is None:
        fs_dir = fetch_fsaverage(verbose=False)
        subjects_dir = str(fs_dir.parent)
    else:
        fs_dir = Path(subjects_dir) / 'fsaverage'

    # ── build ROI labels ──────────────────────────────────────────────
    if roi_dict is None:
        if atlas == 'aparc':
            roi_dict = build_roi_labels(subjects_dir, atlas='aparc',
                                         composite_rois=SPEECH_ROIS['aparc'])
        else:
            roi_dict = build_roi_labels(subjects_dir, atlas=atlas)

    roi_dict_full = roi_dict

    # ── restrict to source-space vertices if requested ────────────────
    if mode == 'ico5':
        src = mne.read_source_spaces(
            fs_dir / 'bem' / 'fsaverage-ico-5-src.fif', verbose=False
        )
        roi_dict = _restrict_labels_to_src(roi_dict, src)

    if views is None:
        views = ['lateral', 'medial']

    # ── determine hemisphere from labels if not explicitly set ────────
    if hemi is None:
        hemis_present = {label.hemi for label in roi_dict.values()}
        if hemis_present == {'lh'}:
            hemi = 'lh'
        elif hemis_present == {'rh'}:
            hemi = 'rh'
        else:
            hemi = 'both'

    # ── create the Brain ──────────────────────────────────────────────
    bkw = dict(
        subject='fsaverage',
        subjects_dir=subjects_dir,
        hemi=hemi,
        surf='inflated',
        cortex='low_contrast',
        background='white',
        size=(1000, 800),
        views=views,
    )
    if brain_kwargs:
        bkw.update(brain_kwargs)

    brain = mne.viz.Brain(**bkw)

    # ── add each ROI ──────────────────────────────────────────────────
    colours = _get_roi_colours(list(roi_dict.keys()))
    for idx, (roi_name, label) in enumerate(roi_dict.items()):
        colour = colours[idx]
        brain.add_label(label, color=colour, alpha=0.7,
                        borders=False)
        # Skip individual borders for very large atlases (slow to render)
        if len(roi_dict) <= 50:
            brain.add_label(label, color=colour, alpha=1.0,
                            borders=True)

    # ── print vertex summary ──────────────────────────────────────────
    print(f'\n── ROI vertex coverage (atlas={atlas}, mode={mode}, '
          f'{len(roi_dict)} ROIs) ──')
    shown_items = list(roi_dict.items())
    # For large atlases, show a summary instead of all labels
    if len(shown_items) > 20:
        for roi_name, label in shown_items[:10]:
            full = roi_dict_full[roi_name]
            print(f'  {roi_name:45s}  hemi={label.hemi}  '
                  f'full={len(full.vertices):>5d}  '
                  f'shown={len(label.vertices):>5d}')
        print(f'  ... ({len(shown_items) - 20} more ROIs) ...')
        for roi_name, label in shown_items[-10:]:
            full = roi_dict_full[roi_name]
            print(f'  {roi_name:45s}  hemi={label.hemi}  '
                  f'full={len(full.vertices):>5d}  '
                  f'shown={len(label.vertices):>5d}')
    else:
        for roi_name, label in shown_items:
            full = roi_dict_full[roi_name]
            print(f'  {roi_name:45s}  hemi={label.hemi}  '
                  f'full={len(full.vertices):>5d}  '
                  f'shown={len(label.vertices):>5d}')
    print()

    # ── save or show ──────────────────────────────────────────────────
    if save:
        if out_dir is None:
            out_dir = FIGURES_ROOT / 'ROI_maps'
        legend = list(zip(roi_dict.keys(), colours))
        _save_brain_views(brain, views, out_dir,
                          f'fsaverage_{atlas}_ROIs', fmt=fmt,
                          legend_items=legend)
    else:
        print('Interactive viewer open – close the window when done.')

    return brain


def plot_single_roi(roi_name, parcel_names=None, roi_dict=None, subjects_dir=None,
                    views=None, hemi=None, save=False, out_dir=None,
                    mode='ico5', atlas='aparc', fmt='png'):
    """
    Plot one ROI (or a small set of named parcels) highlighted on the brain.

    For the aparc atlas with composite ROIs, this shows the constituent
    sub-parcels in different colours.  For Schaefer200 / HCPMMP1, pass
    the exact parcel name(s) via *roi_name* or *parcel_names*.

    Parameters
    ----------
    roi_name : str
        Key into SPEECH_ROIS['aparc'] (aparc) or an exact parcel name from the atlas.
    parcel_names : list of str | None
        Explicit list of atlas label names.  If None and atlas='aparc',
        looks up roi_name in SPEECH_ROIS['aparc'].  If None and atlas is
        Schaefer200/HCPMMP1, treats roi_name as a substring filter
        (e.g. 'Aud' matches all auditory parcels).
    atlas : str
        'aparc', 'Schaefer200', or 'HCPMMP1'.
    mode : str
        'ico5' to restrict to source-space vertices (default), or
        'full' to show full-resolution parcel vertices.
    """
    if subjects_dir is None:
        fs_dir = fetch_fsaverage(verbose=False)
        subjects_dir = str(fs_dir.parent)
    else:
        fs_dir = Path(subjects_dir) / 'fsaverage'

    src = None
    if mode == 'ico5':
        src = mne.read_source_spaces(
            fs_dir / 'bem' / 'fsaverage-ico-5-src.fif', verbose=False
        )

    if parcel_names is not None:
        # Explicit parcel list provided — load the atlas and look them up
        parc = ATLAS_PARC_MAP.get(atlas, atlas)
        labels_all = mne.read_labels_from_annot(
            'fsaverage', parc=parc, hemi='both',
            subjects_dir=subjects_dir
        )
        label_lookup = {l.name: l for l in labels_all}
        for pname in parcel_names:
            if pname not in label_lookup:
                raise ValueError(
                    f"Label '{pname}' not found in {atlas}. "
                    f"Available (first 10): {sorted(label_lookup.keys())[:10]}"
                )
        labels_to_plot = {pname: label_lookup[pname] for pname in parcel_names}

    elif roi_name in SPEECH_ROIS.get(atlas, {}):
        # Named speech-network ROI → look up parcels from config
        parcel_names = SPEECH_ROIS[atlas][roi_name]
        parc = ATLAS_PARC_MAP.get(atlas, atlas)
        labels_all = mne.read_labels_from_annot(
            'fsaverage', parc=parc, hemi='both',
            subjects_dir=subjects_dir
        )
        label_lookup = {l.name: l for l in labels_all}
        labels_to_plot = {pname: label_lookup[pname]
                          for pname in parcel_names if pname in label_lookup}

    elif atlas == 'aparc' and roi_name in SPEECH_ROIS['aparc']:
        # Legacy: aparc composite ROI → show sub-parcels
        parcel_names = SPEECH_ROIS['aparc'][roi_name]
        labels_all = mne.read_labels_from_annot(
            'fsaverage', parc='aparc', hemi='both',
            subjects_dir=subjects_dir
        )
        label_lookup = {l.name: l for l in labels_all}
        labels_to_plot = {pname: label_lookup[pname] for pname in parcel_names}

    else:
        # Non-aparc atlas or unknown ROI name: use roi_name as substring filter
        if atlas == 'aparc':
            all_roi_dict = build_roi_labels(subjects_dir, atlas='aparc',
                                             composite_rois=SPEECH_ROIS['aparc'])
        else:
            all_roi_dict = build_roi_labels(subjects_dir, atlas=atlas)

        # Filter by substring match
        labels_to_plot = {
            name: label for name, label in all_roi_dict.items()
            if roi_name in name
        }
        if not labels_to_plot:
            available = sorted(all_roi_dict.keys())[:20]
            raise ValueError(
                f'No parcels matching "{roi_name}" in {atlas}. '
                f'First 20 available: {available}'
            )

    # Determine hemisphere
    if hemi is None:
        hemis = {label.hemi for label in labels_to_plot.values()}
        hemi = hemis.pop() if len(hemis) == 1 else 'both'

    if views is None:
        views = ['lateral', 'medial']

    brain = mne.viz.Brain(
        subject='fsaverage',
        subjects_dir=subjects_dir,
        hemi=hemi,
        surf='inflated',
        cortex='low_contrast',
        background='white',
        size=(1000, 800),
        views=views,
    )

    colours = _get_roi_colours(list(labels_to_plot.keys()))
    print(f'\n── {roi_name}: {len(labels_to_plot)} parcels '
          f'(atlas={atlas}, mode={mode}) ──')
    for idx, (pname, label_full) in enumerate(labels_to_plot.items()):
        label = label_full.restrict(src) if src is not None else label_full
        colour = colours[idx]
        brain.add_label(label, color=colour, alpha=0.6, borders=False)
        if len(labels_to_plot) <= 50:
            brain.add_label(label, color=colour, alpha=1.0, borders=True)
        print(f'  {pname:45s}  colour={colour}  '
              f'full={len(label_full.vertices):>5d}  '
              f'shown={len(label.vertices):>5d}')
    print()

    if save:
        if out_dir is None:
            out_dir = FIGURES_ROOT / 'ROI_maps'
        safe_name = roi_name.replace(' ', '_')
        legend = list(zip(labels_to_plot.keys(), colours))
        _save_brain_views(brain, views, out_dir,
                          f'fsaverage_{atlas}_{safe_name}', fmt=fmt,
                          legend_items=legend)
    else:
        print('Interactive viewer open – close the window when done.')

    return brain


def plot_compare_modes(roi_name=None, parcel_names=None, roi_dict=None, subjects_dir=None,
                       views=None, hemi=None, save=False, out_dir=None,
                       atlas='aparc', fmt='png'):
    """
    Plot full-resolution (top row) vs ico-5 (bottom row) for comparison.

    If *roi_name* or *parcel_names* is given, shows that single ROI's
    parcels.  Otherwise shows all ROIs from the specified atlas.

    Parameters
    ----------
    atlas : str
        'aparc', 'Schaefer200', or 'HCPMMP1'.
    roi_name : str | None
        Single ROI to plot.  For aparc, a key in SPEECH_ROIS['aparc'].  For other
        atlases, a substring filter (e.g. 'Aud' for auditory parcels).
    views : list of str | None
        Brain views.  Defaults to ['lateral'].
    save : bool
        Save a combined figure instead of showing interactively.
    """
    import matplotlib.pyplot as plt
    from matplotlib.image import imread
    import tempfile

    if subjects_dir is None:
        fs_dir = fetch_fsaverage(verbose=False)
        subjects_dir = str(fs_dir.parent)
    else:
        fs_dir = Path(subjects_dir) / 'fsaverage'

    if views is None:
        views = ['lateral']

    src = mne.read_source_spaces(
        fs_dir / 'bem' / 'fsaverage-ico-5-src.fif', verbose=False
    )

    # --- Build labels for both modes ---
    if parcel_names is not None:
        # Explicit parcel list
        parc = ATLAS_PARC_MAP.get(atlas, atlas)
        labels_all = mne.read_labels_from_annot(
            'fsaverage', parc=parc, hemi='both',
            subjects_dir=subjects_dir
        )
        label_lookup = {l.name: l for l in labels_all}
        full_labels = {pname: label_lookup[pname] for pname in parcel_names}
        title_tag = roi_name if roi_name else 'CustomROI'

    elif roi_name is not None:
        # Single ROI by name or substring
        if atlas == 'aparc' and roi_name in SPEECH_ROIS['aparc']:
            parcel_names = SPEECH_ROIS['aparc'][roi_name]
            labels_all = mne.read_labels_from_annot(
                'fsaverage', parc='aparc', hemi='both',
                subjects_dir=subjects_dir
            )
            label_lookup = {l.name: l for l in labels_all}
            full_labels = {pname: label_lookup[pname] for pname in parcel_names}
        else:
            if atlas == 'aparc':
                all_roi_dict = build_roi_labels(subjects_dir, atlas='aparc',
                                                 composite_rois=SPEECH_ROIS['aparc'])
            else:
                all_roi_dict = build_roi_labels(subjects_dir, atlas=atlas)
            full_labels = {
                name: label for name, label in all_roi_dict.items()
                if roi_name in name
            }
            if not full_labels:
                raise ValueError(f'No parcels matching "{roi_name}" in {atlas}')
        title_tag = roi_name
    else:
        # All ROIs
        if roi_dict is None:
            if atlas == 'aparc':
                roi_dict = build_roi_labels(subjects_dir, atlas='aparc',
                                             composite_rois=SPEECH_ROIS['aparc'])
            else:
                roi_dict = build_roi_labels(subjects_dir, atlas=atlas)
        full_labels = roi_dict
        title_tag = f'{atlas}_all_ROIs'

    ico5_labels = _restrict_labels_to_src(full_labels, src)

    # Determine hemisphere
    if hemi is None:
        hemis_present = {l.hemi for l in full_labels.values()}
        if hemis_present == {'lh'}:
            hemi = 'lh'
        elif hemis_present == {'rh'}:
            hemi = 'rh'
        else:
            hemi = 'both'

    # --- Render each mode to temp images ---
    colours = _get_roi_colours(list(full_labels.keys()))

    def _render_brain(labels_dict):
        brain = mne.viz.Brain(
            subject='fsaverage',
            subjects_dir=subjects_dir,
            hemi=hemi,
            surf='inflated',
            cortex='low_contrast',
            background='white',
            size=(1000, 800),
            views=views[0],
        )
        for idx, (name, label) in enumerate(labels_dict.items()):
            colour = colours[idx]
            brain.add_label(label, color=colour, alpha=0.7, borders=False)
            if len(labels_dict) <= 50:
                brain.add_label(label, color=colour, alpha=1.0, borders=True)

        images = {}
        for view in views:
            brain.show_view(view)
            tmpfile = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            brain.save_image(tmpfile.name)
            images[view] = imread(tmpfile.name)
            tmpfile.close()
        brain.close()
        return images

    print(f'Rendering {atlas} full-resolution parcels...')
    full_images = _render_brain(full_labels)
    print(f'Rendering {atlas} ico-5 source-space vertices...')
    ico5_images = _render_brain(ico5_labels)

    # --- Print vertex summary ---
    items = list(full_labels.items())
    print(f'\n── Vertex comparison: {title_tag} ({len(items)} ROIs) ──')
    display_items = items[:15] if len(items) > 15 else items
    for name, label in display_items:
        n_full = len(label.vertices)
        n_ico5 = len(ico5_labels[name].vertices)
        print(f'  {name:45s}  full={n_full:>5d}  ico5={n_ico5:>5d}  '
              f'ratio={n_ico5/n_full:.2%}')
    if len(items) > 15:
        print(f'  ... ({len(items) - 15} more ROIs) ...')
    print()

    # --- Compose matplotlib figure: top=full, bottom=ico5 ---
    n_cols = len(views)
    fig, axes = plt.subplots(2, n_cols, figsize=(8 * n_cols, 12),
                              squeeze=False)

    for col, view in enumerate(views):
        axes[0, col].imshow(full_images[view])
        axes[0, col].set_title(f'{atlas} full — {view}', fontsize=16)
        axes[0, col].axis('off')

        axes[1, col].imshow(ico5_images[view])
        axes[1, col].set_title(f'{atlas} ico-5 — {view}', fontsize=16)
        axes[1, col].axis('off')

    fig.suptitle(f'{title_tag}: full vs ico-5 source space',
                 fontsize=20, y=0.95)
    plt.tight_layout()

    if save:
        if out_dir is None:
            out_dir = FIGURES_ROOT / 'ROI_maps'
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        view_tag = '_'.join(views)
        safe_tag = title_tag.replace(' ', '_')
        ext = fmt if fmt in ('png', 'svg') else 'png'
        fpath = out_dir / f'fsaverage_{safe_tag}_full_vs_ico5_{view_tag}.{ext}'
        fig.savefig(fpath, dpi=200, bbox_inches='tight', format=ext)
        print(f'Saved: {fpath}')
        plt.close(fig)
    else:
        plt.show()

    return fig


# ─────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualize source-estimation ROIs on fsaverage brain.'
    )
    parser.add_argument(
        '--save', action='store_true',
        help='Save PNG screenshots instead of opening the interactive viewer.'
    )
    parser.add_argument(
        '--views', nargs='+', default=['lateral', 'medial'],
        help='Brain views to render (default: lateral medial).'
    )
    parser.add_argument(
        '--atlas', default='aparc',
        choices=['aparc', 'Schaefer200', 'HCPMMP1'],
        help='Cortical atlas for ROI parcellation (default: aparc).'
    )
    parser.add_argument(
        '--roi', type=str, default=None,
        help='Plot a single ROI (or substring filter for non-aparc atlases, '
             'e.g. --roi Aud matches all auditory parcels).'
    )
    parser.add_argument(
        '--custom-labels', nargs='+', default=None,
        help='List of aparc labels to form a custom composite ROI. '
             'E.g., --custom-labels superiortemporal-lh middletemporal-lh. '
             'Overrides --roi if both are provided.'
    )
    parser.add_argument(
        '--custom-name', type=str, default='CustomROI',
        help='Name of the custom ROI for title and saving (default: CustomROI).'
    )
    parser.add_argument(
        '--out-dir', type=str, default=None,
        help='Output directory for saved figures.'
    )
    parser.add_argument(
        '--format', choices=['png', 'svg'], default='png',
        dest='fmt',
        help='Output format: png (default) or svg (vector, with embedded '
             'raster brain renders).'
    )
    parser.add_argument(
        '--speech-rois', action='store_true',
        help='Plot the 16 speech-network ROIs defined in config.SPEECH_ROIS '
             'for the selected atlas (one colour per ROI).'
    )
    parser.add_argument(
        '--list-rois', action='store_true',
        help='Print available speech-network ROI names and exit.'
    )
    parser.add_argument(
        '--mode', choices=['ico5', 'full', 'compare'], default='ico5',
        help='Vertex mode: "ico5" shows only ico-5 source-space vertices '
             '(default), "full" shows full-resolution atlas parcels, '
             '"compare" shows full on top and ico-5 on bottom.'
    )
    parser.add_argument(
        '--hemi', choices=['lh', 'rh', 'both'], default=None,
        help='Hemisphere to plot (default: auto-detect from ROI labels).'
    )
    args = parser.parse_args()

    # --list-rois: print available speech ROI names and exit
    if args.list_rois:
        print(f'\nSpeech-network ROIs for atlas={args.atlas}:')
        roi_defs = SPEECH_ROIS.get(args.atlas, {})
        for name in SPEECH_ROI_NAMES:
            parcels = roi_defs.get(name, [])
            print(f'  {name:25s}  ({len(parcels)} parcels)')
        raise SystemExit(0)

    if args.custom_labels:
        roi_name = args.custom_name
        parcel_names = args.custom_labels
        if args.mode == 'compare':
            plot_compare_modes(
                roi_name=roi_name,
                parcel_names=parcel_names,
                views=args.views,
                save=args.save,
                out_dir=args.out_dir,
                hemi=args.hemi,
                atlas=args.atlas,
                fmt=args.fmt,
            )
        else:
            brain = plot_single_roi(
                roi_name=roi_name,
                parcel_names=parcel_names,
                views=args.views,
                save=args.save,
                out_dir=args.out_dir,
                mode=args.mode,
                hemi=args.hemi,
                atlas=args.atlas,
                fmt=args.fmt,
            )
    elif args.speech_rois:
        # Plot all 16 speech-network ROIs as merged composite labels
        speech_dict = build_speech_roi_labels(args.atlas)
        if args.mode == 'compare':
            plot_compare_modes(
                roi_dict=speech_dict,
                views=args.views,
                save=args.save,
                out_dir=args.out_dir,
                hemi=args.hemi,
                atlas=args.atlas,
                fmt=args.fmt,
            )
        else:
            brain = plot_roi_brain(
                roi_dict=speech_dict,
                views=args.views,
                save=args.save,
                out_dir=args.out_dir,
                mode=args.mode,
                hemi=args.hemi,
                atlas=args.atlas,
                fmt=args.fmt,
            )
    elif args.mode == 'compare':
        plot_compare_modes(
            roi_name=args.roi,
            views=args.views,
            save=args.save,
            out_dir=args.out_dir,
            hemi=args.hemi,
            atlas=args.atlas,
            fmt=args.fmt,
        )
    elif args.roi:
        brain = plot_single_roi(
            roi_name=args.roi,
            views=args.views,
            save=args.save,
            out_dir=args.out_dir,
            mode=args.mode,
            hemi=args.hemi,
            atlas=args.atlas,
            fmt=args.fmt,
        )
    else:
        brain = plot_roi_brain(
            views=args.views,
            save=args.save,
            out_dir=args.out_dir,
            mode=args.mode,
            hemi=args.hemi,
            atlas=args.atlas,
            fmt=args.fmt,
        )

    # Keep the interactive viewer alive by running the Qt event loop
    if not args.save and args.mode != 'compare':
        from qtpy.QtWidgets import QApplication
        app = QApplication.instance()
        if app is not None:
            app.exec_()
