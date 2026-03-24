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

Rendering uses ``mne.viz.Brain`` (PyVista/VTK) for publication-quality
3D brain surface renders with proper lighting, shading, and transparency.

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

    # Surface selection
    python visualize_rois.py --speech-rois --save --surf pial       # folded (default)
    python visualize_rois.py --speech-rois --save --surf inflated   # inflated

    # SVG output for publication figures
    python visualize_rois.py --speech-rois --atlas HCPMMP1 --save --format svg

    # Stat-map overlay (continuous data)
    python visualize_rois.py --statmap data.csv --atlas HCPMMP1 --save --cmap hot

    # Full atlas / compare modes
    python visualize_rois.py --atlas Schaefer200 --mode full        # full-res parcels
    python visualize_rois.py --mode compare --format svg --save     # full vs ico-5
"""
import argparse
import os
import tempfile
from pathlib import Path

# Enable offscreen rendering for PyVista (must be set before import)
os.environ.setdefault('PYVISTA_OFF_SCREEN', 'true')

import numpy as np
import mne
from mne.datasets import fetch_fsaverage

from config import (SPEECH_ROIS, SPEECH_ROI_NAMES, ATLAS_PARC_MAP,
                     FIGURES_ROOT, CUSTOM_ROI_DIR)
from forward_model import build_roi_labels, load_custom_volumetric_rois

# Colorblind-friendly palette: Wong (2011) + Tol muted, 20 distinct colours.
# Covers up to 20 ROIs; for larger atlases, cycles with offset luminance.
_CB_PALETTE = [
    '#0077BB',  # blue
    '#EE7733',  # orange
    '#009988',  # teal
    '#CC3311',  # red
    '#33BBEE',  # cyan
    '#EE3377',  # magenta
    # '#BBBBBB',  # grey
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


# ─────────────────────────────────────────────────────────────────────
# Shared rendering helpers (PyVista direct — MATLAB trisurf style)
# ─────────────────────────────────────────────────────────────────────

# Camera direction vectors per view/hemi: (dx, dy, dz) from focal point.
# Matches MATLAB convention: lh lateral = view from left, etc.
_CAMERA_DIRS = {
    'lateral':  {'lh': (-1, 0, 0),  'rh': (1, 0, 0)},
    'medial':   {'lh': (1, 0, 0),   'rh': (-1, 0, 0)},
    'dorsal':   {'lh': (0, 0, 1),   'rh': (0, 0, 1)},
    'ventral':  {'lh': (0, 0, -1),  'rh': (0, 0, -1)},
    'anterior': {'lh': (0, 1, 0),   'rh': (0, 1, 0)},
    'posterior': {'lh': (0, -1, 0),  'rh': (0, -1, 0)},
    'frontal':  {'lh': (0, 1, 0),   'rh': (0, 1, 0)},
}


def _load_surface(subjects_dir, hemi, surf='pial', smooth=0.25):
    """Load and optionally smooth a FreeSurfer surface.

    Returns (coords, faces) as numpy arrays.
    """
    surf_dir = Path(subjects_dir) / 'fsaverage' / 'surf'
    coords, faces = mne.read_surface(str(surf_dir / f'{hemi}.{surf}'))

    if smooth and smooth > 0:
        coords_infl, _ = mne.read_surface(str(surf_dir / f'{hemi}.inflated'))
        coords = (coords * (1 - smooth) + coords_infl * smooth).astype(np.float32)

    return coords, faces


def _make_pv_mesh(coords, faces):
    """Build a PyVista PolyData mesh from FreeSurfer coords and faces."""
    import pyvista as pv
    faces_pv = np.column_stack([np.full(len(faces), 3), faces]).ravel()
    return pv.PolyData(coords, faces_pv)


def _set_camera(plotter, view, hemi, coords):
    """Set camera position for a named view and hemisphere."""
    d = _CAMERA_DIRS.get(view, _CAMERA_DIRS['lateral']).get(hemi, (-1, 0, 0))
    center = coords.mean(axis=0)
    cam_dist = 300
    cam_pos = [center[0] + d[0] * cam_dist,
               center[1] + d[1] * cam_dist,
               center[2] + d[2] * cam_dist]
    # Up vector: Z-up for lateral/medial/anterior/posterior; Y-up for dorsal/ventral
    if view in ('dorsal', 'ventral'):
        up = (0, 1, 0)
    else:
        up = (0, 0, 1)
    plotter.camera_position = [cam_pos, center.tolist(), up]
    plotter.reset_camera()
    plotter.camera.zoom(1.5)


def _render_brain_views(coords, faces, views, hemi, size=(1200, 900),
                        vertex_rgba=None, vertex_scalars=None,
                        cmap='hot', clim=None):
    """
    Render brain surface screenshots using PyVista.

    Uses smooth_shading + interpolate_before_map + backface_culling for
    publication-quality Gouraud-lit renders matching MATLAB trisurf style.

    Supports two colouring modes:
      - *vertex_rgba*: per-vertex RGBA array (uint8) for discrete ROI colours.
      - *vertex_scalars*: per-vertex float array for continuous heatmap overlay
        (NaN = transparent, shows gray base underneath).

    Returns {view_name: ndarray}.
    """
    import pyvista as pv
    from matplotlib.image import imread

    mesh = _make_pv_mesh(coords, faces)
    # Smooth Gouraud shading with per-vertex interpolation and back-face
    # culling — matches the rendering approach from the reference snippet.
    _material = dict(smooth_shading=True, interpolate_before_map=True,
                     backface_culling=True, style='surface', opacity=1.0)

    images = {}
    for view in views:
        pl = pv.Plotter(off_screen=True, window_size=size)
        pl.set_background('white')

        if vertex_rgba is not None:
            mesh['colors'] = vertex_rgba
            pl.add_mesh(mesh, scalars='colors', rgb=True, **_material)
        elif vertex_scalars is not None:
            # Base gray layer
            pl.add_mesh(mesh.copy(), color=[0.7, 0.7, 0.7], **_material)
            # Overlay coloured data (NaN → transparent)
            mesh['data'] = vertex_scalars
            pl.add_mesh(mesh, scalars='data', cmap=cmap, clim=clim,
                        nan_opacity=0, show_scalar_bar=False, **_material)
        else:
            pl.add_mesh(mesh, color=[0.7, 0.7, 0.7], **_material)

        _set_camera(pl, view, hemi, coords)

        tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        tmp_path = tmp.name
        tmp.close()
        try:
            pl.screenshot(tmp_path)
            images[view] = imread(tmp_path)
        finally:
            os.unlink(tmp_path)
        pl.close()

    return images


def _compose_brain_figure(hemi_view_images, views, legend_items=None,
                          suptitle=None, save=False, out_dir=None,
                          name_prefix='brain', fmt='png',
                          colorbar_mappable=None, colorbar_label=None):
    """
    Compose rendered brain images into a matplotlib figure.

    Parameters
    ----------
    hemi_view_images : dict
        {hemi: {view: ndarray}} -- rendered images per hemisphere per view.
    views : list of str
        View names in column order.
    legend_items : list of (name, colour) | None
        Discrete ROI legend entries.
    colorbar_mappable : ScalarMappable | None
        If provided, adds a continuous colorbar instead of discrete legend.
    colorbar_label : str | None
        Label for the colorbar axis.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    hemis = list(hemi_view_images.keys())
    n_rows = len(hemis)
    n_cols = len(views)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(7 * n_cols, 6 * n_rows),
                             squeeze=False)

    for r, h in enumerate(hemis):
        for c, view in enumerate(views):
            ax = axes[r, c]
            if view in hemi_view_images[h]:
                ax.imshow(hemi_view_images[h][view])
            ax.axis('off')

    if legend_items:
        handles = [mpatches.Patch(facecolor=c, edgecolor='grey', label=n)
                   for n, c in legend_items]
        fig.legend(handles=handles, loc='center left',
                   bbox_to_anchor=(1.0, 0.5), fontsize=10, frameon=True)

    if colorbar_mappable is not None:
        cbar = fig.colorbar(colorbar_mappable, ax=axes.ravel().tolist(),
                            shrink=0.6, pad=0.02)
        if colorbar_label:
            cbar.set_label(colorbar_label, fontsize=12)

    if suptitle:
        fig.suptitle(suptitle, fontsize=16, y=0.98)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        fig.tight_layout(rect=[0, 0, 0.95 if legend_items else 1.0, 0.95 if suptitle else 1.0])

    if save:
        if out_dir is None:
            out_dir = FIGURES_ROOT / 'ROI_maps'
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        ext = fmt if fmt in ('png', 'svg') else 'png'
        fpath = out_dir / f'{name_prefix}.{ext}'
        fig.savefig(str(fpath), format=ext, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {fpath}')
    else:
        plt.show(block=True)

    return fig


# ─────────────────────────────────────────────────────────────────────
# Main plotting functions
# ─────────────────────────────────────────────────────────────────────

def plot_roi_brain(roi_dict=None, subjects_dir=None, views=None,
                   hemi=None, save=False, out_dir=None,
                   mode='ico5', atlas='aparc', fmt='png',
                   surf='pial', smooth=0.25, brain_kwargs=None,
                   custom_roi_dir=None):
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
        Directory for saved figures.  Defaults to FIGURES_ROOT / 'ROI_maps'.
    mode : str
        'ico5' to show only source-space vertices (default), or 'full'
        to show full-resolution atlas parcel vertices.
    atlas : str
        Atlas parcellation: 'aparc', 'Schaefer200', 'HCPMMP1', or 'custom'.
    surf : str
        Cortical surface: 'pial' (default), 'inflated', or 'white'.
    brain_kwargs : dict | None
        Extra keyword arguments forwarded to ``mne.viz.Brain``.
    custom_roi_dir : Path | str | None
        Override directory for custom NIfTI ROIs (atlas='custom' only).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The composed figure.
    """
    # ── resolve subjects_dir ──────────────────────────────────────────
    if subjects_dir is None:
        subjects_dir = str(fetch_fsaverage(verbose=False).parent)

    # ── build ROI labels ──────────────────────────────────────────────
    if roi_dict is None:
        if atlas == 'aparc':
            roi_dict = build_roi_labels(subjects_dir, atlas='aparc',
                                         composite_rois=SPEECH_ROIS['aparc'])
        else:
            roi_dict = build_roi_labels(subjects_dir, atlas=atlas,
                                         custom_roi_dir=custom_roi_dir)

    roi_dict_full = roi_dict

    # ── restrict to source-space vertices if requested ────────────────
    if mode == 'ico5':
        fs_path = Path(subjects_dir) / 'fsaverage'
        src = mne.read_source_spaces(
            fs_path / 'bem' / 'fsaverage-ico-5-src.fif', verbose=False
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

    hemis_to_plot = ['lh', 'rh'] if hemi == 'both' else [hemi]
    colours_hex = _get_roi_colours(list(roi_dict.keys()))

    # ── render each hemisphere ────────────────────────────────────────
    from matplotlib.colors import to_rgba

    hemi_view_images = {}
    for h in hemis_to_plot:
        coords, faces = _load_surface(subjects_dir, h, surf=surf, smooth=smooth)

        # Build per-vertex RGBA: base gray + ROI colours
        rgba = np.full((len(coords), 4), [0.7, 0.7, 0.7, 1.0], dtype=np.float32)
        for idx, (name, label) in enumerate(roi_dict.items()):
            if hasattr(label, 'hemi') and label.hemi == h:
                c = np.array(to_rgba(colours_hex[idx]))
                rgba[label.vertices] = c

        vertex_rgba = (rgba * 255).astype(np.uint8)
        hemi_view_images[h] = _render_brain_views(
            coords, faces, views, h, vertex_rgba=vertex_rgba)

    # ── print vertex summary ──────────────────────────────────────────
    print(f'\n── ROI vertex coverage (atlas={atlas}, mode={mode}, '
          f'{len(roi_dict)} ROIs) ──')
    shown_items = list(roi_dict.items())
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

    # ── build legend ──────────────────────────────────────────────────
    legend_items = []
    seen = set()
    for idx, (name, label) in enumerate(roi_dict.items()):
        if label.hemi in [h for h in hemis_to_plot] and name not in seen:
            legend_items.append((name, colours_hex[idx]))
            seen.add(name)

    return _compose_brain_figure(
        hemi_view_images, views,
        legend_items=legend_items,
        save=save, out_dir=out_dir,
        name_prefix=f'fsaverage_{atlas}_ROIs',
        fmt=fmt,
    )


def plot_single_roi(roi_name, parcel_names=None, roi_dict=None, subjects_dir=None,
                    views=None, hemi=None, save=False, out_dir=None,
                    mode='ico5', atlas='aparc', fmt='png', surf='pial',
                    smooth=0.25):
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
    surf : str
        Cortical surface: 'pial' (default), 'inflated', or 'white'.
    """
    if subjects_dir is None:
        fs_dir = fetch_fsaverage(verbose=False)
        subjects_dir = str(fs_dir.parent)
    else:
        fs_dir = Path(subjects_dir) / 'fsaverage'

    src = None
    if mode == 'ico5':
        src = mne.read_source_spaces(
            Path(fs_dir) / 'bem' / 'fsaverage-ico-5-src.fif', verbose=False
        )

    if parcel_names is not None:
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
        parcel_names = SPEECH_ROIS['aparc'][roi_name]
        labels_all = mne.read_labels_from_annot(
            'fsaverage', parc='aparc', hemi='both',
            subjects_dir=subjects_dir
        )
        label_lookup = {l.name: l for l in labels_all}
        labels_to_plot = {pname: label_lookup[pname] for pname in parcel_names}

    else:
        if atlas == 'aparc':
            all_roi_dict = build_roi_labels(subjects_dir, atlas='aparc',
                                             composite_rois=SPEECH_ROIS['aparc'])
        else:
            all_roi_dict = build_roi_labels(subjects_dir, atlas=atlas)

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

    if hemi is None:
        hemis = {label.hemi for label in labels_to_plot.values()}
        hemi = hemis.pop() if len(hemis) == 1 else 'both'

    if views is None:
        views = ['lateral', 'medial']

    hemis_to_plot = ['lh', 'rh'] if hemi == 'both' else [hemi]
    colours_hex = _get_roi_colours(list(labels_to_plot.keys()))

    print(f'\n── {roi_name}: {len(labels_to_plot)} parcels '
          f'(atlas={atlas}, mode={mode}) ──')

    # ── render each hemisphere ────────────────────────────────────────
    from matplotlib.colors import to_rgba

    hemi_view_images = {}
    legend_items = []
    printed_labels = set()

    for h in hemis_to_plot:
        coords, faces = _load_surface(subjects_dir, h, surf=surf, smooth=smooth)
        rgba = np.full((len(coords), 4), [0.7, 0.7, 0.7, 1.0], dtype=np.float32)

        for idx, (pname, label_full) in enumerate(labels_to_plot.items()):
            label = label_full.restrict(src) if src is not None else label_full

            if hasattr(label, 'hemi') and label.hemi == h:
                colour = colours_hex[idx]
                c = np.array(to_rgba(colour))
                rgba[label.vertices] = c

                if pname not in printed_labels:
                    print(f'  {pname:45s}  colour={colour}  '
                          f'full={len(label_full.vertices):>5d}  '
                          f'shown={len(label.vertices):>5d}')
                    printed_labels.add(pname)
                    legend_items.append((pname, colour))

        vertex_rgba = (rgba * 255).astype(np.uint8)
        hemi_view_images[h] = _render_brain_views(
            coords, faces, views, h, vertex_rgba=vertex_rgba)

    print()

    safe_name = roi_name.replace(' ', '_')
    return _compose_brain_figure(
        hemi_view_images, views,
        legend_items=legend_items,
        save=save, out_dir=out_dir,
        name_prefix=f'fsaverage_{atlas}_{safe_name}',
        fmt=fmt,
    )


def plot_compare_modes(roi_name=None, parcel_names=None, roi_dict=None, subjects_dir=None,
                       views=None, hemi=None, save=False, out_dir=None,
                       atlas='aparc', fmt='png', surf='pial', smooth=0.25):
    """
    Plot full-resolution (top row) vs ico-5 (bottom row) for comparison.

    If *roi_name* or *parcel_names* is given, shows that single ROI's
    parcels.  Otherwise shows all ROIs from the specified atlas.
    """
    import matplotlib.pyplot as plt
    from matplotlib.image import imread

    if subjects_dir is None:
        fs_dir = fetch_fsaverage(verbose=False)
        subjects_dir = str(fs_dir.parent)
    else:
        fs_dir = Path(subjects_dir) / 'fsaverage'

    if views is None:
        views = ['lateral']

    src = mne.read_source_spaces(
        Path(fs_dir) / 'bem' / 'fsaverage-ico-5-src.fif', verbose=False
    )

    # --- Build labels for both modes ---
    if parcel_names is not None:
        parc = ATLAS_PARC_MAP.get(atlas, atlas)
        labels_all = mne.read_labels_from_annot(
            'fsaverage', parc=parc, hemi='both',
            subjects_dir=subjects_dir
        )
        label_lookup = {l.name: l for l in labels_all}
        full_labels = {pname: label_lookup[pname] for pname in parcel_names}
        title_tag = roi_name if roi_name else 'CustomROI'

    elif roi_name is not None:
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
        if roi_dict is None:
            if atlas == 'aparc':
                roi_dict = build_roi_labels(subjects_dir, atlas='aparc',
                                             composite_rois=SPEECH_ROIS['aparc'])
            else:
                roi_dict = build_roi_labels(subjects_dir, atlas=atlas)
        full_labels = roi_dict
        title_tag = f'{atlas}_all_ROIs'

    ico5_labels = _restrict_labels_to_src(full_labels, src)

    if hemi is None:
        hemis_present = {l.hemi for l in full_labels.values()}
        if hemis_present == {'lh'}:
            hemi = 'lh'
        elif hemis_present == {'rh'}:
            hemi = 'rh'
        else:
            hemi = 'both'

    hemis_to_plot = ['lh', 'rh'] if hemi == 'both' else [hemi]
    colours = _get_roi_colours(list(full_labels.keys()))

    def _render_labels(labels_dict, label_tag):
        """Render a set of labels across hemispheres. Returns {hemi: {view: img}}."""
        from matplotlib.colors import to_rgba
        result = {}
        for h in hemis_to_plot:
            coords, faces = _load_surface(subjects_dir, h, surf=surf, smooth=smooth)
            rgba = np.full((len(coords), 4), [0.7, 0.7, 0.7, 1.0], dtype=np.float32)
            for idx, (name, label) in enumerate(labels_dict.items()):
                if hasattr(label, 'hemi') and label.hemi == h:
                    c = np.array(to_rgba(colours[idx]))
                    rgba[label.vertices] = c
            vertex_rgba = (rgba * 255).astype(np.uint8)
            result[h] = _render_brain_views(
                coords, faces, views, h, vertex_rgba=vertex_rgba)
        return result

    print(f'Rendering {atlas} full-resolution parcels...')
    full_images = _render_labels(full_labels, 'full')
    print(f'Rendering {atlas} ico-5 source-space vertices...')
    ico5_images = _render_labels(ico5_labels, 'ico5')

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

    # --- Compose: full on top, ico-5 on bottom ---
    # Flatten into a single grid: for each hemi, top row = full, bottom = ico5
    all_rows = []
    row_labels = []
    for h in hemis_to_plot:
        all_rows.append(full_images[h])
        row_labels.append(f'{h} full')
        all_rows.append(ico5_images[h])
        row_labels.append(f'{h} ico-5')

    n_rows = len(all_rows)
    n_cols = len(views)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(8 * n_cols, 6 * n_rows),
                             squeeze=False)

    for r, (images, rlabel) in enumerate(zip(all_rows, row_labels)):
        for c, view in enumerate(views):
            axes[r, c].imshow(images[view])
            axes[r, c].set_title(f'{rlabel} — {view}', fontsize=14)
            axes[r, c].axis('off')

    fig.suptitle(f'{title_tag}: full vs ico-5 source space',
                 fontsize=18, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save:
        if out_dir is None:
            out_dir = FIGURES_ROOT / 'ROI_maps'
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        view_tag = '_'.join(views)
        safe_tag = title_tag.replace(' ', '_')
        ext = fmt if fmt in ('png', 'svg') else 'png'
        fpath = out_dir / f'fsaverage_{safe_tag}_full_vs_ico5_{view_tag}.{ext}'
        fig.savefig(str(fpath), dpi=200, bbox_inches='tight', format=ext)
        print(f'Saved: {fpath}')
        plt.close(fig)
    else:
        plt.show()

    return fig


def plot_mesh_only(subjects_dir=None, views=None, hemi=None, save=False,
                   out_dir=None, fmt='png', surf='pial', smooth=0.25):
    """Plot the bare fsaverage cortical surface without any ROI overlays."""
    if subjects_dir is None:
        subjects_dir = str(fetch_fsaverage(verbose=False).parent)

    if hemi is None:
        hemi = 'both'
    if views is None:
        views = ['lateral', 'medial']

    hemis_to_plot = ['lh', 'rh'] if hemi == 'both' else [hemi]

    hemi_view_images = {}
    for h in hemis_to_plot:
        coords, faces = _load_surface(subjects_dir, h, surf=surf, smooth=smooth)
        hemi_view_images[h] = _render_brain_views(coords, faces, views, h)

    return _compose_brain_figure(
        hemi_view_images, views,
        save=save, out_dir=out_dir,
        name_prefix=f'fsaverage_{surf}_mesh',
        fmt=fmt,
    )


def plot_statmap_brain(data_dict, subjects_dir=None, views=None,
                       hemi=None, surf='pial', save=False, out_dir=None,
                       atlas='aparc', fmt='png', cmap='hot',
                       fmin=None, fmid=None, fmax=None,
                       mode='ico5', smooth=0.25, brain_kwargs=None):
    """
    Overlay continuous stat-map data (e.g. SVM accuracy) on the cortical surface.

    Parameters
    ----------
    data_dict : dict
        {roi_name: float} mapping ROI names to scalar values.
    cmap : str
        Colormap for the continuous overlay (default: 'hot').
    fmin, fmid, fmax : float | None
        Thresholds for the colormap. If None, derived from data range.
    mode : str
        'ico5' or 'full' -- vertex restriction mode.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize

    if subjects_dir is None:
        subjects_dir = str(fetch_fsaverage(verbose=False).parent)

    if views is None:
        views = ['lateral', 'medial']

    # Build speech ROI labels
    speech_labels = build_speech_roi_labels(atlas, subjects_dir)

    if mode == 'ico5':
        src = mne.read_source_spaces(
            Path(subjects_dir) / 'fsaverage' / 'bem' / 'fsaverage-ico-5-src.fif',
            verbose=False
        )
        speech_labels = _restrict_labels_to_src(speech_labels, src)

    if hemi is None:
        hemis_present = {l.hemi for l in speech_labels.values()}
        if hemis_present == {'lh'}:
            hemi = 'lh'
        elif hemis_present == {'rh'}:
            hemi = 'rh'
        else:
            hemi = 'both'

    hemis_to_plot = ['lh', 'rh'] if hemi == 'both' else [hemi]

    # Compute colormap range from data
    values = [v for k, v in data_dict.items() if k in speech_labels]
    if not values:
        raise ValueError('No matching ROI names between data_dict and speech labels.')

    if fmin is None:
        fmin = min(values)
    if fmax is None:
        fmax = max(values)
    if fmid is None:
        fmid = (fmin + fmax) / 2.0

    hemi_view_images = {}
    for h in hemis_to_plot:
        coords, faces = _load_surface(subjects_dir, h, surf=surf, smooth=smooth)

        # Build per-vertex scalar data (NaN for non-ROI → transparent)
        scalars = np.full(len(coords), np.nan, dtype=np.float32)
        for roi_name, label in speech_labels.items():
            if label.hemi == h and roi_name in data_dict:
                scalars[label.vertices] = data_dict[roi_name]

        hemi_view_images[h] = _render_brain_views(
            coords, faces, views, h,
            vertex_scalars=scalars, cmap=cmap, clim=[fmin, fmax])

    # Build colorbar mappable
    norm = Normalize(vmin=fmin, vmax=fmax)
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])

    return _compose_brain_figure(
        hemi_view_images, views,
        colorbar_mappable=mappable,
        colorbar_label='Value',
        save=save, out_dir=out_dir,
        name_prefix=f'fsaverage_{atlas}_statmap',
        fmt=fmt,
    )


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
        choices=['aparc', 'Schaefer200', 'HCPMMP1', 'custom'],
        help='Cortical atlas for ROI parcellation (default: aparc). '
             '"custom" uses volumetric NIfTI ROIs from --custom-rois-dir.'
    )
    parser.add_argument(
        '--surf', choices=['pial', 'inflated', 'white'],
        default='pial',
        help='Base cortical surface (default: pial). Combined with --smooth '
             'to control sulcal depth.'
    )
    parser.add_argument(
        '--smooth', type=float, default=0.25,
        help='Blend ratio between base surface and inflated (0.0 = pure base, '
             '1.0 = fully inflated). Default 0.25 gives shallow sulci like '
             'publication figures. Use 0 for the raw surface.'
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
        '--mesh-only', action='store_true',
        help='Plot the bare fsaverage cortical surface without any ROIs.'
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
    parser.add_argument(
        '--statmap', type=str, default=None,
        help='Path to a CSV file with ROI-level stat values for heatmap '
             'overlay. Format: roi_name,value (one per line, no header).'
    )
    parser.add_argument(
        '--cmap', type=str, default='hot',
        help='Colormap for stat-map overlay (default: hot). '
             'Any matplotlib colormap name.'
    )
    parser.add_argument(
        '--custom-rois-dir', type=str, default=None,
        help='Override directory for custom volumetric NIfTI ROI masks '
             '(used with --atlas custom). Defaults to config.CUSTOM_ROI_DIR.'
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

    # --custom-rois-dir implies --atlas custom
    if args.custom_rois_dir is not None:
        args.atlas = 'custom'

    # --statmap: continuous heatmap overlay
    if args.statmap:
        import pandas as pd
        df = pd.read_csv(args.statmap, header=None, names=['roi', 'value'])
        data_dict = dict(zip(df.roi, df.value))
        plot_statmap_brain(
            data_dict=data_dict,
            views=args.views, hemi=args.hemi, surf=args.surf, smooth=args.smooth,
            save=args.save, out_dir=args.out_dir,
            atlas=args.atlas, fmt=args.fmt, cmap=args.cmap,
            mode=args.mode,
        )
        raise SystemExit(0)

    if args.mesh_only:
        fig = plot_mesh_only(
            views=args.views,
            hemi=args.hemi,
            save=args.save,
            out_dir=args.out_dir,
            fmt=args.fmt,
            surf=args.surf, smooth=args.smooth,
        )
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
                surf=args.surf, smooth=args.smooth,
            )
        else:
            fig = plot_single_roi(
                roi_name=roi_name,
                parcel_names=parcel_names,
                views=args.views,
                save=args.save,
                out_dir=args.out_dir,
                mode=args.mode,
                hemi=args.hemi,
                atlas=args.atlas,
                fmt=args.fmt,
                surf=args.surf, smooth=args.smooth,
            )
    elif args.speech_rois:
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
                surf=args.surf, smooth=args.smooth,
            )
        else:
            fig = plot_roi_brain(
                roi_dict=speech_dict,
                views=args.views,
                save=args.save,
                out_dir=args.out_dir,
                mode=args.mode,
                hemi=args.hemi,
                atlas=args.atlas,
                fmt=args.fmt,
                surf=args.surf, smooth=args.smooth,
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
            surf=args.surf, smooth=args.smooth,
        )
    elif args.roi:
        fig = plot_single_roi(
            roi_name=args.roi,
            views=args.views,
            save=args.save,
            out_dir=args.out_dir,
            mode=args.mode,
            hemi=args.hemi,
            atlas=args.atlas,
            fmt=args.fmt,
            surf=args.surf, smooth=args.smooth,
        )
    else:
        fig = plot_roi_brain(
            views=args.views,
            save=args.save,
            out_dir=args.out_dir,
            mode=args.mode,
            hemi=args.hemi,
            atlas=args.atlas,
            fmt=args.fmt,
            surf=args.surf, smooth=args.smooth,
            custom_roi_dir=args.custom_rois_dir,
        )
