"""
Build the forward model using the fsaverage template anatomy.

No individual MRIs are available, so we use the pre-computed fsaverage
BEM solution and source space shipped with MNE-Python. This module
handles fetching fsaverage, creating the forward solution, and building
cortical ROI labels from supported atlases (including custom volumetric
NIfTI ROI masks projected onto the fsaverage surface).
"""
from pathlib import Path

import numpy as np
import mne
from mne.datasets import fetch_fsaverage

from config import SPEECH_ROIS, ATLAS_PARC_MAP, CUSTOM_ROI_DIR


def setup_fsaverage():
    """
    Fetch fsaverage files and return paths.

    Returns
    -------
    subjects_dir : Path
        The subjects_dir containing 'fsaverage'.
    fs_dir : Path
        Path to the fsaverage directory itself.
    src : mne.SourceSpaces
        The fsaverage ico-5 source space.
    bem : dict
        The 3-layer BEM solution.
    """
    fs_dir = fetch_fsaverage(verbose=True)
    subjects_dir = fs_dir.parent

    src = mne.read_source_spaces(
        fs_dir / 'bem' / 'fsaverage-ico-5-src.fif'
    )
    bem = mne.read_bem_solution(
        fs_dir / 'bem' / 'fsaverage-5120-5120-5120-bem-sol.fif'
    )

    return subjects_dir, fs_dir, src, bem


def make_forward(epochs_info, src, bem):
    """
    Compute the forward solution for the given EEG info.

    Parameters
    ----------
    epochs_info : mne.Info
        Info object from the EEG epochs.
    src : mne.SourceSpaces
        Source space (fsaverage ico-5).
    bem : dict
        BEM solution.

    Returns
    -------
    fwd : mne.Forward
        The forward solution.
    """
    fwd = mne.make_forward_solution(
        epochs_info,
        trans='fsaverage',
        src=src,
        bem=bem,
        eeg=True,
        meg=False,
        mindist=5.0,
        n_jobs=None,
        verbose=True,
    )
    print(f'Forward solution: {fwd["nsource"]} sources')
    return fwd


def load_custom_volumetric_rois(roi_dir=None, threshold=0.25,
                                subjects_dir=None, surf='pial'):
    """
    Project volumetric NIfTI ROI masks onto the fsaverage surface.

    Loads binary NIfTI masks from *roi_dir*, projects each onto the
    fsaverage surface using nilearn's ``vol_to_surf``, thresholds the
    projected values, and returns MNE Label objects.

    Parameters
    ----------
    roi_dir : Path | str | None
        Directory containing ``*_anatrestrict.nii`` masks.  Defaults to
        ``CUSTOM_ROI_DIR`` from config.
    threshold : float
        Fraction of the max projected value above which a vertex is
        included in the label (default 0.25).  Lower = larger labels.
    subjects_dir : str | None
        MNE subjects_dir containing 'fsaverage'.
    surf : str
        Surface mesh for projection ('pial' or 'white').

    Returns
    -------
    roi_labels : dict
        ``{roi_name: mne.Label}`` for each ROI, on the fsaverage surface.
    """
    from nilearn.surface import vol_to_surf

    if roi_dir is None:
        roi_dir = CUSTOM_ROI_DIR
    roi_dir = Path(roi_dir)

    if subjects_dir is None:
        subjects_dir = str(fetch_fsaverage(verbose=False).parent)

    fs_path = Path(subjects_dir) / 'fsaverage'

    # Find all non-edge NIfTI masks
    nii_files = sorted(
        f for f in roi_dir.glob('*.nii')
        if not f.name.startswith('edge_')
    )
    if not nii_files:
        raise FileNotFoundError(f'No .nii ROI masks found in {roi_dir}')

    roi_labels = {}
    for nii_path in nii_files:
        # Derive ROI name: e.g. "awfa-audioLoc_15mm_anatrestrict.nii" → "awfa"
        roi_name = nii_path.stem.split('-')[0]

        # Project onto both hemispheres
        for hemi in ['lh', 'rh']:
            surf_mesh = str(fs_path / 'surf' / f'{hemi}.{surf}')
            projected = vol_to_surf(
                str(nii_path), surf_mesh,
                interpolation='nearest_most_frequent', radius=3.0,
            )
            projected = np.nan_to_num(projected, nan=0.0)

            max_val = projected.max()
            if max_val <= 0:
                continue
            mask = projected >= (threshold * max_val)
            verts = np.where(mask)[0]
            if len(verts) == 0:
                continue

            label = mne.Label(
                vertices=verts,
                hemi=hemi,
                name=f'{roi_name}-{hemi}',
                subject='fsaverage',
            )
            roi_labels[f'{roi_name}-{hemi}'] = label

    # ── Mask awfa: remove vertices falling in vSMC (precentral/postcentral) ─
    _vsmc_aparc = ['precentral', 'postcentral']
    for key in list(roi_labels):
        if not key.startswith('awfa'):
            continue
        label = roi_labels[key]
        hemi = label.hemi
        vsmc_labels = mne.read_labels_from_annot(
            'fsaverage', parc='aparc', hemi=hemi,
            subjects_dir=subjects_dir, verbose=False,
        )
        vsmc_verts = set()
        for al in vsmc_labels:
            if any(al.name.startswith(n) for n in _vsmc_aparc):
                vsmc_verts.update(al.vertices)
        clean_verts = np.array(
            sorted(v for v in label.vertices if v not in vsmc_verts)
        )
        n_removed = len(label.vertices) - len(clean_verts)
        if n_removed > 0:
            print(f'  {key}: masked {n_removed} vertices overlapping '
                  f'aparc vSMC (precentral/postcentral)')
            roi_labels[key] = mne.Label(
                vertices=clean_verts, hemi=hemi,
                name=label.name, subject='fsaverage',
            )

    print(f'Projected {len(nii_files)} NIfTI masks → '
          f'{len(roi_labels)} surface labels (threshold={threshold})')
    for name, label in roi_labels.items():
        print(f'  {name:30s}  hemi={label.hemi}  '
              f'vertices={len(label.vertices):>5d}')
    return roi_labels


def build_roi_labels(subjects_dir, atlas='aparc', composite_rois=None,
                     custom_roi_dir=None):
    """
    Build cortical ROI labels from a supported atlas.

    Parameters
    ----------
    subjects_dir : Path
        The subjects_dir containing 'fsaverage'.
    atlas : str
        Atlas name: 'aparc', 'HCPMMP1', 'Schaefer200', or 'custom'.
    composite_rois : dict or None
        When provided, merges atlas parcels into composite ROIs.
        Maps ROI name to list of native parcel names.  Works with
        any non-custom atlas (aparc, HCPMMP1, Schaefer200).
    custom_roi_dir : Path | str | None
        Directory with NIfTI masks for atlas='custom'.  Defaults to
        ``CUSTOM_ROI_DIR`` from config.

    Returns
    -------
    roi_dict : dict
        Mapping from ROI name to MNE Label object.
    """
    # ── Custom volumetric ROIs ────────────────────────────────────────
    if atlas == 'custom':
        return load_custom_volumetric_rois(
            roi_dir=custom_roi_dir, subjects_dir=subjects_dir,
        )

    parc = ATLAS_PARC_MAP[atlas]

    labels_all = mne.read_labels_from_annot(
        'fsaverage', parc=parc, hemi='both',
        subjects_dir=subjects_dir
    )

    # Composite-ROI mode: merge parcels into speech-network ROIs
    if composite_rois is not None:
        available = {l.name: l for l in labels_all}

        roi_dict = {}
        for roi_name, label_names in composite_rois.items():
            found = [available[n] for n in label_names if n in available]
            missing = [n for n in label_names if n not in available]
            if missing:
                print(f'  Warning: {roi_name}: labels not found in '
                      f'{parc} atlas (skipped): {missing}')
            if not found:
                print(f'  Warning: {roi_name}: no valid labels — '
                      f'ROI omitted entirely')
                continue
            combined = found[0]
            for lbl in found[1:]:
                combined = combined + lbl
            combined.name = roi_name
            roi_dict[roi_name] = combined

        print(f'Built {len(roi_dict)} composite ROIs ({atlas}): '
              f'{list(roi_dict.keys())}')
        return roi_dict

    # Standard mode: return all parcels from the atlas (excluding ???)
    roi_dict = {}
    for label in labels_all:
        if label.name.startswith('???'):
            continue
        roi_dict[label.name] = label

    print(f'Loaded {len(roi_dict)} ROIs from {atlas} atlas')
    return roi_dict
