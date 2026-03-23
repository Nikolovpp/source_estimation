"""
Build the forward model using the fsaverage template anatomy.

No individual MRIs are available, so we use the pre-computed fsaverage
BEM solution and source space shipped with MNE-Python. This module
handles fetching fsaverage, creating the forward solution, and building
cortical ROI labels from supported atlases.
"""
import mne
from mne.datasets import fetch_fsaverage

from config import SOURCE_ROIS, ATLAS_PARC_MAP


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


def build_roi_labels(subjects_dir, atlas='aparc', composite_rois=None):
    """
    Build cortical ROI labels from a supported atlas.

    Parameters
    ----------
    subjects_dir : Path
        The subjects_dir containing 'fsaverage'.
    atlas : str
        Atlas name: 'aparc', 'HCPMMP1', or 'Schaefer200'.
    composite_rois : dict or None
        When provided with atlas='aparc', merges aparc labels into
        composite ROIs (backward-compatible mode).  Maps ROI name to
        list of aparc label names.

    Returns
    -------
    roi_dict : dict
        Mapping from ROI name to MNE Label object.
    """
    parc = ATLAS_PARC_MAP[atlas]

    labels_all = mne.read_labels_from_annot(
        'fsaverage', parc=parc, hemi='both',
        subjects_dir=subjects_dir
    )

    # Legacy composite-ROI mode (backward compat with existing 8-ROI setup)
    if composite_rois is not None and atlas == 'aparc':
        def _get_label(name):
            matches = [l for l in labels_all if l.name == name]
            if not matches:
                raise ValueError(f'Label "{name}" not found in {parc} atlas')
            return matches[0]

        roi_dict = {}
        for roi_name, label_names in composite_rois.items():
            combined = _get_label(label_names[0])
            for ln in label_names[1:]:
                combined = combined + _get_label(ln)
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
