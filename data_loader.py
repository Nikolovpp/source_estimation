"""
Load EEGLAB-preprocessed .mat files and convert to MNE Epochs objects.

Handles both perception (popthresh120) and production (ProdOnset) data.
Extracts behavioral data (word lists), trial info (artifact rejection),
and builds proper MNE Epochs with event codes for SVM classification.
"""
import numpy as np
import mat73
import mne

from config import (
    PROD_DIFF_TH, PROD_DIFF_F, PERC_DIFF_S, PERC_DIFF_T,
    COMPLETE_WORD_LIST,
    get_perception_data_path, get_production_data_path,
)


def _load_eeglab_mat(mat_path):
    """Load an EEGLAB .mat file and return the dictionary."""
    print(f'Loading: {mat_path}')
    return mat73.loadmat(str(mat_path))


def _extract_word_list(eeg_dict, task_cond):
    """
    Extract the trial word list from behavioralData.

    Production and perception data have slightly different nesting in
    the behavioralData structure:
      - Production: trial_list needs flattening (nested lists)
      - Perception: trial_list is already flat, but 'MEEV' entries must be skipped
    """
    word_list = []
    b_data = eeg_dict['behavioralData']
    for bd in b_data:
        em_list = bd[0]['eventMarker']
        trial_list = em_list[1]
        trial_list = trial_list[0]

        if task_cond == 'overtProd':
            # Production data has nested lists that need flattening
            trial_list = [item for sublist in trial_list for item in sublist]
            for word in trial_list:
                word_list.append(word)
        else:
            # Perception data — filter out 'MEEV' filler trials
            for word in trial_list:
                if word != 'MEEV':
                    word_list.append(word)

    return word_list


def _get_good_trial_mask(eeg_dict):
    """Return boolean mask of good (non-artifact) trials."""
    trial_info = eeg_dict['trialInfo']
    return trial_info != 0


def _build_class_labels(word_list, stim_class):
    """
    Assign binary class labels based on stimulus class.

    Returns:
        y_labels: array of 0/1 labels for relevant trials
        keep_mask: boolean array indicating which trials to keep
    """
    y_labels = []
    keep_mask = []

    if stim_class == 'prodDiff':
        class_0, class_1 = PROD_DIFF_TH, PROD_DIFF_F
    elif stim_class == 'percDiff':
        class_0, class_1 = PERC_DIFF_S, PERC_DIFF_T
    else:
        raise ValueError(f'Unknown stim_class: {stim_class}')

    for w in word_list:
        if w in class_0:
            y_labels.append(0)
            keep_mask.append(True)
        elif w in class_1:
            y_labels.append(1)
            keep_mask.append(True)
        else:
            # Trial belongs to the other contrast or is unknown — drop
            keep_mask.append(False)

    return np.array(y_labels), np.array(keep_mask)


def load_subject_epochs(subj_id, task_cond, stim_class):
    """
    Load EEGLAB data for one subject and return MNE Epochs + class labels.

    Parameters
    ----------
    subj_id : str
        Subject ID, e.g. 'EEGPROD4001'.
    task_cond : str
        'perception' or 'overtProd'.
    stim_class : str
        'prodDiff' or 'percDiff'.

    Returns
    -------
    epochs : mne.EpochsArray
        MNE Epochs object with average reference and biosemi64 montage.
    y : np.ndarray
        Binary class labels (0 or 1) for each epoch.
    sfreq : float
        Sampling frequency in Hz.
    """
    # Load the .mat file
    if task_cond == 'perception':
        mat_path = get_perception_data_path(subj_id)
    elif task_cond == 'overtProd':
        mat_path = get_production_data_path(subj_id)
    else:
        raise ValueError(f'Unknown task_cond: {task_cond}')

    eeg_dict = _load_eeglab_mat(mat_path)

    # Channel info
    chan_info = eeg_dict['chanlocs']['labels']

    # Data tensor: EEGLAB stores as (channels, timepoints, trials)
    data = eeg_dict['data']
    # Swap to (trials, channels, timepoints) for MNE
    data = data.swapaxes(2, 0)  # now (trials, timepoints, channels)
    data = data.swapaxes(1, 2)  # now (trials, channels, timepoints)

    # Sampling frequency — read from mat file, fallback to 2048 Hz (BIOSEMI native)
    sfreq = eeg_dict.get('srate', 2048.0)
    if isinstance(sfreq, np.ndarray):
        sfreq = float(sfreq.flat[0])
    sfreq = float(sfreq)

    # Time vector (EEGLAB stores times in milliseconds)
    times = eeg_dict['times']
    if isinstance(times, np.ndarray):
        times = times.flatten()

    # Derive tmin from the actual times array (ms → seconds)
    tmin = times[0] / 1000.0

    # Cross-check sfreq against the times array
    sfreq_from_times = 1000.0 / np.mean(np.diff(times))
    if abs(sfreq - sfreq_from_times) > 1.0:
        print(f'  WARNING: srate={sfreq} Hz but times array implies '
              f'{sfreq_from_times:.1f} Hz — using srate value')
    else:
        print(f'  sfreq={sfreq} Hz (confirmed by times array: '
              f'{sfreq_from_times:.2f} Hz)')

    # Extract word list and artifact mask
    word_list = _extract_word_list(eeg_dict, task_cond)
    good_mask = _get_good_trial_mask(eeg_dict).flatten()

    # Apply artifact rejection
    good_indices = np.where(good_mask)[0]
    data_clean = data[good_indices]
    word_list_clean = [word_list[i] for i in good_indices]

    # Build class labels and filter to relevant trials
    y_labels, class_mask = _build_class_labels(word_list_clean, stim_class)
    data_final = data_clean[class_mask]

    # EEGLAB stores in µV, MNE expects V
    # Check scale: if max > 1e-3, data is likely in µV
    if np.abs(data_final).max() > 1e-3:
        data_final = data_final * 1e-6  # µV → V

    # Create MNE Info
    # Fix channel name case to match biosemi64 montage (e.g. EEGLAB has typo 'Afz', MNE montage expects 'AFz')
    ch_names = list(chan_info)
    montage = mne.channels.make_standard_montage('biosemi64')
    montage_lookup = {name.lower(): name for name in montage.ch_names}
    ch_names = [montage_lookup.get(ch.lower(), ch) for ch in ch_names]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')

    # Create dummy events array (required by MNE Epochs)
    n_epochs = data_final.shape[0]
    events = np.column_stack([
        np.arange(n_epochs),
        np.zeros(n_epochs, dtype=int),
        y_labels
    ])

    # Create EpochsArray
    epochs = mne.EpochsArray(data_final, info, events=events, tmin=tmin,
                             event_id={'class_0': 0, 'class_1': 1},
                             verbose=False)

    # Set montage
    montage = mne.channels.make_standard_montage('biosemi64')
    epochs.set_montage(montage, on_missing='warn')

    # Set average reference (mandatory for source estimation)
    epochs.set_eeg_reference('average', projection=True)
    epochs.apply_proj()

    print(f'  {subj_id}: {n_epochs} epochs, {len(ch_names)} channels, '
          f'sfreq={sfreq} Hz, tmin={tmin}')

    return epochs, y_labels, sfreq
