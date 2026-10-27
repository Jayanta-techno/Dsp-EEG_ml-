# utils/preprocessing.py
# -----------------------------------
# Streamlit helper functions ONLY
# (ML training code is NOT touched)
# -----------------------------------

import os
import tempfile

import numpy as np
import mne


# 22 EEG channels of BCI Competition IV 2a (in recording order)
CH_NAMES_2A = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2", "POz"
]

# MI cue codes -> class index (same mapping as training)
MI_CODES = {"769": 0, "770": 1, "771": 2, "772": 3}
# cue of unknown class (used in the evaluation "E" files)
UNKNOWN_CUE = "783"


def _epochable_event_mask(raw, events, tmin=2.0, tmax=6.0):
    start = events[:, 0] + int(round(tmin * raw.info["sfreq"]))
    stop = events[:, 0] + int(round(tmax * raw.info["sfreq"]))
    return (start >= raw.first_samp) & (stop <= raw.last_samp)


def read_uploaded_gdf(file_bytes):
    """
    Read a GDF file uploaded through Streamlit.
    MNE needs a real file path, so the bytes are written to a temp file.

    Parameters
    ----------
    file_bytes : bytes

    Returns
    -------
    raw : mne.io.Raw
    """
    fd, path = tempfile.mkstemp(suffix=".gdf")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(file_bytes)
        raw = mne.io.read_raw_gdf(path, preload=True, verbose=False)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass  # file may still be locked on Windows

    return raw


def get_mi_events(raw):
    """
    Find the motor imagery cue events in a recording.

    Parameters
    ----------
    raw : mne.io.Raw

    Returns
    -------
    events : np.ndarray
        Shape (n_trials, 3), one row per MI trial
    labels : np.ndarray or None
        True class per trial, or None if the file has no labels
    """
    events, event_dict = mne.events_from_annotations(raw, verbose=False)

    present = {code: event_dict[code] for code in MI_CODES if code in event_dict}
    if present:
        mi_events = events[np.isin(events[:, 2], list(present.values()))]
        inv_map = {event_dict[c]: MI_CODES[c] for c in present}
        labels = np.array([inv_map[e] for e in mi_events[:, 2]], dtype=int)
        valid = _epochable_event_mask(raw, mi_events)
        return mi_events[valid], labels[valid]

    if UNKNOWN_CUE in event_dict:
        mi_events = events[events[:, 2] == event_dict[UNKNOWN_CUE]]
        return mi_events[_epochable_event_mask(raw, mi_events)], None

    return events[:0], None


def zscore_single_trial(X):
    """
    Z-score normalize ONE EEG trial.

    Parameters
    ----------
    X : np.ndarray
        Shape (channels, samples)

    Returns
    -------
    X_norm : np.ndarray
        Shape (channels, samples)
    """
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True) + 1e-8
    return (X - mean) / std


def basic_preprocess_raw(raw):
    """
    Minimal preprocessing for Streamlit demo.
    (Same idea as training, but lightweight)

    Steps:
    - pick first 22 EEG channels
    - notch filter (50 Hz)
    - bandpass filter (8–30 Hz)

    Parameters
    ----------
    raw : mne.io.Raw

    Returns
    -------
    raw : mne.io.Raw
    """
    raw = raw.copy()
    raw.pick(raw.ch_names[:22])
    raw.notch_filter(50, verbose=False)
    raw.filter(8, 30, verbose=False)
    return raw


def extract_single_epoch(raw, events, trial_idx,
                         tmin=2.0, tmax=6.0):
    """
    Extract ONE EEG trial for prediction/demo.

    Parameters
    ----------
    raw : mne.io.Raw
    events : ndarray
        MI events (see get_mi_events)
    trial_idx : int
    tmin, tmax : float

    Returns
    -------
    X_trial : np.ndarray
        Shape (channels, samples)
    """
    epochs = mne.Epochs(
        raw,
        events[trial_idx:trial_idx + 1],
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose=False
    )

    # return (channels, samples)
    return epochs.get_data()[0]
