# utils/fbcsp.py
# -----------------------------------
# FBCSP feature extraction (INFERENCE ONLY)
# Uses pre-trained CSP models
# -----------------------------------

import numpy as np
from mne.filter import filter_data


def extract_fbcsp_features_single_trial(
    trial_data,
    csp_models,
    bands,
    fs=250
):
    """
    Extract FBCSP features for ONE EEG trial using
    pre-trained CSP models.

    Parameters
    ----------
    trial_data : np.ndarray
        Shape (channels, samples)
    csp_models : list
        List of trained CSP objects (one per band)
    bands : list of tuple
        [(low_freq, high_freq), ...]
    fs : int
        Sampling frequency

    Returns
    -------
    feat_vec : np.ndarray
        Shape (1, total_features)
    """

    # reshape to (1, ch, time) for MNE
    X = trial_data[np.newaxis, :, :]

    feats = []

    for (l_freq, h_freq), csp in zip(bands, csp_models):
        # Band-pass filter
        Xf = filter_data(
            X,
            sfreq=fs,
            l_freq=l_freq,
            h_freq=h_freq,
            verbose=False
        )

        # CSP transform (NO fitting here)
        feat = csp.transform(Xf)  # shape (1, n_components)
        feats.append(feat)

    # concatenate features from all bands
    feat_vec = np.concatenate(feats, axis=1)

    return feat_vec


def extract_fbcsp_features_batch(
    X,
    csp_models,
    bands,
    fs=250
):
    """
    Extract FBCSP features for MULTIPLE trials.
    Useful for plotting / evaluation inside Streamlit.

    Parameters
    ----------
    X : np.ndarray
        Shape (n_trials, channels, samples)
    csp_models : list
    bands : list of tuple
    fs : int

    Returns
    -------
    X_feat : np.ndarray
        Shape (n_trials, total_features)
    """

    feats_all = []

    for (l_freq, h_freq), csp in zip(bands, csp_models):
        Xf = filter_data(
            X,
            sfreq=fs,
            l_freq=l_freq,
            h_freq=h_freq,
            verbose=False
        )

        feats = csp.transform(Xf)
        feats_all.append(feats)

    return np.concatenate(feats_all, axis=1)
