# utils/streaming.py
# -----------------------------------
# Streaming / sliding-window prediction
# (Inference ONLY – no training)
# -----------------------------------

import numpy as np
from mne.filter import filter_data


CLASS_NAMES = ["Left", "Right", "Foot", "Tongue"]


def zscore_window(X):
    """
    Z-score normalize a single EEG window.

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


def fbcsp_stream_predict(
    trial_data,
    csp_models,
    clf,
    bands,
    fs=250,
    win_sec=1.0,
    step_sec=1.0
):
    """
    Perform streaming-style prediction on one EEG trial
    using sliding windows.

    Parameters
    ----------
    trial_data : np.ndarray
        Shape (channels, samples)
    csp_models : list
        Pre-trained CSP models (one per band)
    clf : sklearn classifier
        Pre-trained classifier
    bands : list of tuple
        [(low, high), ...]
    fs : int
        Sampling frequency
    win_sec : float
        Window size in seconds
    step_sec : float
        Step size in seconds

    Returns
    -------
    times : list
        Time (in seconds) of each prediction
    preds : list
        Predicted class indices
    pred_names : list
        Predicted class names
    """

    win_samp = int(win_sec * fs)
    step_samp = int(step_sec * fs)

    times = []
    preds = []

    start = 0
    n_samples = trial_data.shape[1]

    while start + win_samp <= n_samples:
        end = start + win_samp

        # causal window (past data only)
        X_win = trial_data[:, start:end]

        # normalize window
        X_win = zscore_window(X_win)

        # reshape to (1, ch, time)
        X_win = X_win[np.newaxis, :, :]

        feat_list = []

        for (l_freq, h_freq), csp in zip(bands, csp_models):
            Xf = filter_data(
                X_win,
                sfreq=fs,
                l_freq=l_freq,
                h_freq=h_freq,
                verbose=False
            )

            feat = csp.transform(Xf)  # (1, n_csp)
            feat_list.append(feat)

        # concatenate features from all bands
        feat_all = np.concatenate(feat_list, axis=1)

        # predict
        pred = int(clf.predict(feat_all)[0])

        times.append(end / fs)
        preds.append(pred)

        start += step_samp

    pred_names = [CLASS_NAMES[p] for p in preds]

    return times, preds, pred_names
