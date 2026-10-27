# utils/plots.py
# -----------------------------------
# All visualization helpers for Streamlit
# -----------------------------------

import numpy as np
import matplotlib.pyplot as plt
import mne
from sklearn.metrics import confusion_matrix


# -------------------------------------------------
# 1) EEG sensor layout (motor cortex explanation)
# -------------------------------------------------
def plot_sensor_layout(epochs):
    """
    Plot EEG sensor positions with channel names.
    Used to explain motor cortex regions (C3, Cz, C4).

    Parameters
    ----------
    epochs : mne.Epochs

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig = epochs.plot_sensors(show_names=True)
    return fig


# -------------------------------------------------
# 2) CSP Topomap (spatial patterns)
# -------------------------------------------------
def plot_csp_topomap(csp, epochs_info, ch_names, components=(0, 1), title_prefix="CSP"):
    """
    Plot CSP spatial patterns as topomaps.

    Parameters
    ----------
    csp : fitted CSP object
    epochs_info : mne.Info
    ch_names : list
        Channel names used during training
    components : tuple
        CSP components to visualize
    title_prefix : str

    Returns
    -------
    figs : list of matplotlib figures
    """
    figs = []

    montage = mne.channels.make_standard_montage("standard_1020")
    valid_1020 = set(montage.ch_names)
    valid_chs = [ch for ch in ch_names if ch in valid_1020]

    idx = [ch_names.index(ch) for ch in valid_chs]

    for comp in components:
        fig, ax = plt.subplots(figsize=(4, 4))
        mne.viz.plot_topomap(
            csp.patterns_[comp][idx],
            epochs_info,
            contours=0,
            axes=ax,
            show=False
        )
        ax.set_title(f"{title_prefix} – Pattern {comp+1}")
        figs.append(fig)

    return figs


# -------------------------------------------------
# 3) Frequency relevance plot (band vs accuracy)
# -------------------------------------------------
def plot_band_accuracy(bands, accuracies):
    """
    Plot accuracy vs frequency band.

    Parameters
    ----------
    bands : list of tuple
    accuracies : list

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    band_labels = [f"{l}-{h}" for l, h in bands]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(band_labels, accuracies, marker="o")
    ax.set_xlabel("Frequency Band (Hz)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Frequency Band Relevance")
    ax.grid(True)

    return fig


# -------------------------------------------------
# 4) Confusion matrix plot
# -------------------------------------------------
def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plot confusion matrix.

    Parameters
    ----------
    y_true : array
    y_pred : array
    class_names : list

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    fig.colorbar(im, ax=ax)
    return fig
