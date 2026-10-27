import json
from pathlib import Path

import streamlit as st
import joblib
import numpy as np
import mne
import matplotlib.pyplot as plt

from utils.preprocessing import (
    CH_NAMES_2A,
    read_uploaded_gdf,
    get_mi_events,
    basic_preprocess_raw,
    extract_single_epoch,
    zscore_single_trial
)

from utils.calibration import (
    fit_subject_classifier,
    split_calibration_indices,
)
from utils.fbcsp import (
    extract_fbcsp_features_batch,
    extract_fbcsp_features_single_trial,
)
from utils.streaming import fbcsp_stream_predict
from utils.plots import (
    plot_sensor_layout,
    plot_csp_topomap,
    plot_band_accuracy,
    plot_confusion_matrix
)

# -------------------------------------------------
# BASIC CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Motor Imagery EEG System",
    layout="wide"
)

CLASS_NAMES = ["Left", "Right", "Foot", "Tongue"]
BANDS = [(8,12), (12,20), (20,30)]
FS = 250
MODELS_DIR = Path(__file__).parent / "models"

# -------------------------------------------------
# LOAD TRAINED MODELS (NO TRAINING HERE)
# -------------------------------------------------
@st.cache_resource
def load_models():
    clf = joblib.load(MODELS_DIR / "classifier.pkl")
    csp_models = joblib.load(MODELS_DIR / "csp_models.pkl")
    return clf, csp_models


@st.cache_data
def load_metrics():
    path = MODELS_DIR / "metrics.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_recording(file_bytes):
    raw = read_uploaded_gdf(file_bytes)
    raw = basic_preprocess_raw(raw)
    events, labels = get_mi_events(raw)
    return raw, events, labels


def montage_info():
    info = mne.create_info(ch_names=CH_NAMES_2A, sfreq=FS, ch_types="eeg")
    info.set_montage("standard_1020")
    return info


try:
    clf, csp_models = load_models()
except FileNotFoundError:
    st.error(
        "Trained models not found in `streamlit_app/models/`. "
        "Run `python cross_subject_mode_training.py --data-path <gdf folder>` first."
    )
    st.stop()

metrics = load_metrics()

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Problem Statement",
        "Motor Cortex",
        "Frequency Analysis",
        "Model Pipeline",
        "Prediction",
        "Streaming Prediction",
        "Performance"
    ]
)

# -------------------------------------------------
# PAGE 1 — PROBLEM STATEMENT
# -------------------------------------------------
if page == "Problem Statement":
    st.title("📘 Problem Statement")

    st.markdown("""
Motor Imagery (MI) refers to the mental simulation of movement
without physical execution. EEG-based MI classification is a
core problem in Brain–Computer Interfaces (BCI).

### Challenges
- Low signal-to-noise ratio
- High inter-subject variability
- Overlapping EEG rhythms

### Objective
Classify **four motor imagery tasks**:
- Left Hand
- Right Hand
- Foot
- Tongue

in a **zero-shot cross-subject setting**: the model is trained on
subjects A01–A07 of BCI Competition IV 2a and tested on the unseen
subjects A08–A09.
""")

# -------------------------------------------------
# PAGE 2 — MOTOR CORTEX
# -------------------------------------------------
elif page == "Motor Cortex":
    st.title("🧠 Motor Cortex & EEG Channels")

    st.markdown("""
Motor imagery activates the **sensorimotor cortex**.

Key EEG channels:
- **C3** → Right hand (contralateral)
- **C4** → Left hand (contralateral)
- **Cz** → Foot
""")

    st.markdown("### EEG Sensor Layout (22 channels, BCI IV 2a)")

    # Dummy epochs only for visualization
    dummy_epochs = mne.EpochsArray(
        np.random.randn(1, len(CH_NAMES_2A), FS),
        montage_info(),
        verbose=False
    )

    fig = plot_sensor_layout(dummy_epochs)
    st.pyplot(fig)

# -------------------------------------------------
# PAGE 3 — FREQUENCY ANALYSIS
# -------------------------------------------------
elif page == "Frequency Analysis":
    st.title("📊 Frequency Band Relevance")

    st.markdown("""
Motor imagery information is mainly present in:
- **Mu rhythm (8–12 Hz)**
- **Beta rhythm (12–30 Hz)**

The plot shows the cross-subject test accuracy of a classifier
trained on the CSP features of each band alone (chance = 25%).
""")

    if metrics and "band_accuracy" in metrics:
        fig = plot_band_accuracy(BANDS, metrics["band_accuracy"])
        st.pyplot(fig)
    else:
        st.info("No band accuracies found. Re-run the training script to generate `models/metrics.json`.")

# -------------------------------------------------
# PAGE 4 — MODEL PIPELINE
# -------------------------------------------------
elif page == "Model Pipeline":
    st.title("🧩 Model Pipeline")

    st.markdown("""
**Pipeline Overview**

1. EEG preprocessing (50 Hz notch + 8–30 Hz bandpass, 22 EEG channels)
2. Epoching 2–6 s after the cue + trial-wise z-score normalization
3. Filter Bank Common Spatial Patterns (FBCSP): 3 bands × 6 CSP components
4. StandardScaler → SelectKBest (mutual information, k=15) → Logistic Regression

This classical pipeline is robust and interpretable
for EEG-based MI classification.
""")

    st.markdown("### Learned CSP spatial patterns")
    band_idx = st.selectbox(
        "Frequency band",
        range(len(BANDS)),
        format_func=lambda i: f"{BANDS[i][0]}–{BANDS[i][1]} Hz"
    )
    figs = plot_csp_topomap(
        csp_models[band_idx],
        montage_info(),
        CH_NAMES_2A,
        components=(0, 1, 2),
        title_prefix=f"{BANDS[band_idx][0]}–{BANDS[band_idx][1]} Hz"
    )
    for col, fig in zip(st.columns(len(figs)), figs):
        col.pyplot(fig)

# -------------------------------------------------
# PAGE 5 — PREDICTION (OFFLINE DEMO)
# -------------------------------------------------
elif page == "Prediction":
    st.title("🎯 Motor Imagery Prediction")

    uploaded = st.file_uploader("Upload EEG GDF file", type=["gdf"])

    if uploaded:
        raw, events, labels = load_recording(uploaded.getvalue())

        if len(events) == 0:
            st.error("No motor imagery cue events (769–772 or 783) found in this file.")
            st.stop()

        prediction_mode = "Zero-shot"
        calibration_indices = prediction_indices = None
        calibration_per_class = 0
        if labels is not None:
            prediction_mode = st.radio(
                "Prediction mode",
                ["Zero-shot", "Subject-calibrated"],
                horizontal=True,
                help=(
                    "Subject calibration uses labeled trials from this recording "
                    "and predicts only the remaining trials."
                ),
            )

        if prediction_mode == "Subject-calibrated":
            class_counts = np.bincount(labels, minlength=len(CLASS_NAMES))
            if np.any(class_counts == 0):
                st.error("Calibration requires labeled trials from all four classes.")
                st.stop()

            max_calibration = min(int(class_counts.min()) - 1, 50)
            if max_calibration < 5:
                st.error("Each class needs at least six labeled trials for calibration.")
                st.stop()
            calibration_per_class = st.slider(
                "Labeled calibration trials per class",
                min_value=5,
                max_value=max_calibration,
                value=min(20, max_calibration),
            )
            calibration_indices, prediction_indices = split_calibration_indices(
                labels, calibration_per_class
            )
            st.caption(
                f"Using {len(calibration_indices)} labeled trials for calibration; "
                f"predictions are limited to the other {len(prediction_indices)} trials."
            )
            prediction_position = st.slider(
                "Select prediction trial",
                0,
                len(prediction_indices) - 1,
                0,
            )
            trial_idx = int(prediction_indices[prediction_position])
        else:
            trial_idx = st.slider("Select trial index", 0, len(events)-1, 0)

        if st.button("Predict"):
            active_classifier = clf
            if prediction_mode == "Subject-calibrated":
                with st.spinner("Calibrating on the labeled trials..."):
                    epochs = mne.Epochs(
                        raw,
                        events,
                        tmin=2.0,
                        tmax=6.0,
                        baseline=None,
                        preload=True,
                        verbose=False,
                    )
                    if len(epochs) != len(events):
                        st.error("Could not extract every trial from this recording.")
                        st.stop()

                    trial_data = epochs.get_data()
                    trial_data = (
                        trial_data
                        - trial_data.mean(axis=2, keepdims=True)
                    ) / (trial_data.std(axis=2, keepdims=True) + 1e-8)
                    recording_features = extract_fbcsp_features_batch(
                        trial_data, csp_models, BANDS, fs=FS
                    )
                    active_classifier = fit_subject_classifier(
                        recording_features[calibration_indices],
                        labels[calibration_indices],
                    )
                    feat = recording_features[[trial_idx]]
            else:
                X_trial = extract_single_epoch(raw, events, trial_idx)
                X_trial = zscore_single_trial(X_trial)
                feat = extract_fbcsp_features_single_trial(
                    X_trial, csp_models, BANDS
                )

            pred = active_classifier.predict(feat)[0]
            proba = active_classifier.predict_proba(feat)[0]

            st.subheader(f"🧠 Predicted Class: **{CLASS_NAMES[pred]}**")
            if labels is not None:
                true = labels[trial_idx]
                mark = "✅" if true == pred else "❌"
                st.write(f"True class: **{CLASS_NAMES[true]}** {mark}")

            fig, ax = plt.subplots(figsize=(6, 3))
            ax.bar(CLASS_NAMES, proba)
            ax.set_ylim(0, 1)
            ax.set_ylabel("Probability")
            ax.set_title("Class probabilities")
            st.pyplot(fig)

# -------------------------------------------------
# PAGE 6 — STREAMING PREDICTION
# -------------------------------------------------
elif page == "Streaming Prediction":
    st.title("📈 Streaming Prediction")

    st.markdown("""
Sliding-window decoding simulates real-time BCI inference.
""")

    uploaded = st.file_uploader("Upload EEG GDF file for streaming", type=["gdf"])

    if uploaded:
        raw, events, labels = load_recording(uploaded.getvalue())

        if len(events) == 0:
            st.error("No motor imagery cue events (769–772 or 783) found in this file.")
            st.stop()

        trial_idx = st.slider("Select trial index", 0, len(events)-1, 0)
        win_sec = st.slider("Window length (s)", 0.5, 2.0, 1.0, 0.5)
        step_sec = st.slider("Step (s)", 0.25, 1.0, 0.5, 0.25)

        X_trial = extract_single_epoch(raw, events, trial_idx)

        times, preds, names = fbcsp_stream_predict(
            X_trial,
            csp_models,
            clf,
            BANDS,
            fs=FS,
            win_sec=win_sec,
            step_sec=step_sec
        )

        if labels is not None:
            st.write(f"True class: **{CLASS_NAMES[labels[trial_idx]]}**")

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.step(times, preds, where="post", marker="o")
        ax.set_yticks(range(len(CLASS_NAMES)))
        ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel("Time within trial (s)")
        ax.set_ylabel("Predicted class")
        ax.set_title("Causal Continuous Prediction (FBCSP)")
        ax.grid(True)
        st.pyplot(fig)

        st.write("Predicted sequence:", names)

# -------------------------------------------------
# PAGE 7 — PERFORMANCE
# -------------------------------------------------
elif page == "Performance":
    st.title("📊 Model Performance")

    st.markdown("""
Model evaluated in **zero-shot cross-subject** setting
(trained on A01–A07, tested on A08–A09; chance level = 25%).
""")

    if metrics is None:
        st.info("No metrics found. Re-run the training script to generate `models/metrics.json`.")
    else:
        col1, col2 = st.columns(2)
        col1.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
        col2.metric("Kappa", f"{metrics['kappa']:.2f}")

        fig = plot_confusion_matrix(metrics["y_true"], metrics["y_pred"], CLASS_NAMES)
        st.pyplot(fig)
