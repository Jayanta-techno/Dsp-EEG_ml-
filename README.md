# EEG_ml

**Cross-subject motor imagery classification using Filter Bank Common Spatial Patterns (FBCSP)**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://projectdsp.streamlit.app/)
[![Dataset](https://img.shields.io/badge/Dataset-BCI%20Competition%20IV%202a-00897B)](https://www.bbci.de/competition/iv/#dataset2a)

[Live Demo](https://projectdsp.streamlit.app/) · [Results](#results) · [Methodology](#methodology) · [Quick Start](#quick-start)
 
---

## Overview

EEG_ml is a machine-learning pipeline that classifies four motor-imagery tasks from EEG recordings. It uses FBCSP to extract spatial-frequency features and Logistic Regression to predict the imagined movement.

The central experiment is **zero-shot cross-subject classification**: the model is trained on subjects A01–A07 and evaluated on A08–A09, whose data is never used for fitting.

**Goal:** measure how well a model trained on some people generalises to motor-imagery EEG from previously unseen people, without subject-specific calibration.

| Class 1 | Class 2 | Class 3 | Class 4 |
|:-------:|:-------:|:-------:|:-------:|
| Left hand | Right hand | Feet | Tongue |

## Results

| Metric | Main FBCSP model | One-vs-Rest FBCSP |
|--------|:----------------:|:-----------------:|
| Cross-subject accuracy | **42.9%** | **48.6%** |
| Cohen's kappa | 0.238 | 0.315 |
| Test subjects | A08–A09 | A08–A09 |

- Four-class chance-level accuracy is 25%.
- The one-vs-rest variant achieved the best accuracy among the configurations reported.
- Performance varies between subjects, reflecting the inter-subject variability of EEG.

> These are reported experimental results and may not reproduce exactly on every run. See `metrics.json` and the project report for run-specific details.

## Methodology

```text
EEG recordings
      │
      ▼
Preprocessing      22 channels · 50 Hz notch · 8–30 Hz band-pass
      │
      ▼
Epoching           trial extraction · channel-wise z-score
      │
      ▼
FBCSP              bands 8–12 / 12–20 / 20–30 Hz · 6 CSP components per band
      │
      ▼
Feature selection  standardise · 18 features → 15 (mutual information)
      │
      ▼
Classification     class-balanced Logistic Regression · 4 classes
```

| Stage | Description |
|-------|-------------|
| Preprocessing | Select 22 EEG channels; apply a 50 Hz notch filter and an 8–30 Hz band-pass filter. |
| Epoching | Extract trials around cue events `769–772`; z-score each trial channel-wise. |
| FBCSP | Extract six CSP features from each of three frequency bands, yielding 18 features. |
| Feature selection | Standardise features and keep 15 using mutual information. |
| Classification | Predict left hand, right hand, foot, or tongue with class-balanced Logistic Regression. |

> **Epoch window:** the project report specifies 0.5–2.5 s after the cue. Verify the current preprocessing code before comparing results across versions.

## Evaluation Modes

| Mode | Description | Zero-shot |
|------|-------------|:---------:|
| Cross-subject evaluation | Train on A01–A07; test on unseen A08–A09 without using their labels. | Yes |
| Subject-calibrated prediction | Fit a subject-specific classifier on labelled target-subject trials, then predict the remaining trials. | No |
| Unlabelled evaluation file | Predict on an unlabelled `E` recording with no target-subject labels for calibration. | Yes |

Subject-calibrated mode requires at least five labelled trials per class reserved for target-specific training. Report calibrated results separately from zero-shot results.

## Live Application

**[Open the Streamlit demo](https://projectdsp.streamlit.app/)**

| Section | Purpose |
|---------|---------|
| Problem Statement | Motivation and classification objective |
| Motor Cortex | Electrode layout and motor-cortex context |
| Frequency Analysis | Performance of individual frequency bands |
| Model Pipeline | Pipeline overview and CSP spatial-pattern topomaps |
| Prediction | Upload a `.gdf` file, choose a trial, and view class probabilities |
| Subject-Calibrated Prediction | Train a target-specific model from labelled trials |
| Streaming Prediction | Sliding-window predictions with adjustable window and step |
| Performance | Accuracy, Cohen's kappa, and confusion matrix |

## Project Structure

```text
EEG_ml/
├── data_preprocessing.py
├── fbcsp_feature_extraction.py
├── cross_subject_mode_training.py
├── evaluation+prediction_cross_subject.py
├── ovrFBCSP_feature Extraction+training.py
├── requirements.txt
└── streamlit_app/
    ├── app.py
    ├── models/                 # Generated after training
    │   ├── classifier.pkl
    │   ├── csp_models.pkl
    │   └── metrics.json
    └── utils/                  # Inference and visualisation helpers
```

| File | Responsibility |
|------|----------------|
| `data_preprocessing.py` | Loads GDF files, filters EEG, extracts epochs, normalises trials |
| `fbcsp_feature_extraction.py` | Filter-bank CSP feature extraction |
| `cross_subject_mode_training.py` | Trains and evaluates the model; saves models and metrics |
| `evaluation+prediction_cross_subject.py` | Re-evaluates saved models; runs a sliding-window demo |
| `ovrFBCSP_feature Extraction+training.py` | Alternative one-vs-rest FBCSP experiment |
| `streamlit_app/app.py` | Entry point for the web application |
| `streamlit_app/models/` | Generated classifier, CSP models, and metrics |
| `streamlit_app/utils/` | Inference, preprocessing, streaming, and plotting helpers |

## Quick Start

### Requirements

- Python 3.10 or newer
- Packages listed in `requirements.txt`
- BCI Competition IV Dataset 2a training files `A01T.gdf` to `A09T.gdf`

### Installation

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Dataset

Download [BCI Competition IV Dataset 2a](https://www.bbci.de/competition/iv/#dataset2a) and place the labelled training files in a `data/` folder:

```text
data/
├── A01T.gdf
├── A02T.gdf
├── ...
└── A09T.gdf
```

### Training

```bash
python cross_subject_mode_training.py --data-path data
```

The script reports accuracy, kappa, a classification report, a confusion matrix, and per-band accuracy. It saves `classifier.pkl`, `csp_models.pkl`, and `metrics.json` to `streamlit_app/models/`.

### Evaluation and streaming (optional)

```bash
python evaluation+prediction_cross_subject.py --data-path data --trial 0
```

Re-evaluates the saved model on A08–A09 and demonstrates causal sliding-window prediction. The plot is saved as `streaming_prediction.png`.

### One-vs-rest experiment (optional)

```bash
python "ovrFBCSP_feature Extraction+training.py" --data-path data
```

### Launch the app

```bash
streamlit run streamlit_app/app.py
```

Then open http://localhost:8501.

## Google Colab

Upload your dataset archive to Google Drive, then replace the placeholders with your repository URL and file path.

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
!git clone <YOUR_REPOSITORY_URL>
%cd EEG_ml
!pip install -q -r requirements.txt
```

If using a dataset archive:

```python
from data_preprocessing import extract_dataset

extract_dataset(
    "/content/drive/MyDrive/<dataset>.zip",
    "/content/eeg_data"
)
```

```bash
!python cross_subject_mode_training.py --data-path /content/eeg_data
```

If the dataset is already extracted, skip the extraction step and pass its folder to `--data-path`.

## Tech Stack

| Technology | Role |
|------------|------|
| MNE-Python | EEG/GDF loading, filtering, epoching, CSP, topographic plots |
| NumPy / SciPy | Numerical computing and signal processing |
| scikit-learn | Scaling, feature selection, classification, evaluation |
| Matplotlib | Plots and visualisations |
| joblib | Saving and loading trained models |
| Streamlit | Interactive web interface |

## Dataset and Reference

This project uses [BCI Competition IV Dataset 2a](https://www.bbci.de/competition/iv/#dataset2a), which contains EEG recordings for four motor-imagery classes.

For background on FBCSP:

> Ang, K. K., Chin, Z. Y., Wang, C., Guan, C., and Zhang, H. *Filter Bank Common Spatial Pattern Algorithm on BCI Competition IV Datasets 2a and 2b.*

---

**[Launch the live app](https://projectdsp.streamlit.app/)**
