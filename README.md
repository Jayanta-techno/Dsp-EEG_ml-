# EEG_ml — Cross-Subject Motor Imagery Classification

Classifies four motor imagery tasks (**Left hand, Right hand, Foot, Tongue**) from EEG using
**Filter Bank Common Spatial Patterns (FBCSP)** and logistic regression, in a **zero-shot
cross-subject** setting: the model is trained on subjects A01–A07 of
[BCI Competition IV dataset 2a](https://www.bbci.de/competition/iv/#dataset2a) and tested on
subjects A08–A09, whose data it has never seen. A Streamlit app shows the pipeline and makes
predictions on uploaded recordings.

## Pipeline

1. Pick the 22 EEG channels, apply a 50 Hz notch filter and an 8–30 Hz bandpass filter
2. Cut epochs from 2 to 6 s after each cue (codes 769–772) and z-score each trial
3. FBCSP: bands 8–12, 12–20 and 20–30 Hz, with a regularised 6-component CSP per band (18 features)
4. StandardScaler → SelectKBest (mutual information, k=15) → LogisticRegression

## Project structure

```
data_preprocessing.py                   # GDF loading, filtering, epoching, z-scoring
fbcsp_feature_extraction.py             # FBCSP feature extraction
cross_subject_mode_training.py          # train + evaluate, save models and metrics for the app
evaluation+prediction_cross_subject.py  # re-evaluate saved models + sliding-window prediction demo
ovrFBCSP_feature Extraction+training.py # alternative experiment: one-vs-rest FBCSP
streamlit_app/
  app.py                                # Streamlit web app
  models/                               # created by training (classifier.pkl, csp_models.pkl, metrics.json)
  utils/                                # inference helpers: preprocessing, FBCSP, streaming, plots
```

## Setup

Requires Python 3.10 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

Download the dataset files `A01T.gdf` to `A09T.gdf` (BCI Competition IV 2a) into one folder,
for example `data/`.

## Usage

Run everything from the repository root.

**1. Train**

```bash
python cross_subject_mode_training.py --data-path data
```

This prints accuracy, kappa, a classification report, a confusion matrix and the accuracy of each
band. It saves `classifier.pkl`, `csp_models.pkl` and `metrics.json` to `streamlit_app/models/`.

**2. Evaluate + streaming demo (optional)**

```bash
python evaluation+prediction_cross_subject.py --data-path data --trial 0
```

This re-evaluates the saved models on A08–A09, runs a causal 1 s sliding-window prediction on
one test trial, and saves the plot to `streaming_prediction.png`.

**3. One-vs-rest FBCSP experiment (optional)**

```bash
python "ovrFBCSP_feature Extraction+training.py" --data-path data
```

**4. Streamlit app**

```bash
streamlit run streamlit_app/app.py
```

Open http://localhost:8501. The app has these pages:

- Problem statement and motor cortex sensor layout
- Accuracy of each frequency band
- Learned CSP spatial patterns as topomaps
- Prediction: upload a `.gdf` file, pick a trial, and see the predicted class with its probabilities (and the true class for labelled `A0xT` files)
- Subject-calibrated prediction: for labelled files, reserve at least five trials per class for a target-specific classifier; predictions are made only on the remaining trials. This uses target-subject labels and is not zero-shot. Unlabelled `E` files remain zero-shot only.
- Streaming prediction with adjustable window and step
- Cross-subject performance (accuracy, kappa, confusion matrix)

### Google Colab

```python
from google.colab import drive
drive.mount('/content/drive')
!git clone <this repo>
%cd EEG_ml
!pip install -q -r requirements.txt

from data_preprocessing import extract_dataset
extract_dataset("/content/drive/MyDrive/<dataset>.zip", "/content/eeg_data")
!python cross_subject_mode_training.py --data-path /content/eeg_data
```

### Deployed web app link
https://projectdsp.streamlit.app/
