#loading + preprocessing for BCI Competition IV 2a (.gdf files)

import os
import zipfile

import numpy as np
import mne


CLASS_NAMES = ["Left", "Right", "Foot", "Tongue"]
TRAIN_SUBJECTS = [f"A0{i}T.gdf" for i in range(1, 8)]
TEST_SUBJECTS = ["A08T.gdf", "A09T.gdf"]


#unzip the dataset archive (e.g. on Colab after mounting Google Drive)
def extract_dataset(zip_path, extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Dataset extracted!")


#for scaling all subjects to a similar scale
def zscore_trials(X):
    mean = X.mean(axis=2, keepdims=True)
    std  = X.std(axis=2, keepdims=True) + 1e-8
    return (X - mean) / std


#loading subjects and make epochs
def load_subject_mi_epochs(subject_file, data_path="/content/eeg_data", tmin=2.0, tmax=6.0):
    gdf_file = os.path.join(data_path, subject_file)
    raw = mne.io.read_raw_gdf(gdf_file, preload=True, verbose=False)
    events, event_dict = mne.events_from_annotations(raw, verbose=False)
    code_to_class = {"769": 0, "770": 1, "771": 2, "772": 3}
    for code in code_to_class.keys():
      if code not in event_dict:
        raise ValueError(f"Missing MI event {code} in {subject_file}")
    mi_event_ints = [event_dict[c] for c in code_to_class.keys()]
    events_mi = events[np.isin(events[:, 2], mi_event_ints)]

    #extract eeg channels
    raw_eeg = raw.copy().pick(raw.ch_names[:22])
    #notch filter
    raw_eeg.notch_filter(50, verbose=False)
    #bandpass filter
    raw_eeg.filter(8, 30, verbose=False) #mu (8-12) and beta (13-30)

    #make epochs
    epochs = mne.Epochs(
    raw_eeg,
    events_mi,
    event_id={k: event_dict[k] for k in code_to_class.keys()},
    tmin=tmin,
    tmax=tmax,
    baseline=None,
    preload=True,
    verbose=False)
    rename_map = {}
    for ch in epochs.ch_names:
        if ch.startswith("EEG-"):
            rename_map[ch] = ch.replace("EEG-", "")
    if len(rename_map) > 0:
        epochs.rename_channels(rename_map)

    montage = mne.channels.make_standard_montage("standard_1020")
    epochs.set_montage(montage, on_missing="ignore")

    X = epochs.get_data()
    y_evt = epochs.events[:, 2]

    inv_map = {event_dict[k]: v for k, v in code_to_class.items()}
    y = np.array([inv_map[e] for e in y_evt], dtype=int)

    return X, y,epochs #X is epochs (trials,channels,samples) and Y denotes labels {0,1,2,3}


#load several subjects and stack their trials
def load_subjects(subject_files, data_path, tmin=2.0, tmax=6.0):
    X_list, y_list = [], []
    for sub in subject_files:
        Xs, ys, _ = load_subject_mi_epochs(sub, data_path, tmin=tmin, tmax=tmax)
        X_list.append(Xs)
        y_list.append(ys)
    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)
