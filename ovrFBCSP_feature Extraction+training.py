#alternative experiment: one-vs-rest FBCSP (one CSP per class per band)
#usage: python "ovrFBCSP_feature Extraction+training.py" --data-path path/to/gdf_folder

import argparse

import numpy as np
from mne.decoding import CSP
from mne.filter import filter_data
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report, confusion_matrix

from data_preprocessing import CLASS_NAMES, TRAIN_SUBJECTS, TEST_SUBJECTS, load_subjects, zscore_trials
from fbcsp_feature_extraction import BANDS


def ovr_fbcsp_features(X_train, y_train, X_test, fs=250, n_csp=4):

    classes = [0, 1, 2, 3]

    feat_train_all = []
    feat_test_all = []

    for (l, h) in BANDS:
        Xtr_f = filter_data(X_train, sfreq=fs, l_freq=l, h_freq=h, verbose=False)
        Xte_f = filter_data(X_test,  sfreq=fs, l_freq=l, h_freq=h, verbose=False)

        band_train_feats = []
        band_test_feats = []

        for c in classes:

            y_bin = (y_train == c).astype(int)

            csp = CSP(n_components=n_csp, reg="ledoit_wolf", log=True, norm_trace=False)

            Ft = csp.fit_transform(Xtr_f, y_bin)   # (Ntrain, n_csp)
            Fs = csp.transform(Xte_f)              # (Ntest, n_csp)

            band_train_feats.append(Ft)
            band_test_feats.append(Fs)


        band_train_feats = np.concatenate(band_train_feats, axis=1)  # (Ntrain, 4*n_csp)
        band_test_feats  = np.concatenate(band_test_feats, axis=1)   # (Ntest,  4*n_csp)

        feat_train_all.append(band_train_feats)
        feat_test_all.append(band_test_feats)


    feat_train_all = np.concatenate(feat_train_all, axis=1)
    feat_test_all  = np.concatenate(feat_test_all, axis=1)

    return feat_train_all, feat_test_all


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True, help="folder containing A01T.gdf ... A09T.gdf")
    args = parser.parse_args()

    #training pipeline
    X_train, y_train = load_subjects(TRAIN_SUBJECTS, args.data_path)
    X_test, y_test = load_subjects(TEST_SUBJECTS, args.data_path)

    print("Train:", X_train.shape, np.bincount(y_train))
    print("Test :", X_test.shape,  np.bincount(y_test))

    # normalize
    X_train = zscore_trials(X_train)
    X_test  = zscore_trials(X_test)

    # OVR-FBCSP features
    X_train_feat, X_test_feat = ovr_fbcsp_features(X_train, y_train, X_test, fs=250, n_csp=4)
    print("Features:", X_train_feat.shape, X_test_feat.shape)

    # classifier
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=5000, class_weight="balanced"))
    ])

    clf.fit(X_train_feat, y_train)

    # evaluation
    y_pred = clf.predict(X_test_feat)
    print("ZERO-SHOT CROSS-SUBJECT RESULTS (OVR-FBCSP + LogisticRegression)")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Kappa   :", cohen_kappa_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    main()
