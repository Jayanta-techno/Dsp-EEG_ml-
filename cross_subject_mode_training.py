#zero-shot cross-subject training: train on A01-A07, test on A08-A09
#usage: python cross_subject_mode_training.py --data-path path/to/gdf_folder

import argparse
import json
import os

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report, confusion_matrix

from data_preprocessing import CLASS_NAMES, TRAIN_SUBJECTS, TEST_SUBJECTS, load_subjects, zscore_trials
from fbcsp_feature_extraction import BANDS, fbcsp_features_cross_subject


N_CSP = 6


def make_classifier(k=15):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(mutual_info_classif, k=k)),
        ("lr", LogisticRegression(max_iter=3000, class_weight="balanced"))
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True, help="folder containing A01T.gdf ... A09T.gdf")
    parser.add_argument("--models-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "streamlit_app", "models"))
    args = parser.parse_args()

    X_train, y_train = load_subjects(TRAIN_SUBJECTS, args.data_path)
    X_test, y_test = load_subjects(TEST_SUBJECTS, args.data_path)

    print("Train shape:", X_train.shape, "Labels:", np.bincount(y_train))
    print("Test  shape:", X_test.shape,  "Labels:", np.bincount(y_test))

    X_train = zscore_trials(X_train)
    X_test  = zscore_trials(X_test)

    X_train_feat, X_test_feat, csp_models = fbcsp_features_cross_subject(X_train, y_train, X_test, fs=250, n_csp=N_CSP)
    print("Feature shapes:", X_train_feat.shape, X_test_feat.shape)

    clf = make_classifier()
    clf.fit(X_train_feat, y_train)

    #evaluation on unseen subjects
    y_pred = clf.predict(X_test_feat)
    acc = accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)

    print("ZERO-SHOT CROSS-SUBJECT RESULTS (FBCSP + LogisticRegression)")
    print("Accuracy:", acc)
    print("Kappa   :", kappa)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))
    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))

    #accuracy of each band on its own (frequency relevance)
    band_acc = []
    for b in range(len(BANDS)):
        cols = slice(b * N_CSP, (b + 1) * N_CSP)
        band_clf = make_classifier(k="all").fit(X_train_feat[:, cols], y_train)
        band_acc.append(float(accuracy_score(y_test, band_clf.predict(X_test_feat[:, cols]))))
    for (l, h), a in zip(BANDS, band_acc):
        print(f"Band {l}-{h} Hz accuracy: {a:.3f}")

    # =========================================
    # SAVE TRAINED MODELS FOR STREAMLIT
    # =========================================
    os.makedirs(args.models_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(args.models_dir, "classifier.pkl"))
    joblib.dump(csp_models, os.path.join(args.models_dir, "csp_models.pkl"))

    metrics = {
        "accuracy": float(acc),
        "kappa": float(kappa),
        "y_true": y_test.tolist(),
        "y_pred": y_pred.tolist(),
        "bands": BANDS,
        "band_accuracy": band_acc,
        "train_subjects": TRAIN_SUBJECTS,
        "test_subjects": TEST_SUBJECTS,
    }
    with open(os.path.join(args.models_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"✅ Classifier, CSP models and metrics saved to {args.models_dir}")


if __name__ == "__main__":
    main()
