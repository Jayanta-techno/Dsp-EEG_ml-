#evaluate saved models on the test subjects + causal continuous prediction demo
#usage: python evaluation+prediction_cross_subject.py --data-path path/to/gdf_folder [--trial 0]

import argparse
import os

import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report, confusion_matrix

from data_preprocessing import CLASS_NAMES, TEST_SUBJECTS, load_subjects, zscore_trials
from fbcsp_feature_extraction import BANDS
from streamlit_app.utils.fbcsp import extract_fbcsp_features_batch
from streamlit_app.utils.streaming import fbcsp_stream_predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True, help="folder containing A08T.gdf and A09T.gdf")
    parser.add_argument("--models-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "streamlit_app", "models"))
    parser.add_argument("--trial", type=int, default=0, help="test trial used for the streaming demo")
    parser.add_argument("--plot", default="streaming_prediction.png", help="where to save the streaming plot")
    args = parser.parse_args()

    clf = joblib.load(os.path.join(args.models_dir, "classifier.pkl"))
    csp_models = joblib.load(os.path.join(args.models_dir, "csp_models.pkl"))

    X_test, y_test = load_subjects(TEST_SUBJECTS, args.data_path)
    X_test = zscore_trials(X_test)

    #evaluation
    X_test_feat = extract_fbcsp_features_batch(X_test, csp_models, BANDS)
    y_pred = clf.predict(X_test_feat)

    print("ZERO-SHOT CROSS-SUBJECT RESULTS (FBCSP + LogisticRegression)")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Kappa   :", cohen_kappa_score(y_test, y_pred))

    print("\nTrue counts:", np.bincount(y_test, minlength=4))
    print("Pred counts:", np.bincount(y_pred, minlength=4))

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))

    #causal+continous prediction
    trial_data = X_test[args.trial]
    true_label = y_test[args.trial]

    times, preds, names = fbcsp_stream_predict(
        trial_data, csp_models, clf, BANDS, fs=250, win_sec=1, step_sec=1
    )

    print("\nTRUE:", CLASS_NAMES[true_label])
    print("Continuous predictions (every second):")
    for t, n in zip(times, names):
        print(f"At {t:.1f}s -> {n}")

    #plotting of predicted outputs
    plt.figure(figsize=(8,3))
    plt.step(times, preds, where="post")
    plt.yticks([0,1,2,3], CLASS_NAMES)
    plt.xlabel("Time (sec)")
    plt.ylabel("Predicted class")
    plt.title(f"Causal Continuous Prediction (FBCSP) - true: {CLASS_NAMES[true_label]}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.plot)
    print(f"\nStreaming plot saved to {args.plot}")


if __name__ == "__main__":
    main()
