import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def split_calibration_indices(labels, trials_per_class, random_state=42):
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("Labels must be a one-dimensional array.")
    if trials_per_class < 1:
        raise ValueError("At least one calibration trial per class is required.")

    rng = np.random.default_rng(random_state)
    calibration_indices = []
    prediction_indices = []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        if len(indices) <= trials_per_class:
            raise ValueError(
                f"Class {label} needs more than {trials_per_class} trials "
                "to leave at least one prediction trial."
            )
        shuffled = rng.permutation(indices)
        calibration_indices.extend(shuffled[:trials_per_class])
        prediction_indices.extend(shuffled[trials_per_class:])

    return (
        np.sort(np.asarray(calibration_indices, dtype=int)),
        np.sort(np.asarray(prediction_indices, dtype=int)),
    )


def fit_subject_classifier(calibration_features, calibration_labels):
    calibration_labels = np.asarray(calibration_labels)
    if len(calibration_labels) != len(calibration_features):
        raise ValueError("Calibration features and labels must have equal lengths.")
    if len(np.unique(calibration_labels)) < 2:
        raise ValueError("Calibration data must include at least two classes.")

    classifier = Pipeline([
        ("scaler", StandardScaler()),
        ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
    ])
    classifier.fit(calibration_features, calibration_labels)
    return classifier
