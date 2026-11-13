from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from src.molecules import morgan_fingerprint, scaffold_key

FINGERPRINT_RADIUS = 2
FINGERPRINT_BITS = 2048
CV_FOLDS = 5


def _make_estimators(random_state: int) -> dict:
    return {
        "Logistic regression": LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=random_state
        ),
        "Random forest": RandomForestClassifier(
            class_weight="balanced_subsample",
            n_estimators=300,
            n_jobs=-1,
            random_state=random_state,
        ),
        "Extra trees": ExtraTreesClassifier(
            class_weight="balanced",
            n_estimators=300,
            n_jobs=-1,
            random_state=random_state,
        ),
    }


def _classification_metrics(
    labels: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray
) -> dict[str, float]:
    return {
        "auprc": float(average_precision_score(labels, probabilities)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
    }


def _scaffold_split(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if len(np.unique(groups)) < 2:
        raise ValueError("At least two distinct molecular scaffolds are required.")

    splitter = GroupShuffleSplit(
        n_splits=100, test_size=test_size, random_state=random_state
    )
    best_split = None
    best_split_score = float("inf")

    for train_indices, test_indices in splitter.split(features, labels, groups):
        if len(np.unique(labels[train_indices])) < 2:
            continue
        if len(np.unique(labels[test_indices])) < 2:
            continue
        size_error = abs(len(test_indices) / len(labels) - test_size)
        class_balance_error = abs(
            float(np.mean(labels[test_indices])) - float(np.mean(labels))
        )
        split_score = size_error + class_balance_error
        if split_score < best_split_score:
            best_split = (train_indices, test_indices)
            best_split_score = split_score

    if best_split is None:
        raise ValueError(
            "Could not create a scaffold split containing both labels in "
            "the training and test sets. Check the dataset size and class "
            "distribution."
        )
    return best_split


def train_from_csv(
    csv_path: str | Path,
    smiles_column: str = "smiles",
    label_column: str = "label",
    model_path: str | Path = "models/herg_model.joblib",
    test_size: float = 0.2,
    random_state: int = 42,
    cv_folds: int = CV_FOLDS,
) -> dict:
    dataset = pd.read_csv(csv_path)
    missing_columns = {
        column
        for column in (smiles_column, label_column)
        if column not in dataset.columns
    }
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {', '.join(sorted(missing_columns))}."
        )
    if dataset.empty:
        raise ValueError("Dataset contains no rows.")
    if dataset[smiles_column].isna().any():
        raise ValueError("Dataset contains missing SMILES values.")

    labels = pd.to_numeric(dataset[label_column], errors="raise")
    if labels.isna().any() or not set(labels.unique()).issubset({0, 1}):
        raise ValueError(
            f"'{label_column}' must contain only binary values 0 and 1 "
            "(1 means hERG inhibitor)."
        )
    labels_array = labels.to_numpy(dtype=np.int8)
    if len(np.unique(labels_array)) != 2:
        raise ValueError("Dataset must contain both label classes, 0 and 1.")

    smiles_values = dataset[smiles_column].astype(str).tolist()
    fingerprint_rows = []
    for row_number, smiles in enumerate(smiles_values, start=2):
        try:
            fingerprint_rows.append(
                morgan_fingerprint(
                    smiles, radius=FINGERPRINT_RADIUS, n_bits=FINGERPRINT_BITS
                )
            )
        except ValueError as error:
            raise ValueError(
                f"Invalid SMILES at CSV row {row_number}: {smiles!r}."
            ) from error
    features = np.vstack(fingerprint_rows)
    groups = np.asarray([scaffold_key(smiles) for smiles in smiles_values])
    train_indices, test_indices = _scaffold_split(
        features, labels_array, groups, test_size, random_state
    )
    if cv_folds < 2:
        raise ValueError("cv_folds must be at least 2.")
    training_groups = groups[train_indices]
    max_folds = min(cv_folds, len(np.unique(training_groups)))
    if max_folds < 2:
        raise ValueError(
            "At least two distinct scaffolds are required in the training "
            "partition for cross-validation."
        )

    fold_indices = None
    for candidate_folds in range(max_folds, 1, -1):
        splitter = StratifiedGroupKFold(
            n_splits=candidate_folds, shuffle=True, random_state=random_state
        )
        candidate_indices = list(
            splitter.split(
                features[train_indices],
                labels_array[train_indices],
                training_groups,
            )
        )
        if all(
            len(np.unique(labels_array[train_indices[validation]])) == 2
            and len(np.unique(labels_array[train_indices[fold_train]])) == 2
            for fold_train, validation in candidate_indices
        ):
            cv_folds = candidate_folds
            fold_indices = candidate_indices
            break
    if fold_indices is None:
        raise ValueError(
            "Could not create scaffold cross-validation folds containing "
            "both label classes. Check scaffold diversity and class "
            "distribution."
        )

    fold_scores = {name: [] for name in _make_estimators(random_state)}
    for fold_train, fold_validation in fold_indices:
        fold_train_labels = labels_array[train_indices[fold_train]]
        fold_validation_labels = labels_array[train_indices[fold_validation]]
        for name, estimator in _make_estimators(random_state).items():
            estimator.fit(
                features[train_indices[fold_train]], fold_train_labels
            )
            predictions = estimator.predict(
                features[train_indices[fold_validation]]
            )
            positive_index = list(estimator.classes_).index(1)
            probabilities = estimator.predict_proba(
                features[train_indices[fold_validation]]
            )[:, positive_index]
            fold_scores[name].append(
                _classification_metrics(
                    fold_validation_labels, predictions, probabilities
                )
            )

    cv_metrics = {
        name: {
            metric: {
                "mean": float(
                    np.mean([scores[metric] for scores in scores_by_fold])
                ),
                "std": float(
                    np.std([scores[metric] for scores in scores_by_fold])
                ),
            }
            for metric in scores_by_fold[0]
        }
        for name, scores_by_fold in fold_scores.items()
    }
    best_model_name = max(
        cv_metrics,
        key=lambda name: (cv_metrics[name]["auprc"]["mean"], name),
    )

    selected_model = _make_estimators(random_state)[best_model_name]
    selected_model.fit(features[train_indices], labels_array[train_indices])
    test_predictions = selected_model.predict(features[test_indices])
    positive_index = list(selected_model.classes_).index(1)
    test_probabilities = selected_model.predict_proba(features[test_indices])[
        :, positive_index
    ]
    test_metrics = _classification_metrics(
        labels_array[test_indices], test_predictions, test_probabilities
    )
    true_negatives = int(
        np.sum((labels_array[test_indices] == 0) & (test_predictions == 0))
    )
    false_positives = int(
        np.sum((labels_array[test_indices] == 0) & (test_predictions == 1))
    )
    false_negatives = int(
        np.sum((labels_array[test_indices] == 1) & (test_predictions == 0))
    )
    true_positives = int(
        np.sum((labels_array[test_indices] == 1) & (test_predictions == 1))
    )

    selected_model.fit(features, labels_array)
    artifact = {
        "model": selected_model,
        "model_name": best_model_name,
        "fingerprint_radius": FINGERPRINT_RADIUS,
        "fingerprint_bits": FINGERPRINT_BITS,
        "positive_label": 1,
        "selection_metric": "auprc",
        "cv_folds": cv_folds,
        "cv_metrics": cv_metrics,
        "test_metrics": test_metrics,
        "test_confusion": {
            "true_negatives": true_negatives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "true_positives": true_positives,
        },
        "test_size": len(test_indices),
        "train_size": len(train_indices),
    }

    output_path = Path(model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path, compress=3)
    return artifact
