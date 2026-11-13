from pathlib import Path
from hashlib import sha256
from tempfile import NamedTemporaryFile
import logging

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.molecules import molecular_features, scaffold_key

FINGERPRINT_RADIUS = 2
FINGERPRINT_BITS = 2048
CV_FOLDS = 5
logger = logging.getLogger(__name__)


def inhibition_scores(estimator, features) -> np.ndarray:
    """Return class-1 scores; sigmoid margins are rankings, not calibrated risks."""
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(features)[:, list(estimator.classes_).index(1)]
    if list(estimator.classes_) != [0, 1]:
        raise ValueError("Decision-function scoring requires binary classes [0, 1].")
    return expit(estimator.decision_function(features))


def _make_representation_estimators(random_state):
    """A fixed three-way experiment separating model and input changes."""
    def fingerprints_only():
        return ColumnTransformer([("fingerprints", "passthrough", slice(0, FINGERPRINT_BITS))])

    def forest():
        return RandomForestClassifier(class_weight="balanced_subsample", n_estimators=300,
                                      n_jobs=-1, random_state=random_state)

    return {
        "Random forest (fingerprints)": Pipeline([
            ("features", fingerprints_only()), ("model", forest())]),
        "RBF SVM (fingerprints)": Pipeline([
            ("features", fingerprints_only()),
            ("model", SVC(C=1.0, kernel="rbf", gamma="scale", class_weight="balanced",
                          probability=False, cache_size=512, random_state=random_state))]),
        "Random forest (fingerprints + descriptors)": forest(),
    }


def _make_tuning_estimators(random_state):
    """Prespecified bounded search around the current descriptor RF."""
    def forest(**kwargs):
        return RandomForestClassifier(class_weight="balanced_subsample", n_jobs=-1,
                                      random_state=random_state, **kwargs)

    return {
        "Random forest (fingerprints + descriptors)": forest(n_estimators=300),
        "RF descriptors (600 trees)": forest(n_estimators=600),
        "RF descriptors (leaf size 2)": forest(n_estimators=300, min_samples_leaf=2),
        "RF descriptors (5% features)": forest(n_estimators=300, max_features=0.05),
        "Extra trees descriptors (600 trees)": ExtraTreesClassifier(
            n_estimators=600, class_weight="balanced", n_jobs=-1, random_state=random_state),
    }


def save_artifact(artifact: dict, path: str | Path) -> None:
    """Replace an artifact only after its full contents are written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".joblib", delete=False) as temp:
        temporary = Path(temp.name)
    try:
        joblib.dump(artifact, temporary, compress=3)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _threshold_curve(labels: np.ndarray, scores: np.ndarray):
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    if (labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores)
            or set(np.unique(labels)) != {0, 1}
            or not np.isfinite(scores).all() or not ((scores >= 0) & (scores <= 1)).all()):
        raise ValueError("Expected binary labels containing both classes and finite scores in [0, 1].")
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    return precision[:-1], recall[:-1], thresholds


def select_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> dict:
    """Maximize training OOF F1; ties prefer closest to 0.5, then higher."""
    precision, recall, thresholds = _threshold_curve(labels, scores)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(precision), where=(precision + recall) > 0)
    index = max(range(len(thresholds)),
                key=lambda i: (f1[i], -abs(thresholds[i] - 0.5), thresholds[i]))
    return {
        "threshold": float(thresholds[index]),
        "criterion": "f1",
        "source": "training_scaffold_oof",
        "precision": float(precision[index]),
        "recall": float(recall[index]),
        "f1": float(f1[index]),
    }


def select_precision_threshold(
    labels: np.ndarray, scores: np.ndarray, min_precision: float = 0.9
) -> dict:
    """Maximize training OOF recall subject to an empirical precision target.

    Positive means score >= threshold. Ties prefer higher precision, then
    higher threshold. The target is not a future-performance guarantee.
    """
    if not np.isfinite(min_precision) or not 0 < min_precision <= 1:
        raise ValueError("min_precision must be in (0, 1].")
    precision, recall, thresholds = _threshold_curve(labels, scores)
    feasible = np.flatnonzero(precision >= min_precision)
    if not len(feasible):
        raise ValueError(
            f"No training OOF threshold reaches precision {min_precision:.1%}. "
            "The target was not relaxed."
        )
    index = max(feasible, key=lambda i: (recall[i], precision[i], thresholds[i]))
    p, r = float(precision[index]), float(recall[index])
    return {
        "threshold": float(thresholds[index]),
        "criterion": "precision",
        "min_precision": float(min_precision),
        "source": "training_scaffold_oof",
        "precision": p,
        "recall": r,
        "f1": 2 * p * r / (p + r) if p + r else 0.0,
    }


def select_threshold(labels, scores, policy="precision", min_precision=0.9) -> dict:
    if policy == "precision":
        return select_precision_threshold(labels, scores, min_precision)
    if policy == "f1":
        return select_f1_threshold(labels, scores)
    raise ValueError("threshold_policy must be 'precision' or 'f1'.")


def apply_threshold_policy(artifact: dict, policy="precision", min_precision=0.9) -> dict:
    """Reselect from saved training OOF scores; evaluate only after selection.

    Does not refit or modify the model, baseline results, or supplied artifact.
    """
    if "oof_predictions" not in artifact or "test_predictions" not in artifact:
        raise ValueError("Saved OOF/test scores are missing. Run make train first.")
    oof = artifact["oof_predictions"]
    if (not oof.row_index.is_unique
            or set(oof.row_index) != set(artifact["train_indices"])
            or not set(oof.row_index).isdisjoint(artifact["test_predictions"].row_index)
            or not oof.groupby("scaffold").fold.nunique().eq(1).all()
            or not set(oof.scaffold).isdisjoint(artifact["test_predictions"].scaffold)):
        raise ValueError("Invalid training OOF/test partition metadata.")
    selection = select_threshold(oof.label, oof.score, policy, min_precision)
    test = artifact["test_predictions"].copy()
    prediction = (test.score.to_numpy() >= selection["threshold"]).astype(np.int8)
    test["threshold_prediction"] = prediction
    metrics = _classification_metrics(test.label.to_numpy(), prediction, test.score.to_numpy())
    metrics.update({
        "precision": float(precision_score(test.label, prediction, zero_division=0)),
        "recall": float(recall_score(test.label, prediction, zero_division=0)),
    })
    result = dict(artifact)
    result.update({
        "decision_threshold": selection["threshold"],
        "threshold_selection": selection,
        "test_predictions": test,
        "threshold_test_metrics": metrics,
        "threshold_test_confusion": dict(zip(
            ["true_negatives", "false_positives", "false_negatives", "true_positives"],
            map(int, confusion_matrix(test.label, prediction, labels=[0, 1]).ravel()),
        )),
    })
    return result


def select_model_at_precision(cv_metrics, oof_scores, labels, min_precision=0.9):
    """Select using training data only: recall at the fixed precision target.

    Ties prefer mean CV AP, then model name. Infeasible candidates are recorded
    and excluded; the precision target is never relaxed.
    """
    candidates = {}
    for name, scores in oof_scores.items():
        try:
            candidates[name] = select_precision_threshold(labels, scores, min_precision)
        except ValueError as error:
            if "No training OOF threshold" not in str(error):
                raise
            candidates[name] = {"feasible": False, "reason": str(error)}
    feasible = [name for name, result in candidates.items() if "threshold" in result]
    if not feasible:
        raise ValueError("No candidate meets the training OOF precision target.")
    selected = max(feasible, key=lambda name: (
        candidates[name]["recall"], cv_metrics[name]["auprc"]["mean"], name
    ))
    return selected, candidates


def _make_estimators(random_state: int, expanded: bool = False) -> dict:
    estimators = {
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
    if expanded:
        estimators.update({
            "Random forest (20% features)": RandomForestClassifier(
                class_weight="balanced_subsample", max_features=0.2,
                n_estimators=300, n_jobs=-1, random_state=random_state,
            ),
            "Extra trees (20% features)": ExtraTreesClassifier(
                class_weight="balanced", max_features=0.2,
                n_estimators=300, n_jobs=-1, random_state=random_state,
            ),
        })
    return estimators


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
    threshold_policy: str = "precision",
    min_precision: float = 0.9,
    model_selection: str = "auprc",
    expanded_search: bool = False,
    representation_search: bool = False,
    tuning_search: bool = False,
) -> dict:
    if sum([expanded_search, representation_search, tuning_search]) > 1:
        raise ValueError("Choose only one search mode: expanded, representation, or tuning.")
    include_descriptors = representation_search or tuning_search
    if threshold_policy not in {"precision", "f1"}:
        raise ValueError("threshold_policy must be 'precision' or 'f1'.")
    if not np.isfinite(min_precision) or not 0 < min_precision <= 1:
        raise ValueError("min_precision must be in (0, 1].")
    if model_selection == "auto":
        model_selection = "recall_at_precision" if threshold_policy == "precision" else "auprc"
    if model_selection not in {"auprc", "recall_at_precision"}:
        raise ValueError("model_selection must be auto, auprc, or recall_at_precision.")
    if model_selection == "recall_at_precision" and threshold_policy != "precision":
        raise ValueError("recall_at_precision selection requires the precision threshold policy.")
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
                molecular_features(
                    smiles, radius=FINGERPRINT_RADIUS, n_bits=FINGERPRINT_BITS,
                    include_descriptors=include_descriptors,
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

    def make_candidates():
        if tuning_search:
            return _make_tuning_estimators(random_state)
        if representation_search:
            return _make_representation_estimators(random_state)
        return _make_estimators(random_state, expanded_search)

    fold_scores = {name: [] for name in make_candidates()}
    oof_scores = {name: np.full(len(train_indices), np.nan) for name in fold_scores}
    oof_folds = np.full(len(train_indices), -1, dtype=int)
    for fold_number, (fold_train, fold_validation) in enumerate(fold_indices):
        oof_folds[fold_validation] = fold_number
        fold_train_labels = labels_array[train_indices[fold_train]]
        fold_validation_labels = labels_array[train_indices[fold_validation]]
        for name, estimator in make_candidates().items():
            logger.info("Scaffold CV fold %s/%s: %s", fold_number + 1, cv_folds, name)
            estimator.fit(
                features[train_indices[fold_train]], fold_train_labels
            )
            predictions = estimator.predict(
                features[train_indices[fold_validation]]
            )
            probabilities = inhibition_scores(estimator,
                features[train_indices[fold_validation]]
            )
            oof_scores[name][fold_validation] = probabilities
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
    candidate_thresholds = {}
    if model_selection == "recall_at_precision":
        best_model_name, candidate_thresholds = select_model_at_precision(
            cv_metrics, oof_scores, labels_array[train_indices], min_precision
        )
    else:
        best_model_name = max(
            cv_metrics,
            key=lambda name: (cv_metrics[name]["auprc"]["mean"], name),
        )
    threshold_selection = select_threshold(
        labels_array[train_indices], oof_scores[best_model_name],
        threshold_policy, min_precision,
    )
    decision_threshold = threshold_selection["threshold"]
    logger.info("Selected from training OOF: %s, threshold %.6f", best_model_name, decision_threshold)
    oof_results = pd.DataFrame({
        "row_index": train_indices,
        "label": labels_array[train_indices],
        "score": oof_scores[best_model_name],
        "fold": oof_folds,
        "scaffold": training_groups,
    })

    selected_model = make_candidates()[best_model_name]
    selected_model.fit(features[train_indices], labels_array[train_indices])
    test_predictions = selected_model.predict(features[test_indices])
    test_probabilities = inhibition_scores(selected_model, features[test_indices])
    test_metrics = _classification_metrics(
        labels_array[test_indices], test_predictions, test_probabilities
    )
    threshold_predictions = (test_probabilities >= decision_threshold).astype(np.int8)
    threshold_test_metrics = _classification_metrics(
        labels_array[test_indices], threshold_predictions, test_probabilities
    )
    threshold_test_metrics.update({
        "precision": float(precision_score(labels_array[test_indices], threshold_predictions, zero_division=0)),
        "recall": float(recall_score(labels_array[test_indices], threshold_predictions, zero_division=0)),
    })
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

    # Snapshot held-out predictions before refitting the demo model on all rows.
    test_results = pd.DataFrame({
        "row_index": test_indices,
        "smiles": dataset.iloc[test_indices][smiles_column].to_numpy(),
        "label": labels_array[test_indices],
        "prediction": test_predictions.copy(),
        "score": test_probabilities.copy(),
        "scaffold": groups[test_indices],
        "threshold_prediction": threshold_predictions,
    })
    if "compound_id" in dataset.columns:
        test_results["compound_id"] = dataset.iloc[test_indices]["compound_id"].to_numpy()

    selected_model.fit(features, labels_array)
    artifact = {
        "model": selected_model,
        "model_name": best_model_name,
        "fingerprint_radius": FINGERPRINT_RADIUS,
        "fingerprint_bits": FINGERPRINT_BITS,
        "positive_label": 1,
        "selection_metric": model_selection,
        "model_selection_min_precision": min_precision if model_selection == "recall_at_precision" else None,
        "candidate_thresholds": candidate_thresholds,
        "candidate_oof_scores": oof_scores,
        "expanded_search": expanded_search,
        "representation_search": representation_search,
        "tuning_search": tuning_search,
        "include_descriptors": include_descriptors,
        "score_method": "probability" if hasattr(selected_model, "predict_proba") else "sigmoid_margin",
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
        "test_predictions": test_results,
        "train_indices": train_indices.copy(),
        "random_state": random_state,
        "dataset_sha256": sha256(Path(csv_path).read_bytes()).hexdigest(),
        "decision_threshold": decision_threshold,
        "threshold_selection": threshold_selection,
        "oof_predictions": oof_results,
        "threshold_test_metrics": threshold_test_metrics,
        "threshold_test_confusion": dict(zip(
            ["true_negatives", "false_positives", "false_negatives", "true_positives"],
            map(int, confusion_matrix(labels_array[test_indices], threshold_predictions, labels=[0, 1]).ravel()),
        )),
    }

    output_path = Path(model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_artifact(artifact, output_path)
    return artifact
