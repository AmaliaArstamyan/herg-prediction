import joblib
import numpy as np
import pandas as pd

from src.molecules import morgan_fingerprint
from src.modeling import _scaffold_split, train_from_csv


def test_scaffold_split_keeps_scaffolds_in_one_partition():
    features = np.zeros((12, 4))
    labels = np.tile([0, 1], 6)
    groups = np.asarray([f"scaffold-{index}" for index in range(12)])

    train_indices, test_indices = _scaffold_split(
        features, labels, groups, test_size=0.25, random_state=42
    )

    assert set(groups[train_indices]).isdisjoint(groups[test_indices])


def test_training_uses_cv_and_untouched_test_set_then_saves_predictor(tmp_path):
    dataset_path = tmp_path / "training.csv"
    model_path = tmp_path / "model.joblib"
    compound_count = 30
    pd.DataFrame(
        {
            "smiles": ["C" * length for length in range(1, compound_count + 1)],
            "label": [length % 2 for length in range(1, compound_count + 1)],
        }
    ).to_csv(dataset_path, index=False)

    artifact = train_from_csv(dataset_path, model_path=model_path)
    loaded_artifact = joblib.load(model_path)

    assert set(artifact["cv_metrics"]) == {
        "Logistic regression",
        "Random forest",
        "Extra trees",
    }
    assert all(
        "auprc" in scores and "mean" in scores["auprc"]
        for scores in artifact["cv_metrics"].values()
    )
    assert set(artifact["test_metrics"]) == {
        "auprc",
        "roc_auc",
        "balanced_accuracy",
        "f1",
        "mcc",
    }
    assert loaded_artifact["model_name"] == artifact["model_name"]
    assert artifact["train_size"] + artifact["test_size"] == compound_count
    assert sum(artifact["test_confusion"].values()) == artifact["test_size"]
    assert 2 <= artifact["cv_folds"] <= 5
    assert artifact["selection_metric"] == "auprc"
    fingerprint = morgan_fingerprint("CCO").reshape(1, -1)
    probabilities = loaded_artifact["model"].predict_proba(fingerprint)
    assert probabilities.shape == (1, 2)
    assert np.isclose(probabilities.sum(), 1)
