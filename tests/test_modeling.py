import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, confusion_matrix

from src.molecules import morgan_fingerprint
from src.modeling import (
    _make_estimators, _scaffold_split, apply_threshold_policy,
    select_f1_threshold, select_precision_threshold, train_from_csv,
    save_artifact,
    select_model_at_precision,
    inhibition_scores, _make_representation_estimators,
    _make_tuning_estimators,
)


def test_f1_threshold_handles_tied_scores_and_matches_exhaustive_search():
    labels = np.array([0, 1, 1, 0, 1, 0])
    scores = np.array([0.1, 0.3, 0.3, 0.4, 0.8, 0.9])
    from sklearn.metrics import f1_score
    selected = select_f1_threshold(labels, scores)
    expected = max(f1_score(labels, scores >= threshold) for threshold in np.unique(scores))
    assert np.isclose(selected["f1"], expected)
    assert selected["threshold"] == 0.3
    assert selected["source"] == "training_scaffold_oof"


@pytest.mark.parametrize("labels,scores", [([0, 0], [0.1, 0.2]), ([0, 1], [np.nan, 0.8]),
    ([0, 1], [0.1]), ([0, 1], [-0.1, 1.1])])
def test_f1_threshold_rejects_invalid_inputs(labels, scores):
    with pytest.raises(ValueError):
        select_f1_threshold(labels, scores)


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

    artifact = train_from_csv(dataset_path, model_path=model_path, threshold_policy="f1", expanded_search=False)
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
    results = loaded_artifact["test_predictions"]
    oof = loaded_artifact["oof_predictions"]
    assert oof.row_index.is_unique
    assert set(oof.row_index) == set(artifact["train_indices"])
    assert set(oof.row_index).isdisjoint(results.row_index)
    assert oof.groupby("scaffold").fold.nunique().eq(1).all()
    assert np.isfinite(oof.score).all()
    selected = select_f1_threshold(oof.label, oof.score)
    assert selected == artifact["threshold_selection"]
    np.testing.assert_array_equal(results.threshold_prediction,
                                  results.score >= selected["threshold"])
    assert len(results) == artifact["test_size"]
    assert set(results.row_index).isdisjoint(artifact["train_indices"])
    assert np.isclose(average_precision_score(results.label, results.score),
                      artifact["test_metrics"]["auprc"])
    counts = artifact["test_confusion"]
    assert confusion_matrix(results.label, results.prediction).ravel().tolist() == [
        counts["true_negatives"], counts["false_positives"],
        counts["false_negatives"], counts["true_positives"],
    ]
    # Reconstruct the training-only estimator to catch saving predictions
    # from the all-data demo refit instead of the held-out evaluation.
    all_features = np.vstack([
        morgan_fingerprint("C" * length)
        for length in range(1, compound_count + 1)
    ])
    labels = np.array([length % 2 for length in range(1, compound_count + 1)])
    training_rows = artifact["train_indices"]
    evaluation_model = _make_estimators(42)[artifact["model_name"]]
    evaluation_model.fit(all_features[training_rows], labels[training_rows])
    expected_scores = evaluation_model.predict_proba(all_features[results.row_index])[:, 1]
    np.testing.assert_allclose(results.score, expected_scores)
    # Policy changes may read test data for evaluation, but cannot use it
    # to select the threshold. Original artifacts and scores stay intact.
    target = 0.5
    updated = apply_threshold_policy(loaded_artifact, min_precision=target)
    changed_test = dict(loaded_artifact)
    changed_test["test_predictions"] = loaded_artifact["test_predictions"].copy()
    changed_test["test_predictions"]["label"] = 1 - results.label
    changed_test["test_predictions"]["score"] = 1 - results.score
    alternate = apply_threshold_policy(changed_test, min_precision=target)
    assert alternate["threshold_selection"] == updated["threshold_selection"]
    assert updated["model"] is loaded_artifact["model"]
    assert loaded_artifact["threshold_selection"]["criterion"] == "f1"
    pd.testing.assert_series_equal(updated["test_predictions"].score, results.score)
    fingerprint = morgan_fingerprint("CCO").reshape(1, -1)
    probabilities = loaded_artifact["model"].predict_proba(fingerprint)
    assert probabilities.shape == (1, 2)
    assert np.isclose(probabilities.sum(), 1)


def test_precision_policy_maximizes_recall_within_constraint():
    labels = np.array([0, 1, 0, 1, 1, 1])
    scores = np.array([0.1, 0.4, 0.5, 0.7, 0.8, 0.9])
    selected = select_precision_threshold(labels, scores, min_precision=0.9)
    assert selected["threshold"] == 0.7
    assert selected["precision"] == 1.0
    assert selected["recall"] == 0.75
    # Include a tied positive/negative score as a single decision group.
    tied = select_precision_threshold([0, 1, 1], [0.7, 0.7, 0.9], 0.9)
    assert tied["threshold"] == 0.9


def test_precision_policy_rejects_unreachable_target():
    # Predicting nobody cannot satisfy the constraint via undefined precision.
    with pytest.raises(ValueError, match="target was not relaxed"):
        select_precision_threshold([0, 1], [0.9, 0.1], 0.9)


@pytest.mark.parametrize("target", [0, -0.1, 1.1, np.nan])
def test_precision_policy_rejects_invalid_target(target):
    with pytest.raises(ValueError, match="min_precision"):
        select_precision_threshold([0, 1], [0.1, 0.9], target)


def test_failed_artifact_save_preserves_previous_file(tmp_path, monkeypatch):
    path = tmp_path / "model.joblib"
    save_artifact({"version": "original"}, path)

    def broken_dump(artifact, destination, **kwargs):
        destination.write_bytes(b"partial")
        raise OSError("simulated write failure")

    monkeypatch.setattr(joblib, "dump", broken_dump)
    with pytest.raises(OSError, match="simulated"):
        save_artifact({"version": "new"}, path)
    assert joblib.load(path) == {"version": "original"}
    assert list(tmp_path.iterdir()) == [path]


def test_model_selection_prioritizes_recall_at_fixed_precision_over_ap():
    labels = np.array([1, 1, 1, 0, 0, 0])
    scores = {
        "Higher AP": np.array([0.9, 0.3, 0.2, 0.8, 0.7, 0.6]),
        "Higher recall": np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4]),
        "Infeasible": np.array([0.1, 0.2, 0.3, 0.9, 0.8, 0.7]),
    }
    metrics = {name: {"auprc": {"mean": ap}} for name, ap in
               [("Higher AP", 0.99), ("Higher recall", 0.8), ("Infeasible", 0.7)]}
    selected, candidates = select_model_at_precision(metrics, scores, labels, 0.9)
    assert selected == "Higher recall"
    assert candidates[selected]["precision"] >= 0.9
    assert candidates[selected]["recall"] == 1.0
    assert candidates["Infeasible"]["feasible"] is False


def test_model_selection_fails_when_all_candidates_miss_precision_target():
    with pytest.raises(ValueError, match="No candidate meets"):
        select_model_at_precision({"Only": {"auprc": {"mean": 0.5}}},
                                  {"Only": np.array([0.9, 0.1])}, np.array([0, 1]), 0.9)


def test_training_saves_policy_aware_selection_and_all_oof_scores(tmp_path, monkeypatch):
    from sklearn.dummy import DummyClassifier
    import src.modeling as modeling

    monkeypatch.setattr(modeling, "_make_estimators", lambda random_state, expanded=False: {
        "Prior A": DummyClassifier(strategy="prior"),
        "Prior B": DummyClassifier(strategy="prior"),
    })
    path = tmp_path / "training.csv"
    pd.DataFrame({"smiles": ["C" * length for length in range(1, 31)],
                  "label": [length % 2 for length in range(1, 31)]}).to_csv(path, index=False)
    artifact = train_from_csv(path, model_path=tmp_path / "model.joblib", min_precision=0.1,
                              model_selection="recall_at_precision")
    assert artifact["selection_metric"] == "recall_at_precision"
    assert artifact["model_selection_min_precision"] == 0.1
    assert artifact["model_name"] == "Prior B"
    assert set(artifact["candidate_oof_scores"]) == {"Prior A", "Prior B"}
    assert artifact["threshold_selection"]["precision"] >= 0.1
    assert artifact["threshold_selection"]["recall"] == 1.0


def test_representation_models_ignore_or_use_descriptors_as_declared():
    from src.molecules import molecular_features
    from scipy.special import expit
    smiles = ["CCO", "CCN", "CCCl", "CCBr", "c1ccccc1", "Cc1ccccc1"]
    features = np.vstack([molecular_features(s, include_descriptors=True) for s in smiles])
    labels = np.array([0, 0, 0, 1, 1, 1])
    models = _make_representation_estimators(42)
    for name, model in models.items():
        model.fit(features, labels)
        scores = inhibition_scores(model, features)
        assert scores.shape == (6,)
        assert np.isfinite(scores).all()
        if "(fingerprints)" in name:
            changed = features.copy()
            changed[:, 2048:] += 1000
            np.testing.assert_allclose(scores, inhibition_scores(model, changed))
        if "SVM" in name:
            np.testing.assert_allclose(scores, expit(model.decision_function(features)))


def test_representation_training_round_trip_uses_matching_inference_features(tmp_path, monkeypatch):
    import src.modeling as modeling
    from src.molecules import molecular_features

    factory = modeling._make_representation_estimators

    def small_candidates(random_state):
        models = factory(random_state)
        for model in models.values():
            params = model.get_params()
            if "model__n_estimators" in params:
                model.set_params(model__n_estimators=5)
            elif "n_estimators" in params:
                model.set_params(n_estimators=5)
        return models

    monkeypatch.setattr(modeling, "_make_representation_estimators", small_candidates)
    path = tmp_path / "training.csv"
    pd.DataFrame({"smiles": ["C" * length for length in range(1, 31)],
                  "label": [length % 2 for length in range(1, 31)]}).to_csv(path, index=False)
    output = tmp_path / "model.joblib"
    artifact = train_from_csv(path, model_path=output, representation_search=True,
                              model_selection="recall_at_precision", min_precision=0.1)
    loaded = joblib.load(output)
    assert loaded["include_descriptors"] is True
    assert loaded["representation_search"] is True
    assert len(loaded["candidate_thresholds"]) == 3
    assert loaded["model"].n_features_in_ == 2054
    features = molecular_features("CCO", include_descriptors=loaded["include_descriptors"]).reshape(1, -1)
    np.testing.assert_allclose(inhibition_scores(loaded["model"], features),
                               inhibition_scores(artifact["model"], features))


def test_tuning_candidates_include_unchanged_control_and_fixed_variations():
    models = _make_tuning_estimators(42)
    assert len(models) == 5
    control = models["Random forest (fingerprints + descriptors)"]
    previous = _make_representation_estimators(42)["Random forest (fingerprints + descriptors)"]
    assert control.get_params() == previous.get_params()
    assert models["RF descriptors (600 trees)"].n_estimators == 600
    assert models["RF descriptors (leaf size 2)"].min_samples_leaf == 2
    assert models["RF descriptors (5% features)"].max_features == 0.05


def test_tuning_search_cannot_be_combined_with_other_search_modes(tmp_path):
    with pytest.raises(ValueError, match="only one search mode"):
        train_from_csv(tmp_path / "unused.csv", tuning_search=True, representation_search=True)


def test_tuning_training_persists_descriptor_schema(tmp_path, monkeypatch):
    import src.modeling as modeling
    from sklearn.dummy import DummyClassifier
    monkeypatch.setattr(modeling, "_make_tuning_estimators", lambda random_state: {
        "Tuning control": DummyClassifier(strategy="prior")})
    path = tmp_path / "training.csv"
    pd.DataFrame({"smiles": ["C" * length for length in range(1, 31)],
                  "label": [length % 2 for length in range(1, 31)]}).to_csv(path, index=False)
    artifact = train_from_csv(path, model_path=tmp_path / "model.joblib", tuning_search=True,
                              model_selection="recall_at_precision", min_precision=0.1)
    assert artifact["tuning_search"] is True
    assert artifact["include_descriptors"] is True
    assert artifact["representation_search"] is False
    assert artifact["threshold_selection"]["precision"] >= 0.1
