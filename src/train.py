import argparse
import logging
from hashlib import sha256
from pathlib import Path

import joblib

from src.modeling import apply_threshold_policy, save_artifact, train_from_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare hERG classifiers with scaffold cross-validation, "
            "evaluate the selected model on a held-out scaffold test set, "
            "and select using a training-only objective."
        )
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="data/raw/herg_karim.csv",
        help="CSV file containing molecular SMILES and binary labels.",
    )
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--model-path", default="models/herg_model.joblib")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--threshold-policy", choices=["precision", "f1"], default="precision")
    parser.add_argument("--min-precision", type=float, default=0.9)
    parser.add_argument("--model-selection", choices=["auto", "auprc", "recall_at_precision"], default="auprc")
    parser.add_argument("--expanded-search", action="store_true",
                        help="Compare five candidates, including RF/Extra Trees using 20%% of features per split.")
    parser.add_argument("--baseline-search", action="store_false", dest="expanded_search",
                        help="Compare only the original three classifiers.")
    parser.add_argument("--representation-search", action="store_true",
                        help="Compare RF fingerprints, RBF SVM fingerprints, and RF with descriptors.")
    parser.add_argument("--tuning-search", action="store_true",
                        help="Compare five fixed RF/Extra Trees settings with descriptors.")
    parser.add_argument("--verbose", action="store_true", help="Report CV progress.")
    parser.add_argument("--reselect-threshold", action="store_true",
                        help="Reuse saved training OOF scores without retraining the model.")
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.reselect_threshold:
        artifact = joblib.load(args.model_path)
        if sha256(Path(args.csv_path).read_bytes()).hexdigest() != artifact.get("dataset_sha256"):
            parser.error("Dataset differs from the saved training run. Retrain first.")
        artifact = apply_threshold_policy(artifact, args.threshold_policy, args.min_precision)
        save_artifact(artifact, args.model_path)
    else:
        artifact = train_from_csv(
            csv_path=args.csv_path,
            smiles_column=args.smiles_column,
            label_column=args.label_column,
            model_path=args.model_path,
            test_size=args.test_size,
            cv_folds=args.cv_folds,
            threshold_policy=args.threshold_policy,
            min_precision=args.min_precision,
            model_selection=args.model_selection,
            expanded_search=args.expanded_search,
            representation_search=args.representation_search,
            tuning_search=args.tuning_search,
        )
    print(f"Selected model: {artifact['model_name']}")
    print(f"Model selection: {artifact['selection_metric']}; scaffold folds: {artifact['cv_folds']}")
    for name, policy in artifact.get("candidate_thresholds", {}).items():
        print(f"Training OOF candidate {name}: {policy}")
    for model_name, scores in artifact["cv_metrics"].items():
        auprc = scores["auprc"]
        print(
            f"{model_name} CV AUPRC: "
            f"{auprc['mean']:.3f} +/- {auprc['std']:.3f}"
        )
    print(
        f"Reserved scaffold test set: {artifact['test_size']} compounds "
        f"(training partition: {artifact['train_size']})"
    )
    for metric, value in artifact["test_metrics"].items():
        print(f"Test {metric}: {value:.3f}")
    print("Test confusion counts:")
    for metric, value in artifact["test_confusion"].items():
        print(f"  {metric}: {value}")
    print(f"Saved held-out predictions: {len(artifact['test_predictions'])} rows")
    print(f"Training OOF policy: {artifact['threshold_selection']}")
    print(f"Decision threshold: {artifact['decision_threshold']:.6f}")
    for metric, value in artifact["threshold_test_metrics"].items():
        print(f"Threshold test {metric}: {value:.3f}")
    print(f"Threshold test confusion: {artifact['threshold_test_confusion']}")
    print(f"Saved model: {args.model_path}")


if __name__ == "__main__":
    main()
