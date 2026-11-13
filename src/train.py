import argparse

from src.modeling import train_from_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare hERG classifiers with scaffold cross-validation, "
            "evaluate the selected model on a held-out scaffold test set, "
            "and select by mean AUPRC."
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
    args = parser.parse_args()

    artifact = train_from_csv(
        csv_path=args.csv_path,
        smiles_column=args.smiles_column,
        label_column=args.label_column,
        model_path=args.model_path,
        test_size=args.test_size,
        cv_folds=args.cv_folds,
    )
    print(f"Selected model: {artifact['model_name']}")
    print(f"Selection: mean CV AUPRC over {artifact['cv_folds']} scaffold folds")
    for model_name, scores in artifact["cv_metrics"].items():
        auprc = scores["auprc"]
        print(
            f"{model_name} CV AUPRC: "
            f"{auprc['mean']:.3f} +/- {auprc['std']:.3f}"
        )
    print(
        f"Untouched scaffold test set: {artifact['test_size']} compounds "
        f"(training partition: {artifact['train_size']})"
    )
    for metric, value in artifact["test_metrics"].items():
        print(f"Test {metric}: {value:.3f}")
    print("Test confusion counts:")
    for metric, value in artifact["test_confusion"].items():
        print(f"  {metric}: {value}")
    print(f"Saved model: {args.model_path}")


if __name__ == "__main__":
    main()
