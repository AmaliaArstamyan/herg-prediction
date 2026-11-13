
# hERG Inhibition Prediction

## Overview

This project develops a machine learning pipeline to predict
whether a chemical compound inhibits the hERG channel.

The goal is to explore molecular data, train classification
models, evaluate their performance, and build a web application
for demonstrating predictions.

## Project Goals

- Perform exploratory data analysis (EDA).
- Prepare molecular features using chemical representations.
- Train and compare classification models.
- Evaluate model performance and prediction errors.
- Test data processing and model behavior.
- Deploy a web application using Docker.

## Technology Stack

- Python 3.10-compatible environment for the local build
- Jupyter Notebook
- pandas, NumPy, Matplotlib, scikit-learn
- RDKit
- Streamlit
- Git and GitHub
- Docker

## Local Setup

This repository currently uses a Python 3.10-compatible dependency set for the
local environment. If you are working in a 3.12 environment, update the pinned
versions in `requirements.txt` as needed.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
```

## Run the Web Demo

```bash
make app
```

The app validates a SMILES string and displays molecular properties. The
Karim hERG dataset is available at `data/raw/herg_karim.csv`; it has `smiles`
and binary `label` columns, where `1` means hERG inhibitor and `0` means
non-inhibitor. Training reserves a scaffold-based test set, selects among
baseline classifiers using scaffold-grouped cross-validation on the remaining
data, and evaluates only the selected model on the reserved test set:

Choose a compound from the searchable, SMILES-sorted dropdown, or click
**Enter new** to type a SMILES string manually. The expandable table lists all
dataset compounds.

```bash
make train
```

For different column names or a different CSV path:

```bash
python -m src.train path/to/dataset.csv \
  --smiles-column SMILES --label-column activity
```

Training saves the selected model to `models/herg_model.joblib`; the app uses it
for predictions when present. The current demo also displays molecular
properties. The model score is an estimate, not experimental evidence.
The interface supports English and Russian; select the language at the top.
Models are selected by mean cross-validation AUPRC, the metric recommended for
this dataset. The app displays the cross-validation AUPRC mean and standard
deviation, plus the selected model's metrics and confusion counts on the
untouched scaffold test set. ROC-AUC, balanced accuracy, F1, and MCC are also
reported. After evaluation, the selected model is refit on all rows for the
interactive demo. These are educational estimates, not independent clinical
validation.

Use `--cv-folds` to change the maximum number of scaffold-grouped
cross-validation folds (default: 5). Training reduces this number if needed
to ensure each fold contains both label classes:

```bash
PYTHONPATH=. .venv/bin/python -m src.train --cv-folds 3
```

## Training Data

The [Karim et al. hERG dataset](https://huggingface.co/datasets/jablonkagroup/herg_karim_et_al)
contains 13,445 compounds. The CSV includes `compound_id`, `smiles`, binary
`label`, and the source-provided `source_split`. That source split is retained
for reference; the model pipeline creates its own scaffold split.

The dataset is provided under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Please attribute it to the Karim et al. dataset and cite:

> Karim, A., Lee, M., Balle, T., & Sattar, A. (2021). CardioTox net: a robust
> predictor for hERG channel blockade based on deep learning meta-feature
> ensembles. *Journal of Cheminformatics, 13*, 60.
> https://doi.org/10.1186/s13321-021-00541-z

## Project Status

The Streamlit demo and scaffold cross-validation training pipeline are in
place. The training dataset is downloaded; run `make train` to refresh the
model and its displayed evaluation.

## Disclaimer

This project is for educational and research purposes.
Predictions are not a substitute for experimental validation
or professional medical assessment.
