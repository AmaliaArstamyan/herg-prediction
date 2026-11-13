# hERG Inhibition Prediction

A research and educational pipeline for predicting hERG inhibition from SMILES,
with a Streamlit explorer and executed Jupyter notebooks in Armenian and English.

The current local demo uses **Random Forest with 600 trees**, Morgan fingerprints
(radius 2, 2048 bits), and six molecular descriptors. Model settings and the
classification threshold are selected using training scaffold cross-validation:
**maximize recall subject to at least 90% empirical OOF precision**.

On the reserved 2,680-compound scaffold test set, the selected rule achieved
**91.22% precision, 65.05% recall, 84 false positives, and 469 false negatives**.
The threshold is **0.642925**. This test set has been inspected across successive
experiments; these comparisons are descriptive, not new independent validation.

## Setup and run

The pinned dependencies target the project's Python 3.10 environment. The stack
includes pandas, NumPy, SciPy, Matplotlib, scikit-learn, RDKit, Streamlit, Jupyter,
and pytest. Docker packaging is not implemented yet.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
make test
```

The raw CSV and trained artifacts are local files ignored by Git. Before training,
obtain the dataset described below and place it at `data/raw/herg_karim.csv`.

```bash
make train
make app
```

`make train` runs the five-candidate descriptor-tuning search with progress output,
selects the model and threshold on training OOF scores, evaluates the selected
model on test, then refits it on all rows and saves `models/herg_model.joblib`.
The app loads that file when available. Without it, molecular exploration still
works; without the CSV, manual SMILES entry remains available.

The English/Russian interface supports a searchable compound dropdown, manual
SMILES entry, molecular drawings, calculated properties, predictions, CV model
comparisons, and test metrics at both the classifier's original decision rule
and the selected threshold. The displayed score is a model estimate.

## Training data and quality

The [Karim et al. hERG dataset](https://huggingface.co/datasets/jablonkagroup/herg_karim_et_al)
contains 13,445 local rows with these columns:

| Column | Meaning |
|---|---|
| `compound_id` | Source compound identifier |
| `smiles` | Molecular structure |
| `label` | 1 = inhibitor; 0 = non-inhibitor |
| `source_split` | Original source partition, retained for reference |

EDA found no missing values, invalid/empty molecular structures, repeated raw or
canonical SMILES, or conflicting canonical-structure labels. Labels are nearly
balanced: **6,727 non-inhibitors and 6,718 inhibitors**. Canonicalization is not
full salt, protonation, or tautomer standardization.

Our pipeline creates its own scaffold partitions rather than using `source_split`.
Murcko scaffold keys keep the same scaffold in one partition. For acyclic
molecules, the key uses the complete canonical structure. Different keys can
still represent chemically similar molecules.

The dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Attribute it to Karim et al. and cite:

> Karim, A., Lee, M., Balle, T., & Sattar, A. (2021). CardioTox net: a robust
> predictor for hERG channel blockade based on deep learning meta-feature
> ensembles. *Journal of Cheminformatics, 13*, 60.
> https://doi.org/10.1186/s13321-021-00541-z

## Notebooks

All **14 notebooks** include saved outputs, charts, and explanations. Open one in
VS Code, select `.venv/bin/python` as the kernel, and use **Run All** or
**Shift + Enter**. Saved results can be viewed without rerunning training.

```bash
.venv/bin/python -m jupyter notebook notebooks/01_data_exploration.ipynb
```

| Analysis | Armenian | English |
|---|---|---|
| Data quality, class balance, descriptors, correlations, scaffold drawings | [01](notebooks/01_data_exploration.ipynb) | [01 EN](notebooks/01_data_exploration_en.ipynb) |
| Confusion matrices, PR/ROC, score distributions, error examples | [02](notebooks/02_model_error_analysis.ipynb) | [02 EN](notebooks/02_model_error_analysis_en.ipynb) |
| Maximum-F1 threshold selection | [03](notebooks/03_threshold_selection.ipynb) | [03 EN](notebooks/03_threshold_selection_en.ipynb) |
| Precision target versus the original and F1 rules | [04](notebooks/04_precision_policy.ipynb) | [04 EN](notebooks/04_precision_policy_en.ipynb) |
| Five fingerprint-only model candidates | [05](notebooks/05_model_improvement.ipynb) | [05 EN](notebooks/05_model_improvement_en.ipynb) |
| Model family versus molecular representation | [06](notebooks/06_model_vs_representation.ipynb) | [06 EN](notebooks/06_model_vs_representation_en.ipynb) |
| Descriptor-model tuning and changed molecular predictions | [07](notebooks/07_descriptor_tuning.ipynb) | [07 EN](notebooks/07_descriptor_tuning_en.ipynb) |

Notebook 01 describes the full dataset; it is not a basis for selecting model
settings against test. Notebooks 02–04 prefer the saved original precision-model
snapshot when present, so later app updates preserve their historical comparisons.
Notebooks 05–07 require the historical artifacts listed below to rerun.

## Evaluation and decision policy

Training reserves a scaffold test partition and constructs up to five
`StratifiedGroupKFold` folds on the remaining rows. The number of folds is reduced
if needed so both classes occur in each training and validation fold. Each
training row receives one **out-of-fold (OOF)** score from a model that did not
train on that row or its scaffold.

For the current recipe, each candidate gets the threshold maximizing pooled OOF
recall subject to precision >= 90%. Model selection then maximizes that recall;
model ties prefer mean CV AP, then model name. Threshold ties prefer higher
precision, then the higher threshold. Infeasible targets raise an error instead
of being silently relaxed. Predictions use `score >= decision_threshold`.

Only the selected model is evaluated on test, using the already fixed threshold.
Original-rule and selected-threshold results are retained separately. Afterward,
the model is refitted on all 13,445 rows for the demo.

AUPRC in this project means scikit-learn **Average Precision (AP)**, not
trapezoidal integration. AP and ROC-AUC depend on scores; changing only the
threshold does not change them. Precision, recall, F1, and confusion counts can
change. Maximizing F1 does not necessarily minimize total mistakes.

### Experiments completed

The following are **test** results; model/threshold choices used training OOF data.

| Model and rule | Precision | Recall | FP | FN |
|---|---:|---:|---:|---:|
| Original fingerprint RF, classifier rule | 84.36% | 76.75% | 191 | 312 |
| Original RF, maximum-OOF-F1 threshold | 77.99% | 83.68% | 317 | 219 |
| Original RF, precision-target threshold | 91.42% | 57.97% | 73 | 564 |
| Expanded fingerprint search: Extra Trees | 91.52% | 57.90% | 72 | 565 |
| RF + descriptors, 300 trees | 90.59% | 64.53% | 90 | 476 |
| **RF + descriptors, 600 trees — current** | **91.22%** | **65.05%** | **84** | **469** |

The expanded fingerprint search added RF/Extra Trees variants considering 20% of
features per split. It did not reduce missed inhibitors and was not activated.

The representation experiment held settings and splits fixed: OOF recall at the
90% precision target was **53.42%** for fingerprint RF, **49.48%** for RBF SVM,
and **58.71%** for descriptor RF. Adding descriptors helped at these settings;
this does not establish that all SVMs perform worse. SVM used C=1, gamma=scale,
and balanced class weights. Its score is a sigmoid-transformed decision margin,
not a calibrated probability; fingerprint-only pipelines discard the descriptors.

The final tuning search compared descriptor RF with 300/600 trees, minimum leaf
size 2, 5% of features per split, and descriptor Extra Trees with 600 trees.
The 600-tree RF won with **59.10% OOF recall and 90.00% OOF precision**. Relative
to descriptor RF with 300 trees, test FN decreased by 7 and FP decreased by 6:
a small observed improvement, not a statistically established gain.

For the current selected-threshold test result: AP **0.9027**, ROC-AUC **0.8976**,
balanced accuracy **0.7939**, F1 **0.7595**, MCC **0.6133**; TN **1,254**, TP **873**.

## Reproduce experiments and artifacts

`make train` is equivalent to:

```bash
.venv/bin/python -m src.train --tuning-search --model-selection recall_at_precision --verbose
```

The low-level CLI retains its fingerprint-only, mean-CV-AP defaults for
compatibility. Use `--model-selection auto` to select constrained recall for the
precision policy, or AP for the F1 policy. Search modes are mutually exclusive.
Use `--cv-folds`, `--test-size`, `--smiles-column`, and `--label-column` as needed;
changing splits or data changes the reported experiment.

To recreate all snapshots required by notebooks 02–07, run in order:

```bash
# Original fingerprint-only precision model: notebooks 02–06.
.venv/bin/python -m src.train --baseline-search --model-selection auprc \
  --model-path models/herg_precision_baseline.joblib

# Earlier five-model fingerprint experiment: notebook 05.
.venv/bin/python -m src.train --expanded-search --model-selection recall_at_precision \
  --model-path models/herg_candidate.joblib

# RF/SVM/descriptor comparison: notebook 06, and the control for notebook 07.
.venv/bin/python -m src.train --representation-search --model-selection recall_at_precision \
  --model-path models/herg_representation_candidate.joblib
cp models/herg_representation_candidate.joblib models/herg_descriptor_baseline.joblib

# Final tuning experiment: notebook 07, then activate for the demo.
.venv/bin/python -m src.train --tuning-search --model-selection recall_at_precision \
  --model-path models/herg_tuned_candidate.joblib --verbose
cp models/herg_tuned_candidate.joblib models/herg_model.joblib
```

To reselect only the current model's threshold from saved OOF scores:

```bash
.venv/bin/python -m src.train --reselect-threshold --threshold-policy precision --min-precision 0.9
# Alternative objective, without rerunning model selection:
.venv/bin/python -m src.train --reselect-threshold --threshold-policy f1
```

F1 threshold ties prefer the value closest to 0.5, then the higher threshold.
Reselecting a threshold does not rerun model selection across candidates.

Artifacts store candidate OOF scores, selected OOF rows/folds/scaffolds, training
indices, seed, dataset SHA-256, feature/scoring metadata, selected threshold, and
test scores captured **before** the all-data refit. Original-rule results remain
in `test_metrics`, `test_confusion`, and `test_predictions.prediction`; the active
rule uses `threshold_test_metrics`, `threshold_test_confusion`, and
`test_predictions.threshold_prediction`. Artifact writes replace the old file
only after a complete write, so readers cannot load a partially written model.

Training and inference share `molecular_features` and `inhibition_scores`.
Descriptor models use 2,054 columns: 2,048 fingerprint bits followed by molecular
weight, cLogP, TPSA, hydrogen-bond donors, hydrogen-bond acceptors, and rotatable
bonds. Older fingerprint-only artifacts remain supported.

## Validation and limitations

`make test`: **35 passed**. Tests cover molecular features, partition separation,
OOF-only policy selection, infeasible targets, model-selection objectives,
artifact persistence and failed writes, inference with both feature schemas,
and Streamlit interaction/language switching. All 14 notebooks were executed
and checked for cell errors; dependency deprecation warnings remain.

The same CV results select models and thresholds, so the OOF optimum is not an
independent performance estimate. Test results have been repeatedly inspected;
confirmation requires new independent data. The empirical 90% precision target
is not a guarantee on new compounds. Full-data refitting can shift scores, and
predictions for dataset compounds in the demo are not held-out evaluations.

Higher precision does not make dataset labels more accurate or prove safety.
This project is for educational and research purposes; predictions do not
replace experimental validation or professional medical assessment. External
validation and Docker packaging remain future work.
