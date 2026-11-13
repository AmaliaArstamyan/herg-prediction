from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from rdkit.Chem import Draw

from src.molecules import (
    calculate_descriptors,
    molecular_features,
    parse_smiles,
)
from src.modeling import inhibition_scores

MODEL_PATH = Path(__file__).resolve().parents[1] / "models/herg_model.joblib"
DATASET_PATH = Path(__file__).resolve().parents[1] / "data/raw/herg_karim.csv"


@st.cache_resource
def load_model(path: str, modified_at: float) -> dict:
    return joblib.load(path)


@st.cache_data
def load_compounds(path: str, modified_at: float) -> pd.DataFrame:
    compounds = pd.read_csv(path)
    return compounds.sort_values(
        "smiles", key=lambda column: column.str.lower(), kind="stable"
    ).reset_index(drop=True)


st.set_page_config(page_title="hERG Compound Explorer", layout="wide")
_, language_column = st.columns([12, 2])
with language_column:
    language_option = st.selectbox(
        "Language / Язык",
        ("🇬🇧 English", "🇷🇺 Русский"),
        label_visibility="collapsed",
        help="English / Русский",
        key="language_flag",
    )
language = "English" if language_option.endswith("English") else "Русский"
compounds = (
    load_compounds(str(DATASET_PATH), DATASET_PATH.stat().st_mtime)
    if DATASET_PATH.is_file()
    else None
)

translations = {
    "English": {
        "title": "hERG Inhibition Prediction",
        "intro": (
            "Explore a molecule from its SMILES string and, after training, "
            "get an hERG inhibition prediction."
        ),
        "notice": (
            "This educational model is not a substitute for experimental "
            "validation or medical assessment."
        ),
        "model_ready": "A trained hERG model is ready.",
        "evaluation_title": "Model evaluation (held-out scaffold test set)",
        "selected_model": "Selected model: {model}",
        "cv_summary": "Scaffold CV AUPRC (mean +/- SD), folds: {folds}",
        "cv_model": "Model",
        "cv_auprc_mean": "AUPRC mean",
        "cv_auprc_std": "AUPRC SD",
        "test_summary": "Final test-set metrics",
        "threshold_summary": "Decision threshold: {threshold:.3f} (selected on training scaffold-CV)",
        "precision_policy": "Policy: training CV precision at least {target:.0%}, then maximum recall. The target is not guaranteed on new compounds.",
        "f1_policy": "Policy: maximum training CV F1.",
        "model_policy": "Model selected by maximum training OOF recall at precision ≥ {target:.0%}.",
        "candidate_recall": "OOF recall",
        "candidate_precision": "OOF precision",
        "candidate_threshold": "Threshold",
        "baseline_summary": "Original classifier decision rule (baseline)",
        "tuned_summary": "Test metrics at the selected threshold",
        "test_confusion": "Test-set classifications",
        "metric_labels": {
            "auprc": "AUPRC",
            "roc_auc": "ROC-AUC",
            "balanced_accuracy": "Balanced accuracy",
            "f1": "F1",
            "mcc": "MCC",
            "precision": "Precision",
            "recall": "Recall",
        },
        "confusion_labels": {
            "true_negatives": "True negatives",
            "false_positives": "False positives",
            "false_negatives": "False negatives",
            "true_positives": "True positives",
        },
        "model_missing": (
            "No trained model found. Add a labeled dataset and run the "
            "training command in the README to enable predictions."
        ),
        "dataset_missing": (
            "The local compound list is unavailable. You can still enter "
            "a SMILES string manually."
        ),
        "dataset_count": "{count:,} dataset compounds available.",
        "select_compound": "Compound SMILES",
        "select_placeholder": "Type to search compounds",
        "enter_new": "Enter new",
        "choose_existing": "Choose from dataset",
        "list_title": "All dataset compounds (sorted by SMILES)",
        "table_columns": {
            "compound_id": "Compound ID",
            "smiles": "SMILES",
            "label": "Reference label (1=blocker, 0=non-blocker)",
            "source_split": "Source split",
        },
        "smiles_label": "Compound SMILES",
        "placeholder": "For example: CCO",
        "submit": "Explore molecule",
        "invalid_smiles": "Enter a valid SMILES string.",
        "molecule": "Molecule",
        "properties": "Calculated properties",
        "prediction": "Prediction",
        "inhibitor": "Possible hERG inhibitor",
        "non_inhibitor": "Not predicted as a hERG inhibitor",
        "score": "Model score for inhibition",
        "descriptors": {
            "Molecular weight (Da)": "Molecular weight (Da)",
            "cLogP": "cLogP",
            "TPSA (A^2)": "TPSA (A^2)",
            "Hydrogen-bond donors": "Hydrogen-bond donors",
            "Hydrogen-bond acceptors": "Hydrogen-bond acceptors",
            "Rotatable bonds": "Rotatable bonds",
        },
    },
    "Русский": {
        "title": "Прогноз ингибирования hERG",
        "intro": (
            "Изучите молекулу по ее строке SMILES и получите прогноз "
            "ингибирования hERG после обучения модели."
        ),
        "notice": (
            "Эта учебная модель не заменяет экспериментальную проверку "
            "или медицинскую оценку."
        ),
        "model_ready": "Обученная модель hERG готова.",
        "evaluation_title": "Оценка модели (отложенная scaffold-тестовая выборка)",
        "selected_model": "Выбранная модель: {model}",
        "cv_summary": "Scaffold CV AUPRC (среднее +/- SD), фолды: {folds}",
        "cv_model": "Модель",
        "cv_auprc_mean": "Среднее AUPRC",
        "cv_auprc_std": "SD AUPRC",
        "test_summary": "Метрики итоговой тестовой выборки",
        "threshold_summary": "Порог: {threshold:.3f} (выбран на обучающей scaffold-CV)",
        "precision_policy": "Политика: precision на обучающей CV не ниже {target:.0%}, затем максимум recall. Цель не гарантирована для новых соединений.",
        "f1_policy": "Политика: максимум F1 на обучающей CV.",
        "model_policy": "Модель выбрана по максимуму recall на обучающих OOF при precision ≥ {target:.0%}.",
        "candidate_recall": "OOF recall",
        "candidate_precision": "OOF precision",
        "candidate_threshold": "Порог",
        "baseline_summary": "Исходное правило классификатора (базовый вариант)",
        "tuned_summary": "Тестовые метрики с выбранным порогом",
        "test_confusion": "Классификация тестовой выборки",
        "metric_labels": {
            "auprc": "AUPRC",
            "roc_auc": "ROC-AUC",
            "balanced_accuracy": "Сбалансированная точность",
            "f1": "F1",
            "mcc": "MCC",
            "precision": "Точность (precision)",
            "recall": "Полнота (recall)",
        },
        "confusion_labels": {
            "true_negatives": "Истинно отрицательные",
            "false_positives": "Ложноположительные",
            "false_negatives": "Ложноотрицательные",
            "true_positives": "Истинно положительные",
        },
        "model_missing": (
            "Обученная модель не найдена. Добавьте размеченный набор данных "
            "и запустите команду обучения из README, чтобы включить прогноз."
        ),
        "dataset_missing": (
            "Локальный список соединений недоступен. Вы по-прежнему можете "
            "ввести SMILES вручную."
        ),
        "dataset_count": "Соединений в наборе данных: {count:,}.",
        "select_compound": "SMILES соединения",
        "select_placeholder": "Введите запрос для поиска",
        "enter_new": "Ввести новое",
        "choose_existing": "Выбрать из набора",
        "list_title": "Все соединения набора (сортировка по SMILES)",
        "table_columns": {
            "compound_id": "ID соединения",
            "smiles": "SMILES",
            "label": "Исходная метка (1=блокер, 0=не блокер)",
            "source_split": "Исходная выборка",
        },
        "smiles_label": "SMILES соединения",
        "placeholder": "Например: CCO",
        "submit": "Изучить молекулу",
        "invalid_smiles": "Введите корректную строку SMILES.",
        "molecule": "Молекула",
        "properties": "Расчетные свойства",
        "prediction": "Прогноз",
        "inhibitor": "Возможный ингибитор hERG",
        "non_inhibitor": "Не классифицировано как ингибитор hERG",
        "score": "Оценка модели для ингибирования",
        "descriptors": {
            "Molecular weight (Da)": "Молекулярная масса (Да)",
            "cLogP": "cLogP",
            "TPSA (A^2)": "TPSA (A^2)",
            "Hydrogen-bond donors": "Доноры водородных связей",
            "Hydrogen-bond acceptors": "Акцепторы водородных связей",
            "Rotatable bonds": "Вращаемые связи",
        },
    },
}
text = translations[language]

st.title(text["title"])
st.write(text["intro"])
st.info(text["notice"])
if compounds is None:
    st.warning(text["dataset_missing"])
else:
    st.caption(text["dataset_count"].format(count=len(compounds)))
if MODEL_PATH.is_file():
    st.success(text["model_ready"])
    model_artifact = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
    if "cv_metrics" in model_artifact and "test_metrics" in model_artifact:
        with st.expander(text["evaluation_title"]):
            st.write(
                text["selected_model"].format(
                    model=model_artifact["model_name"]
                )
            )
            st.caption(
                text["cv_summary"].format(folds=model_artifact["cv_folds"])
            )
            cv_rows = [
                {
                    text["cv_model"]: model_name,
                    text["cv_auprc_mean"]: round(
                        scores["auprc"]["mean"], 3
                    ),
                    text["cv_auprc_std"]: round(scores["auprc"]["std"], 3),
                }
                for model_name, scores in model_artifact[
                    "cv_metrics"
                ].items()
            ]
            st.dataframe(
                pd.DataFrame(cv_rows),
                use_container_width=True,
                hide_index=True,
            )
            if model_artifact.get("selection_metric") == "recall_at_precision":
                st.caption(text["model_policy"].format(target=model_artifact.get(
                    "model_selection_min_precision", model_artifact["threshold_selection"]["min_precision"])))
                st.dataframe(pd.DataFrame([
                    {text["cv_model"]: name,
                     text["candidate_precision"]: round(candidate.get("precision", float("nan")), 3),
                     text["candidate_recall"]: round(candidate.get("recall", float("nan")), 3),
                     text["candidate_threshold"]: round(candidate.get("threshold", float("nan")), 3)}
                    for name, candidate in model_artifact["candidate_thresholds"].items()
                ]), use_container_width=True, hide_index=True)
            if "decision_threshold" in model_artifact:
                st.caption(text["threshold_summary"].format(threshold=model_artifact["decision_threshold"]))
                policy = model_artifact.get("threshold_selection", {})
                if policy.get("criterion") == "precision":
                    st.caption(text["precision_policy"].format(target=policy["min_precision"]))
                elif policy.get("criterion") == "f1":
                    st.caption(text["f1_policy"])
                st.write(text["baseline_summary"])
            test_rows = [
                {
                    text["test_summary"]: text["metric_labels"][metric],
                    "Score": round(value, 3),
                }
                for metric, value in model_artifact["test_metrics"].items()
            ]
            st.dataframe(
                pd.DataFrame(test_rows),
                use_container_width=True,
                hide_index=True,
            )
            if "test_confusion" in model_artifact:
                confusion_rows = [
                    {
                        text["test_confusion"]: text["confusion_labels"][key],
                        "Count": value,
                    }
                    for key, value in model_artifact[
                        "test_confusion"
                    ].items()
                ]
                st.dataframe(
                    pd.DataFrame(confusion_rows),
                    use_container_width=True,
                    hide_index=True,
                )
            if "threshold_test_metrics" in model_artifact:
                st.write(text["tuned_summary"])
                st.dataframe(pd.DataFrame([
                    {text["test_summary"]: text["metric_labels"][metric], "Score": round(value, 3)}
                    for metric, value in model_artifact["threshold_test_metrics"].items()
                ]), use_container_width=True, hide_index=True)
                st.dataframe(pd.DataFrame([
                    {text["test_confusion"]: text["confusion_labels"][key], "Count": value}
                    for key, value in model_artifact["threshold_test_confusion"].items()
                ]), use_container_width=True, hide_index=True)
else:
    st.warning(text["model_missing"])

manual_entry = st.session_state.get("manual_smiles_mode", compounds is None)
selected_compound = None
if compounds is not None and not manual_entry:
    compound_options = compounds["smiles"].tolist()
    input_column, action_column = st.columns(
        [5, 1], vertical_alignment="bottom"
    )
    with input_column:
        selected_compound = st.selectbox(
            text["select_compound"],
            options=compound_options,
            index=None,
            placeholder=text["select_placeholder"],
            key="selected_compound",
        )
    with action_column:
        if st.button(
            text["enter_new"], key="enter_new", use_container_width=True
        ):
            st.session_state["manual_smiles_mode"] = True
            st.rerun()
    smiles = selected_compound or ""
elif manual_entry:
    if compounds is not None:
        input_column, action_column = st.columns(
            [5, 1], vertical_alignment="bottom"
        )
        with input_column:
            smiles = st.text_input(
                text["smiles_label"],
                placeholder=text["placeholder"],
                key="manual_smiles",
            )
        with action_column:
            if st.button(
                text["choose_existing"],
                key="choose_existing",
                use_container_width=True,
            ):
                st.session_state["manual_smiles_mode"] = False
                st.rerun()
    else:
        smiles = st.text_input(
            text["smiles_label"],
            placeholder=text["placeholder"],
            key="manual_smiles",
        )

submitted = st.button(text["submit"], key="submit_molecule")

if compounds is not None:
    with st.expander(text["list_title"]):
        st.dataframe(
            compounds.rename(columns=text["table_columns"]),
            use_container_width=True,
            hide_index=True,
        )

if submitted:
    molecule = parse_smiles(smiles)
    if molecule is None:
        st.error(text["invalid_smiles"])
    else:
        st.subheader(text["molecule"])
        st.image(Draw.MolToImage(molecule, size=(400, 300)))
        st.subheader(text["properties"])
        descriptors = calculate_descriptors(smiles)
        translated_descriptors = {
            text["descriptors"][name]: value
            for name, value in descriptors.items()
        }
        st.dataframe(translated_descriptors, use_container_width=True)
        if MODEL_PATH.is_file():
            fingerprint = molecular_features(
                smiles,
                radius=model_artifact["fingerprint_radius"],
                n_bits=model_artifact["fingerprint_bits"],
                include_descriptors=model_artifact.get("include_descriptors", False),
            )
            features = fingerprint.reshape(1, -1)
            score = float(inhibition_scores(model_artifact["model"], features)[0])
            prediction = (
                int(score >= model_artifact["decision_threshold"])
                if "decision_threshold" in model_artifact
                else int(model_artifact["model"].predict(features)[0])
            )
            prediction_text = (
                text["inhibitor"] if prediction == 1 else text["non_inhibitor"]
            )
            st.subheader(text["prediction"])
            st.metric(text["prediction"], prediction_text)
            st.metric(text["score"], f"{score:.1%}")
            if "decision_threshold" in model_artifact:
                st.caption(text["threshold_summary"].format(threshold=model_artifact["decision_threshold"]))
