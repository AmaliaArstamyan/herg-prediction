from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from streamlit.testing.v1 import AppTest


DATASET_PATH = Path("data/raw/herg_karim.csv")


@pytest.mark.parametrize("include_descriptors", [False, True])
def test_app_uses_saved_threshold_instead_of_classifier_default(tmp_path, include_descriptors):
    # A 0.6 score is class 1 by default, but class 0 at the saved 0.7 threshold.
    model = RandomForestClassifier(n_estimators=1, bootstrap=False, random_state=42).fit(
        np.zeros((5, 2054 if include_descriptors else 2048)), [0, 0, 1, 1, 1]
    )
    model_path = tmp_path / "model.joblib"
    joblib.dump({"model": model, "positive_label": 1,
                 "fingerprint_radius": 2, "fingerprint_bits": 2048,
                 "decision_threshold": 0.7,
                 "include_descriptors": include_descriptors}, model_path)
    source = Path("src/app.py").read_text().replace(
        'MODEL_PATH = Path(__file__).resolve().parents[1] / "models/herg_model.joblib"',
        f"MODEL_PATH = Path({str(model_path)!r})",
    ).replace(
        'DATASET_PATH = Path(__file__).resolve().parents[1] / "data/raw/herg_karim.csv"',
        f"DATASET_PATH = Path({str(tmp_path / 'missing.csv')!r})",
    )
    app = AppTest.from_string(source).run()
    app.text_input[0].set_value("CCO")
    next(button for button in app.button if button.label == "Explore molecule").click().run()
    assert not app.exception
    assert any(metric.value == "Not predicted as a hERG inhibitor" for metric in app.metric)
    assert any("0.700" in caption.value for caption in app.caption)


def test_app_can_switch_to_russian():
    app = AppTest.from_file("src/app.py").run()

    app.selectbox[0].select("🇷🇺 Русский").run()

    assert any(title.value == "Прогноз ингибирования hERG" for title in app.title)
    if app.text_input:
        assert app.text_input[0].label == "SMILES соединения"
    else:
        assert any(box.label == "SMILES соединения" for box in app.selectbox)


def test_enter_new_switches_compound_dropdown_to_smiles_text():
    app = AppTest.from_file("src/app.py").run()

    enter_new_button = next(
        (button for button in app.button if button.label == "Enter new"), None
    )
    if enter_new_button is not None:
        enter_new_button.click().run()

    assert app.text_input[0].label == "Compound SMILES"


@pytest.mark.skipif(not DATASET_PATH.is_file(), reason="Dataset is not downloaded")
def test_sorted_dropdown_contains_dataset_and_can_submit_compound():
    app = AppTest.from_file("src/app.py").run()
    selector = next(
        box for box in app.selectbox if box.label == "Compound SMILES"
    )
    first_smiles = selector.options[0]

    assert selector.options == sorted(selector.options, key=str.lower)
    selector.select(first_smiles).run()
    next(
        button for button in app.button if button.label == "Explore molecule"
    ).click().run()

    assert not app.exception
    assert any(len(table.value) == 13445 for table in app.dataframe)
    assert any(item.value == "Molecule" for item in app.subheader)
