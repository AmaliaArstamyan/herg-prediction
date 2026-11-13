from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


DATASET_PATH = Path("data/raw/herg_karim.csv")


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
