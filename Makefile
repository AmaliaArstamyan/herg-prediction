PYTHON ?= python3
VENV ?= .venv
PYTHON_VENV := $(VENV)/bin/python
PIP_VENV := $(VENV)/bin/pip

.DEFAULT_GOAL := help

help:
	@echo "Targets: install, test, app, train"

install:
	@$(PYTHON) -m venv $(VENV)
	@$(PIP_VENV) install --upgrade pip
	@$(PIP_VENV) install -r requirements.txt

test:
	@PYTHONPATH=. $(PYTHON_VENV) -m pytest -q

app:
	@PYTHONPATH=. $(VENV)/bin/streamlit run src/app.py

train:
	@PYTHONPATH=. $(PYTHON_VENV) -m src.train
