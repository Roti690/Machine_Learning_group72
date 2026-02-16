PYTHON := python3
VENV := .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python

.PHONY: venv install run clean

venv:
	$(PYTHON) -m venv $(VENV)

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

run:
	$(PY) scripts/run_mnist_size_study.py

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
