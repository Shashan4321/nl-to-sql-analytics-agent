.PHONY: setup data test lint eval eval-gold app

setup:
	pip install -r requirements-dev.txt && pre-commit install

data:
	PYTHONPATH=src python -m nl2sql.warehouse

test:
	pytest -q

lint:
	ruff check . && ruff format --check .

eval-gold: data
	PYTHONPATH=src python -m nl2sql.evaluate --gold-only

eval: data
	PYTHONPATH=src python -m nl2sql.evaluate

app: data
	streamlit run app.py
