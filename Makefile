export MLFLOW_TRACKING_URI ?= http://localhost:5050
export MLFLOW_DISABLE_AGENT_HINT = 1

.PHONY: install lint test mlflow-up mlflow-down baselines sweep notebook log-gbert final serve all

install:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run pytest

mlflow-up:  ## MLflow server + PostgreSQL on http://localhost:5050
	docker compose up -d --build --wait

mlflow-down:
	docker compose down

baselines:  ## one hand-picked run per encoder
	uv run python -m sentiflow.train --encoder e5-small --family logreg
	uv run python -m sentiflow.train --encoder e5-base --family logreg

sweep:  ## 60 Optuna trials, best config retrained and logged
	uv run python -m sentiflow.sweep --trials 60

notebook:  ## regenerate the Colab notebook from src/
	uv run python scripts/make_colab_notebook.py

log-gbert:  ## after Colab: put gbert_results.zip here
	uv run python -m sentiflow.log_finetuned --zip gbert_results.zip

final:  ## champion on dev, test once, register + aliases
	uv run python -m sentiflow.final

serve:  ## REST API for the current @champion on http://localhost:5051
	uv run mlflow models serve -m "models:/sb10k-sentiment@champion" --env-manager local --port 5051

all: mlflow-up baselines sweep log-gbert final
