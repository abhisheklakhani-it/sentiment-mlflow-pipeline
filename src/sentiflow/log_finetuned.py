"""Log the Colab gbert fine-tuning run into MLflow (params, per-epoch metrics, model).

python -m sentiflow.log_finetuned --zip gbert_results.zip
"""

from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from sentiflow.evaluate import write_reports
from sentiflow.models import GbertSentimentModel
from sentiflow.train import setup_mlflow


def log_run(run_json: Path, model_dir: Path) -> str:
    result = json.loads(run_json.read_text())
    with mlflow.start_run(run_name="gbert-finetune") as run:
        mlflow.set_tags({"family": "gbert-finetune", "trained_on": result["device"]})
        mlflow.log_params(result["params"])
        for i, h in enumerate(result["history"], start=1):
            mlflow.log_metrics(
                {"epoch_dev_macro_f1": h["dev_macro_f1"], "epoch_dev_loss": h["dev_loss"]}, step=i
            )
        for h in result["train_loss"]:
            mlflow.log_metric("train_loss", h["train_loss"], step=h["step"])
        mlflow.log_metrics({**result["dev_metrics"], "train_minutes": result["train_minutes"]})
        y, pred = np.asarray(result["dev_labels"]), np.asarray(result["dev_predictions"])
        with tempfile.TemporaryDirectory() as tmp:
            for path in write_reports(y, pred, Path(tmp), "dev"):
                mlflow.log_artifact(str(path), "reports")
        mlflow.log_artifact(str(run_json), "colab")
        mlflow.pyfunc.log_model(
            name="model",
            python_model=GbertSentimentModel(),
            artifacts={"model_dir": str(model_dir)},
            input_example=pd.DataFrame({"text": ["Das Spiel war richtig gut!"]}),
            code_paths=["src/sentiflow"],
            pip_requirements=_requirements(),
        )
        return run.info.run_id


def _requirements() -> list[str]:
    from importlib.metadata import version

    return [f"{p}=={version(p)}" for p in ("mlflow", "transformers", "torch", "pandas", "numpy")]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--zip", type=Path, help="gbert_results.zip from the Colab notebook")
    p.add_argument("--run-json", type=Path, default=Path("outputs/gbert_run.json"))
    p.add_argument("--model-dir", type=Path, default=Path("outputs/gbert"))
    args = p.parse_args()
    if args.zip:
        with zipfile.ZipFile(args.zip) as z:
            z.extractall(".")
    setup_mlflow()
    print(json.dumps({"run_id": log_run(args.run_json, args.model_dir)}))


if __name__ == "__main__":
    main()
