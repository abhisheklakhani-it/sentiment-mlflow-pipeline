"""Train one embedding-based model and log everything to MLflow.

    python -m sentiflow.train --encoder e5-base --family logreg --C 1.0 --class-weight balanced

Fits on train, scores on dev (model selection). The test split is only used by
`sentiflow.final`, once, for the chosen model.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd

from sentiflow.data import LABELS, REVISION, load_split, remove_overlap
from sentiflow.embed import ENCODERS, cached_embeddings
from sentiflow.evaluate import metrics, write_reports
from sentiflow.models import EmbeddingSentimentModel, build_classifier

EXPERIMENT = "sb10k-sentiment"


@dataclass
class Splits:
    train: pd.DataFrame
    dev: pd.DataFrame
    test: pd.DataFrame


def load_splits() -> Splits:
    dev, test = load_split("dev"), load_split("test")
    train = remove_overlap(load_split("train"), dev, test)
    return Splits(train, dev, test)


def label_ids(df: pd.DataFrame) -> np.ndarray:
    return df["label"].map(LABELS.index).to_numpy()


def setup_mlflow() -> None:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(EXPERIMENT)


def train_and_log(
    encoder: str,
    family: str,
    params: dict[str, Any],
    splits: Splits,
    *,
    log_model: bool = True,
    nested: bool = False,
    tags: dict[str, str] | None = None,
) -> tuple[str, dict[str, float]]:
    """One MLflow run. Returns (run_id, dev metrics)."""
    x_train = cached_embeddings(encoder, "train", splits.train["text"].tolist())
    x_dev = cached_embeddings(encoder, "dev", splits.dev["text"].tolist())
    y_train, y_dev = label_ids(splits.train), label_ids(splits.dev)

    with mlflow.start_run(nested=nested) as run:
        mlflow.set_tags({"family": f"e5+{family}", **(tags or {})})
        mlflow.log_params(
            {
                "encoder": encoder,
                "encoder_repo": ENCODERS[encoder][0],
                "encoder_revision": ENCODERS[encoder][1],
                "family": family,
                "dataset_revision": REVISION,
                "n_train": len(y_train),
                **params,
            }
        )
        clf = build_classifier(family, params).fit(x_train, y_train)
        pred = clf.predict(x_dev)
        dev_metrics = metrics(y_dev, pred, "dev")
        mlflow.log_metrics(dev_metrics)

        if log_model:
            with tempfile.TemporaryDirectory() as tmp:
                for path in write_reports(y_dev, pred, Path(tmp), "dev"):
                    mlflow.log_artifact(str(path), "reports")
            mlflow.pyfunc.log_model(
                name="model",
                python_model=EmbeddingSentimentModel(encoder, clf),
                input_example=pd.DataFrame({"text": ["Das Spiel war richtig gut!"]}),
                code_paths=["src/sentiflow"],
                pip_requirements=_requirements(),
            )
        return run.info.run_id, dev_metrics


def _requirements() -> list[str]:
    from importlib.metadata import version

    pkgs = ["mlflow", "scikit-learn", "sentence-transformers", "torch", "pandas", "numpy"]
    return [f"{p}=={version(p)}" for p in pkgs]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--encoder", choices=sorted(ENCODERS), default="e5-base")
    p.add_argument("--family", choices=["logreg", "svm"], default="logreg")
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--class-weight", choices=["none", "balanced"], default="balanced")
    args = p.parse_args()
    setup_mlflow()
    params = {"C": args.C, "class_weight": args.class_weight}
    run_id, dev = train_and_log(args.encoder, args.family, params, load_splits())
    print(json.dumps({"run_id": run_id, **dev}))


if __name__ == "__main__":
    main()
