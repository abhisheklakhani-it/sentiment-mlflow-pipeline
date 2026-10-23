from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
import pytest
from mlflow import MlflowClient

from sentiflow import final
from sentiflow.train import EXPERIMENT


class Constant(mlflow.pyfunc.PythonModel):  # type: ignore[misc]
    def predict(self, context: Any, model_input: Any, params: Any = None) -> pd.DataFrame:
        return pd.DataFrame({"label": ["neutral"] * len(model_input)})


def log(family: str, dev_f1: float, with_model: bool = True) -> str:
    with mlflow.start_run() as run:
        mlflow.set_tag("family", family)
        mlflow.log_metric("dev_macro_f1", dev_f1)
        if with_model:
            mlflow.pyfunc.log_model(name="model", python_model=Constant())
        return run.info.run_id


def test_best_per_family_ranks_by_dev_and_needs_a_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_experiment(EXPERIMENT)
    log("e5+svm", 0.70)
    best_logreg = log("e5+logreg", 0.69)
    log("e5+logreg", 0.65)
    log("gbert-finetune", 0.80, with_model=False)  # no model: must be ignored

    finalists = final.best_per_family(MlflowClient())
    assert [r.data.tags["family"] for r in finalists] == ["e5+svm", "e5+logreg"]
    assert finalists[1].info.run_id == best_logreg
