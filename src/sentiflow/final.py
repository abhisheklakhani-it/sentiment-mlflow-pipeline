"""Pick the champion on dev, score the finalists on test once, register them.

    python -m sentiflow.final

1. Best run per family by dev macro-F1 (runs that logged a model).
2. Champion = best dev macro-F1. Decided BEFORE any test score is seen.
3. Each finalist is scored on the test split; metrics + reports go to its run.
4. Finalists become versions of the registered model `sb10k-sentiment`;
   aliases @champion and @challenger point at the top two (by dev).
Writes results/final.json for the README.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient

from sentiflow.data import LABELS
from sentiflow.evaluate import metrics, write_reports
from sentiflow.train import EXPERIMENT, label_ids, load_splits, setup_mlflow

MODEL_NAME = "sb10k-sentiment"
RESULTS = Path("results/final.json")


def best_per_family(client: MlflowClient) -> list[Any]:
    exp = client.get_experiment_by_name(EXPERIMENT)
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string="metrics.dev_macro_f1 > 0 and attributes.status = 'FINISHED'",
        order_by=["metrics.dev_macro_f1 DESC"],
        max_results=500,
    )
    best: dict[str, Any] = {}
    for run in runs:
        family = run.data.tags.get("family")
        if family and family not in best and _has_model(client, exp.experiment_id, run):
            best[family] = run
    return sorted(best.values(), key=lambda r: r.data.metrics["dev_macro_f1"], reverse=True)


def _has_model(client: MlflowClient, experiment_id: str, run: Any) -> bool:
    found = client.search_logged_models(
        experiment_ids=[experiment_id], filter_string=f"source_run_id = '{run.info.run_id}'"
    )
    return len(found) > 0


def score_on_test(run: Any, texts: list[str], y: np.ndarray) -> dict[str, float]:
    model = mlflow.pyfunc.load_model(f"runs:/{run.info.run_id}/model")
    pred = model.predict(pd.DataFrame({"text": texts}))["label"].map(LABELS.index).to_numpy()
    test_metrics = metrics(y, pred, "test")
    with mlflow.start_run(run_id=run.info.run_id):
        mlflow.log_metrics(test_metrics)
        with tempfile.TemporaryDirectory() as tmp:
            for path in write_reports(y, pred, Path(tmp), "test"):
                mlflow.log_artifact(str(path), "reports")
    return test_metrics


def main() -> None:
    setup_mlflow()
    client = MlflowClient()
    finalists = best_per_family(client)
    if not finalists:
        raise SystemExit("no finished runs with a model; run sentiflow.train or sentiflow.sweep")

    test = load_splits().test
    texts, y = test["text"].tolist(), label_ids(test)
    rows = []
    for rank, run in enumerate(finalists):
        test_metrics = score_on_test(run, texts, y)
        version = mlflow.register_model(f"runs:/{run.info.run_id}/model", MODEL_NAME)
        family = run.data.tags["family"]
        dev_f1 = run.data.metrics["dev_macro_f1"]
        for key, value in {
            "family": family,
            "dev_macro_f1": f"{dev_f1:.4f}",
            "test_macro_f1": f"{test_metrics['test_macro_f1']:.4f}",
        }.items():
            client.set_model_version_tag(MODEL_NAME, version.version, key, value)
        if rank < 2:
            client.set_registered_model_alias(
                MODEL_NAME, "champion" if rank == 0 else "challenger", version.version
            )
        rows.append(
            {
                "rank_by_dev": rank + 1,
                "family": family,
                "run_id": run.info.run_id,
                "version": int(version.version),
                "alias": ["champion", "challenger"][rank] if rank < 2 else None,
                "params": {
                    k: v
                    for k, v in run.data.params.items()
                    if k
                    in ("encoder", "C", "class_weight", "base_model", "epochs", "learning_rate")
                },
                "dev_macro_f1": dev_f1,
                **test_metrics,
            }
        )
    client.update_registered_model(
        MODEL_NAME, "German tweet sentiment (SB10k): negative / neutral / positive. Text in."
    )
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(rows, indent=2) + "\n")
    for r in rows:
        print(
            f"v{r['version']:<3} {r['alias'] or '':<11} {r['family']:<15} "
            f"dev {r['dev_macro_f1']:.4f}  test {r['test_macro_f1']:.4f}  "
            f"(acc {r['test_accuracy']:.4f}, pos/neg F1 {r['test_f1_pos_neg']:.4f})"
        )


if __name__ == "__main__":
    main()
