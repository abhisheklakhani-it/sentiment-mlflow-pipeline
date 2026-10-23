"""Optuna hyperparameter sweep; every trial is a nested MLflow run.

    python -m sentiflow.sweep --trials 60

Trials are scored by 5-fold cross-validated macro-F1 on train, not on dev, so
the 60 trials cannot overfit the dev set. The best configuration is then
retrained on all of train, scored on dev and logged with its model.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold, cross_val_score

from sentiflow.embed import ENCODERS, cached_embeddings
from sentiflow.models import FAMILIES, SEED, build_classifier
from sentiflow.train import Splits, label_ids, load_splits, setup_mlflow, train_and_log

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def suggest(trial: optuna.Trial) -> tuple[str, str, dict[str, Any]]:
    encoder = trial.suggest_categorical("encoder", sorted(ENCODERS))
    family = trial.suggest_categorical("family", list(FAMILIES))
    params = {
        "C": trial.suggest_float("C", 1e-3, 1e2, log=True),
        "class_weight": trial.suggest_categorical("class_weight", ["none", "balanced"]),
    }
    return encoder, family, params


def make_objective(splits: Splits) -> Any:
    y = label_ids(splits.train)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    x_by_encoder = {
        name: cached_embeddings(name, "train", splits.train["text"].tolist()) for name in ENCODERS
    }

    def objective(trial: optuna.Trial) -> float:
        encoder, family, params = suggest(trial)
        with mlflow.start_run(run_name=f"trial-{trial.number:03d}", nested=True):
            mlflow.set_tag("family", f"e5+{family}")
            mlflow.log_params({"encoder": encoder, "family": family, **params})
            scores = cross_val_score(
                build_classifier(family, params),
                x_by_encoder[encoder],
                y,
                cv=folds,
                scoring="f1_macro",
                n_jobs=1,
            )
            mlflow.log_metrics(
                {"cv_macro_f1": float(scores.mean()), "cv_macro_f1_std": float(scores.std())}
            )
        return float(scores.mean())

    return objective


def log_optuna_plots(study: optuna.Study) -> None:
    from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances

    with tempfile.TemporaryDirectory() as tmp:
        for name, fn in (
            ("optimization_history", plot_optimization_history),
            ("param_importances", plot_param_importances),
        ):
            ax = fn(study)
            fig = np.atleast_1d(ax)[0].figure
            fig.set_size_inches(8, 4.5)
            fig.tight_layout()
            path = Path(tmp) / f"{name}.png"
            fig.savefig(path, dpi=120)
            plt.close(fig)
            mlflow.log_artifact(str(path), "optuna")


def run_sweep(trials: int) -> dict[str, Any]:
    setup_mlflow()
    splits = load_splits()
    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name="sb10k")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    with mlflow.start_run(run_name=f"optuna-sweep-{trials}") as parent:
        mlflow.set_tag("kind", "sweep")
        mlflow.log_params({"trials": trials, "sampler": "TPE", "seed": SEED, "cv_folds": 5})
        study.optimize(make_objective(splits), n_trials=trials)
        best = study.best_trial
        mlflow.log_metric("best_cv_macro_f1", best.value)
        mlflow.log_dict(best.params, "best_params.json")
        log_optuna_plots(study)

        p = best.params
        params = {"C": p["C"], "class_weight": p["class_weight"]}
        run_id, dev = train_and_log(
            p["encoder"],
            p["family"],
            params,
            splits,
            nested=True,
            tags={"kind": "sweep-best", "sweep_run_id": parent.info.run_id},
        )
        mlflow.set_tag("best_run_id", run_id)
    return {"best_run_id": run_id, "best_cv_macro_f1": round(best.value, 4), **p, **dev}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--trials", type=int, default=60)
    args = p.parse_args()
    print(json.dumps(run_sweep(args.trials)))


if __name__ == "__main__":
    main()
