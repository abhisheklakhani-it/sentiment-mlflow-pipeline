"""Compare @champion and @challenger on the test split; write the README charts.

    uv run python scripts/compare.py

Loads both aliases from the registry (so it checks what is actually served),
predicts the test split, and writes:
  results/comparison.json   per-class F1, McNemar test, confidence buckets, latency
  docs/per_class_f1.png     per-class test F1 of both models
  docs/confusion_matrices.png
  docs/sweep_history.png, docs/sweep_importances.png   copied from the sweep run
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from scipy.stats import binomtest
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score

from sentiflow.data import LABELS
from sentiflow.final import MODEL_NAME
from sentiflow.train import EXPERIMENT, label_ids, load_splits, setup_mlflow

DOCS = Path("docs")
RESULTS = Path("results/comparison.json")
ALIASES = ("champion", "challenger")
COLORS = {"champion": "#2a78d6", "challenger": "#eb6834"}  # validated pair, light surface
INK, MUTED, SURFACE = "#1f1f1e", "#6b6a64", "#fcfcfb"


def predict(alias: str, texts: list[str]) -> tuple[np.ndarray, np.ndarray | None, float]:
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{alias}")
    start = time.perf_counter()
    out = model.predict(pd.DataFrame({"text": texts}))
    ms_per_text = (time.perf_counter() - start) * 1000 / len(texts)
    conf = out["confidence"].to_numpy() if "confidence" in out else None
    return out["label"].map(LABELS.index).to_numpy(), conf, ms_per_text


def mcnemar(y: np.ndarray, a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Exact McNemar test: only the tweets where exactly one model is right count."""
    only_a = int(((a == y) & (b != y)).sum())
    only_b = int(((a != y) & (b == y)).sum())
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return {"only_champion_right": only_a, "only_challenger_right": only_b, "p_value": float(p)}


def confidence_buckets(y: np.ndarray, pred: np.ndarray, conf: np.ndarray) -> list[dict]:
    rows = []
    for lo, hi in ((0.9, 1.01), (0.7, 0.9), (0.0, 0.7)):
        mask = (conf >= lo) & (conf < hi)
        rows.append(
            {
                "confidence": f"{lo:.1f}-{min(hi, 1.0):.1f}",
                "share_of_tweets": round(float(mask.mean()), 4),
                "accuracy": round(float((pred[mask] == y[mask]).mean()), 4) if mask.any() else None,
            }
        )
    return rows


def plot_per_class(f1s: dict[str, np.ndarray], names: dict[str, str]) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    x, width = np.arange(len(LABELS)), 0.36
    for i, alias in enumerate(ALIASES):
        pos = x + (i - 0.5) * (width + 0.02)  # small surface gap between the paired bars
        bars = ax.bar(pos, f1s[alias], width, color=COLORS[alias], label=names[alias])
        for bar, v in zip(bars, f1s[alias], strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.015,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=INK,
            )
    ax.set_xticks(x, LABELS, color=INK)
    ax.set_ylim(0, 1)
    ax.set_ylabel("test F1", color=MUTED)
    ax.set_title("Per-class F1 on the test split (1,491 tweets)", color=INK, loc="left")
    ax.yaxis.grid(True, color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, length=0)
    ax.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(DOCS / "per_class_f1.png", dpi=150)
    plt.close(fig)


def plot_confusion(y: np.ndarray, preds: dict[str, np.ndarray], names: dict[str, str]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), facecolor=SURFACE)
    for ax, alias in zip(axes, ALIASES, strict=True):
        cm = confusion_matrix(y, preds[alias], labels=range(len(LABELS)))
        ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(names[alias], fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(DOCS / "confusion_matrices.png", dpi=150)
    plt.close(fig)


def copy_sweep_plots(client: MlflowClient) -> None:
    exp = client.get_experiment_by_name(EXPERIMENT)
    runs = client.search_runs(
        [exp.experiment_id], "tags.mlflow.runName LIKE 'optuna-sweep%'", max_results=1
    )
    if not runs:
        return
    folder = Path(client.download_artifacts(runs[0].info.run_id, "optuna"))
    for src, dst in (
        ("optimization_history.png", "sweep_history.png"),
        ("param_importances.png", "sweep_importances.png"),
    ):
        if (folder / src).exists():
            shutil.copy(folder / src, DOCS / dst)


def main() -> None:
    setup_mlflow()
    client = MlflowClient()
    test = load_splits().test
    texts, y = test["text"].tolist(), label_ids(test)
    DOCS.mkdir(exist_ok=True)

    preds, confs, latency, names, f1s = {}, {}, {}, {}, {}
    for alias in ALIASES:
        version = client.get_model_version_by_alias(MODEL_NAME, alias)
        names[alias] = f"@{alias}: {version.tags.get('family', '?')} (v{version.version})"
        preds[alias], confs[alias], latency[alias] = predict(alias, texts)
        f1s[alias] = f1_score(y, preds[alias], average=None, labels=range(len(LABELS)))

    champ, chall = preds["champion"], preds["challenger"]
    report = {
        "models": names,
        "per_class_f1": {
            a: dict(zip(LABELS, np.round(f1s[a], 4).tolist(), strict=True)) for a in ALIASES
        },
        "mcnemar": mcnemar(y, champ, chall),
        "both_wrong": int(((champ != y) & (chall != y)).sum()),
        "ms_per_tweet_batch": {a: round(latency[a], 2) for a in ALIASES},
        "n_test": len(y),
    }
    if confs["champion"] is not None:
        report["champion_confidence_buckets"] = confidence_buckets(y, champ, confs["champion"])
    RESULTS.write_text(json.dumps(report, indent=2) + "\n")

    plot_per_class(f1s, names)
    plot_confusion(y, preds, names)
    copy_sweep_plots(client)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
