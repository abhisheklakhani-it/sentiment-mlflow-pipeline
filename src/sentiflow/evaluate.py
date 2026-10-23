"""Metrics and report artifacts shared by all model families."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from sentiflow.data import LABELS


def metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> dict[str, float]:
    per_class = f1_score(y_true, y_pred, average=None, labels=range(len(LABELS)))
    out = {
        f"{prefix}_macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        f"{prefix}_accuracy": float(accuracy_score(y_true, y_pred)),
        # SB10k papers often report the mean F1 of positive and negative only
        f"{prefix}_f1_pos_neg": float((per_class[0] + per_class[2]) / 2),
    }
    for name, value in zip(LABELS, per_class, strict=True):
        out[f"{prefix}_f1_{name}"] = float(value)
    return {k: round(v, 4) for k, v in out.items()}


def write_reports(y_true: np.ndarray, y_pred: np.ndarray, out_dir: Path, prefix: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / f"{prefix}_classification_report.txt"
    report.write_text(
        classification_report(y_true, y_pred, target_names=LABELS, digits=4, zero_division=0)
    )
    cm = confusion_matrix(y_true, y_pred, labels=range(len(LABELS)))
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{prefix} confusion matrix")
    fig.tight_layout()
    png = out_dir / f"{prefix}_confusion_matrix.png"
    fig.savefig(png, dpi=120)
    plt.close(fig)
    return [report, png]
