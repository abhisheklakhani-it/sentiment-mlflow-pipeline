"""Classifier families on top of the embeddings, and the text-in pyfunc wrapper."""

from __future__ import annotations

from typing import Any

import mlflow.pyfunc
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sentiflow.data import LABELS

SEED = 42
FAMILIES = ("logreg", "svm")


def build_classifier(family: str, params: dict[str, Any]) -> Any:
    weight = "balanced" if params.get("class_weight") == "balanced" else None
    if family == "logreg":
        return LogisticRegression(
            C=params["C"], class_weight=weight, max_iter=2000, random_state=SEED
        )
    if family == "svm":
        return LinearSVC(C=params["C"], class_weight=weight, random_state=SEED)
    raise ValueError(f"unknown family {family!r}; choose from {FAMILIES}")


class EmbeddingSentimentModel(mlflow.pyfunc.PythonModel):  # type: ignore[misc]
    """Text in, label out: e5 encoder (loaded by pinned name) + fitted classifier."""

    def __init__(self, encoder: str, classifier: Any) -> None:
        self.encoder = encoder
        self.classifier = classifier

    def predict(self, context: Any, model_input: Any, params: Any = None) -> pd.DataFrame:
        from sentiflow.embed import encode

        texts = _texts(model_input)
        x = encode(self.encoder, texts)
        pred = self.classifier.predict(x)
        out = pd.DataFrame({"label": [LABELS[int(i)] for i in pred]})
        if hasattr(self.classifier, "predict_proba"):
            proba = self.classifier.predict_proba(x)
            out["confidence"] = np.round(proba.max(axis=1), 4)
        return out


def _texts(model_input: Any) -> list[str]:
    if isinstance(model_input, pd.DataFrame):
        col = "text" if "text" in model_input.columns else model_input.columns[0]
        return [str(t) for t in model_input[col]]
    if isinstance(model_input, str):
        return [model_input]
    return [str(t) for t in model_input]


class GbertSentimentModel(mlflow.pyfunc.PythonModel):  # type: ignore[misc]
    """Text in, label out for the fine-tuned gbert (same schema as the embedding models)."""

    def load_context(self, context: Any) -> None:
        import torch
        from transformers import BertForSequenceClassification, BertTokenizerFast

        path = context.artifacts["model_dir"]
        self.tokenizer = BertTokenizerFast.from_pretrained(path)
        self.model = BertForSequenceClassification.from_pretrained(path).eval()
        self.torch = torch

    def predict(self, context: Any, model_input: Any, params: Any = None) -> pd.DataFrame:
        texts = _texts(model_input)
        rows: list[tuple[str, float]] = []
        for start in range(0, len(texts), 32):
            batch = self.tokenizer(
                texts[start : start + 32],
                truncation=True,
                max_length=128,
                padding=True,
                return_tensors="pt",
            )
            with self.torch.no_grad():
                proba = self.model(**batch).logits.softmax(dim=-1).numpy()
            rows += [(LABELS[int(p.argmax())], round(float(p.max()), 4)) for p in proba]
        return pd.DataFrame(rows, columns=["label", "confidence"])
