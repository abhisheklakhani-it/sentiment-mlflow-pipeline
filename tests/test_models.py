from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sentiflow import embed
from sentiflow.evaluate import metrics, write_reports
from sentiflow.models import EmbeddingSentimentModel, build_classifier


def toy_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1, 2], 30)
    x = rng.normal(size=(90, 8)) + y[:, None] * 2.0  # three well separated blobs
    return x, y


@pytest.mark.parametrize("family", ["logreg", "svm"])
def test_families_fit_and_predict(family: str) -> None:
    x, y = toy_data()
    clf = build_classifier(family, {"C": 1.0, "class_weight": "balanced"}).fit(x, y)
    assert (clf.predict(x) == y).mean() > 0.95


def test_unknown_family_raises() -> None:
    with pytest.raises(ValueError, match="unknown family"):
        build_classifier("forest", {"C": 1.0})


def test_metrics_perfect_and_keys() -> None:
    y = np.array([0, 1, 2, 1])
    m = metrics(y, y, "dev")
    assert m["dev_macro_f1"] == 1.0
    assert m["dev_f1_pos_neg"] == 1.0
    assert set(m) >= {"dev_accuracy", "dev_f1_negative", "dev_f1_neutral", "dev_f1_positive"}


def test_write_reports(tmp_path: Path) -> None:
    y = np.array([0, 1, 2, 1, 0])
    paths = write_reports(y, y, tmp_path, "dev")
    assert all(p.exists() for p in paths)


def test_pyfunc_text_in_label_out(monkeypatch: pytest.MonkeyPatch) -> None:
    x, y = toy_data()
    clf = build_classifier("logreg", {"C": 1.0}).fit(x, y)
    # fake encoder: "schlecht" -> blob 0, "egal" -> blob 1, "super" -> blob 2
    centers = {"schlecht": 0, "egal": 1, "super": 2}
    monkeypatch.setattr(
        embed, "encode", lambda name, texts: np.array([[centers[t] * 2.0] * 8 for t in texts])
    )
    model = EmbeddingSentimentModel("e5-small", clf)
    out = model.predict(None, pd.DataFrame({"text": ["schlecht", "egal", "super"]}))
    assert list(out["label"]) == ["negative", "neutral", "positive"]
    assert out["confidence"].between(0, 1).all()


def test_cached_embeddings_reuses_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def fake_encode(name: str, texts: list[str]) -> np.ndarray:
        calls.append(len(texts))
        return np.ones((len(texts), 4), dtype=np.float32)

    monkeypatch.setattr(embed, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(embed, "encode", fake_encode)
    a = embed.cached_embeddings("e5-small", "train", ["a", "b"])
    b = embed.cached_embeddings("e5-small", "train", ["a", "b"])
    assert calls == [2]
    assert np.array_equal(a, b)
