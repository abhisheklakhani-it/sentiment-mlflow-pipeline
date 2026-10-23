from __future__ import annotations

import pandas as pd

from sentiflow.data import LABELS, normalize, prepare, remove_overlap


def test_normalize_masks_urls_users_and_spaces() -> None:
    assert normalize("RT @Bob  schau  https://t.co/x1 an") == "RT @user schau http an"


def test_prepare_keeps_known_labels_and_drops_duplicates() -> None:
    raw = pd.DataFrame(
        {
            "ID": [1, 2, 3, 4],
            "Text": ["gut", "gut", "schlecht", "egal"],
            "Sentiment": ["positive", "positive", "negative", "mixed"],
        }
    )
    out = prepare(raw)
    assert list(out["text"]) == ["gut", "schlecht"]
    assert set(out["label"]) <= set(LABELS)


def test_remove_overlap() -> None:
    train = pd.DataFrame({"text": ["a", "b", "c"], "label": ["neutral"] * 3})
    dev = pd.DataFrame({"text": ["b"], "label": ["neutral"]})
    test = pd.DataFrame({"text": ["c"], "label": ["neutral"]})
    assert list(remove_overlap(train, dev, test)["text"]) == ["a"]
