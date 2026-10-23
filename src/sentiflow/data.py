"""Load SB10k (German tweets, 3 sentiment classes) at a pinned revision.

Splits: train (tuning folds), dev (model selection), test (touched once, at the end).
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

DATASET = "Alienmaster/SB10k"
REVISION = "114e22a55a6ec9b64757a1c9f76445249f4946be"  # pinned for reproducibility
LABELS = ["negative", "neutral", "positive"]
SPLITS = {"train": "train", "dev": "dev", "test": "test"}
DATA_DIR = Path("data")

_URL = re.compile(r"https?://\S+")
_USER = re.compile(r"@\w+")
_SPACES = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Mask URLs and user names (they carry no sentiment) and squeeze spaces."""
    text = _URL.sub("http", text)
    text = _USER.sub("@user", text)
    return _SPACES.sub(" ", text).strip()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "id": df["ID"].astype(str),
            "text": df["Text"].astype(str).map(normalize),
            "label": df["Sentiment"].astype(str),
        }
    )
    out = out[out["label"].isin(LABELS)]
    out = out.drop_duplicates("text").reset_index(drop=True)
    return out


def download(data_dir: Path = DATA_DIR) -> dict[str, Path]:
    """Download once, save as parquet; later calls read the local files."""
    from datasets import load_dataset

    data_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    ds = None
    for name, hf_split in SPLITS.items():
        path = data_dir / f"{name}.parquet"
        if not path.exists():
            if ds is None:
                ds = load_dataset(DATASET, revision=REVISION)
            prepare(ds[hf_split].to_pandas()).to_parquet(path, index=False)
        paths[name] = path
    return paths


def load_split(name: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    path = data_dir / f"{name}.parquet"
    if not path.exists():
        download(data_dir)
    return pd.read_parquet(path)


def remove_overlap(train: pd.DataFrame, *others: pd.DataFrame) -> pd.DataFrame:
    """Drop train tweets whose text also appears in dev/test (avoids leakage)."""
    seen = set().union(*(set(o["text"]) for o in others))
    return train[~train["text"].isin(seen)].reset_index(drop=True)
