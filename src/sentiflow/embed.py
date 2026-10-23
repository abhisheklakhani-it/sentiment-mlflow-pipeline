"""Sentence embeddings with multilingual-e5, cached on disk per model and split."""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

# name -> pinned Hugging Face revision
ENCODERS = {
    "e5-small": ("intfloat/multilingual-e5-small", "614241f622f53c4eeff9890bdc4f31cfecc418b3"),
    "e5-base": ("intfloat/multilingual-e5-base", "d128750597153bb5987e10b1c3493a34e5a4502a"),
}
CACHE_DIR = Path("cache")
PREFIX = "query: "  # e5 expects this prefix for classification-style inputs


def device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@cache
def load_encoder(name: str) -> Any:
    from sentence_transformers import SentenceTransformer

    repo, revision = ENCODERS[name]
    return SentenceTransformer(repo, revision=revision, device=device())


def encode(name: str, texts: list[str], batch_size: int = 64) -> np.ndarray:
    model = load_encoder(name)
    vectors = model.encode(
        [PREFIX + t for t in texts],
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return np.asarray(vectors, dtype=np.float32)


def cached_embeddings(name: str, split: str, texts: list[str]) -> np.ndarray:
    """Encode once; later calls load the .npy (checked against the text count)."""
    path = CACHE_DIR / f"{name}_{split}.npy"
    if path.exists():
        x = np.load(path)
        if len(x) == len(texts):
            return x
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    x = encode(name, texts)
    np.save(path, x)
    return x
