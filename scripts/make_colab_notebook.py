"""Build notebooks/finetune_gbert_colab.ipynb from the project's own source files.

The notebook writes data.py, evaluate.py and finetune.py into the Colab machine
unchanged, so Colab trains with exactly the same split and metrics as the local
models.

    uv run python scripts/make_colab_notebook.py
"""

import json
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PINS = " ".join(
    f"{p}=={version(p)}" for p in ("transformers", "datasets", "accelerate", "scikit-learn")
)


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip().splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.strip().splitlines(keepends=True),
    }


def write_file(path: str) -> dict:
    return code(f"%%writefile {path}\n" + (ROOT / path).read_text())


cells = [
    md("""
# Fine-tune gbert-base on SB10k German tweets (Colab)

Part of [sentiment-mlflow-pipeline](https://github.com/abhisheklakhani-it/sentiment-mlflow-pipeline).

1. **Runtime → Change runtime type → T4 GPU**, then **Runtime → Run all**.
2. Training takes about 5 minutes. The last cell downloads `gbert_results.zip`.
3. Put the zip in the project folder and run `make log-gbert`: it unzips
   `outputs/gbert/` and logs the run (params, per-epoch metrics, model) to MLflow.
"""),
    code("!nvidia-smi --query-gpu=name,memory.total --format=csv"),
    code(f"!pip -q install {PINS}"),
    code("!mkdir -p src/sentiflow outputs"),
    code("%%writefile src/sentiflow/__init__.py\n"),
    write_file("src/sentiflow/data.py"),
    write_file("src/sentiflow/evaluate.py"),
    write_file("src/sentiflow/finetune.py"),
    code("import os, sys\nsys.path.insert(0, 'src')\nos.environ['PYTHONPATH'] = 'src'"),
    md("Quick check that the whole pipeline runs (10 steps on 200 tweets, about 1 minute):"),
    code("!python -m sentiflow.finetune --smoke"),
    md("The real run: 4 epochs, best epoch kept by dev macro-F1. The test split is not used."),
    code("!python -m sentiflow.finetune"),
    code(
        "!python -c \"import json; r=json.load(open('outputs/gbert_run.json')); "
        "print(r['dev_metrics'], r['train_minutes'], r['device'])\""
    ),
    code("""
!zip -qr gbert_results.zip outputs/gbert outputs/gbert_run.json
from google.colab import files
files.download("gbert_results.zip")
"""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "T4", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}
out = ROOT / "notebooks" / "finetune_gbert_colab.ipynb"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
print(f"wrote {out.relative_to(ROOT)} ({len(cells)} cells)")
