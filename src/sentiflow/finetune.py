"""Fine-tune deepset/gbert-base on SB10k (run on a GPU, e.g. the Colab notebook).

    python -m sentiflow.finetune            # full run (about 5 minutes on a Colab T4)
    python -m sentiflow.finetune --smoke    # 10 steps on 200 tweets, to test the pipeline

Same split and metrics as the embedding models. The epoch with the best dev
macro-F1 is kept. Writes outputs/gbert/ (model + tokenizer) and
outputs/gbert_run.json (params, per-epoch history, dev metrics, dev predictions).
`python -m sentiflow.log_finetuned` then logs that run into MLflow locally.
The test split is not touched here.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from sentiflow.data import LABELS, REVISION, load_split, remove_overlap
from sentiflow.evaluate import metrics

BASE_MODEL = "deepset/gbert-base"
BASE_REVISION = "d50cb1df9660ff2de1af8a5362d322b3d5a1a28a"
SEED = 42
OUT = Path("outputs")


def main() -> None:
    import torch
    from datasets import Dataset

    # gbert's 2020 config names no model_type/tokenizer_class, so Auto* cannot resolve it
    from transformers import (
        BertForSequenceClassification,
        BertTokenizerFast,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--epochs", type=float, default=4)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=128)  # tweets: 99.9% fit
    args = p.parse_args()
    set_seed(SEED)

    dev_df, test_df = load_split("dev"), load_split("test")
    train_df = remove_overlap(load_split("train"), dev_df, test_df)
    if args.smoke:
        train_df, dev_df = train_df.head(200), dev_df.head(200)

    tok = BertTokenizerFast.from_pretrained(
        BASE_MODEL, revision=BASE_REVISION, do_lower_case=False, strip_accents=False
    )

    def to_ds(df: Any) -> Any:
        ds = Dataset.from_dict(
            {"text": df["text"].tolist(), "label": df["label"].map(LABELS.index).tolist()}
        )
        return ds.map(
            lambda b: tok(b["text"], truncation=True, max_length=args.max_length),
            batched=True,
            remove_columns=["text"],
        )

    train_ds, dev_ds = to_ds(train_df), to_ds(dev_df)
    model = BertForSequenceClassification.from_pretrained(
        BASE_MODEL,
        revision=BASE_REVISION,
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id={n: i for i, n in enumerate(LABELS)},
    )

    def compute_metrics(eval_pred: Any) -> dict[str, float]:
        logits, labels = eval_pred
        m = metrics(np.asarray(labels), np.argmax(logits, axis=-1), "dev")
        return {"macro_f1": m["dev_macro_f1"], "accuracy": m["dev_accuracy"]}

    targs = TrainingArguments(
        output_dir=str(OUT / ("checkpoints-smoke" if args.smoke else "checkpoints")),
        num_train_epochs=args.epochs,
        max_steps=10 if args.smoke else -1,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=64,
        warmup_steps=0.1,  # below 1 = ratio of all steps
        weight_decay=0.01,
        eval_strategy="steps" if args.smoke else "epoch",
        eval_steps=5 if args.smoke else None,
        save_strategy="steps" if args.smoke else "epoch",
        save_steps=5 if args.smoke else None,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=20,
        report_to=[],
        fp16=torch.cuda.is_available(),
        seed=SEED,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics,
    )
    start = time.perf_counter()
    trainer.train()
    minutes = (time.perf_counter() - start) / 60

    logits = trainer.predict(dev_ds).predictions
    pred = np.argmax(logits, axis=-1)
    dev_metrics = metrics(np.asarray(dev_ds["label"]), pred, "dev")
    history = [
        {"epoch": h["epoch"], "dev_macro_f1": h["eval_macro_f1"], "dev_loss": h["eval_loss"]}
        for h in trainer.state.log_history
        if "eval_macro_f1" in h
    ]
    train_loss = [
        {"epoch": h["epoch"], "step": h["step"], "train_loss": h["loss"]}
        for h in trainer.state.log_history
        if "loss" in h
    ]
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else str(targs.device)
    result = {
        "params": {
            "base_model": BASE_MODEL,
            "base_revision": BASE_REVISION,
            "dataset_revision": REVISION,
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "seed": SEED,
            "n_train": len(train_ds),
            "smoke": args.smoke,
        },
        "dev_metrics": dev_metrics,
        "history": history,
        "train_loss": train_loss,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "train_minutes": round(minutes, 2),
        "device": gpu,
        "dev_predictions": pred.tolist(),
        "dev_labels": list(dev_ds["label"]),
    }
    tag = "gbert_smoke" if args.smoke else "gbert"
    OUT.mkdir(exist_ok=True)
    (OUT / f"{tag}_run.json").write_text(json.dumps(result, indent=2) + "\n")
    if not args.smoke:
        trainer.save_model(str(OUT / "gbert"))
        tok.save_pretrained(str(OUT / "gbert"))
    print(f"{tag}: dev macro-F1 {dev_metrics['dev_macro_f1']:.4f} ({minutes:.1f} min on {gpu})")


if __name__ == "__main__":
    main()
