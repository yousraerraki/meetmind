"""Fine-tuning de CamemBERT pour classer les phrases : information / decision / action.

Données : CSV ou JSONL avec les colonnes "text" et "label".
    python training/train_classifier.py --data data/annotations/seed.jsonl data/annotations/a_corriger.csv

Le modèle est sauvegardé dans models/camembert-meetmind (chemin lu par l'API via
CLASSIFIER_MODEL_PATH). Le script affiche précision / rappel / F1 par classe
(objectif du cahier des charges : F1 ≥ 0,75 sur les actions).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

LABELS = ["information", "decision", "action"]
L2I = {l: i for i, l in enumerate(LABELS)}


def load(paths: list[str]) -> list[dict]:
    rows = []
    for p in paths:
        p = Path(p)
        if p.suffix == ".jsonl":
            rows += [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        else:
            with p.open(encoding="utf-8") as fh:
                rows += list(csv.DictReader(fh))
    rows = [{"text": r["text"].strip(), "label": r["label"].strip().lower()} for r in rows
            if r.get("text") and r.get("label", "").strip().lower() in L2I]
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--model", default="camembert-base")
    ap.add_argument("--output", default="models/camembert-meetmind")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import numpy as np
    import torch
    from datasets import Dataset
    from sklearn.metrics import classification_report, precision_recall_fscore_support
    from sklearn.model_selection import train_test_split
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                              Trainer, TrainingArguments)

    random.seed(args.seed)
    rows = load(args.data)
    print(f"{len(rows)} exemples :", {l: sum(r['label'] == l for r in rows) for l in LABELS})
    train, test = train_test_split(rows, test_size=args.test_size, random_state=args.seed,
                                   stratify=[r["label"] for r in rows])

    tok = AutoTokenizer.from_pretrained(args.model)

    def encode(batch):
        enc = tok(batch["text"], truncation=True, max_length=128)
        enc["labels"] = [L2I[l] for l in batch["label"]]
        return enc

    ds_train = Dataset.from_list(train).map(encode, batched=True, remove_columns=["text", "label"])
    ds_test = Dataset.from_list(test).map(encode, batched=True, remove_columns=["text", "label"])

    # Pondération des classes (les phrases "information" sont majoritaires)
    counts = np.bincount([L2I[r["label"]] for r in train], minlength=len(LABELS))
    weights = torch.tensor(counts.sum() / (len(LABELS) * np.maximum(counts, 1)), dtype=torch.float)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(LABELS),
        id2label={i: l for i, l in enumerate(LABELS)}, label2id=L2I,
    )

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = torch.nn.functional.cross_entropy(outputs.logits, labels, weight=weights.to(outputs.logits.device))
            return (loss, outputs) if return_outputs else loss

    def metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(-1)
        p, r, f, _ = precision_recall_fscore_support(labels, preds, labels=[1, 2], average=None, zero_division=0)
        return {"f1_decision": f[0], "f1_action": f[1], "macro_f1": float(np.mean(f))}

    # Arguments compatibles transformers 4.x et 5.x (certains noms ont changé)
    import inspect
    steps = max(1, len(train) // args.batch) * args.epochs
    wanted = dict(
        output_dir="models/_checkpoints", num_train_epochs=args.epochs, per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=32, learning_rate=args.lr, weight_decay=0.01, warmup_steps=int(0.1 * steps),
        eval_strategy="epoch", evaluation_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="macro_f1", save_total_limit=1, logging_steps=20, seed=args.seed, report_to=[],
    )
    accepted = set(inspect.signature(TrainingArguments.__init__).parameters)
    if "eval_strategy" in accepted:
        wanted.pop("evaluation_strategy")
    targs = TrainingArguments(**{k: v for k, v in wanted.items() if k in accepted})
    trainer = WeightedTrainer(model=model, args=targs, train_dataset=ds_train, eval_dataset=ds_test,
                              data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics)
    trainer.train()

    preds = trainer.predict(ds_test).predictions.argmax(-1)
    print(classification_report([L2I[r["label"]] for r in test], preds, target_names=LABELS, zero_division=0))

    Path(args.output).mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.output)
    tok.save_pretrained(args.output)
    print(f"Modèle sauvegardé → {args.output}")


if __name__ == "__main__":
    main()
