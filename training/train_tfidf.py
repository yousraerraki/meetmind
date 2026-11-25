"""Classifieur léger TF-IDF + régression logistique (information / décision / action).

Alternative rapide à CamemBERT : s'entraîne en quelques secondes sur CPU, sans GPU
ni téléchargement de modèle. L'API l'utilise automatiquement s'il existe et qu'aucun
CamemBERT fine-tuné n'est présent (ordre : CamemBERT > TF-IDF > règles).

    python training/train_tfidf.py --data data/corpus/annotations_train.csv data/annotations/seed.jsonl data/annotations/reunions_exemple.csv
    python training/train_tfidf.py ... --eval data/corpus/annotations_test.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["information", "decision", "action"]


def load(paths: list[str]) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for p in map(Path, paths):
        if p.suffix == ".jsonl":
            items = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        else:
            items = list(csv.DictReader(p.open(encoding="utf-8")))
        rows += [(r["text"].strip(), r["label"].strip().lower()) for r in items
                 if r.get("text") and r.get("label", "").strip().lower() in LABELS]
    return rows


def build_model():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline, make_union
    features = make_union(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),                  # mots et bigrammes
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True),  # morceaux de mots (darija, fautes)
    )
    return make_pipeline(features, LogisticRegression(max_iter=2000, class_weight="balanced"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--eval", help="CSV de test (text,label) pour mesurer précision / rappel / F1")
    ap.add_argument("--output", default=str(ROOT / "models" / "tfidf-meetmind.joblib"))
    args = ap.parse_args()

    import joblib
    from sklearn.metrics import classification_report

    rows = load(args.data)
    X, y = zip(*rows)
    print(f"{len(rows)} phrases d'entraînement :", {l: y.count(l) for l in LABELS})
    model = build_model().fit(X, y)

    if args.eval:
        Xt, yt = zip(*load([args.eval]))
        print(classification_report(yt, model.predict(Xt), labels=LABELS, digits=3, zero_division=0))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output)
    print(f"Modèle sauvegardé → {args.output}")


if __name__ == "__main__":
    sys.exit(main())
