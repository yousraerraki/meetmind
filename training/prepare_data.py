"""Pré-annotation des phrases pour entraîner CamemBERT.

Prend des transcriptions (.txt — par ex. vos réunions, ou AMI / QMSum traduits),
les découpe en phrases, et propose une étiquette (decision / action / information)
avec les règles. Le binôme corrige ensuite le fichier à la main (colonne "label").

Usage :
    python training/prepare_data.py data/transcripts/*.txt -o data/annotations/a_corriger.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.nlp.classifier import rule_classify  # noqa: E402
from backend.app.nlp.preprocessing import preprocess  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("-o", "--output", default="data/annotations/a_corriger.csv")
    args = ap.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "speaker", "text", "label", "confiance_regles"])
        for f in args.files:
            meeting, _ = preprocess(Path(f).read_text(encoding="utf-8"))
            for u in meeting.utterances:
                label, conf = rule_classify(u.text)
                w.writerow([Path(f).name, u.speaker or "", u.text, label, f"{conf:.2f}"])
                n += 1
    print(f"{n} phrases pré-annotées → {out}")
    print("Corrigez la colonne 'label' (decision | action | information), puis lancez train_classifier.py")


if __name__ == "__main__":
    main()
