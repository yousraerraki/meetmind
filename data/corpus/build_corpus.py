"""Construit le corpus synthétique MeetMind à partir des fichiers annotés.

Entrées  : train_annote.txt, test_annote.txt, test_references.json
Sorties  : train/*.txt, test/*.txt                (transcriptions « Nom : phrase »)
           annotations_train.csv, annotations_test.csv   (text, label) pour CamemBERT
           test/*.gold.json                        (références pour evaluation/evaluate.py)

    python data/corpus/build_corpus.py
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
LABELS = {"I": "information", "D": "decision", "A": "action"}
MEETING_DATE = "2026-11-16"


def parse(path: Path) -> dict[str, dict]:
    meetings: dict[str, dict] = {}
    current = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") and not line.startswith("###"):
            continue
        if line.startswith("###"):
            ident, _, title = line[3:].partition("|")
            current = ident.strip()
            meetings[current] = {"title": title.strip(), "lines": []}
            continue
        m = re.match(r"^([IDA])\|(.+?)\s*:\s*(.+)$", line)
        if not m or current is None:
            raise ValueError(f"Ligne mal formée dans {path.name} : {raw!r}")
        meetings[current]["lines"].append((LABELS[m.group(1)], m.group(2).strip(), m.group(3).strip()))
    return meetings


def write_split(name: str, meetings: dict[str, dict]) -> list[dict]:
    out_dir = HERE / name
    out_dir.mkdir(exist_ok=True)
    rows = []
    for ident, m in meetings.items():
        (out_dir / f"{ident}.txt").write_text(
            "\n".join(f"{spk} : {txt}" for _, spk, txt in m["lines"]) + "\n", encoding="utf-8")
        rows += [{"source": ident, "speaker": spk, "text": txt, "label": lab} for lab, spk, txt in m["lines"]]
    with (HERE / f"annotations_{name}.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["source", "speaker", "text", "label"])
        w.writeheader()
        w.writerows(rows)
    return rows


def main() -> None:
    train = parse(HERE / "train_annote.txt")
    test = parse(HERE / "test_annote.txt")
    refs = json.loads((HERE / "test_references.json").read_text(encoding="utf-8"))
    tr_rows = write_split("train", train)
    te_rows = write_split("test", test)
    for ident, ref in refs.items():
        gold = {"transcript": f"{ident}.txt", "meeting_date": MEETING_DATE, "title": test[ident]["title"], **ref}
        (HERE / "test" / f"{ident}.gold.json").write_text(json.dumps(gold, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Entraînement : {len(train)} réunions, {len(tr_rows)} phrases {dict(Counter(r['label'] for r in tr_rows))}")
    print(f"Test         : {len(test)} réunions, {len(te_rows)} phrases {dict(Counter(r['label'] for r in te_rows))}")


if __name__ == "__main__":
    main()
