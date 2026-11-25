"""Tableau comparatif des modèles de résumé (ROUGE-L, BERTScore, temps).

Chaque modèle est lancé dans un sous-processus (la configuration est lue au démarrage).

    python evaluation/compare_summarizers.py data/samples/*.gold.json
    python evaluation/compare_summarizers.py data/samples/*.gold.json --models extractive barthez mt5 llm
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "extractive": {"SUMMARIZER": "extractive", "SUMMARIZER_MODEL": ""},
    "barthez": {"SUMMARIZER": "barthez", "SUMMARIZER_MODEL": "moussaKam/barthez-orangesum-abstract"},
    "mt5": {"SUMMARIZER": "mt5", "SUMMARIZER_MODEL": "csebuetnlp/mT5_multilingual_XLSum"},
    "llm": {"SUMMARIZER": "llm", "SUMMARIZER_MODEL": ""},
}

WORKER = r"""
import json, sys, time
sys.path.insert(0, sys.argv[1])
from backend.app.nlp.preprocessing import preprocess
from backend.app.nlp.summarizer import summarize_meeting
from evaluation.evaluate import rouge, bertscore
from pathlib import Path
out = []
for p in sys.argv[2:]:
    p = Path(p); gold = json.loads(p.read_text(encoding="utf-8"))
    meeting, _ = preprocess((p.parent / gold["transcript"]).read_text(encoding="utf-8"))
    summarize_meeting(meeting) if not out else None   # chauffe : chargement du modèle hors chrono
    t = time.perf_counter(); summary, _, backend = summarize_meeting(meeting); dt = time.perf_counter() - t
    r = rouge(summary, gold["reference_summary"])
    out.append({"file": p.name, "backend": backend, "rougeL": r["rougeL"], "rouge1": r["rouge1"],
                "bertscore": bertscore(summary, gold["reference_summary"]), "time_s": round(dt, 2), "summary": summary})
print("@@RESULT@@" + json.dumps(out, ensure_ascii=False))
"""


def run(model: str, files: list[str]) -> list[dict]:
    env = {**os.environ, **MODELS[model], "NLP_MODE": "full" if model != "extractive" else "light"}
    proc = subprocess.run([sys.executable, "-c", WORKER, str(ROOT), *files], env=env, capture_output=True, text=True)
    marker = [l for l in proc.stdout.splitlines() if l.startswith("@@RESULT@@")]
    if proc.returncode != 0 or not marker:
        print(f"  ✗ {model} : échec\n{proc.stderr[-800:]}")
        return []
    return json.loads(marker[0][len("@@RESULT@@"):])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("gold", nargs="+")
    ap.add_argument("--models", nargs="+", default=["extractive", "barthez", "mt5"], choices=list(MODELS))
    ap.add_argument("--report", default="evaluation/comparaison_resumes.json")
    args = ap.parse_args()

    table = []
    for m in args.models:
        print(f"→ {m}…", flush=True)
        res = run(m, args.gold)
        if res:
            n = len(res)
            bs = [r["bertscore"] for r in res if r["bertscore"] is not None]
            table.append({
                "modele": m,
                "rougeL": round(sum(r["rougeL"] for r in res) / n, 3),
                "rouge1": round(sum(r["rouge1"] for r in res) / n, 3),
                "bertscore": round(sum(bs) / len(bs), 3) if bs else None,
                "temps_moyen_s": round(sum(r["time_s"] for r in res) / n, 2),
                "details": res,
            })

    print(f"\n{'Modèle':<12}{'ROUGE-L':>9}{'ROUGE-1':>9}{'BERTScore':>11}{'Temps (s)':>11}  Objectif ROUGE-L ≥ 0,35")
    for t in table:
        ok = "✅" if t["rougeL"] >= 0.35 else "❌"
        print(f"{t['modele']:<12}{t['rougeL']:>9}{t['rouge1']:>9}{str(t['bertscore'] or '—'):>11}{t['temps_moyen_s']:>11}  {ok}")
    Path(args.report).write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRapport → {args.report}")


if __name__ == "__main__":
    main()
