"""Évaluation de MeetMind (objectifs du cahier des charges) :

  - Résumé  : ROUGE-1/2/L (objectif ROUGE-L ≥ 0,35) + BERTScore (si installé)
  - Décisions et actions : précision / rappel / F1 (objectif F1 actions ≥ 0,75)
    Une action prédite est correcte si sa tâche est proche d'une action de référence
    (similarité de mots ≥ seuil). On mesure aussi l'exactitude du responsable et de l'échéance.
  - Temps de traitement

Format d'un fichier de référence (*.gold.json) : voir data/samples/reunion_projet.gold.json

Usage :
    python evaluation/evaluate.py data/samples/*.gold.json
    python evaluation/evaluate.py data/samples/*.gold.json --report evaluation/resultats.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.nlp.pipeline import run_pipeline  # noqa: E402
from backend.app.nlp.textutils import content_words, strip_accents  # noqa: E402


def _words(t: str) -> set[str]:
    return {strip_accents(w) for w in content_words(t)}


def similarity(a: str, b: str) -> float:
    wa, wb = _words(a), _words(b)
    return len(wa & wb) / max(min(len(wa), len(wb)), 1) if wa and wb else 0.0


def match(preds: list[str], refs: list[str], threshold: float) -> list[tuple[int, int]]:
    """Appariement glouton prédictions ↔ références."""
    pairs = sorted(((similarity(p, r), i, j) for i, p in enumerate(preds) for j, r in enumerate(refs)), reverse=True)
    used_p, used_r, out = set(), set(), []
    for s, i, j in pairs:
        if s < threshold:
            break
        if i not in used_p and j not in used_r:
            used_p.add(i), used_r.add(j)
            out.append((i, j))
    return out


_TITLES = {"m", "mme", "mlle", "dr", "pr", "prof", "inf", "ing", "chef", "monsieur", "madame", "docteur", "professeur"}


def same_person(pred: str | None, gold: str | None) -> bool:
    """« Ouazzani » == « M. Ouazzani » ; « Rania » == « Rania Bennani » (titres ignorés)."""
    if not pred or not gold:
        return not pred and not gold
    norm = lambda s: {t for t in re.findall(r"[a-zà-ÿ]+", strip_accents(s.lower())) if t not in _TITLES}  # noqa: E731
    return bool(norm(pred) & norm(gold))


def sentence_report(csv_path: str) -> dict:
    """Précision / rappel / F1 de la classification de phrases (décision, action) sur un CSV annoté."""
    import csv as _csv
    from backend.app.nlp.classifier import classify
    rows = list(_csv.DictReader(open(csv_path, encoding="utf-8")))
    out = {}
    for lab in ("decision", "action"):
        tp = sum(1 for r in rows if r["label"] == lab and classify(r["text"])[0] == lab)
        n_pred = sum(1 for r in rows if classify(r["text"])[0] == lab)
        n_ref = sum(1 for r in rows if r["label"] == lab)
        out[lab] = prf(tp, n_pred, n_ref)
    return out


def prf(n_match: int, n_pred: int, n_ref: int) -> dict:
    p = n_match / n_pred if n_pred else 0.0
    r = n_match / n_ref if n_ref else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 3), "rappel": round(r, 3), "f1": round(f, 3)}


def rouge(pred: str, ref: str) -> dict:
    from rouge_score import rouge_scorer
    # Pas de stemmer anglais : on compare des mots français normalisés
    sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    s = sc.score(strip_accents(ref.lower()), strip_accents(pred.lower()))
    return {k: round(v.fmeasure, 3) for k, v in s.items()}


def bertscore(pred: str, ref: str) -> float | None:
    try:
        from bert_score import score
    except ImportError:
        return None
    _, _, f = score([pred], [ref], lang="fr", verbose=False)
    return round(float(f[0]), 3)


def evaluate_file(path: Path, threshold: float) -> dict:
    gold = json.loads(path.read_text(encoding="utf-8"))
    transcript = (path.parent / gold["transcript"]).read_text(encoding="utf-8")
    t = time.perf_counter()
    rep = run_pipeline(text=transcript, meeting_date=date.fromisoformat(gold["meeting_date"]))
    elapsed = time.perf_counter() - t

    res = {"fichier": path.name, "temps_s": round(elapsed, 2), "modeles": rep.models_used}
    res["resume"] = rouge(rep.summary, gold["reference_summary"])
    bs = bertscore(rep.summary, gold["reference_summary"])
    if bs is not None:
        res["resume"]["bertscore_f1"] = bs

    dec_pred = [d.text for d in rep.decisions]
    dec_m = match(dec_pred, gold["decisions"], threshold)
    res["decisions"] = prf(len(dec_m), len(dec_pred), len(gold["decisions"]))

    act_pred = [a.task for a in rep.actions]
    act_ref = [a["task"] for a in gold["actions"]]
    act_m = match(act_pred, act_ref, threshold)
    res["actions"] = prf(len(act_m), len(act_pred), len(act_ref))
    if act_m:
        owner_ok = sum(same_person(rep.actions[i].owner, gold["actions"][j]["owner"]) for i, j in act_m)
        date_ok = sum(rep.actions[i].deadline_date == gold["actions"][j].get("deadline_date") for i, j in act_m)
        res["actions"]["responsable_correct"] = round(owner_ok / len(act_m), 3)
        res["actions"]["echeance_correcte"] = round(date_ok / len(act_m), 3)
    res["details"] = {
        "actions_non_trouvees": [act_ref[j] for j in range(len(act_ref)) if j not in {b for _, b in act_m}],
        "actions_en_trop": [act_pred[i] for i in range(len(act_pred)) if i not in {a for a, _ in act_m}],
        "decisions_non_trouvees": [gold["decisions"][j] for j in range(len(gold["decisions"]))
                                   if j not in {b for _, b in dec_m}],
        "decisions_en_trop": [dec_pred[i] for i in range(len(dec_pred)) if i not in {a for a, _ in dec_m}],
    }
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("gold", nargs="+")
    ap.add_argument("--threshold", type=float, default=0.5, help="similarité minimale pour apparier deux éléments")
    ap.add_argument("--report", help="chemin du rapport JSON")
    ap.add_argument("--sentences", help="CSV annoté (text,label) : évalue aussi la classification de phrases")
    args = ap.parse_args()

    results = [evaluate_file(Path(p), args.threshold) for p in args.gold]
    n = len(results)
    avg = lambda f: round(sum(f(r) for r in results) / n, 3)  # noqa: E731
    summary = {
        "n_reunions": n,
        "rougeL_moyen": avg(lambda r: r["resume"]["rougeL"]),
        "f1_decisions_moyen": avg(lambda r: r["decisions"]["f1"]),
        "f1_actions_moyen": avg(lambda r: r["actions"]["f1"]),
        "temps_moyen_s": avg(lambda r: r["temps_s"]),
    }
    summary["objectif_rougeL_0.35"] = "✅" if summary["rougeL_moyen"] >= 0.35 else "❌"
    summary["objectif_f1_actions_0.75"] = "✅" if summary["f1_actions_moyen"] >= 0.75 else "❌"

    for r in results:
        print(f"\n=== {r['fichier']}  ({r['temps_s']} s)")
        print("  Résumé    :", r["resume"])
        print("  Décisions :", r["decisions"])
        print("  Actions   :", {k: v for k, v in r["actions"].items()})
        for k, v in r["details"].items():
            if v:
                print(f"  {k} : {v}")
    print("\n=== GLOBAL")
    for k, v in summary.items():
        print(f"  {k:28s} {v}")
    if args.sentences:
        sr = sentence_report(args.sentences)
        summary["classification_phrases"] = sr
        print("\n=== CLASSIFICATION DES PHRASES (" + args.sentences + ")")
        for lab, v in sr.items():
            print(f"  {lab:10s} {v}")
    if args.report:
        Path(args.report).write_text(json.dumps({"global": summary, "reunions": results}, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
        print(f"\nRapport → {args.report}")


if __name__ == "__main__":
    main()
