import os
from datetime import date
from pathlib import Path

os.environ.setdefault("NLP_MODE", "light")

from backend.app.nlp.classifier import rule_classify  # noqa: E402
from backend.app.nlp.ner import find_deadline, resolve_date  # noqa: E402
from backend.app.nlp.pipeline import run_pipeline  # noqa: E402
from backend.app.nlp.preprocessing import clean_text, parse_turns  # noqa: E402

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "samples" / "reunion_projet.txt"
REF = date(2026, 11, 11)  # mercredi


def test_clean_text_removes_hesitations_and_repetitions():
    assert clean_text("euh je je pense que, hum, c'est bon") == "Je pense que, c'est bon"


def test_parse_turns_with_timestamps():
    turns = parse_turns("[00:01] Rania : Bonjour\nsuite de phrase\nKarim: Salut")
    assert turns == [("Rania", "Bonjour suite de phrase"), ("Karim", "Salut")]


def test_rule_classifier():
    assert rule_classify("On a décidé de garder PostgreSQL.")[0] == "decision"
    assert rule_classify("Je vais préparer la maquette avant vendredi.")[0] == "action"
    assert rule_classify("Le score ROUGE est de 0,38.")[0] == "information"
    assert rule_classify("D'accord, ça me va.")[0] == "information"
    assert rule_classify("On valide le 9 décembre, tout le monde est d'accord ?")[0] == "decision"


def test_dates():
    assert find_deadline("je le fais avant vendredi") == "avant vendredi"
    assert resolve_date("avant vendredi", REF) == "2026-11-13"
    assert resolve_date("demain", REF) == "2026-11-12"
    assert resolve_date("pour le 20 novembre", REF) == "2026-11-20"
    assert resolve_date("d'ici la fin du mois", REF) == "2026-11-30"
    assert resolve_date("15/01", REF) == "2027-01-15"


def test_full_pipeline_on_sample():
    rep = run_pipeline(text=SAMPLE.read_text(encoding="utf-8"), meeting_date=REF)
    assert set(rep.participants) >= {"Karim", "Yousra", "Rania", "Salma"}
    assert rep.summary
    assert len(rep.decisions) == 3
    owners = {(a.owner, a.deadline_date) for a in rep.actions}
    assert ("Yousra", "2026-11-13") in owners
    assert ("Salma", "2026-11-12") in owners
    assert ("Rania", "2026-11-18") in owners


def test_diarization_helpers():
    from backend.app.nlp.diarization import assign_speakers, parse_speaker_names, rename_speakers, to_turn_text
    segs = [{"start": 0, "end": 2, "text": "Bonjour."}, {"start": 2, "end": 4, "text": "On commence."},
            {"start": 4.5, "end": 6, "text": "Je vais préparer la maquette avant vendredi."}]
    turns = [(0, 4.2, "SPEAKER_00"), (4.3, 7, "SPEAKER_01")]
    text = to_turn_text(assign_speakers(segs, turns))
    assert text == "SPEAKER_00 : Bonjour. On commence.\nSPEAKER_01 : Je vais préparer la maquette avant vendredi."
    names = parse_speaker_names("SPEAKER_00=Karim, SPEAKER_01=Rania")
    renamed = rename_speakers(text, names)
    rep = run_pipeline(text=renamed, meeting_date=REF)
    assert rep.actions[0].owner == "Rania" and rep.actions[0].deadline_date == "2026-11-13"


def test_audio_style_transcript_without_speakers():
    text = ("Bonjour à tous.\nNous avons décidé d'utiliser BARThez pour le résumé.\n"
            "Pour le résumé, Rania, tu peux préparer les slides avant vendredi ?\nOui, je m'en occupe.\n"
            "Yousra va envoyer le rapport à l'encadrant demain.\nNous fixons la soutenance au neuf décembre.")
    rep = run_pipeline(text=text, meeting_date=REF)
    assert len(rep.decisions) == 2
    acts = {(a.task, a.owner, a.deadline_date) for a in rep.actions}
    assert ("Préparer les slides", "Rania", "2026-11-13") in acts
    assert ("Envoyer le rapport à l'encadrant", "Yousra", "2026-11-12") in acts
