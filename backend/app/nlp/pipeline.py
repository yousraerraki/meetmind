"""Orchestration du pipeline MeetMind (architecture section 8 du cahier des charges).

  1. Entrée (texte / .docx / audio)          → api / interface
  2. Transcription Whisper                    → ignorée si texte
  3. Prétraitement spaCy / NLTK               → nettoyage, locuteurs, sujets
  4a/4b/4c. Résumé · Décisions · Actions      → EN PARALLÈLE sur le même texte prétraité
  5. Génération du compte rendu               → fusion en un document unique
  6. Stockage + export                        → api (db.py / export/)
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

from .actions import extract_actions
from .classifier import classifier_backend
from .decisions import extract_decisions
from .diarization import rename_speakers
from .ner import ner_backend
from .preprocessing import preprocess
from .schemas import MeetingReport
from .summarizer import summarize_meeting
from .transcription import transcribe

log = logging.getLogger(__name__)


def read_docx(path: str | Path) -> str:
    import docx
    return "\n".join(p.text for p in docx.Document(str(path)).paragraphs if p.text.strip())


def read_pdf(path: str | Path) -> str:
    """Extrait le texte d'un PDF (transcription exportée de Teams/Word, compte rendu brut…)."""
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    text = "\n".join((page.extract_text() or "") for page in reader.pages)
    if not text.strip():
        raise ValueError("Le PDF ne contient pas de texte (PDF scanné ?). Utilisez un PDF texte ou un .txt.")
    return text


def run_pipeline(
    text: str | None = None,
    audio_path: str | Path | None = None,
    title: str = "Réunion",
    meeting_date: date | None = None,
    audio_language: str | None = "fr",
    speaker_names: dict[str, str] | None = None,
    num_speakers: int | None = None,
    vocabulary: str | None = None,
) -> MeetingReport:
    meeting_date = meeting_date or date.today()
    timings: dict[str, float] = {}
    models: dict[str, str] = {}
    t0 = time.perf_counter()

    # 2. Transcription
    if audio_path:
        t = time.perf_counter()
        result = transcribe(audio_path, language=audio_language, num_speakers=num_speakers, vocabulary=vocabulary)
        text = result["text"]
        timings["transcription"] = round(time.perf_counter() - t, 2)
        models["transcription"] = "Whisper" + (" + pyannote (diarisation)" if result.get("diarized") else "")
    if speaker_names and text:
        text = rename_speakers(text, speaker_names)
    if not text or not text.strip():
        raise ValueError("La transcription est vide.")

    # 3. Prétraitement
    t = time.perf_counter()
    meeting, seg_method = preprocess(text)
    timings["pretraitement"] = round(time.perf_counter() - t, 2)
    models["segmentation"] = seg_method
    if not meeting.utterances:
        raise ValueError("Aucune phrase exploitable après nettoyage.")

    # 4a / 4b / 4c en parallèle
    models["classification"] = classifier_backend()
    models["ner"] = ner_backend()

    def timed(name, fn, *args):
        s = time.perf_counter()
        out = fn(*args)
        timings[name] = round(time.perf_counter() - s, 2)
        return out

    with ThreadPoolExecutor(max_workers=3) as pool:
        f_sum = pool.submit(timed, "resume", summarize_meeting, meeting)
        f_dec = pool.submit(timed, "decisions", extract_decisions, meeting)
        f_act = pool.submit(timed, "actions", extract_actions, meeting, meeting_date)
        summary, topics, sum_backend = f_sum.result()
        decisions = f_dec.result()
        actions = f_act.result()
    models["resume"] = sum_backend

    # 5. Assemblage du compte rendu
    participants = list(meeting.speakers)
    for a in actions:
        if a.owner and a.owner not in participants:
            participants.append(a.owner)

    timings["total"] = round(time.perf_counter() - t0, 2)
    return MeetingReport(
        title=title,
        meeting_date=meeting_date.isoformat(),
        language=meeting.language,
        participants=participants,
        summary=summary,
        topics=topics,
        decisions=decisions,
        actions=actions,
        clean_transcript="\n".join(
            f"{u.speaker + ' : ' if u.speaker else ''}{u.text}" for u in meeting.utterances
        ),
        models_used=models,
        timings=timings,
    )
