"""Étape 2 — Transcription audio → texte avec Whisper (OpenAI).

Ignorée si l'entrée est déjà du texte. Le modèle tourne en local :
aucune donnée de réunion ne quitte la machine (besoin de confidentialité).
"""
from __future__ import annotations

import logging
from pathlib import Path

from ..config import model_loader, settings

log = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm"}


class TranscriptionUnavailable(RuntimeError):
    pass


@model_loader
def _load_whisper():
    try:
        import whisper  # openai-whisper
    except ImportError as exc:
        raise TranscriptionUnavailable(
            "Whisper n'est pas installé. Lancez : pip install openai-whisper "
            "(et installez ffmpeg)."
        ) from exc
    log.info("Chargement de Whisper '%s'…", settings.whisper_model)
    return whisper.load_model(settings.whisper_model)


def build_prompt(vocabulary: str | None, language: str | None) -> str:
    """Texte d'amorce (initial_prompt) : aide Whisper à bien écrire les noms propres
    et le vocabulaire du domaine (« BARThez », « Yousra », « encadrant »…)."""
    import os
    base = {"fr": "Compte rendu d'une réunion d'équipe en français.",
            "en": "Transcript of a team meeting in English."}.get(language or "", "")
    extra = ", ".join(w.strip() for w in [os.getenv("WHISPER_PROMPT", ""), vocabulary or ""] if w and w.strip())
    return f"{base} Vocabulaire : {extra}." if extra else base


def transcribe(audio_path: str | Path, language: str | None = "fr", num_speakers: int | None = None,
               vocabulary: str | None = None) -> dict:
    """Retourne {'text': str, 'language': str, 'segments': [{start, end, text}]}.

    language=None laisse Whisper détecter la langue (fr / en / ar…).
    """
    audio_path = Path(audio_path)
    if audio_path.suffix.lower() not in AUDIO_EXTENSIONS:
        raise ValueError(f"Format audio non supporté : {audio_path.suffix}")
    model = _load_whisper()
    prompt = build_prompt(vocabulary, language)
    result = model.transcribe(str(audio_path), language=language, fp16=False,
                              initial_prompt=prompt or None, condition_on_previous_text=True)
    segments = [
        {"start": round(s["start"], 1), "end": round(s["end"], 1), "text": s["text"].strip()}
        for s in result.get("segments", [])
    ]
    diarized = False
    from .diarization import assign_speakers, diarization_enabled, diarize, to_turn_text
    if diarization_enabled() and segments:
        try:
            segments = assign_speakers(segments, diarize(str(audio_path), num_speakers))
            text = to_turn_text(segments)
            diarized = True
        except Exception as exc:  # pyannote absent / token manquant : on continue sans locuteurs
            log.warning("Diarisation indisponible : %s", exc)
    if not diarized:
        # Une ligne par segment : le prétraitement les traitera comme des tours de parole.
        text = "\n".join(s["text"] for s in segments) or result.get("text", "")
    return {"text": text, "language": result.get("language", language or "fr"),
            "segments": segments, "diarized": diarized}
