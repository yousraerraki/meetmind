"""API MeetMind (FastAPI).

Lancer :  uvicorn backend.app.main:app --reload
Docs   :  http://localhost:8000/docs
"""
from __future__ import annotations

import logging
import re
import shutil
import uuid
from datetime import date
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from sqlalchemy.orm import Session

from . import db as store
from .config import settings
from .export.documents import to_docx, to_pdf, to_text
from .export.mailer import MailNotConfigured, send_report
from .nlp.diarization import parse_speaker_names
from .nlp.pipeline import read_docx, read_pdf, run_pipeline
from .nlp.textutils import strip_accents
from .nlp.transcription import AUDIO_EXTENSIONS, TranscriptionUnavailable

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(
    title="MeetMind API",
    description="Assistant intelligent de comptes rendus de réunion — résumé, décisions, actions.",
    version="0.1.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
store.init_db()

TEXT_EXTENSIONS = {".txt", ".md", ".vtt", ".srt"}
MAX_UPLOAD_MB = 200


# ------------------------------------------------------------------ schémas
class DecisionIn(BaseModel):
    text: str
    speaker: Optional[str] = None


class ActionIn(BaseModel):
    task: str
    owner: Optional[str] = None
    deadline: Optional[str] = None
    deadline_date: Optional[str] = None
    status: Optional[str] = "à faire"
    source: Optional[str] = ""


class MeetingUpdate(BaseModel):
    title: Optional[str] = None
    meeting_date: Optional[str] = None
    summary: Optional[str] = None
    participants: Optional[list[str]] = None
    topics: Optional[list[dict]] = None
    decisions: Optional[list[DecisionIn]] = None
    actions: Optional[list[ActionIn]] = None


class ActionStatus(BaseModel):
    status: str


class EmailRequest(BaseModel):
    recipients: list[str]
    attach_pdf: bool = True


# ------------------------------------------------------------------ helpers
def _get_meeting(db: Session, meeting_id: int) -> store.Meeting:
    m = db.get(store.Meeting, meeting_id)
    if not m:
        raise HTTPException(404, "Réunion introuvable")
    return m


def _strip_subtitles(text: str) -> str:
    """Nettoie les fichiers .vtt / .srt (Teams, Zoom, Meet) : numéros et horodatages."""
    text = re.sub(r"^WEBVTT.*$", "", text, flags=re.M)
    text = re.sub(r"^\d+\s*$", "", text, flags=re.M)
    text = re.sub(r"^\d{2}:\d{2}:\d{2}[.,]\d{3} --> .*$", "", text, flags=re.M)
    text = re.sub(r"<v ([^>]+)>", r"\1: ", text)  # locuteurs Teams
    return re.sub(r"</?v[^>]*>", "", text)


# ---------------------------------------------------------------- endpoints
@app.get("/health")
def health():
    from .nlp.classifier import classifier_backend
    from .nlp.ner import ner_backend
    from .nlp.summarizer import summarizer_backend
    return {
        "status": "ok",
        "nlp_mode": settings.nlp_mode,
        "models": {
            "resume": summarizer_backend(),
            "classification": classifier_backend(),
            "ner": ner_backend(),
        },
    }


@app.post("/meetings/analyze")
async def analyze(
    title: str = Form("Réunion"),
    meeting_date: Optional[str] = Form(None),
    text: Optional[str] = Form(None),
    audio_language: Optional[str] = Form("fr"),
    speaker_names: Optional[str] = Form(None, description="SPEAKER_00=Yousra, SPEAKER_01=Rania"),
    num_speakers: Optional[int] = Form(None),
    vocabulary: Optional[str] = Form(None, description="Noms et mots importants pour Whisper"),
    file: Optional[UploadFile] = File(None),
    db: Session = Depends(store.get_session),
):
    """Étapes 1 → 6 : importe une transcription (texte, .txt/.docx/.vtt) ou un audio, analyse et enregistre."""
    try:
        ref_date = date.fromisoformat(meeting_date) if meeting_date else date.today()
    except ValueError:
        raise HTTPException(422, "meeting_date doit être au format AAAA-MM-JJ")

    audio_path: Path | None = None
    raw_text = (text or "").strip()
    if file is not None and file.filename:
        ext = Path(file.filename).suffix.lower()
        tmp = settings.upload_dir / f"{uuid.uuid4().hex}{ext}"
        with tmp.open("wb") as fh:
            shutil.copyfileobj(file.file, fh)
        if tmp.stat().st_size > MAX_UPLOAD_MB * 1024 * 1024:
            tmp.unlink(missing_ok=True)
            raise HTTPException(413, f"Fichier trop volumineux (max {MAX_UPLOAD_MB} Mo)")
        try:
            if ext in TEXT_EXTENSIONS:
                content = tmp.read_text(encoding="utf-8", errors="ignore")
                raw_text = _strip_subtitles(content) if ext in {".vtt", ".srt"} else content
            elif ext == ".docx":
                raw_text = read_docx(tmp)
            elif ext == ".pdf":
                try:
                    raw_text = read_pdf(tmp)
                except ValueError as exc:
                    raise HTTPException(422, str(exc))
            elif ext in AUDIO_EXTENSIONS:
                audio_path = tmp
            else:
                raise HTTPException(415, f"Format non supporté : {ext}")
        finally:
            if audio_path is None:
                tmp.unlink(missing_ok=True)

    if not raw_text and audio_path is None:
        raise HTTPException(422, "Fournissez un texte, un fichier .txt/.docx ou un audio .mp3/.wav")

    try:
        report = run_pipeline(text=raw_text or None, audio_path=audio_path, title=title,
                              meeting_date=ref_date, audio_language=audio_language or None,
                              speaker_names=parse_speaker_names(speaker_names), num_speakers=num_speakers,
                              vocabulary=vocabulary)
    except TranscriptionUnavailable as exc:
        raise HTTPException(503, str(exc))
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    finally:
        if audio_path:
            audio_path.unlink(missing_ok=True)  # confidentialité : on ne garde pas l'audio

    if audio_path:
        raw_text = report.clean_transcript
    meeting = store.save_report(db, report, raw_text)
    return store.meeting_to_dict(meeting)


@app.get("/meetings")
def list_meetings(q: Optional[str] = None, db: Session = Depends(store.get_session)):
    """Historique + recherche par mot-clé (titre, résumé, transcription, décisions, actions, responsables)."""
    return [store.meeting_to_dict(m, full=False) for m in store.search_meetings(db, q)]


@app.get("/meetings/{meeting_id}")
def get_meeting(meeting_id: int, db: Session = Depends(store.get_session)):
    return store.meeting_to_dict(_get_meeting(db, meeting_id))


@app.put("/meetings/{meeting_id}")
def edit_meeting(meeting_id: int, body: MeetingUpdate, db: Session = Depends(store.get_session)):
    """Correction manuelle du compte rendu généré."""
    m = store.update_meeting(db, _get_meeting(db, meeting_id), body.model_dump(exclude_none=True))
    return store.meeting_to_dict(m)


@app.delete("/meetings/{meeting_id}", status_code=204)
def delete_meeting(meeting_id: int, db: Session = Depends(store.get_session)):
    db.delete(_get_meeting(db, meeting_id))
    db.commit()
    return Response(status_code=204)


@app.get("/actions")
def list_actions(owner: Optional[str] = None, status: Optional[str] = None,
                 db: Session = Depends(store.get_session)):
    """Suivi des actions après la réunion (toutes réunions confondues)."""
    q = db.query(store.ActionRow, store.Meeting).join(store.Meeting, store.ActionRow.meeting_id == store.Meeting.id)
    if owner:
        q = q.filter(store.ActionRow.owner.ilike(f"%{owner}%"))
    if status:
        q = q.filter(store.ActionRow.status == status)
    rows = q.order_by(store.ActionRow.deadline_date.is_(None), store.ActionRow.deadline_date).all()
    return [
        {"id": a.id, "task": a.task, "owner": a.owner, "deadline": a.deadline, "deadline_date": a.deadline_date,
         "status": a.status, "meeting_id": m.id, "meeting_title": m.title}
        for a, m in rows
    ]


@app.patch("/actions/{action_id}")
def set_action_status(action_id: int, body: ActionStatus, db: Session = Depends(store.get_session)):
    if body.status not in {"à faire", "en cours", "fait"}:
        raise HTTPException(422, "Statut attendu : à faire | en cours | fait")
    a = db.get(store.ActionRow, action_id)
    if not a:
        raise HTTPException(404, "Action introuvable")
    a.status = body.status
    db.commit()
    return {"id": a.id, "status": a.status}


@app.get("/meetings/{meeting_id}/export/{fmt}")
def export(meeting_id: int, fmt: str, db: Session = Depends(store.get_session)):
    m = store.meeting_to_dict(_get_meeting(db, meeting_id))
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", strip_accents(m["title"]))[:60].strip("_") or "reunion"
    if fmt == "pdf":
        data, mime = to_pdf(m), "application/pdf"
    elif fmt == "docx":
        data, mime = to_docx(m), "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    elif fmt == "txt":
        data, mime = to_text(m).encode("utf-8"), "text/plain; charset=utf-8"
    else:
        raise HTTPException(400, "Format : pdf | docx | txt")
    return Response(data, media_type=mime,
                    headers={"Content-Disposition": f'attachment; filename="compte_rendu_{safe}.{fmt}"'})


@app.post("/meetings/{meeting_id}/email")
def email(meeting_id: int, body: EmailRequest, db: Session = Depends(store.get_session)):
    m = store.meeting_to_dict(_get_meeting(db, meeting_id))
    try:
        send_report(m, body.recipients, body.attach_pdf)
    except MailNotConfigured as exc:
        raise HTTPException(503, str(exc))
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    except Exception as exc:  # erreur SMTP
        raise HTTPException(502, f"Échec de l'envoi : {exc}")
    return {"sent_to": body.recipients}
