"""Étape 6a — Stockage : historique et recherche des réunions (SQLite ou PostgreSQL)."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, Column, DateTime, ForeignKey, Integer, String, Text, create_engine, or_
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from .config import settings

connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False)
Base = declarative_base()


class Meeting(Base):
    __tablename__ = "meetings"
    id = Column(Integer, primary_key=True)
    title = Column(String(255), nullable=False)
    meeting_date = Column(String(10))
    language = Column(String(10))
    participants = Column(JSON, default=list)
    summary = Column(Text, default="")
    topics = Column(JSON, default=list)
    raw_transcript = Column(Text, default="")
    clean_transcript = Column(Text, default="")
    models_used = Column(JSON, default=dict)
    timings = Column(JSON, default=dict)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc),
                        onupdate=lambda: datetime.now(timezone.utc))

    decisions = relationship("DecisionRow", cascade="all, delete-orphan", order_by="DecisionRow.position")
    actions = relationship("ActionRow", cascade="all, delete-orphan", order_by="ActionRow.position")


class DecisionRow(Base):
    __tablename__ = "decisions"
    id = Column(Integer, primary_key=True)
    meeting_id = Column(Integer, ForeignKey("meetings.id", ondelete="CASCADE"), index=True)
    position = Column(Integer, default=0)
    text = Column(Text, nullable=False)
    speaker = Column(String(120))
    confidence = Column(String(10))


class ActionRow(Base):
    __tablename__ = "actions"
    id = Column(Integer, primary_key=True)
    meeting_id = Column(Integer, ForeignKey("meetings.id", ondelete="CASCADE"), index=True)
    position = Column(Integer, default=0)
    task = Column(Text, nullable=False)
    owner = Column(String(120))
    deadline = Column(String(120))
    deadline_date = Column(String(10))
    status = Column(String(20), default="à faire")
    source = Column(Text, default="")


def init_db() -> None:
    Base.metadata.create_all(engine)


def get_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ------------------------------------------------------------------ helpers
def save_report(db, report, raw_text: str) -> Meeting:
    m = Meeting(
        title=report.title,
        meeting_date=report.meeting_date,
        language=report.language,
        participants=report.participants,
        summary=report.summary,
        topics=report.topics,
        raw_transcript=raw_text,
        clean_transcript=report.clean_transcript,
        models_used=report.models_used,
        timings=report.timings,
    )
    m.decisions = [
        DecisionRow(position=i, text=d.text, speaker=d.speaker, confidence=str(d.confidence))
        for i, d in enumerate(report.decisions)
    ]
    m.actions = [
        ActionRow(position=i, task=a.task, owner=a.owner, deadline=a.deadline,
                  deadline_date=a.deadline_date, status=a.status, source=a.source)
        for i, a in enumerate(report.actions)
    ]
    db.add(m)
    db.commit()
    db.refresh(m)
    return m


def meeting_to_dict(m: Meeting, full: bool = True) -> dict:
    d = {
        "id": m.id,
        "title": m.title,
        "meeting_date": m.meeting_date,
        "language": m.language,
        "participants": m.participants or [],
        "summary": m.summary,
        "n_decisions": len(m.decisions),
        "n_actions": len(m.actions),
        "n_actions_open": sum(a.status != "fait" for a in m.actions),
        "created_at": m.created_at.isoformat() if m.created_at else None,
    }
    if full:
        d.update({
            "topics": m.topics or [],
            "decisions": [{"id": x.id, "text": x.text, "speaker": x.speaker} for x in m.decisions],
            "actions": [
                {"id": a.id, "task": a.task, "owner": a.owner, "deadline": a.deadline,
                 "deadline_date": a.deadline_date, "status": a.status, "source": a.source}
                for a in m.actions
            ],
            "clean_transcript": m.clean_transcript,
            "raw_transcript": m.raw_transcript,
            "models_used": m.models_used or {},
            "timings": m.timings or {},
        })
    return d


def search_meetings(db, q: str | None = None, limit: int = 100) -> list[Meeting]:
    query = db.query(Meeting)
    if q:
        like = f"%{q}%"
        query = query.filter(or_(
            Meeting.title.ilike(like),
            Meeting.summary.ilike(like),
            Meeting.clean_transcript.ilike(like),
            Meeting.decisions.any(DecisionRow.text.ilike(like)),
            Meeting.actions.any(ActionRow.task.ilike(like)),
            Meeting.actions.any(ActionRow.owner.ilike(like)),
        ))
    return query.order_by(Meeting.created_at.desc()).limit(limit).all()


def update_meeting(db, m: Meeting, data: dict) -> Meeting:
    """Édition manuelle du compte rendu généré."""
    for field in ("title", "meeting_date", "summary", "participants", "topics"):
        if field in data and data[field] is not None:
            setattr(m, field, data[field])
    if data.get("decisions") is not None:
        m.decisions = [
            DecisionRow(position=i, text=d["text"], speaker=d.get("speaker"), confidence="manuel")
            for i, d in enumerate(data["decisions"]) if d.get("text", "").strip()
        ]
    if data.get("actions") is not None:
        m.actions = [
            ActionRow(position=i, task=a["task"], owner=a.get("owner"), deadline=a.get("deadline"),
                      deadline_date=a.get("deadline_date"), status=a.get("status") or "à faire",
                      source=a.get("source", ""))
            for i, a in enumerate(data["actions"]) if a.get("task", "").strip()
        ]
    db.commit()
    db.refresh(m)
    return m


__all__ = ["init_db", "get_session", "save_report", "meeting_to_dict", "search_meetings",
           "update_meeting", "Meeting", "ActionRow"]
