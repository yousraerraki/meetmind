"""Structures de données partagées par le pipeline."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class Utterance:
    """Une phrase prononcée, après nettoyage."""
    index: int
    speaker: Optional[str]
    text: str
    topic_id: int = 0


@dataclass
class Topic:
    id: int
    label: str
    utterance_ids: list[int] = field(default_factory=list)


@dataclass
class Decision:
    text: str
    speaker: Optional[str] = None
    topic_id: Optional[int] = None
    confidence: float = 1.0


@dataclass
class Action:
    task: str
    owner: Optional[str] = None
    deadline: Optional[str] = None        # texte tel qu'il apparaît ("vendredi prochain")
    deadline_date: Optional[str] = None   # ISO (AAAA-MM-JJ) si résolu
    source: str = ""                      # phrase d'origine
    topic_id: Optional[int] = None
    confidence: float = 1.0
    status: str = "à faire"               # à faire | en cours | fait (suivi)


@dataclass
class PreprocessedMeeting:
    language: str
    utterances: list[Utterance]
    topics: list[Topic]
    speakers: list[str]

    def text(self) -> str:
        return " ".join(u.text for u in self.utterances)

    def topic_text(self, topic_id: int) -> str:
        return " ".join(u.text for u in self.utterances if u.topic_id == topic_id)


@dataclass
class MeetingReport:
    title: str
    meeting_date: str
    language: str
    participants: list[str]
    summary: str
    topics: list[dict]          # [{id, label, summary}]
    decisions: list[Decision]
    actions: list[Action]
    clean_transcript: str
    models_used: dict
    timings: dict

    def to_dict(self) -> dict:
        return asdict(self)
