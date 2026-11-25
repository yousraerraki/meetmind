"""Étape 4b — Extraction des décisions (phrases d'accord ou de choix final)."""
from __future__ import annotations

import re

from .classifier import classify
from .schemas import Decision, PreprocessedMeeting
from .textutils import content_words

_LEAD_RE = re.compile(r"^(?:(?:ok|okay|d'accord|bon|alors|donc|très bien|parfait|super|bref|voilà|oui)[,.!]?\s+)+", re.I)


_TAG_RE = re.compile(r",?\s*(?:tout le monde est d['’]accord|on est d['’]accord|ça vous va|ok|non|d['’]accord)\s*\?\s*$", re.I)


def _normalize(text: str) -> str:
    text = _LEAD_RE.sub("", text).strip()
    text = _TAG_RE.sub(".", text)
    return text[:1].upper() + text[1:] if text else text


def extract_decisions(meeting: PreprocessedMeeting, min_conf: float = 0.5) -> list[Decision]:
    decisions: list[Decision] = []
    seen: list[set[str]] = []
    for u in meeting.utterances:
        label, conf = classify(u.text)
        if label != "decision" or conf < min_conf:
            continue
        text = _normalize(u.text)
        words = set(content_words(text))
        if len(words) < 2:
            continue  # simple approbation (« C'est validé pour moi »)
        # dédoublonnage : une décision répétée par plusieurs participants
        if any(len(words & s) / max(len(words | s), 1) > 0.6 for s in seen):
            continue
        seen.append(words)
        decisions.append(Decision(text=text, speaker=u.speaker, topic_id=u.topic_id, confidence=round(conf, 2)))
    return decisions
