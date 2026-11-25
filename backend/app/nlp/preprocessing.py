"""Étape 3 — Prétraitement : nettoyage, locuteurs, segmentation par sujet.

- Nettoyage : suppression des hésitations (euh, hum…), répétitions (« je je »), bruit.
- Locuteurs : détection des tours de parole « Nom : texte » / « [00:12] Nom: texte ».
- Segmentation thématique : Sentence-BERT + clustering hiérarchique contraint
  (segments contigus). Fallback : TF-IDF + TextTiling.
"""
from __future__ import annotations

import logging
import re

from ..config import model_loader, settings
from .schemas import PreprocessedMeeting, Topic, Utterance
from .textutils import cosine, detect_language, split_sentences, tfidf_vectors, top_keywords

log = logging.getLogger(__name__)

# ---------------------------------------------------------------- nettoyage
HESITATIONS = [
    r"euh+", r"heu+", r"hum+", r"hm+", r"mmh+", r"bah", r"ben", r"beh", r"pff+",
    r"tu vois", r"tu sais", r"en fait", r"du coup", r"quoi(?=[\s,.!?]*$)",
    r"uh+", r"um+", r"you know", r"i mean",
]
_HESIT_RE = re.compile(r"(?<![\w'’])(?:" + "|".join(HESITATIONS) + r")(?![\w'’])[\s,]*", re.IGNORECASE)
_REPEAT_RE = re.compile(r"\b(\w+(?:['’]\w+)?)(?:[\s,]+\1\b)+", re.IGNORECASE)
_TIMESTAMP_RE = re.compile(r"^\s*[\[(]?\d{1,2}:\d{2}(?::\d{2})?[\])]?\s*[-–]?\s*")
_SPEAKER_RE = re.compile(
    r"^\s*(?P<speaker>[A-ZÀ-Ö][\wÀ-ÖØ-öø-ÿ.'’-]*(?:\s+[A-ZÀ-Ö][\wÀ-ÖØ-öø-ÿ.'’-]*){0,2}|SPEAKER_\d+|Locuteur\s*\d+|Intervenant\s*\d+)"
    r"(?:\s*\((?P<name>[^)]{2,40})\))?"            # « Présidente (Fatima Zahra) : »
    r"\s*[:：]\s*(?P<text>.+)$"
)
NOT_SPEAKERS = {"action", "actions", "décision", "decision", "note", "objet", "date", "ordre du jour",
                "remarque", "important", "todo", "to do", "question", "réponse", "conclusion"}
_NOISE_RE = re.compile(r"\[(?:rires?|inaudible|bruit|silence|musique|laughs?|noise)\]|\((?:rires?|inaudible)\)", re.I)


def clean_text(text: str) -> str:
    text = _NOISE_RE.sub(" ", text)
    text = _HESIT_RE.sub("", text)
    text = _REPEAT_RE.sub(r"\1", text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r"([,.!?])\1+", r"\1", text)
    text = re.sub(r"^[\s,.;:]+", "", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    # majuscule en début de phrase (après suppression d'un « euh, » initial par ex.)
    text = re.sub(r"(^|[.!?…]\s+)([a-zà-ÿ])", lambda m: m.group(1) + m.group(2).upper(), text)
    return text


def parse_turns(raw: str) -> list[tuple[str | None, str]]:
    """Découpe la transcription en tours de parole (locuteur, texte)."""
    turns: list[tuple[str | None, str]] = []
    for line in raw.splitlines():
        line = _TIMESTAMP_RE.sub("", line).strip()
        if not line:
            continue
        m = _SPEAKER_RE.match(line)
        if m and len(m.group("speaker")) <= 40 and m.group("speaker").lower() not in NOT_SPEAKERS:
            speaker = (m.group("name") or m.group("speaker")).strip()
            turns.append((speaker, m.group("text").strip()))
        elif turns:
            # ligne de continuation du locuteur précédent
            spk, txt = turns[-1]
            turns[-1] = (spk, f"{txt} {line}")
        else:
            turns.append((None, line))
    return turns


# ------------------------------------------------------- segmentation thématique
@model_loader
def _load_sbert():
    from sentence_transformers import SentenceTransformer
    log.info("Chargement de Sentence-BERT '%s'…", settings.sbert_model)
    return SentenceTransformer(settings.sbert_model)


def _n_topics(n: int) -> int:
    # ~1 sujet pour 12 phrases, entre 1 et 8
    return max(1, min(8, round(n / 12)))


def _segment_sbert(sentences: list[str]) -> list[int]:
    import numpy as np
    from scipy.sparse import diags
    from sklearn.cluster import AgglomerativeClustering

    model = _load_sbert()
    emb = model.encode(sentences, normalize_embeddings=True)
    # Lissage par fenêtre glissante pour stabiliser les frontières
    smoothed = np.array([emb[max(0, i - 1): i + 2].mean(axis=0) for i in range(len(emb))])
    n = len(sentences)
    connectivity = diags([1, 1, 1], [-1, 0, 1], shape=(n, n))  # segments contigus
    clustering = AgglomerativeClustering(
        n_clusters=_n_topics(n), connectivity=connectivity, linkage="ward"
    ).fit(smoothed)
    return _relabel_in_order(list(clustering.labels_))


def _segment_texttiling(sentences: list[str], window: int = 3) -> list[int]:
    n = len(sentences)
    k = _n_topics(n)
    if k == 1:
        return [0] * n
    vecs = tfidf_vectors(sentences)
    gaps = []
    for i in range(1, n):
        left = {}
        for v in vecs[max(0, i - window): i]:
            for w, x in v.items():
                left[w] = left.get(w, 0) + x
        right = {}
        for v in vecs[i: i + window]:
            for w, x in v.items():
                right[w] = right.get(w, 0) + x
        gaps.append((cosine(left, right), i))
    # Les k-1 frontières les moins similaires, espacées d'au moins 4 phrases
    boundaries: list[int] = []
    for _, i in sorted(gaps):
        if all(abs(i - b) >= 4 for b in boundaries) and 3 <= i <= n - 3:
            boundaries.append(i)
        if len(boundaries) == k - 1:
            break
    labels, t = [], 0
    for i in range(n):
        if i in boundaries:
            t += 1
        labels.append(t)
    return labels


def _relabel_in_order(labels: list[int]) -> list[int]:
    mapping: dict[int, int] = {}
    out = []
    for lab in labels:
        if lab not in mapping:
            mapping[lab] = len(mapping)
        out.append(mapping[lab])
    return out


def segment_topics(sentences: list[str]) -> tuple[list[int], str]:
    if len(sentences) < 8:
        return [0] * len(sentences), "aucune (réunion courte)"
    if settings.use_models:
        try:
            return _segment_sbert(sentences), f"Sentence-BERT ({settings.sbert_model}) + clustering"
        except Exception as exc:  # modèle absent ou erreur
            if settings.nlp_mode == "full":
                raise
            log.info("Sentence-BERT indisponible (%s) → fallback TextTiling", exc)
    return _segment_texttiling(sentences), "TF-IDF + TextTiling (fallback)"


# ------------------------------------------------------------------ principal
def preprocess(raw_text: str) -> tuple[PreprocessedMeeting, str]:
    turns = parse_turns(raw_text)
    utterances: list[Utterance] = []
    speakers: list[str] = []
    for speaker, text in turns:
        if speaker and speaker not in speakers:
            speakers.append(speaker)
        for sent in split_sentences(clean_text(text)):
            if len(sent.split()) < 2 and not re.search(r"\w{4,}", sent):
                continue  # « Ok. », « Oui. »
            utterances.append(Utterance(index=len(utterances), speaker=speaker, text=sent))

    labels, method = segment_topics([u.text for u in utterances])
    topics: dict[int, Topic] = {}
    for u, lab in zip(utterances, labels):
        u.topic_id = lab
        topics.setdefault(lab, Topic(id=lab, label="")).utterance_ids.append(u.index)
    for t in topics.values():
        names = {n.lower() for sp in speakers for n in sp.split()}
        kws = top_keywords([utterances[i].text for i in t.utterance_ids], k=3, exclude=names)
        t.label = " · ".join(kws).capitalize() if kws else f"Sujet {t.id + 1}"

    meeting = PreprocessedMeeting(
        language=detect_language(raw_text),
        utterances=utterances,
        topics=sorted(topics.values(), key=lambda t: t.id),
        speakers=speakers,
    )
    return meeting, method
