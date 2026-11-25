"""Étape 4a — Résumé global + résumé par sujet.

Backends (variable SUMMARIZER) :
  - barthez / mt5 / bart : résumé abstractif Hugging Face (BARThez = français)
  - llm                  : Mistral / Llama 3 en local via Ollama (données non envoyées à l'extérieur)
  - extractive           : TextRank sur TF-IDF (fallback, sans modèle)
"""
from __future__ import annotations

import logging
import re

from ..config import model_loader, settings
from .schemas import PreprocessedMeeting
from .textutils import cosine, tfidf_vectors

log = logging.getLogger(__name__)

DEFAULT_MODELS = {
    "barthez": "moussaKam/barthez-orangesum-abstract",
    "mt5": "csebuetnlp/mT5_multilingual_XLSum",
    "bart": "facebook/bart-large-cnn",  # anglais
}


# ----------------------------------------------------------------- extractif
def textrank(sentences: list[str], k: int, damping: float = 0.85, iters: int = 30) -> list[str]:
    if len(sentences) <= k:
        return sentences
    vecs = tfidf_vectors(sentences)
    n = len(sentences)
    sim = [[cosine(vecs[i], vecs[j]) if i != j else 0.0 for j in range(n)] for i in range(n)]
    out_sum = [sum(row) or 1.0 for row in sim]
    scores = [1.0 / n] * n
    for _ in range(iters):
        scores = [
            (1 - damping) / n + damping * sum(sim[j][i] * scores[j] / out_sum[j] for j in range(n))
            for i in range(n)
        ]
    # léger bonus aux phrases informatives (longueur)
    ranked = sorted(range(n), key=lambda i: scores[i] * min(len(sentences[i].split()), 25) ** 0.5, reverse=True)
    keep = sorted(ranked[:k])
    return [sentences[i] for i in keep]


_FILLER = re.compile(r"^(?:(?:oui|ok|okay|alors|donc|bon|et|euh|parfait|super|d'accord)[,.!]?\s+)+", re.I)


def _extractive(sentences: list[str], ratio: float = 0.25, max_sent: int = 5) -> str:
    # on écarte les questions et les phrases trop courtes, puis on nettoie les mots de remplissage
    cands = [_FILLER.sub("", s).strip() for s in sentences if not s.rstrip().endswith("?") and len(s.split()) >= 5]
    cands = [c[:1].upper() + c[1:] for c in cands if c] or sentences
    k = max(1, min(max_sent, round(len(cands) * ratio)))
    return " ".join(textrank(cands, k))


# ---------------------------------------------------------------- abstractif
def _load_tokenizer(name: str):
    """Charge le tokenizer en essayant plusieurs méthodes : avec transformers v5,
    AutoTokenizer échoue sur BARThez (« 'dict' object is not an instance of 'Sequence' »).
    Dernier recours : construire le tokenizer directement depuis tokenizer.json."""
    import os
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    errors = []
    for kwargs in ({}, {"use_fast": False}):
        try:
            return AutoTokenizer.from_pretrained(name, **kwargs)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{type(exc).__name__}: {exc}")
    if os.path.isdir(name):
        tok_file = os.path.join(name, "tokenizer.json")
    else:
        from huggingface_hub import hf_hub_download
        tok_file = hf_hub_download(name, "tokenizer.json")
    log.info("Tokenizer chargé depuis tokenizer.json (AutoTokenizer : %s)", " | ".join(errors))
    return PreTrainedTokenizerFast(tokenizer_file=tok_file, bos_token="<s>", eos_token="</s>",
                                   unk_token="<unk>", pad_token="<pad>", mask_token="<mask>")


@model_loader
def _load_hf():
    """Modèle seq2seq (BARThez / mT5 / BART) chargé directement : le pipeline
    « summarization » a été supprimé dans transformers v5."""
    import torch
    from transformers import AutoModelForSeq2SeqLM
    name = settings.summarizer_model or DEFAULT_MODELS.get(settings.summarizer)
    log.info("Chargement du modèle de résumé '%s'…", name)
    tok = _load_tokenizer(name)
    model = AutoModelForSeq2SeqLM.from_pretrained(name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return tok, model, device


def _generate(text: str, max_len: int, min_len: int) -> str:
    import torch
    tok, model, device = _load_hf()
    inputs = tok(text, truncation=True, max_length=1024, return_tensors="pt").to(device)
    inputs.pop("token_type_ids", None)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_len, min_new_tokens=min_len, num_beams=4,
                             no_repeat_ngram_size=3, early_stopping=True)
    return tok.decode(out[0], skip_special_tokens=True).strip()


def _chunks(sentences: list[str], max_words: int = 350) -> list[str]:
    chunks, cur, count = [], [], 0
    for s in sentences:
        w = len(s.split())
        if cur and count + w > max_words:
            chunks.append(" ".join(cur))
            cur, count = [], 0
        cur.append(s)
        count += w
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def _abstractive(sentences: list[str], max_len: int = 120) -> str:
    parts = [_generate(chunk, max_len, 20) for chunk in _chunks(sentences)]
    if len(parts) > 2:  # résumé de résumés
        return _generate(" ".join(parts), max_len, 30)
    return " ".join(parts)


def _llm(text: str, instruction: str) -> str:
    import requests
    prompt = (
        "Tu es un assistant qui rédige des comptes rendus de réunion en français, "
        "clairs, factuels et concis. N'invente rien.\n\n"
        f"{instruction}\n\nTranscription :\n\"\"\"\n{text[:12000]}\n\"\"\"\n\nRésumé :"
    )
    r = requests.post(
        f"{settings.llm_base_url}/api/generate",
        json={"model": settings.llm_model, "prompt": prompt, "stream": False, "options": {"temperature": 0.2}},
        timeout=180,
    )
    r.raise_for_status()
    return r.json()["response"].strip()


# ----------------------------------------------------------------- interface
def summarizer_backend() -> str:
    if not settings.use_models or settings.summarizer == "extractive":
        return "TextRank extractif (fallback)"
    if settings.summarizer == "llm":
        return f"LLM {settings.llm_model} (Ollama, local)"
    name = f"{settings.summarizer} ({settings.summarizer_model or DEFAULT_MODELS.get(settings.summarizer)})"
    if _load_hf.failed():
        return "TextRank extractif (fallback : modèle de résumé indisponible)"
    return name if _load_hf.loaded() else f"{name} — chargé à la première réunion"


def _summarize(sentences: list[str], instruction: str, max_sent: int) -> tuple[str, str]:
    backend = summarizer_backend()
    if not sentences:
        return "", backend
    try:
        if settings.use_models and settings.summarizer == "llm":
            return _llm(" ".join(sentences), instruction), backend
        if settings.use_models and settings.summarizer in DEFAULT_MODELS:
            summary = _abstractive(sentences)
            return summary, summarizer_backend()
    except Exception as exc:
        if settings.nlp_mode == "full":
            raise
        log.warning("Résumé abstractif indisponible (%s: %s) → TextRank", type(exc).__name__, exc, exc_info=True)
    return _extractive(sentences, max_sent=max_sent), "TextRank extractif (fallback)"


def summarize_meeting(meeting: PreprocessedMeeting) -> tuple[str, list[dict], str]:
    """Retourne (résumé global, [{id, label, summary}], backend utilisé)."""
    all_sents = [u.text for u in meeting.utterances]
    topics_out = []
    backend = summarizer_backend()
    if len(meeting.topics) > 1:
        for t in meeting.topics:
            sents = [meeting.utterances[i].text for i in t.utterance_ids]
            summary, backend = _summarize(sents, f"Résume ce passage (sujet : {t.label}) en 2 à 3 phrases.", max_sent=2)
            topics_out.append({"id": t.id, "label": t.label, "summary": summary})
    global_summary, backend = _summarize(
        all_sents,
        "Rédige un résumé global de la réunion en 4 à 6 phrases : objectif, points discutés, conclusions.",
        max_sent=5,
    )
    if not topics_out and meeting.topics:
        topics_out = [{"id": 0, "label": meeting.topics[0].label, "summary": global_summary}]
    return global_summary, topics_out, backend
