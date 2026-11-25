"""Diarisation (« qui parle quand ? ») pour les fichiers audio — pyannote.audio.

Whisper donne le texte avec des horodatages mais sans locuteurs. pyannote
détecte les tours de parole (SPEAKER_00, SPEAKER_01…) ; on attribue chaque
segment Whisper au locuteur qui le recouvre le plus, puis on fusionne les
segments consécutifs du même locuteur. L'utilisateur peut ensuite renommer
SPEAKER_00 → « Yousra » (paramètre speaker_names de l'API).

Activation (.env) :  DIARIZATION=on   HF_TOKEN=hf_xxx
(accepter les conditions du modèle pyannote/speaker-diarization-3.1 sur Hugging Face)
"""
from __future__ import annotations

import logging
import os
import re

from ..config import model_loader

log = logging.getLogger(__name__)

DIARIZATION_MODEL = os.getenv("DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1")


def diarization_enabled() -> bool:
    return os.getenv("DIARIZATION", "off").lower() in {"on", "1", "true", "yes"}


@model_loader
def _load_pyannote():
    from pyannote.audio import Pipeline
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN manquant pour pyannote (voir .env.example)")
    log.info("Chargement de %s…", DIARIZATION_MODEL)
    pipe = Pipeline.from_pretrained(DIARIZATION_MODEL, use_auth_token=token)
    try:
        import torch
        if torch.cuda.is_available():
            pipe.to(torch.device("cuda"))
    except ImportError:
        pass
    return pipe


def diarize(audio_path: str, num_speakers: int | None = None) -> list[tuple[float, float, str]]:
    """Retourne [(début, fin, locuteur)]."""
    pipe = _load_pyannote()
    kwargs = {"num_speakers": num_speakers} if num_speakers else {}
    annotation = pipe(str(audio_path), **kwargs)
    return [(turn.start, turn.end, spk) for turn, _, spk in annotation.itertracks(yield_label=True)]


def assign_speakers(segments: list[dict], turns: list[tuple[float, float, str]]) -> list[dict]:
    """Attribue à chaque segment Whisper {start, end, text} le locuteur de recouvrement maximal."""
    out = []
    for seg in segments:
        best, best_ov = None, 0.0
        for start, end, spk in turns:
            ov = min(seg["end"], end) - max(seg["start"], start)
            if ov > best_ov:
                best, best_ov = spk, ov
        if best is None and turns:  # pas de recouvrement : locuteur le plus proche
            mid = (seg["start"] + seg["end"]) / 2
            best = min(turns, key=lambda t: min(abs(t[0] - mid), abs(t[1] - mid)))[2]
        out.append({**seg, "speaker": best})
    return out


def to_turn_text(segments: list[dict]) -> str:
    """Fusionne les segments consécutifs d'un même locuteur → « SPEAKER_00 : texte »."""
    lines: list[list] = []
    for seg in segments:
        spk = seg.get("speaker") or "SPEAKER_?"
        if lines and lines[-1][0] == spk:
            lines[-1][1] += " " + seg["text"].strip()
        else:
            lines.append([spk, seg["text"].strip()])
    return "\n".join(f"{spk} : {txt}" for spk, txt in lines)


def rename_speakers(text: str, names: dict[str, str]) -> str:
    """{'SPEAKER_00': 'Yousra'} → remplace les étiquettes en début de ligne."""
    for label, name in names.items():
        if name.strip():
            text = re.sub(rf"(?m)^{re.escape(label)}(\s*[:：])", name.strip() + r"\1", text)
    return text


def parse_speaker_names(raw: str | None) -> dict[str, str]:
    """« SPEAKER_00=Yousra, SPEAKER_01=Rania » → dict."""
    if not raw:
        return {}
    pairs = [p.split("=", 1) for p in re.split(r"[,;\n]", raw) if "=" in p]
    return {k.strip(): v.strip() for k, v in pairs if k.strip() and v.strip()}
