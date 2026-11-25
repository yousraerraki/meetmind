"""Configuration centrale de MeetMind (lue depuis les variables d'environnement / .env)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]


def _load_dotenv() -> None:
    env_file = ROOT_DIR / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()


@dataclass(frozen=True)
class Settings:
    database_url: str = os.getenv("DATABASE_URL", f"sqlite:///{ROOT_DIR / 'meetmind.db'}")
    nlp_mode: str = os.getenv("NLP_MODE", "auto")  # auto | light | full

    whisper_model: str = os.getenv("WHISPER_MODEL", "small")
    summarizer: str = os.getenv("SUMMARIZER", "barthez")
    summarizer_model: str = os.getenv("SUMMARIZER_MODEL", "moussaKam/barthez-orangesum-abstract")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "http://localhost:11434")
    llm_model: str = os.getenv("LLM_MODEL", "mistral")
    spacy_model: str = os.getenv("SPACY_MODEL", "fr_core_news_lg")
    classifier_model_path: str = os.getenv(
        "CLASSIFIER_MODEL_PATH", str(ROOT_DIR / "models" / "camembert-meetmind")
    )
    tfidf_model_path: str = os.getenv("TFIDF_MODEL_PATH", str(ROOT_DIR / "models" / "tfidf-meetmind.joblib"))
    sbert_model: str = os.getenv("SBERT_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")

    smtp_host: str = os.getenv("SMTP_HOST", "")
    smtp_port: int = int(os.getenv("SMTP_PORT", "587"))
    smtp_user: str = os.getenv("SMTP_USER", "")
    smtp_password: str = os.getenv("SMTP_PASSWORD", "")
    smtp_from: str = os.getenv("SMTP_FROM", "")

    upload_dir: Path = ROOT_DIR / "uploads"

    @property
    def use_models(self) -> bool:
        return self.nlp_mode != "light"


settings = Settings()
settings.upload_dir.mkdir(exist_ok=True)


def model_loader(fn):
    """Comme lru_cache, mais mémorise aussi les échecs : un modèle absent
    n'est pas re-téléchargé / re-chargé à chaque requête (mode auto)."""
    import functools

    state: dict = {}

    @functools.wraps(fn)
    def wrapper():
        if "value" in state:
            return state["value"]
        if "error" in state:
            raise state["error"]
        try:
            state["value"] = fn()
            return state["value"]
        except Exception as exc:  # noqa: BLE001
            state["error"] = exc
            raise

    wrapper.reset = state.clear
    wrapper.loaded = lambda: "value" in state
    wrapper.failed = lambda: "error" in state
    return wrapper
