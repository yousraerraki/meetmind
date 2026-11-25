"""Petits outils texte sans dépendance (utilisés par les fallbacks)."""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from functools import lru_cache

STOPWORDS_FR = set("""
a à ai aie ait alors as au aucun aussi autre aux avec avoir avons avez bon c ça car ce ceci cela celle
celles celui ces cet cette chaque ci comme comment d dans de des deux dit doit donc du elle elles en
encore est et été être eu fait faire fais faut font hors ici il ils j je jusqu l la le les leur leurs
lui m ma mais me même mes moi mon n ne ni nos notre nous on ont ou où par parce pas peu peut plus
pour pourquoi qu quand que quel quelle quelles quels qui sa sans se ses si sien son sont sous sur ta
te tes toi ton tous tout toute toutes très tu un une va vais vas vers voilà vos votre vous vu y
ok okay oui non bien alors donc voilà juste vraiment après avant aussi déjà là ah oh bon ben bah
d'accord c'est j'ai ça n'est qu'on faut veux peux peut bonjour merci point premier deuxième troisième
idée bonne super parfait est-ce qu'il s'il chose choses petit petite fois tout tous
the and to of is in it we i you that for on be this will with are have as at so do
""".split())

_WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+(?:['’-][A-Za-zÀ-ÖØ-öø-ÿ]+)*")


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def content_words(text: str) -> list[str]:
    return [w for w in tokenize(text) if w not in STOPWORDS_FR and len(w) > 2 and not w.isdigit()]


@lru_cache(maxsize=1)
def _nltk_tokenizer():
    try:
        import nltk
        nltk.sent_tokenize("Test. Test.", language="french")
        return nltk.sent_tokenize
    except Exception:  # NLTK absent ou ressource « punkt » non téléchargée
        return None


def split_sentences(text: str) -> list[str]:
    """Découpage en phrases (NLTK si dispo, sinon regex)."""
    text = text.strip()
    if not text:
        return []
    tok = _nltk_tokenizer()
    if tok:
        return [s.strip() for s in tok(text, language="french") if s.strip()]
    parts = re.split(r"(?<=[.!?…])\s+(?=[A-ZÀ-Ö0-9«\"])", text)
    return [p.strip() for p in parts if p.strip()]


def tfidf_vectors(docs: list[str]) -> list[dict[str, float]]:
    tokenized = [content_words(d) for d in docs]
    df = Counter(w for toks in tokenized for w in set(toks))
    n = len(docs)
    vectors = []
    for toks in tokenized:
        tf = Counter(toks)
        vec = {w: (c / max(len(toks), 1)) * (math.log((1 + n) / (1 + df[w])) + 1) for w, c in tf.items()}
        vectors.append(vec)
    return vectors


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    num = sum(a[w] * b[w] for w in common)
    den = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
    return num / den if den else 0.0


def top_keywords(texts: list[str], k: int = 3, exclude: set[str] | None = None) -> list[str]:
    exclude = exclude or set()
    counts = Counter(w for t in texts for w in content_words(t) if w not in exclude)
    return [w for w, _ in counts.most_common(k)]


def detect_language(text: str) -> str:
    """Détection simple fr / en / darija (latin)."""
    toks = tokenize(text[:5000])
    if not toks:
        return "fr"
    fr = {"le", "la", "les", "est", "et", "on", "pour", "que", "des", "une", "pas", "nous", "je"}
    en = {"the", "and", "is", "we", "to", "for", "that", "this", "will", "are", "you", "i"}
    da = {"wach", "dyal", "bghit", "ghadi", "chno", "hna", "mzyan", "daba", "bach", "wla", "f", "3la", "kayn", "lli"}
    scores = {
        "fr": sum(t in fr for t in toks),
        "en": sum(t in en for t in toks),
        "darija": sum(t in da for t in toks) * 1.5,
    }
    return max(scores, key=scores.get) if max(scores.values()) > 0 else "fr"
