"""NER — extraction des responsables (PER) et des échéances (DATE).

- PER : spaCy fr_core_news_lg (ou CamemBERT-NER), avec repli sur la liste des
  participants détectés + mots capitalisés.
- DATE : les modèles spaCy français n'ont pas d'étiquette DATE ; on utilise donc
  des règles qui repèrent ET résolvent les expressions temporelles
  (« vendredi », « demain », « le 15 novembre », « d'ici la fin du mois »…)
  par rapport à la date de la réunion.
"""
from __future__ import annotations

import calendar
import logging
import re
from datetime import date, timedelta

from ..config import model_loader, settings

log = logging.getLogger(__name__)

WEEKDAYS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
MONTHS = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août",
          "septembre", "octobre", "novembre", "décembre"]
_MONTHS_NOACC = [m.replace("é", "e").replace("û", "u") for m in MONTHS]
_MONTH_ALT = "|".join(sorted(set(MONTHS + _MONTHS_NOACC), key=len, reverse=True))

_PREFIX = r"(?:(?:au plus tard |avant |pour |d'ici |d’ici |jusqu'à |jusqu’à |dès |d'ici à |)(?:le |la |ce |cette |)?)"
DATE_PATTERNS = [
    rf"{_PREFIX}(?:{'|'.join(WEEKDAYS)})(?:\s+(?:prochain|matin|soir|après-midi))?(?:\s+\d{{1,2}}(?:\s+(?:{_MONTH_ALT}))?)?",
    rf"{_PREFIX}\d{{1,2}}(?:er)?\s+(?:{_MONTH_ALT})(?:\s+\d{{4}})?",
    rf"{_PREFIX}\d{{1,2}}/\d{{1,2}}(?:/\d{{2,4}})?",
    rf"{_PREFIX}(?:demain|après-demain|aujourd'hui|aujourd’hui|ce soir)",
    rf"{_PREFIX}(?:la |)semaine prochaine",
    rf"{_PREFIX}(?:le |)mois prochain",
    rf"{_PREFIX}(?:la |)fin (?:de la |du |de )(?:semaine|mois|journée|{_MONTH_ALT})",
    rf"{_PREFIX}(?:la |)prochaine réunion",
    rf"{_PREFIX}(?:ce |le )week-?end(?: prochain)?",
    rf"{_PREFIX}cette semaine",
    r"had (?:simana|ssimana)",
    r"(?:9bel |qbel |9bl |)(?:ghedda|ghda|lyouma|jem3a|ljem3a|tnin|ttnin|tlat|larb3|lkhmis|sebt|l7ed)",  # darija
    r"(?:dans|sous) (?:\d+|deux|trois|quatre|cinq|huit|quinze) (?:jours|semaines)",
    r"(?:by |before |until |)(?:next |)(?:monday|tuesday|wednesday|thursday|friday|tomorrow|next week|end of (?:the )?(?:week|month))",
    r"asap|dès que possible|au plus vite|rapidement",
]
DATE_RE = re.compile(r"(?<![\w'’])(?:" + "|".join(f"(?:{p})" for p in DATE_PATTERNS) + ")", re.IGNORECASE)
_NUM_WORDS = {"deux": 2, "trois": 3, "quatre": 4, "cinq": 5, "huit": 8, "quinze": 15}


def find_deadline(text: str) -> str | None:
    matches = [m.group(0).strip(" ,.") for m in DATE_RE.finditer(text) if m.group(0).strip()]
    return max(matches, key=len) if matches else None


def resolve_date(expr: str | None, ref: date) -> str | None:
    """Convertit une expression temporelle en date ISO, relative à la date de réunion."""
    if not expr:
        return None
    e = expr.lower().replace("’", "'")
    e_noacc = e.replace("é", "e").replace("û", "u")

    m = re.search(r"(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?", e)
    if m:
        d, mo = int(m.group(1)), int(m.group(2))
        y = int(m.group(3)) if m.group(3) else ref.year
        y = y + 2000 if y < 100 else y
        try:
            res = date(y, mo, d)
            if not m.group(3) and res < ref:
                res = date(y + 1, mo, d)
            return res.isoformat()
        except ValueError:
            return None

    m = re.search(rf"(\d{{1,2}})(?:er)?\s+({_MONTH_ALT})(?:\s+(\d{{4}}))?", e)
    if m:
        name = m.group(2).replace("é", "e").replace("û", "u")
        mo = _MONTHS_NOACC.index(name) + 1
        y = int(m.group(3)) if m.group(3) else ref.year
        try:
            res = date(y, mo, int(m.group(1)))
            if not m.group(3) and res < ref:
                res = date(y + 1, mo, int(m.group(1)))
            return res.isoformat()
        except ValueError:
            return None

    darija = {"tnin": 0, "ttnin": 0, "tlat": 1, "larb3": 2, "lkhmis": 3, "jem3a": 4, "ljem3a": 4, "sebt": 5, "l7ed": 6}
    for word, wd in darija.items():
        if re.search(rf"\b{word}\b", e):
            return (ref + timedelta(days=(wd - ref.weekday()) % 7 or 7)).isoformat()
    if re.search(r"\b(?:ghedda|ghda)\b", e):
        return (ref + timedelta(days=1)).isoformat()
    if re.search(r"had s?simana", e):
        return (ref + timedelta(days=(4 - ref.weekday()) % 7)).isoformat()
    if "lyouma" in e:
        return ref.isoformat()
    if "cette semaine" in e:
        return (ref + timedelta(days=(4 - ref.weekday()) % 7)).isoformat()  # vendredi
    if "après-demain" in e:
        return (ref + timedelta(days=2)).isoformat()
    if "demain" in e or "tomorrow" in e:
        return (ref + timedelta(days=1)).isoformat()
    if "aujourd'hui" in e or "ce soir" in e:
        return ref.isoformat()

    m = re.search(r"(?:dans|sous) (\d+|\w+) (jours|semaines)", e)
    if m:
        n = int(m.group(1)) if m.group(1).isdigit() else _NUM_WORDS.get(m.group(1), 0)
        return (ref + timedelta(days=n * (7 if m.group(2).startswith("semaine") else 1))).isoformat()

    en_days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    for i, (fr, en) in enumerate(zip(WEEKDAYS, en_days)):
        if fr in e or en in e:
            # « vendredi » ou « vendredi prochain » = le prochain vendredi à venir
            delta = (i - ref.weekday()) % 7 or 7
            return (ref + timedelta(days=delta)).isoformat()

    if "week-end" in e or "weekend" in e:
        return (ref + timedelta(days=(5 - ref.weekday()) % 7)).isoformat()  # samedi
    if "fin de la semaine" in e or "fin de semaine" in e or "end of the week" in e or "end of week" in e:
        return (ref + timedelta(days=(4 - ref.weekday()) % 7)).isoformat()
    if "semaine prochaine" in e or "next week" in e:
        return (ref + timedelta(days=7 - ref.weekday())).isoformat()  # lundi prochain
    if "fin du mois" in e or "end of the month" in e or "end of month" in e:
        return date(ref.year, ref.month, calendar.monthrange(ref.year, ref.month)[1]).isoformat()
    for i, name in enumerate(_MONTHS_NOACC):
        if f"fin {name}" in e_noacc or f"fin de {name}" in e_noacc or f"fin d'{name}" in e_noacc:
            y = ref.year + (1 if i + 1 < ref.month else 0)
            return date(y, i + 1, calendar.monthrange(y, i + 1)[1]).isoformat()
    if "mois prochain" in e:
        y, mo = (ref.year + 1, 1) if ref.month == 12 else (ref.year, ref.month + 1)
        return date(y, mo, 1).isoformat()
    return None


# ------------------------------------------------------------------ personnes
@model_loader
def _load_spacy():
    import spacy
    log.info("Chargement de spaCy '%s'…", settings.spacy_model)
    return spacy.load(settings.spacy_model)


_CAP_RE = re.compile(r"\b([A-ZÀ-Ö][a-zà-öø-ÿ]+(?:[- ][A-ZÀ-Ö][a-zà-öø-ÿ]+)?)\b")
_NOT_NAMES = {"Je", "Tu", "Il", "Elle", "On", "Nous", "Vous", "Ils", "Le", "La", "Les", "Un", "Une",
              "Ok", "Oui", "Non", "Bon", "Alors", "Donc", "Et", "Mais", "Pour", "Avant", "Lundi",
              "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche", "Ce", "Cette", "Il",
              "Super", "Parfait", "Merci", "Bonjour", "Très", "Ça", "Si", "Est", "Qui", "De", "En"}


def ner_backend() -> str:
    if settings.use_models:
        try:
            _load_spacy()
            return f"spaCy {settings.spacy_model} (PER) + règles (DATE)"
        except Exception:
            if settings.nlp_mode == "full":
                raise
    return "règles + liste des participants (fallback)"


def find_persons(text: str, participants: list[str] | None = None) -> list[str]:
    participants = participants or []
    found: list[str] = []
    if settings.use_models:
        try:
            doc = _load_spacy()(text)
            found = [ent.text for ent in doc.ents if ent.label_ in ("PER", "PERSON")]
        except Exception:
            found = []
    # participants connus (prénom ou nom complet)
    for p in participants:
        for part in {p, p.split()[0]}:
            if re.search(rf"\b{re.escape(part)}\b", text) and p not in found:
                found.append(p)
    if not found and not settings.use_models:
        for m in _CAP_RE.finditer(text):
            cand = m.group(1)
            if cand not in _NOT_NAMES and m.start() > 0:
                found.append(cand)
    return found


def canonical_person(name: str | None, participants: list[str]) -> str | None:
    """Associe « Rania » à « Rania Bennani » si c'est un participant."""
    if not name:
        return None
    low = name.lower()
    for p in participants:
        if p.lower() == low or p.lower().split()[0] == low.split()[0]:
            return p
    return name
