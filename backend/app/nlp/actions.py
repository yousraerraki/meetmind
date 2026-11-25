"""Étape 4c — Extraction des actions : tâche · responsable · échéance.

CamemBERT (classification) + NER (responsable, date) + règles de résolution :
  « je vais… »            → le locuteur
  « Rania, tu peux… »     → la personne interpellée
  « tu peux… ? » / « Oui je m'en occupe » → la personne qui accepte
  « Yousra va… »          → la personne nommée
"""
from __future__ import annotations

import re
from datetime import date

from .classifier import classify
from .ner import canonical_person, find_deadline, find_persons, resolve_date
from .schemas import Action, PreprocessedMeeting

_ACCEPT_RE = re.compile(
    r"^(?:(?:oui|ok|okay|d'accord|ça marche|pas de (?:souci|problème)|entendu|parfait|sans problème|yes|sure)[,.! ]*)*"
    r"(?:je (?:m'en|m’en) (?:occupe|charge)|je le fais|je (?:prends|gère)|c'est noté|ça marche|pas de (?:souci|problème)|"
    r"je peux le faire|je (?:peux|vais) m['’]en (?:occuper|charger)|i'll do it|i'll handle it|no problem)",
    re.I,
)
_FIRST_PERSON = re.compile(r"(?:\b(?:je|moi je|i|i'll|ana)\b|\bj['’]|\bghadi n|\bghan|\bkhassni)", re.I)
_SECOND_PERSON = re.compile(
    r"(?:\b(?:tu|vous|te|you|nta|nti|ntouma)\b|\bt['’]|\bkhass(?:ek|k|kom)\b|\bma tnsa\w*"
    r"|\b(?:sifet|sifti|dir|diri|kteb|ktbi|3ayet|3ayti) (?:lia|lina)\b)", re.I)
_FOLLOWUP_RE = re.compile(r"^(?:oui|ok|okay|d'accord|ça marche|safi|wakha|iyeh|yes)\b", re.I)
_AGENDA_RE = re.compile(
    r"\b(?:aujourd['’]hui|lyouma|ce soir|ordre du jour)\b.*\b(?:parler|discuter|aborder|nhedro|nhdro|ntkellmo|nchoufo)\b", re.I)
_WHO_QUESTION = re.compile(r"\bqui (?:est-ce qui )?(?:s['’]occupe de |se charge de |va |peut )?(?P<task>[^?]+)\?", re.I)
_NAMED_SUBJECT = re.compile(
    r"\b([A-ZÀ-Ö][a-zà-ÿ]+(?: [A-ZÀ-Ö][a-zà-ÿ]+)?) (?:va|doit|devra|s'occupe|s’occupe|se charge|prépare|enverra|fera|prendra)\b"
)
_VOCATIVE = re.compile(r"^(?:\w+[, ]+)?([A-ZÀ-Ö][a-zà-ÿ]+)\s*,|,\s*([A-ZÀ-Ö][a-zà-ÿ]+)\s*[?.!]?$")

_STRIP_PATTERNS = [
    r"^(?:(?:ok|okay|d'accord|bon|alors|donc|et|du coup|parfait|super|oui|très bien|merci(?: à tous| beaucoup)?)[,.!]?\s+)+",
    r"^[A-ZÀ-Ö][a-zà-ÿ]+\s*,\s*",                                   # vocatif en tête
    r",\s*[A-ZÀ-Ö][a-zà-ÿ]+\s*(?=[?.!]?$)",                           # vocatif en fin
    r"^(?:il faut (?:que|qu['’])\s*\w+\s+)", r"^il faut\s+(?=\w)", r"^(?:aussi|également|ensuite|puis)\s+", r"^(?:w|ou|et)\s+(?=ana\b|ghadi\b|ma\b)", r"^ma tnsa\w*\s+",
    r"^(?:ana |nta |nti )?(?:ghadi |ghan)(?=n?\w)", r"^khass(?:ek|k|ni|na|kom)\s+",
    r"^(?:est-ce que |est-ce qu['’])",
    r"^(?:moi )?(?:je|tu|vous|il|elle|on|[A-ZÀ-Ö][a-zà-ÿ]+(?: [A-ZÀ-Ö][a-zà-ÿ]+)?)\s+"
    r"(?:vais|vas|va|allons|allez|peux|peut|pourrais|pourrait|pourriez|pouvez|dois|doit|devez|devra|"
    r"(?:m['’]|t['’]|s['’]|vous )occupe[sz]? (?=d)|(?:me |te |se |vous )charge[sz]? (?=d))\s*",
    r"^(?:me |te |nous |vous |lui |leur )(?=\w)",
    r"^(?:tu|vous)\s+(?=\w)", r"^(?:je|j['’])\s*(?:te |vous |lui |leur )?(?=\w)",
    r"^(?:merci de|pensez à|pense à|n['’]oublie pas de|n['’]oubliez pas de)\s+",
    r"^(?:i will|i'll|can you|could you|please)\s+",
]
_STRIP_RE = [re.compile(p, re.I) for p in _STRIP_PATTERNS]


_VOCATIVE_PREFIX = re.compile(r"^.*?\b[A-ZÀ-Ö][a-zà-ÿ]+\s*,\s*(?=(?:tu|vous|est-ce que)\b)")


def _clean_task(text: str, deadline: str | None) -> str:
    task = text.strip()
    # « Pour le résumé, Rania, tu peux préparer… » → « tu peux préparer… »
    task = _VOCATIVE_PREFIX.sub("", task)
    if deadline:
        task = re.sub(r"[,\s]*" + re.escape(deadline) + r"[,\s]*", " ", task, flags=re.I)
    for _ in range(3):
        for r in _STRIP_RE:
            task = r.sub("", task).strip()
    task = re.sub(r",\s*(?:je|j['’]|on|tu|il|elle)\b.*$", "", task)   # « …, je peux la finir »
    task = re.sub(r"\s*[?.!]+$", "", task).strip(" ,")
    task = re.sub(r"\s{2,}", " ", task)
    task = _to_infinitive(task)
    return task[:1].upper() + task[1:] if task else text


# « lance l'évaluation » → « Lancer l'évaluation » (verbes fréquents dans les réunions)
_IRREGULAR = {"fait": "faire", "fais": "faire", "met": "mettre", "mets": "mettre", "prend": "prendre",
              "prends": "prendre", "écrit": "écrire", "écris": "écrire", "revoit": "revoir", "voit": "voir",
              "envoie": "envoyer", "envoies": "envoyer", "renvoie": "renvoyer", "finit": "finir",
              "finis": "finir", "remplit": "remplir", "relit": "relire", "relis": "relire"}
_ER_VERBS = {"lance", "prépare", "ajoute", "finalise", "rédige", "contacte", "vérifie", "corrige", "appelle",
             "relance", "organise", "présente", "teste", "valide", "installe", "documente", "intègre",
             "planifie", "réserve", "partage", "compare", "annote", "entraîne", "évalue", "déploie",
             "termine", "modifie", "programme", "demande", "regarde", "cherche", "contacte", "transmet",
             "dépose", "réduit", "envoie", "imprime", "achète", "paie", "publie", "affiche", "signe",
             "informe", "confirme", "examine", "rappelle", "récupère", "rédige", "achète", "mesure"}


def _to_infinitive(task: str) -> str:
    if not task:
        return task
    first, _, rest = task.partition(" ")
    low = first.lower()
    # « la page historique » (issu de « je me charge de… ») → « S'occuper de la page historique »
    if low in {"la", "l'", "un", "une"} or low.startswith(("l'", "l’")):
        return f"s'occuper de {task[:1].lower()}{task[1:]}"
    if low == "le":
        return f"s'occuper du {rest}"
    if low == "les":
        return f"s'occuper des {rest}"
    if low in {"du", "des"}:
        return f"s'occuper {task[:1].lower()}{task[1:]}"
    if low in {"de", "d'", "d’"}:
        nxt = rest.split(" ", 1)[0].lower()
        if re.search(r"(?:er|ir|re|oir)$", nxt):  # « de demander l'autorisation » → « demander… »
            return rest
        return f"s'occuper {task[:1].lower()}{task[1:]}"
    if low in _IRREGULAR:
        verb = _IRREGULAR[low]
    elif low in _ER_VERBS or (low.endswith("es") and low[:-1] in _ER_VERBS) or (
            low.endswith("ez") and low[:-2] + "e" in _ER_VERBS):
        stem = low[:-1] if low.endswith("es") else low[:-2] + "e" if low.endswith("ez") else low
        verb = stem + "r"
        verb = "transmettre" if verb.startswith("transmet") else verb
    else:
        return task
    return f"{verb} {rest}".strip()


def _resolve_owner(u, utterances, participants: list[str]) -> str | None:
    text = u.text
    m = _NAMED_SUBJECT.search(text)
    if m and m.group(1).split()[0] not in {"Il", "Elle", "On", "Ça", "Ce", "Qui"}:
        return canonical_person(m.group(1), participants)
    m = _VOCATIVE.search(text)
    if m and (_SECOND_PERSON.search(text) or re.search(r"\bil faut\b", text, re.I)):
        name = m.group(1) or m.group(2)
        if name and name not in {"Ok", "Bon", "Alors", "Donc", "Oui", "Merci", "Mzyan", "Safi", "Wakha", "Iyeh",
                                 "Walakin", "Daba", "Parfait", "Super", "Bien", "Bref", "Voilà", "Akhir"}:
            return canonical_person(name, participants)
    persons = [p for p in find_persons(text, participants) if p != u.speaker]
    first, second = _FIRST_PERSON.search(text), _SECOND_PERSON.search(text)
    if first and (not second or first.start() < second.start()):
        return u.speaker  # « je vous envoie… » : le sujet est « je »
    if second:
        if persons:
            return canonical_person(persons[0], participants)
        # « tu peux… ? » → la personne qui répond juste après
        for nxt in utterances[u.index + 1: u.index + 3]:
            if nxt.speaker and nxt.speaker != u.speaker:
                return nxt.speaker
        return None
    if persons:
        return canonical_person(persons[0], participants)
    return None


def extract_actions(meeting: PreprocessedMeeting, meeting_date: date, min_conf: float = 0.5) -> list[Action]:
    actions: list[Action] = []
    utts = meeting.utterances
    for u in utts:
        # « Oui, je m'en occupe » : confirme l'action précédente au lieu d'en créer une
        if _ACCEPT_RE.search(u.text) and len(u.text.split()) <= 12:
            # « Qui dépose le dossier ? » → « Je peux le faire » : l'action vient de la question
            prev_q = next((p for p in reversed(utts[max(0, u.index - 2): u.index]) if _WHO_QUESTION.search(p.text)), None)
            if prev_q and not (actions and actions[-1].source == prev_q.text):
                q_task = _WHO_QUESTION.search(prev_q.text).group("task")
                dl = find_deadline(prev_q.text) or find_deadline(u.text)
                actions.append(Action(task=_clean_task(q_task, dl), owner=u.speaker, deadline=dl,
                                      deadline_date=resolve_date(dl, meeting_date), source=prev_q.text,
                                      topic_id=u.topic_id, confidence=0.8))
                continue
            if actions and u.index - _index_of(actions[-1], utts) <= 2:
                if u.speaker:  # sans diarisation (audio), on garde le responsable déjà trouvé
                    actions[-1].owner = u.speaker
                dl = find_deadline(u.text)
                if dl and not actions[-1].deadline:
                    actions[-1].deadline, actions[-1].deadline_date = dl, resolve_date(dl, meeting_date)
            continue
        label, conf = classify(u.text)
        if label != "action" or conf < min_conf or _AGENDA_RE.search(u.text):
            continue
        deadline = find_deadline(u.text)
        # « Oui, je l'appelle demain matin » : précise l'action qu'on vient de confier à ce locuteur
        if (_FOLLOWUP_RE.match(u.text) and actions and actions[-1].owner == u.speaker
                and u.index - _index_of(actions[-1], utts) <= 2
                and len(u.text.split()) <= 7):
            if deadline:
                actions[-1].deadline, actions[-1].deadline_date = deadline, resolve_date(deadline, meeting_date)
            continue
        owner = _resolve_owner(u, utts, meeting.speakers)
        if owner is None and deadline is None and conf < 0.8:
            continue  # contrainte générale (« on doit rester sous une minute »), pas une tâche
        actions.append(Action(
            task=_clean_task(u.text, deadline),
            owner=owner,
            deadline=deadline,
            deadline_date=resolve_date(deadline, meeting_date),
            source=u.text,
            topic_id=u.topic_id,
            confidence=round(conf, 2),
        ))
    return actions


def _index_of(action: Action, utts) -> int:
    for u in utts:
        if u.text == action.source:
            return u.index
    return -99
