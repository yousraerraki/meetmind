"""Classification de phrases : DECISION / ACTION / INFORMATION.

Modèle principal : CamemBERT fine-tuné (voir training/train_classifier.py).
Fallback : règles linguistiques (marqueurs d'accord, de choix final, d'engagement).
Les règles servent aussi à pré-annoter les données d'entraînement.
"""
from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path

from ..config import model_loader, settings

log = logging.getLogger(__name__)

LABELS = ["information", "decision", "action"]
MODEL_MIN_CONFIDENCE = float(__import__("os").getenv("MODEL_MIN_CONFIDENCE", "0.6"))

# ---- marqueurs de décision : accord ou choix final
DECISION_PATTERNS = [
    r"\bon (?:a |l'a |)(?:décid|tranch|valid|retenu|choisi|opté|adopté|acté|arrêté)",
    r"\b(?:c'est|c’est) (?:décidé|validé|acté|tranché|entendu|approuvé|bon pour moi|réglé)",
    r"\b(?:il est|il a été) (?:décidé|convenu|validé|retenu|acté)",
    r"\bdécision\b", r"\bon part (?:sur|là-dessus)", r"\bon garde\b", r"\bon retient\b",
    r"\bon (?:valide|choisit|adopte|opte|abandonne|maintient|reporte|annule|fixe|fige)\b",
    r"\b(?:nous avons|on a) (?:convenu|décidé|validé|choisi|retenu)",
    r"\b(?:tout le monde est|on est tous|nous sommes) d['’]accord",
    r"\bvalidé\b", r"\badopté\b", r"\bapprouvé\b", r"\bunanimité\b",
    r"\bla (?:solution|option) retenue\b", r"\bfinalement,? on\b",
    r"\bwe (?:decided|agreed|chose|will go with)\b", r"\bit'?s (?:decided|agreed)\b",
    r"\b(?:donc|alors|du coup|bon|ok)[,.]? on (?!a\b|est\b|peut|pourrait|devrait|doit|va\b|n['’]a\b|verra|en parle)\w+",
    r"\bon en reparle (?:en|le|à|au|la)\b", r"\b(?:on (?:le |les |la |)reporte|est reporté|sera reporté)\b",
    r"\bla décision est prise\b",
    r"\bnous (?:fixons|validons|choisissons|retenons|gardons|adoptons|décidons|maintenons|reportons|annulons|abandonnons|optons)\b",
    r"\b(?:tfahmna|ttafa9na|9rrna|qrrna)\b",  # darija
]
# Une proposition n'est pas encore une décision
_PROPOSAL_RE = re.compile(r"\b(?:je propose|on pourrait|il faudrait|je pense qu|peut-être|je suggère)\b", re.I)
# ---- marqueurs d'action : engagement, demande, tâche à réaliser
ACTION_PATTERNS = [
    r"\bje (?:te |vous |lui |leur |le |la |les |l['’]|t['’])?(?:vais|m'occupe|m’occupe|me charge|prépare|fais|ferai|enverrai|envoie|rédige|contacte|"
    r"regarde|vérifie|mets|mettrai|corrige|finalise|appelle|relance|peux|dois|reprends|prends)\b",
    r"\b(?:tu|vous) (?:peux|pouvez|pourrais|pourriez|dois|devez|vas|allez|t'occupes|vous occupez|"
    r"te charges|vous chargez|prépares|préparez|envoies|envoyez)\b",
    r"\b(?:il|elle|on) (?:va|doit|devra|s'occupe|s’occupe|se charge)\b",
    r"\bil faut (?:que|qu['’])", r"\bil faut (?!que\b|qu['’]|\w+ (?:pas|plus)\b)\w+(?:er|ir|re)\b", r"\b(?:à faire|action|tâche)\s*:", r"\bpense(?:z)? à\b",
    r"\b(?:merci de|n'oublie pas|n’oublie pas|n'oubliez pas)\b", r"\bqui s'occupe\b",
    r"\b(?:i will|i'll|you will|can you|could you|please)\b",
    r"\b(?:ghadi n|ghan|khassek|khassk|khassni|khassna|khasskom)\w*",  # darija
    r"\bma tnsa\w*\b", r"\b(?:sifet|sifti|dir|diri|kteb|ktbi|3ayet|3ayti) (?:lia|lina|liha|lih)\b",
]
_DEC_RE = [re.compile(p, re.IGNORECASE) for p in DECISION_PATTERNS]
_ACT_RE = [re.compile(p, re.IGNORECASE) for p in ACTION_PATTERNS]
# Sujet nommé (« Rania va… ») : sensible à la casse pour ne pas confondre avec « ça me va »
_ACT_RE.append(re.compile(
    r"\b(?<![.!?] )(?!(?:Il|Elle|On|Ça|Ce|Qui|Cela|Tout|Ca)\b)[A-ZÀ-Ö][a-zà-ÿ]+ "
    r"(?:va|doit|devra|s'occupe|s’occupe|se charge|prépare|enverra|fera)\b"
))
# Questions-tags : « on valide le 9, tout le monde est d'accord ? » reste une décision
_TAG_QUESTION = re.compile(
    r"[,\s]*(?:tout le monde est d['’]accord|on est d['’]accord|ça vous va|ça va pour (?:tout le monde|vous)|"
    r"c['’]est bon pour (?:tout le monde|vous)|d['’]accord|ok|non|right|ok for everyone)\s*\?\s*$",
    re.IGNORECASE,
)
_TIME_HINT = re.compile(
    r"\b(?:avant|d'ici|d’ici|demain|lundi|mardi|mercredi|jeudi|vendredi|semaine prochaine|fin du mois|week-end|"
    r"ghedda|9bel|had simana|\d{1,2}/\d{1,2}|janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\b",
    re.IGNORECASE,
)
_QUESTION_RE = re.compile(r"\?\s*$")


def rule_classify(sentence: str) -> tuple[str, float]:
    sentence = _TAG_QUESTION.sub(".", sentence)
    dec = sum(bool(r.search(sentence)) for r in _DEC_RE)
    act = sum(bool(r.search(sentence)) for r in _ACT_RE)
    if act and _TIME_HINT.search(sentence):
        act += 1
    # Les questions ouvertes (« on valide ? ») ne sont pas des décisions
    if _PROPOSAL_RE.search(sentence):
        dec = 0
    if _QUESTION_RE.search(sentence):
        dec = 0
        # « Leila, tu prépares… ? » / « tu peux… ? » = demande d'action ; « on envoie un communiqué ? » = question
        if not re.search(r"\b(?:tu|vous|can you|could you)\b", sentence, re.I):
            act = 0
    if dec == 0 and act == 0:
        return "information", 0.6
    if act > dec:
        return "action", min(0.55 + 0.15 * act, 0.95)
    return "decision", min(0.55 + 0.15 * dec, 0.95)


# ------------------------------------------------------------- CamemBERT
@model_loader
def _load_camembert():
    path = Path(settings.classifier_model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Modèle CamemBERT fine-tuné introuvable : {path}. "
            "Entraînez-le avec : python training/train_classifier.py"
        )
    from transformers import pipeline
    log.info("Chargement du CamemBERT fine-tuné depuis %s…", path)
    return pipeline("text-classification", model=str(path), tokenizer=str(path), truncation=True)


@model_loader
def _load_tfidf():
    path = Path(settings.tfidf_model_path)
    if not path.exists():
        raise FileNotFoundError(f"Modèle TF-IDF introuvable : {path} (python training/train_tfidf.py)")
    import joblib
    log.info("Chargement du classifieur TF-IDF depuis %s…", path)
    return joblib.load(path)


def _active_model() -> str | None:
    """Ordre de préférence : CamemBERT fine-tuné > TF-IDF + régression logistique > règles."""
    if settings.nlp_mode == "light":
        return None
    for name, loader in (("camembert", _load_camembert), ("tfidf", _load_tfidf)):
        try:
            loader()
            return name
        except Exception as exc:
            log.debug("%s indisponible : %s", name, exc)
    if settings.nlp_mode == "full":
        raise RuntimeError("Aucun classifieur entraîné (CamemBERT ou TF-IDF) n'est disponible.")
    return None


def classifier_backend() -> str:
    model = _active_model()
    if model == "camembert":
        return f"CamemBERT fine-tuné ({Path(settings.classifier_model_path).name}) + règles si incertain"
    if model == "tfidf":
        return "TF-IDF + régression logistique + règles si incertain"
    return "règles linguistiques (fallback)"


def _predict_model(sentence: str) -> tuple[str, float] | None:
    model = _active_model()
    if model == "camembert":
        pred = _load_camembert()(sentence)[0]
        label = pred["label"].lower()
        if label.startswith("label_"):
            label = LABELS[int(label.split("_")[1])]
        return label, float(pred["score"])
    if model == "tfidf":
        clf = _load_tfidf()
        proba = clf.predict_proba([sentence])[0]
        best = int(proba.argmax())
        return str(clf.classes_[best]), float(proba[best])
    return None


@lru_cache(maxsize=4096)
def classify(sentence: str) -> tuple[str, float]:
    """Retourne (label, confiance). Résultat mis en cache : les modules
    décisions (4b) et actions (4c) partagent la même prédiction."""
    try:
        pred = _predict_model(sentence)
    except Exception:
        if settings.nlp_mode == "full":
            raise
        pred = None
    if pred is None:
        return rule_classify(sentence)
    label, score = pred
    # Modèle hésitant → on s'appuie sur les règles (utile tant que le jeu annoté est petit)
    if score < MODEL_MIN_CONFIDENCE:
        rule_label, rule_conf = rule_classify(sentence)
        if rule_label != label:
            return rule_label, rule_conf
    return label, score
