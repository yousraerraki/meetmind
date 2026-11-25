# MeetMind — Assistant intelligent de comptes rendus de réunion

Projet NLP · **Yousra Erraki** · 2026-2027

MeetMind transforme automatiquement la transcription (ou l'audio) d'une réunion en un **compte rendu structuré** :
résumé global et par sujet, **décisions prises**, **actions à réaliser** (tâche · responsable · échéance).

| Ce qui est livré | Où |
|---|---|
| API FastAPI (pipeline 6 étapes, historique, édition, export, e-mail, suivi des actions) | `backend/` |
| Interface **React** (principale) | `frontend-react/` |
| Interface **Streamlit** (prototype) | `frontend/streamlit_app.py` |
| Lancement complet **Docker + PostgreSQL** | `docker-compose.yml` |
| **Notebook Colab** : fine-tuning CamemBERT, comparaison BARThez/mT5, évaluation | `notebooks/MeetMind_Colab.ipynb` |
| Diarisation audio (qui parle ?) | `backend/app/nlp/diarization.py` |
| 4 réunions d'exemple annotées + 285 phrases étiquetées | `data/` |
| Scripts d'entraînement et d'évaluation | `training/`, `evaluation/` |
| Tests (pipeline + API, SQLite et PostgreSQL) | `tests/` |

---

## 1. Démarrage

### Option A — Docker (le plus simple : tout en une commande)

```bash
docker compose up --build
```
- Interface React : **http://localhost:8080**
- API / Swagger : http://localhost:8000/docs
- Base PostgreSQL incluse (volume persistant)

Options :
```bash
docker compose --profile streamlit up --build      # + interface Streamlit sur :8501
docker compose --profile llm up --build            # + Ollama (résumé Mistral en local) : docker compose exec ollama ollama pull mistral
INSTALL_NLP=true docker compose up --build         # image API avec Whisper, BARThez, spaCy (CPU, ~2,5 Go)
# PowerShell : $env:INSTALL_NLP="true"; docker compose up --build
```

### Option B — En local (développement)

```bash
python -m venv .venv && source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                     # NLP_MODE=light pour démarrer sans modèles

uvicorn backend.app.main:app --reload                    # terminal 1 — API      :8000
cd frontend-react && npm install && npm run dev          # terminal 2 — React    :5173
# ou : streamlit run frontend/streamlit_app.py           #             Streamlit :8501
```

Dans l'interface : **Nouvelle réunion → « Charger un exemple » → Générer le compte rendu**.

## 2. Architecture (section 8 du cahier des charges)

```
1. Entrée (React / Streamlit)   texte · .txt · .docx · .vtt/.srt (Teams, Zoom) · audio .mp3/.wav
        │                                     ┌── backend FastAPI ───────────────────────────────┐
2. Transcription  Whisper (+ pyannote)        │ nlp/transcription.py · nlp/diarization.py
3. Prétraitement  nettoyage, locuteurs, sujets│ nlp/preprocessing.py
        ├──────────────┬──────────────┐       │  exécutés EN PARALLÈLE (ThreadPoolExecutor)
4a. Résumé        4b. Décisions   4c. Actions │ nlp/summarizer.py · decisions.py · actions.py + ner.py
        └──────────────┴──────────────┘       │
5. Compte rendu   fusion en un document       │ nlp/pipeline.py
6a. Stockage      PostgreSQL / SQLite         │ db.py
6b. Export        PDF · Word · e-mail         │ export/
                                              └──────────────────────────────────────────────────┘
```

| Tâche NLP | Modèle (mode complet) | Fallback (sans modèle) |
|---|---|---|
| Speech-to-Text | Whisper | — |
| Diarisation | pyannote 3.1 | une ligne par segment |
| Résumé abstractif | BARThez / mT5 / BART, ou Mistral·Llama 3 via Ollama | TextRank extractif |
| Classification de phrases | CamemBERT fine-tuné (+ règles si le modèle hésite) | règles linguistiques |
| NER responsables / dates | spaCy `fr_core_news_lg` (PER) + règles de dates | participants + règles |
| Segmentation thématique | Sentence-BERT + clustering hiérarchique contigu | TF-IDF + TextTiling |

- **Dates :** les modèles spaCy français n'ont pas d'étiquette `DATE`. `ner.py` détecte **et résout** les échéances
  (« vendredi », « d'ici la fin du mois », « le 20 novembre », darija « ghedda », « 9bel jem3a »…) en vraies dates.
- **Darija :** marqueurs de décision (*ttafa9na, 9rrna*), d'action (*ghadi n…, khassek…*) et de dates.
- **Confidentialité :** tout tourne en local (Ollama pour le LLM) ; l'audio est supprimé après transcription.
- Chaque compte rendu indique les modèles réellement utilisés (`GET /health`, onglet *Transcription*).

## 3. Mode complet (vrais modèles)

```bash
pip install -r requirements-nlp.txt
python -m spacy download fr_core_news_lg
# + ffmpeg pour Whisper
```
Dans `.env` : `NLP_MODE=auto` (modèles si disponibles, sinon fallback) ou `full` (erreur si un modèle manque).

- `SUMMARIZER=barthez` (défaut) · `mt5` · `bart` (anglais) · `llm` (Ollama) · `extractive`
- `WHISPER_MODEL=small` (`medium` = plus précis)
- Diarisation : `DIARIZATION=on` + `HF_TOKEN=…` (accepter les conditions de `pyannote/speaker-diarization-3.1`).
  Dans l'interface, renommez les locuteurs : `SPEAKER_00=Yousra, SPEAKER_01=Rania`.

## 4. Entraînement de CamemBERT → notebook Colab

Ouvrez `notebooks/MeetMind_Colab.ipynb` dans Google Colab (GPU T4), chargez `meetmind.zip`, exécutez tout.
Le notebook : installe tout, **fine-tune CamemBERT**, produit le **tableau comparatif TextRank / BARThez / mT5**,
évalue le pipeline complet, teste Whisper sur un audio, et télécharge le modèle entraîné
(à dézipper dans `models/camembert-meetmind`, chargé automatiquement par l'API).

En ligne de commande :
```bash
python training/prepare_data.py vos_transcriptions/*.txt -o data/annotations/mes_reunions.csv   # pré-annotation
# → corriger la colonne "label" à la main (decision | action | information)
python training/train_classifier.py --data data/annotations/seed.jsonl data/annotations/reunions_exemple.csv data/annotations/mes_reunions.csv
```
Données fournies : `seed.jsonl` (180 phrases écrites à la main, FR/EN/darija) + `reunions_exemple.csv`
(105 phrases des réunions d'exemple, vérifiées). **Il faut ajouter 300–500 phrases de vraies réunions.**

## 5. Données et évaluation

**Corpus synthétique** (`data/corpus/`, à déclarer comme tel dans le rapport) :
10 réunions d'entraînement (146 phrases étiquetées) et **5 réunions de test** jamais utilisées
pendant le développement, avec résumé, décisions et actions de référence.

| Configuration (5 réunions de test) | ROUGE-L | F1 décisions | F1 actions |
|---|---|---|---|
| Règles + TextRank | 0,357 ✅ | 0,53 | 0,32 |
| TF-IDF + règles si incertain (défaut) | 0,357 ✅ | 0,65 | 0,49 |
| TF-IDF seul (`MODEL_MIN_CONFIDENCE=0`) | 0,357 ✅ | 0,46 | 0,81 ✅ |

Analyse complète : **`evaluation/RESULTATS.md`**.

```bash
python training/train_tfidf.py --data data/corpus/annotations_train.csv data/annotations/seed.jsonl data/annotations/reunions_exemple.csv --eval data/corpus/annotations_test.csv
python evaluation/evaluate.py data/corpus/test/*.gold.json --sentences data/corpus/annotations_test.csv
python evaluation/compare_summarizers.py data/corpus/test/*.gold.json --models extractive barthez mt5
```

Classifieur utilisé par l'API, dans l'ordre : **CamemBERT fine-tuné** (`models/camembert-meetmind`)
> **TF-IDF + régression logistique** (entraîné automatiquement dans l'image Docker) > règles.

## 6. API

| Méthode | Route | Rôle |
|---|---|---|
| POST | `/meetings/analyze` | importer (texte / fichier / audio) → compte rendu |
| GET | `/meetings?q=` | historique + recherche (titre, résumé, transcription, décisions, actions, responsables) |
| GET / PUT / DELETE | `/meetings/{id}` | consulter / corriger / supprimer |
| GET | `/meetings/{id}/export/{pdf\|docx\|txt}` | export |
| POST | `/meetings/{id}/email` | envoi aux participants (SMTP dans `.env`) |
| GET | `/actions?owner=&status=` | suivi des actions (toutes réunions) |
| PATCH | `/actions/{id}` | statut : à faire / en cours / fait |
| GET | `/health` | modèles actifs |

## 7. Tests

```bash
pytest -q                                                        # SQLite
DATABASE_URL=postgresql+psycopg2://user@localhost/meetmind pytest -q tests/test_api.py   # PostgreSQL
```

## 8. Ce qui reste à faire par le binôme

1. **Annoter 300–500 phrases de vraies réunions** et lancer le notebook (fine-tuning CamemBERT).
2. Remplir le **tableau comparatif** BARThez / mT5 (notebook, section 3) et choisir le modèle de résumé.
3. Évaluer sur un **jeu de test séparé** et reporter les scores dans le rapport.
4. Tester Whisper + diarisation sur de vrais enregistrements (temps < 1 min pour 1 h ?).

## 9. Arborescence

```
backend/app/        main.py (API) · config.py · db.py
  nlp/              transcription · diarization · preprocessing · summarizer · classifier · ner
                    decisions · actions · pipeline · schemas · textutils
  export/           documents.py (PDF, Word, texte) · mailer.py
frontend-react/     interface React (Vite) + Dockerfile nginx
frontend/           interface Streamlit
notebooks/          MeetMind_Colab.ipynb
training/           prepare_data.py · train_classifier.py
evaluation/         evaluate.py · compare_summarizers.py · résultats JSON
data/samples/       4 réunions (projet, marketing, association, darija) + références *.gold.json
data/annotations/   seed.jsonl · reunions_exemple.csv
tests/              tests du pipeline et de l'API
Dockerfile · Dockerfile.streamlit · docker-compose.yml
```
