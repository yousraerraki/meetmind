# ---------- API MeetMind (FastAPI) ----------
FROM python:3.11-slim

# ffmpeg : nécessaire à Whisper pour lire les fichiers audio
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements-nlp.txt requirements-docker-nlp.txt ./
RUN pip install --no-cache-dir -r requirements.txt psycopg2-binary

# Modèles NLP (Whisper, BARThez, spaCy, Sentence-BERT) — version CPU allégée (~2,5 Go) :
#   $env:INSTALL_NLP="true"; docker compose up --build        (PowerShell)
#   INSTALL_NLP=true docker compose up --build                (Linux / Mac)
# torch CPU d'abord (sinon pip installe la version CUDA de plusieurs Go), puis le reste.
ARG INSTALL_NLP=false
RUN if [ "$INSTALL_NLP" = "true" ]; then \
      pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
      pip install --no-cache-dir -r requirements-docker-nlp.txt && \
      python -m spacy download fr_core_news_md && \
      python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)" ; \
    fi
# Modèle spaCy « md » dans Docker (45 Mo au lieu de 570 Mo pour « lg »)
ENV SPACY_MODEL=fr_core_news_md

COPY backend ./backend
COPY data ./data
COPY training ./training
COPY evaluation ./evaluation
RUN mkdir -p uploads models

# Classifieur TF-IDF (décisions / actions) entraîné pendant le build sur le corpus fourni.
# Un CamemBERT fine-tuné placé dans ./models/camembert-meetmind reste prioritaire.
RUN python training/train_tfidf.py --output /app/builtin_models/tfidf-meetmind.joblib \
      --data data/corpus/annotations_train.csv data/annotations/seed.jsonl data/annotations/reunions_exemple.csv
ENV TFIDF_MODEL_PATH=/app/builtin_models/tfidf-meetmind.joblib

EXPOSE 8000
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
