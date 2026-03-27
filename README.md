# Puls-Events — Chatbot RAG

POC d'un chatbot intelligent basé sur le **Retrieval-Augmented Generation (RAG)** pour recommander des événements culturels.

## Stack technique

| Composant | Technologie |
|-----------|------------|
| Orchestrateur RAG | LangChain |
| Modèle NLP | Mistral (via API) |
| Base vectorielle | Faiss (faiss-cpu) |
| API REST | FastAPI |
| Source de données | API Open Agenda |

| Conteneurisation | Docker |

## Installation

### Option 1 — Docker (recommandé)

```bash
# Build de l'image
docker build -t puls-events .

# Lancer le conteneur (clé API passée au runtime)
docker run -p 8000:8000 -e MISTRAL_API_KEY=votre_clé_ici puls-events
```

L'API est accessible sur `http://localhost:8000/docs`.

> **Note** : L'index Faiss doit être présent dans `data/faiss_index/` avant le build, ou reconstruit via `/rebuild` après le lancement.

Pour monter un index existant :
```bash
docker run -p 8000:8000 \
  -e MISTRAL_API_KEY=votre_clé_ici \
  -v $(pwd)/data:/app/data \
  puls-events
```

### Option 2 — Installation locale

### 1. Cloner le projet

```bash
git clone <url-du-repo>
cd pull-events
```

### 2. Créer un environnement virtuel

```bash
python -m venv env
source env/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Configurer les variables d'environnement

```bash
cp .env.example .env
# Éditer .env et renseigner votre clé API Mistral
```

### 5. Vérifier l'installation

```bash
python scripts/check_imports.py
```

## Lancer l'API

```bash
uvicorn api.main:app --reload
```

L'API est accessible sur `http://localhost:8000`.

### Documentation Swagger

Une documentation interactive est générée automatiquement :
- **Swagger UI** : http://localhost:8000/docs
- **ReDoc** : http://localhost:8000/redoc

### Endpoints

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| GET | `/health` | Vérifier l'état de l'API |
| POST | `/ask` | Poser une question au chatbot |
| POST | `/rebuild` | Reconstruire l'index Faiss |

#### Exemple — POST /ask

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Quels concerts à Paris ce weekend ?"}'
```

Réponse :
```json
{
  "answer": "Voici les concerts prévus à Paris...",
  "sources": [
    {
      "title": "Concert Jazz au Sunset",
      "city": "Paris",
      "date_start": "2026-03-28",
      "date_end": "2026-03-28",
      "url": "https://openagenda.com/...",
      "excerpt": "Un concert exceptionnel..."
    }
  ]
}
```

## Tests

```bash
pytest tests/ -v
```

## Évaluation RAG (Ragas)

```bash
python scripts/evaluate_rag.py
```

Le script évalue le pipeline RAG sur un jeu de test annoté (`data/test_questions.json`) et calcule les métriques : fidélité, pertinence de la réponse, précision du contexte.

## Pipeline de données

```bash
# 1. Ingestion (API Open Agenda → data/raw_events.json)
python scripts/fetch_events.py

# 2. Nettoyage + chunking (→ data/chunks.json)
python scripts/prepare_data.py

# 3. Indexation Faiss (→ data/faiss_index/)
python scripts/build_index.py
```

## Structure du projet

```
pull-events/
├── api/                       # API REST FastAPI
│   └── main.py                # Endpoints /ask, /rebuild, /health
├── scripts/                   # Scripts du pipeline
│   ├── fetch_events.py        # Ingestion Open Agenda
│   ├── prepare_data.py        # Nettoyage + chunking
│   ├── build_index.py         # Indexation Faiss
│   ├── rag_chain.py           # Chaîne RAG (retriever + LLM)
│   ├── evaluate_rag.py        # Évaluation Ragas
│   └── check_imports.py       # Vérification de l'environnement
├── tests/                     # Tests unitaires
│   └── test_api.py            # Tests de l'API
├── data/                      # Données (non versionnées sauf test)
│   └── test_questions.json    # Jeu de test annoté
├── docs/                      # Documentation
├── Dockerfile                 # Conteneurisation
├── .dockerignore
├── .env.example               # Template des variables d'environnement
├── .gitignore
├── README.md
└── requirements.txt
```
