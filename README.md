# Puls-Events — Chatbot RAG

Chatbot **RAG** (Retrieval-Augmented Generation) qui recommande des événements culturels en Île-de-France à partir des données Open Agenda.

La version 1.1 remplace l'index Faiss par **PostgreSQL + pgvector** et ajoute **Langfuse** : chaque question est tracée avec sa latence, ses tokens et son coût.

La version 1.2 tourne aussi sur **Azure** : Container Apps, Azure OpenAI (Foundry), PostgreSQL Flexible Server + pgvector, Key Vault. Voir [docs/azure/deploiement.md](docs/azure/deploiement.md).

## Stack technique

| Composant | Technologie |
|-----------|------------|
| Orchestrateur RAG | LangChain |
| LLM et embeddings | Mistral (`ministral-8b-latest`, `mistral-embed`) en local ; Azure OpenAI (`gpt-4.1-mini`, `text-embedding-3-small`) sur Azure |
| Base vectorielle | PostgreSQL 17 + pgvector (index HNSW, distance cosinus) |
| Observabilité LLM | Langfuse v4 (auto-hébergé en local, Langfuse Cloud UE pour Azure) |
| API REST | FastAPI |
| Évaluation | Ragas (20 questions annotées), juge Claude Haiku 5.5 |
| Source de données | API Open Agenda (OpenDataSoft) |
| Dépendances | uv (`pyproject.toml` + `uv.lock`) |
| Conteneurisation | Docker Compose ; image publiée sur GHCR |
| Cloud | Azure Container Apps, Azure OpenAI (Foundry), PostgreSQL Flexible Server, Key Vault |

## Architecture du système

```mermaid
graph TB
    subgraph "Sources de données"
        OA["API Open Agenda<br/>(OpenDataSoft)"]
    end

    subgraph "Pipeline d'ingestion"
        FE["fetch_events.py<br/>Pagination curseur date<br/>Filtre : IDF, < 1 an"]
        PD["prepare_data.py<br/>Nettoyage HTML<br/>Déduplication<br/>Chunking 2000 chars"]
        BI["build_index.py<br/>Embeddings Mistral<br/>Batch 50 + retry<br/>Table de travail puis échange"]
    end

    subgraph "Base vectorielle - Docker Compose"
        PG["PostgreSQL 17 + pgvector<br/>table event_chunks<br/>~39 000 vecteurs x 1024 dim<br/>index HNSW (cosinus)"]
    end

    subgraph "Chaîne RAG - rag_chain.py"
        RET["Recherche vectorielle<br/>top_k = 5"]
        PROMPT["Prompt Template<br/>Système + Utilisateur"]
        LLM["Mistral LLM<br/>ministral-8b-latest<br/>temperature=0.3"]
    end

    subgraph "Observabilité - Docker Compose"
        LF["Langfuse v4 auto-hébergé<br/>trace par question<br/>latence, tokens, coût"]
    end

    subgraph "API REST - FastAPI"
        HEALTH["/health GET"]
        ASK["/ask POST"]
        REBUILD["/rebuild POST"]
        SWAGGER["Swagger /docs"]
    end

    subgraph "Évaluation"
        EVAL["evaluate_rag.py<br/>Ragas + coûts<br/>20 questions"]
        TEST["pytest<br/>tests unitaires"]
    end

    OA -->|"HTTP JSON"| FE
    FE -->|"raw_events.json"| PD
    PD -->|"chunks.json"| BI
    BI -->|"mistral-embed API"| PG

    ASK -->|"question"| RET
    RET -->|"5 chunks"| PROMPT
    PROMPT --> LLM
    LLM -->|"réponse + sources"| ASK

    REBUILD -->|"relance pipeline"| FE
    RET -.->|"SQL <=> cosinus"| PG
    RET -.->|"span retrieval"| LF
    LLM -.->|"generation : tokens + coût"| LF

    EVAL -.->|"métriques"| ASK
    TEST -.->|"pytest"| ASK
```

## Installation

Prérequis : Docker avec Docker Compose, et [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/clementRbl/pull-events.git
cd pull-events
cp .env.example .env      # puis renseigner MISTRAL_API_KEY
uv sync                   # crée .venv à partir de uv.lock
```

### Lancement rapide

```bash
./launch.sh           # base + Langfuse, pipeline complet, puis API
./launch.sh --api     # base + Langfuse, puis API (index déjà construit)
```

### Tout en Docker Compose

```bash
docker compose up -d                          # db, api, Langfuse
docker compose exec api python -m scripts.fetch_events
docker compose exec api python -m scripts.prepare_data
docker compose exec api python -m scripts.build_index
```

| Service | Adresse |
|---------|---------|
| API (Swagger) | http://localhost:8000/docs |
| Langfuse | http://localhost:3000 (compte : `admin@pull-events.local` / `local-dev-password`) |
| PostgreSQL + pgvector | `localhost:5432` (base `pullevents`) |

Les identifiants Langfuse et Postgres du compose sont des valeurs de développement local. Pour tout autre usage, surchargez-les dans `.env` (liste dans `.env.example`).

## Configuration

Tout ce qui change d'un environnement à l'autre passe par des variables d'environnement (`scripts/config.py`). Le même code tourne en local ou sur un PostgreSQL managé :

| Variable | Défaut | Rôle |
|----------|--------|------|
| `LLM_PROVIDER` | `mistral` | Fournisseur du LLM et des embeddings : `mistral` ou `azure_openai` |
| `MISTRAL_API_KEY` | — | Clé API Mistral (obligatoire avec `mistral`) |
| `AZURE_OPENAI_ENDPOINT` / `AZURE_OPENAI_API_KEY` / `AZURE_OPENAI_API_VERSION` | — / — / `2024-10-21` | Accès Azure OpenAI. `LLM_MODEL` et `EMBEDDING_MODEL` désignent alors les noms des déploiements |
| `DATABASE_URL` | `postgresql+psycopg://pullevents:pullevents@localhost:5432/pullevents` | Base pgvector |
| `VECTOR_TABLE` | `event_chunks` | Table des chunks |
| `LLM_MODEL` / `EMBEDDING_MODEL` / `EMBEDDING_SIZE` | `ministral-8b-latest` / `mistral-embed` / `1024` | Modèles (changer d'embeddings impose de réindexer) |
| `TOP_K` | `5` | Nombre de chunks envoyés au LLM |
| `HNSW_EF_SEARCH` | `100` | Candidats explorés par l'index HNSW (rappel@5 : 0,91 à 40, 0,99 à 100) |
| `LLM_INPUT_PRICE_PER_MTOK` / `LLM_OUTPUT_PRICE_PER_MTOK` | `0.10` / `0.10` | Prix ($ par million de tokens) pour le coût par réponse |
| `JUDGE_PROVIDER` / `JUDGE_MODEL` / `ANTHROPIC_API_KEY` | `anthropic` / `claude-haiku-5-5` | Juge de l'évaluation Ragas (dev uniquement) |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_BASE_URL` | clés du Langfuse local | Traçage. Sans clés, l'API marche mais ne trace rien |
| `API_KEY` | — | Si définie, `/ask` et `/rebuild` exigent l'en-tête `X-API-Key`. Sans elle, l'API est ouverte (avertissement au démarrage) |
| `ENABLE_REBUILD` | `false` | `true` autorise `/rebuild` (poste local) |
| `LLM_MAX_TOKENS` / `LLM_TIMEOUT_SECONDS` | `1024` / `60` | Plafonds par réponse (tokens générés, délai d'attente du LLM et des embeddings) |
| `LANGFUSE_TRACE_CONTENT` | `true` | `false` masque questions, prompts et réponses dans Langfuse ; tokens, coûts et latences restent |

## Base vectorielle : PostgreSQL + pgvector

Les chunks sont stockés dans une vraie table SQL (`langchain-postgres`, `PGVectorStore`) :

```
event_chunks
├── langchain_id   uuid (clé primaire)
├── content        text           -- texte du chunk
├── embedding      vector(1024)   -- index HNSW, vector_cosine_ops
├── title, city, location_name, url, uid, date_start, date_end   text
└── chunk_index, total_chunks     integer
```

Les métadonnées sont des colonnes, et non un blob JSON : on peut filtrer en SQL (par ville, par date) et inspecter la base avec n'importe quel client PostgreSQL.

**Reconstruction sans coupure.** `build_index.py` écrit dans `event_chunks_staging`, construit l'index HNSW, puis échange les deux tables en une seule transaction. Pendant les ~15 minutes d'indexation, l'API continue de répondre avec l'ancienne table.

## Observabilité : Langfuse

Chaque appel à `/ask` produit une trace dans Langfuse :

```
ask (span)
├── retrieval (retriever)   -- titres des chunks, durée embedding / recherche
└── llm-chat (generation)   -- prompt, réponse, tokens entrée/sortie, coût en $
```

Le coût est calculé dans l'application à partir des tokens renvoyés par le LLM et des prix configurés, puis envoyé à Langfuse (`cost_details`). L'évaluation et Langfuse utilisent donc la même grille de prix.

**Les questions des utilisateurs sont tracées** dans Langfuse, avec les prompts et les réponses. `LANGFUSE_TRACE_CONTENT=false` remplace tous les textes par `[masqué]` (paramètre `mask` du client Langfuse) et ne garde que les mesures. Les journaux de l'API n'affichent plus les questions (niveau DEBUG).

## Endpoints

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| GET | `/health` | Vérifier l'état de l'API |
| POST | `/ask` | Poser une question au chatbot |
| POST | `/rebuild` | Reconstruire l'index vectoriel (sans coupure de service). Désactivé par défaut |

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Quels concerts à Paris ce weekend ?"}'
```

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

Codes d'erreur : `401` clé d'API absente ou fausse (si `API_KEY` est définie), `403` reconstruction désactivée, `409` reconstruction déjà en cours, `422` question invalide, `429` limite de débit dépassée, `503` base vectorielle indisponible, `500` configuration manquante. Les messages d'erreur restent génériques : le détail est dans les journaux du serveur.

Limites de débit (`slowapi`, en mémoire) : 10 questions par minute et par IP, 30 par minute et par clé d'API, une reconstruction par jour. En local, l'API n'écoute que sur `127.0.0.1` (`launch.sh` et Docker Compose).

Sur Azure, ajouter l'en-tête `X-API-Key` (la clé est dans le Key Vault, voir [docs/azure/deploiement.md](docs/azure/deploiement.md#appeler-lapi)).

## Tests

```bash
uv run pytest -v
```

- `tests/test_api.py` : endpoints, erreurs génériques, Swagger, clé d'API, limite de débit, `/rebuild` désactivé par défaut et verrouillé
- `tests/test_scripts.py` : nettoyage, chunking, ingestion
- `tests/test_rag.py` : configuration par variables d'environnement, choix du fournisseur (Mistral ou Azure OpenAI), calcul du coût, chaîne RAG instrumentée (LLM simulé), échange de tables pgvector sur une vraie base (ignoré si `docker compose up -d db` n'est pas lancé)

## Évaluation : qualité et coûts

```bash
uv run python -m scripts.evaluate_rag --backend pgvector
```

Le script pose les 20 questions de `data/test_questions.json` (`--max-questions` pour en prendre moins), puis les fait noter par un juge d'un autre fournisseur (Claude Haiku 5.5, pour qu'un modèle ne note pas ses propres réponses). Il affiche une estimation du coût du juge avant de le lancer (environ 0,006 $ par question). Les embeddings du juge restent `mistral-embed` quel que soit le fournisseur évalué, pour que les scores restent comparables. Il calcule :

| Métrique | Ce qu'elle mesure |
|----------|------------------|
| **Faithfulness** | La réponse reste fidèle aux chunks (pas d'hallucination) |
| **Answer Relevancy** | La réponse est pertinente par rapport à la question |
| **Context Precision** | Les chunks récupérés sont les bons (avec référence) |
| Latence | Embedding de la question, recherche vectorielle, génération |
| Tokens et coût | Par question, et extrapolé à 1 000 questions |

Le résumé et le détail par question sont écrits dans `docs/evaluation/results_<backend>.json`.

### Résultats avant/après

Même jeu de 20 questions et même juge partout. Les trois premières colonnes et la colonne A utilisent les mêmes vecteurs Mistral ; la colonne B réindexe tout avec Azure OpenAI. Détail, méthode et limites : [docs/evaluation/README.md](docs/evaluation/README.md).

| | Faiss (v1) | pgvector, `ef_search=40` | pgvector, `ef_search=100` | Azure A : Mistral | Azure B : Azure OpenAI |
|---|---|---|---|---|---|
| Faithfulness | 0,871 | 0,843 | 0,884 | 0,878 | **0,932** |
| Answer relevancy | 0,826 | 0,837 | 0,831 | 0,843 | **0,878** |
| Context precision | 0,582 | 0,518 | 0,518 | 0,527 | **0,707** |
| Mêmes 5 sources que Faiss | — | 16 / 20 | 18 / 20 | 18 / 20 | 0 / 20 |
| Recherche vectorielle (moyenne) | 14,8 ms | 4,3 ms | 5,6 ms | 832 ms | 206 ms |
| Latence totale (moyenne / p95) | 4,31 s / 8,18 s | 3,06 s / 4,38 s | 3,47 s / 6,16 s | 3,89 s / 7,08 s | 2,87 s / 4,49 s |
| Coût LLM pour 1 000 questions | 0,17 $ | 0,17 $ | 0,17 $ | 0,18 $ | 1,24 $ |

En résumé :
- **Faiss → pgvector : qualité inchangée** : les écarts restent dans le bruit du juge, mesuré sur les questions où les chunks récupérés étaient identiques (± 0,06 en faithfulness).
- **Le vrai réglage pgvector est `ef_search`** : à 40 (défaut pgvector), l'index HNSW rate 9 % des plus proches voisins ; à 100, 1 % pour 0,4 ms de plus.
- **Local → Azure : l'infrastructure seule ne change rien** (colonne A) : mêmes sources pour les 20 questions, scores dans le bruit.
- **Azure OpenAI améliore les trois scores** (colonne B), surtout la précision du contexte (+0,18), pour un coût 7 fois plus élevé (1,24 $ pour 1 000 questions).
- **L'API déployée répond en 2,6 s en moyenne** (p95 4,2 s), mesuré en HTTP depuis Paris sur les 20 questions.
- **Le serveur PostgreSQL B1ms est trop petit pour garder les index en mémoire** : la recherche passe de 5 ms à plusieurs centaines de millisecondes, encore loin derrière les 2,6 s de génération.
- **Évaluer coûte plus cher que répondre** : environ 0,12 $ de juge par évaluation, contre 0,0034 $ pour les 20 réponses.

## Pipeline de données

```bash
uv run python -m scripts.fetch_events   # API Open Agenda → data/raw_events.json
uv run python -m scripts.prepare_data   # nettoyage + chunking → data/chunks.json
uv run python -m scripts.build_index    # embeddings → PostgreSQL + pgvector
```

L'ancien index Faiss reste reconstructible pour comparer (dépendances de dev) :

```bash
uv run python -m scripts.faiss_baseline build    # → data/faiss_index/
uv run python -m scripts.faiss_baseline export   # copie les vecteurs Faiss dans pgvector, sans nouvel appel Mistral
uv run python -m scripts.evaluate_rag --backend faiss
```

## Limites connues et améliorations possibles

- **Fraîcheur des données** : l'index se reconstruit en entier (~39 000 embeddings) via `/rebuild`. Une mise à jour incrémentale (par `uid`) réduirait coût et durée.
- **Questions temporelles** : le modèle ne connaît pas la date du jour (« ce weekend »). Un filtre SQL sur `date_start` réglerait ce point.
- **Pas d'historique de conversation** : chaque question est indépendante.
- **Jeu d'évaluation réduit** : 20 questions, avec des références génériques. Assez pour comparer deux versions, pas pour une mesure absolue.
- **Prix** : les tarifs Mistral par défaut sont à vérifier sur la page officielle et à ajuster dans `.env`.
- **Limite de débit en mémoire** : valable avec un seul réplica (c'est le cas sur Azure, 0 à 1). Au-delà, il faudrait un stockage partagé (Redis).
- **`/rebuild` synchrone** : la requête attend la fin du pipeline. Un traitement en tâche de fond (202 puis suivi) serait plus propre ; la route est de toute façon désactivée hors poste local.
- **Chunks longs** : les descriptions sont découpées par ligne, et le HTML nettoyé n'en contient presque plus. Certains chunks font donc près de 10 000 caractères (4 800 tokens au plus après normalisation Unicode).

## Structure du projet

```
pull-events/
├── api/main.py                # Endpoints /ask, /rebuild, /health
├── scripts/
│   ├── config.py              # Configuration par variables d'environnement, prix
│   ├── fetch_events.py        # Ingestion Open Agenda
│   ├── prepare_data.py        # Nettoyage + chunking
│   ├── build_index.py         # Embeddings → pgvector
│   ├── vector_db.py           # Table pgvector, index HNSW, échange de tables
│   ├── rag_chain.py           # Chaîne RAG instrumentée (Langfuse)
│   ├── evaluate_rag.py        # Ragas + latence + coûts
│   ├── faiss_baseline.py      # Index Faiss de référence (dev)
│   └── check_imports.py       # Vérification de l'environnement
├── tests/                     # pytest
├── data/test_questions.json   # Jeu de test annoté (20 questions)
├── docs/
│   ├── azure/                 # Ressources Azure, commandes, coût, destruction ; fiche AI-103
│   ├── evaluation/            # Méthode, résultats avant/après (JSON par configuration)
│   ├── article-blog.md        # Brouillon d'article
│   └── architecture.mmd
├── docker-compose.yml         # pgvector, API, Langfuse
├── Dockerfile                 # Image API (uv)
├── launch.sh
├── pyproject.toml / uv.lock
└── .env.example
```
