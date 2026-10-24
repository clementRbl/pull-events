#!/usr/bin/env bash
# =============================================================
# Puls-Events — Script de lancement local
# =============================================================
# Démarre PostgreSQL + pgvector et Langfuse (Docker Compose), lance
# le pipeline complet (ingestion → chunking → indexation) puis l'API.
#
# Usage :
#   ./launch.sh           # Pipeline complet + API
#   ./launch.sh --api     # API seule (index existant)
# =============================================================

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo "=============================================="
echo "  Puls-Events — Lancement local"
echo "=============================================="

# Vérifier uv
if ! command -v uv > /dev/null; then
    echo -e "${RED}Erreur : uv n'est pas installé.${NC}"
    echo "Installation : https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

# Vérifier le fichier .env
if [ ! -f ".env" ]; then
    echo -e "${RED}Erreur : fichier .env introuvable.${NC}"
    echo "Créez-le avec : cp .env.example .env"
    exit 1
fi

echo "Installation des dépendances (uv sync)..."
uv sync

echo "Démarrage de PostgreSQL + pgvector et de Langfuse..."
docker compose up -d --wait db langfuse-web langfuse-worker

# Mode API seule (sans pipeline)
if [ "$1" = "--api" ]; then
    echo ""
    echo -e "${GREEN}Démarrage de l'API (index existant)...${NC}"
    uv run uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
    exit 0
fi

# Pipeline complet
echo ""
echo "--- Étape 1/3 : Ingestion des données ---"
uv run python -m scripts.fetch_events

echo ""
echo "--- Étape 2/3 : Nettoyage et chunking ---"
uv run python -m scripts.prepare_data

echo ""
echo "--- Étape 3/3 : Indexation pgvector ---"
uv run python -m scripts.build_index

echo ""
echo -e "${GREEN}Pipeline terminé avec succès !${NC}"
echo ""
echo "--- Démarrage de l'API ---"
echo "Swagger  : http://localhost:8000/docs"
echo "Langfuse : http://localhost:3000"
echo ""
uv run uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
