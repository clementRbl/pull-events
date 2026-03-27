#!/usr/bin/env bash
# =============================================================
# Puls-Events — Script de lancement local
# =============================================================
# Lance le pipeline complet (ingestion → chunking → indexation)
# puis démarre l'API FastAPI.
#
# Usage :
#   chmod +x launch.sh
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

# Vérifier l'environnement virtuel
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -d "env" ]; then
        echo "Activation de l'environnement virtuel..."
        source env/bin/activate
    else
        echo -e "${RED}Erreur : pas d'environnement virtuel trouvé.${NC}"
        echo "Créez-le avec : python -m venv env && source env/bin/activate && pip install -r requirements.txt"
        exit 1
    fi
fi

# Vérifier le fichier .env
if [ ! -f ".env" ]; then
    echo -e "${RED}Erreur : fichier .env introuvable.${NC}"
    echo "Créez-le avec : cp .env.example .env"
    exit 1
fi

# Mode API seule (sans pipeline)
if [ "$1" = "--api" ]; then
    echo ""
    echo -e "${GREEN}Démarrage de l'API (index existant)...${NC}"
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
    exit 0
fi

# Pipeline complet
echo ""
echo "--- Étape 1/3 : Ingestion des données ---"
python scripts/fetch_events.py

echo ""
echo "--- Étape 2/3 : Nettoyage et chunking ---"
python scripts/prepare_data.py

echo ""
echo "--- Étape 3/3 : Indexation Faiss ---"
python scripts/build_index.py

echo ""
echo -e "${GREEN}Pipeline terminé avec succès !${NC}"
echo ""
echo "--- Démarrage de l'API ---"
echo "Swagger : http://localhost:8000/docs"
echo ""
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
