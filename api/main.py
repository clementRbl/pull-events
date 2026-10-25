"""
API REST Puls-Events.

Expose le chatbot RAG via FastAPI avec documentation Swagger automatique.
Endpoints : /ask (poser une question) et /rebuild (reconstruire l'index).

Protection, pour un déploiement public (Azure) :
- API_KEY définie : /ask et /rebuild exigent l'en-tête X-API-Key avec cette
  valeur. Sans API_KEY, l'API est ouverte et le signale au démarrage.
- /rebuild est désactivé sauf si ENABLE_REBUILD=true : l'index se construit
  hors de l'API, pour qu'aucun appel ne déclenche une réindexation payante.
- Limites de débit par IP et par clé, une seule reconstruction à la fois.
- Les erreurs renvoient un message générique ; le détail reste dans les logs.
"""

import hashlib
import logging
import os
import secrets
import subprocess
import sys
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

# Ajouter la racine du projet au Python path pour importer les modules de scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.rag_chain import ask, reset_components  # noqa: E402
from scripts.vector_db import VectorStoreUnavailableError  # noqa: E402

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Limites de débit.
ASK_LIMIT_PER_IP = "10/minute"
ASK_LIMIT_PER_KEY = "30/minute"
REBUILD_LIMIT = "1/day"


def client_ip(request: Request) -> str:
    """Adresse IP du client, pour la limite de débit par IP.

    Derrière l'ingress Azure Container Apps (Envoy), l'adresse réelle du
    client est la *dernière* de X-Forwarded-For : Envoy l'ajoute à la fin et
    garde ce que le client a envoyé avant. Prendre la première adresse
    laisserait un client forger l'en-tête pour contourner la limite
    (audit du 25/10/2026, point 3.4). Sans en-tête (poste local), c'est
    l'adresse de la connexion TCP.
    """
    forwarded_for = request.headers.get("X-Forwarded-For", "")
    if forwarded_for:
        return forwarded_for.rsplit(",", 1)[-1].strip()
    if request.client is None:
        return "127.0.0.1"
    return request.client.host


def api_key_bucket(request: Request) -> str:
    """Regroupe les requêtes par clé d'API (empreinte, pas la clé en clair)."""
    provided_key = request.headers.get("X-API-Key", "")
    return hashlib.sha256(provided_key.encode()).hexdigest()[:16]


limiter = Limiter(key_func=client_ip)

# Une seule reconstruction à la fois : deux pipelines en parallèle
# écriraient dans la même table de travail.
rebuild_lock = threading.Lock()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Signale au démarrage une API ouverte à tous."""
    if not os.getenv("API_KEY"):
        logger.warning(
            "API_KEY non définie : /ask est ouvert à tous. "
            "Acceptable sur un poste local uniquement."
        )
    yield


app = FastAPI(
    title="Puls-Events API",
    description=(
        "API de recommandation d'événements culturels en Île-de-France.\n\n"
        "Basée sur un pipeline **RAG** (Retrieval-Augmented Generation) :\n"
        "- Recherche vectorielle PostgreSQL + pgvector sur 39 000+ chunks\n"
        "- Génération de réponses via Mistral ou Azure OpenAI\n"
        "- Sources traçables pour chaque réponse"
    ),
    version="1.2.0",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# Modèles Pydantic


class QuestionRequest(BaseModel):
    """Corps de la requête pour poser une question."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=500,
        examples=["Quels concerts sont prévus à Paris ce weekend ?"],
        description="Question en français sur les événements culturels.",
    )


class Source(BaseModel):
    """Source d'un événement utilisé pour la réponse."""

    title: str
    city: str
    date_start: str
    date_end: str
    url: str
    excerpt: str


class AnswerResponse(BaseModel):
    """Réponse du chatbot avec les sources."""

    answer: str
    sources: list[Source]


class RebuildResponse(BaseModel):
    """Réponse après reconstruction de l'index."""

    status: str
    message: str


# Protection

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(
    provided_key: Annotated[str | None, Depends(api_key_header)],
) -> None:
    """Vérifie la clé d'API quand API_KEY est configurée.

    Sans API_KEY (poste local), l'API reste ouverte comme avant.
    """
    expected_key = os.getenv("API_KEY")
    if not expected_key:
        return
    # compare_digest prend le même temps que la clé soit juste ou fausse :
    # on ne peut pas deviner la clé caractère par caractère en chronométrant.
    if provided_key is None or not secrets.compare_digest(provided_key, expected_key):
        raise HTTPException(status_code=401, detail="Clé d'API absente ou invalide")


def require_rebuild_enabled() -> None:
    """Refuse /rebuild sauf si ENABLE_REBUILD vaut true."""
    if os.getenv("ENABLE_REBUILD", "false").lower() != "true":
        raise HTTPException(
            status_code=403,
            detail="Reconstruction désactivée sur ce déploiement",
        )


# Endpoints


@app.get("/health", tags=["Système"], summary="Vérifier l'état de l'API")
def health_check():
    """Retourne le statut de l'API."""
    return {"status": "ok"}


@app.post(
    "/ask",
    response_model=AnswerResponse,
    tags=["Chatbot"],
    summary="Poser une question au chatbot",
    dependencies=[Depends(require_api_key)],
)
@limiter.limit(ASK_LIMIT_PER_IP)
@limiter.limit(ASK_LIMIT_PER_KEY, key_func=api_key_bucket)
def ask_question(request: Request, body: QuestionRequest):
    """Pose une question sur les événements culturels en Île-de-France.

    Le système recherche les événements pertinents dans la base vectorielle
    pgvector, puis génère une réponse en langage naturel via le LLM configuré.
    """
    try:
        result = ask(body.question)
        return result
    except VectorStoreUnavailableError as e:
        logger.exception("Base vectorielle indisponible")
        raise HTTPException(
            status_code=503,
            detail="Base vectorielle indisponible",
        ) from e
    except ValueError as e:
        logger.exception("Erreur de configuration")
        raise HTTPException(
            status_code=500,
            detail="Erreur de configuration du serveur",
        ) from e


@app.post(
    "/rebuild",
    response_model=RebuildResponse,
    tags=["Système"],
    summary="Reconstruire l'index vectoriel",
    dependencies=[Depends(require_api_key), Depends(require_rebuild_enabled)],
)
@limiter.limit(REBUILD_LIMIT)
def rebuild_index(request: Request):
    """Reconstruit l'index vectoriel en relançant le pipeline complet.

    Étapes exécutées :
    1. Ingestion des données depuis Open Agenda
    2. Nettoyage et découpage en chunks
    3. Vectorisation et indexation pgvector (table de travail, puis
       mise en service : l'API reste disponible pendant la reconstruction)

    **Attention** : cette opération peut prendre plusieurs minutes.
    Désactivée par défaut (ENABLE_REBUILD=true pour l'autoriser).
    """
    if not rebuild_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Reconstruction déjà en cours")
    try:
        _run_rebuild_steps()
    finally:
        rebuild_lock.release()

    # Recharger les composants RAG au prochain appel
    reset_components()

    logger.info("Rebuild terminé avec succès")
    return RebuildResponse(
        status="ok",
        message="Index vectoriel reconstruit avec succès.",
    )


def _run_rebuild_steps() -> None:
    """Lance ingestion, chunking et indexation ; s'arrête à la première erreur."""
    steps = [
        ("Ingestion des données", "scripts.fetch_events"),
        ("Préparation des chunks", "scripts.prepare_data"),
        ("Construction de l'index", "scripts.build_index"),
    ]

    for step_name, module in steps:
        logger.info("Rebuild — %s...", step_name)
        result = subprocess.run(  # noqa: S603
            [sys.executable, "-m", module],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        if result.returncode != 0:
            logger.error("Rebuild échoué (%s) : %s", step_name, result.stderr[-2000:])
            raise HTTPException(
                status_code=500,
                detail=f"Échec lors de '{step_name}' (détail dans les logs)",
            )
