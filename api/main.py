"""
API REST Puls-Events.

Expose le chatbot RAG via FastAPI avec documentation Swagger automatique.
Endpoints : /ask (poser une question) et /rebuild (reconstruire l'index).
"""

import logging
import subprocess
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Ajouter la racine du projet au Python path pour importer les modules de scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.rag_chain import ask  # noqa: E402

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Puls-Events API",
    description=(
        "API de recommandation d'événements culturels en Île-de-France.\n\n"
        "Basée sur un pipeline **RAG** (Retrieval-Augmented Generation) :\n"
        "- Recherche vectorielle Faiss sur 16 000+ événements\n"
        "- Génération de réponses via Mistral LLM\n"
        "- Sources traçables pour chaque réponse"
    ),
    version="1.0.0",
)


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
)
def ask_question(body: QuestionRequest):
    """Pose une question sur les événements culturels en Île-de-France.

    Le système recherche les événements pertinents dans la base vectorielle
    Faiss, puis génère une réponse en langage naturel via Mistral.
    """
    try:
        result = ask(body.question)
        return result
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Index Faiss indisponible : {e}",
        ) from e
    except ValueError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur de configuration : {e}",
        ) from e


@app.post(
    "/rebuild",
    response_model=RebuildResponse,
    tags=["Système"],
    summary="Reconstruire l'index vectoriel",
)
def rebuild_index():
    """Reconstruit l'index Faiss en relançant le pipeline complet.

    Étapes exécutées :
    1. Ingestion des données depuis Open Agenda
    2. Nettoyage et découpage en chunks
    3. Vectorisation et indexation Faiss

    **Attention** : cette opération peut prendre plusieurs minutes.
    """
    scripts_dir = PROJECT_ROOT / "scripts"

    steps = [
        ("Ingestion des données", scripts_dir / "fetch_events.py"),
        ("Préparation des chunks", scripts_dir / "prepare_data.py"),
        ("Construction de l'index", scripts_dir / "build_index.py"),
    ]

    for step_name, script_path in steps:
        logger.info("Rebuild — %s...", step_name)
        result = subprocess.run(  # noqa: S603
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        if result.returncode != 0:
            logger.error("Rebuild échoué (%s) : %s", step_name, result.stderr[:500])
            raise HTTPException(
                status_code=500,
                detail=f"Échec lors de '{step_name}' : {result.stderr[:500]}",
            )

    # Réinitialiser le singleton pour recharger le nouvel index
    import scripts.rag_chain as rag_module

    rag_module._chain = None
    rag_module._retriever = None

    logger.info("Rebuild terminé avec succès")
    return RebuildResponse(
        status="ok",
        message="Index Faiss reconstruit avec succès.",
    )
