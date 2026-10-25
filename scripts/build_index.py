"""
Construction de l'index vectoriel dans PostgreSQL + pgvector.

Charge les chunks textuels préparés, les vectorise avec les embeddings
du fournisseur configuré (Mistral ou Azure OpenAI) et les écrit dans une
table de travail. La table n'est mise en service qu'une fois complète et
indexée (HNSW).

Usage :
    uv run python -m scripts.build_index
    # variante de découpage dans sa propre table, en reprenant les vecteurs
    # des chunks identiques déjà calculés dans event_chunks :
    CHUNKING=phrase-recoupement uv run python -m scripts.prepare_data
    VECTOR_TABLE=event_chunks_phrase_recoupement \\
        uv run python -m scripts.build_index --reuse-from event_chunks
"""

import argparse
import json
import logging
import time
from pathlib import Path

from langchain_core.documents import Document
from langchain_postgres import PGVectorStore

from scripts.config import Settings, get_settings
from scripts.rag_chain import get_embeddings
from scripts.vector_db import (
    create_staging_store,
    fetch_known_embeddings,
    publish_staging_store,
)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Constantes
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHUNKS_FILE = DATA_DIR / "chunks.json"

# Taille des lots pour l'envoi à l'API (éviter les timeouts)
BATCH_SIZE = 50

# Délai entre les lots (secondes) pour respecter le rate-limiting
BATCH_DELAY = 1.0

# Nombre maximum de tentatives en cas d'erreur API
MAX_RETRIES = 5


def load_chunks() -> list[dict]:
    """Charge les chunks depuis le fichier JSON.

    Returns:
        Liste des chunks avec texte et métadonnées.

    Raises:
        FileNotFoundError: Si le fichier chunks.json n'existe pas.
    """
    if not CHUNKS_FILE.exists():
        msg = (
            f"Fichier {CHUNKS_FILE} introuvable. "
            "Lancez d'abord : python scripts/prepare_data.py"
        )
        raise FileNotFoundError(msg)

    with open(CHUNKS_FILE, encoding="utf-8") as f:
        chunks = json.load(f)

    logger.info("Chargement de %d chunks", len(chunks))
    return chunks


def chunks_to_documents(chunks: list[dict]) -> list[Document]:
    """Convertit les chunks en documents LangChain.

    Chaque document contient le texte du chunk et ses métadonnées,
    format attendu par les vectorstores LangChain.

    Args:
        chunks: Liste des chunks {text, metadata}.

    Returns:
        Liste de Documents LangChain.
    """
    documents = []
    for chunk in chunks:
        doc = Document(
            page_content=chunk["text"],
            metadata=chunk["metadata"],
        )
        documents.append(doc)

    logger.info("Conversion en %d documents LangChain", len(documents))
    return documents


def _retry_with_backoff(func, *args, **kwargs):
    """Exécute une fonction avec retry et backoff exponentiel.

    En cas d'erreur (rate-limiting, timeout), réessaie jusqu'à
    MAX_RETRIES fois avec un délai croissant.

    Args:
        func: Fonction à exécuter.
        *args: Arguments positionnels.
        **kwargs: Arguments nommés.

    Returns:
        Résultat de la fonction.

    Raises:
        Exception: Si toutes les tentatives échouent.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt == MAX_RETRIES:
                logger.error("Échec après %d tentatives : %s", MAX_RETRIES, e)
                raise
            wait_time = BATCH_DELAY * (2**attempt)
            logger.warning(
                "Erreur API (tentative %d/%d) : %s — retry dans %.0fs",
                attempt,
                MAX_RETRIES,
                e,
                wait_time,
            )
            time.sleep(wait_time)
    return None


def index_documents(
    store: PGVectorStore,
    documents: list[Document],
    settings: Settings,
    reuse_from: str | None = None,
) -> None:
    """Vectorise et insère les documents par lots dans la table de travail.

    Gère le rate-limiting de l'API d'embeddings avec retry et délai entre les lots.
    Avec ``reuse_from``, un chunk dont le texte existe déjà dans cette table
    reprend son vecteur au lieu d'être renvoyé à l'API.
    """
    total_batches = (len(documents) + BATCH_SIZE - 1) // BATCH_SIZE
    logger.info(
        "Indexation de %d documents (lots de %d, %d lots au total)...",
        len(documents),
        BATCH_SIZE,
        total_batches,
    )

    reused_count = 0
    for batch_num, start in enumerate(range(0, len(documents), BATCH_SIZE), 1):
        batch = documents[start : start + BATCH_SIZE]

        known = {}
        if reuse_from:
            texts = [doc.page_content for doc in batch]
            known = fetch_known_embeddings(settings, reuse_from, texts)
        reused = [doc for doc in batch if doc.page_content in known]
        to_embed = [doc for doc in batch if doc.page_content not in known]

        if reused:
            store.add_embeddings(
                texts=[doc.page_content for doc in reused],
                embeddings=[known[doc.page_content] for doc in reused],
                metadatas=[doc.metadata for doc in reused],
            )
            reused_count += len(reused)
        if to_embed:
            _retry_with_backoff(store.add_documents, to_embed)
            # Pause entre les lots pour respecter le rate-limiting
            time.sleep(BATCH_DELAY)
        logger.info("Lot %d/%d indexé", batch_num, total_batches)

    logger.info(
        "%d vecteurs repris de %s, %d calculés par l'API",
        reused_count,
        reuse_from,
        len(documents) - reused_count,
    )


def test_search(store: PGVectorStore) -> None:
    """Vérifie l'index avec quelques recherches de similarité."""
    test_queries = [
        "Quels concerts sont prévus à Paris ?",
        "Expositions à Versailles",
        "Spectacle pour enfants en Île-de-France",
    ]

    logger.info("Test de recherche de similarité")
    for query in test_queries:
        results = store.similarity_search(query, k=3)
        logger.info("Requête : '%s'", query)
        for i, doc in enumerate(results, 1):
            logger.info(
                "  Résultat %d : %s (%s)",
                i,
                doc.metadata.get("title", "?"),
                doc.metadata.get("city", "?"),
            )


def main():
    """Point d'entrée : construction de l'index vectoriel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reuse-from",
        default=None,
        help=(
            "Table déjà indexée avec le même modèle d'embeddings : les chunks "
            "au texte identique y reprennent leur vecteur"
        ),
    )
    args = parser.parse_args()

    settings = get_settings()
    documents = chunks_to_documents(load_chunks())

    store = create_staging_store(settings, get_embeddings(settings))
    index_documents(store, documents, settings, args.reuse_from)
    test_search(store)
    publish_staging_store(settings, store)

    logger.info("Indexation terminée avec succès")


if __name__ == "__main__":
    main()
