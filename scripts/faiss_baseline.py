"""
Index Faiss de référence, gardé uniquement pour la comparaison avant/après.

L'API utilise PostgreSQL + pgvector. Ce script (dépendances de dev) sert à :
- `build`  : reconstruire l'ancien index Faiss dans data/faiss_index/ ;
- `export` : copier ses vecteurs dans pgvector sans repayer les embeddings.

L'évaluation charge cet index avec `--backend faiss`.

Usage :
    uv run python -m scripts.faiss_baseline build
    uv run python -m scripts.faiss_baseline export
"""

import argparse
import logging
import time
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

from scripts.build_index import (
    BATCH_DELAY,
    BATCH_SIZE,
    _retry_with_backoff,
    chunks_to_documents,
    load_chunks,
)
from scripts.config import get_settings
from scripts.rag_chain import get_embeddings
from scripts.vector_db import create_staging_store, publish_staging_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

INDEX_DIR = Path(__file__).resolve().parent.parent / "data" / "faiss_index"


def load_faiss_index(embeddings: Embeddings) -> FAISS:
    """Charge l'index Faiss depuis le disque.

    Raises:
        FileNotFoundError: Si l'index n'a pas été construit.
    """
    if not INDEX_DIR.exists():
        msg = (
            f"Index Faiss introuvable dans {INDEX_DIR}. "
            "Lancez d'abord : uv run python -m scripts.faiss_baseline build"
        )
        raise FileNotFoundError(msg)

    # Désérialisation pickle : acceptable car l'index est produit localement
    return FAISS.load_local(
        str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True
    )


def build_faiss_index() -> None:
    """Vectorise les chunks et sauvegarde l'index Faiss (comme la v1)."""
    documents = chunks_to_documents(load_chunks())
    embeddings = get_embeddings(get_settings())

    vectorstore = _retry_with_backoff(
        FAISS.from_documents, documents[:BATCH_SIZE], embeddings
    )
    for start in range(BATCH_SIZE, len(documents), BATCH_SIZE):
        time.sleep(BATCH_DELAY)
        _retry_with_backoff(
            vectorstore.add_documents, documents[start : start + BATCH_SIZE]
        )
        indexed = min(start + BATCH_SIZE, len(documents))
        logger.info("%d/%d documents indexés", indexed, len(documents))

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(INDEX_DIR))
    logger.info("Index Faiss sauvegardé dans %s", INDEX_DIR)


def export_to_pgvector() -> None:
    """Copie les vecteurs Faiss dans pgvector, sans nouvel appel à Mistral."""
    settings = get_settings()
    embeddings = get_embeddings(settings)
    faiss_store = load_faiss_index(embeddings)
    store = create_staging_store(settings, embeddings)

    # Faiss range les vecteurs par position ; le docstore les relie aux textes
    position_to_id = faiss_store.index_to_docstore_id
    total = faiss_store.index.ntotal
    for start in range(0, total, 500):
        positions = range(start, min(start + 500, total))
        documents = [faiss_store.docstore.search(position_to_id[p]) for p in positions]
        store.add_embeddings(
            texts=[doc.page_content for doc in documents],
            embeddings=[faiss_store.index.reconstruct(p).tolist() for p in positions],
            metadatas=[doc.metadata for doc in documents],
        )
        logger.info("%d/%d vecteurs copiés", positions[-1] + 1, total)

    publish_staging_store(settings, store)


def main():
    """Point d'entrée : construction ou export de l'index Faiss."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build", "export"])
    args = parser.parse_args()

    if args.command == "build":
        build_faiss_index()
    else:
        export_to_pgvector()


if __name__ == "__main__":
    main()
