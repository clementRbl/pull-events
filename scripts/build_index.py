"""
Construction de l'index vectoriel Faiss.

Charge les chunks textuels préparés, les vectorise avec les
embeddings Mistral, et construit un index Faiss pour la
recherche de similarité.
"""

import json
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_mistralai import MistralAIEmbeddings

load_dotenv()

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Constantes
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHUNKS_FILE = DATA_DIR / "chunks.json"
INDEX_DIR = DATA_DIR / "faiss_index"

# Modèle d'embeddings Mistral
EMBEDDING_MODEL = "mistral-embed"

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
    format attendu par le vectorstore FAISS de LangChain.

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


def get_embeddings() -> MistralAIEmbeddings:
    """Initialise le modèle d'embeddings Mistral.

    Returns:
        Instance MistralAIEmbeddings configurée.

    Raises:
        ValueError: Si la clé API Mistral n'est pas définie.
    """
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        msg = "MISTRAL_API_KEY non définie. Configurez-la dans le fichier .env"
        raise ValueError(msg)

    return MistralAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=api_key,
    )


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


def build_faiss_index(documents: list[Document]) -> FAISS:
    """Construit l'index Faiss à partir des documents.

    Vectorise les documents par lots et les indexe dans Faiss
    via l'intégration LangChain. Gère le rate-limiting avec
    retry et délai entre les lots.

    Args:
        documents: Liste de Documents LangChain à indexer.

    Returns:
        Instance FAISS contenant l'index vectoriel.
    """
    embeddings = get_embeddings()
    total_batches = (len(documents) // BATCH_SIZE) + 1

    logger.info(
        "Indexation de %d documents (lots de %d, %d lots au total)...",
        len(documents),
        BATCH_SIZE,
        total_batches,
    )

    # Premier lot pour initialiser l'index
    first_batch = documents[:BATCH_SIZE]
    vectorstore = _retry_with_backoff(FAISS.from_documents, first_batch, embeddings)
    logger.info("Lot 1/%d indexé", total_batches)

    # Lots suivants ajoutés à l'index existant
    for i in range(BATCH_SIZE, len(documents), BATCH_SIZE):
        # Pause entre les lots pour respecter le rate-limiting
        time.sleep(BATCH_DELAY)

        batch = documents[i : i + BATCH_SIZE]
        _retry_with_backoff(vectorstore.add_documents, batch)

        batch_num = (i // BATCH_SIZE) + 1
        logger.info(
            "Lot %d/%d indexé (%d documents traités)",
            batch_num,
            total_batches,
            i + len(batch),
        )

    logger.info("Index Faiss construit avec succès")
    return vectorstore


def save_index(vectorstore: FAISS) -> Path:
    """Sauvegarde l'index Faiss sur disque.

    Args:
        vectorstore: Instance FAISS à sauvegarder.

    Returns:
        Chemin du dossier de l'index.
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(INDEX_DIR))
    logger.info("Index sauvegardé dans %s", INDEX_DIR)
    return INDEX_DIR


def test_search(vectorstore: FAISS):
    """Teste l'index avec une recherche de similarité simple.

    Effectue une requête test pour valider que l'index fonctionne
    correctement et retourne des résultats pertinents.

    Args:
        vectorstore: Instance FAISS à tester.
    """
    test_queries = [
        "Quels concerts sont prévus à Paris ?",
        "Expositions à Versailles",
        "Spectacle pour enfants en Île-de-France",
    ]

    logger.info("Test de recherche de similarité")
    for query in test_queries:
        results = vectorstore.similarity_search(query, k=3)
        logger.info("\nRequête : '%s'", query)
        for i, doc in enumerate(results):
            logger.info(
                "  Résultat %d : %s (%s) — %s",
                i + 1,
                doc.metadata.get("title", "?"),
                doc.metadata.get("city", "?"),
                doc.page_content[:100],
            )


def main():
    """Point d'entrée : construction de l'index vectoriel."""
    # Charger les chunks
    chunks = load_chunks()

    # Convertir en documents LangChain
    documents = chunks_to_documents(chunks)

    # Construire l'index Faiss
    vectorstore = build_faiss_index(documents)

    # Sauvegarder l'index
    save_index(vectorstore)

    # Tester avec des recherches simples
    test_search(vectorstore)

    logger.info("Indexation terminée avec succès")


if __name__ == "__main__":
    main()
