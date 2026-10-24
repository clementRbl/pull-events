"""
Accès à la base vectorielle PostgreSQL + pgvector.

Les chunks vivent dans une table dédiée (une colonne par métadonnée, une
colonne `embedding vector(1024)` indexée en HNSW). Une reconstruction écrit
dans une table de travail, puis l'échange avec la table servie en une seule
transaction : l'API continue de répondre pendant toute l'indexation.
"""

import logging

import sqlalchemy
from langchain_core.embeddings import Embeddings
from langchain_postgres import Column, PGEngine, PGVectorStore
from langchain_postgres.v2.indexes import HNSWIndex, HNSWQueryOptions

from scripts.config import Settings

logger = logging.getLogger(__name__)

# Métadonnées stockées comme vraies colonnes SQL (filtrables en SQL)
METADATA_COLUMNS = [
    Column("uid", "TEXT"),
    Column("title", "TEXT"),
    Column("city", "TEXT"),
    Column("location_name", "TEXT"),
    Column("date_start", "TEXT"),
    Column("date_end", "TEXT"),
    Column("url", "TEXT"),
    Column("chunk_index", "INTEGER"),
    Column("total_chunks", "INTEGER"),
]


class VectorStoreUnavailableError(RuntimeError):
    """La base vectorielle est injoignable ou la table n'existe pas encore."""


def _staging_table(settings: Settings) -> str:
    return f"{settings.vector_table}_staging"


def _pg_engine(settings: Settings) -> PGEngine:
    return PGEngine.from_connection_string(url=settings.database_url)


def _table_exists(settings: Settings, table_name: str) -> bool:
    engine = sqlalchemy.create_engine(settings.database_url)
    try:
        return sqlalchemy.inspect(engine).has_table(table_name)
    finally:
        engine.dispose()


def open_store(settings: Settings, embeddings: Embeddings) -> PGVectorStore:
    """Ouvre la table servie par l'API.

    Raises:
        VectorStoreUnavailableError: Base injoignable ou index jamais construit.
    """
    try:
        table_ready = _table_exists(settings, settings.vector_table)
    except sqlalchemy.exc.OperationalError as e:
        msg = "PostgreSQL injoignable (vérifiez DATABASE_URL et le conteneur db)"
        raise VectorStoreUnavailableError(msg) from e

    if not table_ready:
        msg = (
            f"Table {settings.vector_table} absente. "
            "Lancez d'abord : uv run python -m scripts.build_index"
        )
        raise VectorStoreUnavailableError(msg)

    return PGVectorStore.create_sync(
        engine=_pg_engine(settings),
        embedding_service=embeddings,
        table_name=settings.vector_table,
        metadata_columns=[column.name for column in METADATA_COLUMNS],
        # Taille de la liste de candidats explorée par HNSW : plus grande,
        # la recherche approchée retrouve plus souvent les vrais plus proches.
        index_query_options=HNSWQueryOptions(ef_search=settings.hnsw_ef_search),
    )


def create_staging_store(settings: Settings, embeddings: Embeddings) -> PGVectorStore:
    """Crée une table de travail vide pour une nouvelle indexation."""
    engine = _pg_engine(settings)
    engine.init_vectorstore_table(
        table_name=_staging_table(settings),
        vector_size=settings.embedding_size,
        metadata_columns=METADATA_COLUMNS,
        overwrite_existing=True,
    )
    return PGVectorStore.create_sync(
        engine=engine,
        embedding_service=embeddings,
        table_name=_staging_table(settings),
        metadata_columns=[column.name for column in METADATA_COLUMNS],
    )


def publish_staging_store(settings: Settings, store: PGVectorStore) -> None:
    """Indexe la table de travail en HNSW puis la met en service.

    L'index HNSW est construit avant l'échange, pour que la table servie
    soit toujours indexée.
    """
    staging = _staging_table(settings)
    served = settings.vector_table

    logger.info("Construction de l'index HNSW sur %s...", staging)
    store.apply_vector_index(HNSWIndex(name=f"{staging}_hnsw"))

    # Renommer aussi la clé primaire et l'index, sinon leurs noms entreraient
    # en conflit avec ceux de la prochaine table de travail.
    swap_statements = [
        f"DROP TABLE IF EXISTS {served}",
        f"ALTER TABLE {staging} RENAME TO {served}",
        f"ALTER INDEX {staging}_pkey RENAME TO {served}_pkey",
        f"ALTER INDEX {staging}_hnsw RENAME TO {served}_hnsw",
    ]
    engine = sqlalchemy.create_engine(settings.database_url)
    try:
        with engine.begin() as connection:
            for statement in swap_statements:
                connection.execute(sqlalchemy.text(statement))
    finally:
        engine.dispose()
    logger.info("Table %s mise en service", served)
