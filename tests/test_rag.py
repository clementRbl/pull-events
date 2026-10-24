"""Tests de la configuration, de la chaîne RAG instrumentée et de pgvector."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from scripts.config import get_settings, llm_cost_usd
from scripts.rag_chain import RagComponents, answer_question, ask

# === Configuration par variables d'environnement ===


def test_settings_read_from_environment():
    """Base et modèles se changent sans toucher au code (déploiement Azure)."""
    env = {
        "DATABASE_URL": "postgresql+psycopg://user:pw@azure-host:5432/rag",
        "LLM_MODEL": "gpt-4o-mini",
        "TOP_K": "8",
    }
    with patch.dict("os.environ", env):
        settings = get_settings()

    assert settings.database_url == env["DATABASE_URL"]
    assert settings.llm_model == "gpt-4o-mini"
    assert settings.top_k == 8


def test_settings_reject_unsafe_table_name():
    """VECTOR_TABLE est inséré dans du SQL : un nom piégé est refusé."""
    with patch.dict("os.environ", {"VECTOR_TABLE": "chunks; DROP TABLE users"}):
        with pytest.raises(ValueError, match="VECTOR_TABLE"):
            get_settings()


def test_settings_reject_unknown_judge_provider():
    """Seuls les fournisseurs prévus pour le juge Ragas sont acceptés."""
    with patch.dict("os.environ", {"JUDGE_PROVIDER": "openai"}):
        with pytest.raises(ValueError, match="JUDGE_PROVIDER"):
            get_settings()


def test_llm_cost_uses_configured_prices():
    """Le coût suit les prix par million de tokens configurés."""
    env = {"LLM_INPUT_PRICE_PER_MTOK": "1.0", "LLM_OUTPUT_PRICE_PER_MTOK": "3.0"}
    with patch.dict("os.environ", env):
        settings = get_settings()

    cost = llm_cost_usd(settings, input_tokens=2_000, output_tokens=500)
    assert cost == pytest.approx(0.002 + 0.0015)


# === Chaîne RAG (Mistral et base vectorielle simulés) ===


def _fake_components() -> RagComponents:
    """Composants RAG sans réseau : embeddings, base et LLM simulés."""
    with patch.dict("os.environ", {"TOP_K": "2"}):
        settings = get_settings()

    embeddings = MagicMock()
    embeddings.embed_query.return_value = [0.1] * 1024

    vectorstore = MagicMock()
    vectorstore.similarity_search_by_vector.return_value = [
        Document(
            page_content="Concert de jazz au Sunset",
            metadata={"title": "Jazz", "city": "Paris", "url": "https://x/1"},
        ),
        Document(page_content="Expo photo", metadata={"title": "Photo"}),
    ]

    llm = MagicMock()
    llm.invoke.return_value = AIMessage(
        content="Un concert de jazz a lieu au Sunset.",
        usage_metadata={
            "input_tokens": 1200,
            "output_tokens": 80,
            "total_tokens": 1280,
        },
    )
    return RagComponents(settings, embeddings, vectorstore, llm)


def test_answer_question_searches_once_and_measures_cost():
    """Une seule recherche par question, tokens et coût relevés."""
    components = _fake_components()

    result = answer_question("Du jazz à Paris ?", components)

    components.embeddings.embed_query.assert_called_once_with("Du jazz à Paris ?")
    components.vectorstore.similarity_search_by_vector.assert_called_once_with(
        [0.1] * 1024, k=2
    )
    assert result.answer == "Un concert de jazz a lieu au Sunset."
    assert (result.input_tokens, result.output_tokens) == (1200, 80)
    expected_cost = llm_cost_usd(components.settings, 1200, 80)
    assert result.cost_usd == pytest.approx(expected_cost)
    assert result.total_seconds >= 0


def test_answer_question_puts_chunks_in_prompt():
    """Les chunks trouvés sont bien transmis au LLM comme contexte."""
    components = _fake_components()

    answer_question("Du jazz à Paris ?", components)

    messages = components.llm.invoke.call_args.args[0]
    assert "Concert de jazz au Sunset" in messages[-1].content
    assert "Du jazz à Paris ?" in messages[-1].content


def test_ask_keeps_api_contract():
    """ask() renvoie toujours {answer, sources} avec les champs attendus."""
    with patch("scripts.rag_chain.get_components", return_value=_fake_components()):
        result = ask("Du jazz à Paris ?")

    assert set(result) == {"answer", "sources"}
    first_source = result["sources"][0]
    assert first_source["title"] == "Jazz"
    assert first_source["city"] == "Paris"
    # Métadonnée absente : chaîne vide, comme avant la migration
    assert result["sources"][1]["city"] == ""


# === pgvector (base réelle, test ignoré si PostgreSQL est éteint) ===


def _postgres_settings():
    """Paramètres d'une table de test, ou skip si la base est injoignable."""
    import sqlalchemy

    with patch.dict("os.environ", {"VECTOR_TABLE": "test_event_chunks"}):
        settings = get_settings()
    try:
        engine = sqlalchemy.create_engine(settings.database_url)
        with engine.connect():
            pass
        engine.dispose()
    except sqlalchemy.exc.OperationalError:
        pytest.skip("PostgreSQL indisponible (docker compose up -d db)")
    return settings


def test_rebuild_twice_swaps_tables_without_name_conflict():
    """Deux reconstructions de suite : la table servie est remplacée proprement."""
    from langchain_core.embeddings import DeterministicFakeEmbedding

    from scripts.vector_db import (
        create_staging_store,
        open_store,
        publish_staging_store,
    )

    settings = _postgres_settings()
    embeddings = DeterministicFakeEmbedding(size=settings.embedding_size)

    for title in ["Première version", "Seconde version"]:
        store = create_staging_store(settings, embeddings)
        store.add_texts([f"{title} du concert"], metadatas=[{"title": title}])
        publish_staging_store(settings, store)

    served = open_store(settings, embeddings)
    results = served.similarity_search("concert", k=5)
    assert [doc.metadata["title"] for doc in results] == ["Seconde version"]
