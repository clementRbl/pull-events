"""
Chaîne RAG (Retrieval-Augmented Generation) pour Puls-Events.

Assemble le pipeline complet : recherche vectorielle + génération de
réponse via un LLM (Mistral, ou Azure OpenAI selon LLM_PROVIDER). Chaque
question est tracée dans Langfuse (latence, tokens, coût) quand les clés
Langfuse sont configurées.
Expose une fonction `ask()` réutilisable par l'API et les tests.
"""

import logging
import os
import time
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStore
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langfuse import Langfuse

from scripts.config import (
    Settings,
    get_azure_openai_settings,
    get_mistral_api_key,
    get_settings,
    llm_cost_usd,
)
from scripts.vector_db import open_store

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Prompt système pour guider le LLM
SYSTEM_PROMPT = """\
Tu es un assistant spécialisé dans les événements culturels en Île-de-France.
Tu réponds en français, de manière claire et structurée.

RÈGLES STRICTES :
- Réponds UNIQUEMENT à partir des informations fournies dans le contexte ci-dessous.
- Si le contexte ne contient pas l'information demandée, dis-le honnêtement.
- Ne jamais inventer d'événement, de date, de lieu ou de prix.
- Cite le nom de l'événement, le lieu et les dates quand ils sont disponibles.
- Sois concis mais informatif.
"""

USER_PROMPT = """\
Contexte (événements trouvés dans la base) :
{context}

Question : {question}
"""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", USER_PROMPT),
    ]
)


@dataclass
class RagResult:
    """Résultat détaillé d'une question : réponse, sources et mesures."""

    answer: str
    documents: list[Document]
    input_tokens: int
    output_tokens: int
    cost_usd: float
    embedding_seconds: float
    search_seconds: float
    generation_seconds: float

    @property
    def total_seconds(self) -> float:
        """Latence totale de la question, de l'embedding à la réponse."""
        return self.embedding_seconds + self.search_seconds + self.generation_seconds


@dataclass
class RagComponents:
    """Objets coûteux à créer, construits une seule fois par processus."""

    settings: Settings
    embeddings: Embeddings
    vectorstore: VectorStore
    llm: BaseChatModel


def _tracing_configured() -> bool:
    """Le traçage n'est actif que si les deux clés Langfuse sont fournies."""
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


MASKED_TEXT = "[masqué]"


# Any : signature imposée par le client Langfuse (MaskFunction), qui
# transmet n'importe quelle donnée sérialisable en JSON.
def mask_text_content(*, data: Any, **kwargs: Any) -> Any:
    """Remplace tout texte envoyé à Langfuse, en gardant les nombres.

    Questions, prompts et réponses disparaissent des traces ; tokens,
    coûts et latences restent visibles.
    """
    if isinstance(data, str):
        return MASKED_TEXT
    if isinstance(data, dict):
        return {key: mask_text_content(data=value) for key, value in data.items()}
    if isinstance(data, list):
        return [mask_text_content(data=item) for item in data]
    return data


def _trace_content_enabled() -> bool:
    """LANGFUSE_TRACE_CONTENT=false : ne tracer que les mesures, pas le texte."""
    return os.getenv("LANGFUSE_TRACE_CONTENT", "true").lower() != "false"


# Sans clés, le client Langfuse ne fait rien : l'API fonctionne sans traçage.
langfuse = Langfuse(
    tracing_enabled=_tracing_configured(),
    mask=None if _trace_content_enabled() else mask_text_content,
)


def get_embeddings(settings: Settings) -> Embeddings:
    """Initialise le modèle d'embeddings du fournisseur configuré."""
    if settings.llm_provider == "azure_openai":
        azure = get_azure_openai_settings()
        return AzureOpenAIEmbeddings(
            azure_deployment=settings.embedding_model,
            azure_endpoint=azure.endpoint,
            api_key=azure.api_key,
            api_version=azure.api_version,
            timeout=settings.llm_timeout_seconds,
        )
    return get_mistral_embeddings(
        settings.embedding_model, settings.llm_timeout_seconds
    )


def get_mistral_embeddings(model: str, timeout_seconds: float) -> MistralAIEmbeddings:
    """Initialise un modèle d'embeddings Mistral."""
    return MistralAIEmbeddings(
        model=model,
        api_key=get_mistral_api_key(),
        timeout=int(timeout_seconds),
    )


def get_llm(settings: Settings) -> BaseChatModel:
    """Initialise le LLM de génération du fournisseur configuré."""
    if settings.llm_provider == "azure_openai":
        azure = get_azure_openai_settings()
        return AzureChatOpenAI(
            azure_deployment=settings.llm_model,
            azure_endpoint=azure.endpoint,
            api_key=azure.api_key,
            api_version=azure.api_version,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout_seconds,
        )
    return ChatMistralAI(
        model=settings.llm_model,
        api_key=get_mistral_api_key(),
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=int(settings.llm_timeout_seconds),
    )


def _format_docs(docs: list[Document]) -> str:
    """Formate les documents récupérés en un seul bloc de contexte."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        formatted.append(f"--- Événement {i} ---\n{doc.page_content}")
    return "\n\n".join(formatted)


def build_components(vectorstore: VectorStore | None = None) -> RagComponents:
    """Crée les embeddings, la base vectorielle et le LLM.

    Args:
        vectorstore: Base à interroger. Par défaut, la table pgvector ;
            l'évaluation peut fournir un index Faiss pour comparer.

    Raises:
        VectorStoreUnavailableError: Si PostgreSQL ou la table sont absents.
    """
    settings = get_settings()
    embeddings = get_embeddings(settings)
    if vectorstore is None:
        vectorstore = open_store(settings, embeddings)
    llm = get_llm(settings)
    logger.info(
        "Chaîne RAG prête (fournisseur: %s, modèle: %s, top_k: %d)",
        settings.llm_provider,
        settings.llm_model,
        settings.top_k,
    )
    return RagComponents(settings, embeddings, vectorstore, llm)


# Instance globale (chargée une seule fois)
_components: RagComponents | None = None


def get_components() -> RagComponents:
    """Retourne les composants RAG, en les construisant au premier appel."""
    global _components  # noqa: PLW0603
    if _components is None:
        _components = build_components()
    return _components


def reset_components() -> None:
    """Oublie les composants chargés (après une reconstruction de l'index)."""
    global _components  # noqa: PLW0603
    _components = None


def _retrieve(
    components: RagComponents, question: str
) -> tuple[list[Document], float, float]:
    """Cherche les chunks proches de la question.

    L'embedding de la question et la recherche vectorielle sont chronométrés
    séparément : seule la recherche dépend de la base vectorielle.

    Returns:
        Tuple (documents, durée embedding, durée recherche) en secondes.
    """
    start = time.perf_counter()
    query_vector = components.embeddings.embed_query(question)
    embedding_seconds = time.perf_counter() - start

    start = time.perf_counter()
    documents = components.vectorstore.similarity_search_by_vector(
        query_vector, k=components.settings.top_k
    )
    search_seconds = time.perf_counter() - start
    return documents, embedding_seconds, search_seconds


def _generate(
    components: RagComponents, messages: list[BaseMessage]
) -> tuple[str, int, int]:
    """Appelle le LLM et renvoie (réponse, tokens d'entrée, tokens de sortie)."""
    response = components.llm.invoke(messages)
    usage = response.usage_metadata or {}
    return (
        str(response.content),
        usage.get("input_tokens", 0),
        usage.get("output_tokens", 0),
    )


def answer_question(
    question: str, components: RagComponents | None = None
) -> RagResult:
    """Répond à une question et mesure latence, tokens et coût.

    Chaque étape est enregistrée dans Langfuse comme une observation
    rattachée à la trace de la question.
    """
    if components is None:
        components = get_components()
    settings = components.settings

    with langfuse.start_as_current_observation(
        name="ask", as_type="span", input={"question": question}
    ) as trace_span:
        with langfuse.start_as_current_observation(
            name="retrieval", as_type="retriever", input={"question": question}
        ) as retrieval_span:
            documents, embedding_seconds, search_seconds = _retrieve(
                components, question
            )
            retrieval_span.update(
                output=[doc.metadata.get("title", "") for doc in documents],
                metadata={
                    "top_k": settings.top_k,
                    "embedding_seconds": embedding_seconds,
                    "search_seconds": search_seconds,
                },
            )

        messages = PROMPT.format_messages(
            context=_format_docs(documents), question=question
        )
        with langfuse.start_as_current_observation(
            name="llm-chat",
            as_type="generation",
            model=settings.llm_model,
            model_parameters={"temperature": settings.llm_temperature},
            input=[{"role": m.type, "content": m.content} for m in messages],
        ) as generation:
            start = time.perf_counter()
            answer, input_tokens, output_tokens = _generate(components, messages)
            generation_seconds = time.perf_counter() - start

            input_cost = llm_cost_usd(settings, input_tokens, 0)
            output_cost = llm_cost_usd(settings, 0, output_tokens)
            generation.update(
                output=answer,
                usage_details={"input": input_tokens, "output": output_tokens},
                cost_details={"input": input_cost, "output": output_cost},
            )

        trace_span.update(output={"answer": answer})

    return RagResult(
        answer=answer,
        documents=documents,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=input_cost + output_cost,
        embedding_seconds=embedding_seconds,
        search_seconds=search_seconds,
        generation_seconds=generation_seconds,
    )


def ask(question: str) -> dict:
    """Pose une question au chatbot RAG.

    Args:
        question: Question de l'utilisateur en français.

    Returns:
        Dictionnaire avec :
        - "answer": la réponse générée par le LLM
        - "sources": les chunks utilisés comme contexte
    """
    # La question peut contenir des données personnelles : pas en INFO
    logger.debug("Question reçue : '%s'", question)
    result = answer_question(question)

    sources = [
        {
            "title": doc.metadata.get("title", ""),
            "city": doc.metadata.get("city", ""),
            "date_start": doc.metadata.get("date_start", ""),
            "date_end": doc.metadata.get("date_end", ""),
            "url": doc.metadata.get("url", ""),
            "excerpt": doc.page_content[:200],
        }
        for doc in result.documents
    ]

    logger.info(
        "Réponse générée (%d caractères, %d sources, %.2fs, %.6f $)",
        len(result.answer),
        len(sources),
        result.total_seconds,
        result.cost_usd,
    )
    return {"answer": result.answer, "sources": sources}


def main():
    """Point d'entrée : test interactif du chatbot RAG."""
    print("=" * 60)
    print("Chatbot Puls-Events — RAG (Île-de-France)")
    print("=" * 60)

    test_questions = [
        "Quels concerts sont prévus à Paris prochainement ?",
        "Y a-t-il des expositions à Versailles ?",
        "Que faire avec des enfants ce weekend en Île-de-France ?",
    ]

    for question in test_questions:
        print(f"\n{'─' * 60}")
        print(f"Question : {question}")
        print(f"{'─' * 60}")

        result = ask(question)
        print(f"\nRéponse :\n{result['answer']}")
        print(f"\nSources ({len(result['sources'])}) :")
        for src in result["sources"]:
            print(f"  • {src['title']} ({src['city']}) — {src['url']}")

    langfuse.flush()


if __name__ == "__main__":
    main()
