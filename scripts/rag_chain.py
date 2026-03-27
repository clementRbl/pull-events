"""
Chaîne RAG (Retrieval-Augmented Generation) pour Puls-Events.

Assemble le pipeline complet : recherche vectorielle Faiss +
génération de réponse via LLM Mistral orchestré par LangChain.
Expose une fonction `ask()` réutilisable par l'API et les tests.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

# Charger les variables d'environnement
load_dotenv()

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Constantes
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INDEX_DIR = DATA_DIR / "faiss_index"

EMBEDDING_MODEL = "mistral-embed"
LLM_MODEL = "mistral-small-latest"

# Nombre de chunks à récupérer par requête
TOP_K = 5

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


def _get_api_key() -> str:
    """Récupère la clé API Mistral depuis l'environnement.

    Returns:
        Clé API Mistral.

    Raises:
        ValueError: Si la clé n'est pas définie.
    """
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        msg = "MISTRAL_API_KEY non définie. Configurez-la dans le fichier .env"
        raise ValueError(msg)
    return api_key


def load_vectorstore() -> FAISS:
    """Charge l'index Faiss depuis le disque.

    Returns:
        Instance FAISS avec l'index vectoriel chargé.

    Raises:
        FileNotFoundError: Si l'index n'existe pas.
    """
    if not INDEX_DIR.exists():
        msg = (
            f"Index Faiss introuvable dans {INDEX_DIR}. "
            "Lancez d'abord : python scripts/build_index.py"
        )
        raise FileNotFoundError(msg)

    api_key = _get_api_key()
    embeddings = MistralAIEmbeddings(model=EMBEDDING_MODEL, api_key=api_key)
    vectorstore = FAISS.load_local(
        str(INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    logger.info("Index Faiss chargé depuis %s", INDEX_DIR)
    return vectorstore


def _format_docs(docs: list) -> str:
    """Formate les documents récupérés en texte pour le prompt.

    Combine les chunks trouvés en un seul bloc de contexte,
    séparé par des lignes de démarcation.

    Args:
        docs: Liste de Documents LangChain retournés par Faiss.

    Returns:
        Texte formaté avec tous les chunks pertinents.
    """
    formatted = []
    for i, doc in enumerate(docs, 1):
        formatted.append(f"--- Événement {i} ---\n{doc.page_content}")
    return "\n\n".join(formatted)


def build_rag_chain():
    """Construit la chaîne RAG complète avec LangChain.

    Assemble : Retriever Faiss → Prompt augmenté → LLM Mistral → Réponse texte.

    Returns:
        Tuple (chain, retriever) — la chaîne RAG et le retriever pour usage direct.
    """
    api_key = _get_api_key()

    # Charger le vectorstore et créer le retriever
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    # Configurer le LLM Mistral
    llm = ChatMistralAI(
        model=LLM_MODEL,
        api_key=api_key,
        temperature=0.3,
    )

    # Construire le prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", USER_PROMPT),
        ]
    )

    # Assembler la chaîne RAG avec LCEL (LangChain Expression Language)
    # 1. Le retriever cherche les chunks pertinents
    # 2. Les chunks sont formatés en texte (context)
    # 3. Le prompt injecte le contexte + la question
    # 4. Le LLM génère la réponse
    # 5. Le parser extrait le texte brut
    chain = (
        {
            "context": retriever | _format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    logger.info("Chaîne RAG construite (modèle: %s, top_k: %d)", LLM_MODEL, TOP_K)
    return chain, retriever


# Instance globale (chargée une seule fois)
_chain = None
_retriever = None


def _get_chain():
    """Retourne la chaîne RAG, en la construisant si nécessaire.

    Utilise un singleton pour éviter de recharger l'index
    à chaque appel.

    Returns:
        Tuple (chain, retriever).
    """
    global _chain, _retriever  # noqa: PLW0603
    if _chain is None:
        _chain, _retriever = build_rag_chain()
    return _chain, _retriever


def ask(question: str) -> dict:
    """Pose une question au chatbot RAG.

    C'est la fonction principale du système. Elle prend une question
    en langage naturel et retourne une réponse basée sur les
    événements culturels indexés.

    Args:
        question: Question de l'utilisateur en français.

    Returns:
        Dictionnaire avec :
        - "answer": la réponse générée par le LLM
        - "sources": les chunks utilisés comme contexte
    """
    chain, retriever = _get_chain()

    logger.info("Question reçue : '%s'", question)

    # Récupérer les chunks pertinents (pour les retourner comme sources)
    source_docs = retriever.invoke(question)

    # Générer la réponse via la chaîne RAG
    answer = chain.invoke(question)

    # Construire les sources pour la traçabilité
    sources = [
        {
            "title": doc.metadata.get("title", ""),
            "city": doc.metadata.get("city", ""),
            "date_start": doc.metadata.get("date_start", ""),
            "date_end": doc.metadata.get("date_end", ""),
            "url": doc.metadata.get("url", ""),
            "excerpt": doc.page_content[:200],
        }
        for doc in source_docs
    ]

    logger.info(
        "Réponse générée (%d caractères, %d sources)", len(answer), len(sources)
    )
    return {"answer": answer, "sources": sources}


def main():
    """Point d'entrée : test interactif du chatbot RAG."""
    print("=" * 60)
    print("Chatbot Puls-Events — RAG (Île-de-France)")
    print("Tapez 'quit' pour quitter.")
    print("=" * 60)

    # Test automatique avec quelques questions
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


if __name__ == "__main__":
    main()
