"""
Évaluation automatique du pipeline RAG avec Ragas.

Exécute le chatbot sur un jeu de test annoté et calcule
des métriques de qualité : fidélité, pertinence, précision.
"""

import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Ajouter la racine du projet au path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

TEST_FILE = PROJECT_ROOT / "data" / "test_questions.json"
RESULTS_FILE = PROJECT_ROOT / "data" / "evaluation_results.json"


def load_test_set() -> list[dict]:
    """Charge le jeu de test annoté depuis data/test_questions.json."""
    with open(TEST_FILE) as f:
        return json.load(f)


def run_rag_on_test_set(test_set: list[dict]) -> list[dict]:
    """Exécute le RAG sur chaque question du jeu de test.

    Returns:
        Liste de résultats avec question, réponse, contextes et ground truth.
    """
    from scripts.rag_chain import _get_chain

    chain, retriever = _get_chain()

    results = []
    for i, item in enumerate(test_set, 1):
        question = item["question"]
        logger.info("Question %d/%d : %s", i, len(test_set), question)

        # Récupérer les chunks pertinents
        docs = retriever.invoke(question)
        contexts = [doc.page_content for doc in docs]

        # Générer la réponse
        answer = chain.invoke(question)

        results.append(
            {
                "user_input": question,
                "response": answer,
                "retrieved_contexts": contexts,
                "reference": item["ground_truth"],
            }
        )

    return results


def evaluate_with_ragas(results: list[dict]) -> dict:
    """Évalue les résultats avec Ragas et retourne les scores."""
    from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        ResponseRelevancy,
    )

    api_key = os.getenv("MISTRAL_API_KEY")

    # Configurer Ragas avec Mistral (au lieu d'OpenAI par défaut)
    evaluator_llm = LangchainLLMWrapper(
        ChatMistralAI(model="mistral-small-latest", api_key=api_key, temperature=0)
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        MistralAIEmbeddings(model="mistral-embed", api_key=api_key)
    )

    # Construire le dataset Ragas
    samples = [
        SingleTurnSample(
            user_input=r["user_input"],
            response=r["response"],
            retrieved_contexts=r["retrieved_contexts"],
            reference=r["reference"],
        )
        for r in results
    ]
    eval_dataset = EvaluationDataset(samples=samples)

    # Métriques d'évaluation
    metrics = [
        Faithfulness(llm=evaluator_llm),
        ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        LLMContextPrecisionWithReference(llm=evaluator_llm),
    ]

    logger.info("Évaluation Ragas en cours...")
    evaluation = evaluate(dataset=eval_dataset, metrics=metrics)

    return evaluation


def main():
    """Point d'entrée : évaluation complète du pipeline RAG."""
    # Charger le jeu de test
    logger.info("Chargement du jeu de test...")
    test_set = load_test_set()
    logger.info("%d questions chargées", len(test_set))

    # Exécuter le RAG
    logger.info("Exécution du RAG sur le jeu de test...")
    results = run_rag_on_test_set(test_set)

    # Évaluer avec Ragas
    evaluation = evaluate_with_ragas(results)

    # Afficher les résultats
    print("\n" + "=" * 60)
    print("RÉSULTATS DE L'ÉVALUATION RAGAS")
    print("=" * 60)
    for metric_name, score in evaluation.items():
        if isinstance(score, (int, float)):
            print(f"  {metric_name}: {score:.4f}")
    print("=" * 60)

    # Sauvegarder
    scores = {
        k: round(float(v), 4)
        for k, v in evaluation.items()
        if isinstance(v, (int, float))
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)
    logger.info("Résultats sauvegardés dans %s", RESULTS_FILE)


if __name__ == "__main__":
    main()
