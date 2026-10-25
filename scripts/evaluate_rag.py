"""
Évaluation automatique du pipeline RAG avec Ragas.

Exécute le chatbot sur un jeu de test annoté, calcule des métriques de
qualité (fidélité, pertinence, précision) et relève pour chaque question
la latence, les tokens consommés et le coût.

Usage :
    uv run python -m scripts.evaluate_rag --backend pgvector
    uv run python -m scripts.evaluate_rag --backend faiss   # référence v1

Mode API : les mêmes questions sont envoyées en HTTP à une API déployée
(Container Apps, AKS...). On mesure ce que voit un client : code HTTP et
latence. Pas de juge Ragas dans ce mode : l'API ne renvoie qu'un extrait
de chaque source, et la qualité est déjà mesurée depuis le poste avec la
même image, la même base et le même modèle.
    API_KEY=... uv run python -m scripts.evaluate_rag --api-url https://... \\
        --output-label aks
"""

import argparse
import json
import logging
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import requests
from langchain_mistralai import ChatMistralAI
from ragas import EvaluationDataset, RunConfig, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    ResponseRelevancy,
)

from scripts.config import Settings, get_mistral_api_key, get_settings
from scripts.rag_chain import (
    RagComponents,
    RagResult,
    answer_question,
    build_components,
    get_embeddings,
    get_mistral_embeddings,
    langfuse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_FILE = PROJECT_ROOT / "data" / "test_questions.json"
RESULTS_DIR = PROJECT_ROOT / "docs" / "evaluation"

# Embeddings utilisés par la métrique answer_relevancy. Ils font partie du
# juge : on les garde fixes quel que soit le fournisseur évalué, sinon les
# scores de deux configurations ne seraient pas comparables.
JUDGE_EMBEDDING_MODEL = "mistral-embed"

# Coût du juge mesuré dans Langfuse le 24/10/2026 avec claude-haiku-5-5 :
# environ 0,12 $ pour 20 questions. Sert à prévenir avant de lancer le juge.
JUDGE_COST_PER_QUESTION_USD = 0.006

RAGAS_METRICS = [
    "faithfulness",
    "answer_relevancy",
    "llm_context_precision_with_reference",
]


def load_test_set() -> list[dict]:
    """Charge le jeu de test annoté depuis data/test_questions.json."""
    with open(TEST_FILE, encoding="utf-8") as f:
        return json.load(f)


def load_components(backend: str) -> RagComponents:
    """Prépare le RAG sur la base demandée (pgvector, ou Faiss pour comparer)."""
    if backend == "pgvector":
        return build_components()

    # Import local : Faiss n'est installé qu'avec les dépendances de dev
    from scripts.faiss_baseline import load_faiss_index

    embeddings = get_embeddings(get_settings())
    return build_components(vectorstore=load_faiss_index(embeddings))


def run_rag_on_test_set(
    test_set: list[dict], components: RagComponents
) -> list[tuple[dict, RagResult]]:
    """Exécute le RAG sur chaque question du jeu de test.

    Returns:
        Liste de couples (question annotée, résultat détaillé du RAG).
    """
    results = []
    for i, item in enumerate(test_set, 1):
        logger.info("Question %d/%d : %s", i, len(test_set), item["question"])
        results.append((item, answer_question(item["question"], components)))
    return results


@dataclass(frozen=True)
class ApiResult:
    """Résultat d'une question posée en HTTP à une API déployée."""

    question: str
    status_code: int
    seconds: float
    answer: str
    sources: list[str]


def ask_api(api_url: str, api_key: str, question: str, timeout: float) -> ApiResult:
    """Envoie une question à POST /ask et mesure la latence vue par le client."""
    start = time.perf_counter()
    response = requests.post(
        f"{api_url.rstrip('/')}/ask",
        json={"question": question},
        headers={"X-API-Key": api_key},
        timeout=timeout,
    )
    seconds = time.perf_counter() - start
    body = response.json() if response.status_code == 200 else {}
    return ApiResult(
        question=question,
        status_code=response.status_code,
        seconds=seconds,
        answer=body.get("answer", ""),
        sources=[source.get("title", "") for source in body.get("sources", [])],
    )


def run_api_on_test_set(
    test_set: list[dict], api_url: str, api_key: str, pause_seconds: float
) -> list[ApiResult]:
    """Pose chaque question à l'API, avec une pause (limite de débit par IP)."""
    results = []
    for i, item in enumerate(test_set, 1):
        logger.info("Question %d/%d : %s", i, len(test_set), item["question"])
        result = ask_api(api_url, api_key, item["question"], timeout=120)
        logger.info("HTTP %d en %.2fs", result.status_code, result.seconds)
        results.append(result)
        if i < len(test_set):
            time.sleep(pause_seconds)
    return results


def summarize_api(results: list[ApiResult]) -> dict:
    """Agrège codes HTTP et latences des questions posées à l'API."""
    seconds = [r.seconds for r in results]
    return {
        "questions": len(results),
        "http_200": sum(1 for r in results if r.status_code == 200),
        "latency_seconds": {
            "total_mean": round(_mean(seconds), 3),
            "total_p50": round(statistics.median(seconds), 3),
            "total_p95": round(_percentile_95(seconds), 3),
        },
    }


def save_api_results(label: str, api_url: str, results: list[ApiResult]) -> Path:
    """Écrit le résumé et le détail par question dans docs/evaluation/."""
    report = {
        "label": label,
        "mode": "api",
        "api_url": api_url,
        "summary": summarize_api(results),
        "details": [
            {
                "question": r.question,
                "status_code": r.status_code,
                "seconds": round(r.seconds, 3),
                "answer": r.answer,
                "sources": r.sources,
            }
            for r in results
        ],
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = RESULTS_DIR / f"results_{label}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return output_file


def build_judge(settings: Settings) -> LangchainLLMWrapper:
    """Crée le LLM juge de Ragas (Claude par défaut, ou Mistral)."""
    if settings.judge_provider == "anthropic":
        # Import local : langchain-anthropic n'est installé qu'en dev.
        # La clé est lue dans ANTHROPIC_API_KEY.
        from langchain_anthropic import ChatAnthropic

        # Les modèles Claude récents refusent le paramètre temperature,
        # que Ragas envoie par défaut : on le désactive.
        return LangchainLLMWrapper(
            ChatAnthropic(model=settings.judge_model, max_tokens=8192),
            bypass_temperature=True,
        )

    return LangchainLLMWrapper(
        ChatMistralAI(
            model=settings.judge_model, api_key=get_mistral_api_key(), temperature=0
        )
    )


def estimate_judge_cost_usd(question_count: int) -> float:
    """Estime le coût du juge (Ragas fait plusieurs appels par question)."""
    return question_count * JUDGE_COST_PER_QUESTION_USD


def evaluate_with_ragas(results: list[tuple[dict, RagResult]]) -> list[dict]:
    """Évalue les réponses avec Ragas et retourne les scores par question."""
    settings = get_settings()
    logger.info(
        "Juge %s sur %d questions : coût estimé ~%.2f $",
        settings.judge_model,
        len(results),
        estimate_judge_cost_usd(len(results)),
    )
    evaluator_llm = build_judge(settings)
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        get_mistral_embeddings(JUDGE_EMBEDDING_MODEL, settings.llm_timeout_seconds)
    )

    samples = [
        SingleTurnSample(
            user_input=item["question"],
            response=result.answer,
            retrieved_contexts=[doc.page_content for doc in result.documents],
            reference=item["ground_truth"],
        )
        for item, result in results
    ]

    metrics = [
        Faithfulness(llm=evaluator_llm),
        ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        LLMContextPrecisionWithReference(llm=evaluator_llm),
    ]

    logger.info("Évaluation Ragas en cours...")
    # Peu de requêtes en parallèle pour rester sous la limite de débit Mistral
    evaluation = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=metrics,
        run_config=RunConfig(max_workers=2, max_retries=10),
    )
    return evaluation.to_pandas()[RAGAS_METRICS].to_dict(orient="records")


def _mean(values: list[float]) -> float:
    """Moyenne qui ignore les scores manquants (NaN renvoyés par Ragas)."""
    valid = [v for v in values if v == v]
    return statistics.fmean(valid) if valid else float("nan")


def _percentile_95(values: list[float]) -> float:
    """95e centile : la latence que 95 % des questions ne dépassent pas."""
    return statistics.quantiles(values, n=20)[-1]


def summarize(results: list[tuple[dict, RagResult]], scores: list[dict]) -> dict:
    """Agrège qualité, latence, tokens et coût sur tout le jeu de test."""
    rag_results = [result for _, result in results]
    total_seconds = [r.total_seconds for r in rag_results]
    mean_cost = _mean([r.cost_usd for r in rag_results])

    return {
        "questions": len(rag_results),
        "quality": {
            name: round(_mean([s[name] for s in scores]), 4) for name in RAGAS_METRICS
        },
        "latency_seconds": {
            "search_mean": round(_mean([r.search_seconds for r in rag_results]), 4),
            "embedding_mean": round(
                _mean([r.embedding_seconds for r in rag_results]), 3
            ),
            "generation_mean": round(
                _mean([r.generation_seconds for r in rag_results]), 3
            ),
            "total_mean": round(_mean(total_seconds), 3),
            "total_p50": round(statistics.median(total_seconds), 3),
            "total_p95": round(_percentile_95(total_seconds), 3),
        },
        "tokens_mean": {
            "input": round(_mean([r.input_tokens for r in rag_results])),
            "output": round(_mean([r.output_tokens for r in rag_results])),
        },
        "cost_usd": {
            "per_question_mean": round(mean_cost, 6),
            "per_1000_questions": round(mean_cost * 1000, 3),
        },
    }


def save_results(
    label: str,
    backend: str,
    results: list[tuple[dict, RagResult]],
    scores: list[dict],
) -> Path:
    """Écrit le résumé et le détail par question dans docs/evaluation/."""
    settings = get_settings()
    details = [
        {
            "question": item["question"],
            "answer": result.answer,
            "sources": [doc.metadata.get("title", "") for doc in result.documents],
            "scores": score,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "cost_usd": round(result.cost_usd, 6),
            "search_seconds": round(result.search_seconds, 4),
            "total_seconds": round(result.total_seconds, 3),
        }
        for (item, result), score in zip(results, scores, strict=True)
    ]
    report = {
        "label": label,
        "backend": backend,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "judge_model": settings.judge_model,
        "embedding_model": settings.embedding_model,
        "vector_table": settings.vector_table if backend == "pgvector" else None,
        "top_k": settings.top_k,
        "hnsw_ef_search": settings.hnsw_ef_search if backend == "pgvector" else None,
        "summary": summarize(results, scores),
        "details": details,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = RESULTS_DIR / f"results_{label}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return output_file


def main():
    """Point d'entrée : évaluation complète du pipeline RAG."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-label",
        default=None,
        help="Nom du fichier de résultats (par défaut : le nom du backend)",
    )
    parser.add_argument(
        "--backend",
        choices=["pgvector", "faiss"],
        default="pgvector",
        help="Base vectorielle évaluée",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=20,
        help="Nombre maximum de questions évaluées (le juge est facturé par question)",
    )
    parser.add_argument(
        "--api-url",
        default=None,
        help="Mode API : URL d'une API déployée (clé lue dans la variable API_KEY)",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=7.0,
        help="Mode API : pause entre deux questions (10 questions/min par IP)",
    )
    args = parser.parse_args()

    test_set = load_test_set()[: args.max_questions]
    logger.info("%d questions retenues", len(test_set))

    if args.api_url:
        api_key = os.getenv("API_KEY")
        if not api_key:
            msg = "API_KEY doit être définie dans l'environnement en mode API"
            raise ValueError(msg)
        api_results = run_api_on_test_set(
            test_set, args.api_url, api_key, args.pause_seconds
        )
        label = args.output_label or "api"
        output_file = save_api_results(label, args.api_url, api_results)
        print("\n" + "=" * 60)
        print(f"RÉSULTATS API ({label})")
        print("=" * 60)
        print(json.dumps(summarize_api(api_results), indent=2, ensure_ascii=False))
        logger.info("Résultats sauvegardés dans %s", output_file)
        return

    # Base chargée avant les questions, pour ne pas chronométrer le démarrage
    components = load_components(args.backend)
    results = run_rag_on_test_set(test_set, components)
    scores = evaluate_with_ragas(results)
    label = args.output_label or args.backend
    output_file = save_results(label, args.backend, results, scores)
    langfuse.flush()

    summary = summarize(results, scores)
    print("\n" + "=" * 60)
    print(f"RÉSULTATS ({label})")
    print("=" * 60)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    logger.info("Résultats sauvegardés dans %s", output_file)


if __name__ == "__main__":
    main()
