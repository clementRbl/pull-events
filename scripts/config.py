"""
Configuration centrale de Puls-Events, lue depuis les variables d'environnement.

Tout ce qui change entre le poste local et un déploiement cloud (base de
données, modèles, prix) passe par ici, pour qu'aucun script ne code ces
valeurs en dur.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

DEFAULT_DATABASE_URL = (
    "postgresql+psycopg://pullevents:pullevents@localhost:5432/pullevents"
)


def _float_env(name: str, default: float) -> float:
    """Lit une variable d'environnement numérique, avec valeur par défaut."""
    return float(os.getenv(name, str(default)))


@dataclass(frozen=True)
class Settings:
    """Paramètres de l'application.

    Les prix sont en dollars par million de tokens (tarifs publics Mistral,
    à vérifier sur https://mistral.ai/pricing avant toute décision).
    Le juge est le LLM qui note les réponses pendant l'évaluation Ragas :
    un modèle d'un autre fournisseur évite qu'un modèle note ses propres
    réponses.
    """

    database_url: str
    vector_table: str
    llm_provider: str
    embedding_model: str
    embedding_size: int
    llm_model: str
    llm_temperature: float
    llm_max_tokens: int
    llm_timeout_seconds: float
    top_k: int
    hnsw_ef_search: int
    judge_provider: str
    judge_model: str
    llm_input_price_per_mtok: float
    llm_output_price_per_mtok: float
    embedding_price_per_mtok: float


def get_settings() -> Settings:
    """Construit les paramètres à partir de l'environnement courant.

    Raises:
        ValueError: Si VECTOR_TABLE n'est pas un nom de table simple.
    """
    vector_table = os.getenv("VECTOR_TABLE", "event_chunks")
    llm_provider = os.getenv("LLM_PROVIDER", "mistral")
    if llm_provider not in ("mistral", "azure_openai"):
        msg = f"LLM_PROVIDER doit valoir mistral ou azure_openai : {llm_provider!r}"
        raise ValueError(msg)
    judge_provider = os.getenv("JUDGE_PROVIDER", "anthropic")
    if judge_provider not in ("anthropic", "mistral"):
        msg = f"JUDGE_PROVIDER doit valoir anthropic ou mistral : {judge_provider!r}"
        raise ValueError(msg)
    # Le nom de table est inséré tel quel dans du SQL : on n'accepte
    # que lettres, chiffres et soulignés.
    if not vector_table.isidentifier():
        msg = f"VECTOR_TABLE invalide : {vector_table!r}"
        raise ValueError(msg)

    return Settings(
        database_url=os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL),
        vector_table=vector_table,
        llm_provider=llm_provider,
        embedding_model=os.getenv("EMBEDDING_MODEL", "mistral-embed"),
        embedding_size=int(os.getenv("EMBEDDING_SIZE", "1024")),
        llm_model=os.getenv("LLM_MODEL", "ministral-8b-latest"),
        llm_temperature=_float_env("LLM_TEMPERATURE", 0.3),
        # Plafonds : une réponse ne peut pas consommer plus de tokens ni
        # bloquer un worker plus longtemps que ça (réponses mesurées : ~350 tokens).
        llm_max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1024")),
        llm_timeout_seconds=_float_env("LLM_TIMEOUT_SECONDS", 60.0),
        top_k=int(os.getenv("TOP_K", "5")),
        # 40 (défaut pgvector) : rappel@5 mesuré à 0,91 ; 100 : 0,99 pour +0,4 ms
        hnsw_ef_search=int(os.getenv("HNSW_EF_SEARCH", "100")),
        judge_provider=judge_provider,
        judge_model=os.getenv("JUDGE_MODEL", "claude-haiku-5-5"),
        llm_input_price_per_mtok=_float_env("LLM_INPUT_PRICE_PER_MTOK", 0.10),
        llm_output_price_per_mtok=_float_env("LLM_OUTPUT_PRICE_PER_MTOK", 0.10),
        embedding_price_per_mtok=_float_env("EMBEDDING_PRICE_PER_MTOK", 0.10),
    )


def get_mistral_api_key() -> str:
    """Récupère la clé API Mistral depuis l'environnement.

    Raises:
        ValueError: Si la clé n'est pas définie.
    """
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        msg = "MISTRAL_API_KEY non définie. Configurez-la dans le fichier .env"
        raise ValueError(msg)
    return api_key


@dataclass(frozen=True)
class AzureOpenAISettings:
    """Accès à Azure OpenAI : un point de terminaison et, au choix, une clé.

    Sur Azure, on n'appelle pas un modèle par son nom mais un *déploiement*
    (une instance du modèle créée dans la ressource, avec son propre quota).
    LLM_MODEL et EMBEDDING_MODEL contiennent alors les noms des déploiements.

    Sans clé (``api_key`` à None), l'application s'authentifie par son
    identité Entra ID : identité managée de la Container App sur Azure,
    compte ``az login`` sur le poste.
    """

    endpoint: str
    api_key: str | None
    api_version: str


def get_azure_openai_settings() -> AzureOpenAISettings:
    """Lit la configuration Azure OpenAI depuis l'environnement.

    Raises:
        ValueError: Si le point de terminaison n'est pas défini.
    """
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if not endpoint:
        msg = "AZURE_OPENAI_ENDPOINT doit être définie quand LLM_PROVIDER=azure_openai"
        raise ValueError(msg)
    return AzureOpenAISettings(
        endpoint=endpoint,
        # Une clé vide compte comme absente : identité managée.
        api_key=os.getenv("AZURE_OPENAI_API_KEY") or None,
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
    )


def llm_cost_usd(settings: Settings, input_tokens: int, output_tokens: int) -> float:
    """Calcule le coût en dollars d'un appel LLM à partir des tokens consommés."""
    input_cost = input_tokens * settings.llm_input_price_per_mtok / 1_000_000
    output_cost = output_tokens * settings.llm_output_price_per_mtok / 1_000_000
    return input_cost + output_cost
