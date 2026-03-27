"""
Ingestion des événements culturels depuis l'API OpenDataSoft.

Récupère les événements publics Open Agenda pour la région
Île-de-France, datant de moins d'un an, avec pagination automatique.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import requests

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# === Constantes ===
API_BASE_URL = (
    "https://public.opendatasoft.com/api/explore/v2.1"
    "/catalog/datasets/evenements-publics-openagenda/records"
)
REGION = "Île-de-France"
PAGE_SIZE = 100  # Maximum autorisé par l'API
OFFSET_LIMIT = 9900  # Limite d'offset de l'API OpenDataSoft
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_FILE = DATA_DIR / "raw_events.json"


def _date_limit() -> str:
    """Retourne la date limite (il y a 1 an) au format ISO."""
    return (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")


def build_query_params(
    offset: int = 0,
    date_upper_bound: str | None = None,
) -> dict:
    """Construit les paramètres de requête pour l'API.

    Filtre sur la région Île-de-France et les événements
    dont la dernière date est dans les 12 derniers mois.
    Utilise un curseur de date pour contourner la limite offset=10000.

    Args:
        offset: Décalage pour la pagination.
        date_upper_bound: Borne supérieure de date (curseur) pour
            contourner la limite d'offset de l'API.

    Returns:
        Dictionnaire des paramètres de requête.
    """
    where_parts = [
        f"location_region='{REGION}'",
        f"lastdate_end >= '{_date_limit()}'",
    ]

    # Curseur de date : filtrer les événements antérieurs à la borne
    if date_upper_bound:
        where_parts.append(f"lastdate_end < '{date_upper_bound}'")

    return {
        "where": " AND ".join(where_parts),
        "limit": PAGE_SIZE,
        "offset": offset,
        "order_by": "lastdate_end DESC",
    }


def fetch_page(
    offset: int,
    date_upper_bound: str | None = None,
) -> dict:
    """Récupère une page de résultats depuis l'API.

    Args:
        offset: Position de départ dans les résultats.
        date_upper_bound: Borne supérieure de date pour le curseur.

    Returns:
        Réponse JSON de l'API.

    Raises:
        requests.HTTPError: Si la requête échoue.
    """
    params = build_query_params(offset, date_upper_bound)
    response = requests.get(API_BASE_URL, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_all_events() -> list[dict]:
    """Récupère tous les événements avec pagination par curseur de date.

    L'API OpenDataSoft limite l'offset à environ 10 000. Pour contourner
    cette limite, on pagine normalement jusqu'à l'offset max, puis
    on utilise la date du dernier résultat comme curseur pour
    relancer une nouvelle fenêtre de pagination.

    Returns:
        Liste de tous les événements récupérés.
    """
    all_events = []
    seen_uids = set()  # Éviter les doublons entre fenêtres
    date_upper_bound = None

    # Première requête pour connaître le total global
    data = fetch_page(0)
    total_count = data.get("total_count", 0)
    logger.info("Total d'événements à récupérer : %d", total_count)

    while True:
        offset = 0

        # Pagination dans la fenêtre courante
        while True:
            data = fetch_page(offset, date_upper_bound)
            results = data.get("results", [])

            if not results:
                break

            # Ajouter uniquement les événements non encore vus
            for event in results:
                uid = event.get("uid")
                if uid not in seen_uids:
                    seen_uids.add(uid)
                    all_events.append(event)

            logger.info(
                "Progression : %d événements récupérés (offset=%d, curseur=%s)",
                len(all_events),
                offset,
                date_upper_bound or "aucun",
            )

            offset += PAGE_SIZE

            # Si on atteint la limite d'offset, passer au curseur de date
            if offset > OFFSET_LIMIT:
                break

        # Déterminer la nouvelle borne de date à partir du dernier résultat
        if results:
            last_date = results[-1].get("lastdate_end", "")
            if last_date and last_date != date_upper_bound:
                date_upper_bound = last_date
                logger.info("Nouveau curseur de date : %s", date_upper_bound)
                continue

        # Plus de résultats ou curseur inchangé → fin
        break

    logger.info(
        "Ingestion terminée : %d événements récupérés au total",
        len(all_events),
    )
    return all_events


def save_events(events: list[dict]) -> Path:
    """Sauvegarde les événements bruts en JSON.

    Args:
        events: Liste des événements à sauvegarder.

    Returns:
        Chemin du fichier de sortie.
    """
    # Créer le dossier data/ s'il n'existe pas
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(events, f, ensure_ascii=False, indent=2)

    logger.info("Événements sauvegardés dans %s", OUTPUT_FILE)
    return OUTPUT_FILE


def main():
    """Point d'entrée : ingestion complète des événements."""
    logger.info("Démarrage de l'ingestion — Région : %s", REGION)

    events = fetch_all_events()
    output_path = save_events(events)

    logger.info("Fichier de sortie : %s (%d événements)", output_path, len(events))


if __name__ == "__main__":
    main()
