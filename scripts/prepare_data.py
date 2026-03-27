"""
Préparation et nettoyage des données d'événements.

Transforme les événements bruts (JSON) en chunks textuels
structurés, prêts pour la vectorisation Faiss.
"""

import json
import logging
import re
from pathlib import Path

import pandas as pd

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Constantes
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INPUT_FILE = DATA_DIR / "raw_events.json"
OUTPUT_FILE = DATA_DIR / "chunks.json"

# Taille cible des chunks en caractères
CHUNK_MAX_CHARS = 2000

# Champs utiles extraits des données brutes
FIELDS_TO_KEEP = [
    "uid",
    "title_fr",
    "description_fr",
    "longdescription_fr",
    "keywords_fr",
    "conditions_fr",
    "daterange_fr",
    "firstdate_begin",
    "lastdate_end",
    "location_name",
    "location_address",
    "location_city",
    "location_department",
    "location_coordinates",
    "canonicalurl",
]


def load_raw_events() -> list[dict]:
    """Charge les événements bruts depuis le fichier JSON.

    Returns:
        Liste des événements bruts.

    Raises:
        FileNotFoundError: Si le fichier raw_events.json n'existe pas.
    """
    if not INPUT_FILE.exists():
        msg = (
            f"Fichier {INPUT_FILE} introuvable. "
            "Lancez d'abord : python scripts/fetch_events.py"
        )
        raise FileNotFoundError(msg)

    with open(INPUT_FILE, encoding="utf-8") as f:
        events = json.load(f)

    logger.info("Chargement de %d événements bruts", len(events))
    return events


def strip_html(text: str) -> str:
    """Supprime les balises HTML d'un texte.

    Args:
        text: Texte contenant potentiellement du HTML.

    Returns:
        Texte nettoyé sans balises HTML.
    """
    # Supprimer les balises HTML
    clean = re.sub(r"<[^>]+>", " ", text)
    # Normaliser les espaces multiples
    clean = re.sub(r"\s+", " ", clean)
    return clean.strip()


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie et filtre le DataFrame des événements.

    - Supprime les doublons (par uid)
    - Supprime les lignes sans titre ni description
    - Nettoie le HTML des descriptions
    - Remplit les valeurs manquantes

    Args:
        df: DataFrame brut des événements.

    Returns:
        DataFrame nettoyé.
    """
    initial_count = len(df)

    # Suppression des doublons par uid
    df = df.drop_duplicates(subset="uid", keep="first")
    logger.info(
        "Doublons supprimés : %d → %d événements",
        initial_count,
        len(df),
    )

    # Suppression des lignes sans titre ET sans description
    df = df.dropna(subset=["title_fr", "description_fr"], how="all")
    logger.info("Après filtrage (titre/description requis) : %d événements", len(df))

    # Nettoyage HTML des descriptions longues
    df["longdescription_fr"] = df["longdescription_fr"].fillna("").apply(strip_html)
    df["description_fr"] = df["description_fr"].fillna("")
    df["conditions_fr"] = df["conditions_fr"].fillna("")
    df["keywords_fr"] = df["keywords_fr"].fillna("")
    df["daterange_fr"] = df["daterange_fr"].fillna("")
    df["location_name"] = df["location_name"].fillna("")
    df["location_address"] = df["location_address"].fillna("")
    df["location_city"] = df["location_city"].fillna("")

    return df


def build_event_text(row: pd.Series) -> str:
    """Construit un texte structuré à partir d'une ligne d'événement.

    Combine titre, dates, lieu, description et conditions en un
    texte lisible et exploitable pour le RAG.

    Args:
        row: Ligne du DataFrame (un événement).

    Returns:
        Texte structuré de l'événement.
    """
    parts = [f"Titre : {row['title_fr']}"]

    if row["daterange_fr"]:
        parts.append(f"Dates : {row['daterange_fr']}")

    # Informations de localisation
    lieu_parts = []
    if row["location_name"]:
        lieu_parts.append(row["location_name"])
    if row["location_address"]:
        lieu_parts.append(row["location_address"])
    if row["location_city"]:
        lieu_parts.append(row["location_city"])
    if lieu_parts:
        parts.append(f"Lieu : {', '.join(lieu_parts)}")

    if row["conditions_fr"]:
        parts.append(f"Conditions : {row['conditions_fr']}")

    if row["keywords_fr"]:
        parts.append(f"Mots-clés : {row['keywords_fr']}")

    # Description : préférer la longue, sinon la courte
    description = row["longdescription_fr"] or row["description_fr"]
    if description:
        parts.append(f"Description : {description}")

    return "\n".join(parts)


def split_text_into_chunks(text: str, max_chars: int = CHUNK_MAX_CHARS) -> list[str]:
    """Découpe un texte en chunks de taille limitée.

    Découpe au niveau des paragraphes (double saut de ligne),
    puis des phrases si nécessaire.

    Args:
        text: Texte à découper.
        max_chars: Taille maximale d'un chunk en caractères.

    Returns:
        Liste de chunks textuels.
    """
    # Si le texte est assez court pas besoin de découper
    if len(text) <= max_chars:
        return [text]

    chunks = []
    current_chunk = ""

    # Découper par paragraphes d'abord
    paragraphs = text.split("\n")

    for paragraph in paragraphs:
        # Si ajouter ce paragraphe dépasse la limite
        if len(current_chunk) + len(paragraph) + 1 > max_chars:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = paragraph
        else:
            current_chunk += "\n" + paragraph if current_chunk else paragraph

    # Ajouter le dernier chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def create_chunks(df: pd.DataFrame) -> list[dict]:
    """Crée les chunks à partir du DataFrame nettoyé.

    Chaque chunk contient le texte et les métadonnées associées
    (uid, titre, lieu, dates, url) pour enrichir le contexte RAG.

    Args:
        df: DataFrame nettoyé des événements.

    Returns:
        Liste de dictionnaires {text, metadata}.
    """
    all_chunks = []

    for _, row in df.iterrows():
        # Construire le texte complet de l'événement
        full_text = build_event_text(row)

        # Découper en chunks
        text_chunks = split_text_into_chunks(full_text)

        for i, chunk_text in enumerate(text_chunks):
            chunk = {
                "text": chunk_text,
                "metadata": {
                    "uid": row.get("uid", ""),
                    "title": row.get("title_fr", ""),
                    "city": row.get("location_city", ""),
                    "location_name": row.get("location_name", ""),
                    "date_start": row.get("firstdate_begin", ""),
                    "date_end": row.get("lastdate_end", ""),
                    "url": row.get("canonicalurl", ""),
                    "chunk_index": i,
                    "total_chunks": len(text_chunks),
                },
            }
            all_chunks.append(chunk)

    logger.info(
        "%d événements → %d chunks (moy. %.1f chunks/événement)",
        len(df),
        len(all_chunks),
        len(all_chunks) / len(df) if len(df) > 0 else 0,
    )
    return all_chunks


def save_chunks(chunks: list[dict]) -> Path:
    """Sauvegarde les chunks en JSON.

    Args:
        chunks: Liste des chunks à sauvegarder.

    Returns:
        Chemin du fichier de sortie.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    logger.info("Chunks sauvegardés dans %s", OUTPUT_FILE)
    return OUTPUT_FILE


def main():
    """Point d'entrée : nettoyage et chunking des données."""
    # Charger les données brutes
    raw_events = load_raw_events()

    # Créer un DataFrame avec uniquement les champs utiles
    df = pd.DataFrame(raw_events)[FIELDS_TO_KEEP]
    logger.info("DataFrame créé : %d lignes × %d colonnes", len(df), len(df.columns))

    # Nettoyer les données
    df = clean_dataframe(df)

    # Créer les chunks
    chunks = create_chunks(df)

    # Sauvegarder
    output_path = save_chunks(chunks)

    # Afficher un aperçu
    logger.info("=== Aperçu du premier chunk ===")
    if chunks:
        logger.info("Texte (200 premiers chars) : %s", chunks[0]["text"][:200])
        logger.info("Métadonnées : %s", chunks[0]["metadata"])

    logger.info("Fichier de sortie : %s", output_path)


if __name__ == "__main__":
    main()
