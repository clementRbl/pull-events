"""
Préparation et nettoyage des données d'événements.

Transforme les événements bruts (JSON) en chunks textuels
structurés, prêts pour la vectorisation.
"""

import json
import logging
import re
import statistics
import unicodedata
from pathlib import Path

import pandas as pd

from scripts.config import get_chunking

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

# Découpage phrase-recoupement : phrases reprises d'un chunk au suivant,
# et taille maximale de l'en-tête répété (titre, dates, lieu)
OVERLAP_SENTENCES = 2
HEADER_MAX_CHARS = 500

# Fins de mot qui terminent une phrase
SENTENCE_ENDINGS = (".", "!", "?", "…")

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


def build_event_header(row: pd.Series) -> str:
    """Construit l'en-tête d'un événement : titre, dates et lieu.

    Le découpage ``phrase-recoupement`` le répète dans chaque chunk, pour
    qu'un morceau de description reste rattaché à son événement.

    Args:
        row: Ligne du DataFrame (un événement).

    Returns:
        En-tête de l'événement, une information par ligne.
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

    return _normalize("\n".join(parts))


def build_event_body(row: pd.Series) -> str:
    """Construit le corps d'un événement : conditions, mots-clés, description.

    Args:
        row: Ligne du DataFrame (un événement).

    Returns:
        Corps de l'événement (vide si aucun de ces champs n'est rempli).
    """
    parts = []

    if row["conditions_fr"]:
        parts.append(f"Conditions : {row['conditions_fr']}")

    if row["keywords_fr"]:
        parts.append(f"Mots-clés : {row['keywords_fr']}")

    # Description : préférer la longue, sinon la courte
    description = row["longdescription_fr"] or row["description_fr"]
    if description:
        parts.append(f"Description : {description}")

    return _normalize("\n".join(parts))


def _normalize(text: str) -> str:
    # NFKC ramène les lettres décoratives (ex. « 𝗣𝗥𝗢𝗚𝗥𝗔𝗠𝗠𝗘 » copié des
    # réseaux sociaux) à leur forme simple. Sans cela, ces caractères coûtent
    # plusieurs tokens chacun et un chunk peut dépasser la limite de
    # 8 192 tokens de mistral-embed.
    return unicodedata.normalize("NFKC", text)


def build_event_text(row: pd.Series) -> str:
    """Construit un texte structuré à partir d'une ligne d'événement.

    Combine titre, dates, lieu, description et conditions en un
    texte lisible et exploitable pour le RAG.

    Args:
        row: Ligne du DataFrame (un événement).

    Returns:
        Texte structuré de l'événement.
    """
    body = build_event_body(row)
    if not body:
        return build_event_header(row)
    return build_event_header(row) + "\n" + body


def split_text_into_chunks(text: str, max_chars: int = CHUNK_MAX_CHARS) -> list[str]:
    """Découpage ``ligne`` : regroupe les lignes du texte jusqu'à max_chars.

    Une ligne plus longue que max_chars n'est pas coupée : le chunk dépasse
    alors la limite (référence de la v1, conservée telle quelle).

    Args:
        text: Texte à découper.
        max_chars: Taille maximale visée d'un chunk en caractères.

    Returns:
        Liste de chunks textuels.
    """
    # Si le texte est assez court pas besoin de découper
    if len(text) <= max_chars:
        return [text]
    return _pack_lines(text.split("\n"), max_chars)


def split_lines_with_cap(text: str, max_chars: int = CHUNK_MAX_CHARS) -> list[str]:
    """Découpage ``ligne-plafond`` : comme ``ligne``, sans jamais dépasser max_chars.

    Une ligne trop longue est d'abord coupée entre deux mots.

    Args:
        text: Texte à découper.
        max_chars: Taille maximale d'un chunk en caractères.

    Returns:
        Liste de chunks textuels, aucun au-delà de max_chars.
    """
    if len(text) <= max_chars:
        return [text]

    lines = []
    for line in text.split("\n"):
        lines.extend(split_on_words(line, max_chars))
    return _pack_lines(lines, max_chars)


def _pack_lines(lines: list[str], max_chars: int) -> list[str]:
    """Regroupe des lignes successives tant que le chunk reste sous max_chars."""
    chunks = []
    current_chunk = ""

    for line in lines:
        # Si ajouter cette ligne dépasse la limite
        if len(current_chunk) + len(line) + 1 > max_chars:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = line
        else:
            current_chunk += "\n" + line if current_chunk else line

    # Ajouter le dernier chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def split_on_words(text: str, max_chars: int) -> list[str]:
    """Coupe un texte en morceaux de max_chars au plus, entre deux mots.

    Un mot plus long que max_chars (une URL interminable, par exemple) est
    coupé au caractère près.

    Args:
        text: Texte à couper.
        max_chars: Taille maximale d'un morceau.

    Returns:
        Morceaux dans l'ordre du texte.
    """
    pieces = []
    remaining = text
    while len(remaining) > max_chars:
        cut = remaining.rfind(" ", 0, max_chars + 1)
        if cut <= 0:
            cut = max_chars
        pieces.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        pieces.append(remaining)
    return pieces


def split_sentences(text: str) -> list[str]:
    """Découpe un texte en phrases.

    Une phrase se termine par « . », « ! », « ? » ou « … » en fin de mot, ou
    par une fin de ligne. Volontairement simple : « M. Dupont » donne deux
    « phrases », ce qui ne gêne pas le découpage en chunks.

    Args:
        text: Texte à découper.

    Returns:
        Phrases dans l'ordre du texte, sans phrase vide.
    """
    sentences = []
    for line in text.split("\n"):
        current = ""
        for word in line.split():
            current = f"{current} {word}" if current else word
            if word.endswith(SENTENCE_ENDINGS):
                sentences.append(current)
                current = ""
        if current:
            sentences.append(current)
    return sentences


def split_sentences_with_overlap(
    header: str,
    body: str,
    max_chars: int = CHUNK_MAX_CHARS,
    overlap_sentences: int = OVERLAP_SENTENCES,
) -> list[str]:
    """Découpage ``phrase-recoupement`` : phrases regroupées, en-tête répété.

    Chaque chunk commence par l'en-tête de l'événement (titre, dates, lieu),
    puis des phrases entières jusqu'à max_chars. Les dernières phrases d'un
    chunk sont reprises au début du suivant, pour qu'une information à
    cheval sur deux chunks reste lisible dans l'un d'eux.

    Args:
        header: En-tête de l'événement, répété dans chaque chunk.
        body: Corps de l'événement, découpé en phrases.
        max_chars: Taille maximale d'un chunk, en-tête compris.
        overlap_sentences: Nombre maximal de phrases reprises d'un chunk au suivant.

    Returns:
        Liste de chunks textuels, aucun au-delà de max_chars.
    """
    full_text = f"{header}\n{body}" if body else header
    if len(full_text) <= max_chars:
        return [full_text]

    # Un en-tête démesuré (une longue liste de dates) ne doit pas manger
    # toute la place des phrases.
    header = split_on_words(header, HEADER_MAX_CHARS)[0]
    budget = max_chars - len(header) - 1

    sentences = []
    for sentence in split_sentences(body):
        sentences.extend(split_on_words(sentence, budget))

    groups = []
    current: list[str] = []
    for sentence in sentences:
        if current and len(" ".join([*current, sentence])) > budget:
            groups.append(current)
            current = _overlap(current, sentence, budget, overlap_sentences)
        current.append(sentence)
    if current:
        groups.append(current)

    return [f"{header}\n{' '.join(group)}" for group in groups]


def _overlap(
    previous: list[str], next_sentence: str, budget: int, overlap_sentences: int
) -> list[str]:
    """Dernières phrases du chunk précédent, autant qu'il en tient avec la suivante."""
    tail = previous[-overlap_sentences:]
    while tail and len(" ".join([*tail, next_sentence])) > budget:
        tail = tail[1:]
    return tail


def chunk_event(row: pd.Series, chunking: str) -> list[str]:
    """Découpe un événement selon la stratégie demandée.

    Un événement court (max_chars ou moins) donne le même chunk unique
    avec les trois stratégies.

    Args:
        row: Ligne du DataFrame (un événement).
        chunking: ``ligne``, ``ligne-plafond`` ou ``phrase-recoupement``.

    Returns:
        Liste de chunks textuels.
    """
    if chunking == "phrase-recoupement":
        return split_sentences_with_overlap(
            build_event_header(row), build_event_body(row)
        )
    if chunking == "ligne-plafond":
        return split_lines_with_cap(build_event_text(row))
    return split_text_into_chunks(build_event_text(row))


def chunk_length_stats(chunks: list[dict]) -> dict[str, int]:
    """Statistiques de longueur des chunks (en caractères).

    Returns:
        Nombre, médiane, 95e centile, maximum et nombre au-delà de CHUNK_MAX_CHARS.
    """
    lengths = sorted(len(chunk["text"]) for chunk in chunks)
    if not lengths:
        return {"count": 0, "median": 0, "p95": 0, "max": 0, "over_max": 0}
    return {
        "count": len(lengths),
        "median": int(statistics.median(lengths)),
        "p95": lengths[int(0.95 * (len(lengths) - 1))],
        "max": lengths[-1],
        "over_max": sum(1 for length in lengths if length > CHUNK_MAX_CHARS),
    }


def create_chunks(df: pd.DataFrame, chunking: str = "ligne") -> list[dict]:
    """Crée les chunks à partir du DataFrame nettoyé.

    Chaque chunk contient le texte et les métadonnées associées
    (uid, titre, lieu, dates, url) pour enrichir le contexte RAG.

    Args:
        df: DataFrame nettoyé des événements.
        chunking: Stratégie de découpage (voir chunk_event).

    Returns:
        Liste de dictionnaires {text, metadata}.
    """
    all_chunks = []

    for _, row in df.iterrows():
        text_chunks = chunk_event(row, chunking)

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
    chunking = get_chunking()
    chunks = create_chunks(df, chunking)
    logger.info("Découpage %s : %s", chunking, chunk_length_stats(chunks))

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
