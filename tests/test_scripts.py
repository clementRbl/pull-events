"""Tests unitaires pour les scripts du pipeline de données."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from scripts.config import get_chunking
from scripts.prepare_data import (
    build_event_text,
    chunk_event,
    chunk_length_stats,
    clean_dataframe,
    split_lines_with_cap,
    split_on_words,
    split_sentences,
    split_sentences_with_overlap,
    split_text_into_chunks,
    strip_html,
)

# === Tests strip_html ===


def test_strip_html_removes_tags():
    """strip_html supprime les balises HTML."""
    assert strip_html("<p>Bonjour</p>") == "Bonjour"


def test_strip_html_nested_tags():
    """strip_html gère les balises imbriquées."""
    result = strip_html("<div><p>Un <strong>concert</strong> de jazz</p></div>")
    assert "Un" in result
    assert "concert" in result
    assert "<" not in result


def test_strip_html_empty_string():
    """strip_html retourne une chaîne vide si l'entrée est vide."""
    assert strip_html("") == ""


def test_strip_html_no_tags():
    """strip_html retourne le texte tel quel s'il n'y a pas de balises."""
    assert strip_html("Texte sans HTML") == "Texte sans HTML"


# === Tests split_text_into_chunks ===


def test_split_short_text_no_split():
    """Un texte court n'est pas découpé."""
    text = "Texte court."
    chunks = split_text_into_chunks(text, max_chars=2000)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_split_long_text():
    """Un texte long est découpé en plusieurs chunks."""
    paragraphs = [f"Paragraphe numéro {i}." * 10 for i in range(20)]
    text = "\n".join(paragraphs)
    chunks = split_text_into_chunks(text, max_chars=200)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 200 or "\n" not in chunk


def test_split_respects_max_chars():
    """Chaque chunk respecte la taille maximale quand c'est possible."""
    text = "A" * 100 + "\n" + "B" * 100 + "\n" + "C" * 100
    chunks = split_text_into_chunks(text, max_chars=150)
    assert len(chunks) >= 2


# === Tests des découpages ligne-plafond et phrase-recoupement ===

HEADER = "Titre : Exposition\nDates : 1er mars 2027\nLieu : Musée, Paris"


def make_event(description: str) -> pd.Series:
    """Événement minimal dont seule la description varie."""
    return pd.Series(
        {
            "title_fr": "Exposition",
            "daterange_fr": "1er mars 2027",
            "location_name": "Musée",
            "location_address": "",
            "location_city": "Paris",
            "conditions_fr": "",
            "keywords_fr": "",
            "description_fr": "",
            "longdescription_fr": description,
        }
    )


def long_body(sentence_count: int) -> str:
    """Description d'une seule ligne, faite de phrases numérotées."""
    sentences = [
        f"Phrase numéro {i} qui décrit une œuvre de l'exposition."
        for i in range(1, sentence_count + 1)
    ]
    return "Description : " + " ".join(sentences)


def test_split_on_words_respects_max_and_keeps_words():
    """Les morceaux tiennent dans la limite et aucun mot n'est perdu."""
    text = " ".join(f"mot{i}" for i in range(500))
    pieces = split_on_words(text, max_chars=100)
    assert all(len(piece) <= 100 for piece in pieces)
    assert " ".join(pieces).split() == text.split()


def test_split_on_words_cuts_a_word_longer_than_max():
    """Un mot sans espace plus long que la limite est coupé au caractère près."""
    pieces = split_on_words("x" * 250, max_chars=100)
    assert [len(piece) for piece in pieces] == [100, 100, 50]


def test_line_with_cap_splits_a_line_longer_than_max():
    """ligne-plafond coupe la ligne trop longue que ligne laissait passer."""
    text = "Titre : Concert\n" + "mot " * 1000
    assert max(len(chunk) for chunk in split_text_into_chunks(text, 2000)) > 2000
    assert all(len(chunk) <= 2000 for chunk in split_lines_with_cap(text, 2000))


def test_split_sentences_cuts_on_punctuation_and_line_breaks():
    """Une phrase finit sur « . ! ? … » ou en fin de ligne."""
    text = "Bienvenue ! Venez voir. Entrée libre\nOuvert le dimanche… Et après ?"
    assert split_sentences(text) == [
        "Bienvenue !",
        "Venez voir.",
        "Entrée libre",
        "Ouvert le dimanche…",
        "Et après ?",
    ]


def test_sentence_overlap_respects_max_chars():
    """Aucun chunk ne dépasse la limite, en-tête compris."""
    chunks = split_sentences_with_overlap(HEADER, long_body(200), max_chars=2000)
    assert len(chunks) > 3
    assert all(len(chunk) <= 2000 for chunk in chunks)


def test_sentence_overlap_repeats_header_in_every_chunk():
    """Chaque chunk commence par l'en-tête de l'événement."""
    chunks = split_sentences_with_overlap(HEADER, long_body(200), max_chars=2000)
    assert all(chunk.startswith(HEADER + "\n") for chunk in chunks)


def test_sentence_overlap_repeats_last_sentences_in_next_chunk():
    """Les deux dernières phrases d'un chunk ouvrent le chunk suivant."""
    chunks = split_sentences_with_overlap(HEADER, long_body(200), max_chars=2000)
    for previous, following in zip(chunks, chunks[1:], strict=False):
        previous_sentences = split_sentences(previous.removeprefix(HEADER + "\n"))
        following_sentences = split_sentences(following.removeprefix(HEADER + "\n"))
        assert following_sentences[:2] == previous_sentences[-2:]


def test_sentence_overlap_loses_no_sentence():
    """Toutes les phrases de la description se retrouvent dans un chunk."""
    body = long_body(200)
    chunks = split_sentences_with_overlap(HEADER, body, max_chars=2000)
    kept = set()
    for chunk in chunks:
        kept.update(split_sentences(chunk.removeprefix(HEADER + "\n")))
    assert set(split_sentences(body)) <= kept


def test_sentence_overlap_truncates_a_huge_header():
    """Un en-tête démesuré (longue liste de dates) est raccourci."""
    header = "Titre : Visites\nDates : " + "lundi 3 mars, " * 300
    chunks = split_sentences_with_overlap(header, long_body(200), max_chars=2000)
    assert all(len(chunk) <= 2000 for chunk in chunks)
    assert all(chunk.startswith("Titre : Visites") for chunk in chunks)


@pytest.mark.parametrize("chunking", ["ligne", "ligne-plafond", "phrase-recoupement"])
def test_short_event_gives_the_same_single_chunk(chunking):
    """Un événement court reste un chunk unique, identique pour les trois
    stratégies : c'est ce qui permet de réutiliser ses vecteurs."""
    row = make_event("Une courte description.")
    assert chunk_event(row, chunking) == [build_event_text(row)]


def test_get_chunking_rejects_unknown_strategy(monkeypatch):
    """Une stratégie inconnue est refusée au lieu d'être ignorée."""
    monkeypatch.setenv("CHUNKING", "paragraphe")
    with pytest.raises(ValueError, match="CHUNKING"):
        get_chunking()


def test_chunk_length_stats():
    """Les statistiques comptent les chunks au-delà de 2 000 caractères."""
    chunks = [{"text": "a" * length} for length in (100, 500, 900, 2500)]
    stats = chunk_length_stats(chunks)
    assert stats["count"] == 4
    assert stats["max"] == 2500
    assert stats["over_max"] == 1


# === Tests clean_dataframe ===


def test_clean_dataframe_removes_duplicates():
    """clean_dataframe supprime les doublons par uid."""
    df = pd.DataFrame(
        {
            "uid": ["1", "1", "2"],
            "title_fr": ["A", "A", "B"],
            "description_fr": ["Desc A", "Desc A", "Desc B"],
            "longdescription_fr": [None, None, None],
            "conditions_fr": [None, None, None],
            "keywords_fr": [None, None, None],
            "daterange_fr": [None, None, None],
            "location_name": [None, None, None],
            "location_address": [None, None, None],
            "location_city": [None, None, None],
        }
    )
    result = clean_dataframe(df)
    assert len(result) == 2


def test_clean_dataframe_drops_empty_rows():
    """clean_dataframe supprime les lignes sans titre ni description."""
    df = pd.DataFrame(
        {
            "uid": ["1", "2"],
            "title_fr": ["Concert", None],
            "description_fr": ["Super concert", None],
            "longdescription_fr": [None, None],
            "conditions_fr": [None, None],
            "keywords_fr": [None, None],
            "daterange_fr": [None, None],
            "location_name": [None, None],
            "location_address": [None, None],
            "location_city": [None, None],
        }
    )
    result = clean_dataframe(df)
    assert len(result) == 1


def test_clean_dataframe_strips_html():
    """clean_dataframe nettoie le HTML des longues descriptions."""
    df = pd.DataFrame(
        {
            "uid": ["1"],
            "title_fr": ["Test"],
            "description_fr": ["Desc"],
            "longdescription_fr": ["<p>Texte <b>propre</b></p>"],
            "conditions_fr": [None],
            "keywords_fr": [None],
            "daterange_fr": [None],
            "location_name": [None],
            "location_address": [None],
            "location_city": [None],
        }
    )
    result = clean_dataframe(df)
    assert "<p>" not in result.iloc[0]["longdescription_fr"]


# === Tests build_event_text ===


def test_build_event_text_complete():
    """build_event_text construit un texte structuré complet."""
    row = pd.Series(
        {
            "title_fr": "Concert Jazz",
            "daterange_fr": "Samedi 15 mars 2026",
            "location_name": "Le Sunset",
            "location_address": "60 rue des Lombards",
            "location_city": "Paris",
            "conditions_fr": "Entrée libre",
            "keywords_fr": "jazz, musique",
            "description_fr": "Un super concert",
            "longdescription_fr": "",
        }
    )
    text = build_event_text(row)
    assert "Concert Jazz" in text
    assert "Samedi 15 mars" in text
    assert "Le Sunset" in text
    assert "Paris" in text
    assert "Entrée libre" in text
    assert "jazz" in text
    assert "Un super concert" in text


def test_build_event_text_minimal():
    """build_event_text fonctionne avec un minimum de données."""
    row = pd.Series(
        {
            "title_fr": "Événement test",
            "daterange_fr": "",
            "location_name": "",
            "location_address": "",
            "location_city": "",
            "conditions_fr": "",
            "keywords_fr": "",
            "description_fr": "",
            "longdescription_fr": "",
        }
    )
    text = build_event_text(row)
    assert "Événement test" in text


def test_build_event_text_normalizes_decorative_letters():
    """Les lettres « gras mathématiques » deviennent des lettres simples."""
    row = pd.Series(
        {
            "title_fr": "Festival",
            "daterange_fr": "",
            "location_name": "",
            "location_address": "",
            "location_city": "",
            "conditions_fr": "",
            "keywords_fr": "",
            "description_fr": "",
            "longdescription_fr": "𝗣𝗥𝗢𝗚𝗥𝗔𝗠𝗠𝗘 𝗖𝗢𝗠𝗣𝗟𝗘𝗧",
        }
    )
    text = build_event_text(row)
    assert "PROGRAMME COMPLET" in text


# === Tests fetch_events (fonctions unitaires) ===


def test_build_query_params_basic():
    """build_query_params construit les paramètres avec les bons filtres."""
    from scripts.fetch_events import build_query_params

    params = build_query_params(offset=0)
    assert params["limit"] == 100
    assert params["offset"] == 0
    assert "Île-de-France" in params["where"]


def test_build_query_params_with_cursor():
    """build_query_params ajoute le curseur de date quand fourni."""
    from scripts.fetch_events import build_query_params

    params = build_query_params(offset=50, date_upper_bound="2026-01-01")
    assert "2026-01-01" in params["where"]
    assert params["offset"] == 50


def test_date_limit_format():
    """_date_limit retourne une date au format YYYY-MM-DD."""
    import re

    from scripts.fetch_events import _date_limit

    result = _date_limit()
    assert re.match(r"\d{4}-\d{2}-\d{2}", result)


# === Tests build_index (fonctions unitaires) ===


def test_index_documents_reuses_known_embeddings():
    """Un chunk déjà vectorisé reprend son vecteur ; seul le nouveau part à l'API."""
    from langchain_core.documents import Document

    from scripts.build_index import index_documents

    known = Document(page_content="déjà vu", metadata={"uid": "1"})
    new = Document(page_content="nouveau", metadata={"uid": "2"})
    store = MagicMock()

    with (
        patch(
            "scripts.build_index.fetch_known_embeddings",
            return_value={"déjà vu": [0.1, 0.2]},
        ),
        patch("scripts.build_index.time.sleep"),
    ):
        index_documents(store, [known, new], MagicMock(), reuse_from="event_chunks")

    store.add_embeddings.assert_called_once_with(
        texts=["déjà vu"], embeddings=[[0.1, 0.2]], metadatas=[{"uid": "1"}]
    )
    store.add_documents.assert_called_once_with([new])


def test_fetch_known_embeddings_rejects_unsafe_table_name():
    """Le nom de la table source est inséré dans le SQL : il doit être simple."""
    from scripts.vector_db import fetch_known_embeddings

    with pytest.raises(ValueError, match="Table source"):
        fetch_known_embeddings(MagicMock(), "event_chunks; DROP TABLE x", ["texte"])


def test_chunks_to_documents():
    """chunks_to_documents convertit les chunks en Documents LangChain."""
    from scripts.build_index import chunks_to_documents

    chunks = [
        {
            "text": "Titre : Concert\nLieu : Paris",
            "metadata": {"title": "Concert", "city": "Paris"},
        },
        {
            "text": "Titre : Expo\nLieu : Lyon",
            "metadata": {"title": "Expo", "city": "Lyon"},
        },
    ]
    docs = chunks_to_documents(chunks)
    assert len(docs) == 2
    assert docs[0].page_content == "Titre : Concert\nLieu : Paris"
    assert docs[0].metadata["city"] == "Paris"


# === Tests rag_chain (fonctions unitaires) ===


def test_format_docs():
    """_format_docs formate les documents en texte lisible."""
    from scripts.rag_chain import _format_docs

    mock_doc = MagicMock()
    mock_doc.page_content = "Concert de jazz au Sunset"
    result = _format_docs([mock_doc])
    assert "Événement 1" in result
    assert "Concert de jazz" in result


def test_format_docs_multiple():
    """_format_docs numérote correctement les documents multiples."""
    from scripts.rag_chain import _format_docs

    docs = []
    for i in range(3):
        doc = MagicMock()
        doc.page_content = f"Événement {i}"
        docs.append(doc)
    result = _format_docs(docs)
    assert "Événement 1" in result
    assert "Événement 2" in result
    assert "Événement 3" in result


def test_get_api_key_missing():
    """get_mistral_api_key lève ValueError si la clé est absente."""
    import pytest

    from scripts.config import get_mistral_api_key

    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="MISTRAL_API_KEY"):
            get_mistral_api_key()


def test_get_api_key_present():
    """get_mistral_api_key retourne la clé si elle est définie."""
    from scripts.config import get_mistral_api_key

    with patch.dict("os.environ", {"MISTRAL_API_KEY": "sk-test123"}):
        assert get_mistral_api_key() == "sk-test123"
