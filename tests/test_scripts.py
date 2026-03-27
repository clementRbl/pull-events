"""Tests unitaires pour les scripts du pipeline de données."""

from unittest.mock import MagicMock, patch

import pandas as pd

from scripts.prepare_data import (
    build_event_text,
    clean_dataframe,
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
    """_get_api_key lève ValueError si la clé est absente."""
    import pytest

    from scripts.rag_chain import _get_api_key

    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="MISTRAL_API_KEY"):
            _get_api_key()


def test_get_api_key_present():
    """_get_api_key retourne la clé si elle est définie."""
    from scripts.rag_chain import _get_api_key

    with patch.dict("os.environ", {"MISTRAL_API_KEY": "sk-test123"}):
        assert _get_api_key() == "sk-test123"
