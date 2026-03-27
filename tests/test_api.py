"""Tests unitaires pour l'API REST Puls-Events."""

from unittest.mock import patch

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


# === Tests /health ===


def test_health_check():
    """GET /health retourne status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# === Tests /ask ===


@patch("api.main.ask")
def test_ask_returns_answer(mock_ask):
    """POST /ask retourne une réponse structurée avec sources."""
    mock_ask.return_value = {
        "answer": "Il y a un concert de jazz au Sunset, samedi 28 mars.",
        "sources": [
            {
                "title": "Concert Jazz au Sunset",
                "city": "Paris",
                "date_start": "2026-03-28",
                "date_end": "2026-03-28",
                "url": "https://openagenda.com/event/123",
                "excerpt": "Concert de jazz au Sunset...",
            }
        ],
    }

    response = client.post("/ask", json={"question": "Concerts à Paris ?"})

    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert len(data["sources"]) == 1
    assert data["sources"][0]["city"] == "Paris"
    mock_ask.assert_called_once_with("Concerts à Paris ?")


@patch("api.main.ask")
def test_ask_multiple_sources(mock_ask):
    """POST /ask peut retourner plusieurs sources."""
    mock_ask.return_value = {
        "answer": "Voici deux événements.",
        "sources": [
            {
                "title": "Événement 1",
                "city": "Paris",
                "date_start": "2026-04-01",
                "date_end": "2026-04-01",
                "url": "https://example.com/1",
                "excerpt": "Premier événement...",
            },
            {
                "title": "Événement 2",
                "city": "Versailles",
                "date_start": "2026-04-02",
                "date_end": "2026-04-02",
                "url": "https://example.com/2",
                "excerpt": "Deuxième événement...",
            },
        ],
    }

    response = client.post("/ask", json={"question": "Événements cette semaine ?"})

    assert response.status_code == 200
    assert len(response.json()["sources"]) == 2


def test_ask_empty_question():
    """POST /ask avec question vide retourne 422."""
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422


def test_ask_missing_question():
    """POST /ask sans champ question retourne 422."""
    response = client.post("/ask", json={})
    assert response.status_code == 422


def test_ask_no_body():
    """POST /ask sans body retourne 422."""
    response = client.post("/ask")
    assert response.status_code == 422


@patch("api.main.ask")
def test_ask_index_not_found(mock_ask):
    """POST /ask retourne 503 si l'index Faiss est absent."""
    mock_ask.side_effect = FileNotFoundError("Index introuvable")

    response = client.post("/ask", json={"question": "Test ?"})

    assert response.status_code == 503
    assert "Index Faiss indisponible" in response.json()["detail"]


@patch("api.main.ask")
def test_ask_missing_api_key(mock_ask):
    """POST /ask retourne 500 si la clé API Mistral est manquante."""
    mock_ask.side_effect = ValueError("MISTRAL_API_KEY non définie")

    response = client.post("/ask", json={"question": "Test ?"})

    assert response.status_code == 500
    assert "Erreur de configuration" in response.json()["detail"]


# === Tests /rebuild ===


@patch("subprocess.run")
def test_rebuild_success(mock_run):
    """POST /rebuild retourne un succès quand les scripts s'exécutent."""
    mock_run.return_value = type("Result", (), {"returncode": 0, "stderr": ""})()

    with patch("api.main.ask"):  # Éviter le chargement réel du singleton
        response = client.post("/rebuild")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert mock_run.call_count == 3  # 3 scripts exécutés


@patch("subprocess.run")
def test_rebuild_script_failure(mock_run):
    """POST /rebuild retourne 500 si un script échoue."""
    mock_run.return_value = type(
        "Result", (), {"returncode": 1, "stderr": "Erreur réseau"}
    )()

    response = client.post("/rebuild")

    assert response.status_code == 500
    assert "Échec" in response.json()["detail"]


# === Test documentation Swagger ===


def test_swagger_docs_accessible():
    """GET /docs (Swagger UI) est accessible."""
    response = client.get("/docs")
    assert response.status_code == 200


def test_openapi_schema():
    """GET /openapi.json retourne le schéma OpenAPI valide."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert schema["info"]["title"] == "Puls-Events API"
    assert "/ask" in schema["paths"]
    assert "/rebuild" in schema["paths"]
    assert "/health" in schema["paths"]
