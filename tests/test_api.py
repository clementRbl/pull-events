"""Tests unitaires pour l'API REST Puls-Events."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api.main import app, limiter, rebuild_lock
from scripts.vector_db import VectorStoreUnavailableError

client = TestClient(app)


@pytest.fixture(autouse=True)
def open_api(monkeypatch):
    """Par défaut, API ouverte comme en local, même si .env définit API_KEY.

    Les compteurs de limite de débit repartent de zéro à chaque test.
    """
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("ENABLE_REBUILD", raising=False)
    limiter.reset()


@pytest.fixture
def rebuild_enabled(monkeypatch):
    """Autorise /rebuild, désactivé par défaut."""
    monkeypatch.setenv("ENABLE_REBUILD", "true")


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
def test_ask_vector_store_unavailable(mock_ask):
    """POST /ask retourne 503 si la base vectorielle est indisponible."""
    mock_ask.side_effect = VectorStoreUnavailableError("PostgreSQL injoignable")

    response = client.post("/ask", json={"question": "Test ?"})

    assert response.status_code == 503
    assert "Base vectorielle indisponible" in response.json()["detail"]


@patch("api.main.ask")
def test_ask_missing_api_key(mock_ask):
    """POST /ask retourne 500 si la clé API Mistral est manquante."""
    mock_ask.side_effect = ValueError("MISTRAL_API_KEY non définie")

    response = client.post("/ask", json={"question": "Test ?"})

    assert response.status_code == 500
    assert "Erreur de configuration" in response.json()["detail"]


@patch("api.main.ask")
def test_ask_errors_do_not_leak_details(mock_ask):
    """Le message d'erreur interne reste dans les logs, pas dans la réponse."""
    mock_ask.side_effect = ValueError("postgresql://admin:secret@host/db")

    response = client.post("/ask", json={"question": "Test ?"})

    assert "secret" not in response.text


@patch("api.main.ask")
def test_ask_rate_limited_per_ip(mock_ask):
    """Au-delà de 10 questions par minute depuis une IP : 429."""
    mock_ask.return_value = {"answer": "Réponse", "sources": []}

    statuses = [
        client.post("/ask", json={"question": "Test ?"}).status_code for _ in range(11)
    ]

    assert statuses[:10] == [200] * 10
    assert statuses[10] == 429


@patch("api.main.ask")
def test_ask_rate_limit_keys_on_last_forwarded_address(mock_ask):
    """Seule la dernière adresse de X-Forwarded-For (ajoutée par l'ingress) compte.

    Changer le début de l'en-tête ne contourne pas la limite ; un autre
    client réel (dernière adresse différente) a son propre compteur.
    """
    mock_ask.return_value = {"answer": "Réponse", "sources": []}

    def post(forwarded_for: str) -> int:
        response = client.post(
            "/ask",
            json={"question": "Test ?"},
            headers={"X-Forwarded-For": forwarded_for},
        )
        return response.status_code

    first_ten = [post(f"10.0.0.{i}, 203.0.113.9") for i in range(10)]

    assert first_ten == [200] * 10
    assert post("10.0.0.99, 203.0.113.9") == 429
    assert post("203.0.113.9, 198.51.100.1") == 200


# === Protection par clé d'API (déploiement public) ===


@pytest.mark.parametrize(
    "headers",
    [{}, {"X-API-Key": "mauvaise-cle"}],
    ids=["sans-cle", "mauvaise-cle"],
)
@patch("api.main.ask")
def test_ask_rejects_missing_or_wrong_api_key(mock_ask, monkeypatch, headers):
    """Avec API_KEY, /ask refuse sans la bonne clé et n'appelle pas le LLM."""
    monkeypatch.setenv("API_KEY", "cle-secrete")

    response = client.post("/ask", json={"question": "Test ?"}, headers=headers)

    assert response.status_code == 401
    mock_ask.assert_not_called()


@patch("api.main.ask")
def test_ask_accepts_right_api_key(mock_ask, monkeypatch):
    """Avec la bonne clé dans X-API-Key, /ask répond normalement."""
    monkeypatch.setenv("API_KEY", "cle-secrete")
    mock_ask.return_value = {"answer": "Réponse", "sources": []}

    response = client.post(
        "/ask", json={"question": "Test ?"}, headers={"X-API-Key": "cle-secrete"}
    )

    assert response.status_code == 200


def test_health_stays_open_with_api_key(monkeypatch):
    """/health reste accessible sans clé : la sonde d'Azure doit pouvoir l'appeler."""
    monkeypatch.setenv("API_KEY", "cle-secrete")

    assert client.get("/health").status_code == 200


# === Tests /rebuild ===


@patch("subprocess.run")
def test_rebuild_disabled_by_default(mock_run):
    """Sans ENABLE_REBUILD=true, /rebuild est refusé sans lancer aucun script."""
    response = client.post("/rebuild")

    assert response.status_code == 403
    mock_run.assert_not_called()


@patch("subprocess.run")
def test_rebuild_refused_while_running(mock_run, rebuild_enabled):
    """Une seule reconstruction à la fois : la seconde reçoit 409."""
    with rebuild_lock:
        response = client.post("/rebuild")

    assert response.status_code == 409
    mock_run.assert_not_called()


@patch("subprocess.run")
def test_rebuild_success(mock_run, rebuild_enabled):
    """POST /rebuild retourne un succès quand les scripts s'exécutent."""
    mock_run.return_value = type("Result", (), {"returncode": 0, "stderr": ""})()

    with patch("api.main.ask"):  # Éviter le chargement réel du singleton
        response = client.post("/rebuild")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert mock_run.call_count == 3  # 3 scripts exécutés


@patch("subprocess.run")
def test_rebuild_script_failure(mock_run, rebuild_enabled):
    """POST /rebuild retourne 500 si un script échoue."""
    mock_run.return_value = type(
        "Result", (), {"returncode": 1, "stderr": "Erreur réseau"}
    )()

    response = client.post("/rebuild")

    assert response.status_code == 500
    assert "Échec" in response.json()["detail"]
    assert "Erreur réseau" not in response.json()["detail"]


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
