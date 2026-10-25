"""Tests du mode API de l'évaluation (questions posées en HTTP à une API déployée)."""

from unittest.mock import MagicMock, patch

from scripts.evaluate_rag import (
    ApiResult,
    ask_api,
    run_api_on_test_set,
    summarize_api,
)


def _fake_response(status_code: int, body: dict | None = None) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = body or {}
    return response


def test_ask_api_sends_key_in_header_and_reads_sources():
    """La clé part dans l'en-tête X-API-Key, et les titres des sources sont relevés."""
    body = {"answer": "Oui.", "sources": [{"title": "Expo A"}, {"title": "Expo B"}]}
    with patch("scripts.evaluate_rag.requests.post") as post:
        post.return_value = _fake_response(200, body)
        result = ask_api("https://api.example/", "secret", "Des expos ?", timeout=5)

    post.assert_called_once()
    assert post.call_args.args[0] == "https://api.example/ask"
    assert post.call_args.kwargs["headers"] == {"X-API-Key": "secret"}
    assert result.status_code == 200
    assert result.answer == "Oui."
    assert result.sources == ["Expo A", "Expo B"]
    assert result.seconds >= 0


def test_ask_api_keeps_error_status_without_parsing_body():
    """Un 401 est conservé tel quel : réponse vide, pas d'exception."""
    with patch("scripts.evaluate_rag.requests.post") as post:
        post.return_value = _fake_response(401)
        result = ask_api("https://api.example", "bad", "Q ?", timeout=5)

    assert result.status_code == 401
    assert result.answer == ""
    assert result.sources == []
    post.return_value.json.assert_not_called()


def test_run_api_pauses_between_questions_but_not_after_the_last():
    """La pause respecte la limite de débit : n-1 pauses pour n questions."""
    test_set = [{"question": "A"}, {"question": "B"}, {"question": "C"}]
    with (
        patch("scripts.evaluate_rag.requests.post") as post,
        patch("scripts.evaluate_rag.time.sleep") as sleep,
    ):
        post.return_value = _fake_response(200, {"answer": "", "sources": []})
        results = run_api_on_test_set(test_set, "https://api.example", "k", 7.0)

    assert [r.question for r in results] == ["A", "B", "C"]
    assert sleep.call_count == 2
    sleep.assert_called_with(7.0)


def test_summarize_api_counts_success_and_latency():
    """Le résumé compte les HTTP 200 et donne moyenne, médiane et p95."""
    results = [
        ApiResult("a", 200, 1.0, "", []),
        ApiResult("b", 200, 3.0, "", []),
        ApiResult("c", 429, 0.5, "", []),
    ]
    summary = summarize_api(results)

    assert summary["questions"] == 3
    assert summary["http_200"] == 2
    assert summary["latency_seconds"]["total_mean"] == 1.5
    assert summary["latency_seconds"]["total_p50"] == 1.0
