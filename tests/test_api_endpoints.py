"""
tests/test_api_endpoints.py — FastAPI endpoint verification tests using TestClient.
"""

import pytest
from fastapi.testclient import TestClient
from proxy.model import app

client = TestClient(app)


def test_health_endpoint():
    """Verify /health returns 200 and reports runtime status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "onnx" in data
    assert isinstance(data["onnx"], bool)


def test_score_endpoint_benign_request():
    """Verify /score correctly labels clean traffic with browser UA as benign."""
    payload = {
        "method": "GET",
        "path": "/index.php",
        "query": "view=home",
        "body": "",
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Firefox/149.0",
        "cookie": "PHPSESSID=session_clean_123"
    }
    response = client.post("/score", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "score" in data
    assert "blocked" in data
    assert "latency_ms" in data
    assert data["blocked"] is False
    assert data["label"] == "benign"
    assert data["score"] < 0.5


def test_score_endpoint_malicious_request():
    """Verify /score detects SQL injection payload and marks as blocked."""
    payload = {
        "method": "POST",
        "path": "/login.php",
        "query": "",
        "body": "username=admin' OR 1=1--",
        "user_agent": "",
        "cookie": ""
    }
    response = client.post("/score", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["blocked"] is True
    assert data["label"] == "attack"
    assert data["score"] >= 0.5
    assert data["windows"] >= 1


def test_batch_score_endpoint():
    """Verify /batch_score handles mixed payloads in a single request."""
    batch_payload = {
        "payloads": [
            "admin' OR 1=1--",
            "1; DROP TABLE users; --",
            "<script>alert('xss')</script>"
        ],
        "threshold": 0.5
    }
    response = client.post("/batch_score", json=batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 3
    assert data["malicious_count"] + data["benign_count"] == 3
    assert len(data["results"]) == 3
    for res in data["results"]:
        assert "score" in res
        assert "label" in res
        assert "latency_ms" in res
        assert 0.0 <= res["score"] <= 1.0


def test_batch_score_abuse_prevention():
    """Verify /batch_score rejects payloads exceeding the 500-item limit."""
    excessive_payload = {
        "payloads": ["test"] * 501,
        "threshold": 0.5
    }
    response = client.post("/batch_score", json=excessive_payload)
    assert response.status_code == 400
    assert "Max 500 payloads" in response.json()["detail"]
