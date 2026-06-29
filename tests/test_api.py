"""Tests for the FastAPI application."""

from fastapi.testclient import TestClient

from facsimile.api.main import app

client = TestClient(app)


def test_health_returns_200() -> None:
    response = client.get("/health")
    assert response.status_code == 200


def test_health_returns_ok_status() -> None:
    data = client.get("/health").json()
    assert data["status"] == "ok"


def test_health_includes_version() -> None:
    data = client.get("/health").json()
    assert "version" in data
