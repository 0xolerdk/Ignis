"""Basic health check test placeholder."""

from backend.app.main import create_app
from fastapi.testclient import TestClient


def test_health_endpoint() -> None:
    app = create_app()
    client = TestClient(app)
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
