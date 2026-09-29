import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_dashboard_summary():
    response = client.get("/api/dashboard/summary")
    assert response.status_code == 200
    assert "total_creators" in response.json()

def test_descriptive_stats():
    response = client.get("/api/statistics/descriptive?variable=subscribers")
    assert response.status_code == 200
    assert "mean" in response.json()

def test_ml_models():
    response = client.get("/api/ml/models")
    assert response.status_code == 200
    assert len(response.json()) > 0
