import pytest
from fastapi.testclient import TestClient
from middleware.app import app
from middleware.db import init_db

# Initialize DB for tests (TestClient doesn't run startup hooks automatically outside a 'with' block)
init_db()

client = TestClient(app)

def test_valid_predict_request():
    response = client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]}, headers={"X-Session-ID": "test-session-1"})
    assert response.status_code == 200
    assert "X-Threat-Score" in response.headers
    assert "X-Defense-Tier" in response.headers

def test_wrong_feature_count():
    response = client.post("/predict", json={"features": [5.1, 3.5, 1.4]}) # Only 3 features
    assert response.status_code == 422

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "fingerprint_model_loaded" in data

def test_rapid_requests_trigger_defense():
    # Send 60 rapid requests to trigger high threat score / rate limits
    final_tier = 0
    for i in range(60):
        response = client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]}, headers={"X-Session-ID": "test-session-rapid"})
        if response.status_code == 200:
            tier = int(response.headers.get("X-Defense-Tier", 0))
            final_tier = max(final_tier, tier)
        elif response.status_code == 429:
            final_tier = 4
            break
            
    assert final_tier >= 3

def test_get_stats():
    # The IP used by TestClient is usually "testclient"
    response = client.get("/stats/testclient")
    assert response.status_code == 200
    data = response.json()
    assert "query_count" in data
    assert data["query_count"] > 0
