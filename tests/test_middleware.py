from datetime import timedelta

from middleware.db import QueryLog, DefenseLog, Ban, utcnow

IRIS = [5.1, 3.5, 1.4, 0.2]


def predict(client, features=IRIS):
    return client.post("/predict", json={"features": features}, headers={"X-Session-ID": "test"})


def test_valid_request_returns_real_answer(client, fixed_score):
    fixed_score(0.0)
    response = predict(client)
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "iris-random-forest-v1"
    assert body["prediction"] == 0


def test_threat_headers_hidden_by_default(client, fixed_score):
    fixed_score(0.0)
    response = predict(client)
    assert "X-Threat-Score" not in response.headers
    assert "X-Defense-Tier" not in response.headers


def test_threat_headers_sent_when_enabled(client, fixed_score, monkeypatch):
    monkeypatch.setenv("EXPOSE_THREAT_HEADERS", "1")
    fixed_score(0.4)
    response = predict(client)
    assert response.headers["X-Threat-Score"] == "0.4"
    assert response.headers["X-Defense-Tier"] == "1"


def test_wrong_feature_count(client):
    assert predict(client, [5.1, 3.5, 1.4]).status_code == 422


def test_rejects_non_finite_features(client):
    for bad in ("NaN", "Infinity", "-Infinity"):
        response = client.post(
            "/predict",
            content='{"features": [%s, 3.5, 1.4, 0.2]}' % bad,
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422, bad


def test_rejects_out_of_range_features(client):
    assert predict(client, [-1.0, 3.5, 1.4, 0.2]).status_code == 422
    assert predict(client, [5.1, 3.5, 1.4, 1e6]).status_code == 422


def test_health_does_not_leak_internal_url(client):
    data = client.get("/health").json()
    assert data["status"] == "ok"
    assert "protected_model_url" not in data


def test_fake_tier_looks_real_and_logs_real_output(client, fixed_score, protected_model, db):
    fixed_score(0.8)  # tier 3
    response = predict(client)
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "iris-random-forest-v1"
    assert body["prediction"] == body["probabilities"].index(max(body["probabilities"]))
    # The real model is still queried so features use true outputs
    assert len(protected_model) == 1
    row = db.query(QueryLog).one()
    assert row.raw_prediction == 0
    assert row.raw_probabilities is not None
    assert row.tier_applied == 3


def test_tier4_bans_and_ban_outlasts_the_score(client, fixed_score, protected_model, db):
    fixed_score(0.95)
    first = predict(client)
    assert first.status_code == 429
    assert 295 <= int(first.headers["Retry-After"]) <= 300
    assert protected_model == []  # blocked requests never reach the model

    # Score drops, but the ban holds
    fixed_score(0.0)
    second = predict(client)
    assert second.status_code == 429
    assert protected_model == []

    # Blocked requests are logged, so the window keeps seeing the attack
    assert db.query(QueryLog).filter(QueryLog.tier_applied == 4).count() == 2
    assert db.query(DefenseLog).filter(DefenseLog.action_taken == "BLOCK").count() == 2
    assert client.get("/stats/testclient").json()["is_blocked"] is True


def test_expired_ban_lets_client_back_in(client, fixed_score, db):
    db.add(Ban(client_ip="testclient", strikes=1, banned_until=utcnow() - timedelta(seconds=1)))
    db.commit()
    fixed_score(0.0)
    assert predict(client).status_code == 200
    assert client.get("/stats/testclient").json()["is_blocked"] is False


def test_tier2_rate_limit_is_not_a_ban(client, fixed_score, db):
    fixed_score(0.6)  # tier 2; default limit is 5 per window
    statuses = [predict(client).status_code for _ in range(6)]
    assert statuses == [200] * 5 + [429]
    assert db.query(Ban).count() == 0
    assert db.query(DefenseLog).filter(DefenseLog.action_taken == "RATE_LIMIT").count() == 1


def test_rapid_grid_sweep_gets_blocked(client):
    # Real fingerprint model, real protected model: a fast sweep must escalate
    statuses = []
    for a in range(10):
        for b in range(4):
            statuses.append(predict(client, [0.5 + 0.7 * a, 0.5 + 1.5 * b, 2.0, 1.0]).status_code)
    assert 429 in statuses
