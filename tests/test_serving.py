"""Integration tests for the FastAPI prediction service.

We use httpx's TestClient (FastAPI's recommended path) and the existing
registered model. If no model is registered, the service should respond
with /health = ok but no_model status, and 503 from other endpoints.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from gridsentinel.predict.registry import load_latest
from gridsentinel.serving.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint_returns_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["version"]
    assert data["status"] in ("ok", "no_model")


def test_version_endpoint_when_model_loaded(client):
    if load_latest() is None:
        pytest.skip("no registered model")
    resp = client.get("/version")
    assert resp.status_code == 200
    data = resp.json()
    assert "feature_names" in data
    assert data["n_features"] == len(data["feature_names"])


def test_predict_with_explicit_features(client):
    if load_latest() is None:
        pytest.skip("no registered model")
    # Build a zeroed-out feature dict from the model's declared schema.
    v = client.get("/version").json()
    feats = {name: 0.0 for name in v["feature_names"]}
    feats[v["feature_names"][0]] = 1.0  # nudge one feature so it's not all zero
    resp = client.post("/predict", json={"features": feats})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert 0.0 <= data["failure_risk_24h"] <= 1.0
    assert 0.0 <= data["anomaly_score"] <= 1.0


def test_predict_rejects_empty_request(client):
    if load_latest() is None:
        pytest.skip("no registered model")
    resp = client.post("/predict", json={})
    assert resp.status_code == 400


def test_at_risk_returns_leaderboard(client):
    if load_latest() is None:
        pytest.skip("no registered model")
    resp = client.get("/at-risk?n=5")
    assert resp.status_code == 200
    data = resp.json()
    assert data["n"] <= 5
    if data["nodes"]:
        # confirm risks are sorted descending
        risks = [n["failure_risk_24h"] for n in data["nodes"]]
        assert risks == sorted(risks, reverse=True)


def test_explain_returns_top_features(client):
    if load_latest() is None:
        pytest.skip("no registered model")
    v = client.get("/version").json()
    feats = {name: 0.0 for name in v["feature_names"]}
    resp = client.post("/explain", json={"features": feats, "top_n": 4})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["top_features"]) <= 4
    if data["top_features"]:
        entry = data["top_features"][0]
        assert "feature" in entry and "shap" in entry
