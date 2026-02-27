"""Tests for the REST API."""

from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.models.implementations import get_model


@pytest.fixture()
def client() -> TestClient:
    app = create_app()
    return TestClient(app)


@pytest.fixture()
def trained_model():
    model = get_model("random_forest", n_estimators=5, random_state=42)
    X = pd.DataFrame({"f0": [1, 2, 3], "f1": [4, 5, 6]})
    y = pd.Series([0, 1, 0])
    model.fit(X, y)
    return model


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_health_returns_200(self, client: TestClient) -> None:
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "version" in data

    def test_health_schema(self, client: TestClient) -> None:
        resp = client.get("/api/v1/health")
        data = resp.json()
        assert set(data.keys()) == {"status", "version"}


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------


class TestPredictEndpoint:
    def test_predict_no_model_returns_503(self, client: TestClient) -> None:
        with patch("src.api.routes.get_model", return_value=None):
            resp = client.post("/api/v1/predict", json={"features": {"f0": 1, "f1": 4}})
            assert resp.status_code == 503

    def test_predict_with_model(self, client: TestClient, trained_model) -> None:
        with patch("src.api.routes.get_model", return_value=trained_model):
            resp = client.post("/api/v1/predict", json={"features": {"f0": 1, "f1": 4}})
            assert resp.status_code == 200
            data = resp.json()
            assert "prediction" in data
            assert "model_version" in data

    def test_predict_invalid_body_returns_422(self, client: TestClient) -> None:
        resp = client.post("/api/v1/predict", json={"wrong": "payload"})
        assert resp.status_code == 422

    def test_predict_model_error_returns_500(self, client: TestClient, trained_model) -> None:
        def broken_predict(df):
            raise RuntimeError("boom")

        trained_model.predict = broken_predict
        with patch("src.api.routes.get_model", return_value=trained_model):
            resp = client.post("/api/v1/predict", json={"features": {"f0": 1, "f1": 4}})
            assert resp.status_code == 500
