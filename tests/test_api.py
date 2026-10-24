"""Testes da API de inferência."""

from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from pos_mle_tech_challenge_2.serving.api import create_app, create_app_from_env
from pos_mle_tech_challenge_2.serving.examples import EXAMPLE_SESSIONS
from pos_mle_tech_challenge_2.serving.model_loader import (
    LoadedModel,
    ModelNotFoundError,
)
from pos_mle_tech_challenge_2.serving.schemas import ModelInfo
from pos_mle_tech_challenge_2.training.model_factory import ModelFactory
from pos_mle_tech_challenge_2.training.registry import ModelRegistry
from pos_mle_tech_challenge_2.training.tracker import ExperimentTracker

XY = tuple[pd.DataFrame, pd.Series]
SESSION: dict[str, Any] = EXAMPLE_SESSIONS["navegacao_curta"]


class FakePipeline:
    """Probabilidade = PageValues / 10 (limitada a 1)."""

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        positive = np.clip(features["PageValues"].to_numpy() / 10, 0, 1)
        return np.column_stack([1 - positive, positive])


def fake_model() -> LoadedModel:
    info = ModelInfo(
        name="purchase-propensity-classifier",
        version=3,
        requested="champion",
        run_id="run-abc",
        threshold=0.73,
        data_md5="abc123",
        git_commit="deadbeef",
        test_average_precision=0.7574,
        loaded_at=datetime.now(UTC),
    )
    return LoadedModel(FakePipeline(), info)


def client_for(load_model: Callable[[], LoadedModel]) -> Iterator[TestClient]:
    with TestClient(create_app(load_model)) as client:
        yield client


@pytest.fixture
def client() -> Iterator[TestClient]:
    yield from client_for(fake_model)


def failing(error: Exception) -> Callable[[], LoadedModel]:
    def _load() -> LoadedModel:
        raise error

    return _load


def test_health_reports_the_loaded_version(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_version": 3}


def test_model_endpoint_describes_the_loaded_model(client: TestClient) -> None:
    body = client.get("/model").json()

    assert body["name"] == "purchase-propensity-classifier"
    assert body["version"] == 3
    assert body["requested"] == "champion"
    assert body["threshold"] == 0.73
    assert body["data_md5"] == "abc123"
    assert body["test_average_precision"] == 0.7574


def test_predict_returns_probability_and_class_per_session(
    client: TestClient,
) -> None:
    high = {**SESSION, "PageValues": 8.0}
    low = {**SESSION, "PageValues": 1.0}

    response = client.post("/predict", json={"sessions": [high, low]})

    assert response.status_code == 200
    assert response.json() == {
        "model_version": 3,
        "threshold": 0.73,
        "predictions": [
            {"probability": 0.8, "predicted_class": 1},
            {"probability": 0.1, "predicted_class": 0},
        ],
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"sessions": []},
        {},
        {"sessions": [{**SESSION, "PageValues": -1}]},
        {"sessions": [{**SESSION, "Month": "Novv"}]},
        {"sessions": [{**SESSION, "Revenue": True}]},
        {"sessions": [{k: v for k, v in SESSION.items() if k != "Browser"}]},
    ],
)
def test_invalid_payloads_get_a_422(client: TestClient, payload: dict) -> None:
    assert client.post("/predict", json=payload).status_code == 422


@pytest.mark.parametrize(
    "error",
    [
        ModelNotFoundError("Modelo 'x' (champion) não encontrado no Registry"),
        ConnectionError("MLflow fora do ar"),
    ],
)
def test_without_a_model_the_service_stays_up_but_not_ready(
    error: Exception,
) -> None:
    for client in client_for(failing(error)):
        for method, path in [("get", "/health"), ("get", "/model")]:
            response = getattr(client, method)(path)
            assert response.status_code == 503
            assert str(error) in response.json()["detail"]
        predict = client.post("/predict", json={"sessions": [SESSION]})
        assert predict.status_code == 503


def test_swagger_lists_the_named_examples(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()

    body = schema["paths"]["/predict"]["post"]["requestBody"]
    examples = body["content"]["application/json"]["examples"]
    assert set(examples) == {"lote_com_as_tres_sessoes", *EXAMPLE_SESSIONS}
    assert client.get("/docs").status_code == 200


def test_all_documentation_examples_are_accepted_by_predict(
    client: TestClient,
) -> None:
    schema = client.get("/openapi.json").json()
    examples = schema["paths"]["/predict"]["post"]["requestBody"]["content"][
        "application/json"
    ]["examples"]

    for name, example in examples.items():
        response = client.post("/predict", json=example["value"])
        assert response.status_code == 200, name


# --- integração: create_app_from_env com um MLflow real (SQLite temporário) ---


@pytest.fixture
def registered_models(
    mlflow_uri: str, monkeypatch: pytest.MonkeyPatch, make_xy: Callable[..., XY]
) -> int:
    """Registra duas versões; a 2 vira champion. Devolve a versão fixa (1)."""
    registry = ModelRegistry(mlflow_uri, "test-model", "champion")
    tracker = ExperimentTracker(mlflow_uri, "api-test")
    features, target = make_xy()
    for depth in (1, 3):
        pipeline = ModelFactory(seed=1).build_pipeline({"max_depth": depth})
        pipeline.fit(features, target)
        with tracker.start_run() as run_id:
            uri = tracker.log_model(pipeline, features)
        version = registry.register(uri, tags={"threshold": "0.6", "run_id": run_id})
    registry.promote(version)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", mlflow_uri)
    monkeypatch.setenv("MLFLOW_REGISTERED_MODEL_NAME", "test-model")
    return 1


def test_app_from_env_serves_the_champion_by_default(
    registered_models: int,
) -> None:
    with TestClient(create_app_from_env()) as client:
        assert client.get("/health").json() == {"status": "ok", "model_version": 2}
        response = client.post("/predict", json={"sessions": [SESSION]})

    assert response.status_code == 200
    assert response.json()["threshold"] == 0.6
    assert response.json()["predictions"][0]["predicted_class"] in (0, 1)


def test_app_from_env_pins_the_version_from_model_version(
    registered_models: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MODEL_VERSION", str(registered_models))

    with TestClient(create_app_from_env()) as client:
        body = client.get("/model").json()

    assert body["version"] == registered_models
    assert body["requested"] == str(registered_models)


def test_app_from_env_reports_a_nonexistent_version(
    registered_models: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MODEL_VERSION", "99")

    with TestClient(create_app_from_env()) as client:
        response = client.get("/health")

    assert response.status_code == 503
    assert "99" in response.json()["detail"]
