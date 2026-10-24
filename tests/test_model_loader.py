"""Testes do carregamento do modelo do Registry (MLflow real, em SQLite temporário)."""

from collections.abc import Callable

import pandas as pd
import pytest

from pos_mle_tech_challenge_2.serving.model_loader import (
    ModelLoader,
    ModelNotFoundError,
)
from pos_mle_tech_challenge_2.training.model_factory import ModelFactory
from pos_mle_tech_challenge_2.training.registry import ModelRegistry
from pos_mle_tech_challenge_2.training.tracker import ExperimentTracker

XY = tuple[pd.DataFrame, pd.Series]
MODEL_NAME = "test-model"


@pytest.fixture
def registry(mlflow_uri: str) -> ModelRegistry:
    return ModelRegistry(mlflow_uri, MODEL_NAME, "champion")


@pytest.fixture
def register_version(
    mlflow_uri: str, registry: ModelRegistry, make_xy: Callable[..., XY]
) -> Callable[..., int]:
    """Treina, loga e registra uma versão (com tags como as do trainer)."""
    tracker = ExperimentTracker(mlflow_uri, "serving-test")
    features, target = make_xy()

    def _register(threshold: str = "0.73", max_depth: int = 2) -> int:
        pipeline = ModelFactory(seed=1).build_pipeline({"max_depth": max_depth})
        pipeline.fit(features, target)
        with tracker.start_run() as run_id:
            tracker.log_metrics({"test_average_precision": 0.77})
            uri = tracker.log_model(pipeline, features)
        tags = {"threshold": threshold, "run_id": run_id, "data_md5": "abc123"}
        return registry.register(uri, tags=tags)

    return _register


@pytest.fixture
def loader(mlflow_uri: str) -> ModelLoader:
    return ModelLoader(mlflow_uri, MODEL_NAME, "champion")


def test_champion_is_resolved_to_its_version_number(
    loader: ModelLoader, registry: ModelRegistry, register_version: Callable[..., int]
) -> None:
    register_version()
    second = register_version()
    registry.promote(second)

    loaded = loader.load("champion")

    assert loaded.info.version == second
    assert loaded.info.requested == "champion"


def test_a_numbered_version_pins_that_model(
    loader: ModelLoader, registry: ModelRegistry, register_version: Callable[..., int]
) -> None:
    first = register_version()
    registry.promote(register_version())

    loaded = loader.load(str(first))

    assert loaded.info.version == first
    assert loaded.info.requested == str(first)


def test_info_comes_from_version_tags_and_run(
    loader: ModelLoader, registry: ModelRegistry, register_version: Callable[..., int]
) -> None:
    registry.promote(register_version(threshold="0.6"))

    info = loader.load("champion").info

    assert info.threshold == 0.6
    assert info.data_md5 == "abc123"
    assert info.run_id is not None
    assert info.test_average_precision == pytest.approx(0.77)
    assert info.name == MODEL_NAME


def test_loaded_pipeline_predicts_probabilities(
    loader: ModelLoader,
    registry: ModelRegistry,
    register_version: Callable[..., int],
    make_xy: Callable[..., XY],
) -> None:
    features, _ = make_xy()
    registry.promote(register_version())

    pipeline = loader.load("champion").pipeline

    assert pipeline.predict_proba(features).shape == (len(features), 2)


def test_missing_threshold_tag_falls_back_to_default(
    mlflow_uri: str,
    loader: ModelLoader,
    registry: ModelRegistry,
    make_xy: Callable[..., XY],
) -> None:
    features, target = make_xy()
    pipeline = (
        ModelFactory(seed=1).build_pipeline({"max_depth": 2}).fit(features, target)
    )
    tracker = ExperimentTracker(mlflow_uri, "serving-test")
    with tracker.start_run():
        uri = tracker.log_model(pipeline, features)
    registry.promote(registry.register(uri))

    assert loader.load("champion").info.threshold == 0.5


def test_unknown_version_is_reported_clearly(
    loader: ModelLoader, register_version: Callable[..., int]
) -> None:
    register_version()

    with pytest.raises(ModelNotFoundError, match="99"):
        loader.load("99")


def test_champion_without_alias_is_reported_clearly(
    loader: ModelLoader, register_version: Callable[..., int]
) -> None:
    register_version()  # registrado, mas nunca promovido

    with pytest.raises(ModelNotFoundError):
        loader.load("champion")


def test_model_that_was_never_registered_is_reported_clearly(
    loader: ModelLoader,
) -> None:
    with pytest.raises(ModelNotFoundError):
        loader.load("champion")


@pytest.mark.parametrize("requested", ["latest", "v3", "", "-1"])
def test_invalid_requested_version_is_rejected(
    loader: ModelLoader, requested: str
) -> None:
    with pytest.raises(ModelNotFoundError, match="inválido"):
        loader.load(requested)
