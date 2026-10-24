"""Testes do registro de modelos e do alias de champion."""

from collections.abc import Callable

import pandas as pd
import pytest
from mlflow import MlflowClient
from sklearn.pipeline import Pipeline

from pos_mle_tech_challenge_2.training.model_factory import ModelFactory
from pos_mle_tech_challenge_2.training.registry import ModelRegistry
from pos_mle_tech_challenge_2.training.tracker import ExperimentTracker

XY = tuple[pd.DataFrame, pd.Series]
MODEL_NAME = "test-model"


@pytest.fixture
def registry(mlflow_uri: str) -> ModelRegistry:
    return ModelRegistry(mlflow_uri, MODEL_NAME, "champion")


@pytest.fixture
def log_model(
    mlflow_uri: str, make_xy: Callable[..., XY]
) -> Callable[..., tuple[str, Pipeline]]:
    """Treina e loga um modelo; devolve (uri, pipeline)."""
    tracker = ExperimentTracker(mlflow_uri, "registry-experiment")
    features, target = make_xy()

    def _log(max_depth: int = 2) -> tuple[str, Pipeline]:
        pipeline = (
            ModelFactory(seed=1)
            .build_pipeline("hist_gradient_boosting", {"max_depth": max_depth})
            .fit(features, target)
        )
        with tracker.start_run():
            return tracker.log_model(pipeline, features), pipeline

    return _log


def test_there_is_no_champion_before_any_registration(
    registry: ModelRegistry,
) -> None:
    assert registry.champion_version() is None
    with pytest.raises(LookupError):
        registry.load_champion()


def test_registering_creates_incremental_versions(
    registry: ModelRegistry, log_model: Callable[..., tuple[str, Pipeline]]
) -> None:
    first = registry.register(log_model()[0])
    second = registry.register(log_model()[0])

    assert (first, second) == (1, 2)


def test_registered_model_without_alias_has_no_champion(
    registry: ModelRegistry, log_model: Callable[..., tuple[str, Pipeline]]
) -> None:
    registry.register(log_model()[0])

    assert registry.champion_version() is None


def test_promote_moves_the_champion_alias(
    registry: ModelRegistry, log_model: Callable[..., tuple[str, Pipeline]]
) -> None:
    first = registry.register(log_model()[0])
    second = registry.register(log_model()[0])

    registry.promote(first)
    assert registry.champion_version() == first
    registry.promote(second)

    assert registry.champion_version() == second


def test_load_champion_returns_the_promoted_model(
    registry: ModelRegistry,
    log_model: Callable[..., tuple[str, Pipeline]],
    make_xy: Callable[..., XY],
) -> None:
    features, _ = make_xy()
    uri, pipeline = log_model()
    registry.promote(registry.register(uri))

    champion = registry.load_champion()

    pd.testing.assert_frame_equal(
        pd.DataFrame(champion.predict_proba(features)),
        pd.DataFrame(pipeline.predict_proba(features)),
    )


def test_version_tags_are_stored(
    registry: ModelRegistry,
    log_model: Callable[..., tuple[str, Pipeline]],
    mlflow_uri: str,
) -> None:
    version = registry.register(log_model()[0], tags={"threshold": "0.73"})

    stored = MlflowClient(mlflow_uri).get_model_version(MODEL_NAME, str(version))
    assert stored.tags["threshold"] == "0.73"
