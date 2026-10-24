"""Testes do rastreamento de experimentos (MLflow Tracking)."""

from collections.abc import Callable

import mlflow
import pandas as pd
import pytest
from mlflow import MlflowClient

from pos_mle_tech_challenge_2.training.model_factory import ModelFactory
from pos_mle_tech_challenge_2.training.reporting import ReportBuilder
from pos_mle_tech_challenge_2.training.tracker import ExperimentTracker, flatten_params

XY = tuple[pd.DataFrame, pd.Series]


@pytest.fixture
def tracker(mlflow_uri: str) -> ExperimentTracker:
    return ExperimentTracker(mlflow_uri, "test-experiment")


def test_flatten_params_joins_nested_keys_with_dots() -> None:
    nested = {"seed": 42, "model": {"type": "rf", "params": {"max_depth": 3}}}

    assert flatten_params(nested) == {
        "seed": 42,
        "model.type": "rf",
        "model.params.max_depth": 3,
    }


def test_run_records_params_metrics_and_tags(
    tracker: ExperimentTracker, mlflow_uri: str
) -> None:
    with tracker.start_run() as run_id:
        tracker.log_params({"seed": 42, "model": {"params": {"max_depth": 3}}})
        tracker.log_metrics({"test_average_precision": 0.75})
        tracker.set_tags({"data_md5": "abc123"})

    run = MlflowClient(mlflow_uri).get_run(run_id)
    assert run.data.params["model.params.max_depth"] == "3"
    assert run.data.metrics["test_average_precision"] == 0.75
    assert run.data.tags["data_md5"] == "abc123"


def test_run_is_closed_after_the_block(tracker: ExperimentTracker) -> None:
    with tracker.start_run():
        assert mlflow.active_run() is not None

    assert mlflow.active_run() is None


def test_figure_is_saved_as_run_artifact(
    tracker: ExperimentTracker, mlflow_uri: str
) -> None:
    figure = ReportBuilder().precision_recall_figure(
        pd.Series([0, 0, 1, 1]), pd.Series([0.1, 0.4, 0.35, 0.8]).to_numpy()
    )

    with tracker.start_run() as run_id:
        tracker.log_figure(figure, "plots/precision_recall.png")

    artifacts = MlflowClient(mlflow_uri).list_artifacts(run_id, "plots")
    assert [artifact.path for artifact in artifacts] == ["plots/precision_recall.png"]


def test_logged_model_has_signature_and_reproduces_predictions(
    tracker: ExperimentTracker, make_xy: Callable[..., XY]
) -> None:
    features, target = make_xy()
    pipeline = (
        ModelFactory(seed=1)
        .build_pipeline("hist_gradient_boosting", {"max_depth": 3})
        .fit(features, target)
    )

    with tracker.start_run():
        model_uri = tracker.log_model(pipeline, features)

    loaded = mlflow.sklearn.load_model(model_uri)
    assert mlflow.models.get_model_info(model_uri).signature is not None
    pd.testing.assert_frame_equal(
        pd.DataFrame(loaded.predict_proba(features)),
        pd.DataFrame(pipeline.predict_proba(features)),
    )
