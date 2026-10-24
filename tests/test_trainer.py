"""Testes do treino ponta a ponta: runs, registro e portão de promoção."""

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient

from pos_mle_tech_challenge_2.config import (
    TEST_FILENAME,
    TRAIN_FILENAME,
    Settings,
    load_settings,
)
from pos_mle_tech_challenge_2.data.splitter import DataSplitter
from pos_mle_tech_challenge_2.params import (
    Params,
    SplitParams,
    TrainParams,
)
from pos_mle_tech_challenge_2.pipelines.train import (
    SUMMARY_METRICS,
    TrainPipeline,
    build_trainer,
)
from pos_mle_tech_challenge_2.training.trainer import TrainingResult

STRONG = {"max_depth": 3, "max_iter": 50}
WEAK = {"max_depth": 1, "max_iter": 1}  # um único toco: modelo propositalmente fraco


def make_params(model: dict, primary_metric: str = "average_precision") -> Params:
    return Params(
        seed=42,
        split=SplitParams(test_size=0.25, drop_duplicates=True),
        train=TrainParams(cv_folds=3, primary_metric=primary_metric, threshold=0.6),
        model=model,
    )


@pytest.fixture
def settings(
    tmp_path: Path,
    mlflow_uri: str,
    monkeypatch: pytest.MonkeyPatch,
    make_sessions: Callable[..., pd.DataFrame],
) -> Settings:
    """Dados processados sintéticos (com sinal em PageValues) e MLflow isolado."""
    df = make_sessions(n_rows=600, positive_rate=0.2)
    noise = np.random.default_rng(1).normal(0, 0.25, len(df))
    df["Revenue"] = df["PageValues"] + noise > 0.75
    train, test = DataSplitter(test_size=0.25, seed=42).split(df)
    processed = tmp_path / "processed"
    processed.mkdir()
    train.to_csv(processed / TRAIN_FILENAME, index=False)
    test.to_csv(processed / TEST_FILENAME, index=False)
    monkeypatch.setenv("DATA_PROCESSED_DIR", str(processed))
    monkeypatch.setenv("DATA_RAW_PATH", str(tmp_path / "raw.csv"))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", mlflow_uri)
    return load_settings(tmp_path / "missing.env")


def run_training(settings: Settings, model: dict) -> TrainingResult:
    return build_trainer(settings, make_params(model)).run()


def test_first_run_is_registered_and_becomes_champion(
    settings: Settings, mlflow_uri: str
) -> None:
    result = run_training(settings, STRONG)

    assert result.version == 1
    assert result.decision.promote
    assert result.decision.reason == "primeiro modelo"
    registry = MlflowClient(mlflow_uri)
    model = registry.get_registered_model(settings.mlflow_registered_model_name)
    assert int(model.aliases[settings.mlflow_champion_alias]) == 1


def test_run_records_params_metrics_figures_and_tags(
    settings: Settings, mlflow_uri: str
) -> None:
    result = run_training(settings, STRONG)

    client = MlflowClient(mlflow_uri)
    run = client.get_run(result.run_id)
    assert run.data.params["model.max_depth"] == "3"
    assert run.data.params["train.threshold"] == "0.6"
    for metric in (*SUMMARY_METRICS, "test_f1_at_0.5", "test_f1_at_0.6"):
        assert metric in run.data.metrics, metric
    plots = {a.path for a in client.list_artifacts(result.run_id, "plots")}
    assert plots == {"plots/precision_recall.png", "plots/confusion_matrix.png"}
    assert run.data.tags["promotion"] == "promoted"
    assert run.data.tags["registered_version"] == "1"


def test_model_version_carries_threshold_and_run_id(
    settings: Settings, mlflow_uri: str
) -> None:
    result = run_training(settings, STRONG)

    version = MlflowClient(mlflow_uri).get_model_version(
        settings.mlflow_registered_model_name, str(result.version)
    )
    assert version.tags["threshold"] == "0.6"
    assert version.tags["run_id"] == result.run_id
    assert version.tags["primary_metric"] == "average_precision"


def test_rerun_without_improvement_keeps_the_champion(
    settings: Settings, mlflow_uri: str
) -> None:
    run_training(settings, STRONG)

    second = run_training(settings, STRONG)

    assert second.version == 2
    assert not second.decision.promote
    assert second.decision.champion_score == second.decision.candidate_score
    model = MlflowClient(mlflow_uri).get_registered_model(
        settings.mlflow_registered_model_name
    )
    assert int(model.aliases[settings.mlflow_champion_alias]) == 1


def test_better_model_replaces_the_champion(
    settings: Settings, mlflow_uri: str
) -> None:
    run_training(settings, WEAK)

    second = run_training(settings, STRONG)

    assert second.decision.promote
    assert second.decision.candidate_score > second.decision.champion_score
    model = MlflowClient(mlflow_uri).get_registered_model(
        settings.mlflow_registered_model_name
    )
    assert int(model.aliases[settings.mlflow_champion_alias]) == 2


def test_worse_model_does_not_replace_the_champion(
    settings: Settings, mlflow_uri: str
) -> None:
    run_training(settings, STRONG)

    second = run_training(settings, WEAK)

    assert not second.decision.promote
    assert second.decision.candidate_score < second.decision.champion_score
    run = MlflowClient(mlflow_uri).get_run(second.run_id)
    assert run.data.tags["promotion"] == "kept_champion"
    assert "champion_test_average_precision" in run.data.metrics


def test_primary_metric_must_not_depend_on_a_threshold(settings: Settings) -> None:
    params = make_params(STRONG, primary_metric="f1")

    with pytest.raises(ValueError, match="average_precision"):
        build_trainer(settings, params)


def test_pipeline_writes_the_metrics_summary(
    settings: Settings, tmp_path: Path
) -> None:
    metrics_path = tmp_path / "metrics.json"
    trainer = build_trainer(settings, make_params(STRONG))

    result = TrainPipeline(trainer, metrics_path).run()

    summary = json.loads(metrics_path.read_text())
    assert set(summary) == {
        *SUMMARY_METRICS,
        "promoted",
        "registered_version",
        "mlflow_run_id",
    }
    assert summary["promoted"] is True
    assert summary["mlflow_run_id"] == result.run_id
    assert replace(result).version == summary["registered_version"]
