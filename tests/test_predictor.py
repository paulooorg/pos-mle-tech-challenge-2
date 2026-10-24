"""Testes da predição em lote (com um modelo falso)."""

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from pos_mle_tech_challenge_2.data.schema import CATEGORICAL_FEATURES, FEATURE_COLUMNS
from pos_mle_tech_challenge_2.serving.examples import EXAMPLE_SESSIONS
from pos_mle_tech_challenge_2.serving.model_loader import LoadedModel
from pos_mle_tech_challenge_2.serving.predictor import Predictor
from pos_mle_tech_challenge_2.serving.schemas import ModelInfo, SessionFeatures


class FakePipeline:
    """Probabilidade = PageValues / 10 (limitada a 1); guarda a última entrada."""

    def __init__(self) -> None:
        self.received: pd.DataFrame | None = None

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        self.received = features
        positive = np.clip(features["PageValues"].to_numpy() / 10, 0, 1)
        return np.column_stack([1 - positive, positive])


def make_predictor(threshold: float = 0.73) -> tuple[Predictor, FakePipeline]:
    pipeline = FakePipeline()
    info = ModelInfo(
        name="m",
        version=3,
        requested="champion",
        run_id="abc",
        threshold=threshold,
        data_md5=None,
        git_commit=None,
        test_average_precision=None,
        loaded_at=datetime.now(UTC),
    )
    return Predictor(LoadedModel(pipeline, info)), pipeline


def session(page_values: float) -> SessionFeatures:
    return SessionFeatures(
        **{**EXAMPLE_SESSIONS["navegacao_curta"], "PageValues": page_values}
    )


def test_class_follows_the_model_threshold() -> None:
    predictor, _ = make_predictor(threshold=0.73)

    response = predictor.predict([session(8.0), session(5.0), session(0.0)])

    assert [p.predicted_class for p in response.predictions] == [1, 0, 0]
    assert [p.probability for p in response.predictions] == [0.8, 0.5, 0.0]


def test_a_probability_equal_to_the_threshold_is_class_one() -> None:
    predictor, _ = make_predictor(threshold=0.5)

    response = predictor.predict([session(5.0)])

    assert response.predictions[0].predicted_class == 1


def test_response_reports_the_model_version_and_threshold() -> None:
    predictor, _ = make_predictor(threshold=0.6)

    response = predictor.predict([session(1.0)])

    assert (response.model_version, response.threshold) == (3, 0.6)


def test_order_of_the_batch_is_preserved() -> None:
    predictor, _ = make_predictor()

    response = predictor.predict([session(1.0), session(9.0), session(4.0)])

    assert [p.probability for p in response.predictions] == [0.1, 0.9, 0.4]


def test_features_have_training_columns_and_text_categories() -> None:
    predictor, pipeline = make_predictor()

    predictor.predict([SessionFeatures(**EXAMPLE_SESSIONS["visitante_recorrente"])])

    received = pipeline.received
    assert list(received.columns) == list(FEATURE_COLUMNS)
    for column in CATEGORICAL_FEATURES:
        assert received[column].map(type).eq(str).all(), column
    assert received.loc[0, "Weekend"] == "True"
    assert received.loc[0, "Region"] == "3"


def test_probability_is_rounded_to_four_decimals() -> None:
    predictor, _ = make_predictor()

    response = predictor.predict([session(1.23456789)])

    assert response.predictions[0].probability == pytest.approx(0.1235, abs=1e-9)
