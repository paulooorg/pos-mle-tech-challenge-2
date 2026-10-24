"""Testes da criação de estimadores e pipelines."""

from collections.abc import Callable

import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from pos_mle_tech_challenge_2.training.model_factory import ModelFactory

XY = tuple[pd.DataFrame, pd.Series]

CASES = [
    ("hist_gradient_boosting", HistGradientBoostingClassifier, {"max_depth": 3}),
    ("random_forest", RandomForestClassifier, {"n_estimators": 10}),
    ("logistic_regression", LogisticRegression, {"C": 0.5}),
]


@pytest.mark.parametrize(("algorithm", "expected_type", "params"), CASES)
def test_creates_the_requested_estimator(
    algorithm: str, expected_type: type, params: dict
) -> None:
    estimator = ModelFactory(seed=7).create_estimator(algorithm, params)

    assert isinstance(estimator, expected_type)
    assert estimator.random_state == 7
    for name, value in params.items():
        assert getattr(estimator, name) == value


def test_unknown_algorithm_lists_the_valid_options() -> None:
    with pytest.raises(ValueError, match="hist_gradient_boosting"):
        ModelFactory(seed=1).create_estimator("xgboost", {})


def test_pipeline_has_preprocessor_then_model() -> None:
    pipeline = ModelFactory(seed=1).build_pipeline("random_forest", {})

    assert [name for name, _ in pipeline.steps] == ["preprocessor", "model"]


@pytest.mark.parametrize(("algorithm", "_", "params"), CASES)
def test_pipeline_fits_and_predicts_probabilities(
    make_xy: Callable[..., XY], algorithm: str, _: type, params: dict
) -> None:
    features, target = make_xy()
    pipeline = ModelFactory(seed=1).build_pipeline(algorithm, params)

    pipeline.fit(features, target)
    proba = pipeline.predict_proba(features)

    assert proba.shape == (len(features), 2)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_same_seed_gives_same_predictions(make_xy: Callable[..., XY]) -> None:
    features, target = make_xy()
    params = {"n_estimators": 20}

    first = ModelFactory(seed=3).build_pipeline("random_forest", params)
    second = ModelFactory(seed=3).build_pipeline("random_forest", params)

    first.fit(features, target)
    second.fit(features, target)
    pd.testing.assert_frame_equal(
        pd.DataFrame(first.predict_proba(features)),
        pd.DataFrame(second.predict_proba(features)),
    )
