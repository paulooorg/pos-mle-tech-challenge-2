"""Testes da criação do estimador e do pipeline."""

from collections.abc import Callable

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from pos_mle_tech_challenge_2.training.model_factory import ModelFactory

XY = tuple[pd.DataFrame, pd.Series]


def test_creates_the_estimator_with_seed_and_hyperparameters() -> None:
    estimator = ModelFactory(seed=7).create_estimator(
        {"max_depth": 3, "learning_rate": 0.2}
    )

    assert isinstance(estimator, HistGradientBoostingClassifier)
    assert estimator.random_state == 7
    assert estimator.max_depth == 3
    assert estimator.learning_rate == 0.2


def test_pipeline_has_preprocessor_then_model() -> None:
    pipeline = ModelFactory(seed=1).build_pipeline({})

    assert [name for name, _ in pipeline.steps] == ["preprocessor", "model"]


def test_pipeline_fits_and_predicts_probabilities(
    make_xy: Callable[..., XY],
) -> None:
    features, target = make_xy()
    pipeline = ModelFactory(seed=1).build_pipeline({"max_iter": 20})

    pipeline.fit(features, target)
    proba = pipeline.predict_proba(features)

    assert proba.shape == (len(features), 2)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_same_seed_gives_same_predictions(make_xy: Callable[..., XY]) -> None:
    features, target = make_xy()
    params = {"max_iter": 20, "max_features": 0.5}

    first = ModelFactory(seed=3).build_pipeline(params).fit(features, target)
    second = ModelFactory(seed=3).build_pipeline(params).fit(features, target)

    pd.testing.assert_frame_equal(
        pd.DataFrame(first.predict_proba(features)),
        pd.DataFrame(second.predict_proba(features)),
    )
