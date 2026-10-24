"""Testes das métricas de avaliação."""

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from pos_mle_tech_challenge_2.training.evaluator import CV_SCORING, Evaluator
from pos_mle_tech_challenge_2.training.model_factory import ModelFactory

XY = tuple[pd.DataFrame, pd.Series]


class FixedProbabilityModel:
    """Modelo falso que devolve probabilidades conhecidas."""

    def __init__(self, positive_probabilities: list[float]) -> None:
        self._positive = np.array(positive_probabilities)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return np.column_stack([1 - self._positive, self._positive])


@pytest.fixture
def known_case() -> tuple[FixedProbabilityModel, pd.DataFrame, pd.Series]:
    """Exemplo clássico da documentação do scikit-learn (AP = 0,8333; AUC = 0,75)."""
    model = FixedProbabilityModel([0.1, 0.4, 0.35, 0.8])
    return model, pd.DataFrame({"x": [0, 0, 0, 0]}), pd.Series([0, 0, 1, 1])


def test_threshold_free_metrics_match_known_values(known_case: tuple) -> None:
    model, features, target = known_case

    metrics = Evaluator(cv_folds=3, seed=1).evaluate(model, features, target, [0.5])

    assert metrics["test_average_precision"] == pytest.approx(0.8333, abs=1e-4)
    assert metrics["test_roc_auc"] == pytest.approx(0.75)


def test_class_metrics_depend_on_the_threshold(known_case: tuple) -> None:
    model, features, target = known_case

    metrics = Evaluator(cv_folds=3, seed=1).evaluate(
        model, features, target, [0.5, 0.3]
    )

    assert metrics["test_precision_at_0.5"] == pytest.approx(1.0)
    assert metrics["test_recall_at_0.5"] == pytest.approx(0.5)
    assert metrics["test_f1_at_0.5"] == pytest.approx(2 / 3)
    assert metrics["test_recall_at_0.3"] == pytest.approx(1.0)
    assert metrics["test_precision_at_0.3"] == pytest.approx(2 / 3)


def test_repeated_thresholds_are_not_duplicated(known_case: tuple) -> None:
    model, features, target = known_case

    metrics = Evaluator(cv_folds=3, seed=1).evaluate(
        model, features, target, [0.5, 0.5]
    )

    assert sum(name.startswith("test_f1_") for name in metrics) == 1


def test_cross_validation_reports_mean_and_std_for_each_metric(
    make_xy: Callable[..., XY],
) -> None:
    features, target = make_xy()
    pipeline = ModelFactory(seed=1).build_pipeline("logistic_regression", {})

    metrics = Evaluator(cv_folds=3, seed=1).cross_validate(pipeline, features, target)

    expected = {f"cv_{name}_{stat}" for name in CV_SCORING for stat in ("mean", "std")}
    assert set(metrics) == expected
    assert all(0 <= metrics[f"cv_{name}_mean"] <= 1 for name in CV_SCORING)


def test_cross_validation_is_reproducible(make_xy: Callable[..., XY]) -> None:
    features, target = make_xy()
    pipeline = ModelFactory(seed=1).build_pipeline("logistic_regression", {})

    first = Evaluator(cv_folds=3, seed=5).cross_validate(pipeline, features, target)
    second = Evaluator(cv_folds=3, seed=5).cross_validate(pipeline, features, target)

    assert first == second
