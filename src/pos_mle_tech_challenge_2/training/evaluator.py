"""Avaliação do modelo: validação cruzada e métricas no conjunto de teste."""

from collections.abc import Iterable
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

Metrics = dict[str, float]

CV_SCORING = ("average_precision", "roc_auc", "f1", "precision", "recall")

# Métricas que não dependem de threshold: podem ser usadas para escolher o modelo.
THRESHOLD_FREE_METRICS = ("average_precision", "roc_auc")


class ProbabilisticModel(Protocol):
    """Qualquer modelo que devolve probabilidades por classe."""

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Probabilidades com formato (n_amostras, 2)."""


class Evaluator:
    """Calcula as métricas de seleção (validação cruzada) e de relatório (teste)."""

    def __init__(self, cv_folds: int, seed: int) -> None:
        self._cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    def cross_validate(
        self, pipeline: Pipeline, features: pd.DataFrame, target: pd.Series
    ) -> Metrics:
        """Validação cruzada estratificada no treino.

        Returns:
            Média e desvio de cada métrica, com chaves `cv_<métrica>_mean` e
            `cv_<métrica>_std`.
        """
        scores = cross_validate(
            pipeline, features, target, cv=self._cv, scoring=list(CV_SCORING)
        )
        metrics: Metrics = {}
        for name in CV_SCORING:
            values = scores[f"test_{name}"]
            metrics[f"cv_{name}_mean"] = float(values.mean())
            metrics[f"cv_{name}_std"] = float(values.std())
        return metrics

    def evaluate(
        self,
        model: ProbabilisticModel,
        features: pd.DataFrame,
        target: pd.Series,
        thresholds: Iterable[float],
    ) -> Metrics:
        """Métricas no teste, com F1, precision e recall para cada threshold.

        PR-AUC (`average_precision`) e ROC-AUC não dependem de threshold.

        Returns:
            Métricas `test_*`; as de classe levam o sufixo `_at_<threshold>`.
        """
        proba = model.predict_proba(features)[:, 1]
        metrics: Metrics = {
            "test_average_precision": float(average_precision_score(target, proba)),
            "test_roc_auc": float(roc_auc_score(target, proba)),
        }
        for threshold in sorted(set(thresholds)):
            metrics.update(self._threshold_metrics(target, proba, threshold))
        return metrics

    @staticmethod
    def _threshold_metrics(
        target: pd.Series, proba: np.ndarray, threshold: float
    ) -> Metrics:
        predicted = (proba >= threshold).astype(int)
        suffix = f"at_{threshold:g}"
        return {
            f"test_f1_{suffix}": float(f1_score(target, predicted, zero_division=0)),
            f"test_precision_{suffix}": float(
                precision_score(target, predicted, zero_division=0)
            ),
            f"test_recall_{suffix}": float(
                recall_score(target, predicted, zero_division=0)
            ),
        }
