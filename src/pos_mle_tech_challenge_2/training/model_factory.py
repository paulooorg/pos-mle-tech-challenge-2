"""Criação do estimador e do pipeline completo a partir do params.yaml."""

from collections.abc import Mapping
from typing import Any

from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from pos_mle_tech_challenge_2.features.preprocessor import PreprocessorFactory

ESTIMATORS: dict[str, type[BaseEstimator]] = {
    "hist_gradient_boosting": HistGradientBoostingClassifier,
    "random_forest": RandomForestClassifier,
    "logistic_regression": LogisticRegression,
}


class ModelFactory:
    """Instancia o algoritmo escolhido em `model.type` com seus hiperparâmetros."""

    def __init__(
        self, seed: int, preprocessor_factory: PreprocessorFactory | None = None
    ) -> None:
        self._seed = seed
        self._preprocessor_factory = preprocessor_factory or PreprocessorFactory()

    def create_estimator(
        self, algorithm: str, hyperparameters: Mapping[str, Any]
    ) -> BaseEstimator:
        """Cria o estimador com a seed do projeto.

        Args:
            algorithm: chave do algoritmo (`hist_gradient_boosting`,
                `random_forest` ou `logistic_regression`).
            hyperparameters: hiperparâmetros do estimador.

        Raises:
            ValueError: se o algoritmo não for suportado.
        """
        if algorithm not in ESTIMATORS:
            options = ", ".join(sorted(ESTIMATORS))
            raise ValueError(
                f"Algoritmo desconhecido: {algorithm!r}. Opções: {options}"
            )
        return ESTIMATORS[algorithm](random_state=self._seed, **hyperparameters)

    def build_pipeline(
        self, algorithm: str, hyperparameters: Mapping[str, Any]
    ) -> Pipeline:
        """Cria o pipeline completo: pré-processamento seguido do estimador."""
        return Pipeline(
            [
                ("preprocessor", self._preprocessor_factory.build()),
                ("model", self.create_estimator(algorithm, hyperparameters)),
            ]
        )
