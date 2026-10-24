"""Criação do estimador e do pipeline completo a partir do params.yaml."""

from collections.abc import Mapping
from typing import Any

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

from pos_mle_tech_challenge_2.features.preprocessor import PreprocessorFactory

ALGORITHM_NAME = "hist_gradient_boosting"


class ModelFactory:
    """Cria o HistGradientBoostingClassifier com os hiperparâmetros de `model`."""

    def __init__(
        self, seed: int, preprocessor_factory: PreprocessorFactory | None = None
    ) -> None:
        self._seed = seed
        self._preprocessor_factory = preprocessor_factory or PreprocessorFactory()

    def create_estimator(
        self, hyperparameters: Mapping[str, Any]
    ) -> HistGradientBoostingClassifier:
        """Cria o estimador com a seed do projeto.

        Args:
            hyperparameters: hiperparâmetros do `HistGradientBoostingClassifier`.
        """
        return HistGradientBoostingClassifier(
            random_state=self._seed, **hyperparameters
        )

    def build_pipeline(self, hyperparameters: Mapping[str, Any]) -> Pipeline:
        """Cria o pipeline completo: pré-processamento seguido do estimador."""
        return Pipeline(
            [
                ("preprocessor", self._preprocessor_factory.build()),
                ("model", self.create_estimator(hyperparameters)),
            ]
        )
