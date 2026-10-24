"""Predição em lote com o modelo carregado."""

from collections.abc import Sequence

import pandas as pd

from pos_mle_tech_challenge_2.data.schema import CATEGORICAL_FEATURES, FEATURE_COLUMNS
from pos_mle_tech_challenge_2.serving.model_loader import LoadedModel
from pos_mle_tech_challenge_2.serving.schemas import (
    ModelInfo,
    PredictionItem,
    PredictResponse,
    SessionFeatures,
)

PROBABILITY_DECIMALS = 4


class Predictor:
    """Converte sessões em features, aplica o modelo e o threshold da versão."""

    def __init__(self, loaded: LoadedModel) -> None:
        self._pipeline = loaded.pipeline
        self._info = loaded.info

    @property
    def info(self) -> ModelInfo:
        """Informações do modelo carregado."""
        return self._info

    def predict(self, sessions: Sequence[SessionFeatures]) -> PredictResponse:
        """Classifica um lote de sessões, preservando a ordem da entrada.

        Returns:
            Probabilidade de compra e classe de cada sessão, com a versão do modelo
            e o threshold usados.
        """
        probabilities = self._pipeline.predict_proba(self._to_dataframe(sessions))[:, 1]
        threshold = self._info.threshold
        predictions = [
            PredictionItem(
                probability=round(float(probability), PROBABILITY_DECIMALS),
                predicted_class=int(probability >= threshold),
            )
            for probability in probabilities
        ]
        return PredictResponse(
            model_version=self._info.version,
            threshold=threshold,
            predictions=predictions,
        )

    @staticmethod
    def _to_dataframe(sessions: Sequence[SessionFeatures]) -> pd.DataFrame:
        """Monta as features com os mesmos tipos do treino (categorias como texto)."""
        frame = pd.DataFrame(
            [session.model_dump() for session in sessions],
            columns=list(FEATURE_COLUMNS),
        )
        frame[list(CATEGORICAL_FEATURES)] = frame[list(CATEGORICAL_FEATURES)].astype(
            str
        )
        return frame
