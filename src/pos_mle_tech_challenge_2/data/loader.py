"""Leitura do dataset a partir de CSV."""

import logging
from pathlib import Path

import pandas as pd

from pos_mle_tech_challenge_2.data.schema import (
    ALL_COLUMNS,
    CATEGORICAL_FEATURES,
    TARGET,
)

logger = logging.getLogger(__name__)


class DataLoader:
    """Lê um CSV do dataset e normaliza os tipos das colunas."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> pd.DataFrame:
        """Lê o CSV, valida as colunas e converte os tipos.

        As features categóricas viram texto e o alvo vira inteiro (0/1), de forma
        idêntica para o dado bruto e para o dado processado (que perde os tipos
        ao ser salvo em CSV).

        Returns:
            DataFrame com todas as colunas do esquema, na ordem original.

        Raises:
            ValueError: se alguma coluna esperada estiver ausente.
        """
        df = pd.read_csv(self._path)
        self._validate_columns(df)
        df[list(CATEGORICAL_FEATURES)] = df[list(CATEGORICAL_FEATURES)].astype(str)
        df[TARGET] = df[TARGET].astype(int)
        logger.info("Dados carregados de %s: %s linhas", self._path, len(df))
        return df

    @staticmethod
    def _validate_columns(df: pd.DataFrame) -> None:
        missing = [column for column in ALL_COLUMNS if column not in df.columns]
        if missing:
            raise ValueError(f"Colunas ausentes no dataset: {missing}")
