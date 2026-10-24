"""Divisão estratificada em treino e teste."""

import pandas as pd
from sklearn.model_selection import train_test_split

from pos_mle_tech_challenge_2.data.schema import TARGET


class DataSplitter:
    """Divide o dataset mantendo a proporção do alvo, com seed fixa."""

    def __init__(self, test_size: float, seed: int) -> None:
        self._test_size = test_size
        self._seed = seed

    def split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Divide em treino e teste de forma estratificada pelo alvo.

        Args:
            df: dados limpos, incluindo a coluna alvo.

        Returns:
            Tupla (treino, teste), ambos com as colunas originais.
        """
        train, test = train_test_split(
            df,
            test_size=self._test_size,
            random_state=self._seed,
            stratify=df[TARGET],
        )
        return train.reset_index(drop=True), test.reset_index(drop=True)
