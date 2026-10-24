"""Limpeza básica do dataset."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class DataCleaner:
    """Aplica a limpeza definida na EDA (remoção de linhas duplicadas)."""

    def __init__(self, drop_duplicates: bool = True) -> None:
        self._drop_duplicates = drop_duplicates

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicatas (se configurado) e reinicia o índice.

        A remoção deve acontecer antes do split para que a mesma sessão não caia
        em treino e teste ao mesmo tempo.

        Args:
            df: dados carregados.

        Returns:
            Novo DataFrame limpo, sem alterar o original.
        """
        if not self._drop_duplicates:
            return df.reset_index(drop=True)
        cleaned = df.drop_duplicates().reset_index(drop=True)
        logger.info("Duplicatas removidas: %s", len(df) - len(cleaned))
        return cleaned
