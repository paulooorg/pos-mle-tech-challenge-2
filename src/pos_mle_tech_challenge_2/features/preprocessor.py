"""Pré-processamento das features para o modelo."""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pos_mle_tech_challenge_2.data.schema import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)


class PreprocessorFactory:
    """Monta o transformador de colunas aplicado antes do estimador.

    O transformador é treinado junto com o modelo (dentro de um `Pipeline`), o que
    evita vazamento de dados do conjunto de teste para o treino.
    """

    def build(self) -> ColumnTransformer:
        """Cria o `ColumnTransformer` do projeto.

        - Numéricas: padronização (média 0, desvio 1).
        - Categóricas: one-hot, ignorando categorias nunca vistas no treino (por
          exemplo, meses que só aparecem em uma versão futura do dado).

        Returns:
            Transformador ainda não treinado.
        """
        return ColumnTransformer(
            [
                ("numeric", StandardScaler(), list(NUMERIC_FEATURES)),
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    list(CATEGORICAL_FEATURES),
                ),
            ]
        )
