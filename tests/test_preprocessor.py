"""Testes do pré-processamento das features."""

from collections.abc import Callable

import numpy as np
import pandas as pd

from pos_mle_tech_challenge_2.data.schema import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)
from pos_mle_tech_challenge_2.features.preprocessor import PreprocessorFactory

XY = tuple[pd.DataFrame, pd.Series]


def test_numeric_features_are_standardized(
    make_xy: Callable[..., XY],
) -> None:
    features, _ = make_xy()
    preprocessor = PreprocessorFactory().build()

    transformed = preprocessor.fit_transform(features)

    numeric = transformed[:, : len(NUMERIC_FEATURES)]
    assert np.allclose(numeric.mean(axis=0), 0, atol=1e-8)
    assert np.allclose(numeric.std(axis=0), 1, atol=1e-8)


def test_categorical_features_are_one_hot_encoded(
    make_xy: Callable[..., XY],
) -> None:
    features, _ = make_xy()
    preprocessor = PreprocessorFactory().build()

    transformed = preprocessor.fit_transform(features)

    n_categories = sum(features[column].nunique() for column in CATEGORICAL_FEATURES)
    assert transformed.shape == (len(features), len(NUMERIC_FEATURES) + n_categories)
    assert not np.isnan(transformed).any()


def test_unseen_category_is_ignored(make_xy: Callable[..., XY]) -> None:
    features, _ = make_xy()
    preprocessor = PreprocessorFactory().build().fit(features)
    new_data = features.head(3).copy()
    new_data["Month"] = "Jan"  # mês que não existe no treino

    transformed = preprocessor.transform(new_data)

    assert transformed.shape[0] == 3
    assert not np.isnan(transformed).any()


def test_each_build_returns_a_fresh_transformer() -> None:
    factory = PreprocessorFactory()

    assert factory.build() is not factory.build()
