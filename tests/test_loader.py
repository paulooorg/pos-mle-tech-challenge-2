"""Testes do carregamento e normalização de tipos do dataset."""

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from pos_mle_tech_challenge_2.data.loader import DataLoader
from pos_mle_tech_challenge_2.data.schema import CATEGORICAL_FEATURES, TARGET


def test_categorical_features_are_loaded_as_text(
    tmp_path: Path, make_sessions: Callable[..., pd.DataFrame]
) -> None:
    csv_path = tmp_path / "sessions.csv"
    make_sessions().to_csv(csv_path, index=False)

    df = DataLoader(csv_path).load()

    for column in CATEGORICAL_FEATURES:
        assert df[column].map(type).eq(str).all(), column


def test_codes_stay_categorical_after_csv_round_trip(
    tmp_path: Path, make_sessions: Callable[..., pd.DataFrame]
) -> None:
    csv_path = tmp_path / "sessions.csv"
    make_sessions().to_csv(csv_path, index=False)
    first = DataLoader(csv_path).load()
    first.to_csv(csv_path, index=False)

    second = DataLoader(csv_path).load()

    pd.testing.assert_frame_equal(first, second)


def test_target_is_loaded_as_integer(
    tmp_path: Path, make_sessions: Callable[..., pd.DataFrame]
) -> None:
    csv_path = tmp_path / "sessions.csv"
    make_sessions().to_csv(csv_path, index=False)

    df = DataLoader(csv_path).load()

    assert set(df[TARGET].unique()) <= {0, 1}
    assert pd.api.types.is_integer_dtype(df[TARGET])


def test_missing_columns_raise_clear_error(
    tmp_path: Path, make_sessions: Callable[..., pd.DataFrame]
) -> None:
    csv_path = tmp_path / "sessions.csv"
    make_sessions().drop(columns=["PageValues"]).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="PageValues"):
        DataLoader(csv_path).load()
