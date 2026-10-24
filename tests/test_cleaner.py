"""Testes da limpeza do dataset."""

from collections.abc import Callable

import pandas as pd

from pos_mle_tech_challenge_2.data.cleaner import DataCleaner


def _with_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([df, df.head(5)], ignore_index=True)


def test_duplicates_are_removed(make_sessions: Callable[..., pd.DataFrame]) -> None:
    df = make_sessions(n_rows=50)

    cleaned = DataCleaner(drop_duplicates=True).clean(_with_duplicates(df))

    assert len(cleaned) == len(df)
    assert not cleaned.duplicated().any()


def test_duplicates_are_kept_when_disabled(
    make_sessions: Callable[..., pd.DataFrame],
) -> None:
    df = _with_duplicates(make_sessions(n_rows=50))

    cleaned = DataCleaner(drop_duplicates=False).clean(df)

    assert len(cleaned) == len(df)


def test_index_is_reset_and_input_is_not_modified(
    make_sessions: Callable[..., pd.DataFrame],
) -> None:
    df = _with_duplicates(make_sessions(n_rows=50))
    original_length = len(df)

    cleaned = DataCleaner().clean(df)

    assert list(cleaned.index) == list(range(len(cleaned)))
    assert len(df) == original_length
