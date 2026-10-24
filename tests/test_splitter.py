"""Testes da divisão estratificada em treino e teste."""

from collections.abc import Callable

import pandas as pd
import pytest

from pos_mle_tech_challenge_2.data.schema import NUMERIC_FEATURES, TARGET
from pos_mle_tech_challenge_2.data.splitter import DataSplitter


def test_split_sizes_follow_test_size(
    make_sessions: Callable[..., pd.DataFrame],
) -> None:
    df = make_sessions(n_rows=200)

    train, test = DataSplitter(test_size=0.2, seed=42).split(df)

    assert len(test) == 40
    assert len(train) == 160


def test_split_preserves_target_proportion(
    make_sessions: Callable[..., pd.DataFrame],
) -> None:
    df = make_sessions(n_rows=200, positive_rate=0.15)

    train, test = DataSplitter(test_size=0.2, seed=42).split(df)

    assert train[TARGET].mean() == pytest.approx(0.15, abs=0.01)
    assert test[TARGET].mean() == pytest.approx(0.15, abs=0.03)


def test_same_seed_gives_same_split(
    make_sessions: Callable[..., pd.DataFrame],
) -> None:
    df = make_sessions()

    first_train, first_test = DataSplitter(0.2, seed=7).split(df)
    second_train, second_test = DataSplitter(0.2, seed=7).split(df)

    pd.testing.assert_frame_equal(first_train, second_train)
    pd.testing.assert_frame_equal(first_test, second_test)


def test_train_and_test_do_not_overlap(
    make_sessions: Callable[..., pd.DataFrame],
) -> None:
    df = make_sessions()

    train, test = DataSplitter(0.2, seed=42).split(df)

    key = list(NUMERIC_FEATURES)
    assert train.merge(test, on=key).empty
