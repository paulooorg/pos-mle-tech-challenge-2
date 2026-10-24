"""Teste do estágio preprocess do pipeline."""

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from pos_mle_tech_challenge_2.config import (
    TEST_FILENAME,
    TRAIN_FILENAME,
    Settings,
    load_settings,
)
from pos_mle_tech_challenge_2.params import load_params
from pos_mle_tech_challenge_2.pipelines.preprocess import PreprocessPipeline


@pytest.fixture
def settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_sessions: Callable[..., pd.DataFrame],
) -> Settings:
    """Configuração apontando para dados sintéticos em um diretório temporário."""
    raw_path = tmp_path / "raw.csv"
    duplicated = make_sessions(n_rows=200)
    pd.concat([duplicated, duplicated.head(10)]).to_csv(raw_path, index=False)
    monkeypatch.setenv("DATA_RAW_PATH", str(raw_path))
    monkeypatch.setenv("DATA_PROCESSED_DIR", str(tmp_path / "processed"))
    return load_settings(tmp_path / "missing.env")


def test_preprocess_writes_train_and_test_files(settings: Settings) -> None:
    PreprocessPipeline(settings, load_params()).run()

    train = pd.read_csv(settings.data_processed_dir / TRAIN_FILENAME)
    test = pd.read_csv(settings.data_processed_dir / TEST_FILENAME)
    assert len(train) + len(test) == 200  # 10 duplicatas removidas
    assert len(test) == 40
