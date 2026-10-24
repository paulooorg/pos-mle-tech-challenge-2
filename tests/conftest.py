"""Fixtures compartilhadas dos testes."""

from collections.abc import Callable, Iterator
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest

from pos_mle_tech_challenge_2.data.schema import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
)

SETTINGS_ENV_VARS = (
    "MLFLOW_TRACKING_URI",
    "MLFLOW_EXPERIMENT_NAME",
    "MLFLOW_REGISTERED_MODEL_NAME",
    "MLFLOW_CHAMPION_ALIAS",
    "DATA_RAW_PATH",
    "DATA_PROCESSED_DIR",
    "LOG_LEVEL",
)


@pytest.fixture(autouse=True)
def clean_settings_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isola cada teste das variáveis de ambiente de configuração.

    O `setenv` seguido de `delenv` registra o estado original no monkeypatch,
    então variáveis definidas por `load_dotenv` durante o teste são removidas
    ao final.
    """
    for name in SETTINGS_ENV_VARS:
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)


@pytest.fixture
def make_sessions() -> Callable[..., pd.DataFrame]:
    """Fábrica de datasets sintéticos com o mesmo esquema do dataset real."""

    def _make(n_rows: int = 200, positive_rate: float = 0.2) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        data: dict[str, object] = {
            column: rng.random(n_rows).round(4) for column in NUMERIC_FEATURES
        }
        data["Month"] = rng.choice(["Feb", "Mar", "Nov", "Dec"], n_rows)
        data["VisitorType"] = rng.choice(["Returning_Visitor", "New_Visitor"], n_rows)
        data["Weekend"] = rng.choice([True, False], n_rows)
        for column in ("OperatingSystems", "Browser", "Region", "TrafficType"):
            data[column] = rng.integers(1, 5, n_rows)
        n_positive = int(n_rows * positive_rate)
        data[TARGET] = [True] * n_positive + [False] * (n_rows - n_positive)
        return pd.DataFrame(data)

    return _make


@pytest.fixture
def make_xy(
    make_sessions: Callable[..., pd.DataFrame],
) -> Callable[..., tuple[pd.DataFrame, pd.Series]]:
    """Fábrica de (features, alvo) sintéticos já com os tipos do DataLoader."""

    def _make(
        n_rows: int = 200, positive_rate: float = 0.2
    ) -> tuple[pd.DataFrame, pd.Series]:
        df = make_sessions(n_rows, positive_rate)
        df[list(CATEGORICAL_FEATURES)] = df[list(CATEGORICAL_FEATURES)].astype(str)
        df[TARGET] = df[TARGET].astype(int)
        return df.drop(columns=[TARGET]), df[TARGET]

    return _make


@pytest.fixture
def mlflow_uri(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """MLflow isolado: banco SQLite e artefatos dentro do diretório temporário."""
    monkeypatch.chdir(tmp_path)
    previous_uri = mlflow.get_tracking_uri()
    yield f"sqlite:///{tmp_path}/mlflow.db"
    mlflow.set_tracking_uri(previous_uri)
    mlflow.set_registry_uri(previous_uri)
