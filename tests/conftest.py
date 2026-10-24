"""Fixtures compartilhadas dos testes."""

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from pos_mle_tech_challenge_2.data.schema import NUMERIC_FEATURES, TARGET

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
