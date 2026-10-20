"""Fixtures compartilhadas dos testes."""

import pytest

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
