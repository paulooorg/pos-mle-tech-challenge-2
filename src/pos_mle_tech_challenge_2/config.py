"""Configurações do projeto lidas de variáveis de ambiente (arquivo .env)."""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_FILENAME = "train.csv"
TEST_FILENAME = "test.csv"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

DEFAULTS: dict[str, str] = {
    "MLFLOW_TRACKING_URI": "sqlite:///mlflow.db",
    "MLFLOW_EXPERIMENT_NAME": "purchase-propensity",
    "MLFLOW_REGISTERED_MODEL_NAME": "purchase-propensity-classifier",
    "MLFLOW_CHAMPION_ALIAS": "champion",
    "DATA_RAW_PATH": "data/raw/online_shoppers_intention.csv",
    "DATA_PROCESSED_DIR": "data/processed",
    "LOG_LEVEL": "INFO",
}


@dataclass(frozen=True)
class Settings:
    """Configuração de ambiente da aplicação (imutável)."""

    mlflow_tracking_uri: str
    mlflow_experiment_name: str
    mlflow_registered_model_name: str
    mlflow_champion_alias: str
    data_raw_path: Path
    data_processed_dir: Path
    log_level: str


def _read(name: str) -> str:
    """Lê uma variável de ambiente, usando o valor padrão se ausente."""
    return os.getenv(name, DEFAULTS[name])


def _read_path(name: str) -> Path:
    """Lê uma variável de caminho, resolvendo relativos a partir da raiz do projeto."""
    return PROJECT_ROOT / _read(name)


def load_settings(env_file: Path | None = None) -> Settings:
    """Carrega as configurações do ambiente, usando o .env como complemento.

    Variáveis já definidas no ambiente têm prioridade sobre as do arquivo .env.

    Args:
        env_file: caminho do arquivo .env. Por padrão, `.env` na raiz do projeto.

    Returns:
        Configurações com valores padrão para variáveis ausentes.
    """
    load_dotenv(env_file or PROJECT_ROOT / ".env")
    return Settings(
        mlflow_tracking_uri=_read("MLFLOW_TRACKING_URI"),
        mlflow_experiment_name=_read("MLFLOW_EXPERIMENT_NAME"),
        mlflow_registered_model_name=_read("MLFLOW_REGISTERED_MODEL_NAME"),
        mlflow_champion_alias=_read("MLFLOW_CHAMPION_ALIAS"),
        data_raw_path=_read_path("DATA_RAW_PATH"),
        data_processed_dir=_read_path("DATA_PROCESSED_DIR"),
        log_level=_read("LOG_LEVEL").upper(),
    )


def configure_logging(level: str) -> None:
    """Configura o logging da aplicação com o nível informado (ex.: INFO)."""
    logging.basicConfig(level=level, format=LOG_FORMAT)
