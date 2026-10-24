"""Testes das configurações carregadas do ambiente."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from pos_mle_tech_challenge_2.config import DEFAULTS, PROJECT_ROOT, load_settings


@pytest.fixture
def missing_env_file(tmp_path: Path) -> Path:
    """Caminho de um .env que não existe."""
    return tmp_path / "missing.env"


def test_defaults_when_environment_is_empty(missing_env_file: Path) -> None:
    settings = load_settings(missing_env_file)

    assert settings.mlflow_tracking_uri == DEFAULTS["MLFLOW_TRACKING_URI"]
    assert settings.mlflow_champion_alias == DEFAULTS["MLFLOW_CHAMPION_ALIAS"]
    assert settings.log_level == "INFO"


def test_relative_paths_are_resolved_from_project_root(
    missing_env_file: Path,
) -> None:
    settings = load_settings(missing_env_file)

    assert settings.data_raw_path.is_absolute()
    assert settings.data_raw_path.is_relative_to(PROJECT_ROOT)
    assert settings.data_processed_dir == PROJECT_ROOT / "data/processed"


def test_environment_variables_override_defaults(
    monkeypatch: pytest.MonkeyPatch, missing_env_file: Path
) -> None:
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "custom-experiment")
    monkeypatch.setenv("LOG_LEVEL", "debug")

    settings = load_settings(missing_env_file)

    assert settings.mlflow_experiment_name == "custom-experiment"
    assert settings.log_level == "DEBUG"


def test_values_are_loaded_from_env_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("MLFLOW_CHAMPION_ALIAS=production\n")

    settings = load_settings(env_file)

    assert settings.mlflow_champion_alias == "production"


def test_environment_has_priority_over_env_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("MLFLOW_CHAMPION_ALIAS=from-file\n")
    monkeypatch.setenv("MLFLOW_CHAMPION_ALIAS", "from-env")

    settings = load_settings(env_file)

    assert settings.mlflow_champion_alias == "from-env"


def test_settings_are_immutable(missing_env_file: Path) -> None:
    settings = load_settings(missing_env_file)

    with pytest.raises(FrozenInstanceError):
        settings.log_level = "DEBUG"  # type: ignore[misc]


def test_model_version_defaults_to_the_champion(missing_env_file: Path) -> None:
    assert load_settings(missing_env_file).model_version == "champion"


def test_model_version_can_pin_a_number(
    monkeypatch: pytest.MonkeyPatch, missing_env_file: Path
) -> None:
    monkeypatch.setenv("MODEL_VERSION", "3")

    assert load_settings(missing_env_file).model_version == "3"
