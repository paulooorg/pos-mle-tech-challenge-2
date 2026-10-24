"""Testes do script utilitário que simula a chegada de dado novo."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "release_data.py"

SAMPLE_CSV = (
    "A,Month,Revenue\r\n"
    "1,Feb,FALSE\r\n"
    "2,Oct,TRUE\r\n"
    "3,Nov,TRUE\r\n"
    "4,Dec,FALSE\r\n"
    "5,May,FALSE\r\n"
)


@pytest.fixture(scope="module")
def release_data() -> ModuleType:
    """Carrega o script (fora do pacote) como módulo."""
    spec = importlib.util.spec_from_file_location("release_data", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def source(tmp_path: Path) -> Path:
    path = tmp_path / "source.csv"
    path.write_bytes(SAMPLE_CSV.encode())
    return path


def test_v1_excludes_november_and_december(
    release_data: ModuleType, source: Path, tmp_path: Path
) -> None:
    destination = tmp_path / "raw" / "data.csv"

    release_data.release("v1", source, destination)

    lines = destination.read_bytes().decode().splitlines()
    assert lines == ["A,Month,Revenue", "1,Feb,FALSE", "2,Oct,TRUE", "5,May,FALSE"]


def test_v1_preserves_original_text_format(
    release_data: ModuleType, source: Path, tmp_path: Path
) -> None:
    destination = tmp_path / "data.csv"

    release_data.release("v1", source, destination)

    assert destination.read_bytes().startswith(b"A,Month,Revenue\r\n1,Feb,FALSE\r\n")


def test_v2_is_an_exact_copy_of_the_source(
    release_data: ModuleType, source: Path, tmp_path: Path
) -> None:
    destination = tmp_path / "data.csv"

    release_data.release("v2", source, destination)

    assert destination.read_bytes() == source.read_bytes()


def test_unknown_version_is_rejected(release_data: ModuleType) -> None:
    with pytest.raises(SystemExit):
        release_data.parse_args(["v3"])
