"""Simula a chegada de dado novo: libera uma versão do dataset em data/raw/.

Utilitário de demonstração, fora do código do trainer e da inferência.

Versões:
    v1: sessões de Feb a Oct (sem Nov e Dec).
    v2: dataset completo (acrescenta Nov e Dec).

Uso:
    uv run python scripts/release_data.py v1
    uv run python scripts/release_data.py v2
"""

import argparse
import csv
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "data/source/online_shoppers_intention.csv"
RAW_PATH = ROOT / "data/raw/online_shoppers_intention.csv"

VERSIONS = ("v1", "v2")
V1_EXCLUDED_MONTHS = frozenset({"Nov", "Dec"})
MONTH_COLUMN = "Month"
LINE_TERMINATOR = "\r\n"  # mesmo formato do arquivo original


def write_v1(source: Path, destination: Path) -> int:
    """Grava a v1 (sem Nov e Dec) preservando o texto original das linhas.

    Returns:
        Número de sessões gravadas (sem contar o cabeçalho).
    """
    with source.open(newline="") as src, destination.open("w", newline="") as dst:
        reader = csv.reader(src)
        writer = csv.writer(dst, lineterminator=LINE_TERMINATOR)
        header = next(reader)
        writer.writerow(header)
        month_index = header.index(MONTH_COLUMN)
        rows = [row for row in reader if row[month_index] not in V1_EXCLUDED_MONTHS]
        writer.writerows(rows)
    return len(rows)


def release(
    version: str, source: Path = SOURCE_PATH, destination: Path = RAW_PATH
) -> Path:
    """Libera a versão pedida do dataset em `destination`.

    Args:
        version: `v1` (Feb a Oct) ou `v2` (completo).
        source: dataset completo.
        destination: arquivo de entrada do pipeline.

    Returns:
        Caminho do arquivo liberado.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if version == "v1":
        write_v1(source, destination)
    else:
        shutil.copyfile(source, destination)
    return destination


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Lê os argumentos de linha de comando."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("version", choices=VERSIONS, help="versão do dado a liberar")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Ponto de entrada do script."""
    version = parse_args(argv).version
    destination = release(version)
    print(f"Dado {version} liberado em {destination.relative_to(ROOT)}")
    print("Próximo passo: dvc add data/raw/online_shoppers_intention.csv")


if __name__ == "__main__":
    main()
