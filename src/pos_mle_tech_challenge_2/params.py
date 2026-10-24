"""Parâmetros de experimento lidos do params.yaml."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from pos_mle_tech_challenge_2.config import PROJECT_ROOT

DEFAULT_PARAMS_PATH = PROJECT_ROOT / "params.yaml"


@dataclass(frozen=True)
class SplitParams:
    """Parâmetros da divisão treino/teste."""

    test_size: float
    drop_duplicates: bool


@dataclass(frozen=True)
class TrainParams:
    """Parâmetros do protocolo de treino e avaliação."""

    cv_folds: int
    primary_metric: str
    threshold: float


@dataclass(frozen=True)
class Params:
    """Conjunto completo de parâmetros do experimento."""

    seed: int
    split: SplitParams
    train: TrainParams
    model: dict[str, Any]


def load_params(path: Path = DEFAULT_PARAMS_PATH) -> Params:
    """Carrega e valida os parâmetros do arquivo YAML.

    Args:
        path: caminho do params.yaml.

    Returns:
        Parâmetros tipados do experimento.

    Raises:
        KeyError: se alguma seção obrigatória estiver ausente.
    """
    raw = yaml.safe_load(path.read_text())
    return Params(
        seed=raw["seed"],
        split=SplitParams(**raw["split"]),
        train=TrainParams(**raw["train"]),
        model=raw["model"],
    )
