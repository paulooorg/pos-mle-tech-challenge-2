"""Estágio `train` do DVC.

Esqueleto da Etapa 3: lê os dados processados e grava um `metrics.json` mínimo,
para o pipeline rodar de ponta a ponta. O treino real (modelo, MLflow e Registry)
entra na Etapa 4.
"""

import json
import logging

from pos_mle_tech_challenge_2.config import (
    PROJECT_ROOT,
    TEST_FILENAME,
    TRAIN_FILENAME,
    Settings,
    configure_logging,
    load_settings,
)
from pos_mle_tech_challenge_2.data.loader import DataLoader
from pos_mle_tech_challenge_2.data.schema import TARGET

logger = logging.getLogger(__name__)

METRICS_PATH = PROJECT_ROOT / "metrics.json"


class TrainPipeline:
    """Esqueleto do estágio de treino (substituído na Etapa 4)."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def run(self) -> dict[str, float | int]:
        """Lê treino e teste e grava métricas descritivas em `metrics.json`.

        Returns:
            Métricas gravadas (tamanhos e taxa de compra de cada conjunto).
        """
        processed_dir = self._settings.data_processed_dir
        train = DataLoader(processed_dir / TRAIN_FILENAME).load()
        test = DataLoader(processed_dir / TEST_FILENAME).load()
        metrics = {
            "n_train": len(train),
            "n_test": len(test),
            "positive_rate_train": round(float(train[TARGET].mean()), 4),
            "positive_rate_test": round(float(test[TARGET].mean()), 4),
        }
        METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n")
        logger.info("Métricas gravadas em %s: %s", METRICS_PATH, metrics)
        return metrics


def main() -> None:
    """Ponto de entrada do estágio (`python -m ...pipelines.train`)."""
    settings = load_settings()
    configure_logging(settings.log_level)
    TrainPipeline(settings).run()


if __name__ == "__main__":
    main()
