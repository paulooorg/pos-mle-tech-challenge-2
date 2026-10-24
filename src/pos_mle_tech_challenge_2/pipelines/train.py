"""Estágio `train` do DVC: treina, registra no MLflow e decide a promoção."""

import json
import logging
from pathlib import Path

from pos_mle_tech_challenge_2.config import (
    PROJECT_ROOT,
    Settings,
    configure_logging,
    load_settings,
)
from pos_mle_tech_challenge_2.params import Params, load_params
from pos_mle_tech_challenge_2.training.evaluator import Evaluator
from pos_mle_tech_challenge_2.training.lineage import RunLineage
from pos_mle_tech_challenge_2.training.model_factory import ModelFactory
from pos_mle_tech_challenge_2.training.promotion import PromotionGate
from pos_mle_tech_challenge_2.training.registry import ModelRegistry
from pos_mle_tech_challenge_2.training.reporting import ReportBuilder
from pos_mle_tech_challenge_2.training.tracker import ExperimentTracker
from pos_mle_tech_challenge_2.training.trainer import (
    Trainer,
    TrainerComponents,
    TrainingResult,
)

logger = logging.getLogger(__name__)

METRICS_PATH = PROJECT_ROOT / "metrics.json"
SUMMARY_METRICS = (
    "cv_average_precision_mean",
    "test_average_precision",
    "test_roc_auc",
)


def build_trainer(settings: Settings, params: Params) -> Trainer:
    """Monta o `Trainer` com todas as dependências (raiz de composição)."""
    components = TrainerComponents(
        model_factory=ModelFactory(params.seed),
        evaluator=Evaluator(params.train.cv_folds, params.seed),
        tracker=ExperimentTracker(
            settings.mlflow_tracking_uri, settings.mlflow_experiment_name
        ),
        registry=ModelRegistry(
            settings.mlflow_tracking_uri,
            settings.mlflow_registered_model_name,
            settings.mlflow_champion_alias,
        ),
        gate=PromotionGate(),
        reporter=ReportBuilder(),
        lineage=RunLineage(PROJECT_ROOT, Path(f"{settings.data_raw_path}.dvc")),
    )
    return Trainer(settings, params, components)


def summarize(result: TrainingResult) -> dict[str, float | int | str | bool]:
    """Resumo do run para o `metrics.json` (lido por `dvc metrics`)."""
    summary: dict[str, float | int | str | bool] = {
        name: round(result.metrics[name], 4) for name in SUMMARY_METRICS
    }
    summary["promoted"] = result.decision.promote
    summary["registered_version"] = result.version
    summary["mlflow_run_id"] = result.run_id
    return summary


class TrainPipeline:
    """Executa o treino e grava o resumo em `metrics.json`."""

    def __init__(self, trainer: Trainer, metrics_path: Path = METRICS_PATH) -> None:
        self._trainer = trainer
        self._metrics_path = metrics_path

    def run(self) -> TrainingResult:
        """Treina, registra o modelo e grava o resumo das métricas."""
        result = self._trainer.run()
        self._metrics_path.write_text(json.dumps(summarize(result), indent=2) + "\n")
        logger.info("Resumo gravado em %s", self._metrics_path)
        return result


def main() -> None:
    """Ponto de entrada do estágio (`python -m ...pipelines.train`)."""
    settings = load_settings()
    configure_logging(settings.log_level)
    TrainPipeline(build_trainer(settings, load_params())).run()


if __name__ == "__main__":
    main()
