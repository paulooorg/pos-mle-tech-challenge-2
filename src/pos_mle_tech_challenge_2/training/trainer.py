"""Orquestração do treino: avalia, registra e decide a promoção do modelo."""

import logging
from dataclasses import asdict, dataclass

import pandas as pd
from sklearn.pipeline import Pipeline

from pos_mle_tech_challenge_2.config import TEST_FILENAME, TRAIN_FILENAME, Settings
from pos_mle_tech_challenge_2.data.loader import DataLoader
from pos_mle_tech_challenge_2.data.schema import TARGET
from pos_mle_tech_challenge_2.params import Params
from pos_mle_tech_challenge_2.training.evaluator import (
    THRESHOLD_FREE_METRICS,
    Evaluator,
    Metrics,
)
from pos_mle_tech_challenge_2.training.lineage import RunLineage
from pos_mle_tech_challenge_2.training.model_factory import (
    ALGORITHM_NAME,
    ModelFactory,
)
from pos_mle_tech_challenge_2.training.promotion import (
    PromotionDecision,
    PromotionGate,
)
from pos_mle_tech_challenge_2.training.registry import ModelRegistry
from pos_mle_tech_challenge_2.training.reporting import ReportBuilder
from pos_mle_tech_challenge_2.training.tracker import ExperimentTracker

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.5


@dataclass(frozen=True)
class SplitData:
    """Features e alvo de treino e de teste."""

    x_train: pd.DataFrame
    y_train: pd.Series
    x_test: pd.DataFrame
    y_test: pd.Series


@dataclass(frozen=True)
class TrainingResult:
    """Resultado de uma execução do treino."""

    run_id: str
    version: int
    decision: PromotionDecision
    metrics: Metrics


@dataclass(frozen=True)
class TrainerComponents:
    """Dependências do `Trainer` (injetadas na montagem em `pipelines/train.py`)."""

    model_factory: ModelFactory
    evaluator: Evaluator
    tracker: ExperimentTracker
    registry: ModelRegistry
    gate: PromotionGate
    reporter: ReportBuilder
    lineage: RunLineage


class Trainer:
    """Treina o modelo definido em `params.yaml` e controla a promoção a champion."""

    def __init__(
        self,
        settings: Settings,
        params: Params,
        components: TrainerComponents,
    ) -> None:
        self._settings = settings
        self._params = params
        self._parts = components
        self._score_key = self._validated_score_key(params.train.primary_metric)

    @staticmethod
    def _validated_score_key(primary_metric: str) -> str:
        if primary_metric not in THRESHOLD_FREE_METRICS:
            options = ", ".join(THRESHOLD_FREE_METRICS)
            raise ValueError(
                f"Métrica principal inválida: {primary_metric!r}. Opções: {options}"
            )
        return f"test_{primary_metric}"

    def run(self) -> TrainingResult:
        """Executa um run completo e devolve o resultado."""
        data = self._load_data()
        pipeline = self._parts.model_factory.build_pipeline(self._params.model)
        tracker = self._parts.tracker
        with tracker.start_run(run_name=ALGORITHM_NAME) as run_id:
            tracker.log_params(asdict(self._params))
            tracker.set_tags(self._parts.lineage.collect())
            metrics = self._fit_and_evaluate(pipeline, data)
            model_uri = tracker.log_model(pipeline, data.x_train)
            version, decision = self._register_and_decide(
                run_id, model_uri, data, metrics
            )
        return TrainingResult(run_id, version, decision, metrics)

    def _load_data(self) -> SplitData:
        processed_dir = self._settings.data_processed_dir
        train = DataLoader(processed_dir / TRAIN_FILENAME).load()
        test = DataLoader(processed_dir / TEST_FILENAME).load()
        return SplitData(
            x_train=train.drop(columns=[TARGET]),
            y_train=train[TARGET],
            x_test=test.drop(columns=[TARGET]),
            y_test=test[TARGET],
        )

    def _fit_and_evaluate(self, pipeline: Pipeline, data: SplitData) -> Metrics:
        parts = self._parts
        metrics = parts.evaluator.cross_validate(pipeline, data.x_train, data.y_train)
        pipeline.fit(data.x_train, data.y_train)
        thresholds = {DEFAULT_THRESHOLD, self._params.train.threshold}
        metrics |= parts.evaluator.evaluate(
            pipeline, data.x_test, data.y_test, thresholds
        )
        parts.tracker.log_metrics(metrics)
        self._log_figures(pipeline, data)
        return metrics

    def _log_figures(self, pipeline: Pipeline, data: SplitData) -> None:
        proba = pipeline.predict_proba(data.x_test)[:, 1]
        reporter, tracker = self._parts.reporter, self._parts.tracker
        tracker.log_figure(
            reporter.precision_recall_figure(data.y_test, proba),
            "plots/precision_recall.png",
        )
        tracker.log_figure(
            reporter.confusion_matrix_figure(
                data.y_test, proba, self._params.train.threshold
            ),
            "plots/confusion_matrix.png",
        )

    def _register_and_decide(
        self, run_id: str, model_uri: str, data: SplitData, metrics: Metrics
    ) -> tuple[int, PromotionDecision]:
        registry = self._parts.registry
        version = registry.register(model_uri, tags=self._version_tags(run_id))
        decision = self._parts.gate.decide(
            metrics[self._score_key], self._champion_score(data)
        )
        if decision.promote:
            registry.promote(version)
        self._log_decision(version, decision)
        logger.info("Versão %s: %s (%s)", version, decision.promote, decision.reason)
        return version, decision

    def _champion_score(self, data: SplitData) -> float | None:
        """Reavalia o champion atual no mesmo teste, para uma comparação justa."""
        registry = self._parts.registry
        if registry.champion_version() is None:
            return None
        metrics = self._parts.evaluator.evaluate(
            registry.load_champion(), data.x_test, data.y_test, thresholds=[]
        )
        return metrics[self._score_key]

    def _version_tags(self, run_id: str) -> dict[str, str]:
        return {
            "run_id": run_id,
            "threshold": str(self._params.train.threshold),
            "primary_metric": self._params.train.primary_metric,
            **self._parts.lineage.collect(),
        }

    def _log_decision(self, version: int, decision: PromotionDecision) -> None:
        tracker = self._parts.tracker
        tracker.set_tags(
            {
                "registered_version": str(version),
                "promotion": "promoted" if decision.promote else "kept_champion",
                "promotion_reason": decision.reason,
            }
        )
        if decision.champion_score is not None:
            tracker.log_metrics(
                {f"champion_{self._score_key}": decision.champion_score}
            )
