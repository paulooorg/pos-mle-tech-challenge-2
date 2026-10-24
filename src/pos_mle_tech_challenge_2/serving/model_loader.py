"""Carregamento do modelo registrado no MLflow Model Registry."""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import mlflow
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
    RESOURCE_DOES_NOT_EXIST,
    ErrorCode,
)

from pos_mle_tech_challenge_2.serving.schemas import ModelInfo

logger = logging.getLogger(__name__)

CHAMPION = "champion"  # valor de MODEL_VERSION que segue o alias do champion
DEFAULT_THRESHOLD = 0.5
NOT_FOUND_CODES = frozenset(
    ErrorCode.Name(code) for code in (RESOURCE_DOES_NOT_EXIST, INVALID_PARAMETER_VALUE)
)


class ModelNotFoundError(LookupError):
    """A versão ou o alias pedido não existe no Registry."""


@dataclass(frozen=True)
class LoadedModel:
    """Pipeline treinado e as informações de onde ele veio."""

    pipeline: Any
    info: ModelInfo


class ModelLoader:
    """Resolve `MODEL_VERSION` (champion ou número) e carrega o modelo do Registry."""

    def __init__(self, tracking_uri: str, model_name: str, champion_alias: str) -> None:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_registry_uri(tracking_uri)
        self._client = MlflowClient()
        self._model_name = model_name
        self._champion_alias = champion_alias

    def load(self, requested: str) -> LoadedModel:
        """Carrega o modelo pedido.

        Args:
            requested: `champion` (segue o alias) ou o número de uma versão.

        Raises:
            ModelNotFoundError: se o pedido for inválido ou não existir no Registry.
        """
        version = self._resolve_version(requested)
        uri = f"models:/{self._model_name}/{version.version}"
        pipeline = mlflow.sklearn.load_model(uri)
        info = self._build_info(requested, version)
        logger.info("Modelo carregado: %s versão %s", info.name, info.version)
        return LoadedModel(pipeline, info)

    def _resolve_version(self, requested: str) -> ModelVersion:
        try:
            if requested == CHAMPION:
                return self._client.get_model_version_by_alias(
                    self._model_name, self._champion_alias
                )
            if requested.isdigit():
                return self._client.get_model_version(self._model_name, requested)
        except MlflowException as error:
            if error.error_code not in NOT_FOUND_CODES:
                raise
            raise ModelNotFoundError(
                f"Modelo {self._model_name!r} ({requested}) não encontrado no Registry"
            ) from error
        raise ModelNotFoundError(
            f"MODEL_VERSION inválido: {requested!r} (use 'champion' ou o número da versão)"
        )

    def _build_info(self, requested: str, version: ModelVersion) -> ModelInfo:
        tags = version.tags
        run_id = version.run_id or tags.get("run_id")
        return ModelInfo(
            name=self._model_name,
            version=int(version.version),
            requested=requested,
            run_id=run_id,
            threshold=self._threshold(tags),
            data_md5=tags.get("data_md5"),
            git_commit=tags.get("git_commit"),
            test_average_precision=self._run_metric(run_id, "test_average_precision"),
            loaded_at=datetime.now(UTC),
        )

    @staticmethod
    def _threshold(tags: dict[str, str]) -> float:
        if "threshold" not in tags:
            logger.warning("Versão sem tag 'threshold': usando %s", DEFAULT_THRESHOLD)
            return DEFAULT_THRESHOLD
        return float(tags["threshold"])

    def _run_metric(self, run_id: str | None, name: str) -> float | None:
        if run_id is None:
            return None
        return self._client.get_run(run_id).data.metrics.get(name)
