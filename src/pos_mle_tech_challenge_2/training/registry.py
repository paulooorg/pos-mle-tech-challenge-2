"""Registro e promoção de modelos no MLflow Model Registry."""

from collections.abc import Mapping
from typing import Any

import mlflow
from mlflow import MlflowClient
from mlflow.entities.model_registry import RegisteredModel
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST, ErrorCode


class ModelRegistry:
    """Registra versões do modelo e controla qual delas é o `champion`."""

    def __init__(self, tracking_uri: str, model_name: str, champion_alias: str) -> None:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_registry_uri(tracking_uri)
        self._client = MlflowClient()
        self._model_name = model_name
        self._champion_alias = champion_alias

    @property
    def champion_uri(self) -> str:
        """URI que sempre aponta para o modelo champion atual."""
        return f"models:/{self._model_name}@{self._champion_alias}"

    def register(self, model_uri: str, tags: Mapping[str, str] | None = None) -> int:
        """Cria uma nova versão do modelo registrado.

        Args:
            model_uri: URI do modelo logado no run.
            tags: tags da versão (ex.: threshold, hash do dado, commit).

        Returns:
            Número da nova versão.
        """
        version = mlflow.register_model(
            model_uri, self._model_name, tags=dict(tags or {})
        )
        return int(version.version)

    def champion_version(self) -> int | None:
        """Versão que tem o alias de champion, ou None se ainda não existe."""
        registered = self._find_registered_model()
        if registered is None:
            return None
        version = registered.aliases.get(self._champion_alias)
        return int(version) if version is not None else None

    def load_champion(self) -> Any:
        """Carrega o pipeline do champion atual.

        Raises:
            LookupError: se ainda não houver champion.
        """
        if self.champion_version() is None:
            raise LookupError(f"Não há champion para {self._model_name!r}")
        return mlflow.sklearn.load_model(self.champion_uri)

    def promote(self, version: int) -> None:
        """Move o alias de champion para a versão informada."""
        self._client.set_registered_model_alias(
            self._model_name, self._champion_alias, str(version)
        )

    def _find_registered_model(self) -> RegisteredModel | None:
        try:
            return self._client.get_registered_model(self._model_name)
        except MlflowException as error:
            if error.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST):
                return None
            raise
