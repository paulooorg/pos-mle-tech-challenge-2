"""Rastreamento de experimentos no MLflow Tracking."""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import mlflow
import pandas as pd
from matplotlib.figure import Figure
from mlflow.models import infer_signature
from sklearn.pipeline import Pipeline

MODEL_ARTIFACT_NAME = "model"

# O MLflow serializa modelos sklearn com `skops`, que só aceita tipos confiáveis.
# O HistGradientBoosting usa este tipo interno, que precisa ser liberado de forma
# explícita (preferido a `cloudpickle`, que executa código ao carregar o arquivo).
TRUSTED_SKLEARN_TYPES = [
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
]


def flatten_params(params: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Achata dicionários aninhados em chaves com ponto (`model.params.max_depth`)."""
    flat: dict[str, Any] = {}
    for key, value in params.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(flatten_params(value, prefix=f"{name}."))
        else:
            flat[name] = value
    return flat


class ExperimentTracker:
    """Encapsula o MLflow Tracking: runs, parâmetros, métricas, figuras e modelo."""

    def __init__(self, tracking_uri: str, experiment_name: str) -> None:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    @contextmanager
    def start_run(self, run_name: str | None = None) -> Iterator[str]:
        """Abre um run e devolve o seu id; o run é fechado ao sair do bloco."""
        with mlflow.start_run(run_name=run_name) as run:
            yield run.info.run_id

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Registra parâmetros, achatando as seções aninhadas."""
        mlflow.log_params(flatten_params(params))

    def log_metrics(self, metrics: Mapping[str, float]) -> None:
        """Registra métricas numéricas."""
        mlflow.log_metrics(dict(metrics))

    def set_tags(self, tags: Mapping[str, str]) -> None:
        """Registra tags (ex.: hash do dado e commit do git)."""
        mlflow.set_tags(dict(tags))

    def log_figure(self, figure: Figure, artifact_file: str) -> None:
        """Salva uma figura do matplotlib como artefato do run."""
        mlflow.log_figure(figure, artifact_file)

    def log_model(self, pipeline: Pipeline, features: pd.DataFrame) -> str:
        """Registra o pipeline treinado, com assinatura e exemplo de entrada.

        Args:
            pipeline: pipeline já treinado.
            features: amostra de features usada para inferir a assinatura.

        Returns:
            URI do modelo registrado no run (`models:/m-...`).
        """
        info = mlflow.sklearn.log_model(
            pipeline,
            name=MODEL_ARTIFACT_NAME,
            signature=infer_signature(features, pipeline.predict_proba(features)),
            input_example=features.head(5),
            skops_trusted_types=TRUSTED_SKLEARN_TYPES,
        )
        return info.model_uri
