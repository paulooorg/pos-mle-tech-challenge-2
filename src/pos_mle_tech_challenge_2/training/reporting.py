"""Figuras de avaliação registradas como artefatos do run."""

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay


class ReportBuilder:
    """Cria as figuras (curva Precision-Recall e matriz de confusão) do teste."""

    def precision_recall_figure(self, target: pd.Series, proba: np.ndarray) -> Figure:
        """Curva Precision-Recall, com a taxa de compra como referência."""
        figure = Figure(figsize=(6, 5))
        axes = figure.subplots()
        PrecisionRecallDisplay.from_predictions(target, proba, ax=axes)
        axes.axhline(target.mean(), color="gray", linestyle="--", label="baseline")
        axes.set_title("Curva Precision-Recall (teste)")
        axes.legend()
        return figure

    def confusion_matrix_figure(
        self, target: pd.Series, proba: np.ndarray, threshold: float
    ) -> Figure:
        """Matriz de confusão para o threshold informado."""
        figure = Figure(figsize=(5, 4))
        axes = figure.subplots()
        predicted = (proba >= threshold).astype(int)
        ConfusionMatrixDisplay.from_predictions(
            target, predicted, ax=axes, colorbar=False
        )
        axes.set_title(f"Matriz de confusão (threshold {threshold:g})")
        return figure
