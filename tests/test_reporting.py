"""Testes das figuras de avaliação."""

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from pos_mle_tech_challenge_2.training.reporting import ReportBuilder

TARGET = pd.Series([0, 0, 1, 1, 0, 1])
PROBA = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.9])


def test_precision_recall_figure_has_title_and_baseline() -> None:
    figure = ReportBuilder().precision_recall_figure(TARGET, PROBA)

    assert isinstance(figure, Figure)
    assert "Precision-Recall" in figure.axes[0].get_title()
    assert figure.axes[0].get_legend() is not None


def test_confusion_matrix_figure_mentions_the_threshold() -> None:
    figure = ReportBuilder().confusion_matrix_figure(TARGET, PROBA, threshold=0.73)

    assert isinstance(figure, Figure)
    assert "0.73" in figure.axes[0].get_title()
