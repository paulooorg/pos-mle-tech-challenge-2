"""Estágio `preprocess` do DVC: limpeza e split estratificado."""

import logging

import pandas as pd

from pos_mle_tech_challenge_2.config import (
    TEST_FILENAME,
    TRAIN_FILENAME,
    Settings,
    configure_logging,
    load_settings,
)
from pos_mle_tech_challenge_2.data.cleaner import DataCleaner
from pos_mle_tech_challenge_2.data.loader import DataLoader
from pos_mle_tech_challenge_2.data.splitter import DataSplitter
from pos_mle_tech_challenge_2.params import Params, load_params

logger = logging.getLogger(__name__)


class PreprocessPipeline:
    """Lê o dado bruto, limpa, divide e grava treino e teste."""

    def __init__(self, settings: Settings, params: Params) -> None:
        self._settings = settings
        self._params = params

    def run(self) -> None:
        """Executa o estágio e grava `train.csv` e `test.csv`."""
        df = DataLoader(self._settings.data_raw_path).load()
        df = DataCleaner(self._params.split.drop_duplicates).clean(df)
        splitter = DataSplitter(self._params.split.test_size, self._params.seed)
        train, test = splitter.split(df)
        self._save(train, test)

    def _save(self, train: pd.DataFrame, test: pd.DataFrame) -> None:
        output_dir = self._settings.data_processed_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        train.to_csv(output_dir / TRAIN_FILENAME, index=False)
        test.to_csv(output_dir / TEST_FILENAME, index=False)
        logger.info("Treino: %s linhas | Teste: %s linhas", len(train), len(test))


def main() -> None:
    """Ponto de entrada do estágio (`python -m ...pipelines.preprocess`)."""
    settings = load_settings()
    configure_logging(settings.log_level)
    PreprocessPipeline(settings, load_params()).run()


if __name__ == "__main__":
    main()
