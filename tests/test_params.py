"""Testes do carregamento do params.yaml."""

from pathlib import Path

import pytest

from pos_mle_tech_challenge_2.params import DEFAULT_PARAMS_PATH, load_params


def test_real_params_file_is_valid() -> None:
    params = load_params(DEFAULT_PARAMS_PATH)

    assert params.seed == 42
    assert 0 < params.split.test_size < 1
    assert params.train.primary_metric == "average_precision"
    assert params.model.algorithm == "hist_gradient_boosting"
    assert params.model.params["class_weight"] == "balanced"


def test_params_are_parsed_into_typed_sections(tmp_path: Path) -> None:
    params_file = tmp_path / "params.yaml"
    params_file.write_text(
        "seed: 1\n"
        "split: {test_size: 0.3, drop_duplicates: false}\n"
        "train: {cv_folds: 3, primary_metric: roc_auc, threshold: 0.5}\n"
        "model: {type: random_forest, params: {n_estimators: 10}}\n"
    )

    params = load_params(params_file)

    assert params.seed == 1
    assert params.split.drop_duplicates is False
    assert params.train.cv_folds == 3
    assert params.model.algorithm == "random_forest"
    assert params.model.params == {"n_estimators": 10}


def test_missing_section_raises_key_error(tmp_path: Path) -> None:
    params_file = tmp_path / "params.yaml"
    params_file.write_text("seed: 1\n")

    with pytest.raises(KeyError):
        load_params(params_file)
