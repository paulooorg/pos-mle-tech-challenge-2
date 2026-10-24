"""Testes dos contratos da API de inferência."""

from typing import Any

import pytest
from pydantic import ValidationError

from pos_mle_tech_challenge_2.data.schema import FEATURE_COLUMNS
from pos_mle_tech_challenge_2.serving.examples import EXAMPLE_SESSIONS
from pos_mle_tech_challenge_2.serving.schemas import (
    MAX_SESSIONS_PER_REQUEST,
    PredictRequest,
    SessionFeatures,
)

VALID: dict[str, Any] = EXAMPLE_SESSIONS["visitante_recorrente"]


def test_schema_has_exactly_the_training_feature_columns() -> None:
    assert set(SessionFeatures.model_fields) == set(FEATURE_COLUMNS)


@pytest.mark.parametrize("name", list(EXAMPLE_SESSIONS))
def test_documentation_examples_are_valid(name: str) -> None:
    assert SessionFeatures(**EXAMPLE_SESSIONS[name])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("PageValues", -1.0),
        ("BounceRates", 1.5),
        ("ExitRates", -0.1),
        ("SpecialDay", 2),
        ("Administrative", -1),
        ("Administrative", 2.5),
        ("Month", "Novv"),
        ("VisitorType", "Robot"),
        ("Browser", 0),
        ("Weekend", "talvez"),
    ],
)
def test_invalid_values_are_rejected(field: str, value: Any) -> None:
    with pytest.raises(ValidationError):
        SessionFeatures(**{**VALID, field: value})


def test_missing_column_is_rejected() -> None:
    incomplete = {k: v for k, v in VALID.items() if k != "PageValues"}

    with pytest.raises(ValidationError, match="PageValues"):
        SessionFeatures(**incomplete)


def test_unknown_column_is_rejected() -> None:
    with pytest.raises(ValidationError, match="extra"):
        SessionFeatures(**{**VALID, "Revenue": True})


def test_request_needs_at_least_one_session() -> None:
    with pytest.raises(ValidationError):
        PredictRequest(sessions=[])


def test_request_has_a_batch_size_limit() -> None:
    too_many = [VALID] * (MAX_SESSIONS_PER_REQUEST + 1)

    with pytest.raises(ValidationError):
        PredictRequest(sessions=too_many)
