"""Portão de promoção: decide se um modelo novo vira champion."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PromotionDecision:
    """Resultado do portão, com o motivo para ficar registrado no run."""

    promote: bool
    reason: str
    candidate_score: float
    champion_score: float | None


class PromotionGate:
    """Promove o candidato só se ele superar o champion na métrica principal."""

    def decide(
        self, candidate_score: float, champion_score: float | None
    ) -> PromotionDecision:
        """Compara o candidato com o champion avaliado no mesmo conjunto de teste.

        Args:
            candidate_score: métrica principal do modelo novo.
            champion_score: métrica principal do champion atual, ou None se não
                houver champion.

        Returns:
            Decisão, com o motivo.
        """
        if champion_score is None:
            return PromotionDecision(True, "primeiro modelo", candidate_score, None)
        if candidate_score > champion_score:
            return PromotionDecision(
                True, "supera o champion", candidate_score, champion_score
            )
        return PromotionDecision(
            False, "não supera o champion", candidate_score, champion_score
        )
