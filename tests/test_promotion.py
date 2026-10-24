"""Testes do portão de promoção."""

from pos_mle_tech_challenge_2.training.promotion import PromotionGate


def test_first_model_is_always_promoted() -> None:
    decision = PromotionGate().decide(candidate_score=0.5, champion_score=None)

    assert decision.promote is True
    assert decision.reason == "primeiro modelo"


def test_better_candidate_is_promoted() -> None:
    decision = PromotionGate().decide(candidate_score=0.76, champion_score=0.75)

    assert decision.promote is True
    assert decision.champion_score == 0.75


def test_worse_candidate_is_not_promoted() -> None:
    decision = PromotionGate().decide(candidate_score=0.70, champion_score=0.75)

    assert decision.promote is False


def test_tie_does_not_replace_the_champion() -> None:
    decision = PromotionGate().decide(candidate_score=0.75, champion_score=0.75)

    assert decision.promote is False
