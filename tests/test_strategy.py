import numpy as np, pytest
from apex.strategy.pace_model import ParametricPace
from apex.strategy.race_simulator import Assumptions, RaceState, Strategy, candidates, evaluate, is_legal, recommend

ST = RaceState(lap=20, total_laps=55, compound="MEDIUM", tyre_age=18, fuel_kg=65.0, base_pace=88.0)
PACE = ParametricPace()


def test_legality_two_compound_rule():
    a = Assumptions()
    assert not is_legal(Strategy(((25, "MEDIUM"),)), ST, a)
    assert is_legal(Strategy(((25, "HARD"),)), ST, a)
    assert not is_legal(Strategy(((60, "HARD"),)), ST, a)
    assert all(is_legal(s, ST, a) for s in candidates(ST, a))


def test_reproducible_and_paired():
    t1, _ = evaluate(ST, PACE, n=500, seed=1)
    t2, _ = evaluate(ST, PACE, n=500, seed=1)
    assert t1.equals(t2)
    assert t1.prob_best.sum() == pytest.approx(1.0)


def test_cheap_pit_loss_means_pit_early_and_expensive_means_later():
    cheap = evaluate(ST, PACE, Assumptions(pit_loss_mean=8.0), n=600)[0].iloc[0]
    dear = evaluate(ST, PACE, Assumptions(pit_loss_mean=60.0), n=600)[0].iloc[0]
    assert cheap.n_stops >= dear.n_stops


def test_old_tyres_trigger_pit_soon():
    old = RaceState(30, 55, "SOFT", 30, 40.0, 88.0)
    rec = recommend(old, PACE, n=800)
    assert rec["best"].startswith("Pit") and rec["action"] in {"PIT_NOW", "STAY_OUT"}
    assert rec["confidence"] in {"HIGH", "MEDIUM", "LOW"} and rec["would_change_if"] is not None


def test_fallback_on_stale_or_ood():
    r = recommend(ST, PACE, data_ok=False, data_msg="session data incomplete")
    assert r["action"] == "INSUFFICIENT_CONFIDENCE" and "incomplete" in r["reasons"][0]
    r = recommend(ST, PACE, ood_reasons=["tyre_age=70 outside training range"])
    assert r["action"] == "INSUFFICIENT_CONFIDENCE"
