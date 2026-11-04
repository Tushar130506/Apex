"""Transparent Monte-Carlo strategy simulator. Explicit assumptions, common random numbers, no LLM calls."""
from __future__ import annotations
from dataclasses import dataclass, field, replace
from itertools import product
import numpy as np
import pandas as pd

from apex.strategy.pace_model import DRY, ParametricPace
from apex.models.safety_hazard import sample_safety_car_starts


@dataclass(frozen=True)
class RaceState:
    lap: int                 # lap about to be driven (1-based)
    total_laps: int
    compound: str
    tyre_age: int
    fuel_kg: float
    base_pace: float         # session base pace (s), see ParametricPace.estimate_base
    available_compounds: tuple = DRY


@dataclass(frozen=True)
class Assumptions:
    pit_loss_mean: float = 22.0
    pit_loss_sd: float = 1.2
    lap_noise_sd: float = 0.25
    deg_mult_sigma: float = 0.25      # lognormal sigma on tyre degradation rate (per stint)
    sc_prob_per_race_remaining: float = 0.25   # chance of one safety car in remaining laps
    sc_laps: int = 4
    sc_slowdown: float = 1.35
    sc_pit_loss_factor: float = 0.45
    deg_mult_scale: float = 1.0       # sensitivity knob
    require_two_compounds: bool = True
    sc_hazard_by_lap: tuple[tuple[int, float], ...] = ()


@dataclass(frozen=True)
class Strategy:
    stops: tuple = ()        # ((pit_at_end_of_lap, compound_after), ...)

    def label(self, state: RaceState) -> str:
        if not self.stops:
            return "No stop"
        return " / ".join(f"Pit L{l} -> {c[:1]}" for l, c in self.stops)


def is_legal(s: Strategy, st: RaceState, a: Assumptions) -> bool:
    laps = [l for l, _ in s.stops]
    if laps != sorted(set(laps)) or any(not st.lap <= l < st.total_laps for l in laps):
        return False
    if a.require_two_compounds and len({st.compound, *[c for _, c in s.stops]} & set(DRY)) < 2:
        return False
    return all(c in st.available_compounds for _, c in s.stops)


def candidates(st: RaceState, a: Assumptions = Assumptions(), step: int = 3, max_stops: int = 2) -> list[Strategy]:
    grid = list(range(st.lap, st.total_laps, step))
    out = []
    for n in range(0, max_stops + 1):
        for laps in product(grid, repeat=n):
            if list(laps) != sorted(set(laps)) or (n == 2 and laps[1] - laps[0] < 8):
                continue
            for comps in product(st.available_compounds, repeat=n):
                s = Strategy(tuple(zip(laps, comps)))
                if is_legal(s, st, a):
                    out.append(s)
    return out


def draw(n: int, st: RaceState, a: Assumptions, seed: int = 0) -> dict:
    """Shared random draws so strategy comparisons are paired (common random numbers)."""
    r = np.random.default_rng(seed)
    remaining = st.total_laps - st.lap + 1
    hazard = dict(a.sc_hazard_by_lap)
    if not hazard:
        hazard = {lap: a.sc_prob_per_race_remaining for lap in range(1, remaining + 1)}
    sc_start = sample_safety_car_starts(r, remaining, hazard, n)
    return dict(
        deg_mult=np.exp(r.normal(0, a.deg_mult_sigma, (n, 4))) * a.deg_mult_scale,
        pit_loss=r.normal(a.pit_loss_mean, a.pit_loss_sd, (n, 3)),
        noise=r.normal(0, a.lap_noise_sd, (n, remaining)),
        sc_start=np.where(sc_start >= 0, st.lap + sc_start - 1, -10**6))


def simulate(s: Strategy, st: RaceState, pace: ParametricPace, a: Assumptions, d: dict) -> np.ndarray:
    n = d["deg_mult"].shape[0]
    total = np.zeros(n)
    comp, age, fuel, stint = st.compound, st.tyre_age, st.fuel_kg, 0
    stops = dict(s.stops)
    for i, lap in enumerate(range(st.lap, st.total_laps + 1)):
        t = st.base_pace + pace.rel(comp, age, fuel, d["deg_mult"][:, stint]) + d["noise"][:, i]
        under_sc = (lap >= d["sc_start"]) & (lap < d["sc_start"] + a.sc_laps)
        t = np.where(under_sc, st.base_pace * a.sc_slowdown, t)
        if lap in stops:
            loss = d["pit_loss"][:, stint] * np.where(under_sc, a.sc_pit_loss_factor, 1.0)
            t = t + loss
            comp, age, stint = stops[lap], 0, stint + 1
        else:
            age += 1
        total += t
        fuel = max(0.0, fuel - pace.burn_kg_per_lap)
    return total


def evaluate(st: RaceState, pace: ParametricPace, a: Assumptions = Assumptions(), n: int = 2000, seed: int = 0,
             strategies: list[Strategy] | None = None) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    strategies = strategies or candidates(st, a)
    d = draw(n, st, a, seed)
    sims = {s.label(st): simulate(s, st, pace, a, d) for s in strategies}
    M = np.column_stack(list(sims.values()))
    best = M.argmin(axis=1)
    rows = []
    for j, (lbl, v) in enumerate(sims.items()):
        tail = np.sort(v)[int(0.95 * n):]
        rows.append(dict(strategy=lbl, mean_s=v.mean(), p10_s=np.quantile(v, .1), p90_s=np.quantile(v, .9),
                         cvar95_s=tail.mean(), prob_best=float(np.mean(best == j)),
                         n_stops=len(strategies[j].stops), first_stop=strategies[j].stops[0][0] if strategies[j].stops else None))
    df = pd.DataFrame(rows).sort_values("mean_s").reset_index(drop=True)
    df["delta_vs_best_s"] = df["mean_s"] - df["mean_s"].iloc[0]
    return df, sims


def recommend(st: RaceState, pace: ParametricPace, a: Assumptions = Assumptions(), n: int = 2000, seed: int = 0,
              data_ok: bool = True, data_msg: str = "ok", ood_reasons: list[str] | None = None) -> dict:
    """Ranked, explained recommendation - or an explicit 'insufficient confidence' fallback."""
    if not data_ok or ood_reasons:
        return dict(action="INSUFFICIENT_CONFIDENCE", reasons=[] if data_ok else [data_msg],
                    ood=ood_reasons or [], message="No strategy call: inputs are stale, incomplete or outside the training domain.")
    table, sims = evaluate(st, pace, a, n, seed)
    best, second = table.iloc[0], table.iloc[1]
    diff = sims[best.strategy] - sims[second.strategy]
    p_better = float(np.mean(diff < 0))
    conf = "HIGH" if p_better >= 0.9 and second.delta_vs_best_s >= 1.0 else "MEDIUM" if p_better >= 0.7 else "LOW"
    action = "PIT_NOW" if best.first_stop == st.lap else "STAY_OUT"
    # sensitivity: does the winner change under harsher/milder assumptions?
    flips = {}
    for name, a2 in {"pit_loss +3s": replace(a, pit_loss_mean=a.pit_loss_mean + 3),
                     "pit_loss -3s": replace(a, pit_loss_mean=a.pit_loss_mean - 3),
                     "degradation x1.3": replace(a, deg_mult_scale=1.3),
                     "degradation x0.7": replace(a, deg_mult_scale=0.7),
                     "no safety car": replace(a, sc_prob_per_race_remaining=0.0)}.items():
        t2, _ = evaluate(st, pace, a2, max(500, n // 4), seed)
        flips[name] = t2.iloc[0].strategy
    changes = sorted({v for v in flips.values() if v != best.strategy})
    return dict(action=action, best=best.strategy, confidence=conf, p_better_than_second=p_better,
                expected_gain_vs_second_s=float(second.delta_vs_best_s), table=table.head(8),
                sensitivity_winners=flips, would_change_if=[k for k, v in flips.items() if v != best.strategy],
                explanation=(f"{best.strategy}: expected {best.mean_s:.1f}s (P10-P90 {best.p10_s:.1f}-{best.p90_s:.1f}), "
                             f"{second.delta_vs_best_s:.1f}s ahead of '{second.strategy}' in {p_better:.0%} of paired draws. "
                             f"Pace model: {pace.fit_provenance}. Assumes pit loss {a.pit_loss_mean:.0f}±{a.pit_loss_sd:.1f}s, "
                             f"SC chance {a.sc_prob_per_race_remaining:.0%}."),
                alternatives_in_sensitivity=changes)
