"""Monte Carlo finishing-order simulation from driver strengths.

The earlier version perturbed win probabilities with unvalidated tyre and
fuel noise and then drew one winner per simulation, which produced coarse,
tied and sometimes zero probabilities. Uncertainty about pace, tyres, fuel
and safety cars is already inside the fitted strengths (they are trained on
real finishing orders), so the simulation now does one job: draw complete
Plackett-Luce finishing orders and count where each driver ends up.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from apex.models.plackett_luce import position_probabilities


def _log_strengths(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("scores must be finite")
    # Scores in (0, 1) are win scores; anything else is already a log-strength.
    if ((values > 0) & (values < 1)).all():
        clipped = np.clip(values, 1e-7, 1 - 1e-7)
        return np.log(clipped / (1.0 - clipped))
    return values


def simulate_finish_distribution(
    data: pd.DataFrame, scores: np.ndarray, *, simulations: int = 20000, seed: int = 0
) -> np.ndarray:
    """Return ``P[row, position]`` for every race in ``data``."""
    if simulations < 1:
        raise ValueError("simulations must be positive")
    if len(data) != len(scores):
        raise ValueError("data and scores must have the same length")
    strengths = _log_strengths(scores)
    sizes = data.groupby("race_id").size()
    matrix = np.zeros((len(data), int(sizes.max())))
    frame = data[["race_id"]].reset_index(drop=True)
    for offset, (_, positions) in enumerate(frame.groupby("race_id", sort=False).groups.items()):
        rows = np.asarray(list(positions), dtype=int)
        matrix[np.ix_(rows, np.arange(len(rows)))] = position_probabilities(
            strengths[rows], simulations=simulations, seed=seed + offset
        )
    return matrix


def monte_carlo_race_probabilities(
    data: pd.DataFrame, scores: np.ndarray, *, simulations: int = 20000, seed: int = 0, **_: object
) -> np.ndarray:
    """Mutually exclusive race-win probabilities (exact, strictly positive)."""
    return simulate_finish_distribution(data, scores, simulations=simulations, seed=seed)[:, 0]
