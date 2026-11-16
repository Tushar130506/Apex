"""Finish-order race model: fit, calibrate chronologically, and predict.

The model is a blend of a regularised linear Plackett-Luce model and
gradient-boosted trees trained on the same exploded-logit loss. It outputs a
log-strength per driver. Everything shown to a user (win probability, podium
probability, expected finishing position, predicted order) is derived from
those strengths, so the numbers are mutually consistent.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from apex.features.form_features import ORDER_MODEL_COLUMNS, race_order
from apex.models.plackett_luce import (
    DEFAULT_STAGE_WEIGHTS,
    BoostedPlackettLuce,
    LinearPlackettLuce,
    break_ties,
    build_layout,
    fit_temperature,
    position_probabilities,
    win_probabilities,
)

MIN_CALIBRATION_RACES = 12
POINTS_POSITIONS = 10


def _has_lightgbm() -> bool:
    try:
        import lightgbm  # noqa: F401
    except Exception:  # pragma: no cover - optional dependency
        return False
    return True


@dataclass
class RaceOrderModel:
    """Blend of linear and boosted Plackett-Luce models."""

    columns: list[str] = field(default_factory=lambda: list(ORDER_MODEL_COLUMNS))
    seed: int = 0
    halflife_races: float | None = 60.0
    use_boosting: bool = True
    stage_weights: tuple[float, ...] = DEFAULT_STAGE_WEIGHTS
    linear_: LinearPlackettLuce | None = None
    boosted_: BoostedPlackettLuce | None = None
    training_races_: int = 0

    def fit(self, data: pd.DataFrame) -> "RaceOrderModel":
        """Fit on completed races. Rows without a finishing position are ignored
        as outcomes but still count as rivals that were beaten."""
        missing = {"race_id", "finish_position", *self.columns} - set(data.columns)
        if missing:
            raise ValueError(f"training data missing columns: {sorted(missing)}")
        finish = pd.to_numeric(data["finish_position"], errors="coerce")
        completed = finish.gt(0).groupby(data["race_id"]).transform("any")
        train = data[completed]
        if train.empty:
            raise ValueError("no completed races to train on")
        weight = None
        if self.halflife_races is not None:
            order = race_order(train["race_id"])
            age = order.rank(method="dense", ascending=False) - 1.0
            weight = 0.5 ** (age / self.halflife_races)
        layout = build_layout(train["race_id"], train["finish_position"], weight)
        features = train[self.columns]
        self.linear_ = LinearPlackettLuce(stage_weights=self.stage_weights).fit(features, layout)
        self.boosted_ = None
        # Trees need a reasonable number of races before they add anything.
        if self.use_boosting and layout.races >= 30 and _has_lightgbm():
            self.boosted_ = BoostedPlackettLuce(
                stage_weights=self.stage_weights, seed=self.seed
            ).fit(features, layout)
        self.training_races_ = layout.races
        return self

    def strengths(self, data: pd.DataFrame) -> np.ndarray:
        """Uncalibrated log-strengths (higher is better)."""
        if self.linear_ is None:
            raise RuntimeError("model is not fitted")
        values = self.linear_.strengths(data[self.columns])
        if self.boosted_ is not None:
            values = 0.5 * (values + self.boosted_.strengths(data[self.columns]))
        return values


def describe_race(
    race: pd.DataFrame,
    strengths: np.ndarray,
    *,
    later_strengths: np.ndarray | None = None,
    simulations: int = 20000,
    seed: int = 0,
) -> pd.DataFrame:
    """Turn one race's calibrated strengths into the published forecast columns.

    ``strengths`` decide the win; ``later_strengths`` (same drivers, usually a
    higher temperature) decide the places behind the winner.
    """
    if race["race_id"].nunique() != 1:
        raise ValueError("describe_race expects rows from exactly one race")
    result = race.copy()
    tiebreakers = [
        result[column].to_numpy()
        for column in ("grid_filled", "driver_points_fast_rank")
        if column in result
    ]
    tiebreakers.append(result["driver"].astype(str).rank(method="dense").to_numpy())
    strict = break_ties(np.asarray(strengths, dtype=float), *tiebreakers)
    later = None
    if later_strengths is not None:
        later = break_ties(np.asarray(later_strengths, dtype=float), *tiebreakers)
    matrix = position_probabilities(
        strict, later_strengths=later, simulations=simulations, seed=seed
    )
    positions = np.arange(1, len(result) + 1)
    result["strength"] = strict - strict.max()
    result["race_probability"] = matrix[:, 0]
    result["podium_probability"] = matrix[:, :3].sum(axis=1)
    result["points_probability"] = matrix[:, :POINTS_POSITIONS].sum(axis=1)
    result["expected_position"] = matrix @ positions
    # Strict order: strength first, never row order.
    result["predicted_position"] = (
        pd.Series(-strict, index=result.index).rank(method="first").astype(int)
    )
    return result


def _calibration_temperatures(history: list[pd.DataFrame]) -> tuple[float, float]:
    """Temperatures for the win and for the places behind it."""
    if len(history) < MIN_CALIBRATION_RACES:
        return 1.0, 1.0
    scored = pd.concat(history, ignore_index=True)
    layout = build_layout(scored["race_id"], scored["finish_position"])
    raw = scored["raw_strength"].to_numpy()
    win = fit_temperature(raw, layout)
    # Stage 1 gets zero weight: only second and third place are scored here.
    later = fit_temperature(raw, layout, stage_weights=(0.0, 1.0, 1.0), bounds=(0.4, 4.0))
    return win, later


def walk_forward_order(
    data: pd.DataFrame,
    target_orders: list[int],
    *,
    columns: list[str] | None = None,
    seed: int = 0,
    halflife_races: float | None = 60.0,
    use_boosting: bool = True,
    simulations: int = 20000,
    calibration_history: list[pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Predict each target race from strictly earlier races.

    For every target the model is refitted on all earlier completed races. The
    temperature is fitted only on earlier *out-of-sample* predictions from this
    same loop, so calibration never sees the race it is applied to.
    """
    frame = data.copy()
    frame["_order"] = race_order(frame["race_id"])
    history = list(calibration_history or [])
    predictions = []
    for position, target_order in enumerate(sorted(target_orders)):
        target = frame[frame["_order"].eq(target_order)]
        training = frame[frame["_order"].lt(target_order)]
        if target.empty:
            raise ValueError(f"no rows for target race order {target_order}")
        model = RaceOrderModel(
            columns=list(columns or ORDER_MODEL_COLUMNS),
            seed=seed,
            halflife_races=halflife_races,
            use_boosting=use_boosting,
        ).fit(training)
        raw = model.strengths(target)
        temperature, order_temperature = _calibration_temperatures(history)
        described = describe_race(
            target.drop(columns="_order"),
            raw / temperature,
            later_strengths=raw / order_temperature,
            simulations=simulations,
            seed=seed + position,
        )
        described["raw_strength"] = raw
        described["temperature"] = temperature
        described["order_temperature"] = order_temperature
        described["training_races"] = model.training_races_
        described["training_through_race"] = int(training["_order"].max())
        predictions.append(described)
        finish = pd.to_numeric(target["finish_position"], errors="coerce")
        if finish.gt(0).any():
            history.append(
                pd.DataFrame(
                    {
                        "race_id": target["race_id"].to_numpy(),
                        "finish_position": finish.to_numpy(),
                        "raw_strength": raw,
                    }
                )
            )
    return pd.concat(predictions, ignore_index=True)


def evaluate_predictions(predictions: pd.DataFrame, probability: str = "race_probability") -> dict:
    """Race-level metrics for completed races in a prediction table."""
    frame = predictions.copy()
    finish = pd.to_numeric(frame["finish_position"], errors="coerce")
    frame = frame[finish.gt(0).groupby(frame["race_id"]).transform("any")].copy()
    if frame.empty:
        return {"races": 0}
    frame["_finish"] = pd.to_numeric(frame["finish_position"], errors="coerce")
    frame["_win"] = frame["_finish"].eq(1).astype(float)
    frame["_rank"] = frame.groupby("race_id")[probability].rank(method="first", ascending=False)
    winners = frame[frame["_win"].eq(1)]
    top3 = frame[frame["_rank"].le(3)].assign(hit=lambda rows: rows["_finish"].le(3))
    metrics = {
        "races": int(frame["race_id"].nunique()),
        "winner_accuracy": float(winners["_rank"].eq(1).mean()),
        "winner_mean_reciprocal_rank": float((1.0 / winners["_rank"]).mean()),
        "winner_in_top_three": float(winners["_rank"].le(3).mean()),
        "top_three_overlap": float(top3.groupby("race_id")["hit"].sum().mean() / 3.0),
        "race_log_loss": float(-np.log(winners[probability].clip(lower=1e-12)).mean()),
        "race_brier_score": float(
            ((frame[probability] - frame["_win"]) ** 2).groupby(frame["race_id"]).sum().mean()
        ),
    }
    if "podium_probability" in frame:
        podium = frame["_finish"].le(3).astype(float)
        metrics["podium_brier_score"] = float(((frame["podium_probability"] - podium) ** 2).mean())
    if "expected_position" in frame:
        ranked = frame.dropna(subset=["_finish"])
        correlation = ranked.groupby("race_id").apply(
            lambda rows: rows["expected_position"].corr(rows["_finish"], method="spearman"),
            include_groups=False,
        )
        metrics["finish_order_spearman"] = float(correlation.mean())
    return metrics


def reliability_table(
    predictions: pd.DataFrame,
    probability: str,
    outcome: pd.Series,
    edges: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0),
) -> pd.DataFrame:
    """Predicted versus observed frequency inside fixed probability bands."""
    frame = pd.DataFrame(
        {"p": predictions[probability].to_numpy(), "y": np.asarray(outcome, dtype=float)}
    ).dropna()
    frame["band"] = pd.cut(frame["p"], bins=list(edges), include_lowest=True)
    table = frame.groupby("band", observed=True).agg(
        rows=("y", "size"), predicted=("p", "mean"), observed=("y", "mean")
    )
    table.insert(0, "band", [f"{band.left:.2f}-{band.right:.2f}" for band in table.index])
    return table.reset_index(drop=True)


def paired_bootstrap(
    model_loss: np.ndarray, baseline_loss: np.ndarray, *, draws: int = 10000, seed: int = 0
) -> dict:
    """Bootstrap the per-race loss difference (model minus baseline)."""
    difference = np.asarray(model_loss, dtype=float) - np.asarray(baseline_loss, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(difference, size=(draws, len(difference)), replace=True).mean(axis=1)
    return {
        "mean_difference": float(difference.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "probability_model_better": float((samples < 0).mean()),
    }


__all__ = [
    "RaceOrderModel",
    "describe_race",
    "evaluate_predictions",
    "paired_bootstrap",
    "reliability_table",
    "walk_forward_order",
    "win_probabilities",
]
