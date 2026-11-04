"""Race-by-race walk-forward winner evaluation."""
from __future__ import annotations

import pandas as pd
import numpy as np

from apex.features.winner_features import MODEL_COLUMNS, validate_race_targets
from apex.models.calibration import calibrated_race_probabilities
from apex.models.overfit import diagnose_fit
from apex.models.race_monte_carlo import monte_carlo_race_probabilities
from apex.models.winner import _race_metrics, race_probabilities


def _baseline_accuracy(data: pd.DataFrame, column: str, ascending: bool) -> float | None:
    if column not in data or data[column].notna().groupby(data["race_id"]).sum().eq(0).all():
        return None
    predicted = (
        data.assign(_available=data[column].notna())
        .sort_values(["race_id", "_available", column], ascending=[True, False, ascending])
        .groupby("race_id")
        .head(1)
    )
    return float(predicted["winner"].mean())


def _calibrated_race_probabilities(
    training: pd.DataFrame,
    target: pd.DataFrame,
    scores: np.ndarray,
    prior_scores: list[np.ndarray],
    prior_targets: list[np.ndarray],
) -> np.ndarray:
    """Calibrate only against outcomes from earlier walk-forward races."""
    if not prior_scores:
        return race_probabilities(target, scores)
    history = pd.DataFrame(
        {
            "race_id": [
                f"history_{index}" for index, scores_for_race in enumerate(prior_scores)
                for _ in scores_for_race
            ],
            "score": np.concatenate(prior_scores),
            "winner": np.concatenate(prior_targets),
        }
    )
    return calibrated_race_probabilities(history, target, scores)


def _race_order(frame: pd.DataFrame) -> pd.Series:
    parts = frame["race_id"].astype(str).str.extract(r"^(?P<season>\d{4})_(?P<round>\d+)$")
    if parts.isna().any().any():
        raise ValueError("race_id values must have the form YYYY_round")
    return parts["season"].astype(int) * 100 + parts["round"].astype(int)


def walk_forward_predictions(
    data: pd.DataFrame,
    season: int,
    half_races: int | None = None,
    seed: int = 0,
    model_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Predict post-half-season races and append each completed race in order."""
    validate_race_targets(data)
    columns = model_columns or MODEL_COLUMNS
    missing = set(columns) - set(data.columns)
    if missing:
        raise ValueError(f"walk-forward data missing model columns: {sorted(missing)}")
    frame = data.copy()
    frame["_race_order"] = _race_order(frame)
    season_rows = frame[frame["season"].eq(season)]
    race_orders = sorted(season_rows["_race_order"].unique())
    if len(race_orders) < 2:
        raise ValueError("selected season must contain at least two races")
    half = half_races if half_races is not None else len(race_orders) // 2
    if half < 1 or half >= len(race_orders):
        raise ValueError("half_races must be between 1 and the number of races minus 1")
    predictions: list[pd.DataFrame] = []
    metrics: list[dict] = []
    prior_scores: list[np.ndarray] = []
    prior_targets: list[np.ndarray] = []
    for target_order in race_orders[half:]:
        target = frame[frame["_race_order"].eq(target_order)].copy()
        training = frame[frame["_race_order"].lt(target_order)]
        if training.empty:
            raise ValueError(f"no prior races available for {target['race_id'].iloc[0]}")
        model = make_walk_forward_model(seed)
        model.fit(training[columns], training["winner"])
        target["score"] = model.predict_proba(target[columns])[:, 1]
        target["race_probability"] = monte_carlo_race_probabilities(
            target, target["score"].to_numpy(), simulations=1000, seed=seed
        )
        target["calibrated_race_probability"] = _calibrated_race_probabilities(
            training,
            target,
            target["score"].to_numpy(),
            prior_scores,
            prior_targets,
        )
        target["predicted_position"] = (
            target["race_probability"].rank(method="first", ascending=False).astype(int)
        )
        target["training_races"] = int(training["race_id"].nunique())
        target["training_through_race"] = int(target_order - 1)
        predictions.append(target)
        race_metrics = _race_metrics(target, target["score"].to_numpy())
        race_metrics.update(
            {
                "race_id": str(target["race_id"].iloc[0]),
                "training_races": int(training["race_id"].nunique()),
                "grid_baseline_accuracy": _baseline_accuracy(
                    target, "grid_position", ascending=True
                ),
                "qualifying_baseline_accuracy": _baseline_accuracy(
                    target, "qualifying_pace_rank", ascending=True
                ),
                "form_baseline_accuracy": _baseline_accuracy(
                    target, "driver_form_score", ascending=False
                ),
                "calibrated_winner_accuracy": float(
                    target.loc[
                        target.groupby("race_id")["calibrated_race_probability"].idxmax(),
                        "winner",
                    ].eq(1).mean()
                ),
                "calibrated_race_log_loss": float(
                    -np.log(
                        target.loc[target["winner"].eq(1), "calibrated_race_probability"]
                        .clip(lower=1e-12)
                    ).mean()
                ),
            }
        )
        metrics.append(race_metrics)
        prior_scores.append(target["score"].to_numpy())
        prior_targets.append(target["winner"].to_numpy())
    return (
        pd.concat(predictions, ignore_index=True).drop(columns="_race_order"),
        metrics,
    )


def make_walk_forward_model(seed: int = 0) -> object:
    """Build the same classifier pipeline used by the winner model."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline

    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.05,
            max_leaf_nodes=7,
            random_state=seed,
        ),
    )
