"""Race-level winner ranking with chronological holdout evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from apex.features.winner_features import MODEL_COLUMNS, validate_race_targets
from apex.models.overfit import diagnose_fit


def race_probabilities(
    data: pd.DataFrame, scores: np.ndarray, *, temperature: float = 1.0
) -> np.ndarray:
    """Normalize driver scores into mutually exclusive race-win probabilities."""
    if len(data) != len(scores):
        raise ValueError("data and scores must have the same length")
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero")
    probabilities = np.empty(len(scores), dtype=float)
    clipped = np.clip(np.asarray(scores, dtype=float), 1e-7, 1 - 1e-7)
    strengths = np.log(clipped / (1.0 - clipped)) / temperature
    frame = data[["race_id"]].reset_index(drop=True)
    for _, positions in frame.groupby("race_id", sort=False).groups.items():
        indexes = np.asarray(list(positions), dtype=int)
        local = strengths[indexes]
        local -= local.max()
        exponentials = np.exp(local)
        probabilities[indexes] = exponentials / exponentials.sum()
    return probabilities


def _calibration_error(probabilities: np.ndarray, targets: np.ndarray, bins: int = 10) -> float:
    """Return expected calibration error for binary winner events."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (probabilities >= lower) & (
            (probabilities < upper) if upper < 1.0 else (probabilities <= upper)
        )
        if selected.any():
            error += selected.mean() * abs(
                probabilities[selected].mean() - targets[selected].mean()
            )
    return float(error)


def _race_metrics(
    data: pd.DataFrame,
    scores: np.ndarray,
    *,
    probabilities: np.ndarray | None = None,
) -> dict:
    columns = ["race_id", "winner"]
    if "driver" in data.columns:
        columns.append("driver")
    scored = data[columns].copy().reset_index(drop=True)
    scored["score"] = np.asarray(scores, dtype=float)
    scored["race_probability"] = (
        race_probabilities(scored, scored["score"].to_numpy())
        if probabilities is None
        else np.asarray(probabilities, dtype=float)
    )
    winners = scored.loc[scored.groupby("race_id")["race_probability"].idxmax()]
    actual = scored.loc[scored["winner"].eq(1)].set_index("race_id")
    actual_probability = actual["race_probability"].clip(1e-12, 1.0)
    race_groups = scored.groupby("race_id", sort=False)
    winner_ranks = race_groups["race_probability"].rank(method="min", ascending=False)
    actual_ranks = winner_ranks[scored["winner"].eq(1)]
    multiclass_brier = (
        (scored["race_probability"] - scored["winner"]) ** 2
    ).groupby(scored["race_id"]).sum().mean()
    metrics = {
        "races": int(scored["race_id"].nunique()),
        "winner_accuracy": float((winners.set_index("race_id")["winner"] == 1).mean()),
        "actual_winner_mean_score": float(actual["score"].mean()),
        "winner_mean_reciprocal_rank": float((1.0 / actual_ranks).mean()),
        "race_log_loss": float(-np.log(actual_probability).mean()),
        "race_brier_score": float(multiclass_brier),
        "calibration_error": _calibration_error(
            scored["race_probability"].to_numpy(),
            scored["winner"].to_numpy(),
        ),
    }
    if "finish_position" in data.columns:
        scored["finish_position"] = data["finish_position"].to_numpy()
        actual_top3 = scored[scored["finish_position"].le(3)]
        predicted_top3 = (
            scored.sort_values(["race_id", "race_probability"], ascending=[True, False])
            .groupby("race_id")
            .head(3)
        )
        if "driver" in scored:
            overlap = predicted_top3.merge(
                actual_top3[["race_id", "driver"]],
                on=["race_id", "driver"],
                how="inner",
            ).groupby("race_id").size()
        else:
            overlap = pd.Series(dtype=float)
        metrics["top_three_overlap"] = float(
            overlap.reindex(actual_top3["race_id"].unique(), fill_value=0).mean() / 3.0
        )
    return metrics


def train_winner_model(
    data: pd.DataFrame,
    train_years: list[int],
    val_years: list[int],
    test_years: list[int],
    seed: int = 0,
) -> tuple[dict, object]:
    """Train and evaluate a winner ranking model without race-level leakage."""
    validate_race_targets(data)
    sets = {"train": set(train_years), "val": set(val_years), "test": set(test_years)}
    if any(sets[a] & sets[b] for i, a in enumerate(sets) for b in list(sets)[i + 1:]):
        raise ValueError("train, validation, and test seasons must be disjoint")
    if data.groupby("race_id")["season"].nunique().max() > 1:
        raise ValueError("race_id maps to multiple seasons")
    masks = {name: data["season"].isin(years) for name, years in sets.items()}
    if any(not mask.any() for mask in masks.values()):
        raise ValueError("every split must contain race rows")
    model = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_leaf_nodes=7, random_state=seed),
    )
    model.fit(data.loc[masks["train"], MODEL_COLUMNS], data.loc[masks["train"], "winner"])
    metrics = {}
    for name, mask in masks.items():
        scores = model.predict_proba(data.loc[mask, MODEL_COLUMNS])[:, 1]
        metrics[name] = _race_metrics(data.loc[mask], scores)
    metrics["fit_diagnostics"] = diagnose_fit(
        metrics["train"], metrics["val"], metrics["test"]
    )
    return metrics, model
