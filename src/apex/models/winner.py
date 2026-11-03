"""Race-level winner ranking with chronological holdout evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from apex.features.winner_features import MODEL_COLUMNS, validate_race_targets


def _race_metrics(data: pd.DataFrame, scores: np.ndarray) -> dict:
    scored = data[["race_id", "winner"]].copy()
    scored["score"] = scores
    winners = scored.loc[scored.groupby("race_id")["score"].idxmax()]
    actual = scored.loc[scored["winner"].eq(1)].set_index("race_id")
    return {
        "races": int(scored["race_id"].nunique()),
        "winner_accuracy": float((winners.set_index("race_id")["winner"] == 1).mean()),
        "actual_winner_mean_score": float(actual["score"].mean()),
    }


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
    return metrics, model
