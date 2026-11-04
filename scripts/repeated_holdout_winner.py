#!/usr/bin/env python
"""Expanding holdout evaluation and future winner forecasts.

The final model uses 2018-2025 for the 2026 forecast. Races after the midpoint
of 2025 are scored by a model trained only through 2024.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import pandas as pd

sys.path.insert(0, "src")
from apex.data_io import load_processed
from apex.features.winner_features import (
    MODEL_COLUMNS,
    attach_grid_features,
    attach_historical_features,
    attach_race_results,
    build_pre_race_features,
    validate_race_targets,
)
from apex.models.winner import _race_metrics, race_probabilities
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline


def fit(data: pd.DataFrame, years: list[int]) -> object:
    model = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_leaf_nodes=7, random_state=0),
    )
    train = data[data["season"].isin(years)]
    model.fit(train[MODEL_COLUMNS], train["winner"])
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--results", default="data/processed/race_results_2018_2025.parquet")
    parser.add_argument("--future-results", default="data/processed/race_results_2026.parquet")
    parser.add_argument("--out", default="runs/winner_repeated")
    args = parser.parse_args()

    laps = load_processed(args.data, list(range(2018, 2027)))
    features = build_pre_race_features(laps)
    results = pd.concat(
        [pd.read_parquet(args.results), pd.read_parquet(args.future_results)],
        ignore_index=True,
    )
    data = attach_race_results(features, results)
    validate_race_targets(data)

    folds = []
    for year in range(2019, 2026):
        train_years = list(range(2018, year))
        validation = data[data["season"].eq(year)]
        model = fit(data, train_years)
        scores = model.predict_proba(validation[MODEL_COLUMNS])[:, 1]
        metrics = _race_metrics(validation, scores)
        metrics.update({"train_through": year - 1, "validation_year": year})
        folds.append(metrics)

    # Post-midpoint 2025: do not train on any 2025 outcomes.
    model_2025 = fit(data, list(range(2018, 2025)))
    post_2025 = data[(data["season"] == 2025) & (data["race_id"].str.split("_").str[-1].astype(int) > 12)].copy()
    post_2025["score"] = model_2025.predict_proba(post_2025[MODEL_COLUMNS])[:, 1]
    post_2025["race_probability"] = race_probabilities(post_2025, post_2025["score"].to_numpy())

    # 2026 is a future/out-of-domain forecast using all completed historical seasons.
    model_2026 = fit(data, list(range(2018, 2026)))
    future = attach_grid_features(
        features[features["season"].eq(2026)].copy(),
        results[results["season"].eq(2026)],
    )
    future = future.merge(
        results[results["season"].eq(2026)][["race_id", "driver", "team"]]
        .drop_duplicates(["race_id", "driver"]),
        on=["race_id", "driver"],
        how="left",
        suffixes=("", "_result"),
    )
    if "team_result" in future:
        future["team"] = future["team"].fillna(future.pop("team_result"))
    historical_results = results[results["season"].lt(2026)]
    future = attach_historical_features(future, historical_results)
    future["score"] = model_2026.predict_proba(future[MODEL_COLUMNS])[:, 1]
    future["race_probability"] = race_probabilities(future, future["score"].to_numpy())
    top3 = future.sort_values(
        ["race_id", "race_probability"], ascending=[True, False]
    ).groupby("race_id").head(3)
    model_all_labeled = fit(data, sorted(data["season"].unique().tolist()))

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(folds).to_json(output / "repeated_holdout_metrics.json", orient="records", indent=2)
    post_2025.to_parquet(output / "post_mid_2025_predictions.parquet", index=False)
    top3.to_parquet(output / "2026_top3_predictions.parquet", index=False)
    joblib.dump(model_2026, output / "model_2018_2025.joblib")
    joblib.dump(model_all_labeled, output / "model_all_labeled_seasons.joblib")

    print(json.dumps({"folds": folds, "post_mid_2025_races": int(post_2025.race_id.nunique()),
                      "2026_races": int(top3.race_id.nunique())}, indent=2))
    for race_id, group in top3.groupby("race_id", sort=True):
        print(race_id, group.sort_values("score", ascending=False)["driver"].tolist())


if __name__ == "__main__":
    main()
