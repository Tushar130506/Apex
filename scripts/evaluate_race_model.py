#!/usr/bin/env python
"""Walk-forward evaluation of the finish-order race model against baselines.

Every race from ``--first-season`` onward is predicted from strictly earlier
races. The report compares the model with picking the pole-sitter, with a
grid-only probability model refitted the same way, and (when its stored
predictions are available) with the previous classifier.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from apex.features.form_features import build_order_features, race_order
from apex.models.race_order import (
    evaluate_predictions,
    paired_bootstrap,
    reliability_table,
    walk_forward_order,
)
from apex.models.walk_forward import MIN_TRAINING_RACES, baseline_pick, grid_only_probabilities


WIN_EDGES = (0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0)
PODIUM_EDGES = (0.0, 0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 1.0)


def _per_race(frame: pd.DataFrame, probability: str) -> pd.DataFrame:
    finish = pd.to_numeric(frame["finish_position"], errors="coerce")
    rank = frame.groupby("race_id")[probability].rank(method="first", ascending=False)
    winners = frame[finish.eq(1)]
    return pd.DataFrame(
        {
            "race_id": winners["race_id"].to_numpy(),
            "log_loss": -np.log(winners[probability].clip(lower=1e-12)).to_numpy(),
            "hit": rank[finish.eq(1)].eq(1).astype(float).to_numpy(),
        }
    ).set_index("race_id")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default="runs/walk_forward_championship/input_features.parquet")
    parser.add_argument("--first-season", type=int, default=2021)
    parser.add_argument("--previous-predictions", default="runs/walk_forward_championship_2023_2026/race_predictions.parquet")
    parser.add_argument("--out", default="runs/race_order_walk_forward")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    features = build_order_features(pd.read_parquet(args.features))
    order = race_order(features["race_id"])
    orders = sorted(order.unique())
    targets = [value for value in orders if value // 100 >= args.first_season]
    warmup = [
        value for position, value in enumerate(orders)
        if position >= MIN_TRAINING_RACES and value < targets[0]
    ]
    scored = walk_forward_order(features, warmup + targets, seed=args.seed)
    scored["_order"] = race_order(scored["race_id"])
    predictions = scored[scored["_order"].isin(targets)].drop(columns="_order").reset_index(drop=True)
    grid_only = features[["race_id", "driver"]].assign(
        grid_only_probability=grid_only_probabilities(features, targets)
    )
    predictions = predictions.merge(grid_only, on=["race_id", "driver"], validate="one_to_one")

    finish = pd.to_numeric(predictions["finish_position"], errors="coerce")
    winners = predictions[finish.eq(1)].set_index("race_id")["driver"]
    pole = baseline_pick(predictions, "grid_position", True)
    pole_hit = (pole.reindex(winners.index) == winners).astype(float)
    model_races = _per_race(predictions, "race_probability")
    grid_races = _per_race(predictions, "grid_only_probability")

    def summarise(frame: pd.DataFrame) -> dict:
        ids = frame["race_id"].unique()
        row = evaluate_predictions(frame)
        row["pole_sitter_accuracy"] = float(pole_hit.reindex(ids).mean())
        row["grid_only_log_loss"] = float(grid_races.loc[ids, "log_loss"].mean())
        row["grid_only_accuracy"] = float(grid_races.loc[ids, "hit"].mean())
        return row

    report = {
        "protocol": {
            "first_scored_season": args.first_season,
            "warmup_races_for_calibration": len(warmup),
            "refit": "before every race, on all earlier races",
            "seed": args.seed,
        },
        "overall": summarise(predictions),
        "by_season": {
            int(season): summarise(group) for season, group in predictions.groupby("season")
        },
        "model_vs_grid_only_log_loss": paired_bootstrap(
            model_races["log_loss"].to_numpy(), grid_races.loc[model_races.index, "log_loss"].to_numpy()
        ),
        "model_vs_pole_sitter_accuracy": paired_bootstrap(
            -model_races["hit"].to_numpy(), -pole_hit.reindex(model_races.index).to_numpy()
        ),
        "win_reliability": reliability_table(
            predictions, "race_probability", finish.eq(1), edges=WIN_EDGES
        ).to_dict("records"),
        "podium_reliability": reliability_table(
            predictions, "podium_probability", finish.le(3), edges=PODIUM_EDGES
        ).to_dict("records"),
    }
    previous_path = Path(args.previous_predictions)
    if previous_path.exists():
        previous = pd.read_parquet(previous_path)
        shared = sorted(set(previous["race_id"]) & set(predictions["race_id"]))
        previous = previous[previous["race_id"].isin(shared)].copy()
        previous["race_probability"] = previous["race_probability"].clip(lower=1e-4)
        current = predictions[predictions["race_id"].isin(shared)]
        previous_races = _per_race(previous, "race_probability")
        report["previous_model_same_races"] = {
            "races": len(shared),
            "previous": {
                "winner_accuracy": float(previous_races["hit"].mean()),
                "race_log_loss": float(previous_races["log_loss"].mean()),
                "tied_or_zero_probability_rows": int(
                    previous.duplicated(["race_id", "race_probability"], keep=False).sum()
                ),
            },
            "current": {
                "winner_accuracy": float(model_races.loc[shared, "hit"].mean()),
                "race_log_loss": float(model_races.loc[shared, "log_loss"].mean()),
                "tied_or_zero_probability_rows": int(
                    current.duplicated(["race_id", "race_probability"], keep=False).sum()
                ),
            },
            "log_loss_difference": paired_bootstrap(
                model_races.loc[shared, "log_loss"].to_numpy(),
                previous_races.loc[shared, "log_loss"].to_numpy(),
            ),
        }
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output / "race_predictions.parquet", index=False)
    shown = [
        "season", "race_id", "track", "driver", "team", "grid_position", "predicted_position",
        "finish_position", "race_probability", "podium_probability", "expected_position",
    ]
    predictions[[column for column in shown if column in predictions]].to_csv(
        output / "race_prediction_actual.csv", index=False, float_format="%.5f"
    )
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
