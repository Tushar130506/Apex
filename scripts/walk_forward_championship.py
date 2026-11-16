#!/usr/bin/env python
"""Run race-by-race predictions and WDC checkpoints across seasons."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from apex.models.season_walk_forward import expanding_season_predictions


def checkpoint_scores(checkpoints: pd.DataFrame, data: pd.DataFrame) -> dict:
    """Score title probabilities against the champion of each finished season.

    The champion is whoever leads the points derived from the table's own
    results, so a season still in progress is skipped.
    """
    rows = []
    for season, group in checkpoints.groupby("season"):
        season_rows = data[data["season"].eq(season)]
        if group["before_race_id"].nunique() < season_rows["race_id"].nunique():
            continue
        champion = season_rows.groupby("driver")["points"].sum().idxmax()
        for race_id, board in group.groupby("before_race_id", sort=False):
            probability = float(board.loc[board["driver"].eq(champion), "win_probability"].sum())
            rows.append(
                {
                    "season": int(season),
                    "before_race_id": race_id,
                    "remaining_races": int(board["remaining_races"].iloc[0]),
                    "champion": champion,
                    "champion_probability": probability,
                    "favourite": board.sort_values("win_probability", ascending=False)["driver"].iloc[0],
                }
            )
    table = pd.DataFrame(rows)
    if table.empty:
        return {"checkpoints": 0}
    return {
        "checkpoints": len(table),
        "seasons": sorted(table["season"].unique().tolist()),
        "champion_was_favourite": float((table["favourite"] == table["champion"]).mean()),
        "mean_champion_probability": float(table["champion_probability"].mean()),
        "champion_log_loss": float(-np.log(table["champion_probability"].clip(lower=1e-4)).mean()),
        "by_phase": {
            phase: {
                "checkpoints": int(len(rows_)),
                "champion_was_favourite": float((rows_["favourite"] == rows_["champion"]).mean()),
                "mean_champion_probability": float(rows_["champion_probability"].mean()),
            }
            for phase, rows_ in table.groupby(
                pd.cut(table["remaining_races"], [0, 6, 12, 18, 30], labels=["1-6 left", "7-12 left", "13-18 left", "19+ left"]),
                observed=True,
            )
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default="runs/walk_forward_championship/input_features.parquet")
    parser.add_argument("--out", default="runs/race_order_championship_walk_forward")
    parser.add_argument("--seed-through", type=int, default=2022, help="last season used only for training")
    parser.add_argument("--simulations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data = pd.read_parquet(args.features)
    seasons = sorted(data["season"].unique().tolist())
    predictions, checkpoints = expanding_season_predictions(
        data,
        seed_seasons=[season for season in seasons if season <= args.seed_through],
        forecast_seasons=[season for season in seasons if season > args.seed_through],
        championship_simulations=args.simulations,
        seed=args.seed,
    )
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output / "race_predictions.parquet", index=False)
    checkpoints.to_parquet(output / "wdc_checkpoints.parquet", index=False)
    scores = checkpoint_scores(checkpoints, data)
    (output / "wdc_checkpoint_scores.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "predicted_races": int(predictions["race_id"].nunique()),
                "wdc_checkpoints": int(checkpoints["before_race_id"].nunique()),
                "wdc_checkpoint_scores": scores,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
