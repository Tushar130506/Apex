#!/usr/bin/env python
"""Run race-by-race predictions and WDC probabilities across seasons."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, "src")
from apex.features.winner_features import MODEL_COLUMNS
from apex.models.season_walk_forward import expanding_season_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--out", default="runs/walk_forward_championship")
    parser.add_argument("--simulations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data = pd.read_parquet(args.features)
    predictions, checkpoints = expanding_season_predictions(
        data,
        seed_seasons=list(range(2018, 2025)),
        forecast_seasons=[2025, 2026],
        model_columns=MODEL_COLUMNS,
        championship_simulations=args.simulations,
        seed=args.seed,
    )
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output / "race_predictions.parquet", index=False)
    checkpoints.to_parquet(output / "wdc_checkpoints.parquet", index=False)
    print(
        {
            "predicted_races": int(predictions["race_id"].nunique()),
            "prediction_rows": len(predictions),
            "wdc_checkpoints": int(checkpoints["after_race_id"].nunique()),
        }
    )


if __name__ == "__main__":
    main()
