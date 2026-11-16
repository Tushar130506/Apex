#!/usr/bin/env python
"""Forecast a race that has not been run yet.

Before qualifying only prior-race form is used. Pass ``--entries`` with a
starting grid (and optionally ``qualifying_best_lap_s``) for the
post-qualifying model.

  python scripts/predict_race.py --race-id 2026_21 --track "Las Vegas Grand Prix"
  python scripts/predict_race.py --race-id 2026_21 --track "Las Vegas Grand Prix" \
      --entries data/manual/2026_21_grid.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, "src")
from apex.models.forecast import add_upcoming_race, append_results, predict_race

SHOWN = [
    "predicted_position", "driver", "team", "grid_position", "race_probability",
    "podium_probability", "points_probability", "expected_position",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", default="runs/walk_forward_championship/input_features.parquet")
    parser.add_argument("--extra-results", nargs="*", default=[], help="CSV files of completed races missing from --features")
    parser.add_argument("--race-id", required=True, help="YYYY_round of the race to forecast")
    parser.add_argument("--track", default=None, help="Event name as used in the feature table")
    parser.add_argument("--entries", default=None, help="CSV with driver and optionally team, grid_position, qualifying_best_lap_s")
    parser.add_argument("--calibration-races", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    table = pd.read_parquet(args.features)
    for path in args.extra_results:
        table = append_results(table, pd.read_csv(path))
    entries = pd.read_csv(args.entries) if args.entries else None
    table = add_upcoming_race(table, args.race_id, args.track, entries)
    forecast = predict_race(
        table, args.race_id, calibration_races=args.calibration_races, seed=args.seed
    )
    forecast["team"] = forecast["team_canonical"]
    output = Path(args.out or f"runs/forecast_{args.race_id}")
    output.mkdir(parents=True, exist_ok=True)
    forecast.to_parquet(output / "race_forecast.parquet", index=False)
    forecast[SHOWN].to_csv(output / "race_forecast.csv", index=False, float_format="%.4f")
    summary = {
        "race_id": args.race_id,
        "track": args.track,
        "forecast_mode": str(forecast["forecast_mode"].iloc[0]),
        "trained_on_races": int(forecast["training_races"].iloc[0]),
        "trained_through_race": int(forecast["training_through_race"].iloc[0]),
        "win_temperature": float(forecast["temperature"].iloc[0]),
        "predicted_winner": str(forecast["driver"].iloc[0]),
        "predicted_winner_probability": float(forecast["race_probability"].iloc[0]),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(forecast[SHOWN].to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
