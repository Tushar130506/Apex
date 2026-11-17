#!/usr/bin/env python
"""Forecast driver and constructor championships from race predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, "src")
from apex.models.championship import forecast_championship


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completed", required=True, help="Results known at the prediction timestamp")
    parser.add_argument("--future-predictions", required=True, help="Race-level predictions for remaining races")
    parser.add_argument("--out", default="runs/championship_forecast")
    parser.add_argument("--simulations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help="Filter completed and future races to this season before applying --as-of-race",
    )
    parser.add_argument(
        "--as-of-race",
        type=int,
        default=None,
        help="Use completed races through this round and predictions after it",
    )
    args = parser.parse_args()

    completed = pd.read_parquet(args.completed)
    future = pd.read_parquet(args.future_predictions)
    if args.season is not None:
        completed = completed[
            completed["race_id"].astype(str).str.startswith(f"{args.season}_")
        ].copy()
        future = future[
            future["race_id"].astype(str).str.startswith(f"{args.season}_")
        ].copy()
    if "team" in future and future["team"].astype(str).str.strip().isin({"", "None", "nan"}).any():
        team_lookup = (
            completed[["race_id", "driver", "team"]]
            .dropna(subset=["team"])
            .drop_duplicates(["race_id", "driver"])
            .set_index(["race_id", "driver"])["team"]
        )
        keys = pd.MultiIndex.from_frame(future[["race_id", "driver"]])
        missing_team = future["team"].astype(str).str.strip().isin({"", "None", "nan"})
        future["team"] = future["team"].mask(
            missing_team,
            keys.map(team_lookup),
        )
        driver_lookup = (
            completed.dropna(subset=["team"])
            .sort_values(["driver", "race_id"])
            .drop_duplicates("driver", keep="last")
            .set_index("driver")["team"]
        )
        missing_team = future["team"].isna() | future["team"].astype(str).str.strip().isin({"", "None", "nan"})
        future["team"] = future["team"].mask(
            missing_team, future["driver"].map(driver_lookup)
        )
        unresolved = future["team"].isna() | future["team"].astype(str).str.strip().isin({"", "None", "nan"})
        if unresolved.any():
            print(
                "warning: excluding future predictions without a resolvable team: "
                + ", ".join(sorted(future.loc[unresolved, "driver"].astype(str).unique()))
            )
            future = future.loc[~unresolved].copy()
    if args.as_of_race is not None:
        completed = completed[
            completed["race_id"].astype(str).str.extract(r"_(\d+)$")[0].astype(int)
            <= args.as_of_race
        ]
        future = future[
            future["race_id"].astype(str).str.extract(r"_(\d+)$")[0].astype(int)
            > args.as_of_race
        ]
    forecast = forecast_championship(
        completed,
        future,
        simulations=args.simulations,
        seed=args.seed,
    )
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    forecast.driver.to_parquet(output / "driver_forecast.parquet", index=False)
    forecast.constructor.to_parquet(output / "constructor_forecast.parquet", index=False)
    summary = {
        "simulations": forecast.simulations,
        "completed_races": forecast.completed_races,
        "projected_races": forecast.projected_races,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("\nDriver championship:\n", forecast.driver.head(10).to_string(index=False))
    print("\nConstructor championship:\n", forecast.constructor.to_string(index=False))


if __name__ == "__main__":
    main()
