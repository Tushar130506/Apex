#!/usr/bin/env python
"""Forecast every remaining race of a season and both championships.

  python scripts/forecast_season.py --season 2026 \
      --remaining "2026_21=Las Vegas Grand Prix" "2026_22=Qatar Grand Prix" \
                  "2026_23=Abu Dhabi Grand Prix"

Remaining races are forecast from prior-race form only, because their
practice and qualifying sessions have not happened. Add ``--sprint-rounds``
for remaining rounds that also hold a sprint.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, "src")
from apex.models.championship import describe_title_chances
from apex.models.forecast import append_results, season_outlook
from apex.models.plackett_luce import win_probabilities


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", default="runs/walk_forward_championship/input_features.parquet")
    parser.add_argument("--extra-results", nargs="*", default=[])
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--remaining", nargs="+", required=True, help='"YYYY_round=Event name" for each race still to run')
    parser.add_argument("--sprint-rounds", nargs="*", default=[], help="race ids among --remaining that hold a sprint")
    parser.add_argument("--sprint-results", default=None, help="parquet or CSV of completed sprint results (scripts/build_race_results.py --event-type sprint)")
    parser.add_argument("--driver", nargs="*", default=[], help="driver codes to describe, e.g. VER NOR")
    parser.add_argument("--team", nargs="*", default=[], help="team names to describe, e.g. Ferrari")
    parser.add_argument("--simulations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    table = pd.read_parquet(args.features)
    for path in args.extra_results:
        table = append_results(table, pd.read_csv(path))
    remaining = pd.DataFrame(
        [dict(zip(("race_id", "track"), item.split("=", 1))) for item in args.remaining]
    )
    sprints = None
    if args.sprint_results:
        reader = pd.read_csv if args.sprint_results.endswith(".csv") else pd.read_parquet
        sprints = reader(args.sprint_results)
    outlook = season_outlook(
        table,
        args.season,
        remaining,
        sprint_race_ids=args.sprint_rounds,
        sprint_results=sprints,
        simulations=args.simulations,
        seed=args.seed,
    )
    races = outlook.races.copy()
    races["race_probability"] = win_probabilities(races["race_id"], races["strength"].to_numpy())
    output = Path(args.out or f"runs/season_forecast_{args.season}")
    output.mkdir(parents=True, exist_ok=True)
    races.to_parquet(output / "remaining_race_strengths.parquet", index=False)
    outlook.championship.driver.to_parquet(output / "driver_forecast.parquet", index=False)
    outlook.championship.constructor.to_parquet(output / "constructor_forecast.parquet", index=False)
    outlook.championship.driver.to_csv(output / "driver_forecast.csv", index=False, float_format="%.4f")
    outlook.championship.constructor.to_csv(output / "constructor_forecast.csv", index=False, float_format="%.4f")
    summary = {
        "season": args.season,
        "simulations": outlook.championship.simulations,
        "completed_races": outlook.championship.completed_races,
        "projected_races": outlook.championship.projected_races,
        "projected_sprints": outlook.championship.projected_sprints,
        "form_temperature": outlook.temperature,
        "form_drift_per_race": outlook.form_drift_per_race,
        "remaining": remaining.to_dict("records"),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    fmt = lambda value: f"{value:.3f}"  # noqa: E731
    print("\nDriver championship:\n", outlook.championship.driver.head(10).to_string(index=False, float_format=fmt))
    print("\nConstructor championship:\n", outlook.championship.constructor.to_string(index=False, float_format=fmt))
    top = races.sort_values(["race_id", "race_probability"], ascending=[True, False]).groupby("race_id").head(5)
    print("\nRemaining races (form-only win probability):\n", top[["race_id", "track", "driver", "team", "race_probability"]].to_string(index=False, float_format=fmt))
    for name in args.driver:
        print("\n" + describe_title_chances(outlook.championship, name))
    for name in args.team:
        print("\n" + describe_title_chances(outlook.championship, name, constructor=True))


if __name__ == "__main__":
    main()
