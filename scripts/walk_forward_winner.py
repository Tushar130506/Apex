#!/usr/bin/env python
"""Race-by-race walk-forward winner evaluation.

The first half of the selected season seeds training. Each later race is
predicted using rows from earlier races only; after its result is available,
that race becomes part of the next training set.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, "src")
from apex.data_io import load_processed_season
from apex.features.winner_features import (
    attach_prior_safety_features,
    attach_race_results,
    build_race_safety_summary,
    build_pre_race_features,
)
from apex.models.walk_forward import walk_forward_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--results", default="data/processed/race_results_2018_2025_team.parquet")
    parser.add_argument("--future-results", default=None)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--half-races", type=int, default=None)
    parser.add_argument("--out", default="runs/winner_walk_forward")
    args = parser.parse_args()

    results = pd.read_parquet(args.results)
    if args.future_results:
        results = pd.concat([results, pd.read_parquet(args.future_results)], ignore_index=True)
    season_features = []
    safety_summaries = []
    for year in range(2018, args.season + 1):
        laps = load_processed_season(args.data, year)
        season_features.append(build_pre_race_features(laps))
        safety_summaries.append(build_race_safety_summary(laps))
        del laps
    features = pd.concat(season_features, ignore_index=True)
    features = attach_prior_safety_features(
        features, pd.concat(safety_summaries, ignore_index=True)
    )
    del season_features
    data = attach_race_results(features, results)
    predictions, metrics = walk_forward_predictions(
        data, season=args.season, half_races=args.half_races
    )
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output / f"{args.season}_predictions.parquet", index=False)
    (output / f"{args.season}_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps({"season": args.season, "races": len(metrics), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
