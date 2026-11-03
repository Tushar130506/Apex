#!/usr/bin/env python
"""Train and evaluate the chronological race-winner baseline.

Example:
  python scripts/train_winner.py --train-years 2022 2023 --val-years 2024 --test-years 2025
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
from apex.features.winner_features import attach_race_results, build_pre_race_features, validate_race_targets
from apex.models.winner import train_winner_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-years", type=int, nargs="+", required=True)
    parser.add_argument("--val-years", type=int, nargs="+", required=True)
    parser.add_argument("--test-years", type=int, nargs="+", required=True)
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--results", default="data/processed/race_results.parquet")
    parser.add_argument("--out", default="runs/winner_v1")
    args = parser.parse_args()
    years = args.train_years + args.val_years + args.test_years
    laps = load_processed(args.data, years)
    features = build_pre_race_features(laps)
    results = pd.read_parquet(args.results)
    data = attach_race_results(features, results)
    validate_race_targets(data)
    metrics, model = train_winner_model(data, args.train_years, args.val_years, args.test_years)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output / "model.joblib")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    data.to_parquet(output / "features.parquet", index=False)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
