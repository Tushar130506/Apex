#!/usr/bin/env python
"""Walk-forward test of extra race-model features against the current model.

  python scripts/compare_features.py --set temperature
  python scripts/compare_features.py --columns track_temp_z --first-season 2021 --last-season 2025
  python scripts/compare_features.py --set race_pace   # needs runs/race_pace/teams.csv
  python scripts/compare_features.py --set layout      # needs runs/track_layout/ (track_layout_study.py)

Both models are refitted before every race on earlier races only, exactly as
in ``evaluate_race_model.py``. Choose features on 2021-2025 and look at 2026
once, after the choice is made. The verdict uses the paired bootstrap interval
of the per-race log-loss difference: a feature set earns its place only when
the whole interval is below zero.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from apex.features.form_features import (
    ORDER_MODEL_COLUMNS,
    RACE_PACE_COLUMNS,
    TEMPERATURE_COLUMNS,
    add_race_pace_form,
    build_order_features,
    race_order,
)
from apex.features.track_layout import LAYOUT_MODEL_COLUMNS, add_layout_features
from apex.models.race_order import paired_bootstrap, walk_forward_order
from apex.models.walk_forward import MIN_TRAINING_RACES

FEATURE_SETS = {"temperature": TEMPERATURE_COLUMNS, "race_pace": RACE_PACE_COLUMNS, "layout": LAYOUT_MODEL_COLUMNS}


def per_race(predictions: pd.DataFrame) -> pd.DataFrame:
    """Winner log loss and top-pick hit for every race."""
    finish = pd.to_numeric(predictions["finish_position"], errors="coerce")
    rank = predictions.groupby("race_id")["race_probability"].rank(ascending=False, method="first")
    winners = predictions[finish.eq(1)]
    return pd.DataFrame(
        {
            "log_loss": -np.log(winners["race_probability"].clip(lower=1e-12)).to_numpy(),
            "hit": rank[finish.eq(1)].eq(1).astype(float).to_numpy(),
        },
        index=winners["race_id"].to_numpy(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", default="runs/walk_forward_championship/input_features.parquet")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--set", choices=sorted(FEATURE_SETS))
    group.add_argument("--columns", nargs="+")
    parser.add_argument("--first-season", type=int, default=2021)
    parser.add_argument("--last-season", type=int, default=2025)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layout-dir", default="runs/track_layout",
                        help="layouts.csv and race_circuits.csv from track_layout_study.py (for the layout set)")
    parser.add_argument("--race-pace", default="runs/race_pace/teams.csv",
                        help="team race-pace table from race_pace_study.py (for the race_pace set)")
    parser.add_argument("--out", default=None, help="optional JSON file for the result")
    args = parser.parse_args()

    extra = list(FEATURE_SETS[args.set]) if args.set else list(args.columns)
    features = build_order_features(pd.read_parquet(args.features))
    if set(extra) & set(RACE_PACE_COLUMNS):
        features = add_race_pace_form(features, pd.read_csv(args.race_pace))
    if set(extra) & set(LAYOUT_MODEL_COLUMNS):
        layout_dir = Path(args.layout_dir)
        features = add_layout_features(
            features, pd.read_csv(layout_dir / "race_circuits.csv"), pd.read_csv(layout_dir / "layouts.csv")
        )
    unknown = set(extra) - set(features.columns)
    if unknown:
        parser.error(f"unknown columns: {sorted(unknown)}")
    orders = sorted(race_order(features["race_id"]).unique())
    targets = [value for value in orders if args.first_season <= value // 100 <= args.last_season]
    warmup = [
        value for position, value in enumerate(orders)
        if position >= MIN_TRAINING_RACES and value < targets[0]
    ]
    races = {}
    for name, columns in (("current", ORDER_MODEL_COLUMNS), ("with_extra", ORDER_MODEL_COLUMNS + extra)):
        scored = walk_forward_order(features, warmup + targets, columns=list(columns), seed=args.seed)
        races[name] = per_race(scored[race_order(scored["race_id"]).isin(targets)])
    current = races["current"]
    candidate = races["with_extra"].loc[current.index]
    difference = paired_bootstrap(candidate["log_loss"].to_numpy(), current["log_loss"].to_numpy())
    season = pd.Series(current.index.str[:4].astype(int), index=current.index)
    report = {
        "extra_columns": extra,
        "seasons": [args.first_season, args.last_season],
        "races": len(current),
        "current": {"log_loss": float(current["log_loss"].mean()), "winner_accuracy": float(current["hit"].mean())},
        "with_extra": {"log_loss": float(candidate["log_loss"].mean()), "winner_accuracy": float(candidate["hit"].mean())},
        "log_loss_difference": difference,
        "by_season_log_loss": {
            int(year): {"current": float(current.loc[ids, "log_loss"].mean()),
                        "with_extra": float(candidate.loc[ids, "log_loss"].mean())}
            for year, ids in season.groupby(season).groups.items()
        },
        "verdict": "helps" if difference["ci95_high"] < 0 else "no measurable gain",
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
