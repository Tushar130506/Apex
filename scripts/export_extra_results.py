#!/usr/bin/env python
"""Write official results of races missing from the feature table as CSV files.

  python scripts/export_extra_results.py --results data/processed/race_results_2026.parquet

Lap timing needs an F1TV account, official results do not. When a race has
results but no processed laps, this writes ``data/manual/<race_id>.csv`` in the
format ``predict_race.py`` and ``forecast_season.py`` take as
``--extra-results``. Such races feed prior-race form only; their practice and
qualifying pace columns stay empty.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, "src")
from apex.features.form_features import canonical_team, race_order

COLUMNS = ["race_id", "track", "driver", "team", "grid_position", "finish_position", "status"]


def missing_races(results: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Rows of ``results`` for races later than everything in ``table``."""
    absent = set(COLUMNS) - set(results.columns)
    if absent:
        raise ValueError(f"results missing columns: {sorted(absent)}")
    last = race_order(table["race_id"]).max()
    rows = results[race_order(results["race_id"]).gt(last)].copy()
    rows["team"] = canonical_team(rows["team"])
    return rows[COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", required=True, help="parquet from scripts/build_race_results.py")
    parser.add_argument("--features", default="runs/walk_forward_championship/input_features.parquet")
    parser.add_argument("--out", default="data/manual")
    args = parser.parse_args()
    rows = missing_races(pd.read_parquet(args.results), pd.read_parquet(args.features))
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    for race_id, group in rows.groupby("race_id", sort=False):
        path = output / f"{race_id}.csv"
        group.to_csv(path, index=False)
        print(f"wrote {path} ({len(group)} drivers)")
    if rows.empty:
        print("no races newer than the feature table")


if __name__ == "__main__":
    main()
