#!/usr/bin/env python
"""Write official results of races missing from the feature table as CSV files.

  python scripts/export_extra_results.py --results data/processed/race_results_2026.parquet

Lap timing needs an F1TV account, official results do not. When a race has
results but no processed laps, this writes ``data/manual/<race_id>.csv`` in the
format ``predict_race.py`` and ``forecast_season.py`` take as
``--extra-results``. Such races feed prior-race form only; their practice pace
stays empty, and so does qualifying pace unless ``--with-qualifying`` adds it.
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
    """Rows of ``results`` for races that are absent from ``table``."""
    absent = set(COLUMNS) - set(results.columns)
    if absent:
        raise ValueError(f"results missing columns: {sorted(absent)}")
    rows = results[~results["race_id"].isin(set(table["race_id"]))].copy()
    rows = rows.iloc[race_order(rows["race_id"]).argsort(kind="stable")]
    rows["team"] = canonical_team(rows["team"])
    return rows[COLUMNS]


def add_qualifying(rows: pd.DataFrame) -> pd.DataFrame:
    """Join each race's qualifying best laps, so its pace counts in later form."""
    from apex.ingestion.jolpica import qualifying

    frames = []
    for race_id in rows["race_id"].unique():
        season, round_number = (int(part) for part in str(race_id).split("_"))
        frames.append(qualifying(season, round_number))
    times = pd.concat(frames, ignore_index=True)
    if times.empty:
        return rows.assign(qualifying_best_lap_s=float("nan"))
    return rows.merge(
        times[["race_id", "driver", "qualifying_best_lap_s"]], on=["race_id", "driver"], how="left",
        validate="one_to_one",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", required=True, help="parquet from scripts/build_race_results.py")
    parser.add_argument("--features", default="runs/walk_forward_championship/input_features.parquet")
    parser.add_argument("--out", default="data/manual")
    parser.add_argument("--with-qualifying", action="store_true",
                        help="add qualifying_best_lap_s from the Jolpica API (needs network)")
    args = parser.parse_args()
    rows = missing_races(pd.read_parquet(args.results), pd.read_parquet(args.features))
    if args.with_qualifying and not rows.empty:
        rows = add_qualifying(rows)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    for race_id, group in rows.groupby("race_id", sort=False):
        path = output / f"{race_id}.csv"
        group.to_csv(path, index=False)
        print(f"wrote {path} ({len(group)} drivers)")
    if rows.empty:
        print("every race with results is already in the feature table")


if __name__ == "__main__":
    main()
