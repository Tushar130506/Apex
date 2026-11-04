#!/usr/bin/env python
"""Materialize official FastF1 race results for processed lap sessions."""
from __future__ import annotations

import argparse
import sys

import pandas as pd

sys.path.insert(0, "src")
from apex.ingestion.fastf1_loader import _enable_cache
from apex.models.points import points_for_position


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("years", type=int, nargs="+")
    parser.add_argument("--cache-dir", default=".fastf1-cache")
    parser.add_argument("--out", default="data/processed/race_results.parquet")
    parser.add_argument(
        "--allow-skips",
        action="store_true",
        help="Continue and write partial results when one or more rounds fail.",
    )
    args = parser.parse_args()
    import fastf1

    _enable_cache(args.cache_dir)
    rows = []
    failures = []
    for year in args.years:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        for _, event in schedule.iterrows():
            round_number = int(event["RoundNumber"])
            try:
                session = fastf1.get_session(year, round_number, "R")
                session.load(telemetry=False, weather=False, messages=False)
                for _, row in session.results.iterrows():
                    position = pd.to_numeric(row.get("Position"), errors="coerce")
                    grid = pd.to_numeric(row.get("GridPosition"), errors="coerce")
                    status = str(row.get("Status", ""))
                    disqualified = "disqual" in status.lower()
                    rows.append({
                        "race_id": f"{year}_{round_number}",
                        "season": year,
                        "track": str(event["EventName"]),
                        "driver": str(row["Abbreviation"]),
                        "finish_position": position,
                        "grid_position": grid,
                        "classified": status == "Finished" or pd.notna(row.get("ClassifiedPosition")),
                        "points": points_for_position(
                            position, classified=(status == "Finished" or pd.notna(row.get("ClassifiedPosition"))),
                            disqualified=disqualified,
                        ),
                        "derived_points": points_for_position(
                            position, classified=(status == "Finished" or pd.notna(row.get("ClassifiedPosition"))),
                            disqualified=disqualified,
                        ),
                        "event_type": "race",
                        "status": status,
                        "team": str(row.get("TeamName", row.get("Team", "UNKNOWN"))),
                    })
            except Exception as error:
                print(f"skip {year} round {round_number}: {error}")
                failures.append((year, round_number, str(error)))
    if failures and not args.allow_skips:
        details = "; ".join(
            f"{year} round {round_number}: {error}"
            for year, round_number, error in failures
        )
        raise RuntimeError(f"failed race-result rounds ({len(failures)}): {details}")
    result = pd.DataFrame(rows).drop_duplicates(["race_id", "driver"])
    if result.empty:
        raise RuntimeError("no race results were materialized")
    result.to_parquet(args.out, index=False)
    print(f"wrote {len(result)} driver results across {result.race_id.nunique()} races to {args.out}")


if __name__ == "__main__":
    main()
