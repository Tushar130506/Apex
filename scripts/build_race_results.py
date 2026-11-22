#!/usr/bin/env python
"""Materialize official FastF1 race or sprint results.

  python scripts/build_race_results.py 2024 2025 2026
  python scripts/build_race_results.py 2024 2025 2026 --event-type sprint \
      --out data/processed/sprint_results.parquet

Rounds that have not been run yet are skipped, and so, with a warning, is the
latest round when its classification is not published yet. Sprint results go to their own
file: they share the ``race_id`` of the Grand Prix held that weekend, so the
two must not be mixed in one table.
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

sys.path.insert(0, "src")
from apex.ingestion.race_results import SESSION_NAMES, completed_rounds, result_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("years", type=int, nargs="+")
    parser.add_argument("--event-type", choices=sorted(SESSION_NAMES), default="race")
    parser.add_argument("--cache-dir", default=".fastf1-cache")
    parser.add_argument("--out", default="data/processed/race_results.parquet")
    parser.add_argument(
        "--allow-skips",
        action="store_true",
        help="Continue and write partial results when one or more rounds fail.",
    )
    args = parser.parse_args()
    import fastf1
    from apex.ingestion.fastf1_loader import _enable_cache

    _enable_cache(args.cache_dir)
    rows = []
    failures = []
    newest = max(args.years)
    for year in args.years:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        rounds = completed_rounds(schedule, args.event_type)
        for round_number, track in rounds:
            try:
                session = fastf1.get_session(year, round_number, SESSION_NAMES[args.event_type])
                session.load(telemetry=False, weather=False, messages=False)
                found = result_rows(session.results, year, round_number, track, args.event_type)
                if not found:
                    raise RuntimeError("session has no classification")
                rows.extend(found)
            except Exception as error:  # noqa: BLE001 - collect every failed round
                if year == newest and round_number == rounds[-1][0]:
                    # The classification of the latest round can lag the race itself.
                    print(f"WARNING: {year} round {round_number} skipped, its results are not available yet: {error}")
                    continue
                print(f"skip {year} round {round_number}: {error}")
                failures.append((year, round_number, str(error)))
    if failures and not args.allow_skips:
        details = "; ".join(
            f"{year} round {round_number}: {error}"
            for year, round_number, error in failures
        )
        raise RuntimeError(f"failed {args.event_type} rounds ({len(failures)}): {details}")
    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError(f"no {args.event_type} results were materialized")
    result = result.drop_duplicates(["race_id", "driver"])
    result.to_parquet(args.out, index=False)
    print(f"wrote {len(result)} driver results across {result.race_id.nunique()} {args.event_type}s to {args.out}")


if __name__ == "__main__":
    main()
