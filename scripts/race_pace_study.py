#!/usr/bin/env python
"""Measure race-progress (fuel + track) and tyre degradation for every race.

  python scripts/race_pace_study.py --seasons 2024 2025 2026

Uses official lap times and pit stops from the Jolpica API (no account). The
first run downloads about 13 pages per race at the service's rate limit
(roughly 500 requests an hour) and caches them in data/raw/jolpica, so
later runs are instant. Writes runs/race_pace/races.csv (one row per race)
and runs/race_pace/circuits.csv (median per circuit).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, "src")
from apex.features.race_pace import clean_race_laps, fit_race_pace
from apex.ingestion.jolpica import race_laps, races


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seasons", type=int, nargs="+", required=True)
    parser.add_argument("--out", default="runs/race_pace")
    parser.add_argument("--bootstrap", type=int, default=200)
    args = parser.parse_args()
    rows, skipped = [], []
    for season in args.seasons:
        for race in races(f"{season}"):
            round_number = int(race["round"])
            race_id = f"{season}_{round_number}"
            try:
                laps, pits = race_laps(season, round_number)
                cleaned = clean_race_laps(laps, pits)
                result = fit_race_pace(cleaned, bootstrap=args.bootstrap)
            except ValueError as error:
                skipped.append({"race_id": race_id, "reason": str(error)})
                continue
            excluded = cleaned["exclusion_reason"].value_counts().to_dict()
            rows.append({
                "race_id": race_id, "season": season, "track": race["raceName"],
                "circuit": race["Circuit"]["circuitId"],
                "clean_laps": result["clean_laps"], "drivers": result["drivers"],
                "progress_s_per_lap": result["progress_s_per_lap"],
                "progress_low": result["progress_ci95"][0], "progress_high": result["progress_ci95"][1],
                "deg_s_per_lap": result["deg_s_per_lap"], "deg_quad_s_per_lap2": result["deg_quad_s_per_lap2"],
                "deg_at_20_laps_s": result["deg_at_20_laps_s"],
                "deg20_low": result["deg_at_20_laps_ci95"][0], "deg20_high": result["deg_at_20_laps_ci95"][1],
                "neutralised_laps": int(excluded.get("neutralised", 0)),
                "pit_stops": int(len(pits)),
            })
            print(f"{race_id:8s} {race['raceName']:28s} progress {result['progress_s_per_lap']:+.3f} s/lap  "
                  f"deg@20 {result['deg_at_20_laps_s']:+.2f} s", flush=True)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(output / "races.csv", index=False, float_format="%.4f")
    if not table.empty:
        circuits = table.groupby("circuit").agg(
            track=("track", "last"), races=("race_id", "count"),
            progress_s_per_lap=("progress_s_per_lap", "median"),
            deg_s_per_lap=("deg_s_per_lap", "median"),
            deg_quad_s_per_lap2=("deg_quad_s_per_lap2", "median"),
            deg_at_20_laps_s=("deg_at_20_laps_s", "median"),
            pit_stops_per_race=("pit_stops", "mean"),
        ).sort_values("deg_at_20_laps_s", ascending=False)
        circuits.to_csv(output / "circuits.csv", float_format="%.4f")
    (output / "skipped.json").write_text(json.dumps(skipped, indent=2), encoding="utf-8")
    print(f"\n{len(table)} races measured, {len(skipped)} skipped -> {output}")


if __name__ == "__main__":
    main()
