#!/usr/bin/env python
"""Convert verified FastF1 cache sessions into analysis-ready parquet files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import fastf1
import pandas as pd

from src.apex.ingestion.fastf1_loader import _enable_cache, laps_to_contract


def materialize(
    report_path: Path,
    output_dir: Path,
    cache_dir: str,
    include_failed: bool = False,
) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    _enable_cache(cache_dir)
    by_year: dict[int, list[pd.DataFrame]] = {}
    failures: list[dict[str, str]] = []

    for index, item in enumerate(report, 1):
        if item.get("status") != "verified" and not include_failed:
            continue
        year = int(item["year"])
        try:
            session = fastf1.get_session(year, item["event"], item["session"])
            session.load(telemetry=False, weather=True, messages=True)
            laps = session.laps
            if laps is None or laps.empty:
                raise RuntimeError("no lap timing data")
            total_laps = max(1, int(laps["LapNumber"].max()))
            frame = laps_to_contract(
                laps,
                session.weather_data,
                f"fastf1_{year}_{session.event['RoundNumber']}_{item['session']}",
                str(session.event["EventName"]),
                total_laps,
                year,
            )
            by_year.setdefault(year, []).append(frame)
            print(f"[{index}/{len(report)}] loaded {year} {item['event']} {item['session']}")
        except Exception as error:  # noqa: BLE001 - report individual failures
            failure = {**item, "error": str(error)}
            failures.append(failure)
            print(f"[{index}/{len(report)}] failed {item['key']}: {error}")

    for year, frames in by_year.items():
        combined = pd.concat(frames, ignore_index=True)
        combined = combined.drop_duplicates(["session_id", "driver", "lap_no"])
        combined.to_parquet(output_dir / f"laps_{year}.parquet", index=False)
        print(f"WROTE {year}: {len(combined):,} laps")

    Path("fastf1_materialize_failures.json").write_text(
        json.dumps(failures, indent=2), encoding="utf-8"
    )
    print(f"COMPLETE: {sum(len(frames) for frames in by_year.values())} sessions materialized; "
          f"{len(failures)} failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path("fastf1_download_report.json"))
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    parser.add_argument("--cache-dir", default=".fastf1-cache")
    args = parser.parse_args()
    materialize(args.report, args.output, args.cache_dir)


if __name__ == "__main__":
    main()
