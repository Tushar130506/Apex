#!/usr/bin/env python
"""Resumable FastF1 downloader with rate-limit recovery and verification.

Examples:
    python download_fastf1.py --years 2018-2024 --sessions all
    python download_fastf1.py --years 2018-2024 --sessions all --telemetry

The manifest is updated after every session, so stopping this process and
running it again resumes from the first session that is not verified.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd

import fastf1
from fastf1.exceptions import RateLimitExceededError

from src.apex.ingestion.fastf1_loader import _enable_cache


DEFAULT_YEARS = list(range(2018, 2025))
RATE_LIMIT_WAIT_SECONDS = 60 * 60 + 30


def parse_years(value: str) -> list[int]:
    match = re.fullmatch(r"(\d{4})(?:-(\d{4}))?", value.strip())
    if not match:
        raise argparse.ArgumentTypeError("years must look like 2018-2024 or 2024")
    start = int(match.group(1))
    end = int(match.group(2) or start)
    if end < start:
        raise argparse.ArgumentTypeError("year range must be ascending")
    return list(range(start, end + 1))


def session_jobs(years: list[int], sessions: str) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for year in years:
        while True:
            try:
                schedule = fastf1.get_event_schedule(year, include_testing=False)
                break
            except RateLimitExceededError as error:
                print(f"RATE LIMIT reached while reading {year} ({error}); waiting {RATE_LIMIT_WAIT_SECONDS} seconds...")
                time.sleep(RATE_LIMIT_WAIT_SECONDS)
        for _, event in schedule.iterrows():
            names = ["R"]
            if sessions == "all":
                names = [
                    str(event[column])
                    for column in schedule.columns
                    if re.fullmatch(r"Session\d+", column)
                    and pd.notna(event[column])
                    and str(event[column]).strip()
                ]
                names = list(dict.fromkeys(names))
            for name in names:
                jobs.append(
                    {
                        "year": year,
                        "event": str(event["EventName"]),
                        "session": name,
                    }
                )
    return jobs


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    return {str(item["key"]): item for item in data}


def save_reports(
    manifest: dict[str, dict[str, Any]], json_path: Path, csv_path: Path
) -> None:
    rows = list(manifest.values())
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    fields = ["key", "year", "event", "session", "status", "laps", "error"]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def download(
    years: list[int],
    cache_dir: str,
    sessions: str,
    telemetry: bool,
    manifest_path: Path,
    report_path: Path,
) -> None:
    _enable_cache(cache_dir)
    manifest = load_manifest(manifest_path)
    jobs = session_jobs(years, sessions)
    print(f"Found {len(jobs)} sessions; verified {sum(x.get('status') == 'verified' for x in manifest.values())}.")

    for index, job in enumerate(jobs, 1):
        key = f"{job['year']}|{job['event']}|{job['session']}"
        old = manifest.get(key, {})
        if old.get("status") == "verified":
            print(f"[{index}/{len(jobs)}] already verified: {key}")
            continue

        while True:
            try:
                session = fastf1.get_session(job["year"], job["event"], job["session"])
                session.load(telemetry=telemetry, weather=True, messages=True)
                laps = getattr(session, "laps", None)
                if laps is None or laps.empty:
                    raise RuntimeError("session loaded without lap timing data")
                row = {**job, "key": key, "status": "verified", "laps": len(laps), "error": ""}
                manifest[key] = row
                save_reports(manifest, manifest_path, report_path)
                print(f"[{index}/{len(jobs)}] VERIFIED {key}: {len(laps)} laps")
                break
            except RateLimitExceededError as error:
                print(f"RATE LIMIT reached ({error}); waiting {RATE_LIMIT_WAIT_SECONDS} seconds...")
                time.sleep(RATE_LIMIT_WAIT_SECONDS)
            except Exception as error:  # noqa: BLE001 - report and continue to next session
                row = {**job, "key": key, "status": "failed", "laps": 0, "error": str(error)}
                manifest[key] = row
                save_reports(manifest, manifest_path, report_path)
                print(f"[{index}/{len(jobs)}] FAILED {key}: {error}")
                break

    verified = sum(row.get("status") == "verified" for row in manifest.values())
    failed = sum(row.get("status") == "failed" for row in manifest.values())
    print(f"COMPLETE: {verified}/{len(jobs)} verified; {failed} failed.")
    print(f"Reports: {manifest_path} and {report_path}")
    if failed:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=parse_years, default=DEFAULT_YEARS)
    parser.add_argument("--sessions", choices=("race-only", "all"), default="all")
    parser.add_argument("--telemetry", action="store_true")
    parser.add_argument("--cache-dir", default=".fastf1-cache")
    parser.add_argument("--manifest", type=Path, default=Path("fastf1_download_report.json"))
    parser.add_argument("--report", type=Path, default=Path("fastf1_download_report.csv"))
    args = parser.parse_args()
    download(
        args.years,
        args.cache_dir,
        args.sessions,
        args.telemetry,
        args.manifest,
        args.report,
    )


if __name__ == "__main__":
    main()
