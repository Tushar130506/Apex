# APEX - FastF1 race-pace ML + strategy + aero surrogate

**Pure FastF1.** No game telemetry, no generated datasets, no dashboard. The repo ships with *no data*;
you import it with FastF1. (`tests/fixtures.py` holds fake laps used only by unit tests; training refuses them.)

```bash
pip install -e ".[ml,fastf1,app,dev]"
pytest

# Browse all processed seasons with filters and CSV export
pip install -e ".[dashboard]"
python materialize_fastf1.py
streamlit run streamlit_dashboard.py

# 1. import real data, one file per season  -> data/processed/laps_<year>.parquet
python scripts/build_dataset.py season 2022 2023 2024 2025
# Cache every FastF1 session (resumable; telemetry is optional and large)
python scripts/build_dataset.py cache 2018 2019 2020 2021 2022 2023 2024 2025 --sessions all --telemetry
# 2. train / validate / test year-wise (years in configs/split.json or via flags)
python scripts/train.py --run-name pace_v1
# 3. score laps with a saved run (e.g. a newer season) and store predictions
python scripts/predict.py --run runs/pace_v1 --years 2025 --out data/predictions/pace_v1/scored_2025.parquet
```

## Where things go
| Path | What |
|---|---|
| `data/processed/laps_<year>.parquet` | cleaned FastF1 laps per season (contract in `schemas/contracts.py`) |
| `configs/split.json` | which seasons are train / validation / test |
| `runs/<name>/` | `model.joblib` (GBM + conformal + fitted pace model), `metrics.json`, `config.json` |
| `data/predictions/<name>/predictions.parquet` | every scored lap labelled `train`/`val`/`test`: actual, baselines, GBM, 90% interval, in_interval, error |
| `.fastf1-cache/` | FastF1 download cache (first import is slow) |

## How the year-wise protocol works
- **train** seasons fit the GBM and the strategy pace model. **val** seasons calibrate the conformal intervals and
  are where you compare models/features. **test** seasons are reported alongside but must be read once, after choices are frozen.
- Default: train 2022-23, val 2024, test 2025 (same ground-effect era). 2026 is a new regulation era: scoring it will
  set `season_seen_in_training=False` - treat results as out-of-domain.
- Targets are one-step-ahead: each lap is predicted from earlier clean laps in its stint. Baselines: previous lap and rolling mean.

## v0.2 race-winner foundation

`APEX_RACE_WINNER_CONTEXT.md` defines the holdout-safe plan for race-winner
prediction. The first implementation slice is available through
`apex.features.race_features.build_race_features`: it aggregates clean laps into
driver/session rows, calculates fuel-adjusted pace with an explicit sensitivity
assumption, and carries robust stint degradation estimates plus fuel provenance.

This is intentionally not a winner model yet. Official race results, qualifying
or grid position, and verified race-session identity must be joined before
training a race-level ranking model. Full-stint degradation is retrospective and
must not be used as a pre-race feature; use earlier-session degradation for
pre-race scoring and current-stint history only for live prediction.

To materialize official results from the FastF1 cache:

```bash
python scripts/build_race_results.py 2022 2023 2024 2025
python scripts/train_winner.py --train-years 2022 2023 --val-years 2024 --test-years 2025
python scripts/repeated_holdout_winner.py
```

Result materialization fails if any round cannot be loaded. Use
`--allow-skips` only when a partial result file is explicitly acceptable.

## Caveats
- Fuel mass is an estimate (linear burn); FastF1 has no fuel data. Neutralised/pit/inaccurate laps are excluded with a stored reason.
- Driver, team, circuit, weather and traffic are confounders: read `mae_by_track` / `mae_by_compound` in `metrics.json`.
- Not yet run against the live FastF1 API in the build environment; run `season 2024 --max-events 3` first.
- Aero (`src/apex/aero`) needs your ANSYS results; `validate_cfd` rejects anything not marked `ansys_fluent`.
- Optional strategy API: `APEX_RUN=runs/pace_v1 uvicorn apex.api.app:app` uses the run's fitted pace model.
