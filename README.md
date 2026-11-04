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
# Walk-forward: seed with half of a season, then add each completed race
# before predicting the next one.
python scripts/walk_forward_winner.py --season 2025 \
  --results data/processed/race_results_2018_2025_team.parquet
```

### Race, WDC, and constructor forecasts

The winner predictions can be converted into a championship forecast with a
chronological Plackett-Luce Monte Carlo simulation. Completed points are kept
fixed; each future race is sampled without replacement from its predicted
driver strengths, then official race points are accumulated for both drivers
and teams:

```bash
python scripts/forecast_championship.py \
  --completed data/processed/race_results_2026_team.parquet \
  --future-predictions runs/winner_walk_forward_2026_mexico/2026_predictions.parquet \
  --as-of-race 9 \
  --out runs/championship_forecast_2026_as_of_9
```

Use `--as-of-race` for historical replay validation. It prevents results after
the chosen timestamp from becoming current points and prevents earlier
predictions from being treated as future races. The output contains driver WDC
win/top-three probabilities, constructor win probabilities, and expected final
points. The prediction file itself must also have been generated using only
data available at that timestamp; filtering a later walk-forward artifact
does not undo look-ahead in how its later predictions were trained. A forecast
is only meaningful when predictions exist for every remaining race; the
current Mexico artifact covers rounds 10--19 only.

`apex.models.replay.replay_snapshots` provides the leakage-safe replay
boundary: at lap `t`, pit stops, tyre state, lap times, and safety-car events
are calculated only from rows with `lap_no <= t`. This is the foundation for
re-running the race Monte Carlo after each observed lap. FastF1 cache data is
historical; a live feed adapter is still required for real-time operation.

The walk-forward runner reads `laps_<year>.parquet` one season at a time,
converts it immediately to compact race-level features, and releases the raw
lap table before loading the next season. This is the recommended weekly
retraining path because it is reproducible and does not require loading the
entire historical cache into memory.

Result materialization fails if any round cannot be loaded. Use
`--allow-skips` only when a partial result file is explicitly acceptable.

Grid position is now included as a pre-race winner feature. In the repeated
holdout run it improved 2025 accuracy from 58.3% to 66.7%; this remains a
validation result, not a production guarantee.

Points are derived from classified finishing position rather than imported
fastest-lap flags or unverified source totals. Grand Prix points use the
top-ten 25-18-15-12-10-8-6-4-2-1 scale; sprint rows use 8-7-6-5-4-3-2-1;
disqualified/unclassified rows score zero; and no fastest-lap bonus is added.
Shortened-race scaling is supported through `distance_fraction` when that
field is available. Sprint results must be materialized as separate
`event_type="sprint"` rows before they can affect championship forecasts.

The reproducible baseline split is chronological and fixed before inspection:
2018-2023 train, 2024 validation, and 2025 test. This keeps the test season
untouched and gives the model multiple regulation-era examples while leaving
the latest completed season for final evaluation. The 2026 walk-forward run is
the separate future-era check.

Practice and qualifying are now represented by separate pace, best-lap, rank,
and gap features. The full labeled-data model is saved by the repeated holdout
runner under `runs/winner_repeated_sessions/`; future inference must provide
grid position from a pre-race source rather than importing finishing results.

## Caveats
- Fuel mass is an estimate (linear burn); FastF1 has no fuel data. Neutralised/pit/inaccurate laps are excluded with a stored reason.
- Driver, team, circuit, weather and traffic are confounders: read `mae_by_track` / `mae_by_compound` in `metrics.json`.
- Not yet run against the live FastF1 API in the build environment; run `season 2024 --max-events 3` first.
- Aero (`src/apex/aero`) needs your ANSYS results; `validate_cfd` rejects anything not marked `ansys_fluent`.
- Optional strategy API: `APEX_RUN=runs/pace_v1 uvicorn apex.api.app:app` uses the run's fitted pace model.

## Current race-intelligence modules

The race system is split by responsibility so each feature has one clear source:

| Module | Responsibility |
|---|---|
| `apex.features.fuel` | Fuel correction and uncertain fuel-burn sampling |
| `apex.features.tyre_degradation` | Compound-specific linear/quadratic degradation |
| `apex.models.race_monte_carlo` | Race-winner probabilities with fuel, tyres, and safety-car uncertainty |
| `apex.models.safety_hazard` | Historical lap-level safety-car hazard estimation |
| `apex.models.calibration` | Chronological temperature calibration |
| `apex.models.points` | Grand Prix, sprint, shortened-race, and disqualification scoring |
| `apex.models.overfit` | Train/validation/test overfit and underfit warnings |
| `apex.ingestion.live_timing` | Provider-neutral live event/state contract and replay |

The Monte Carlo race layer is validated as a probability layer, but it is
still a simplified race model: it does not yet include a real-time timing
provider, full strategy search, traffic, or incident-specific retirement
hazards. `live_timing.py` is therefore an integration boundary, not a claim
that a live feed is already connected. Historical sprint rows and reliable
race-distance fractions must also be materialized before they can affect
championship forecasts.
