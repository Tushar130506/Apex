# APEX - F1 race prediction, strategy and aero from FastF1 data

APEX has four parts:

| Part | What it does | State |
|---|---|---|
| Race model | Predicts finishing order, win and podium probabilities, and both championships. | Evaluated on 133 races; see `RACE_MODEL_REPORT.md`. |
| Lap-pace model | Predicts the next lap time in a stint, with 90% intervals. | Works; needs imported lap data. |
| Strategy simulator | Compares pit-stop strategies for one car by Monte Carlo. | Simplified: no traffic, default assumptions. |
| Aero surrogate | Fits drag and downforce from CFD runs and estimates lap-time effect. | Code only; no CFD results in the repo yet. |

All race data comes from [FastF1](https://docs.fastf1.dev/). The repository
ships no lap data; you import it yourself. `tests/fixtures.py` holds fake laps
for unit tests only, and training refuses them.

## Install and test

```bash
pip install -e ".[ml,fastf1,app,dashboard,dev]"
```

```bash
pytest
```

CI runs the same tests on every push (`.github/workflows/ci.yml`).

## Race model

A Plackett-Luce finish-order model: 61.7% winner accuracy over 133
walk-forward races against 56.4% for picking the pole-sitter, with calibrated
win probabilities. The accuracy edge is not statistically established; the
gain in probability quality is. Results, protocol and limits are in
`RACE_MODEL_REPORT.md`.

The scripts below read `runs/walk_forward_championship/input_features.parquet`,
a race-level feature table that is committed (2018 to 2026 round 19), so they
run without downloading anything.

```bash
# Walk-forward evaluation against baselines (2021 onward)
python scripts/evaluate_race_model.py
```

```bash
# Title-race checkpoints before every race
python scripts/walk_forward_championship.py
```

```bash
# Forecast a race that has not been run (add --entries grid.csv after qualifying)
python scripts/predict_race.py --race-id 2026_21 --track "Las Vegas Grand Prix"
```

```bash
# Remaining races and both championships
python scripts/forecast_season.py --season 2026 \
  --remaining "2026_21=Las Vegas Grand Prix" "2026_22=Qatar Grand Prix"
```

`predict_race.py` and `forecast_season.py` accept `--extra-results` for
official results of races missing from the feature table
(`race_id, track, driver, team, grid_position, finish_position, status`).

### Rebuilding the feature table

This needs network access to the FastF1 data servers.

```bash
# 1. Laps for every session, one file per season -> data/processed/laps_<year>.parquet
python download_fastf1.py --years 2018-2026 --sessions all
python materialize_fastf1.py
```

```bash
# 2. Official race results
python scripts/build_race_results.py 2018 2019 2020 2021 2022 2023 2024 2025 \
  --out data/processed/race_results_2018_2025_team.parquet
python scripts/build_race_results.py 2026 --out data/processed/race_results_2026_team.parquet
```

```bash
# 3. Race-level features -> <out>/input_features.parquet
python scripts/walk_forward_winner.py --season 2026 --half-races 9 \
  --results data/processed/race_results_2018_2025_team.parquet \
  --future-results data/processed/race_results_2026_team.parquet \
  --out runs/walk_forward_championship
```

`walk_forward_winner.py` reads one season of laps at a time and reduces it to
race-level rows before loading the next, so the full history never has to fit
in memory. `build_race_results.py` fails if any round cannot be loaded; pass
`--allow-skips` only when a partial file is acceptable.

## Lap-pace model

```bash
# 1. Import race laps, one file per season -> data/processed/laps_<year>.parquet
python scripts/build_dataset.py season 2022 2023 2024 2025
```

```bash
# 2. Train, calibrate and report (years from configs/split.json or flags)
python scripts/train.py --run-name pace_v1
```

```bash
# 3. Score laps with a saved run
python scripts/predict.py --run runs/pace_v1 --years 2025 \
  --out data/predictions/pace_v1/scored_2025.parquet
```

- **Split by season:** train seasons fit the model, validation seasons
  calibrate the intervals and are where models are compared, and test seasons
  are read once after choices are frozen. Default: train 2022-23, validation
  2024, test 2025.
- **2026 is a new regulation era:** scoring it sets
  `season_seen_in_training=False`; treat those results as out-of-domain.
- **One step ahead:** each lap is predicted from earlier clean laps in its
  stint. Baselines are the previous lap and a rolling mean.

Try `python scripts/build_dataset.py season 2024 --max-events 3` first; the
first import is slow. `build_dataset.py cache <years> --sessions all
--telemetry` fills the FastF1 cache with every session (telemetry is large and
not yet used by any model).

## Dashboard and API

```bash
streamlit run streamlit_dashboard.py
```

The dashboard reads stored predictions and forecasts from `runs/`, and lap and
result files from `data/processed/` when they exist.

```bash
APEX_RUN=runs/pace_v1 uvicorn apex.api.app:app
```

The optional strategy API uses the pace model fitted by that run.

## Where things are

| Path | What |
|---|---|
| `src/apex/` | The library (modules below) |
| `scripts/` | Command-line entry points for the current pipeline |
| `scripts/legacy/` | Scripts for the earlier winner classifier; see its README |
| `configs/split.json` | Train, validation and test seasons for the lap-pace model |
| `data/processed/` | Imported laps and race results (not committed) |
| `runs/<name>/` | Stored models, metrics, predictions and forecasts |
| `.fastf1-cache/` | FastF1 download cache (not committed) |

| Module | Responsibility |
|---|---|
| `apex.ingestion.fastf1_loader` | FastF1 sessions to the lap contract in `apex.schemas.contracts` |
| `apex.features.form_features` | Race-relative weekend features and prior-race form |
| `apex.features.winner_features` | Per-race rows from laps: practice and qualifying pace, results join |
| `apex.models.plackett_luce` | Exploded-logit loss, linear and boosted models, temperature fit |
| `apex.models.race_order` | Blended race model, walk-forward loop, metrics |
| `apex.models.forecast` | Unrun-race forecasts and season outlook |
| `apex.models.championship` | Title simulation with countback and sprint rounds |
| `apex.models.points` | Grand Prix, sprint, shortened-race and disqualification scoring |
| `apex.models.pace`, `apex.features.lap_features` | Lap-pace GBM, conformal intervals, lap features |
| `apex.features.fuel` | Fuel correction and fuel-burn sampling |
| `apex.features.tyre_degradation`, `apex.models.degradation` | Degradation by compound and by stint |
| `apex.strategy` | Parametric pace model and pit-strategy simulator |
| `apex.models.race_monte_carlo`, `apex.models.safety_hazard` | Lap-level race simulation and safety-car hazard |
| `apex.models.replay` | Leakage-safe lap-by-lap replay of a finished race |
| `apex.ingestion.live_timing` | Live event contract; no feed is connected |
| `apex.aero`, `apex.models.aero_surrogate` | CFD run sheets, result checks, surrogate, lap-time estimate |
| `apex.models.winner` | Earlier winner classifier, kept as a baseline |

## Aero

`apex.aero.doe.run_sheet` writes the list of CFD cases to run. After solving,
fill in the coefficients and convergence columns; `validate_cfd` accepts only
rows marked `openfoam` (the default), `ansys_fluent` or `star_ccm`, one solver
per table. The lap-time estimate uses a generic car on a made-up circuit, so
it compares designs under the same assumptions and is not a prediction for a
real car. Candidates proposed by the surrogate must be re-run in CFD.

## Limits

- **Winner accuracy:** the race model does not reliably beat the pole-sitter
  on the top pick (2025: worse, 2026: equal).
- **Missing data:** no sprint results, no pre-race weather forecasts, and the
  feature table ends at 2026 round 19. Championship points are race points
  only.
- **Fuel is estimated:** FastF1 has no fuel data, so fuel mass is a linear
  burn from an assumed start load, and the fuel sensitivity is an assumption.
- **Points:** derived from classified finishing position on the 25-18-15-12-10-8-6-4-2-1
  scale, with no fastest-lap bonus. Sprint rows must be materialized as
  `event_type="sprint"` before they count.
- **Confounders in lap pace:** driver, team, circuit, weather and traffic;
  read `mae_by_track` and `mae_by_compound` in a run's `metrics.json`.
- **Strategy and race simulation:** no traffic, no full strategy search, no
  incident-specific retirement hazards, and no live timing feed.

## Other documents

| File | Content |
|---|---|
| `RACE_MODEL_REPORT.md` | Current race model: design, results, calibration, forecasts |
| `scripts/legacy/README.md` | What the legacy scripts were and what replaced them |
| `WINNER_WALK_FORWARD_2026_MEXICO_REPORT.md` | Earlier classifier's 2026 results (history) |
| `WINNER_MODEL_FEATURE_SPEC.md`, `APEX_RACE_WINNER_CONTEXT.md` | Earlier feature spec and leakage rules |
