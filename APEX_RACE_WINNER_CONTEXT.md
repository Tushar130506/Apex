# APEX v0.2 race-winner prediction context

## Goal

Build a race-level prediction system that ranks every driver and returns a winner
probability, rather than treating individual laps as independent winner labels.
The first v0.2 slice is the feature foundation: tyre degradation and fuel-adjusted
pace must be calculated before a winner model is trained.

## Current scope

- Keep the existing lap-level pace model unchanged.
- Add reusable race-level aggregation helpers.
- Estimate degradation from clean, non-pit laps using robust per-stint slopes.
- Calculate fuel-adjusted lap pace using an explicit sensitivity parameter.
- Preserve provenance and uncertainty flags because FastF1 does not provide direct
  fuel measurements.
- Support either a supplied `race_id` column or `session_id` as a safe fallback
  grouping key. Do not silently combine unrelated sessions.
- Use the official-result, winner-feature, and race-session workflow implemented
  by `scripts/build_race_results.py` and `scripts/train_winner.py`.

## Holdout and leakage rules

1. Split by complete season and never by individual laps.
2. Any feature used before a race must be computed from data available before that
   race. Never use finishing position, final race time, or post-race pit outcomes.
3. A feature based on the current stint may use earlier laps in that stint only
   for live prediction. Full-stint degradation is retrospective and is for
   training/analysis, not pre-race scoring.
4. The untouched holdout season is read once after feature and model choices are
   frozen.
5. Results must include sample counts and fuel provenance so weak evidence is
   visible.

## v0.2 feature contract

One row represents one driver in one race/session grouping:

- `race_id`, `season`, `track`, `driver`
- `valid_laps`, `stints`
- `median_lap_time_s`, `best_lap_time_s`
- `fuel_corrected_median_lap_s`
- `fuel_sensitivity_s_per_kg`
- `degradation_s_per_lap`, `degradation_stints`
- `estimated_fuel_fraction`
- `feature_quality`

`feature_quality` is `usable` when there are enough clean laps for a descriptive
summary, otherwise `sparse`. It is not a prediction confidence score.

## Planned next slices

1. Add forecast weather and a full strategy-aware race simulator.
2. Evaluate race-level ranking/probability model with race-by-race walk-forward validation.
3. Add Monte Carlo strategy simulation and dashboard explanations.

The first official-result path is now implemented:

- `scripts/build_race_results.py` materializes `race_results.parquet` from cached
  FastF1 sessions.
- `apex.features.winner_features` parses race/session identity and builds only
  pre-race pace features, with separate practice and qualifying pace summaries.
- `apex.models.winner.train_winner_model` evaluates winner ranking by complete
  seasons, with train/validation/test kept disjoint.
- `scripts/walk_forward_winner.py` seeds a season with its first half, predicts
  each later race, and adds each completed race before the next prediction.

The first real-data run exposed a generalization problem: the initial model
achieved 0% validation accuracy and 8.3% holdout accuracy. Race-relative pace
gaps and rank features were added to reduce circuit-specific overfitting. The
re-run achieved 41.7% validation accuracy and 66.7% holdout accuracy across
24 races in 2025. Because the feature change was made after inspecting the
2025 holdout, this result is exploratory and tuned against the test period,
not an untouched holdout. It is not a production claim; forecast weather,
strategy simulation, and uncertainty calibration are still missing.

## Known limitations

- Fuel mass is estimated and the fuel sensitivity is an assumption, not ground truth.
- Full-stint degradation cannot be used as a pre-race feature without leakage.
- The current winner model now includes prior-race driver/team form, smoothed
  reliability, circuit history, practice weather summaries, and
  race-consistent probabilities. Forecast weather, strategy simulation, and
  uncertainty calibration remain future improvements. Grid position is
  included as a pre-race feature derived from official qualifying results.

Adding grid position improved the 2025 repeated-holdout fold from 58.3% to
66.7% winner accuracy. This is a validation-period result, not a guarantee of
future performance; team form, reliability, weather, and calibration remain
the next upgrades. The latest implementation uses refreshed cached race
results with team and status fields; all historical features exclude the
target race.

The session-aware rerun reached 75.0% on the 2025 repeated-holdout fold. The
final artifact `runs/winner_repeated_sessions/model_all_labeled_seasons.joblib`
is trained on all labeled seasons currently available (2018-2026); it should
only be used for future races when pre-race grid data is supplied separately
from post-race finishing results.

## Championship and replay layer

The repository now includes `apex.models.championship.forecast_championship`
for Monte Carlo WDC and constructor forecasts. It samples each future race
using a without-replacement Plackett-Luce order, applies the official
top-ten race points, and accumulates driver and team totals. Completed results
and future predictions are separate inputs so future outcomes cannot alter the
current standings.

The prediction-generation timestamp is a separate contract from an
as-of-race filter: filtering predictions generated later cannot remove
look-ahead from their training data. For valid historical championship
evaluation, generate every remaining-race prediction from the same
information cutoff, or replay the season race by race.

`apex.models.replay.replay_snapshots` creates one observable state per lap.
Pit-stop counts, tyre state, lap times, and safety-car counts at lap `t` use
only observations through lap `t`. This supports historical replay tests for
live-race updating. It does not itself provide a live data connection, and
FastF1 fuel remains estimated rather than measured.
