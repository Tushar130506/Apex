# 2026 walk-forward winner validation through Mexico

## Protocol

- Training starts with the first nine races of 2026 plus all prior seasons.
- Race 10 is predicted from prior races only.
- After each completed race, its labeled result becomes eligible for the next
  race's training set.
- The target race's finishing result, race laps, pit stops, incidents, and
  safety-car status are excluded from its features.
- Raw laps are loaded one season at a time and reduced to compact race-level
  features before the next season is read.
- Mexico is race `2026_19` in the cached result schedule used by this run.

Command:

```powershell
python scripts\walk_forward_winner.py --season 2026 --half-races 9 `
  --results data\processed\race_results_2018_2025_team.parquet `
  --future-results data\processed\race_results_2026_team.parquet `
  --out runs\winner_walk_forward_2026_mexico
```

## Features used

| Group | Features | Timing / leakage rule |
|---|---|---|
| Pace | Fuel-corrected median, best lap, ranks, gaps | Practice and qualifying only |
| Tyres | Robust degradation slope, tyre age, compound-derived lap filters | Pre-race sessions only |
| Grid | Grid position, rank, gap to pole | Available after qualifying |
| Practice | Median/best pace, lap/session counts, availability | Practice only |
| Qualifying | Median/best pace, lap/session counts, availability | Qualifying only |
| Environment | Practice track temperature, air temperature, wet/neutralized fraction | Pre-race sessions only |
| Driver state | Recent finish, form, smoothed finish/DNF rates | Prior races only |
| Team state | Recent finish, form, smoothed finish/DNF rates | Prior races only |
| Circuit history | Driver/team prior track finish and starts | Prior races at the track only |
| Safety context | Prior same-track safety-car and neutralized-lap rates | Earlier race sessions only |
| Ranking | Within-race normalized win probability | Derived from model score; sums to 1 per race |

Safety-car flags use FastF1 `TrackStatus` codes already present in the cached
race laps. The target race's own `TrackStatus` values are never used.

## Aggregate results

| Metric | Result |
|---|---:|
| Scored races | 10 |
| Winner accuracy | 70.0% |
| Mean reciprocal rank | 0.808 |
| Mean top-three overlap | 70.0% |
| Mean race log loss | 0.860 |
| Mean race Brier score | 0.422 |
| Mean calibration error | 0.039 |

The simple baselines scored as follows on the same ten races:

| Predictor | Winner accuracy |
|---|---:|
| Walk-forward model | 70.0% |
| Best grid position | 70.0% |
| Best qualifying pace rank | 40.0% |
| Best prior driver form | 20.0% |

The model therefore matched the grid baseline in winner accuracy rather than
beating it. Its additional value in this sample is ranking quality: it reached
0.808 mean reciprocal rank and 70.0% top-three overlap while incorporating
practice, tyre, environmental, historical, reliability, team, and safety
context.

The two safety-context columns were present for all 220 scored driver rows
(100% coverage). An ablation with the same walk-forward splits and seed should
be used when comparing this integration; correlation with actual race safety
events is not evidence that target-race safety information was available.

### Safety-feature ablation

| Metric | With safety context | Without safety context |
|---|---:|---:|
| Winner accuracy | 70.0% | 70.0% |
| Mean reciprocal rank | 0.808 | 0.808 |
| Mean top-three overlap | 70.0% | 66.7% |
| Mean race log loss | 0.860 | 0.877 |
| Mean race Brier score | 0.422 | 0.416 |
| Mean calibration error | 0.0390 | 0.0391 |

The safety features improved mean top-three overlap and log loss in this
10-race sample, but did not change winner accuracy. Their Mexico prediction
was still wrong (HAM predicted, VER actual); the safety-aware model assigned
VER 7.4% win probability versus 5.3% without the safety columns. This is
useful evidence of integration, not proof of a statistically significant
accuracy improvement.

Driver-row confusion matrix for winner membership:

| Actual \ Predicted | Not winner | Winner |
|---|---:|---:|
| Not winner | 207 | 3 |
| Winner | 3 | 7 |

The row-level accuracy is 97.3%, but it is not the primary measure because
each race has one winner and many non-winners. The race-level winner accuracy
of 70.0% is the meaningful headline result.

## Race-by-race results

| Race | Predicted winner | Actual winner | Winner hit | Top-three overlap |
|---|---|---|---:|---:|
| 2026_10 | ANT | ANT | Yes | 2/3 |
| 2026_11 | NOR | NOR | Yes | 1/3 |
| 2026_12 | NOR | NOR | Yes | 3/3 |
| 2026_13 | LEC | ANT | No | 1/3 |
| 2026_14 | NOR | ANT | No | 2/3 |
| 2026_15 | RUS | RUS | Yes | 2/3 |
| 2026_16 | VER | VER | Yes | 2/3 |
| 2026_17 | VER | VER | Yes | 3/3 |
| 2026_18 | ANT | ANT | Yes | 3/3 |
| 2026_19 Mexico | HAM | VER | No | 2/3 |

The complete driver-level table is stored in
`runs/winner_walk_forward_2026_mexico/2026_predictions.parquet`, and the
machine-readable metrics are in `2026_metrics.json`.

## Mexico result

| Predicted position | Driver | Actual finish | Win probability |
|---:|---|---:|---:|
| 1 | HAM | 2 | 53.1% |
| 2 | RUS | 6 | 31.3% |
| 3 | VER | 1 | 7.4% |
| 4 | LEC | 4 | 3.6% |
| 5 | ANT | 3 | 3.2% |

Mexico received these leakage-safe historical safety features:

```text
prior_track_safety_car_lap_rate = 0.052736
prior_track_neutralized_lap_rate = 0.053171
```

These are historical circuit rates, not Mexico's observed race conditions.

## Probability status

The model now produces mutually exclusive within-race probabilities and
reports log loss, Brier score, and calibration error. It is not yet a fully
calibrated probability model:

1. The primary race probability is the winner frequency from the integrated
   Monte Carlo race simulation, which samples adjusted race outcomes after
   tyre, fuel, and safety-car uncertainty. The classifier softmax remains the
   unadjusted diagnostic baseline.
2. The exploratory online calibration layer is fitted only from earlier
   walk-forward races, but it is unstable with this small rolling sample:
   its mean race log loss was 2.267 versus 0.860 for the uncalibrated
   normalized probabilities. It is therefore reported for diagnosis, not used
   as the primary probability output.
3. P2/P3 are ranking positions, not separately calibrated finish-position
   probabilities.
4. Strategy-aware Monte Carlo is available as a separate strategy analysis;
   the winner model still uses aggregate tyre, fuel, and safety-car adjustments
   rather than simulating every pit-window decision.
5. Fuel mass remains estimated, not measured.
6. The current cache has observed session weather, not a historical archive of
   pre-race weather forecasts.

The mean walk-forward calibration error of 0.039 is a useful baseline, but the
model should not be marketed as production-calibrated until a more stable
chronological calibration method and untouched calibration reports are added.

## Reproducibility artifacts

- Predictions: `runs/winner_walk_forward_2026_mexico/2026_predictions.parquet`
- Metrics: `runs/winner_walk_forward_2026_mexico/2026_metrics.json`
- Feature construction: `src/apex/features/winner_features.py`
- Walk-forward engine: `src/apex/models/walk_forward.py`
- CLI: `scripts/walk_forward_winner.py`
