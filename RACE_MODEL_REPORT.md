# Race model report (finish-order model, November 2026)

This supersedes `WINNER_WALK_FORWARD_2026_MEXICO_REPORT.md`, which describes
the earlier winner classifier and is kept only for history.

## What changed

| Earlier caveat | Status | What was done |
|---|---|---|
| Model only matched "pick the pole-sitter" | Improved, not eliminated | New model is 61.7% vs 56.4% over 133 races. On 2026 alone it still ties pole (13/19 each). See results. |
| Monte Carlo layer produced coarse, tied and zero probabilities | Fixed | Win probability is an exact softmax of fitted strengths. 254 tied/zero rows in the old predictions for 2023-2026, 0 now. |
| Predicted positions among tied drivers depended on row order | Fixed | Strict order; level strengths are separated by grid slot, then recent form. |
| Calibration was unstable (log loss 2.27 calibrated vs 0.86 raw) | Fixed | Continuous temperature fit on earlier out-of-sample races, shrunk towards 1. |
| P2/P3 were ranks, not probabilities | Fixed | Full finishing-position distribution per driver: podium, points, expected position. |
| Reliability features were constant | Fixed | The cached `classified` flag is true on every row, retirements included. DNF history is now read from the status text. |
| Team history broke at every rename | Fixed | Toro Rosso/AlphaTauri/RB/Racing Bulls and similar lineages share one history; unlabeled rows are filled from the driver's nearest race. |
| No way to predict a race without its result row | Fixed | `scripts/predict_race.py` forecasts a race that has not been run, before or after qualifying. |
| Future-circuit forecast copied last year's practice and qualifying pace into the target | Fixed | Unrun races use prior-race form only. |
| Championship ties were split equally | Fixed | Countback: most wins, then seconds, and so on. |
| Championship treated classifier scores as log-strengths, making future races nearly uniform | Fixed | Remaining races use calibrated strengths plus a form-drift term for long horizons. |
| No sprint points in forecasts | Fixed | `--sprint-rounds` simulates future sprints; `--sprint-results` adds completed ones. |
| Evaluation on 10 races | Fixed | 133 races (2021-2026), every one predicted from earlier races only, with bootstrap intervals. |

Not fixable from this repository: fuel load is estimated, there is no archive
of pre-race weather forecasts, and no live timing feed. Sprint results are
used for championship points only, not as model features.

## Model

One row per driver per race. The model is a blend of a regularised linear
Plackett-Luce model and gradient-boosted trees trained on the same
exploded-logit loss (winner weight 1, second 0.5, third 0.25). It learns from
the finishing order rather than from one winner label per race.

Features (36, `apex.features.form_features`):

- **Weekend, pre-race:** grid slot, qualifying gap to pole in percent, best-lap
  and practice-pace gaps, relative tyre degradation, team's best qualifying
  gap, grid versus the driver's usual grid, wet-practice fraction, prior
  safety-car rate at the circuit.
- **Form, prior races only:** fast (3-race) and slow (10-race) half-life
  averages of points, finishing position, wins, podiums, grid, qualifying gap,
  places gained and retirements for driver and team; qualifying gap to
  team-mate; season points share; shrunk circuit history.

Absolute lap times are no longer features: they describe the circuit, not the
driver.

## Protocol

- The model design was chosen on 2021-2025 only. 2026 was scored once, after
  the design was frozen. One later change was made after seeing 2021-2026: a
  separate temperature for places behind the winner. It does not affect any
  winner metric below.
- Before every race the model is refitted on all earlier races. Races are
  weighted with a 60-race half-life.
- Temperatures are fitted only on earlier out-of-sample forecasts.
- Tests assert that changing a race's result changes none of its features.

Reproduce: `python scripts/evaluate_race_model.py`

## Results: 133 races, 2021-2026

| Metric | Model | Pole-sitter / grid-only |
|---|---:|---:|
| Winner accuracy | 61.7% | 56.4% |
| Winner in model's top three | 91.7% | - |
| Mean reciprocal rank of winner | 0.761 | - |
| Top-three overlap | 68.4% | - |
| Race log loss | 1.135 | 1.483 |

Grid-only is a one-feature Plackett-Luce model on grid position, refitted
before each race like the main model.

- Log loss versus grid-only: -0.349 per race, 95% interval -0.524 to -0.176.
  This difference is solid.
- Accuracy versus pole-sitter: +5.3 points, 95% interval -2.3 to +12.8. This
  is **not** statistically established.

| Season | Races | Model accuracy | Pole-sitter | Model log loss | Grid-only log loss |
|---|---:|---:|---:|---:|---:|
| 2021 | 22 | 59.1% | 50.0% | 1.474 | 1.659 |
| 2022 | 22 | 50.0% | 45.5% | 1.239 | 1.805 |
| 2023 | 22 | 86.4% | 63.6% | 0.394 | 1.399 |
| 2024 | 24 | 50.0% | 45.8% | 1.598 | 1.644 |
| 2025 | 24 | 58.3% | 66.7% | 1.011 | 1.122 |
| 2026 (scored once) | 19 | 68.4% | 68.4% | 1.049 | 1.258 |

The model loses to the pole-sitter in 2025 and ties it in 2026. Its advantage
in those seasons is in the probabilities, not in the top pick.

Against the earlier classifier on the 89 races both predicted (2023-2026):
65.2% vs 60.7% accuracy, log loss 1.025 vs 1.105. The log-loss interval
(-0.229 to +0.056) includes zero.

### Calibration

Win probability:

| Predicted band | Rows | Mean predicted | Observed |
|---|---:|---:|---:|
| 0-2% | 2047 | 0.3% | 0.1% |
| 2-5% | 222 | 3.3% | 2.3% |
| 5-10% | 117 | 7.3% | 7.7% |
| 10-20% | 98 | 13.9% | 15.3% |
| 20-35% | 87 | 27.1% | 23.0% |
| 35-50% | 47 | 42.5% | 55.3% |
| 50-70% | 45 | 59.8% | 55.6% |
| 70-100% | 33 | 83.6% | 90.9% |

Podium probability is less well calibrated: drivers given 85-100% reached the
podium 83.9% of the time (112 rows), and those given 15-50% did so more often
than predicted. An explicit retirement component was tested on 2021-2025 and
did not help, so it was not added.

### Championship checkpoints

`python scripts/walk_forward_championship.py` forecasts the title before every
race of 2023-2026 (89 checkpoints) using only earlier races. The eventual
champion was the favourite at 82% of checkpoints, with a mean probability of
79% (69% with 19+ races left, 91% with six or fewer). Four seasons is a small
sample, three of them with a dominant champion.

## Forecasts

Updated 22 November 2026. The feature table ends at round 19 (Mexico City).
Rounds 20 (São Paulo) and 21 (Las Vegas) are added from their official
classifications through `--extra-results`, so they feed prior-race form but
have no practice or qualifying pace. Lap timing now needs an F1TV account, so
no laps were processed for them.

How the pre-weekend (form-only) forecasts did:

| Round | Forecast top three | Result |
|---|---|---|
| 20, São Paulo | VER 35.8%, ANT 25.2%, LEC 10.7% (NOR 5.6%) | NOR, VER, HAD |
| 21, Las Vegas | VER 30.1%, ANT 22.0%, NOR 12.2% | VER, NOR, PIA |

Round 22, Qatar (pre-weekend, form only, data through round 21):

| | 1st | 2nd | 3rd | 4th | 5th |
|---|---|---|---|---|---|
| Win probability | VER 46.3% | ANT 13.9% | NOR 12.5% | RUS 7.3% | LEC 6.2% |

Championship after round 21, sprints included (the six 2026 sprints are in
`data/manual/sprint_results_2026.csv`), two races left and no sprints
remaining:

| Drivers | Points | | Constructors | Points |
|---|---:|---|---|---:|
| Antonelli | 402 | | Mercedes | 666 |
| Verstappen | 304 | | Ferrari | 517 |
| Hamilton | 267 | | Red Bull Racing | 440 |
| Norris | 264 | | McLaren | 431 |
| Russell | 264 | | Racing Bulls | 114 |
| Leclerc | 250 | | Alpine | 87 |

Both titles are settled (`title_status` clinched for Antonelli and Mercedes).
Verstappen finishes second in every simulation; third place is open between
Norris (35%), Russell (31%) and Hamilton (29%). Points are derived from
classified positions, so check them against the official standings before
quoting them.

Files: `runs/forecast_2026_22/`, `runs/season_forecast_2026/`. The stored
`runs/forecast_2026_20/` and `runs/forecast_2026_21/` are the pre-race
forecasts for those rounds.

To refresh after a race, or after qualifying:

```bash
python scripts/build_race_results.py 2026 --out data/processed/race_results_2026.parquet
python scripts/export_extra_results.py --results data/processed/race_results_2026.parquet
python scripts/predict_race.py --race-id 2026_22 --track "Qatar Grand Prix" \
    --extra-results data/manual/2026_20.csv data/manual/2026_21.csv \
    --entries data/manual/2026_22_grid.csv
```

`export_extra_results.py` writes one CSV per race that has official results
but is missing from the feature table
(`race_id, track, driver, team, grid_position, finish_position, status`).
`--entries` takes the starting grid and, optionally, qualifying best laps.
