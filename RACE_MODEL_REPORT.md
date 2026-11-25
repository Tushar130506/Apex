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

## Temperature features (tested, not adopted)

Four features were tested on 2021-2025 (114 races), with 2026 left untouched:
the practice track temperature (`track_temp_z`), relative tyre degradation
times heat (`degradation_x_heat`), and how each driver and each team has done
against their usual form in hot or cold races before (`driver_heat_effect`,
`team_heat_effect`, prior races only, shrunk towards zero).

| | Current model | With temperature |
|---|---:|---:|
| Winner log loss | 1.149 | 1.149 |
| Winner accuracy | 60.5% | 58.8% |

Log-loss difference +0.0001, 95% interval
-0.023 to +0.023. Tested one
at a time, no feature moved log loss by more than 0.003 in either direction.
They are therefore not in the model; 2026 was not scored with them.

What this does and does not show: weekend temperature adds nothing that grid,
qualifying pace and form do not already carry, at the level of who wins. It
does not show that temperature is irrelevant to lap time or tyre wear; that
needs lap data and belongs to the degradation work. The features stay in
`apex.features.form_features.TEMPERATURE_COLUMNS` and the test reproduces
with `python scripts/compare_features.py --set temperature`.

## Fuel, track and tyre wear from race lap times

Measured for 134 races (2021 to 2026 round 21; Belgium 2021 had no racing
laps) from official lap times and pit stops (`scripts/race_pace_study.py`,
Jolpica API, no account). Per race, clean laps are fitted with one pace per
driver, a term for each lap run (fuel burn and the track gaining grip, which
cannot be told apart here) and a tyre-age curve; tyre age resets at each stop
while the lap count keeps rising, which separates the two. Pit, first,
safety-car and traffic laps are dropped. Output: `runs/race_pace/races.csv`,
`circuits.csv` and `teams.csv`.

- **Fuel and track:** laps get 0.057 s faster per lap (median of the 126 dry
  races; 10th-90th percentile 0.036 to 0.084). The strategy model's built-in
  assumption, 0.032 s/kg at 1.6 kg/lap = 0.051 s/lap, is consistent.
- **Tyre wear:** a 20-lap-old tyre is 0.9 s slower than a new one (median),
  highest at Suzuka and Bahrain (2.4 s) and Barcelona (1.7 s), lowest at
  Monaco and Jeddah (0.2 s or less). The data has no compounds, so this is an
  average over the tyres used.
- **Wet or drying races** are flagged when the field's lap times spread more
  than 6% (dry races: 1.5 to 4.9%) and left out of the circuit table: Imola
  2021, Monaco 2022, Singapore 2022, Canada 2024, Australia 2025, Silverstone
  2025 and Brazil 2026, plus Monaco 2024 (dry; its red-flag restart and slow
  tyre-saving laps spread the times).
- **Wear that looks negative** (14 of 126 dry races, among them Canada and
  Las Vegas 2026) means laps got faster with tyre age beyond the lap-count
  term: graining clearing or the track rubbering in, not wear. The strategy
  model turns such a curve into no wear rather than tyres improving.
- **Qatar** varies too much to rely on: four races range from -1.2 s (2024,
  many stops and safety cars) to +0.7 s, and 2025 had a 25-lap stint limit.

**Out-of-sample check** (`python scripts/benchmark_race_pace.py`, 95 dry
races at circuits measured before, each predicted from earlier races only):

| Prediction from | Progress error (s/lap) | 20-lap tyre loss error (s) |
|---|---:|---:|
| Median of all earlier races | 0.015 | 0.65 |
| Median of earlier races at the same circuit | 0.011 | 0.72 |
| Last race at the same circuit | 0.012 | 0.72 |
| Strategy model's built-in default | 0.015 | 0.75 |

The circuit's own history predicts the fuel-and-track gain better (difference
-0.004 s/lap, 95% interval -0.007 to -0.001). It does **not** predict tyre
wear better than one figure for every track (+0.06 s, interval -0.08 to
+0.21): a race's measured wear swings by about 0.7 s from year to year, more
than circuits differ, because compounds, temperature and rules change. The
built-in default wear (1.5 s at 20 laps) is too high against the 0.9 s
median. So `ParametricPace.for_circuit` is right to use the circuit's
progress term, but its per-circuit wear should be read as one noisy sample.

Each team's race pace and tyre wear against the field are also measured
(`team_race_pace`). 2026 so far, mean gap to the median team at a 12-lap-old
tyre: Mercedes -1.6%, Ferrari -1.4%, McLaren -1.2%, Red Bull -1.1%, the
midfield within 0.7%, Aston Martin +1.8% and Cadillac +2.0%. Team wear
differences are small (within 0.01 s/lap for all but Cadillac).

**As race-model features (tested, not adopted).** Each team's race pace and
wear in earlier races (`RACE_PACE_COLUMNS`, exponentially weighted, wet races
skipped) were tested on 2022-2025 (92 races) with 2026 left untouched:

| | Current model | With race pace |
|---|---:|---:|
| Winner log loss | 1.071 | 1.072 |
| Winner accuracy | 60.9% | 60.9% |

Log-loss difference +0.0006, 95% interval -0.010 to +0.011: no gain. Grid,
qualifying pace and points form already carry what a team's recent race pace
says about who wins. Reproduce with `python scripts/compare_features.py --set
race_pace --first-season 2022 --last-season 2025`.

## Forecasts

Updated 22 November 2026. The feature table ends at round 19 (Mexico City).
Rounds 20 (São Paulo) and 21 (Las Vegas) are added from their official
classifications and qualifying times (Jolpica API) through `--extra-results`,
so they feed prior-race form and qualifying pace but no practice pace. Lap
timing from FastF1 now needs an F1TV account, so no laps were processed.

How the pre-weekend (form-only) forecasts did:

| Round | Forecast top three | Result |
|---|---|---|
| 20, São Paulo | VER 35.8%, ANT 25.2%, LEC 10.7% (NOR 5.6%) | NOR, VER, HAD |
| 21, Las Vegas | VER 30.1%, ANT 22.0%, NOR 12.2% | VER, NOR, PIA |

Replayed after qualifying (actual grid and qualifying times, data through
round 20), the Las Vegas forecast becomes VER 70.8%, RUS 11.5%, ANT 9.7%
(Antonelli took pole but started sixth after a penalty). Forecasts made after
qualifying are the model's evaluated mode; pre-weekend ones are much weaker.

Round 22, Qatar (pre-weekend, form only, data through round 21):

| | 1st | 2nd | 3rd | 4th | 5th |
|---|---|---|---|---|---|
| Win probability | VER 40.1% | NOR 15.5% | ANT 15.3% | RUS 8.3% | LEC 6.6% |

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
Norris (41%), Russell (30%) and Hamilton (25%). Points are derived from
classified positions, so check them against the official standings before
quoting them.

Files: `runs/forecast_2026_22/`, `runs/season_forecast_2026/`. The stored
`runs/forecast_2026_20/` and `runs/forecast_2026_21/` are the pre-race
forecasts for those rounds.

To refresh after a race, or after qualifying:

```bash
python scripts/build_race_results.py 2026 --out data/processed/race_results_2026.parquet
python scripts/export_extra_results.py --results data/processed/race_results_2026.parquet --with-qualifying
python scripts/fetch_weekend.py --race-id 2026_22          # after qualifying; add --grid ANT=6 for penalties
python scripts/predict_race.py --race-id 2026_22 --track "Qatar Grand Prix" \
    --extra-results data/manual/2026_20.csv data/manual/2026_21.csv \
    --entries data/manual/2026_22_grid.csv
```

`export_extra_results.py` writes one CSV per race that has official results
but is missing from the feature table
(`race_id, track, driver, team, grid_position, finish_position, status`).
`--entries` takes the starting grid and, optionally, qualifying best laps.
