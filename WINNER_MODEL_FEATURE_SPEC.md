# APEX winner-model feature specification

This document defines the feature contract before adding further model
parameters. Every feature must be available at the prediction timestamp and
must be calculated without using the race outcome being predicted.

## Prediction timing

The current model is an **after-qualifying, pre-race** model:

- Practice sessions are available.
- Qualifying/grid position is available.
- Race laps, finishing position, final race time, race pit stops, and race
  incidents are not available.

The model ranks drivers by a winner score. The highest score is predicted P1;
the next two scores are reported as P2 and P3. Scores are not yet calibrated
probabilities.

## Existing feature parameters

### 1. Fuel sensitivity

Parameter:

```text
fuel_sensitivity_s_per_kg = 0.032
```

Formula:

```text
fuel_corrected_lap_s
    = lap_time_s - fuel_sensitivity_s_per_kg * fuel_kg
```

Purpose: remove the estimated time penalty caused by carrying fuel.

Limitation: FastF1 does not provide direct fuel mass. `fuel_kg` is estimated
from a starting mass and linear burn, so this parameter must later be tested
with a sensitivity range rather than treated as truth.

### 2. Clean-lap filter

Only rows satisfying all conditions are used for pace features:

```text
valid == True
is_pit_lap == False
lap_time_s is not missing
fuel_kg is not missing
tyre_age is not missing
```

Purpose: prevent pit, neutralised, inaccurate, and incomplete laps from
distorting pace.

### 3. Minimum degradation stint length

Parameter:

```text
min_degradation_laps = 8
```

Formula:

```text
fuel_corrected_lap_s
    ~= intercept + degradation_s_per_lap * tyre_age
```

Purpose: estimate tyre-time loss per additional tyre-age lap while avoiding
slopes from very short stints.

### 4. Valid-lap quality threshold

Parameter:

```text
min_valid_laps = 5
```

Formula:

```text
feature_quality = "usable" if valid_laps >= 5 else "sparse"
```

Purpose: expose weak samples. This flag is not a calibrated confidence score.

## Pace features

### 5. Overall fuel-corrected median pace

```text
fuel_corrected_median_lap_s
    = median(fuel_corrected_lap_s)
```

Calculated per driver and race using all available pre-race clean laps.
Lower is faster.

### 6. Overall best lap

```text
best_lap_time_s = min(lap_time_s)
```

This measures peak observed pace. It is currently raw rather than
fuel-corrected and should be treated as a sensitive, high-variance feature.

### 7. Overall pace rank

```text
pace_rank = rank(fuel_corrected_median_lap_s within race)
```

Lower rank is better.

### 8. Overall pace gap

```text
pace_gap_to_best_s
    = driver_median_pace - fastest_driver_median_pace
```

This is more transferable than absolute lap time across different circuits.

### 9. Best-lap rank and gap

```text
best_lap_rank = rank(best_lap_time_s within race)

best_lap_gap_to_best_s
    = driver_best_lap - fastest_driver_best_lap
```

These features describe qualifying-like peak pace but can be noisy when
drivers run different fuel or tyre programmes.

## Practice-specific features

Practice features are calculated only from sessions whose names begin with
`Practice`.

### 10. Practice median pace

```text
practice_median_lap_s
    = median(fuel_corrected_lap_s from practice)
```

Purpose: represent longer-run and setup pace before qualifying.

### 11. Practice best lap

```text
practice_best_lap_s
    = min(lap_time_s from practice)
```

Purpose: represent peak practice pace.

### 12. Practice rank and gap

```text
practice_pace_rank
    = rank(practice_median_lap_s within race)

practice_gap_to_best_s
    = driver_practice_median - fastest_practice_median
```

### 13. Practice availability

```text
has_practice_data = 1 if practice_median_lap_s exists else 0
practice_laps = number of clean practice laps
practice_sessions = number of practice session types present
```

These distinguish a genuinely slow driver from a driver with incomplete data.

## Qualifying-specific features

Qualifying, Sprint Qualifying, and Sprint Shootout are treated as qualifying
session types.

### 14. Qualifying median pace

```text
qualifying_median_lap_s
    = median(fuel_corrected_lap_s from qualifying-type sessions)
```

### 15. Qualifying best lap

```text
qualifying_best_lap_s
    = min(lap_time_s from qualifying-type sessions)
```

### 16. Qualifying rank and gap

```text
qualifying_pace_rank
    = rank(qualifying_median_lap_s within race)

qualifying_gap_to_best_s
    = driver_qualifying_median - fastest_qualifying_median
```

### 17. Qualifying availability

```text
has_qualifying_data = 1 if qualifying_median_lap_s exists else 0
qualifying_laps = number of clean qualifying laps
qualifying_sessions = number of qualifying-type sessions present
```

## Tyre and stint features

### 18. Tyre degradation

For each driver/session/stint with at least eight clean laps:

```text
y = lap_time_s - fuel_sensitivity_s_per_kg * fuel_kg

y ~= intercept + slope * tyre_age

degradation_s_per_lap = slope
```

Positive slope means the tyre becomes slower with age. Negative slopes should
be treated as noisy evidence and bounded or shrunk before production use.

### 19. Stint count

```text
stints = count(distinct stint_id)
```

This indicates how much tyre-programme information is available. It is useful
for data quality and strategy context, but does not directly mean a driver is
faster.

## Grid features

Grid fields come from official results but only pre-race fields are attached.

### 20. Grid position

```text
grid_position = official starting position
```

This is the most direct pre-race track-position feature.

### 21. Grid rank

```text
grid_rank = rank(grid_position within race)
```

Lower is better.

### 22. Grid gap to pole

```text
grid_gap_to_pole
    = driver_grid_position - minimum_grid_position
```

This is robust to missing grid numbers and expresses distance from pole.

## Implemented historical and context features

These features are implemented in `MODEL_COLUMNS`. Their inputs are restricted
to information available before the target race.

### 23. Driver recent form

For the previous `N = 5` completed races before the target race:

```text
driver_form_score
    = mean(1 / max(finish_position, 1))
```

Alternative, less position-sensitive formulation:

```text
driver_recent_mean_finish
    = mean(finish_position over previous N races)
```

The target race must be excluded.

### 24. Team recent form

For both team cars over the previous `N = 5` distinct races:

```text
team_form_score
    = mean(1 / max(finish_position, 1))

team_recent_mean_finish
    = mean(finish_position)
```

Team identity must come from pre-race entry data, not inferred from finishing
results in the target race.

### 25. Reliability

Calculated only from prior rows before the target race:

```text
driver_finish_rate
    = classified_finishes / starts

driver_dnf_rate
    = DNFs / starts

team_dnf_rate
    = team DNFs / team starts
```

The implementation applies Laplace smoothing:

```text
smoothed_rate
    = (observed_successes + prior_successes)
      / (observations + prior_observations)
```

### 26. Weather

Practice-session weather fields available before the race are used:

```text
practice_air_temp
practice_track_temp
practice_wet_lap_fraction
```

The wet-lap fraction is based on INTERMEDIATE/WET compounds or an available
weather wetness signal. Actual race weather is forbidden for pre-race
prediction.

### 27. Circuit history

Using only races before the target race:

```text
driver_track_mean_finish
driver_track_start_count
team_track_mean_finish
team_track_start_count
```

Sparse history must be represented with counts and missingness indicators.

### 28. Prior-track safety context

When race-session laps are available in the cache, summarize only completed
earlier races at the same circuit:

```text
prior_track_safety_car_lap_rate
    = mean(safety_car_laps / race_laps over prior track races)

prior_track_neutralized_lap_rate
    = mean(neutralized_laps / race_laps over prior track races)
```

FastF1 `TrackStatus` codes are used to identify safety-car and neutralized
periods. The target race's race-session laps are excluded, and missing race
sessions remain missing rather than being filled from the target outcome.
These are circuit-risk context features, not a forecast of the target race's
actual safety-car timing.

## Industry-grade upgrade design

The following formulations define the target architecture. They are separated
from the currently implemented baseline so that unavailable prediction-time
inputs are not silently replaced with post-race data.

### Hierarchical pace decomposition

The production pace layer should estimate a latent pre-race pace distribution:

```text
lap_time
    = circuit_effect
    + session_effect
    + team_effect
    + driver_effect
    + compound_effect
    + degradation(tyre_age, compound)
    + fuel_effect(fuel_kg)
    + weather_effect
    + traffic_effect
    + observation_error
```

Driver, team, and circuit effects should use hierarchical shrinkage. The
current aggregate features remain the baseline until team, traffic, and
prediction-time weather sources are available.

### Compound-specific state-space degradation

```text
latent_pace_t
    = latent_pace_(t-1) + degradation_rate_(compound,t) + process_noise

observed_lap_t
    = latent_pace_t + fuel_effect_t + traffic_effect_t
    + weather_effect_t + observation_error_t
```

Pit stops reset tyre age and the latent compound state. Degradation rates
should be compound-, circuit-, and temperature-aware, with uncertainty carried
into simulation. The current robust linear stint slope is the fallback
baseline, not a full state-space model.

### Fuel sensitivity uncertainty

Fuel mass is estimated rather than measured, so the fixed correction requires
validation-only sensitivity analysis:

```text
fuel_effect ~ Normal(mu_fuel, sigma_fuel)
```

At minimum evaluate `0.020, 0.025, 0.030, 0.035, 0.040` seconds/kg. A
forecast whose top three changes across that range must be marked
fuel-sensitive.

### Dynamic driver/team state and reliability

Prior-race form uses only completed races before the target:

```text
state_t = state_(t-1) + drift_t
form_t  = alpha * prior_race + (1 - alpha) * form_(t-1)
```

Reliability is a separate finish-probability component:

```text
hazard_t = baseline_hazard * exp(team_reliability
    + driver_incident_rate + circuit_risk + weather_risk)
P(win) = P(competitive pace) * P(finish) * P(strategy succeeds)
```

Both use smoothed rates, observation counts, and target-race exclusion.
Mechanical and driver-incident causes should remain separate when available.

### Pre-race weather and circuit context

Only forecast or otherwise pre-race observations may be used:

```text
weather = [air_temperature, track_temperature, rain_probability,
           wind_speed, humidity]
circuit = [driver_track_history, team_track_history,
           overtaking_difficulty, safety_car_rate]
```

Sparse history requires counts and missingness indicators. Actual race
weather, incidents, pit stops, and final race time remain prohibited.

### Race-consistent ranking probabilities

The binary classifier's independent probabilities are not a race distribution.
Convert each driver's score to a race-level strength and normalize within the
race:

```text
P(P1 = i) = exp(theta_i) / sum_j exp(theta_j)
```

For a complete order, use the Plackett-Luce factorization:

```text
P(order) = product_k exp(theta_(k))
                  / sum_(j in remaining) exp(theta_j)
```

The first implementation uses this deterministic normalization around the
existing classifier. A Bayesian time-weighted Plackett-Luce model is retained
as the next replacement and must be compared on identical expanding folds.

### Monte Carlo race simulation

The final simulator should sample pace, tyre degradation, fuel sensitivity,
strategy, safety-car events, weather, and reliability, then report:

```text
P(P1), P(P2), P(P3), expected_finish
```

Use at least 10,000 seeded simulations per race. Until strategy and hazard
inputs are available, the normalized ranking must not be described as a full
race simulator.

### Probability calibration

Calibration is fit only inside the training/validation workflow and never on
reported test or future forecast rows. Report log loss, Brier score,
calibration error, and reliability diagrams. Platt scaling is preferred for
small chronological samples; isotonic regression requires a larger
calibration sample.

## Implementation sequence

1. **Race-consistent score layer:** within-race normalization, top-three
   overlap, winner rank, log loss, Brier score, and calibration-error metrics.
2. **Validation-only calibration:** chronological Platt calibration with an
   untouched test report.
3. **Dynamic driver/team state and reliability:** prior-race-only rolling
   features with smoothing and explicit counts.
4. **Fuel sensitivity and compound-specific degradation:** bounded sensitivity
   comparisons and a hierarchical/state-space estimator.
5. **Weather and circuit context:** timestamp-valid forecast/context sources.
6. **Full Bayesian Plackett-Luce:** compare against the normalized classifier.
7. **Monte Carlo simulator:** add strategy, safety-car, weather, and
   reliability sampling for P1/P2/P3 distributions.

Each step is retained only if it improves expanding-fold validation without
target-race leakage.

## Model formulation

The current classifier receives the feature vector:

```text
X_driver,race = [all MODEL_COLUMNS]
```

The target is:

```text
winner =
    1 if finish_position == 1 and classified == True
    0 otherwise
```

Model pipeline:

```text
median imputation + missingness indicators
    -> HistGradientBoostingClassifier
```

Current parameters:

```text
max_iter = 150
learning_rate = 0.05
max_leaf_nodes = 7
random_state = 0
```

For each race, the baseline output score is:

```text
score_driver = P(winner = 1 | X_driver,race)
```

The first ranking upgrade converts classifier log-odds to a stable
within-race Plackett-Luce-style distribution. This makes P1 probabilities
race-consistent; full P2/P3 probabilities still require the later full-order
model or simulator. Until validation-only calibration is added, these remain
comparative probabilities rather than guaranteed calibrated probabilities.

## Evaluation requirements

Use expanding season holdouts:

```text
train 2018..Y-1 -> validate Y
```

Report:

- winner accuracy
- top-three overlap
- mean reciprocal rank of actual winner
- log loss
- Brier score
- calibration error
- performance by regulation era
- performance by data completeness
- fuel-sensitivity stability
- top-three probability mass by finishing position

No feature may use target-race finishing position, final race time, target-race
pit stops, target-race incidents, or actual target-race weather.

## Known limitations before the next implementation

- Fuel is estimated, not measured.
- Practice and qualifying lap programmes are not fully comparable.
- Tyre degradation is a descriptive slope, not a race strategy simulation.
- Grid position means the forecast is after qualifying.
- The first ranking layer is deterministic normalization, not yet a Bayesian
  full-order model.
- P2/P3 probabilities and strategy-aware outcomes require the later simulator.
