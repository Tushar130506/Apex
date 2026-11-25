"""Fuel/progress and degradation recovered from a simulated race."""
import numpy as np
import pandas as pd
import pytest

from apex.features.race_pace import circuit_table, clean_race_laps, fit_race_pace, team_race_pace

PROGRESS, DEG, DEG2 = -0.06, 0.05, 0.001


def _race(seed=0, laps=57, drivers=16):
    rng = np.random.default_rng(seed)
    rows, stops = [], []
    for number in range(drivers):
        driver = f"D{number:02d}"
        base = 92.0 + 0.15 * number
        pits = sorted(rng.choice(np.arange(12, laps - 8), size=1 + number % 2, replace=False).tolist())
        if len(pits) == 2 and pits[1] - pits[0] < 6:
            pits = [pits[0]]
        stops += [{"driver": driver, "lap_no": lap} for lap in pits]
        start = 1
        for lap in range(1, laps + 1):
            age = lap - start + 1
            time = base + PROGRESS * lap + DEG * age + DEG2 * age**2 + rng.normal(0, 0.15)
            if lap == 1:
                time += 6
            if lap in pits:
                time += 20
                start = lap + 1
            if lap - 1 in pits:
                time += 4
            if lap in (30, 31):          # safety car for everyone
                time *= 1.35
            if driver == "D03" and lap == 40:
                time += 8                # traffic
            rows.append({"race_id": "2025_5", "driver": driver, "lap_no": lap, "lap_time_s": time})
    return pd.DataFrame(rows), pd.DataFrame(stops)


def test_cleaning_marks_pit_safety_car_and_slow_laps():
    laps, pits = _race()
    clean = clean_race_laps(laps, pits)
    reasons = clean.set_index(["driver", "lap_no"])["exclusion_reason"]
    first_stop = int(pits.loc[pits.driver.eq("D00"), "lap_no"].iloc[0])
    assert reasons[("D00", first_stop)] == "pit_in_lap" and reasons[("D00", first_stop + 1)] == "pit_out_lap"
    assert reasons[("D05", 1)] == "first_lap" and reasons[("D05", 30)] == "neutralised"
    assert reasons[("D03", 40)] == "slow_lap"
    after = clean[clean.driver.eq("D00") & clean.lap_no.eq(first_stop + 1)].iloc[0]
    assert after.tyre_age == 1 and after.stint_id == 1


def test_fit_recovers_progress_and_degradation():
    laps, pits = _race()
    result = fit_race_pace(clean_race_laps(laps, pits), bootstrap=50)
    assert result["progress_s_per_lap"] == pytest.approx(PROGRESS, abs=0.01)
    assert result["deg_at_20_laps_s"] == pytest.approx(DEG * 20 + DEG2 * 400, abs=0.15)
    low, high = result["progress_ci95"]
    assert low < PROGRESS < high
    assert not result["mixed_conditions"] and 1.0 < result["field_spread"] < 1.06
    assert result["driver_pace_gap_s"]["D00"] == 0 and result["driver_pace_gap_s"]["D15"] == pytest.approx(2.25, abs=0.15)


def test_too_little_data_is_refused():
    laps, pits = _race()
    few = ["D00", "D01", "D02", "D03"]
    laps, pits = laps[laps.driver.isin(few)], pits[pits.driver.isin(few)]
    with pytest.raises(ValueError):
        fit_race_pace(clean_race_laps(laps, pits))


def test_drying_track_is_flagged():
    laps, pits = _race()
    drying = laps["lap_no"].clip(upper=25)            # wet early laps, 0.4 s/lap faster as it dries
    laps = laps.assign(lap_time_s=laps["lap_time_s"] + 0.4 * (25 - drying))
    result = fit_race_pace(clean_race_laps(laps, pits), bootstrap=10)
    assert result["mixed_conditions"]


def test_circuit_table_skips_wet_races_and_keeps_coefficients_together():
    races = pd.DataFrame({
        "circuit": ["vegas", "vegas", "vegas", "silverstone"],
        "track": ["Las Vegas Grand Prix"] * 3 + ["British Grand Prix"],
        "progress_s_per_lap": [-0.07, -0.04, -0.05, -0.38],
        "deg_s_per_lap": [0.03, -0.11, 0.05, 0.40],
        "deg_quad_s_per_lap2": [0.0, 0.002, 0.0, -0.003],
        "deg_at_20_laps_s": [0.6, -1.4, 1.0, 6.8],
        "pit_stops": [23, 23, 20, 35],
        "mixed_conditions": [False, False, False, True],
    })
    table = circuit_table(races).set_index("circuit")
    assert list(table.index) == ["vegas"]
    row = table.loc["vegas"]
    assert row["races"] == 3 and row["deg_at_20_laps_s"] == pytest.approx(0.6)
    assert row["deg_s_per_lap"] == pytest.approx(0.03) and row["progress_s_per_lap"] == pytest.approx(-0.05)


def test_team_pace_finds_the_fast_team_and_the_heavy_wear():
    laps, pits = _race()
    team = {f"D{n:02d}": f"T{n // 2}" for n in range(16)}
    laps = laps.assign(team=laps["driver"].map(team))
    clean = clean_race_laps(laps, pits)
    age = clean["tyre_age"]
    heavy = clean["team"].eq("T7")                       # wears 0.05 s/lap faster than the rest
    clean = clean.assign(lap_time_s=clean["lap_time_s"] + np.where(heavy, 0.05 * age, 0.0))
    result = team_race_pace(clean).set_index("team")
    assert result["pace_gap_pct"].idxmin() == "T0"
    assert result.loc["T7", "deg_rel_s_per_lap"] == pytest.approx(0.05, abs=0.015)
    assert result.drop("T7")["deg_rel_s_per_lap"].abs().max() < 0.02


def test_race_pace_form_uses_earlier_races_only():
    from apex.features.form_features import add_race_pace_form
    measured = pd.DataFrame({
        "race_id": ["2025_1", "2025_1", "2025_2", "2025_2"],
        "team": ["Ferrari", "McLaren", "Ferrari", "McLaren"],
        "pace_gap_pct": [0.5, -0.5, 9.9, 9.9],
        "deg_rel_s_per_lap": [0.02, -0.02, 9.9, 9.9],
        "mixed_conditions": [False, False, True, True],
    })
    features = pd.DataFrame({
        "race_id": ["2025_1", "2025_2", "2025_2", "2025_3"],
        "team": ["Ferrari", "Ferrari", "McLaren", "Ferrari"],
    })
    out = add_race_pace_form(features, measured)
    assert pd.isna(out.loc[0, "team_race_pace_form"])             # no earlier race
    assert out.loc[1, "team_race_pace_form"] == pytest.approx(0.5)
    assert out.loc[2, "team_race_deg_form"] == pytest.approx(-0.02)
    assert out.loc[3, "team_race_pace_form"] == pytest.approx(0.5)  # wet 2025_2 ignored


def test_team_pace_ignores_drivers_with_a_handful_of_laps():
    laps, pits = _race()
    team = {f"D{n:02d}": f"T{n // 2}" for n in range(16)}
    laps = laps.assign(team=laps["driver"].map(team))
    laps = laps[~(laps["driver"].eq("D15") & laps["lap_no"].gt(3))]     # early retirement
    pits = pits[pits["driver"].ne("D15")]
    result = team_race_pace(clean_race_laps(laps, pits)).set_index("team")
    assert result["pace_gap_pct"].abs().max() < 3


def test_race_pace_benchmark_uses_earlier_races_only():
    import importlib.util
    spec = importlib.util.spec_from_file_location("bench", "scripts/benchmark_race_pace.py")
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    rounds = range(1, 26)
    races = pd.DataFrame({
        "race_id": [f"2025_{n}" for n in rounds],
        "circuit": ["A" if n % 5 == 0 else f"C{n}" for n in rounds],
        "progress_s_per_lap": -0.05,
        "deg_at_20_laps_s": [5.0 if n == 25 else 1.0 for n in rounds],
        "mixed_conditions": False,
        "season": 2025,
    })
    errors = bench.predictions(races)
    assert set(errors["race_id"]) == {"2025_25"}            # needs 20 earlier races and an earlier visit
    row = errors.set_index("method").loc["same circuit, earlier races"]
    assert row["deg20_error"] == pytest.approx(4.0)        # its own 5.0 s is not used
