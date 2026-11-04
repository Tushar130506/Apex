import json
import numpy as np, pandas as pd, pytest
import fixtures as fx
from apex.data_io import load_processed, load_processed_season
from apex.evaluation.reports import predict_with_run, train_run
from apex.evaluation.splits import LeakageError, assert_no_leakage, season_split
from apex.features.lap_features import build_features
from apex.features.race_features import build_race_features
from apex.features.winner_features import (
    MODEL_COLUMNS,
    add_session_identity,
    build_pre_race_features,
    validate_race_targets,
)
from apex.models.winner import race_probabilities, train_winner_model
from apex.models.walk_forward import walk_forward_predictions
from apex.models.degradation import stint_slopes
from apex.schemas.contracts import ContractError, validate_laps
from apex.strategy.pace_model import ParametricPace


@pytest.fixture(scope="module")
def laps():
    return fx.make_dataset(64, seed=0)


def test_contract_ok_and_rejects_bad(laps):
    validate_laps(laps)
    with pytest.raises(ContractError):
        validate_laps(laps.assign(source="mystery"))
    bad = laps.copy(); bad.loc[bad.valid == False, "exclusion_reason"] = None  # noqa: E712
    with pytest.raises(ContractError):
        validate_laps(bad)


def test_season_split_disjoint_and_guards(laps):
    m = season_split(laps, [2022, 2023], [2024], [2025])
    assert set(laps[m["train"]].season) == {2022, 2023} and set(laps[m["test"]].season) == {2025}
    with pytest.raises(LeakageError):
        season_split(laps, [2022, 2023], [2023], [2025])
    with pytest.raises(ValueError):
        season_split(laps, [2022], [2024], [2031])
    leaky = {"a": laps.session_id.isin(["fx_000", "fx_001"]).to_numpy(), "b": laps.session_id.isin(["fx_001"]).to_numpy()}
    with pytest.raises(LeakageError):
        assert_no_leakage(laps, leaky)


def test_features_use_only_past(laps):
    f = build_features(laps)
    g = f[(f.session_id == "fx_000") & (f.stint_id == 0)].sort_values("lap_no")
    for _, r in g[g.model_row].head(10).iterrows():
        assert r.prev_lap_time in set(g[(g.lap_no < r.lap_no) & g.valid].lap_time_s)
    assert not f.loc[f.model_row, "is_pit_lap"].any()


def test_degradation_and_fuel_recovered(laps):
    p = ParametricPace.fit(laps)
    for c, (_, deg) in fx.COMPOUNDS.items():
        assert p.deg[c] == pytest.approx(deg, abs=0.012)
    assert p.fuel_s_per_kg == pytest.approx(fx.FUEL_EFFECT_S_PER_KG, abs=0.006)
    s = stint_slopes(laps, fuel_s_per_kg=fx.FUEL_EFFECT_S_PER_KG)
    assert len(s) > 10


def test_race_features_include_fuel_and_degradation(laps):
    features = build_race_features(laps, fuel_s_per_kg=fx.FUEL_EFFECT_S_PER_KG)
    assert len(features) == laps.session_id.nunique()
    assert {
        "fuel_corrected_median_lap_s",
        "degradation_s_per_lap",
        "estimated_fuel_fraction",
        "feature_quality",
    } <= set(features.columns)
    row = features.iloc[0]
    assert row.fuel_sensitivity_s_per_kg == pytest.approx(fx.FUEL_EFFECT_S_PER_KG)
    assert row.feature_quality == "usable"
    assert row.estimated_fuel_fraction == pytest.approx(1.0)


def test_race_features_reject_invalid_configuration(laps):
    with pytest.raises(ValueError, match="fuel_s_per_kg"):
        build_race_features(laps, fuel_s_per_kg=-1)
    with pytest.raises(ValueError, match="missing"):
        build_race_features(laps.drop(columns=["fuel_kg"]))


def test_winner_features_and_holdout_model():
    rows = []
    for season in (2022, 2023, 2024, 2025):
        for rnd in range(1, 3):
            for driver, pace in (("AAA", 80.0), ("BBB", 81.0), ("CCC", 82.0)):
                for lap in range(1, 9):
                    rows.append({
                        "source": "fastf1_real", "season": season,
                        "session_id": f"fastf1_{season}_{rnd}_Practice 1",
                        "track": f"Track {rnd}", "driver": driver, "lap_no": lap,
                        "lap_time_s": pace + lap * 0.02, "valid": True,
                        "exclusion_reason": None, "compound": "MEDIUM", "tyre_age": lap,
                        "stint_id": 1, "fuel_kg": 100 - lap, "fuel_provenance": "estimated",
                        "track_temp": 30.0, "air_temp": 20.0, "track_status": "1", "is_pit_lap": False,
                    })
    laps = pd.DataFrame(rows)
    features = build_pre_race_features(laps)
    results = pd.DataFrame([
        {"race_id": f"{season}_{rnd}", "driver": driver, "finish_position": 1 if driver == "AAA" else 2,
         "grid_position": 1 if driver == "AAA" else 2, "classified": True,
         "team": "TEAM_A" if driver == "AAA" else "TEAM_B", "status": "Finished"}
        for season in (2022, 2023, 2024, 2025)
        for rnd in range(1, 3)
        for driver in ("AAA", "BBB", "CCC")
    ])
    from apex.features.winner_features import attach_race_results
    joined = attach_race_results(features, results)
    validate_race_targets(joined)
    assert {"grid_position", "grid_rank", "grid_gap_to_pole"} <= set(joined.columns)
    assert {
        "practice_median_lap_s",
        "qualifying_median_lap_s",
        "practice_pace_rank",
        "qualifying_pace_rank",
    } <= set(joined.columns)
    assert joined.loc[joined.driver == "AAA", "grid_rank"].eq(1).all()
    assert {
        "driver_form_score",
        "team_finish_rate",
        "driver_track_start_count",
        "practice_track_temp",
        "prior_track_safety_car_lap_rate",
        "prior_track_neutralized_lap_rate",
    } <= set(joined.columns)
    first_race = joined[joined["race_id"] == "2022_1"]
    assert first_race["driver_form_score"].isna().all()
    later_race = joined[joined["race_id"] == "2025_2"]
    assert later_race["driver_form_score"].notna().all()
    metrics, _ = train_winner_model(joined, [2022], [2023], [2024, 2025])
    assert {"train", "val", "test", "fit_diagnostics"} <= set(metrics)
    assert metrics["test"]["races"] == 4
    assert {
        "winner_mean_reciprocal_rank",
        "race_log_loss",
        "race_brier_score",
        "calibration_error",
    } <= set(metrics["test"])


def test_race_probabilities_are_normalized_per_race():
    data = pd.DataFrame({"race_id": ["a", "a", "b"], "winner": [1, 0, 1]})
    probabilities = race_probabilities(data, np.array([0.8, 0.2, 0.4]))
    assert probabilities.sum() == pytest.approx(2.0)
    assert probabilities[:2].sum() == pytest.approx(1.0)
    assert probabilities[2] == pytest.approx(1.0)


def test_walk_forward_adds_each_completed_race_in_order():
    rows = []
    for season in (2024, 2025):
        for rnd in range(1, 5):
            for driver, pace in (("AAA", 80.0), ("BBB", 81.0), ("CCC", 82.0)):
                rows.append({
                    "race_id": f"{season}_{rnd}",
                    "season": season,
                    "driver": driver,
                    "winner": int(driver == "AAA"),
                    "finish_position": 1 if driver == "AAA" else 2,
                    **{column: pace for column in MODEL_COLUMNS},
                })
    data = pd.DataFrame(rows)
    predictions, metrics = walk_forward_predictions(data, 2025, half_races=2)
    assert [item["race_id"] for item in metrics] == ["2025_3", "2025_4"]
    assert predictions.groupby("race_id")["training_races"].first().tolist() == [6, 7]
    assert predictions.groupby("race_id")["race_probability"].sum().round(8).eq(1).all()
    assert predictions.groupby("race_id")["calibrated_race_probability"].sum().round(8).eq(1).all()
    assert {"grid_baseline_accuracy", "qualifying_baseline_accuracy", "calibrated_winner_accuracy"} <= set(metrics[0])


def test_training_refuses_fixture_source_by_default(laps, tmp_path):
    with pytest.raises(ValueError, match="refuses sources"):
        train_run(laps, [2022, 2023], [2024], [2025], "x", tmp_path / "r", tmp_path / "p")


def test_train_run_artifacts_and_predictions(laps, tmp_path):
    m = train_run(laps, [2022, 2023], [2024], [2025], "t1", tmp_path / "runs", tmp_path / "pred", allow_fixture=True)
    for split in ("val", "test"):
        assert m[split]["gbm_beats_best_baseline"], m[split]
        assert abs(m[split]["conformal"]["coverage"] - 0.9) < 0.08
    run = tmp_path / "runs" / "t1"
    assert (run / "model.joblib").exists() and json.loads((run / "config.json").read_text())["test_years"] == [2025]
    p = pd.read_parquet(tmp_path / "pred" / "t1" / "predictions.parquet")
    assert set(p.split) == {"train", "val", "test"}
    assert set(p[p.split == "test"].season) == {2025} and set(p[p.split == "train"].season) == {2022, 2023}
    assert {"actual_s", "pred_gbm_s", "pred_lo_s", "pred_hi_s", "in_interval"} <= set(p.columns)
    out = predict_with_run(run, laps[laps.season == 2025], tmp_path / "new.parquet")
    assert (tmp_path / "new.parquet").exists() and (out.split == "predict").all()
    assert not out.season_seen_in_training.any()       # 2025 not in train years -> flagged


def test_load_processed_roundtrip(laps, tmp_path):
    for y, g in laps.groupby("season"):
        g.to_parquet(tmp_path / f"laps_{y}.parquet")
    assert set(load_processed(tmp_path).season) == {2022, 2023, 2024, 2025}
    assert set(load_processed(tmp_path, [2024]).season) == {2024}
    assert set(load_processed_season(tmp_path, 2024).season) == {2024}
