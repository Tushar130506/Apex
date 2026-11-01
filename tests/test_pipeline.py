import json
import numpy as np, pandas as pd, pytest
import fixtures as fx
from apex.data_io import load_processed
from apex.evaluation.reports import predict_with_run, train_run
from apex.evaluation.splits import LeakageError, assert_no_leakage, season_split
from apex.features.lap_features import build_features
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
