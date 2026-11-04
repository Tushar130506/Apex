#!/usr/bin/env python
"""User-facing APEX race prediction and telemetry dashboard."""
from __future__ import annotations

from pathlib import Path
import math
import re
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from apex.data_io import load_processed
from apex.features.winner_features import (
    MODEL_COLUMNS,
    attach_grid_features,
    attach_historical_features,
    attach_prior_safety_features,
    attach_race_results,
    build_pre_race_features,
    build_race_safety_summary,
)
from apex.models.race_monte_carlo import monte_carlo_race_probabilities
from apex.models.walk_forward import make_walk_forward_model


ROOT = Path(__file__).parent
PROCESSED = ROOT / "data" / "processed"

st.set_page_config(
    page_title="APEX Race Intelligence",
    page_icon="🏁",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner="Loading lap data...")
def load_laps(signature: tuple[tuple[str, int], ...]) -> pd.DataFrame:
    return load_processed()


@st.cache_data(show_spinner="Loading race predictions...")
def load_predictions() -> pd.DataFrame:
    preferred = ROOT / "runs" / "walk_forward_championship_2023_2026" / "race_predictions.parquet"
    candidates = [preferred] if preferred.exists() else []
    candidates += sorted(
        (ROOT / "runs").glob("**/*predictions.parquet"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    frames: list[pd.DataFrame] = []
    seen: set[tuple[str, str]] = set()
    for path in candidates:
        try:
            frame = pd.read_parquet(path)
        except (OSError, ValueError):
            continue
        required = {"race_id", "season", "track", "driver"}
        if not required <= set(frame.columns):
            continue
        keys = frame[["race_id", "driver"]].astype(str).apply(tuple, axis=1)
        keep = ~keys.isin(seen)
        if keep.any():
            frames.append(frame.loc[keep].copy())
            seen.update(keys[keep])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


@st.cache_data(show_spinner="Loading official results...")
def load_results() -> pd.DataFrame:
    paths = sorted(PROCESSED.glob("race_results*.parquet"))
    frames = [pd.read_parquet(path) for path in paths]
    return pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["race_id", "driver"], keep="last"
    ) if frames else pd.DataFrame()


@st.cache_data(show_spinner="Loading championship forecast...")
def load_championship_forecast() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    candidates = sorted(
        (ROOT / "runs").glob("**/driver_forecast.parquet"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    for driver_path in candidates:
        constructor_path = driver_path.with_name("constructor_forecast.parquet")
        if constructor_path.exists():
            return (
                pd.read_parquet(driver_path),
                pd.read_parquet(constructor_path),
                str(driver_path.parent),
            )
    return pd.DataFrame(), pd.DataFrame(), ""


@st.cache_data(show_spinner="Loading model comparison...")
def load_model_comparison() -> pd.DataFrame:
    path = ROOT / "runs" / "walk_forward_championship_2023_2026" / "race_prediction_actual.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame["correct_winner"] = frame["predicted_position"].eq(1) & frame["finish_position"].eq(1)
    frame["actual_winner_probability"] = frame["race_probability"].where(
        frame["finish_position"].eq(1)
    )
    return frame


def comparison_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season, group in frame.groupby("season", sort=True):
        races = group["race_id"].nunique()
        winners = group[group["finish_position"].eq(1)]
        predicted_winners = group[group["predicted_position"].eq(1)]
        correct = predicted_winners.merge(
            winners[["race_id", "driver"]], on=["race_id", "driver"], how="inner"
        )["race_id"].nunique()
        top_three = group[group["predicted_position"].le(3)].merge(
            group[group["finish_position"].le(3)][["race_id", "driver"]],
            on=["race_id", "driver"],
            how="inner",
        )
        log_loss = -pd.to_numeric(winners["race_probability"], errors="coerce").clip(
            lower=1e-12
        ).apply(math.log).mean()
        rows.append(
            {
                "season": int(season),
                "races": races,
                "winner_accuracy_pct": 100 * correct / races if races else 0,
                "top_three_overlap_pct": 100 * len(top_three) / (3 * races) if races else 0,
                "winner_log_loss": log_loss,
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Building leakage-safe pre-race features...")
def build_model_inputs(
    lap_signature: tuple[tuple[str, int], ...],
    result_signature: tuple[tuple[str, int], ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build features once; target outcomes are not used in feature construction."""
    lap_data = load_processed()
    result_data = load_results()
    features = build_pre_race_features(lap_data)
    safety = build_race_safety_summary(lap_data)
    features = attach_prior_safety_features(features, safety)
    labeled = attach_race_results(features, result_data) if not result_data.empty else pd.DataFrame()
    return features, labeled


@st.cache_data(show_spinner="Training the model through the previous race...")
def predict_on_demand(
    lap_signature: tuple[tuple[str, int], ...],
    result_signature: tuple[tuple[str, int], ...],
    target_race_id: str,
) -> pd.DataFrame:
    """Train only on races before target_race_id and predict its pre-race rows."""
    features, labeled = build_model_inputs(lap_signature, result_signature)
    target = features[features["race_id"].eq(target_race_id)].copy()
    if target.empty:
        return pd.DataFrame()
    target_order = int(target_race_id.split("_")[0]) * 100 + int(target_race_id.split("_")[1])
    training = labeled[
        labeled["race_id"].astype(str).map(
            lambda value: int(value.split("_")[0]) * 100 + int(value.split("_")[1])
        ) < target_order
    ].copy()
    if training.empty:
        return pd.DataFrame()

    # Historical form may only use results before the target race. Grid and
    # team are pre-race facts, so use them only for the target race itself.
    result_data = load_results()
    target_order_values = result_data["race_id"].astype(str).map(
        lambda value: int(value.split("_")[0]) * 100 + int(value.split("_")[1])
    )
    prior_results = result_data.loc[target_order_values < target_order].copy()
    target_result = result_data[result_data["race_id"].eq(target_race_id)].copy()
    if not target_result.empty:
        target = target.merge(
            target_result[["race_id", "driver", "grid_position", "team"]].drop_duplicates(
                ["race_id", "driver"]
            ),
            on=["race_id", "driver"],
            how="left",
        )
    target = attach_historical_features(target, prior_results)
    target = attach_grid_features(target, target_result) if not target_result.empty else target
    model = make_walk_forward_model()
    model.fit(training[MODEL_COLUMNS], training["winner"])
    target["score"] = model.predict_proba(target[MODEL_COLUMNS])[:, 1]
    target["race_probability"] = monte_carlo_race_probabilities(
        target, target["score"].to_numpy(), simulations=1000, seed=0
    )
    target["predicted_position"] = (
        target["race_probability"].rank(method="first", ascending=False).astype(int)
    )
    target["training_races"] = int(training["race_id"].nunique())
    target["training_through_race"] = target_order - 1
    return target.sort_values("race_probability", ascending=False)


@st.cache_data(show_spinner="Building historical circuit forecast...")
def predict_future_circuit(
    lap_signature: tuple[tuple[str, int], ...],
    result_signature: tuple[tuple[str, int], ...],
    season: int,
    circuit: str,
) -> pd.DataFrame:
    """Forecast a future circuit from prior circuit data and current form."""
    features, labeled = build_model_inputs(lap_signature, result_signature)
    result_data = load_results()
    season_results = result_data[result_data["season"].eq(season)].copy()
    if season_results.empty:
        return pd.DataFrame()
    current_order = season_results["race_id"].astype(str).map(
        lambda value: int(value.split("_")[0]) * 100 + int(value.split("_")[1])
    ).max()
    target_order = current_order + 1
    target_race_id = f"{season}_24" if season == 2026 and circuit == "Abu Dhabi Grand Prix" else f"{season}_{target_order % 100}"
    current_drivers = season_results[
        season_results["race_id"].astype(str).map(
            lambda value: int(value.split("_")[0]) * 100 + int(value.split("_")[1])
        ).eq(current_order)
    ][["driver", "team"]].drop_duplicates("driver")
    circuit_rows = features[features["track"].eq(circuit)].copy()
    if circuit_rows.empty:
        return pd.DataFrame()
    circuit_rows["_order"] = circuit_rows["race_id"].astype(str).map(
        lambda value: int(value.split("_")[0]) * 100 + int(value.split("_")[1])
    )
    circuit_rows = circuit_rows.sort_values("_order").drop_duplicates("driver", keep="last")
    fallback = features[features["race_id"].eq(f"{season}_{current_order % 100}")].copy()
    target = current_drivers.merge(
        circuit_rows.drop(columns=["_order", "season", "race_id", "track"], errors="ignore"),
        on="driver",
        how="left",
        suffixes=("", "_circuit"),
    )
    if not fallback.empty:
        fallback = fallback.drop(columns=["season", "race_id", "track"], errors="ignore")
        target = target.merge(
            fallback,
            on="driver",
            how="left",
            suffixes=("", "_fallback"),
        )
        for column in MODEL_COLUMNS:
            fallback_column = f"{column}_fallback"
            if column not in target and fallback_column in target:
                target[column] = target[fallback_column]
            elif fallback_column in target:
                target[column] = target[column].fillna(target[fallback_column])
    target["race_id"] = target_race_id
    target["season"] = season
    target["track"] = circuit
    target["grid_position"] = float("nan")
    target["grid_rank"] = float("nan")
    target["grid_gap_to_pole"] = float("nan")
    prior_results = result_data[
        result_data["race_id"].astype(str).map(
            lambda value: int(value.split("_")[0]) * 100 + int(value.split("_")[1])
        ).lt(current_order + 1)
    ].copy()
    target = attach_historical_features(target, prior_results)
    training = labeled[
        labeled["race_id"].astype(str).map(
            lambda value: int(value.split("_")[0]) * 100 + int(value.split("_")[1])
        ).lt(current_order + 1)
    ]
    if training.empty:
        return pd.DataFrame()
    model = make_walk_forward_model()
    model.fit(training[MODEL_COLUMNS], training["winner"])
    target["score"] = model.predict_proba(target[MODEL_COLUMNS])[:, 1]
    target["race_probability"] = monte_carlo_race_probabilities(
        target, target["score"].to_numpy(), simulations=1000, seed=0
    )
    target["predicted_position"] = target["race_probability"].rank(
        method="first", ascending=False
    ).astype(int)
    target["training_races"] = int(training["race_id"].nunique())
    target["future_circuit_forecast"] = True
    return target.sort_values("race_probability", ascending=False)


def _session_type(value: object) -> str:
    text = str(value)
    match = re.search(r"_(Practice 1|Practice 2|Practice 3|Qualifying|Sprint Qualifying|Sprint|Race)$", text)
    return match.group(1) if match else "Other"


def _safe_number(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _probability_column(predictions: pd.DataFrame) -> str | None:
    if "calibrated_race_probability" in predictions and predictions[
        "calibrated_race_probability"
    ].notna().any():
        return "calibrated_race_probability"
    return "race_probability" if "race_probability" in predictions else None


def _effective_probability(frame: pd.DataFrame) -> pd.Series:
    """Use calibrated probabilities when present, otherwise raw race probabilities."""
    calibrated = frame.get("calibrated_race_probability")
    raw = frame.get("race_probability")
    if calibrated is None and raw is None:
        return pd.Series(float("nan"), index=frame.index)
    if calibrated is None:
        return pd.to_numeric(raw, errors="coerce")
    if raw is None:
        return pd.to_numeric(calibrated, errors="coerce")
    return pd.to_numeric(calibrated, errors="coerce").fillna(
        pd.to_numeric(raw, errors="coerce")
    )


def _format_probability(value: object) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return "Unavailable"
    percentage = float(numeric) * 100
    if percentage > 0 and percentage < 0.1:
        return "<0.1%"
    return f"{percentage:.1f}%"


def _format_driver(value: object) -> str:
    return str(value) if pd.notna(value) else "Unknown"


try:
    lap_files = sorted(PROCESSED.glob("laps_*.parquet"))
    signature = tuple((str(path), path.stat().st_mtime_ns) for path in lap_files)
    laps = load_laps(signature)
except FileNotFoundError:
    st.error("No processed lap data found.")
    st.info("Run `python scripts/build_dataset.py season 2022 2023 2024 2025` first.")
    st.stop()

if laps.empty:
    st.warning("The processed lap dataset is empty.")
    st.stop()

predictions = load_predictions()
results = load_results()
wdc_forecast, wcc_forecast, forecast_source = load_championship_forecast()
comparison = load_model_comparison()
result_files = sorted(PROCESSED.glob("race_results*.parquet"))
result_signature = tuple((str(path), path.stat().st_mtime_ns) for path in result_files)
laps = laps.copy()
laps["session_type"] = laps["session_id"].map(_session_type)
laps["valid"] = laps["valid"].fillna(False).astype(bool)

seasons_available = sorted(pd.to_numeric(laps["season"], errors="coerce").dropna().astype(int).unique())
latest_season = seasons_available[-1]

st.title("🏁 APEX Race Intelligence")
st.caption("Choose a season, circuit, and driver to see the prediction and the evidence behind it.")

with st.sidebar:
    st.header("Choose your view")
    season = st.selectbox("Season", seasons_available, index=len(seasons_available) - 1)
    season_laps = laps[laps["season"].eq(season)]
    track_options = sorted(season_laps["track"].dropna().astype(str).unique())
    if season == 2026 and "Abu Dhabi Grand Prix" not in track_options:
        track_options.append("Abu Dhabi Grand Prix")
    if season == latest_season and season == 2026:
        loaded_races = (
            results[results["season"].eq(season)]["race_id"].nunique()
            if not results.empty and "season" in results
            else 0
        )
        st.info(
            f"2026 data is currently loaded through race {loaded_races} "
            "(Mexico City). Abu Dhabi is not materialized yet."
        )
    track = st.selectbox("Circuit", track_options)
    track_laps = season_laps[season_laps["track"].eq(track)]
    if track == "Abu Dhabi Grand Prix" and track_laps.empty:
        driver_options = sorted(
            results[results["season"].eq(season)]["driver"].dropna().astype(str).unique()
        )
    else:
        driver_options = sorted(track_laps["driver"].dropna().astype(str).unique())
    driver = st.selectbox("Driver", driver_options)
    st.divider()
    target_race_ids = (
        track_laps["session_id"].astype(str)
        .str.extract(r"^fastf1_(\d{4}_\d+)_")[0]
        .dropna()
        .unique()
    )
    target_race_id = (
        str(target_race_ids[0])
        if len(target_race_ids)
        else ("2026_24" if season == 2026 and track == "Abu Dhabi Grand Prix" else "")
    )
    ask_model = st.button(
        "Ask model now",
        type="primary",
        help="Train only on races before this race and predict using pre-race sessions.",
    )
    if track == "Abu Dhabi Grand Prix" and track_laps.empty:
        st.info(
            "Abu Dhabi 2026 sessions are not loaded. The future forecast uses historical "
            "Abu Dhabi pace plus current 2026 driver/team form; grid, weather, and 2026 telemetry are unavailable."
        )
    st.caption("Use **Prediction** for the pre-race view. Use **Telemetry** to inspect the selected driver's laps.")

on_demand_prediction = (
    predict_future_circuit(signature, result_signature, season, track)
    if ask_model and track == "Abu Dhabi Grand Prix" and track_laps.empty
    else predict_on_demand(signature, result_signature, target_race_id)
    if ask_model and target_race_id
    else pd.DataFrame()
)
prediction_probability = _probability_column(predictions)
selected_prediction = predictions[
    predictions["season"].eq(season) & predictions["track"].eq(track)
].copy()
if not selected_prediction.empty:
    selected_prediction["display_probability"] = _effective_probability(selected_prediction)
selected_driver_prediction = selected_prediction[selected_prediction["driver"].eq(driver)].copy()

tabs = st.tabs(
    ["Prediction", "Championship", "Compare model", "Telemetry", "Season overview", "Data quality"]
)

with tabs[0]:
    st.header(f"{track} · {season}")
    if ask_model:
        if on_demand_prediction.empty:
            st.warning("The model could not generate a prediction for this race. Check that earlier completed races and pre-race sessions are available.")
        else:
            st.subheader("Fresh on-demand prediction")
            st.caption(
                f"Trained on {int(on_demand_prediction['training_races'].iloc[0])} earlier races. "
                "Only practice, qualifying, historical form, and other pre-race features were used."
            )
            fresh = on_demand_prediction.copy()
            fresh["win_probability_pct"] = fresh["race_probability"] * 100
            fresh_columns = ["predicted_position", "driver", "win_probability_pct"]
            for column in ("team", "grid_position", "pace_rank", "driver_form_score"):
                if column in fresh:
                    fresh_columns.append(column)
            st.dataframe(
                fresh[fresh_columns],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "win_probability_pct": st.column_config.ProgressColumn(
                        "Win probability", min_value=0, max_value=100, format="%.3f%%"
                    ),
                    "predicted_position": st.column_config.NumberColumn("Model position", format="P%d"),
                    "grid_position": st.column_config.NumberColumn("Grid", format="P%d"),
                    "pace_rank": st.column_config.NumberColumn("Pace rank", format="P%d"),
                },
            )
            fresh_driver = fresh[fresh["driver"].eq(driver)]
            if not fresh_driver.empty:
                row = fresh_driver.iloc[0]
                st.metric(
                    f"{driver} on-demand win probability",
                    _format_probability(row["race_probability"]),
                )
            st.info(
                "This is the prediction generated by the model now. If the race has already happened, "
                "use the walk-forward section below to judge historical accuracy."
            )

    if selected_prediction.empty:
        st.info("No saved prediction is available for this season and circuit.")
        st.write("The telemetry and official-result views remain available below.")
    else:
        st.subheader("Walk-forward result and model evaluation")
        st.caption(
            "This saved result was generated chronologically and is the correct source for comparing "
            "predicted versus actual outcomes without look-ahead."
        )
        ranked = selected_prediction.sort_values(
            "display_probability" if "display_probability" in selected_prediction else "score",
            ascending=False,
        ).reset_index(drop=True)
        if "display_probability" in ranked:
            ranked["win_probability_pct"] = ranked["display_probability"] * 100
        ranked["model_rank"] = ranked.index + 1
        winner = ranked.iloc[0]
        selected = selected_driver_prediction.iloc[0] if not selected_driver_prediction.empty else None

        st.subheader("Race prediction")
        cards = st.columns(4)
        cards[0].metric("Predicted winner", _format_driver(winner["driver"]))
        cards[1].metric(
            f"{driver} win chance",
            _format_probability(selected["display_probability"]) if selected is not None else "Unavailable",
        )
        cards[2].metric(
            f"{driver} model position",
            f"P{int(selected['predicted_position'])}" if selected is not None and pd.notna(selected.get("predicted_position")) else "Unavailable",
        )
        cards[3].metric(
            "Prediction confidence",
            "High" if float(winner["display_probability"]) >= 0.5 else "Moderate",
        )
        st.caption("This is a pre-race model probability, not a guarantee. A probability is the model's estimated share of comparable outcomes.")

        if selected is not None:
            evidence = st.columns(4)
            evidence[0].metric("Grid position", f"P{int(selected['grid_position'])}" if pd.notna(selected.get("grid_position")) else "Unknown")
            evidence[1].metric("Pace rank", f"P{int(selected['pace_rank'])}" if pd.notna(selected.get("pace_rank")) else "Unknown")
            evidence[2].metric("Driver form", f"{selected['driver_form_score']:.2f}" if pd.notna(selected.get("driver_form_score")) else "Unknown")
            evidence[3].metric("Tyre degradation", f"{selected['degradation_s_per_lap']:.3f}s/lap" if pd.notna(selected.get("degradation_s_per_lap")) else "Unknown")

        st.subheader("All driver probabilities")
        display_columns = ["model_rank", "driver"]
        if "team" in ranked:
            display_columns.append("team")
        if prediction_probability:
            display_columns.append("win_probability_pct")
        for column in ("predicted_position", "grid_position", "pace_rank", "finish_position"):
            if column in ranked:
                display_columns.append(column)
        table = ranked[display_columns].copy()
        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
            column_config={
                "win_probability_pct": st.column_config.ProgressColumn(
                    "Win probability", min_value=0, max_value=100, format="%.1f%%"
                ),
                "predicted_position": st.column_config.NumberColumn("Model finish", format="P%d"),
                "finish_position": st.column_config.NumberColumn("Actual finish", format="P%d"),
                "grid_position": st.column_config.NumberColumn("Grid", format="P%d"),
                "pace_rank": st.column_config.NumberColumn("Pace rank", format="P%d"),
            },
        )

        if not selected_prediction.empty and "finish_position" in selected_prediction:
            actual_winner = selected_prediction.loc[
                _safe_number(selected_prediction, "finish_position").idxmin(), "driver"
            ] if _safe_number(selected_prediction, "finish_position").notna().any() else None
            if actual_winner:
                st.info(f"Recorded result: **{actual_winner}** won this race. Predictions are compared with actual results only when results are available.")

with tabs[1]:
    st.header(f"Championship · {season}")
    season_results = results[results["season"].eq(season)].copy()
    if season_results.empty:
        st.info("No official championship results are available for this season.")
    else:
        points_column = "derived_points" if "derived_points" in season_results else "points"
        season_results[points_column] = pd.to_numeric(
            season_results[points_column], errors="coerce"
        ).fillna(0)
        wdc = (
            season_results.groupby("driver", as_index=False)[points_column]
            .sum()
            .rename(columns={points_column: "points"})
            .sort_values("points", ascending=False)
        )
        wcc = (
            season_results.groupby("team", as_index=False)[points_column]
            .sum()
            .rename(columns={points_column: "points"})
            .sort_values("points", ascending=False)
        )
        st.subheader("World Drivers' Championship standings")
        st.dataframe(
            wdc,
            use_container_width=True,
            hide_index=True,
            column_config={"points": st.column_config.NumberColumn("Points", format="%.1f")},
        )
        st.subheader("Constructors' Championship standings")
        st.dataframe(
            wcc,
            use_container_width=True,
            hide_index=True,
            column_config={"points": st.column_config.NumberColumn("Points", format="%.1f")},
        )
    if season == 2026 and not wdc_forecast.empty and not wcc_forecast.empty:
        st.divider()
        st.subheader("2026 WDC/WCC forecast")
        st.caption(
            f"Forecast loaded from `{Path(forecast_source).name}`. "
            "This is a forecast checkpoint, not a recalculation from the selected circuit."
        )
        forecast_driver = wdc_forecast.copy()
        forecast_driver["win_probability_pct"] = forecast_driver["win_probability"] * 100
        forecast_driver["top_three_probability_pct"] = forecast_driver["top_three_probability"] * 100
        st.dataframe(
            forecast_driver[["driver", "current_points", "expected_final_points", "win_probability_pct", "top_three_probability_pct"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "win_probability_pct": st.column_config.ProgressColumn("WDC win chance", min_value=0, max_value=100, format="%.1f%%"),
                "top_three_probability_pct": st.column_config.ProgressColumn("Top-three chance", min_value=0, max_value=100, format="%.1f%%"),
            },
        )
        forecast_constructor = wcc_forecast.copy()
        forecast_constructor["win_probability_pct"] = forecast_constructor["win_probability"] * 100
        st.dataframe(
            forecast_constructor[["team", "current_points", "expected_final_points", "win_probability_pct"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "win_probability_pct": st.column_config.ProgressColumn("WCC win chance", min_value=0, max_value=100, format="%.1f%%"),
            },
        )

with tabs[2]:
    st.header(f"Model comparison · {season}")
    selected_comparison = comparison[comparison["season"].eq(season)].copy()
    if selected_comparison.empty:
        st.info("No walk-forward comparison artifact is available yet.")
    else:
        summary = comparison_summary(selected_comparison)
        cards = st.columns(3)
        total_races = selected_comparison["race_id"].nunique()
        winner_rows = selected_comparison[selected_comparison["finish_position"].eq(1)]
        predicted = selected_comparison[selected_comparison["predicted_position"].eq(1)]
        correct = predicted.merge(
            winner_rows[["race_id", "driver"]], on=["race_id", "driver"], how="inner"
        )["race_id"].nunique()
        cards[0].metric("Overall winner accuracy", f"{100 * correct / total_races:.1f}%")
        cards[1].metric("Races evaluated", f"{total_races:,}")
        cards[2].metric("Drivers evaluated", f"{selected_comparison['driver'].nunique():,}")
        st.subheader("Accuracy by season")
        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True,
            column_config={
                "winner_accuracy_pct": st.column_config.ProgressColumn("Winner accuracy", min_value=0, max_value=100, format="%.1f%%"),
                "top_three_overlap_pct": st.column_config.ProgressColumn("Top-three overlap", min_value=0, max_value=100, format="%.1f%%"),
                "winner_log_loss": st.column_config.NumberColumn("Winner log loss", format="%.3f"),
            },
        )
        st.subheader("Predicted vs actual winners")
        race_rows = []
        for (race_season, race_id, race_track), group in selected_comparison.groupby(
            ["season", "race_id", "track"], sort=True
        ):
            predicted_row = group.loc[group["predicted_position"].idxmin()]
            actual_row = group.loc[group["finish_position"].idxmin()]
            race_rows.append(
                {
                    "season": race_season,
                    "race_id": race_id,
                    "track": race_track,
                    "predicted_winner": predicted_row["driver"],
                    "actual_winner": actual_row["driver"],
                    "winner_probability": predicted_row["race_probability"],
                }
            )
        race_compare = pd.DataFrame(race_rows)
        race_compare["correct"] = race_compare["predicted_winner"].eq(race_compare["actual_winner"])
        st.dataframe(race_compare, use_container_width=True, hide_index=True)

with tabs[3]:
    st.header(f"Telemetry · {driver} at {track} · {season}")
    driver_laps = track_laps[track_laps["driver"].eq(driver)].copy()
    valid_driver_laps = driver_laps[driver_laps["valid"] & driver_laps["lap_time_s"].notna()].copy()
    session_choices = ["All sessions"] + sorted(driver_laps["session_type"].unique())
    session_choice = st.selectbox("Session", session_choices, key="telemetry_session")
    if session_choice != "All sessions":
        driver_laps = driver_laps[driver_laps["session_type"].eq(session_choice)]
        valid_driver_laps = valid_driver_laps[valid_driver_laps["session_type"].eq(session_choice)]

    metrics = st.columns(5)
    metrics[0].metric("Valid laps", f"{len(valid_driver_laps):,}")
    metrics[1].metric("Best lap", f"{valid_driver_laps['lap_time_s'].min():.3f}s" if not valid_driver_laps.empty else "Unavailable")
    metrics[2].metric("Median lap", f"{valid_driver_laps['lap_time_s'].median():.3f}s" if not valid_driver_laps.empty else "Unavailable")
    metrics[3].metric("Tyre compounds", f"{driver_laps['compound'].dropna().nunique()}" if "compound" in driver_laps else "0")
    metrics[4].metric("Pit laps", f"{int(driver_laps['is_pit_lap'].fillna(False).sum())}" if "is_pit_lap" in driver_laps else "0")

    if valid_driver_laps.empty:
        st.info("No valid timed laps are available for this selection.")
    else:
        st.subheader("Lap pace")
        pace = valid_driver_laps.sort_values(["session_id", "lap_no"])[["lap_no", "lap_time_s", "session_id"]].copy()
        pace["session"] = pace["session_id"].str.replace(r"^.*_", "", regex=True)
        pace_chart = pace.pivot_table(index="lap_no", columns="session", values="lap_time_s", aggfunc="median")
        st.line_chart(pace_chart, y_label="Lap time (seconds)")
        st.caption("Lower is faster. Each line is the median timed lap at that lap number for a session type.")

        chart_columns = st.columns(2)
        with chart_columns[0]:
            st.subheader("Tyre age and pace")
            tyre = valid_driver_laps[["lap_no", "tyre_age", "lap_time_s"]].dropna().sort_values("lap_no")
            st.line_chart(tyre.set_index("lap_no")[["lap_time_s", "tyre_age"]], y_label="Value")
        with chart_columns[1]:
            st.subheader("Fuel and weather")
            weather_columns = [column for column in ("fuel_kg", "track_temp", "air_temp") if column in driver_laps]
            weather = driver_laps.sort_values("lap_no").drop_duplicates(["session_id", "lap_no"])
            if weather_columns:
                st.line_chart(weather.set_index("lap_no")[weather_columns], y_label="Recorded / estimated value")
            else:
                st.info("Fuel and weather fields are not available.")

        st.subheader("Telemetry summary by session")
        summary = (
            driver_laps.groupby(["session_type", "session_id"], as_index=False)
            .agg(
                laps=("lap_no", "count"),
                valid_laps=("valid", "sum"),
                best_lap_s=("lap_time_s", "min"),
                median_lap_s=("lap_time_s", "median"),
                compounds=("compound", "nunique"),
                wet_lap_fraction=("track_status", lambda values: (values.astype(str) != "1").mean()),
            )
            .sort_values("session_id")
        )
        st.dataframe(summary, use_container_width=True, hide_index=True)

with tabs[4]:
    st.header(f"Season overview · {season}")
    season_valid = season_laps[season_laps["valid"] & season_laps["lap_time_s"].notna()]
    cards = st.columns(4)
    cards[0].metric("Circuits", f"{season_laps['track'].nunique():,}")
    cards[1].metric("Drivers", f"{season_laps['driver'].nunique():,}")
    cards[2].metric("Valid laps", f"{len(season_valid):,}")
    cards[3].metric("Sessions", f"{season_laps['session_id'].nunique():,}")
    if not season_valid.empty:
        trend = season_valid.groupby("track", as_index=False).agg(
            median_lap_s=("lap_time_s", "median"), valid_laps=("lap_time_s", "size")
        ).sort_values("median_lap_s")
        st.subheader("Circuit pace overview")
        st.bar_chart(trend.set_index("track")["median_lap_s"], horizontal=True)
        st.caption("This is a descriptive comparison. Lap times are not regulation-adjusted across seasons or circuits.")
    else:
        st.info("No valid laps are available for this season.")

with tabs[5]:
    st.header("Data quality and model limitations")
    valid_rate = laps["valid"].mean() * 100
    estimated_fuel = (laps["fuel_provenance"].eq("estimated")).mean() * 100 if "fuel_provenance" in laps else 0
    cards = st.columns(3)
    cards[0].metric("Valid lap rate", f"{valid_rate:.1f}%")
    cards[1].metric("Estimated fuel rows", f"{estimated_fuel:.1f}%")
    cards[2].metric("Prediction rows", f"{len(predictions):,}")
    st.write(
        "APEX excludes invalid and pit-influenced laps from pace charts. Fuel is estimated rather than measured, "
        "so fuel-adjusted conclusions are directional. Prediction probabilities are historical model outputs and "
        "should be read with the training cutoff and data quality in mind."
    )
    if "exclusion_reason" in laps:
        reasons = laps.loc[~laps["valid"], "exclusion_reason"].fillna("unspecified").value_counts().rename_axis("reason").to_frame("laps")
        st.subheader("Excluded lap reasons")
        st.dataframe(reasons, use_container_width=True)
