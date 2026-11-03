#!/usr/bin/env python
"""Explore and explain processed FastF1 lap data with Streamlit."""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from apex.data_io import load_processed
from apex.features.winner_features import attach_race_results, build_pre_race_features


st.set_page_config(page_title="APEX FastF1 Data", layout="wide")
st.title("APEX race intelligence dashboard")
st.caption("A simple way to understand pace, tyre behaviour, data quality, and the next opportunities for the project.")


@st.cache_data(show_spinner="Loading processed lap data...")
def load_data(signature: tuple[tuple[str, int], ...]) -> pd.DataFrame:
    return load_processed()


@st.cache_data(show_spinner="Building race summaries from practice and qualifying...")
def load_race_summary(
    lap_signature: tuple[tuple[str, int], ...],
    result_signature: tuple[tuple[str, int], ...],
) -> pd.DataFrame:
    laps = load_processed()
    result_files = [
        Path("data/processed/race_results_2018_2025.parquet"),
        Path("data/processed/race_results_2026.parquet"),
    ]
    results = pd.concat([pd.read_parquet(path) for path in result_files if path.exists()], ignore_index=True)
    if results.empty:
        return pd.DataFrame()
    features = build_pre_race_features(laps)
    joined = attach_race_results(features, results)
    joined["actual_label"] = joined["finish_position"].map(lambda value: f"P{int(value)}" if value <= 3 else "")
    return joined


try:
    processed_files = sorted(Path("data/processed").glob("laps_*.parquet"))
    signature = tuple((str(path), path.stat().st_mtime_ns) for path in processed_files)
    data = load_data(signature)
except FileNotFoundError as error:
    st.error(str(error))
    st.info("Create processed files first with scripts/build_dataset.py.")
    st.stop()

if data.empty:
    st.warning("Processed data is empty.")
    st.stop()

result_files = sorted(Path("data/processed").glob("race_results*.parquet"))
result_signature = tuple((str(path), path.stat().st_mtime_ns) for path in result_files)
race_summary = load_race_summary(signature, result_signature) if result_files else pd.DataFrame()

st.sidebar.success(f"{data['season'].nunique()} seasons loaded")
with st.sidebar:
    st.header("Filters")
    seasons = st.multiselect(
        "Season",
        options=sorted(data["season"].dropna().unique()),
        default=sorted(data["season"].dropna().unique()),
    )
    tracks = st.multiselect(
        "Track",
        options=sorted(data["track"].dropna().unique()),
        default=[],
        placeholder="All tracks",
    )
    drivers = st.multiselect(
        "Driver",
        options=sorted(data["driver"].dropna().unique()),
        default=[],
        placeholder="All drivers",
    )
    compounds = st.multiselect(
        "Compound",
        options=sorted(data["compound"].dropna().unique()),
        default=[],
        placeholder="All compounds",
    )
    valid_only = st.checkbox("Valid laps only", value=False)
    search = st.text_input("Search session or track")

filtered = data[data["season"].isin(seasons)]
if tracks:
    filtered = filtered[filtered["track"].isin(tracks)]
if drivers:
    filtered = filtered[filtered["driver"].isin(drivers)]
if compounds:
    filtered = filtered[filtered["compound"].isin(compounds)]
if valid_only:
    filtered = filtered[filtered["valid"]]
if search:
    needle = search.casefold()
    mask = (
        filtered["track"].astype(str).str.casefold().str.contains(needle, na=False)
        | filtered["session_id"].astype(str).str.casefold().str.contains(needle, na=False)
    )
    filtered = filtered[mask]

metric_columns = st.columns(4)
metric_columns[0].metric("Rows", f"{len(filtered):,}")
metric_columns[1].metric("Sessions", f"{filtered['session_id'].nunique():,}")
metric_columns[2].metric("Drivers", f"{filtered['driver'].nunique():,}")
metric_columns[3].metric("Seasons", f"{filtered['season'].nunique():,}")

if filtered.empty:
    st.warning("No laps match these filters. Try clearing one or more selections.")
    st.stop()

valid = filtered[filtered["valid"]].copy()
tabs = st.tabs(["Overview", "Compare performance", "Grand Prix top 3", "Data explorer", "Future prospects"])

with tabs[0]:
    st.subheader("What is happening?")
    valid_rate = filtered["valid"].mean() * 100
    fuel_estimated = (filtered["fuel_provenance"] == "estimated").mean() * 100
    if valid.empty:
        st.info("There are no valid laps in this selection, so performance insights are unavailable.")
    else:
        fastest = valid.loc[valid["lap_time_s"].idxmin()]
        typical_lap = valid["lap_time_s"].median()
        insight_columns = st.columns(2)
        insight_columns[0].metric("Typical valid lap", f"{typical_lap:.3f} s")
        insight_columns[1].metric("Fastest valid lap", f"{fastest['lap_time_s']:.3f} s")
        st.caption(
            f"Fastest recorded lap: **{fastest['driver']}** at **{fastest['track']}** "
            f"({fastest['season']}, {fastest['compound']})."
        )

        trend = (
            valid.groupby("season", as_index=False)
            .agg(median_lap_time_s=("lap_time_s", "median"), laps=("lap_time_s", "size"))
            .sort_values("season")
            .set_index("season")
        )
        st.subheader("Pace trend by season")
        st.line_chart(trend["median_lap_time_s"], y_label="Median valid lap time (seconds)")
        st.caption("Lower is faster. This is a descriptive trend, not a regulation-adjusted comparison.")

    quality_columns = st.columns(2)
    with quality_columns[0]:
        st.subheader("Data quality")
        st.write(
            f"**{valid_rate:.1f}%** of selected laps are valid for pace analysis. "
            f"Invalid laps remain available for audit and are not silently discarded."
        )
        reasons = (
            filtered.loc[~filtered["valid"], "exclusion_reason"]
            .fillna("unspecified")
            .value_counts()
            .rename_axis("reason")
            .to_frame("laps")
        )
        if not reasons.empty:
            st.dataframe(reasons, use_container_width=True)
    with quality_columns[1]:
        st.subheader("How to read this")
        st.write(
            "Use **Overview** for the big picture, **Compare performance** to find patterns, "
            "and **Data explorer** when you need the underlying rows."
        )
        st.info(
            f"Fuel mass is **estimated for {fuel_estimated:.1f}%** of selected rows. "
            "Treat fuel-related conclusions as directional until measured fuel data is available."
        )

with tabs[1]:
    st.subheader("Compare drivers, tyres, and tracks")
    if valid.empty:
        st.info("Select a wider range to include valid laps.")
    else:
        normalized = valid.copy()
        track_baseline = normalized.groupby("track")["lap_time_s"].transform("median")
        normalized["track_relative_lap_s"] = normalized["lap_time_s"] - track_baseline
        driver_summary = (
            normalized.groupby("driver", as_index=False)
            .agg(
                median_track_relative_lap_s=("track_relative_lap_s", "median"),
                best_track_relative_lap_s=("track_relative_lap_s", "min"),
                valid_laps=("lap_time_s", "size"),
                tracks=("track", "nunique"),
            )
            .query("valid_laps >= 3")
            .sort_values("median_track_relative_lap_s")
        )
        st.markdown("**Driver pace** — times are normalized to each track median; lower is better.")
        st.dataframe(
            driver_summary,
            use_container_width=True,
            hide_index=True,
            column_config={
                "median_track_relative_lap_s": st.column_config.NumberColumn("Median vs track (s)", format="%.3f"),
                "best_track_relative_lap_s": st.column_config.NumberColumn("Best vs track (s)", format="%.3f"),
            },
        )

        compare_columns = st.columns(2)
        with compare_columns[0]:
            compound_summary = (
                normalized.groupby("compound", as_index=False)
                .agg(
                    median_track_relative_lap_s=("track_relative_lap_s", "median"),
                    valid_laps=("lap_time_s", "size"),
                )
                .sort_values("median_track_relative_lap_s")
                .set_index("compound")
            )
            st.markdown("**Tyre compound pace vs track baseline**")
            st.bar_chart(compound_summary["median_track_relative_lap_s"], horizontal=True)
        with compare_columns[1]:
            track_summary = (
                valid.groupby("track", as_index=False)
                .agg(median_lap_time_s=("lap_time_s", "median"), valid_laps=("lap_time_s", "size"))
                .query("valid_laps >= 5")
                .sort_values("median_lap_time_s")
                .head(10)
                .set_index("track")
            )
            st.markdown("**Fastest selected tracks**")
            st.bar_chart(track_summary["median_lap_time_s"], horizontal=True)

with tabs[2]:
    st.subheader("Year-wise Grand Prix top three")
    st.caption(
        "Each row shows the official top three and the pre-race evidence available from practice and qualifying. "
        "Practice/qualifying metrics are descriptive and do not use race laps."
    )
    if race_summary.empty:
        st.info("Build official results first with scripts/build_race_results.py.")
    else:
        summary = race_summary[
            race_summary["season"].isin(seasons)
            & (race_summary["track"].isin(tracks) if tracks else True)
        ].copy()
        top_three = summary[summary["finish_position"] <= 3].copy()
        top_three["driver_summary"] = top_three.apply(
            lambda row: (
                f"P{int(row['finish_position'])} {row['driver']} "
                f"(pace rank {int(row['pace_rank'])}, "
                f"{row['fuel_corrected_median_lap_s']:.2f}s corrected median)"
            ),
            axis=1,
        )
        grand_prix = (
            top_three.sort_values(["season", "race_id", "finish_position"])
            .groupby(["season", "race_id", "track"], as_index=False)
            .agg(
                top_three=("driver_summary", " | ".join),
                practice_laps=("practice_laps", "max"),
                qualifying_laps=("qualifying_laps", "max"),
                practice_sessions=("practice_sessions", "max"),
                qualifying_sessions=("qualifying_sessions", "max"),
            )
            .sort_values(["season", "race_id"], ascending=[False, False])
        )
        st.dataframe(
            grand_prix,
            use_container_width=True,
            hide_index=True,
            column_config={
                "top_three": st.column_config.TextColumn("Official top 3 + pre-race pace"),
                "practice_laps": st.column_config.NumberColumn("Practice laps"),
                "qualifying_laps": st.column_config.NumberColumn("Qualifying laps"),
            },
        )
        st.download_button(
            "Download Grand Prix top-three table",
            data=grand_prix.to_csv(index=False).encode("utf-8"),
            file_name="apex_grand_prix_top_three.csv",
            mime="text/csv",
        )

with tabs[3]:
    st.subheader("Inspect the underlying laps")
    columns = st.multiselect(
        "Choose columns to show",
        options=list(data.columns),
        default=["season", "session_id", "track", "driver", "lap_no", "lap_time_s", "valid", "compound"],
    )
    st.dataframe(
        filtered[columns] if columns else filtered.iloc[:, 0:0],
        use_container_width=True,
        hide_index=True,
        column_config={
            "lap_time_s": st.column_config.NumberColumn("Lap time (s)", format="%.3f"),
            "fuel_kg": st.column_config.NumberColumn("Fuel (kg, estimated)", format="%.2f"),
            "track_temp": st.column_config.NumberColumn("Track temp", format="%.1f"),
            "air_temp": st.column_config.NumberColumn("Air temp", format="%.1f"),
        },
    )
    st.download_button(
        "Download filtered CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="apex_filtered_laps.csv",
        mime="text/csv",
    )

with tabs[4]:
    st.subheader("Future prospects for APEX")
    st.write(
        "APEX already provides a strong foundation: verified FastF1 data, a year-wise ML protocol, "
        "strategy modelling, and an aero-surrogate path. The most valuable next steps are:"
    )
    roadmap = pd.DataFrame(
        [
            ("1", "Add grid, team, reliability, and weather features", "The repeated holdout misses are mostly race-context misses, not raw pace misses."),
            ("2", "Use rolling driver/team form and circuit history", "Improve transfer across regulation eras and changing team performance."),
            ("3", "Calibrate probabilities and add top-three metrics", "Make scores interpretable and optimize for the real user goal."),
            ("4", "Add tyre strategy and Monte Carlo race simulation", "Convert pace and degradation into pit-stop and incident-aware outcomes."),
            ("5", "Run ablations and leakage audits every fold", "Keep each feature tied to information available before the race."),
        ],
        columns=["Priority", "Accuracy improvement", "Why it matters"],
    )
    st.dataframe(roadmap, use_container_width=True, hide_index=True)
    st.success(
        "The clearest product direction is an engineer decision cockpit: "
        "what pace is available, why it changes, and which strategy is most promising."
    )
