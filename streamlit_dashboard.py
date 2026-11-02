#!/usr/bin/env python
"""Explore processed FastF1 lap data with Streamlit."""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from apex.data_io import load_processed


st.set_page_config(page_title="APEX FastF1 Data", layout="wide")
st.title("APEX FastF1 lap data")
st.caption("Interactive view of verified, processed parquet data in data/processed.")


@st.cache_data(show_spinner="Loading processed lap data...")
def load_data(signature: tuple[tuple[str, int], ...]) -> pd.DataFrame:
    return load_processed()


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
    columns = st.multiselect(
        "Columns",
        options=list(data.columns),
        default=list(data.columns),
    )

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

with st.expander("Coverage and data quality", expanded=True):
    coverage = (
        data.groupby("season", as_index=False)
        .agg(
            laps=("lap_no", "size"),
            sessions=("session_id", "nunique"),
            tracks=("track", "nunique"),
            drivers=("driver", "nunique"),
            valid_laps=("valid", "sum"),
        )
        .sort_values("season")
    )
    coverage["valid_%"] = (coverage["valid_laps"] / coverage["laps"] * 100).round(1)
    st.dataframe(coverage, use_container_width=True, hide_index=True)

metric_columns = st.columns(4)
metric_columns[0].metric("Rows", f"{len(filtered):,}")
metric_columns[1].metric("Sessions", f"{filtered['session_id'].nunique():,}")
metric_columns[2].metric("Drivers", f"{filtered['driver'].nunique():,}")
metric_columns[3].metric("Seasons", f"{filtered['season'].nunique():,}")

st.dataframe(
    filtered[columns] if columns else filtered.iloc[:, 0:0],
    use_container_width=True,
    hide_index=True,
    column_config={
        "lap_time_s": st.column_config.NumberColumn("Lap time (s)", format="%.3f"),
        "fuel_kg": st.column_config.NumberColumn("Fuel (kg)", format="%.2f"),
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
