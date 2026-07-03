"""Merge index-aligned sensor and questionnaire CSVs into one wide file."""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import force_2d, line_locate_point
from shapely.geometry import Point

from _paths import OUTPUTS, RAW_DATA_DIR


OUT = OUTPUTS
INDEX_FILE = OUT / "00_index_10sec.csv"
OUT_FILE = OUT / "mergeddata_all.csv"
OUT_FILE_11 = OUT / "merged_all_11participants.csv"
MERGE_KEYS = ["ParticipantID", "PhaseID", "Datetime"]
PHASES_5 = {"BikeU", "WalkU", "BikeG", "WalkG", "Tram"}
GPKG_DIR = RAW_DATA_DIR / "Experiment path"
PHASE_GPKG = {p: f"{p}.gpkg" for p in PHASES_5}
VALID_11_PARTICIPANTS = {f"P{i}" for i in [4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]}
CANONICAL_INPUT_NAMES = [
    "00_index_10sec.csv",
    "01_empatica_corrected_10sec.csv",
    "02_ucm_10sec.csv",
    "03_atmo_lys_merged.csv",
    "04_eyetracker_10sec.csv",
    "05_questionnaires_merged_scored.csv",
]


def add_pct_complete(df: pd.DataFrame) -> pd.DataFrame:
    df["pct_complete"] = np.nan

    for phase, gpkg_file in PHASE_GPKG.items():
        mask = df["PhaseID"] == phase
        n_rows = int(mask.sum())
        if n_rows == 0:
            continue

        if phase == "Tram":
            for pid in df.loc[mask, "ParticipantID"].unique():
                pid_mask = mask & (df["ParticipantID"] == pid)
                ts = df.loc[pid_mask, "Datetime"]
                if len(ts) < 2:
                    continue
                t_min, t_max = ts.min(), ts.max()
                span = (t_max - t_min).total_seconds()
                if span <= 0:
                    continue
                pcts = ((ts - t_min).dt.total_seconds() / span * 100).round(1)
                df.loc[pid_mask, "pct_complete"] = pcts.values
            valid = df.loc[mask, "pct_complete"].dropna()
            rng = f"{valid.min():.1f}% -> {valid.max():.1f}%" if len(valid) else "N/A"
            print(f"  [pct_complete] Tram (time-based): {n_rows} rows, {rng}")
            continue

        gpkg_path = GPKG_DIR / gpkg_file
        if not gpkg_path.exists():
            print(f"  [pct_complete] GPKG not found for {phase}, skipping.")
            continue

        path_gdf = gpd.read_file(gpkg_path)
        line = force_2d(path_gdf.geometry.iloc[0])
        line_length = line.length

        pct_vals = []
        for idx in df.index[mask]:
            lat = df.at[idx, "GPS_lat"] if "GPS_lat" in df.columns else np.nan
            lon = df.at[idx, "GPS_lon"] if "GPS_lon" in df.columns else np.nan
            if pd.isna(lat) or pd.isna(lon):
                pct_vals.append(np.nan)
                continue
            dist_along = line_locate_point(line, Point(lon, lat))
            pct_vals.append(round((dist_along / line_length) * 100, 1))

        df.loc[mask, "pct_complete"] = pct_vals
        valid = [v for v in pct_vals if not np.isnan(v)]
        rng = f"{min(valid):.1f}% -> {max(valid):.1f}%" if valid else "N/A"
        skipped = sum(1 for v in pct_vals if np.isnan(v))
        print(f"  [pct_complete] {phase}: {n_rows} rows, {rng} (skipped {skipped} NaN GPS)")

    return df


def script_order(path: Path) -> str:
    stem = path.stem
    return stem.split("_")[0] if stem and stem[0].isdigit() and "_" in stem else "99"


def merge_input_files() -> list[Path]:
    input_names = list(CANONICAL_INPUT_NAMES)
    pending_empatica = OUT / "01_empatica_corrected_10sec_PENDING_CLOSE_OPEN_FILE.csv"
    if pending_empatica.exists():
        input_names[input_names.index("01_empatica_corrected_10sec.csv")] = pending_empatica.name
    return [OUT / name for name in input_names if (OUT / name).exists()]


def display_name(path: Path) -> str:
    return path.stem.replace("_PENDING_CLOSE_OPEN_FILE", "")


def clean_sensor_columns(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if name == "01_empatica_corrected_10sec":
        # heart_rate is the same signal as empatica__pulse_rate_bpm; keep the clearer analysis name.
        df = df.drop(columns=["empatica__pulse_rate_bpm"], errors="ignore")
        # Empatica wrist temperature is skin temperature, not ambient temperature.
        df = df.rename(columns={"temperature": "skin_temperature"})
    return df


def main() -> None:
    if not INDEX_FILE.exists():
        print("Run 00_index_build.py first.")
        return

    index = pd.read_csv(INDEX_FILE, low_memory=False)
    index["Datetime"] = pd.to_datetime(index["Datetime"])
    index["PhaseID"] = index["PhaseID"].where(index["PhaseID"].notna(), "").astype(str).replace({"nan": "", "None": ""})
    print(f"Index: {len(index):,} rows")

    merged = index.copy()
    for path in merge_input_files():
        name = display_name(path)
        if name == "00_index_10sec":
            continue

        df = pd.read_csv(path, low_memory=False)
        df = clean_sensor_columns(df, name)
        if "Datetime" not in df.columns and "questionnaires" in name:
            if "PhaseID" in df.columns:
                df = df[df["PhaseID"].isin(PHASES_5)]
            df = df.drop_duplicates(subset=["ParticipantID", "PhaseID"], keep="last")
            q_cols = [c for c in df.columns if c not in ("ParticipantID", "PhaseID")]
            merged = merged.merge(df[["ParticipantID", "PhaseID"] + q_cols], on=["ParticipantID", "PhaseID"], how="left")
            print(f"  {name} (questionnaire): {len(df):,} rows -> {len(q_cols)} cols")
            continue

        if "Datetime" in df.columns and "ParticipantID" in df.columns:
            df["Datetime"] = pd.to_datetime(df["Datetime"])
            if "PhaseID" in df.columns:
                df["PhaseID"] = df["PhaseID"].where(df["PhaseID"].notna(), "").astype(str).replace({"nan": "", "None": ""})
            merge_keys = [k for k in MERGE_KEYS if k in df.columns]
            df = df.drop_duplicates(subset=merge_keys, keep="first")
            merged = merged.merge(df, on=merge_keys, how="left", suffixes=("", "_dup"))
            merged = merged.drop(columns=[c for c in merged.columns if c.endswith("_dup")])
            sensor_cols = len(df.columns) - len(merge_keys)
            print(f"  {name}: {len(df):,} rows -> {sensor_cols} cols")

    # Preserve the 00_index_10sec.csv row order exactly; downstream notebooks can sort for plotting.
    merged = merged.reset_index(drop=True)
    merged = add_pct_complete(merged)
    merged.to_csv(OUT_FILE, index=False)
    merged_11 = merged[merged["ParticipantID"].astype(str).isin(VALID_11_PARTICIPANTS)].copy()
    merged_11.to_csv(OUT_FILE_11, index=False)
    print(f"\nSaved: {OUT_FILE} - {len(merged):,} rows x {merged.shape[1]} cols")
    print(f"Saved: {OUT_FILE_11} - {len(merged_11):,} rows x {merged_11.shape[1]} cols")
    print(f"  Participants: {sorted(merged['ParticipantID'].unique())}")
    print(f"  Phases: {sorted(merged['PhaseID'].unique())}")
    print(f"  11-participant subset: {sorted(merged_11['ParticipantID'].unique())}")


if __name__ == "__main__":
    main()
