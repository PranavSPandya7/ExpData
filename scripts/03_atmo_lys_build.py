"""Atmo/LYS build: merge four sensors and align minute readings to the 10-sec index."""

import warnings; warnings.filterwarnings("default")
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import (
    KEY_FILE,
    OUTPUTS,
    RAW_ATMO_DIR,
    assert_sensor_folder_clean,
    load_key_unique,
    setup_build_warning_log,
)
setup_build_warning_log(__file__)

RAW_DIR = RAW_ATMO_DIR
INDEX_FILE = OUTPUTS / "00_index_10sec.csv"
UTC_OFFSET = pd.Timedelta(hours=2)
FORCE_RERUN = True

PHASES = ["BikeU", "WalkU", "BikeG", "WalkG", "Tram"]
SENSORS = [
    ("LYS1", "LYS1", "LYS1"),
    ("LYS2", "LYS2", "LYS2"),
    ("atmo_left", "Atmo_left", "atmotube_left"),
    ("atmo_right", "Atmo_right", "atmotube_right"),
]

def parse_key_date(d: str) -> str:
    return datetime.strptime(f"{d}-2025", "%d-%b-%Y").strftime("%Y-%m-%d")


def build_phase_windows(key: pd.DataFrame) -> dict:
    """Return {(pid, phase): (start_ts, end_ts)} in Brussels local time."""
    windows = {}
    for _, row in key.iterrows():
        if pd.isna(row["Participant_ID"]):
            continue
        pid = int(row["Participant_ID"])
        date = parse_key_date(str(row["Date"]))
        for ph in PHASES:
            s_col = f"{ph}_start"
            e_col = f"{ph}_end"
            if s_col not in row or pd.isna(row[s_col]) or pd.isna(row[e_col]):
                continue
            start = pd.Timestamp(f"{date} {row[s_col]}") + UTC_OFFSET
            end = pd.Timestamp(f"{date} {row[e_col]}") + UTC_OFFSET
            if end < start:
                end += pd.Timedelta(days=1)
            windows[(pid, ph)] = (start, end)
    return windows


def participant_bounds(pid: int, windows: dict) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    spans = [(start, end) for (win_pid, _), (start, end) in windows.items() if win_pid == pid]
    if not spans:
        return None
    return min(start for start, _ in spans), max(end for _, end in spans)


def normalize_legacy_sensor_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize older P1-style Atmotube/LYS CSV headers to the staged schema."""
    df = df.copy()
    if "Datetime" not in df.columns:
        if "timestamp" in df.columns:
            # Some processed P1 LYS timestamps include a misleading timezone suffix;
            # keep the recorded wall-clock time so it aligns with the shifted key.
            ts = df["timestamp"].astype(str).str.replace(r"([+-]\d{2}:\d{2})$", "", regex=True)
            df["Datetime"] = pd.to_datetime(ts, errors="coerce")
        elif "Date" in df.columns and "Time" in df.columns:
            df["Datetime"] = pd.to_datetime(
                df["Date"].astype(str) + " " + df["Time"].astype(str),
                dayfirst=True,
                errors="coerce",
            )
        elif "Date" in df.columns:
            df["Datetime"] = pd.to_datetime(df["Date"], errors="coerce")

    rename = {}
    for col in df.columns:
        clean = col.lower().replace("Â", "").replace("Ë", "").replace("š", "").strip()
        if clean.startswith("pm1,"):
            rename[col] = "atmotube_pm1"
        elif clean.startswith("pm2.5,"):
            rename[col] = "atmotube_pm2.5"
        elif clean.startswith("pm10,"):
            rename[col] = "atmotube_pm10"
        elif clean.startswith("temperature"):
            rename[col] = "atmotube_temperature"
        elif clean.startswith("humidity"):
            rename[col] = "atmotube_humidity"
        elif clean == "lux":
            rename[col] = "lys_lux"
        elif clean == "kelvin":
            rename[col] = "lys_kelvin"
        elif clean == "rgbr":
            rename[col] = "lys_rgbr"
        elif clean == "rgbg":
            rename[col] = "lys_rgbg"
        elif clean == "rgbb":
            rename[col] = "lys_rgbb"
        elif clean == "rgbir":
            rename[col] = "lys_rgbir"
        elif clean == "movement":
            rename[col] = "lys_movement"
        elif clean == "medi":
            rename[col] = "lys_medi"
        elif clean == "r'":
            rename[col] = "lys_r'"
        elif clean == "g'":
            rename[col] = "lys_g'"
        elif clean == "b'":
            rename[col] = "lys_b'"
    if rename:
        df = df.rename(columns=rename)
    return df


def load_sensor_file(pid: int, suffix: str, prefix: str, windows: dict) -> pd.DataFrame:
    """Load one staged sensor CSV and align it to 10-second bins."""
    fp = RAW_DIR / f"{pid}_{suffix}.csv"
    if not fp.exists():
        print(f"    SKIP {suffix}: file not found")
        return pd.DataFrame()

    df = normalize_legacy_sensor_columns(pd.read_csv(fp))
    if df.empty or "Datetime" not in df.columns:
        print(f"    SKIP {suffix}: empty or no Datetime column")
        return pd.DataFrame()

    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df = df.dropna(subset=["Datetime"])
    if df.empty:
        print(f"    SKIP {suffix}: no parseable timestamps")
        return pd.DataFrame()

    bounds = participant_bounds(pid, windows)
    if bounds is None:
        print(f"    SKIP {suffix}: no key window for P{pid}")
        return pd.DataFrame()
    span_start, span_end = bounds
    df = df[(df["Datetime"] >= span_start) & (df["Datetime"] <= span_end)].copy()
    if df.empty:
        print(f"    SKIP {suffix}: no rows from {span_start} to {span_end}")
        return pd.DataFrame()

    df["PhaseID"] = "reststop"
    for ph in PHASES:
        if (pid, ph) not in windows:
            continue
        start, end = windows[(pid, ph)]
        mask = (df["Datetime"] >= start) & (df["Datetime"] <= end)
        df.loc[mask, "PhaseID"] = ph

    meas_cols = [
        c for c in df.columns
        if c not in ("ParticipantID", "Datetime", "PhaseID", "Date", "Time", "timestamp", "sensor")
        and c not in ("VOC, ppm", "AQS", "Pressure, mbar", "Latitude", "Longitude")
    ]
    rename = {c: f"{prefix}__{c}" for c in meas_cols}
    df = df.rename(columns=rename)

    df = df.set_index("Datetime").sort_index()
    prefixed_cols = [rename[c] for c in meas_cols]
    numeric_cols = [c for c in prefixed_cols if c in df.columns and df[c].dtype.kind in ("i", "f")]
    if not numeric_cols:
        print(f"    SKIP {suffix}: no numeric signal columns")
        return pd.DataFrame()

    minute_key = df.index.floor("min")
    minute_values = df[numeric_cols].groupby(minute_key).median()
    minute_phase = df["PhaseID"].groupby(minute_key).agg(
        lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]
    )
    expanded_values = []
    expanded_phases = []
    for offset_s in range(0, 60, 10):
        offset = pd.Timedelta(seconds=offset_s)
        value_block = minute_values.copy()
        value_block.index = value_block.index + offset
        phase_block = minute_phase.copy()
        phase_block.index = phase_block.index + offset
        expanded_values.append(value_block)
        expanded_phases.append(phase_block)
    resampled = pd.concat(expanded_values).sort_index()
    resampled["PhaseID"] = pd.concat(expanded_phases).sort_index().reindex(resampled.index).fillna("reststop")
    resampled.index.name = "Datetime"
    resampled = resampled.reset_index()
    resampled["Datetime"] = resampled["Datetime"].dt.floor("10s")
    resampled.insert(0, "ParticipantID", f"P{pid}")

    phase_count = resampled["PhaseID"].notna().sum()
    print(f"    {suffix}: {len(resampled)} rows at 10-sec ({phase_count} in phase windows)")
    return resampled


def clean_atmo_lys(path: Path) -> None:
    """Remove out-of-range Kelvin values (physical impossibility check).

    Note: negative sensor values are NOT masked here. All 343 negatives in the
    raw lys_medi column fall outside experiment phase windows and are already
    excluded by the time-window clipping in load_sensor_file().
    """
    print(f"  Cleaning {path.name} ...")
    df = pd.read_csv(path)
    numeric_cols = [
        col
        for col in df.columns
        if col not in ("ParticipantID", "PhaseID", "Datetime", "Date")
        and not col.startswith("QC__")
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    kelvin_cols = [col for col in numeric_cols if col.endswith("__lys_kelvin")]
    kelvin_bad = 0
    for col in kelvin_cols:
        bad = df[col].notna() & ((df[col] < 1000) | (df[col] > 12000))
        kelvin_bad += int(bad.sum())
        df.loc[bad, col] = pd.NA
    df.to_csv(path, index=False)
    print(f"  Cleaned: set {kelvin_bad} out-of-range Kelvin values to NaN")


def derive_lys_categories(path: Path) -> None:
    print(f"  Deriving LYS categories from {path.name} ...")
    df = pd.read_csv(path)
    added = []
    for sensor in ("LYS1", "LYS2"):
        lux_col = f"{sensor}__lys_lux"
        out_col = f"{sensor}__lux_cat"
        if lux_col not in df.columns:
            continue
        lux = pd.to_numeric(df[lux_col], errors="coerce")
        df[out_col] = pd.cut(
            lux,
            bins=[-float("inf"), 1000, 10000, float("inf")],
            labels=["Shaded", "Daylight", "Bright daylight"],
        )
        added.append(out_col)
    df.to_csv(path, index=False)
    if added:
        print(f"  Added {len(added)} columns: {added}")


def main() -> None:
    out_path = OUTPUTS / "03_atmo_lys_merged.csv"
    if out_path.exists() and not FORCE_RERUN:
        print(f"Output file already exists: {out_path}. Skipping Atmo & LYS build (FORCE_RERUN=False).")
        return

    if not KEY_FILE.exists():
        print(f"ERROR: Key file not found: {KEY_FILE}")
        return

    assert_sensor_folder_clean("atmo_lys", RAW_DIR)

    key = load_key_unique(KEY_FILE)
    windows = build_phase_windows(key)
    pids = sorted(key["Participant_ID"].dropna().astype(int).unique().tolist())

    existing = list(RAW_DIR.glob("*.csv"))
    if not existing:
        print(f"ERROR: no staged Atmo/LYS files found in {RAW_DIR}")
        return
    print(f"  Using staged Atmo/LYS rawdata ({len(existing)} files in {RAW_DIR})")
    print(f"Key file: {len(pids)} participants, {len(windows)} phase windows")

    all_frames = []
    for pid in pids:
        key_date = parse_key_date(str(key.loc[key["Participant_ID"] == pid, "Date"].iloc[0]))
        print(f"Processing P{pid} ({key_date}) ...")

        sensor_frames = []
        missing_sensors = []
        for sensor_id, suffix, prefix in SENSORS:
            sf = load_sensor_file(pid, suffix, prefix, windows)
            if sf.empty:
                missing_sensors.append(sensor_id)
                continue
            sensor_frames.append(sf.drop(columns=["PhaseID"], errors="ignore").set_index(["ParticipantID", "Datetime"]))

        if not sensor_frames:
            print(f"  => P{pid}: no data")
            continue
        if missing_sensors:
            print(f"  => P{pid}: missing sensors {missing_sensors}")

        merged = sensor_frames[0]
        for sf in sensor_frames[1:]:
            merged = merged.join(sf, how="outer")
        merged = merged.reset_index()
        all_frames.append(merged)
        print(f"  => P{pid}: {len(merged):,} merged rows")

    if not all_frames:
        print("\nWARNING: No data processed for any participant.")
        return

    out = pd.concat(all_frames, ignore_index=True).sort_values(["ParticipantID", "Datetime"]).reset_index(drop=True)

    if INDEX_FILE.exists():
        idx = pd.read_csv(INDEX_FILE, low_memory=False)
        idx["Datetime"] = pd.to_datetime(idx["Datetime"])
        idx["ParticipantID"] = idx["ParticipantID"].astype(str)
        out["Datetime"] = pd.to_datetime(out["Datetime"]).dt.floor("10s")
        out["ParticipantID"] = out["ParticipantID"].astype(str)
        out = idx[["ParticipantID", "PhaseID", "Datetime", "Date"]].merge(
            out,
            on=["ParticipantID", "Datetime"],
            how="left",
        )
        n_missing = int(out.iloc[:, 4:].isna().all(axis=1).sum())
        print(f"  Index join: {len(idx):,} index slots, {n_missing:,} have no sensor data")
    else:
        print(f"  NOTE: {INDEX_FILE.name} not found")

    out.to_csv(out_path, index=False)
    print(f"\nSaved {len(out):,} rows => {out_path}")
    print(f"Participants: {sorted(out['ParticipantID'].unique())}")
    print(f"Phases:       {sorted(out['PhaseID'].dropna().unique())}")
    print(f"Columns ({len(out.columns)}): {out.columns.tolist()}")

    clean_atmo_lys(out_path)
    derive_lys_categories(out_path)


if __name__ == "__main__":
    main()
