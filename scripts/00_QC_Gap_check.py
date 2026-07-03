from __future__ import annotations

import re
import csv
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from _paths import KEY_FILE, OUTPUTS, RAW_DATA_DIR

warnings.filterwarnings("ignore", message="Could not infer format")

INDEX_FILE = OUTPUTS / "00_index_10sec.csv"
OUT_MISSING = OUTPUTS / "QC_gap_missing_percent_by_column.csv"
OUT_INVALID = OUTPUTS / "QC_invalid_percent_by_column.csv"

KEYS = ["ParticipantID", "PhaseID", "Datetime"]
PHASES = {"BikeU", "WalkU", "BikeG", "WalkG", "Tram"}
INPUTS = {
    "01_empatica": "01_empatica_corrected_10sec.csv",
    "02_ucm": "02_ucm_10sec.csv",
    "03_atmo_lys": "03_atmo_lys_merged.csv",
    "04_eyetracker": "04_eyetracker_10sec.csv",
}
SKIP = set(KEYS) | {"Date"}
RAW_SKIP = {"participantid", "participant_id", "participant", "pid", "phaseid", "date", "datetime", "timestamp", "time", "gps_time"}
MISSING_STRINGS = {"", "na", "nan", "none", "null", "nat", "-"}
EMPATICA_RAW_COLS = [
    "accelerometer_x",
    "accelerometer_y",
    "accelerometer_z",
    "bvp",
    "eda",
    "steps",
    "systolicPeaks_peaksTimeNanos",
    "temperature",
]

# Broad impossible-value screening ranges. Missingness is reported separately.
EXACT_RANGES = {
    "empatica__pulse_rate_bpm": (35, 220),
    "heart_rate": (35, 220),
    "empatica__eda_scl_usiemens": (0, 100),
    "eda_tonic": (0, 100),
    "eda_phasic": (-5, 5),
    "temperature": (20, 45),
    "acc_x": (-4, 4),
    "acc_y": (-4, 4),
    "acc_z": (-4, 4),
    "vector_magnitude": (0, 4),
    "hrv_td_rmssd": (5, 500),
    "hrv_td_sdnn": (10, 400),
    "hrv_fd_lf": (0, 250000),
    "hrv_fd_hf": (0, 100000),
    "hrv_fd_lf_hf_ratio": (0, 20),
    "hrv_fd_lfnu": (0, 100),
    "hrv_fd_hfnu": (0, 100),
    "hrv_fd_vlf": (0, 250000),
    "hrv_fd_total_power": (0, 500000),
    "hrv_fd_5min_available": (0, 1),
    "hrv_fd_5min_valid_rri_count": (0, 1200),
    "hrv_fd_5min_removed_pct": (0, 100),
    "hrv_fd_5min_max_gap_seconds": (0, 300),
    "hrv_fd_5min_valid_span_seconds": (0, 300),
    "GPS_lat": (50.75, 50.90),
    "GPS_lon": (4.25, 4.55),
    "GPS_alt": (0, 300),
    "GPS_HDOP": (0, 5),
    "GPS_hdg": (0, 360),
    "MAG_hdg": (0, 360),
    "GPS_speed_valid_sample_count": (0, 10),
    "GPS_speed_mps": (0, 25),
    "GPS_speed_kmh": (0, 90),
    "IO_flag": (0, 9),
    "AIR_temp": (0, 45),
    "AIR_RH": (0, 100),
    "AIR_T_bot": (0, 50),
    "AIR_T_mid": (0, 50),
    "AIR_T_top": (0, 50),
    "AH": (0, 35),
    "SUN_Gh": (-20, 1300),
    "SUN_alt": (0, 90),
    "SUN_az": (0, 360),
    "SND_dBA": (30, 120),
    "WIND_AWS": (0, 40),
    "WIND_AWA": (0, 360),
    "WIND_TWS": (0, 40),
    "WIND_TWD": (0, 360),
    "AQ_pm010": (0, 500),
    "AQ_pm025": (0, 500),
    "AQ_pm100": (0, 1000),
    "AQ_CO2": (350, 5000),
    "fp": (0, 1),
    "MRT": (-20, 80),
    "MRT_S": (-20, 90),
    "OPT": (0, 60),
    "HUMIDEX": (0, 60),
    "HUMIDEX_SR": (0, 5),
    "UTCI": (-50, 60),
    "UTCI_SR": (-5, 5),
    "PET": (-20, 70),
    "PET_SR": (-5, 5),
    "pupil_diameter_avg": (1, 8),
    "pupil_change_rate": (-1, 1),
    "in_blink": (0, 1),
    "blink_duration_s": (0.05, 1.0),
    "in_fixation": (0, 1),
    "fixation_duration_s": (0.02, 1.5),
    "fixation_rate": (0, 6),
    "in_saccade": (0, 1),
    "saccade_duration_s": (0.01, 0.30),
    "saccade_amplitude": (0, 40),
    "saccade_peak_velocity": (0, 9000),
    "gaze_velocity": (0, 5000),
    "gaze_dispersion": (0, 120),
    "distance_from_center": (0, 1000),
    "gaze_centrality": (0, 1),
    "eyelid_aperture_avg": (0, 20),
    "stress_composite": (0, 1),
}

PATTERN_RANGES = [
    (r"^IR_", -50, 90),
    (r"^IR_spot_", -50, 90),
    (r"^AQ_(NO2|O3|SO2|CO)_m\d$", 0, np.inf),
    (r"^LYS[12]__lys_lux$", 0, 120000),
    (r"^LYS[12]__lys_kelvin$", 1000, 12000),
    (r"^LYS[12]__lys_medi$", 0, 8750),
    (r"^LYS[12]__lys_movement$", 0, np.inf),
    (r"^LYS[12]__lys_[bgr]'$", 0, np.inf),
    (r"^LYS[12]__lys_rgb", 0, np.inf),
    (r"^atmotube_.*__atmotube_pm1$", 0, 250),
    (r"^atmotube_.*__atmotube_pm2\.5$", 0, 500),
    (r"^atmotube_.*__atmotube_pm10$", 0, 1000),
    (r"^atmotube_.*__atmotube_humidity$", 0, 100),
    (r"^atmotube_.*__atmotube_temperature$", 0, 45),
]


def allowed_range(col: str) -> tuple[float, float] | None:
    if col in EXACT_RANGES:
        return EXACT_RANGES[col]
    for pattern, low, high in PATTERN_RANGES:
        if re.search(pattern, col):
            return low, high
    return None


def read_index() -> pd.DataFrame:
    idx = pd.read_csv(INDEX_FILE, low_memory=False)
    idx = idx[idx["PhaseID"].astype(str).isin(PHASES)].copy()
    idx["Datetime"] = pd.to_datetime(idx["Datetime"])
    idx["ParticipantID"] = idx["ParticipantID"].astype(str)
    return idx[KEYS]


def date_to_pid() -> dict[str, str]:
    key = pd.read_csv(KEY_FILE)
    out = {}
    for _, row in key.dropna(subset=["Participant_ID", "Date"]).iterrows():
        date = pd.to_datetime(f"{row['Date']}-2025", format="%d-%b-%Y", errors="coerce")
        if pd.notna(date):
            out[date.strftime("%Y-%m-%d")] = f"P{int(row['Participant_ID'])}"
    return out


def sensor_name(path: Path) -> str:
    return path.relative_to(RAW_DATA_DIR).parts[0]


def infer_pid(path: Path, date_map: dict[str, str]) -> str | None:
    parts = path.relative_to(RAW_DATA_DIR).parts
    if parts and parts[0] == "01_empatica" and len(parts) > 1:
        return date_map.get(parts[1])
    m = re.match(r"(\d+)_", path.name)
    if m:
        return f"P{int(m.group(1))}"
    for part in parts[1:]:
        for token in re.split(r"[_\-\s]+", part):
            m = re.match(r"P?(\d+)$", token, re.I)
            if m:
                return f"P{int(m.group(1))}"
    return None


def csv_kwargs(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        for i, line in enumerate(f):
            s = line.lstrip("\ufeff").strip()
            if s.startswith("# GPS_time"):
                return {"skiprows": i + 1, "names": next(csv.reader([s.lstrip("# ")]))}
            if s and not s.startswith("#"):
                return {"skiprows": i}
    return {}


def time_col(df: pd.DataFrame) -> str | None:
    lookup = {str(c).lower(): c for c in df.columns}
    for c in ["Datetime", "DateTime", "Timestamp", "timestamp", "GPS_time", "time"]:
        if c.lower() in lookup:
            return lookup[c.lower()]
    return None


def parse_time(values: pd.Series) -> pd.Series:
    ts = pd.to_datetime(values, errors="coerce")
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_convert("Europe/Brussels").dt.tz_localize(None)
    return ts


def fix_swapped_date(ts: pd.Series, pid: str, expected: set[pd.Timestamp]) -> pd.Series:
    if ts.dropna().empty or (set(ts.dt.floor("10s").dropna()) & expected):
        return ts
    swapped = ts.copy()
    swapped.loc[ts.notna()] = pd.to_datetime(ts.dropna().dt.strftime("%Y-%d-%m %H:%M:%S"), errors="coerce")
    return swapped if (set(swapped.dt.floor("10s").dropna()) & expected) else ts


def clean_col(prefix: str, col: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", str(col).strip()).strip("_")
    return f"{prefix}__{safe}"


def good(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.strip().str.lower()
    return values.notna() & ~text.isin(MISSING_STRINGS)


def add_bins(stats, pid: str, col: str, sensor: str, ts: pd.Series, values: pd.Series, expected: dict[str, set[pd.Timestamp]]) -> None:
    if pid not in expected:
        return
    shift = pd.Timedelta(hours=2) if sensor == "02_ucm" else pd.Timedelta(0)
    bins = (ts.reset_index(drop=True) + shift).dt.floor("min" if sensor == "03_atmo_lys" else "10s")
    bins = bins[good(values.reset_index(drop=True)) & bins.notna()]
    if sensor == "03_atmo_lys":
        expanded = set()
        for offset in range(0, 60, 10):
            expanded.update((bins + pd.Timedelta(seconds=offset)).tolist())
        stats[pid][col].update(expanded & expected[pid])
    else:
        stats[pid][col].update(set(bins) & expected[pid])


def scan_raw_csv(path: Path, expected: dict[str, set[pd.Timestamp]], date_map: dict[str, str], stats) -> None:
    sensor = sensor_name(path)
    if sensor == "04_eyetracker" and path.name.lower() != "output.csv":
        return
    pid = infer_pid(path, date_map)
    if not pid:
        return
    try:
        df = pd.read_csv(path, low_memory=False, on_bad_lines="skip", encoding="utf-8-sig", **csv_kwargs(path))
    except Exception:
        return
    if df.empty:
        return
    tcol = time_col(df)
    if not tcol:
        return
    ts = fix_swapped_date(parse_time(df[tcol]), pid, expected.get(pid, set()))
    for col in df.columns:
        if str(col).lower() in RAW_SKIP or str(col).startswith("Unnamed:"):
            continue
        add_bins(stats, pid, clean_col(sensor, col), sensor, ts, df[col], expected)


def scan_empatica_names(expected: dict[str, set[pd.Timestamp]], date_map: dict[str, str], stats) -> None:
    files = {}
    for path in (RAW_DATA_DIR / "01_empatica").rglob("*.avro"):
        pid = infer_pid(path, date_map)
        m = re.search(r"_(\d+)\.avro$", path.name)
        if pid and m:
            files.setdefault(pid, []).append(int(m.group(1)))
    for pid, starts in files.items():
        present = set()
        starts = sorted(set(starts))
        for i, start_s in enumerate(starts):
            start = pd.to_datetime(start_s, unit="s", utc=True).tz_convert("Europe/Brussels").tz_localize(None).floor("10s")
            next_s = starts[i + 1] if i + 1 < len(starts) else start_s + 1800
            end = pd.to_datetime(min(next_s, start_s + 1800), unit="s", utc=True).tz_convert("Europe/Brussels").tz_localize(None).floor("10s")
            present.update({t for t in expected.get(pid, set()) if start <= t <= end})
        for col in EMPATICA_RAW_COLS:
            stats[pid][clean_col("01_empatica", col)].update(present)


def raw_missing_rows(idx: pd.DataFrame, participants: list[str]) -> list[dict]:
    expected = {pid: set(g["Datetime"]) for pid, g in idx.groupby("ParticipantID")}
    from collections import defaultdict
    stats = defaultdict(lambda: defaultdict(set))
    dmap = date_to_pid()
    scan_empatica_names(expected, dmap, stats)
    for path in sorted(RAW_DATA_DIR.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".csv", ".txt", ".tsv"}:
            scan_raw_csv(path, expected, dmap, stats)
    cols = sorted({c for by_col in stats.values() for c in by_col})
    rows = []
    for col in cols:
        row = {"ColumnName": col}
        for pid in participants:
            total = len(expected.get(pid, set()))
            present = len(stats[pid].get(col, set()))
            row[pid] = round((total - present) / total * 100, 3) if total else ""
        rows.append(row)
    return rows


def load_sensor(prefix: str, filename: str, idx: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(OUTPUTS / filename, low_memory=False)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df["ParticipantID"] = df["ParticipantID"].astype(str)
    keep = KEYS + [c for c in df.columns if c not in SKIP]
    df = df[keep].rename(columns={c: f"{prefix}__{c}" for c in keep if c not in KEYS})
    return idx.merge(df, on=KEYS, how="left")


def percent_rows(df: pd.DataFrame, participants: list[str], invalid: bool) -> list[dict]:
    rows = []
    for col in [c for c in df.columns if c not in KEYS]:
        vals = pd.to_numeric(df[col], errors="coerce")
        rng = allowed_range(col.split("__", 1)[1])
        if invalid and rng is None:
            continue
        row = {"ColumnName": col}
        if invalid:
            low, high = rng
            row["MinPossible"] = low
            row["MaxPossible"] = high if np.isfinite(high) else ""
        for pid in participants:
            sub = vals[df["ParticipantID"].eq(pid)]
            if invalid:
                nonmiss = sub.notna()
                bad = nonmiss & ((sub < low) | (sub > high))
                row[pid] = round(bad.sum() / nonmiss.sum() * 100, 3) if nonmiss.any() else ""
            else:
                row[pid] = round(sub.isna().mean() * 100, 3)
        rows.append(row)
    return rows


def main() -> None:
    idx = read_index()
    participants = sorted(idx["ParticipantID"].unique(), key=lambda p: int(p[1:]))
    missing_rows, invalid_rows = raw_missing_rows(idx, participants), []
    for prefix, filename in INPUTS.items():
        mapped = load_sensor(prefix, filename, idx)
        invalid_rows += percent_rows(mapped, participants, invalid=True)

    pd.DataFrame(missing_rows).to_csv(OUT_MISSING, index=False, lineterminator="\n")
    pd.DataFrame(invalid_rows).to_csv(OUT_INVALID, index=False, lineterminator="\n")
    print(f"Saved missing gaps: {OUT_MISSING}")
    print(f"Saved invalid QC:   {OUT_INVALID}")


if __name__ == "__main__":
    main()
