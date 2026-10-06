import pandas as pd

from _paths import OUTPUTS


def normalize_phase_id(series: pd.Series) -> pd.Series:
    return series.map(lambda value: "" if pd.isna(value) else str(value)).replace({"nan": "", "None": ""})


def align_to_index(df: pd.DataFrame, name: str = "") -> pd.DataFrame:
    idx_path = OUTPUTS / "00_index_10sec.csv"
    if not idx_path.exists():
        print("  WARNING: index not found, run 00_index_build.py first")
        return df

    idx = pd.read_csv(idx_path, low_memory=False)
    idx["Datetime"] = pd.to_datetime(idx["Datetime"])
    idx["ParticipantID"] = idx["ParticipantID"].astype(str)
    idx["PhaseID"] = normalize_phase_id(idx["PhaseID"])

    df["Datetime"] = pd.to_datetime(df["Datetime"])
    if hasattr(df["Datetime"].dtype, "tz") and df["Datetime"].dt.tz is not None:
        df["Datetime"] = df["Datetime"].dt.tz_localize(None)
    df["Datetime"] = df["Datetime"].dt.floor("10s")
    df["ParticipantID"] = df["ParticipantID"].astype(str)
    df["PhaseID"] = normalize_phase_id(df["PhaseID"])
    idx["PhaseID"] = normalize_phase_id(idx["PhaseID"])

    n_before = len(idx)
    result = idx[["ParticipantID", "PhaseID", "Datetime"]].merge(
        df, on=["ParticipantID", "PhaseID", "Datetime"], how="left"
    )
    n_missing = result.iloc[:, 3:].isna().all(axis=1).sum()
    if n_missing:
        print(f"  [{name}] {n_missing:,}/{n_before:,} slots have no data")
    return result
