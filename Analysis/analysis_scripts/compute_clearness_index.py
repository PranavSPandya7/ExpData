"""
compute_clearness_index.py

Calculates the Clearness Index (k_t = G_H / G_0) using Duffie & Beckman textbook equations:
  G_0(t) = G_sc * E_0(n) * sin(alpha(t))
  - G_sc = 1367 W/m^2 (solar constant)
  - E_0(n) = 1 + 0.033 * cos(2 * pi * n / 365) (orbit eccentricity correction, n = day of year)
  - alpha(t) = sun altitude angle in radians from ucm_SUN_alt
  - G_H(t) = global horizontal solar radiation from ucm_SUN_Gh
  - Clearness_Index = G_H / G_0
  - Weather inference categories:
      k_t >= 0.65       -> Sunny
      0.30 <= k_t < 0.65 -> Partly Cloudy
      k_t < 0.30        -> Cloudy

Inputs:
  - C:\\Users\\pandya\\Documents\\Github\\docker\\Paper3_Github\\output\\merged_all_11participants.csv

Outputs:
  - C:\\Users\\pandya\\Documents\\Github\\docker\\ExpData\\Analysis\\analysis_output\\Clearness Index\\merged_all_11participants_CI.csv
  - C:\\Users\\pandya\\Documents\\Github\\docker\\ExpData\\Analysis\\analysis_output\\Clearness Index\\merged_all_11participants_8min_CI.csv
  - C:\\Users\\pandya\\Documents\\Github\\docker\\ExpData\\Analysis\\analysis_output\\Clearness Index\\Participant_Clearness_Summary.csv
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd

# ==============================================================================
# CONFIGURATION & PATHS
# ==============================================================================
INPUT_CSV = Path(r"C:\Users\pandya\Documents\Github\docker\Paper3_Github\output\merged_all_11participants.csv")
OUT_DIR = Path(r"C:\Users\pandya\Documents\Github\docker\ExpData\Analysis\analysis_output\Clearness Index")
OUT_CI_FULL = OUT_DIR / "merged_all_11participants_CI.csv"
OUT_CI_8MIN = OUT_DIR / "merged_all_11participants_8min_CI.csv"
OUT_SUMMARY = OUT_DIR / "Participant_Clearness_Summary.csv"

# Solar constant
G_SC = 1367.0         # Solar constant [W/m^2]

SUNNY_THRESHOLD = 0.65
PARTLY_CLOUDY_THRESHOLD = 0.30

# Phases to exclude from 8-minute clipping (preserve full duration)
UNCLIPPED_PHASES = {"Indoor", "reststop"}
MATCHED_DURATION_SEC = 480  # 8 minutes


# ==============================================================================
# CALCULATION FUNCTIONS
# ==============================================================================
def infer_weather(kt):
    """Categorize Clearness Index into weather classifications."""
    if pd.isna(kt):
        return np.nan
    if kt >= SUNNY_THRESHOLD:
        return "Sunny"
    elif kt >= PARTLY_CLOUDY_THRESHOLD:
        return "Partly Cloudy"
    else:
        return "Cloudy"


def add_clearness_index(df: pd.DataFrame) -> pd.DataFrame:
    """Computes extraterrestrial radiation G0, Clearness Index, and weather inference."""
    df = df.copy()
    if "Datetime" not in df.columns:
        raise ValueError("Missing 'Datetime' column in input data.")
    
    dt = pd.to_datetime(df["Datetime"], errors="coerce")
    doy = dt.dt.dayofyear

    # Eccentricity correction E0(n) = 1 + 0.033 * cos(2 * pi * doy / 365)
    e0 = 1.0 + 0.033 * np.cos(2.0 * np.pi * doy / 365.0)

    # Solar altitude angle alpha in degrees -> radians -> sin(alpha)
    alpha_deg = pd.to_numeric(df.get("ucm_SUN_alt"), errors="coerce")
    alpha_rad = np.deg2rad(alpha_deg)
    sin_alpha = np.sin(alpha_rad)

    # Extraterrestrial irradiance on horizontal surface: G0 = G_sc * E0 * sin(alpha)
    g0 = G_SC * e0 * sin_alpha

    # Measured global horizontal radiation GH
    gh = pd.to_numeric(df.get("ucm_SUN_Gh"), errors="coerce")

    # Validity mask: retain only observed, non-negative irradiance with a positive denominator.
    mask_valid = gh.notna() & g0.notna() & (gh >= 0) & (g0 > 0)

    clearness_index = pd.Series(np.nan, index=df.index, dtype=float)
    clearness_index.loc[mask_valid] = gh.loc[mask_valid] / g0.loc[mask_valid]

    df["Clearness_Index_G0"] = g0
    df["Clearness_Index"] = clearness_index
    df["Clearness_Index_Infer"] = df["Clearness_Index"].apply(infer_weather)
    return df


def clip_to_8min(df: pd.DataFrame) -> pd.DataFrame:
    """Clips each participant-phase to the first 8 minutes (<=480s), preserving Indoor & reststop."""
    df = df.copy()
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    clipped_chunks = []
    
    for (pid, ph), grp in df.groupby(["ParticipantID", "PhaseID"], sort=False):
        grp = grp.sort_values("Datetime")
        if ph in UNCLIPPED_PHASES:
            clipped_chunks.append(grp)
        else:
            t0 = grp["Datetime"].iloc[0]
            cutoff = t0 + pd.Timedelta(seconds=MATCHED_DURATION_SEC)
            clipped_chunks.append(grp[grp["Datetime"] <= cutoff])
            
    res = pd.concat(clipped_chunks, ignore_index=True)
    return res


def compute_participant_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Generates participant-wise Clearness Index min/max and weather percentage distribution."""
    # Consider active experimental route phases or all valid measurements
    stats = df.groupby("ParticipantID")["Clearness_Index"].agg(
        Clearness_Index_Min="min",
        Clearness_Index_Max="max",
        Clearness_Index_Mean="mean"
    ).reset_index()

    # Percentage frequency of inferences (excluding NaN/invalid)
    valid_infer = df[df["Clearness_Index_Infer"].notna()]
    counts = valid_infer.groupby(["ParticipantID", "Clearness_Index_Infer"]).size().unstack(fill_value=0)
    pcts = counts.div(counts.sum(axis=1), axis=0) * 100.0
    for col in ["Sunny", "Partly Cloudy", "Cloudy"]:
        if col not in pcts.columns:
            pcts[col] = 0.0
    pcts = pcts[["Sunny", "Partly Cloudy", "Cloudy"]]
    pcts.columns = [f"% {col}" for col in pcts.columns]
    pcts = pcts.reset_index()

    summary = pd.merge(stats, pcts, on="ParticipantID", how="left")
    return summary


# ==============================================================================
# MAIN PIPELINE EXECUTION
# ==============================================================================
def main():
    print("=" * 80)
    print("COMPUTING CLEARNESS INDEX AND GENERATING MATCHED CSVs")
    print("=" * 80)
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Reading input master file: {INPUT_CSV}")
    df_raw = pd.read_csv(INPUT_CSV, low_memory=False)
    print(f"Input records: {len(df_raw):,} rows x {df_raw.shape[1]} columns")

    # 1. Compute Clearness Index on merged_all_11participants
    print("\nCalculating extraterrestrial solar radiation G0, Clearness Index kt, and weather inferences...")
    df_ci = add_clearness_index(df_raw)
    df_ci.to_csv(OUT_CI_FULL, index=False)
    print(f"[OK] Saved full dataset: {OUT_CI_FULL} ({len(df_ci):,} rows x {df_ci.shape[1]} columns)")

    # 2. Clip to 8 minutes (except Indoor and reststop)
    print("\nClipping active route phases to first 8 minutes (480 seconds)...")
    df_ci_8min = clip_to_8min(df_ci)
    df_ci_8min.to_csv(OUT_CI_8MIN, index=False)
    print(f"[OK] Saved 8-minute clipped dataset: {OUT_CI_8MIN} ({len(df_ci_8min):,} rows x {df_ci_8min.shape[1]} columns)")

    # 3. Generate summary table per participant
    print("\nGenerating participant-level Clearness Index summary table...")
    summary = compute_participant_summary(df_ci_8min)
    summary.to_csv(OUT_SUMMARY, index=False)
    print(f"[OK] Saved summary table: {OUT_SUMMARY}")
    print("\nParticipant Summary Table Preview:")
    print(summary.to_string(index=False))

    print("\n" + "=" * 80)
    print("ALL FILES SUCCESSFULLY GENERATED:")
    print(f"1. {OUT_CI_FULL}")
    print(f"2. {OUT_CI_8MIN}")
    print(f"3. {OUT_SUMMARY}")
    print("=" * 80)


if __name__ == "__main__":
    main()
