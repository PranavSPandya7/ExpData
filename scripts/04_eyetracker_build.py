import warnings
from pathlib import Path
import datetime
import sys
from functools import lru_cache

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import KEY_FILE, OUTPUTS, RAW_ET_DIR, assert_sensor_folder_clean, key_participant_ids

STAGED_ROOT = RAW_ET_DIR

PHASES = ["BikeU", "WalkU", "BikeG", "WalkG", "Tram"]
DUPLICATE_PATH_TOKENS = (" copy", "- copy", "_copy", "backup", "old", "temp", "archive")
MIN_SAMPLES_10S = 600
KEY_COLS = ["ParticipantID", "PhaseID", "Datetime", "Date"]
SIGNAL_COLS = [
    "pupil_diameter_avg",
    "pupil_change_rate",
    "in_blink",
    "blink_duration_s",
    "in_fixation",
    "fixation_duration_s",
    "fixation_rate",
    "in_saccade",
    "saccade_duration_s",
    "saccade_amplitude",
    "saccade_peak_velocity",
    "gaze_velocity",
    "gaze_dispersion",
    "distance_from_center",
    "gaze_centrality",
    "eyelid_aperture_avg",
    "stress_composite",
]


def read_safe(path):
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except Exception:
        return None
def parse_export_folder_name(folder_name):
    parts = folder_name.split("_")
    pid = parts[0] if parts else ""
    phase = ""
    for part in parts[1:]:
        if part in PHASES:
            phase = part
            break
    if not phase and len(parts) > 1:
        phase = parts[1]
    return pid, phase


def is_duplicate_like_path(path: Path) -> bool:
    return any(
        any(token in part.lower() for token in DUPLICATE_PATH_TOKENS)
        for part in path.parts
    )


@lru_cache(maxsize=1)
def get_staged_exports():
    assert_sensor_folder_clean("eyetracker", STAGED_ROOT)
    expected_pids = {f"P{pid}" for pid in key_participant_ids(KEY_FILE)}
    STAGED_ROOT.mkdir(parents=True, exist_ok=True)
    staged = []

    folders = sorted(
        d for d in STAGED_ROOT.iterdir()
        if d.is_dir() and not is_duplicate_like_path(d)
    )
    print(f"Found {len(folders)} staged eyetracker folders")

    for src_folder in folders:
        pid, phase = parse_export_folder_name(src_folder.name)
        if not pid or not phase:
            continue
        if pid not in expected_pids or phase not in PHASES:
            print(f"  SKIP staged eyetracker folder outside key/phases: {src_folder.name}")
            continue
        staged.append((src_folder, pid, phase))

    print(f"Using {len(staged)} staged eyetracker exports from {STAGED_ROOT}")
    return staged


def resolve_export_file(pdir, filename):
    base = Path(pdir)
    direct = base / filename
    if direct.exists():
        return direct
    nested = base / "neon_player" / "exports" / "000" / filename
    if nested.exists():
        return nested
    matches = sorted(base.rglob(filename))
    return matches[0] if matches else direct


def process_folder(pdir, pid, phase):
    gaze_path = resolve_export_file(pdir, "gaze_positions.csv")
    gaze = read_safe(gaze_path)
    if gaze is None or len(gaze) == 0:
        return None

    fix = read_safe(resolve_export_file(pdir, "fixations.csv"))
    sac = read_safe(resolve_export_file(pdir, "saccades.csv"))
    blk = read_safe(resolve_export_file(pdir, "blinks.csv"))
    eye3 = read_safe(resolve_export_file(pdir, "3d_eye_states.csv"))

    ts_col = [c for c in gaze.columns if "timestamp" in c.lower() or c == "time"][0]
    gaze[ts_col] = pd.to_numeric(gaze[ts_col], errors="coerce")
    gaze = gaze.dropna(subset=[ts_col]).sort_values(ts_col).drop_duplicates(subset=[ts_col], keep="last").reset_index(drop=True)
    if len(gaze) == 0:
        return None

    t0 = gaze[ts_col].iloc[0]
    gaze["ts_sec"] = gaze[ts_col] / 1e9
    gaze["rel_time"] = gaze["ts_sec"] - gaze["ts_sec"].min()
    ts_values = gaze[ts_col].to_numpy()

    for evt_type, evt_df, sc, ec in [
        ("fixation", fix, "start timestamp [ns]", "end timestamp [ns]"),
        ("saccade", sac, "start timestamp [ns]", "end timestamp [ns]"),
        ("blink", blk, "start timestamp [ns]", "end timestamp [ns]"),
    ]:
        in_event = np.zeros(len(gaze), dtype=np.int8)
        event_dur = np.full(len(gaze), np.nan, dtype=float)

        if evt_df is not None and len(evt_df) > 0:
            dc = [c for c in evt_df.columns if "duration" in c.lower() and "ms" in c.lower()]
            if dc:
                evt_df["dur_s"] = evt_df[dc[0]] / 1000.0
            for _, row in evt_df.iterrows():
                if sc not in row.index or ec not in row.index:
                    continue
                start_idx = np.searchsorted(ts_values, row[sc], side="left")
                end_idx = np.searchsorted(ts_values, row[ec], side="right")
                if start_idx >= end_idx:
                    continue
                in_event[start_idx:end_idx] = 1
                if "dur_s" in evt_df.columns:
                    event_dur[start_idx:end_idx] = row["dur_s"]

        gaze[f"in_{evt_type}"] = in_event
        gaze[f"{evt_type}_dur"] = event_dur

    if eye3 is not None and len(eye3) > 0:
        rmap = {}
        for c in eye3.columns:
            cl = c.lower()
            if "pupil diameter left" in cl:
                rmap[c] = "pd_L"
            elif "pupil diameter right" in cl:
                rmap[c] = "pd_R"
            elif "eyelid aperture left" in cl:
                rmap[c] = "ea_L"
            elif "eyelid aperture right" in cl:
                rmap[c] = "ea_R"
        if rmap:
            er = eye3.rename(columns=rmap)
            ts_e3 = [c for c in er.columns if "timestamp" in c.lower() or c == "time"][0]
            gaze = gaze.merge(er, left_on=ts_col, right_on=ts_e3, how="left", suffixes=("", "_e3"))

    if "pd_L" in gaze.columns and "pd_R" in gaze.columns:
        gaze["pd_avg"] = gaze[["pd_L", "pd_R"]].mean(axis=1)
    if "ea_L" in gaze.columns and "ea_R" in gaze.columns:
        gaze["ea_avg"] = gaze[["ea_L", "ea_R"]].mean(axis=1)

    gx = [c for c in gaze.columns if "gaze x" in c.lower() or c == "point_x"][0]
    gy = [c for c in gaze.columns if "gaze y" in c.lower() or c == "point_y"][0]
    gaze = gaze.dropna(subset=[gx, gy])

    center_x = (gaze[gx].min() + gaze[gx].max()) / 2
    center_y = (gaze[gy].min() + gaze[gy].max()) / 2
    max_distance = np.sqrt((gaze[gx].max() - center_x) ** 2 + (gaze[gy].max() - center_y) ** 2)

    gaze["dt"] = gaze["ts_sec"].diff()
    gaze.loc[gaze["dt"] <= 0, "dt"] = np.nan
    gaze_dx = gaze[gx].diff()
    gaze_dy = gaze[gy].diff()
    gaze["gaze_vel_raw"] = np.sqrt(gaze_dx ** 2 + gaze_dy ** 2) / gaze["dt"]
    gaze["dist_ctr_raw"] = np.sqrt((gaze[gx] - center_x) ** 2 + (gaze[gy] - center_y) ** 2)
    gaze["gaze_ctr_raw"] = 1 - (gaze["dist_ctr_raw"] / max_distance) if max_distance > 0 else np.nan

    # Use raw gaze_positions.csv only. Normalize pixel coordinates per phase so dispersion
    # keeps within-phase variability without inheriting participant-specific pixel scale.
    window = min(10, len(gaze) // 10)
    if window > 1:
        x_vals = pd.to_numeric(gaze[gx], errors="coerce")
        y_vals = pd.to_numeric(gaze[gy], errors="coerce")
        x_q01, x_q99 = x_vals.quantile([0.01, 0.99])
        y_q01, y_q99 = y_vals.quantile([0.01, 0.99])
        x_scale = max(float(x_q99 - x_q01), float(x_vals.std(skipna=True)), 1.0)
        y_scale = max(float(y_q99 - y_q01), float(y_vals.std(skipna=True)), 1.0)
        gaze["gaze_x_norm"] = (x_vals - x_vals.median(skipna=True)) / x_scale
        gaze["gaze_y_norm"] = (y_vals - y_vals.median(skipna=True)) / y_scale
        gaze["gaze_x_std"] = gaze["gaze_x_norm"].rolling(window=window, min_periods=2).std()
        gaze["gaze_y_std"] = gaze["gaze_y_norm"].rolling(window=window, min_periods=2).std()
        gaze["gaze_disp_raw"] = np.sqrt(gaze["gaze_x_std"] ** 2 + gaze["gaze_y_std"] ** 2)
    else:
        gaze["gaze_disp_raw"] = np.nan

    gaze["b10s"] = (gaze["rel_time"] // 10) * 10
    g10s = (
        gaze.groupby("b10s")
        .agg(
            in_fixation=("in_fixation", "mean"),
            in_saccade=("in_saccade", "max"),
            in_blink=("in_blink", "max"),
            blink_dur=("blink_dur", "mean"),
            fixation_dur=("fixation_dur", "mean"),
            gaze_vel=("gaze_vel_raw", "mean"),
            dist_ctr=("dist_ctr_raw", "mean"),
            gaze_ctr=("gaze_ctr_raw", "mean"),
            gaze_disp=("gaze_disp_raw", "mean"),
            pd_avg=("pd_avg", "mean"),
            ea_avg=("ea_avg", "mean"),
            n=(ts_col, "count"),
        )
        .reset_index()
    )

    g10s["in_fixation"] = (g10s["in_fixation"] > 0.5).astype(int)
    g10s = g10s[g10s["n"] >= MIN_SAMPLES_10S].copy()
    if len(g10s) == 0:
        return None

    if sac is not None and len(sac) > 0:
        sac["b10s"] = ((sac["start timestamp [ns]"] - t0) / 1e9 // 10) * 10
        sac["dur_s"] = sac["duration [ms]"] / 1000.0
        if "peak velocity [px/s]" in sac.columns:
            peak = pd.to_numeric(sac["peak velocity [px/s]"], errors="coerce")
            invalid_peak = peak.notna() & ((peak >= 9000) | (peak <= 0))
            valid_denominator = int(peak.notna().sum())
            if invalid_peak.any() and valid_denominator:
                invalid_pct = 100 * int(invalid_peak.sum()) / valid_denominator
                print(
                    f"  {pid} {phase}: retained {int(invalid_peak.sum()):,}/{valid_denominator:,} "
                    f"raw saccade peak velocity values outside 0-9000 px/s ({invalid_pct:.2f}%)"
                )
        amp_col = None
        if "amplitude [deg]" in sac.columns:
            amp_col = "amplitude [deg]"
        elif "amplitude [px]" in sac.columns:
            amp_col = "amplitude [px]"
        sa = (
            sac.groupby("b10s")
            .agg(
                sac_dur=("dur_s", "mean"),
                sac_amp=(amp_col, "mean") if amp_col is not None else ("dur_s", "size"),
                sac_peak=("peak velocity [px/s]", "mean"),
            )
            .reset_index()
        )
        g10s = g10s.merge(sa, on="b10s", how="left")
    else:
        g10s["sac_dur"] = np.nan
        g10s["sac_amp"] = np.nan
        g10s["sac_peak"] = np.nan

    elapsed_s = g10s["b10s"].diff()
    g10s["pd_change"] = g10s["pd_avg"].diff() / elapsed_s

    prev_fixation = gaze["in_fixation"].shift(1)
    gaze["fixation_start"] = (
        (gaze["in_fixation"] == 1) & (prev_fixation != 1)
    ).astype(int)
    fix_starts = gaze.groupby("b10s")["fixation_start"].sum().reset_index()
    fix_starts.columns = ["b10s", "fix_count"]
    g10s = g10s.merge(fix_starts, on="b10s", how="left")
    if g10s["fix_count"].isna().any():
        raise ValueError(f"{pid} {phase}: internal fixation count alignment failed")
    g10s["fix_count"] = g10s["fix_count"].astype(int)
    g10s["fixation_rate"] = g10s["fix_count"] / 10.0

    fc = g10s["fix_count"]
    fc_min, fc_max = fc.min(), fc.max()
    g10s["fixation_count_scaled"] = (fc - fc_min) / (fc_max - fc_min) if fc_max > fc_min else np.nan

    fd = g10s["fixation_dur"]
    fd_min, fd_max = fd.min(), fd.max()
    g10s["fixation_duration_scaled"] = 1 - (fd - fd_min) / (fd_max - fd_min) if fd_max > fd_min else np.nan

    g10s["stress_composite"] = g10s[["fixation_count_scaled", "fixation_duration_scaled"]].mean(
        axis=1, skipna=False
    )

    idx_path = OUTPUTS / "00_index_10sec.csv"
    if idx_path.exists():
        idx_all = pd.read_csv(str(idx_path))
        idx_pp = idx_all[(idx_all["ParticipantID"] == pid) & (idx_all["PhaseID"] == phase)].reset_index(drop=True)
        if len(idx_pp) > 0:
            slot_idx = (g10s["b10s"] / 10).astype(int).to_numpy()
            keep = slot_idx < len(idx_pp)
            g10s = g10s.loc[keep].reset_index(drop=True)
            slot_idx = slot_idx[keep]
            idx_dt = idx_pp["Datetime"].iloc[slot_idx].values
            out = pd.DataFrame(index=range(len(g10s)))
            out["ParticipantID"] = pid
            out["PhaseID"] = phase
            out["Datetime"] = idx_dt
            out["Date"] = pd.to_datetime(idx_dt[0]).strftime("%Y-%m-%d") if len(idx_dt) > 0 else ""
        else:
            dt0 = pd.to_datetime(t0 / 1e9, unit="s") + pd.Timedelta(hours=2)
            out = pd.DataFrame(index=range(len(g10s)))
            out["ParticipantID"] = pid
            out["PhaseID"] = phase
            out["Datetime"] = [
                (dt0 + pd.Timedelta(seconds=int(t))).strftime("%Y-%m-%d %H:%M:%S") for t in g10s["b10s"]
            ]
            out["Date"] = dt0.strftime("%Y-%m-%d")
    else:
        dt0 = pd.to_datetime(t0 / 1e9, unit="s") + pd.Timedelta(hours=2)
        out = pd.DataFrame(index=range(len(g10s)))
        out["ParticipantID"] = pid
        out["PhaseID"] = phase
        out["Datetime"] = [(dt0 + pd.Timedelta(seconds=int(t))).strftime("%Y-%m-%d %H:%M:%S") for t in g10s["b10s"]]
        out["Date"] = dt0.strftime("%Y-%m-%d")

    out["pupil_diameter_avg"] = g10s["pd_avg"].values if "pd_avg" in g10s.columns else np.nan
    out["pupil_change_rate"] = g10s["pd_change"].values
    out["in_blink"] = g10s["in_blink"].values
    out["blink_duration_s"] = g10s["blink_dur"].values
    out["in_fixation"] = g10s["in_fixation"].values
    out["fixation_duration_s"] = g10s["fixation_dur"].values
    out["fixation_rate"] = g10s["fixation_rate"].values
    out["in_saccade"] = g10s["in_saccade"].values
    out["saccade_duration_s"] = g10s["sac_dur"].values
    out["saccade_amplitude"] = g10s["sac_amp"].values
    out["saccade_peak_velocity"] = g10s["sac_peak"].values
    out["gaze_velocity"] = g10s["gaze_vel"].values
    out["gaze_dispersion"] = g10s["gaze_disp"].values
    out["distance_from_center"] = g10s["dist_ctr"].values
    out["gaze_centrality"] = g10s["gaze_ctr"].values
    out["eyelid_aperture_avg"] = g10s["ea_avg"].values if "ea_avg" in g10s.columns else np.nan
    out["stress_composite"] = g10s["stress_composite"].values

    out.to_csv(gaze_path.parent / "output.csv", index=False)
    return out


def main():
    out_path = OUTPUTS / "04_eyetracker_10sec.csv"
    frames = []
    staged_exports = get_staged_exports()

    for pdir, pid, phase in staged_exports:
        result = process_folder(str(pdir), pid, phase)
        if result is not None:
            frames.append(result)
            fr = result["fixation_rate"].mean() if "fixation_rate" in result.columns else 0
            gv = result["gaze_velocity"].mean() if "gaze_velocity" in result.columns else 0
            print(f"{pid:<4} {phase:<8} -> {len(result):3d} rows  source=raw CSVs  fix_rate={fr:.2f}  gaze_vel={gv:.0f}")

    if not frames:
        print("No data.")
        return

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["ParticipantID", "PhaseID", "Datetime"]).reset_index(drop=True)

    from _align_index import align_to_index

    out["Datetime"] = pd.to_datetime(out["Datetime"])
    out = align_to_index(out, "eyetracker")
    expected_pids = {f"P{pid}" for pid in key_participant_ids(KEY_FILE)}
    out = out[out["ParticipantID"].isin(expected_pids)].reset_index(drop=True)

    out.to_csv(str(out_path), index=False)
    print(f"\nSaved: {out_path}")
    print(f"Rows: {len(out):,} x {len(out.columns)} cols")


if __name__ == "__main__":
    main()
