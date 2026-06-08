"""UCM backpack: read GPS/environmental CSVs, clean, resample to 10-sec, align to index."""

import warnings; warnings.filterwarnings('ignore')

from pathlib import Path
import pandas as pd
import numpy as np


# ── GPS QUALITY FILTER (haversine distance + HDOP + IO_flag + jump) ────────
def haversine_m(lat1, lon1, lat2, lon2):
    """Haversine distance in metres. Works with arrays or scalars."""
    dlat, dlon = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2)
    return 6371000 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def flag_bad_gps_points(df, phase):
    """
    Return a boolean mask (True = bad, should be excluded).

    Exclusion criteria:
    1. HDOP > 5              — poor satellite geometry
    2. IO_flag == 9          — receiver reports no fix
    3. Step distance > 50 m in <2 s  — coordinate anomaly
    """
    n = len(df)
    bad = np.zeros(n, dtype=bool)

    # ── 1. HDOP ────────────────────────────────────────────────────────
    if "GPS_HDOP" in df.columns:
        hdop = df["GPS_HDOP"].values
        bad_hdop = hdop > 5
        bad |= bad_hdop
    else:
        bad_hdop = np.zeros(n, dtype=bool)

    # ── 2. IO_flag == 9 ────────────────────────────────────────────────
    if "IO_flag" in df.columns:
        io_bad = df["IO_flag"].values == 9
        bad |= io_bad
    else:
        io_bad = np.zeros(n, dtype=bool)

    lat = df["GPS_lat"].values.astype(float)
    lon = df["GPS_lon"].values.astype(float)
    time_v = df["Datetime"].values
    valid_coord = ~np.isnan(lat) & ~np.isnan(lon)

    # ── 3. Large jump anomaly (>50 m in <2 s) ──────────────────────────
    step_m = np.full(n, np.nan)
    for i in range(1, n):
        if valid_coord[i - 1] and valid_coord[i] and pd.notna(time_v[i - 1]) and pd.notna(time_v[i]):
            dt = (time_v[i] - time_v[i - 1]) / np.timedelta64(1, "s")
            if dt > 0 and dt < 2:
                step_m[i] = haversine_m(lat[i - 1], lon[i - 1], lat[i], lon[i])

    jump_bad = step_m > 50
    bad |= jump_bad

    # Propagate bad one row forward (step INTO bad point is unreliable)
    bad[1:] |= bad[:-1].copy()

    return bad

# ── PATHS ────────────────────────────────────────────────────────────────────
BASE          = Path(r'C:\Users\pandya\Documents\Github\docker\ExpData')
RAW_DATA_ROOT = Path(r'C:\Users\pandya\Documents\Github\docker\rawdata')
OUTPUTS = Path(r'C:\Users\pandya\Documents\Github\docker\ExpData\outputs')
KEY_FILE      = BASE / 'metadata' / 'key.csv'
INDEX_FILE    = OUTPUTS / '00_index_10sec.csv'

# ── SKIP-GATE CONFIG ─────────────────────────────────────────────────────────
FORCE_RERUN = False

PHASES = ['BikeU', 'WalkU', 'BikeG', 'WalkG', 'Tram']


# ── COLLECTION: CPW → rawdata/ucm/ ───────────────────────────────────────────
def collect_ucm(force: bool = False):
    """
    Copy data.csv files from Complete Participantwise data into rawdata/ucm/.
    Skips if already collected.
    """
    cpw = Path(r'C:\Users\pandya\OneDrive - UCL\Field experiment raw data\Complete Participantwise data')
    out_dir = RAW_DATA_ROOT / '02_ucm'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check if already collected
    existing = list(out_dir.glob('P*_*.csv'))
    if existing:
        print(f'  UCM already collected ({len(existing)} files). Skip collection.')
        return

    key = pd.read_csv(KEY_FILE)
    pids = sorted(key['Participant_ID'].dropna().astype(int).tolist())

    count = 0
    for pid in pids:
        for ph in PHASES:
            out_name = out_dir / f'P{pid}_{ph}.csv'
            if out_name.exists() and not force:
                count += 1
                continue
            pdir = cpw / f'P{pid}' / ph / 'ucm'
            if not pdir.exists():
                continue
            dcs = sorted(pdir.rglob('data.csv'))
            if not dcs:
                continue
            import shutil
            shutil.copy2(str(dcs[0]), str(out_name))
            count += 1
            print(f'    Collected P{pid} {ph} -> {out_name.name}')
    print(f'  UCM collection: {count} files')


def find_ucm_csv(pid: int, phase: str):
    """
    Locate data.csv for this participant + phase.
    Flat structure: rawdata/ucm/P{pid}_{phase}.csv
    """
    path = RAW_DATA_ROOT / '02_ucm' / f'P{pid}_{phase}.csv'
    if path.exists():
        return path
    return None


def parse_ucm_columns(path: Path):
    """Extract column names from the '# GPS_time,...' comment line."""
    with open(path, 'r', errors='replace') as f:
        for line in f:
            stripped = line.lstrip('# ').strip()
            if stripped.startswith('GPS_time'):
                return [c.strip() for c in stripped.split(',')]
    return None


def read_ucm_file(path: Path, phase: str, idx_start=None, idx_end=None):
    """Read one UCM data.csv, assign column names, parse GPS_time as Datetime.
    Applies GPS quality filtering (HDOP, IO_flag, implausible speed, jumps).
    If the raw timestamps fall outside the index window by ~2 h, shift them
    by +2 h to convert UTC -> Brussels local time.
    """
    col_names = parse_ucm_columns(path)
    if col_names is None:
        print(f'  ERROR: column header not found in {path}')
        return None

    try:
        df = pd.read_csv(path, comment='#', header=None,
                         names=col_names, low_memory=False)
    except Exception as e:
        print(f'  ERROR reading {path}: {e}')
        return None

    if 'GPS_time' not in df.columns:
        print(f'  ERROR: GPS_time column missing in {path}')
        return None

    df = df.rename(columns={'GPS_time': 'Datetime'})
    df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
    df = df.dropna(subset=['Datetime'])

    if df.empty:
        print(f'  WARNING: no valid timestamps in {path}')
        return None

    # ── GPS quality filter (HDOP, IO_flag, jumps) ──────────────────────
    n_before = len(df)
    bad_mask = flag_bad_gps_points(df, phase)
    # Nullify GPS-dependent columns for bad epochs so they don't
    # pollute the 10-second means. Environmental data (AIR, AQ, SND,
    # WIND, SUN, IR) are kept as-is — the sensor was still recording.
    for col in ['GPS_lat', 'GPS_lon', 'GPS_alt', 'GPS_speed', 'GPS_hdg']:
        if col in df.columns:
            df.loc[bad_mask, col] = np.nan
    print(f'    GPS filter: {bad_mask.sum()}/{n_before} rows flagged ({bad_mask.sum()/n_before*100:.1f}%)')

    # Auto-detect UTC vs local time: if the data starts ~2h before the index
    # window, the device was recording in UTC while the index is Brussels local
    # (UTC+2 in summer). Shift forward by 2 hours.
    if idx_start is not None:
        raw_start = df['Datetime'].iloc[0]
        offset_h  = (idx_start - raw_start).total_seconds() / 3600
        if 1.5 < offset_h < 2.5:
            df['Datetime'] = df['Datetime'] + pd.Timedelta(hours=2)
            print(f'    [UTC fix] shifted timestamps +2 h (raw was {offset_h:.1f}h behind index)')

    return df


# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    # ── Step 1: Collect raw data from CPW (if not already collected) ────
    print("=" * 55)
    print("  [UCM Build] Step 1 — Collect raw data")
    print("=" * 55)
    collect_ucm()

    # ── Step 2: Build 10-sec CSV ────────────────────────────────────────
    out_path = OUTPUTS / '02_ucm_10sec.csv'
    if out_path.exists() and not FORCE_RERUN:
        print(f'Output file already exists: {out_path}. Skipping UCM build (FORCE_RERUN=False).')
        return

    if not KEY_FILE.exists():
        print(f'ERROR: Key file not found: {KEY_FILE}')
        return

    key  = pd.read_csv(KEY_FILE)
    pids = sorted(key['Participant_ID'].dropna().astype(int).tolist())

    # Load index early so we can pass phase windows to read_ucm_file for UTC detection
    idx_df = None
    if INDEX_FILE.exists():
        idx_df = pd.read_csv(INDEX_FILE, low_memory=False)
        idx_df['Datetime']      = pd.to_datetime(idx_df['Datetime'])
        idx_df['ParticipantID'] = idx_df['ParticipantID'].astype(str)
        idx_df['PhaseID']       = idx_df['PhaseID'].astype(str)

    all_frames = []

    for pid in pids:
        for ph in PHASES:
            csv_path = find_ucm_csv(pid, ph)
            if csv_path is None:
                print(f'  SKIP P{pid} {ph}: P{pid}_{ph}.csv not found under {RAW_DATA_ROOT}/02_ucm/')
                continue

            # Get expected index window for UTC offset detection
            idx_start = idx_end = None
            if idx_df is not None:
                win = idx_df[(idx_df['ParticipantID']==f'P{pid}') & (idx_df['PhaseID']==ph)]['Datetime']
                if not win.empty:
                    idx_start = win.min()
                    idx_end   = win.max()

            df = read_ucm_file(csv_path, phase=ph, idx_start=idx_start, idx_end=idx_end)
            if df is None or df.empty:
                print(f'  WARNING: P{pid} {ph} -- empty or unreadable file')
                continue

            print(f'Processing P{pid} {ph}: {len(df):,} rows at 1-sec ...')

            # Resample to 10-second means (floor to nearest 10s first so
            # the output timestamps align to the fixed clock grid)
            df = df.set_index('Datetime').sort_index()
            numeric_cols = df.select_dtypes(include='number').columns.tolist()
            df_10s = df[numeric_cols].resample('10s').mean().reset_index()
            df_10s['Datetime'] = df_10s['Datetime'].dt.floor('10s')

            df_10s.insert(0, 'ParticipantID', f'P{pid}')
            df_10s.insert(1, 'PhaseID',       ph)

            all_frames.append(df_10s)
            print(f'  -> {len(df_10s):,} rows at 10-sec')

    if not all_frames:
        print('WARNING: No UCM data processed. Check that RAW_DATA_ROOT exists:')
        print(f'  {RAW_DATA_ROOT}')
        return

    out = pd.concat(all_frames, ignore_index=True)
    out = out.sort_values(['ParticipantID', 'PhaseID', 'Datetime']).reset_index(drop=True)

    # ── Left-join onto index backbone (guarantees every 10-sec slot is present) ──
    if idx_df is not None:
        out['Datetime']      = pd.to_datetime(out['Datetime'])
        out['ParticipantID'] = out['ParticipantID'].astype(str)
        out['PhaseID']       = out['PhaseID'].astype(str)
        out = idx_df[['ParticipantID', 'PhaseID', 'Datetime', 'Date']].merge(
            out, on=['ParticipantID', 'PhaseID', 'Datetime'], how='left'
        )
        n_missing = out.iloc[:, 4:].isna().all(axis=1).sum()
        # Fill NaN PhaseID from index join (between-phase rows)
        out['PhaseID'] = out['PhaseID'].fillna('').astype(str).replace({'nan':'','None':''})
        print(f'  Index join: {len(idx_df):,} index slots, {n_missing:,} have no UCM data')
    else:
        print(f'  NOTE: {INDEX_FILE.name} not found — run 00_index_build.py first')

    out_path = OUTPUTS / '02_ucm_10sec.csv'
    out.to_csv(out_path, index=False)
    print(f'\nSaved {len(out):,} rows -> {out_path}')
    print(f'Participants: {sorted(out["ParticipantID"].unique())}')
    print(f'Phases:       {sorted(out["PhaseID"].unique())}')

if __name__ == '__main__':
    main()

