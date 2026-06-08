"""Atmo+LYS: read LYS + Atmotube CSVs, merge 4 sensors, resample 1-min to 10-sec, align to index."""

import warnings; warnings.filterwarnings('ignore')
from pathlib import Path
import pandas as pd
from datetime import datetime
import shutil

BASE     = Path(r'C:\Users\pandya\Documents\Github\docker\ExpData')
CPW      = Path(r'C:\Users\pandya\OneDrive - UCL\Field experiment raw data\Complete Participantwise data\Atmo_lys')
RAW_DIR  = Path(r'C:\Users\pandya\Documents\Github\docker\rawdata\03_atmo_lys')
OUTPUTS  = BASE / 'outputs'
KEY_FILE = BASE / 'metadata' / 'key.csv'
INDEX_FILE = OUTPUTS / '00_index_10sec.csv'
UTC_OFFSET = pd.Timedelta(hours=2)

# ── SKIP-GATE CONFIG ─────────────────────────────────────────────────────────
FORCE_RERUN = False

PHASES = ['BikeU', 'WalkU', 'BikeG', 'WalkG', 'Tram']

# Sensor config: (sensor_id, file_suffix, output_prefix)
SENSORS = [
    ('LYS1', 'LYS1', 'LYS1'),
    ('LYS2', 'LYS2', 'LYS2'),
    ('atmo_left', 'Atmo_left', 'atmotube_left'),
    ('atmo_right', 'Atmo_right', 'atmotube_right'),
]


# ── HELPERS ──────────────────────────────────────────────────────────────────
def parse_key_date(d: str) -> str:
    return datetime.strptime(f'{d}-2025', '%d-%b-%Y').strftime('%Y-%m-%d')


def build_phase_windows(key: pd.DataFrame) -> dict:
    """Returns {(pid_int, phase_key): (start_ts, end_ts)} in Brussels local time."""
    windows = {}
    for _, row in key.iterrows():
        if pd.isna(row['Participant_ID']):
            continue
        pid  = int(row['Participant_ID'])
        date = parse_key_date(str(row['Date']))
        for ph in PHASES:
            s_col = f'{ph}_start'
            e_col = f'{ph}_end'
            if s_col not in row or pd.isna(row[s_col]) or pd.isna(row[e_col]):
                continue
            start = pd.Timestamp(f"{date} {row[s_col]}") + UTC_OFFSET
            end   = pd.Timestamp(f"{date} {row[e_col]}") + UTC_OFFSET
            if end < start:
                end += pd.Timedelta(days=1)
            windows[(pid, ph)] = (start, end)
    return windows


def load_sensor_file(pid: int, suffix: str, prefix: str, windows: dict) -> pd.DataFrame:
    """Load one sensor file, keep ALL data, label PhaseID where applicable."""
    fp = RAW_DIR / f'{pid}_{suffix}.csv'
    if not fp.exists():
        print(f'    SKIP {suffix}: file not found')
        return pd.DataFrame()

    df = pd.read_csv(fp)
    if df.empty or 'Datetime' not in df.columns:
        print(f'    SKIP {suffix}: empty or no Datetime column')
        return pd.DataFrame()

    df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
    df = df.dropna(subset=['Datetime'])

    # Assign PhaseID for rows within phase windows (keep rest as NaN)
    df['PhaseID'] = None
    for ph in PHASES:
        if (pid, ph) not in windows:
            continue
        start, end = windows[(pid, ph)]
        mask = (df['Datetime'] >= start) & (df['Datetime'] <= end)
        df.loc[mask, 'PhaseID'] = ph

    # Rename measurement columns with prefix
    meas_cols = [c for c in df.columns if c not in ('ParticipantID', 'Datetime', 'PhaseID')]
    rename = {c: f'{prefix}__{c}' for c in meas_cols}
    df = df.rename(columns=rename)

    # Floor to 10-sec bins and average (numeric cols only)
    df = df.set_index('Datetime').sort_index()
    prefixed_cols = [rename[c] for c in meas_cols]
    numeric_cols = [c for c in prefixed_cols if c in df.columns and df[c].dtype.kind in ('i','f')]
    if not numeric_cols:
        return pd.DataFrame()
    # Resample 1-minute data to 10-sec: forward-fill each value 6 times
    # (limit=5 prevents filling across large gaps. Since raw data is 1-min,
    # this copies each value into the 5 intervening 10-sec bins.)
    resampled = df[numeric_cols].resample('10s').ffill(limit=5)
    if 'PhaseID' in df.columns:
        resampled['PhaseID'] = df['PhaseID'].resample('10s').first()
    resampled = resampled.reset_index()
    resampled['Datetime'] = resampled['Datetime'].dt.floor('10s')
    resampled.insert(0, 'ParticipantID', f'P{pid}')

    phase_count = resampled['PhaseID'].notna().sum()
    print(f'    {suffix}: {len(resampled)} rows at 10-sec ({phase_count} in phase windows)')
    return resampled


# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    out_path = OUTPUTS / '03_atmo_lys_merged.csv'
    if out_path.exists() and not FORCE_RERUN:
        print(f'Output file already exists: {out_path}. Skipping Atmo & LYS build (FORCE_RERUN=False).')
        return

    if not KEY_FILE.exists():
        print(f'ERROR: Key file not found: {KEY_FILE}')
        return

    key     = pd.read_csv(KEY_FILE)
    windows = build_phase_windows(key)
    pids    = sorted(key['Participant_ID'].dropna().astype(int).tolist())

    # ── Collect: CPW → rawdata/atmo_lys/ ──
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    existing = list(RAW_DIR.glob('*.csv'))
    if not existing:
        count = 0
        for _, suffix, _ in SENSORS:
            for f in CPW.glob(f'*_{suffix}.csv'):
                shutil.copy2(f, RAW_DIR / f.name)
                count += 1
        print(f'  Collected {count} files -> {RAW_DIR}')
    else:
        print(f'  Already collected ({len(existing)} files in {RAW_DIR})')

    print(f'Key file: {len(pids)} participants, {len(windows)} phase windows')

    all_frames = []

    for pid in pids:
        key_date = parse_key_date(str(key[key['Participant_ID'] == pid].iloc[0]['Date']))
        print(f'Processing P{pid} ({key_date}) ...')

        sensor_frames = []
        for sensor_id, suffix, prefix in SENSORS:
            sf = load_sensor_file(pid, suffix, prefix, windows)
            if not sf.empty:
                sensor_frames.append(sf.set_index(['ParticipantID', 'PhaseID', 'Datetime']))

        if not sensor_frames:
            print(f'  => P{pid}: no data')
            continue

        merged = sensor_frames[0]
        for sf in sensor_frames[1:]:
            merged = merged.join(sf, how='outer')
        merged = merged.reset_index()
        all_frames.append(merged)
        print(f'  => P{pid}: {len(merged):,} merged rows')

    if not all_frames:
        print('\nWARNING: No data processed for any participant.')
        return

    out = pd.concat(all_frames, ignore_index=True)
    out = out.sort_values(['ParticipantID', 'PhaseID', 'Datetime']).reset_index(drop=True)

    # Left-join onto index backbone
    if INDEX_FILE.exists():
        idx = pd.read_csv(INDEX_FILE, low_memory=False)
        idx['Datetime']      = pd.to_datetime(idx['Datetime'])
        idx['ParticipantID'] = idx['ParticipantID'].astype(str)
        idx['PhaseID']       = idx['PhaseID'].astype(str)
        out['Datetime']      = pd.to_datetime(out['Datetime']).dt.floor('10s')
        out['ParticipantID'] = out['ParticipantID'].astype(str)
        out['PhaseID']       = out['PhaseID'].astype(str)
        result = idx[['ParticipantID', 'PhaseID', 'Datetime', 'Date']].merge(
            out, on=['ParticipantID', 'PhaseID', 'Datetime'], how='left'
        )
        # Forward-fill small gaps (≤3 min / 18 slots) within each phase
        signal_cols = result.columns[4:]
        if len(signal_cols):
            result[signal_cols] = result.groupby(['ParticipantID', 'PhaseID'])[signal_cols].transform(
                lambda g: g.ffill(limit=18)
            )
        out = result
        n_missing = out.iloc[:, 4:].isna().all(axis=1).sum()
        out['PhaseID'] = out['PhaseID'].fillna('').astype(str).replace({'nan':'','None':''})
        print(f'  Index join: {len(idx):,} index slots, {n_missing:,} have no sensor data')
    else:
        print(f'  NOTE: {INDEX_FILE.name} not found')

    out_path = OUTPUTS / '03_atmo_lys_merged.csv'
    out.to_csv(out_path, index=False)
    print(f'\nSaved {len(out):,} rows => {out_path}')
    print(f'Participants: {sorted(out["ParticipantID"].unique())}')
    print(f'Phases:       {sorted(out["PhaseID"].dropna().unique())}')
    print(f'Columns ({len(out.columns)}): {out.columns.tolist()}')

    # ── POST-PROCESS: clean out_path ──
    clean_atmo_lys(out_path)


def clean_atmo_lys(path: Path):
    """Read CSV, clip negatives, fill all NaN, save in-place."""
    print(f'  Cleaning {path.name} ...')
    df = pd.read_csv(path)
    # Clip negatives to 0
    numeric_cols = df.select_dtypes(include=['int','float']).columns
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').clip(lower=0)
    # Normalise PhaseID
    df['PhaseID'] = df['PhaseID'].fillna('').astype(str).replace({'nan':'','None':''})
    # Fill remaining NaN within each (ParticipantID, minute) group
    sig = [c for c in df.columns if c not in ('ParticipantID','PhaseID','Datetime','Date','key_0')]
    df['_minute'] = pd.to_datetime(df['Datetime']).dt.floor('1min')
    for _, grp in df.groupby(['ParticipantID','_minute']):
        df.loc[grp.index, sig] = grp[sig].ffill().bfill()
    df = df.drop(columns=['_minute'])
    df[sig] = df[sig].bfill().ffill()  # catch any remaining
    df.to_csv(path, index=False)
    nan_left = df.isna().sum().sum()
    neg_left = (df[numeric_cols] < 0).sum().sum()
    print(f'  Cleaned: NaN={nan_left}, Negatives={neg_left}')
    if nan_left == 0 and neg_left == 0:
        print('  ✅ Zero blanks, zero negatives')
    else:
        print(f'  ⚠️  Remaining: NaN={nan_left}, Neg={neg_left}')


if __name__ == '__main__':
    main()
