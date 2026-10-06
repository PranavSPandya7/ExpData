"""UCM backpack: read GPS/environmental CSVs, clean, resample to 10-sec, align to index."""

import warnings; warnings.filterwarnings("default")

from pathlib import Path
import pandas as pd
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import KEY_FILE, OUTPUTS, RAW_DATA_DIR, assert_sensor_folder_clean, load_key_unique


def flag_bad_gps_points(df, phase):
    """
    Return a boolean mask (True = bad, should be excluded).

    Exclusion criteria:
    1. HDOP > 5              - poor satellite geometry
    2. IO_flag == 9          - receiver reports no fix
    """
    n = len(df)
    bad = np.zeros(n, dtype=bool)

    # 1. HDOP > 5
    if "GPS_HDOP" in df.columns:
        hdop = df["GPS_HDOP"].values
        bad |= (hdop > 5)

    # 2. IO_flag == 9
    if "IO_flag" in df.columns:
        io_bad = df["IO_flag"].values == 9
        bad |= io_bad

    return bad

# Paths
RAW_DATA_ROOT = RAW_DATA_DIR
INDEX_FILE    = OUTPUTS / '00_index_10sec.csv'

# Skip-gate config
FORCE_RERUN = True

PHASES = ['BikeU', 'WalkU', 'BikeG', 'WalkG', 'Tram']


# Input check: use staged repo rawdata/ucm/
def discover_ucm_candidates(pid: int, phase: str) -> list[Path]:
    """Find plausible UCM CSV candidates without crossing into other sensor folders."""
    root = RAW_DATA_ROOT / '02_ucm'
    candidates = []

    def add_candidate(path: Path) -> None:
        candidates.append(path)

    flat = root / f'P{pid}_{phase}.csv'
    if flat.exists():
        add_candidate(flat)

    phase_roots = [
        root / f'P{pid}_{phase}',
        root / phase,
        root / f'P{pid}' / phase,
    ]
    for base in phase_roots:
        if not base.exists() or not base.is_dir():
            continue
        for path in base.rglob('data.csv'):
            add_candidate(path)
        for path in base.rglob('*.csv'):
            if path.name.lower() != 'data.csv':
                add_candidate(path)

    # For P4+ phase-level recordings, stop here so one phase cannot select
    # another phase's same-duration file. Fall back to old-style participant
    # folders only when no phase-specific candidates exist.
    if candidates:
        return sorted(set(candidates), key=lambda p: str(p).lower())

    fallback_roots = [
        root / f'P{pid}',
        root / f'P{pid}' / 'input data',
        root / f'P{pid}' / 'inputdata',
        root / f'P{pid}_{phase}' / 'input data',
        root / f'P{pid}_{phase}' / 'inputdata',
    ]
    for base in fallback_roots:
        if not base.exists() or not base.is_dir():
            continue
        for path in base.rglob('data.csv'):
            add_candidate(path)
        for path in base.rglob('*.csv'):
            if path.name.lower() != 'data.csv':
                add_candidate(path)

    # Keep deterministic unique paths only.
    return sorted(set(candidates), key=lambda p: str(p).lower())


def select_ucm_csv(pid: int, phase: str, idx_start, idx_end):
    """Choose the candidate with the strongest overlap with the key/index phase window."""
    candidates = discover_ucm_candidates(pid, phase)
    if not candidates:
        return None, None

    scored = []
    for path in candidates:
        df = read_ucm_file(path, phase=phase, idx_start=idx_start, idx_end=idx_end)
        if df is None or df.empty:
            continue
        if idx_start is not None and idx_end is not None:
            in_window = (df['Datetime'] >= idx_start) & (df['Datetime'] <= idx_end)
            overlap = int(in_window.sum())
        else:
            overlap = len(df)
        if overlap > 0:
            scored.append((overlap, path, df))

    if not scored:
        return None, None

    scored.sort(key=lambda item: (-item[0], str(item[1]).lower()))
    best_overlap = scored[0][0]
    tied = [item for item in scored if item[0] == best_overlap]
    if len(tied) > 1:
        paths = '\n    '.join(str(item[1]) for item in tied)
        print(f'  WARNING: tied UCM candidates for P{pid} {phase}; using first deterministic path:\n    {paths}')

    overlap, path, df = scored[0]
    if idx_start is not None and idx_end is not None:
        df = df[(df['Datetime'] >= idx_start) & (df['Datetime'] <= idx_end)].copy()
    print(f'  SELECT P{pid} {phase}: {path} ({overlap} overlapping rows)')
    return path, df


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
    Applies GPS quality filtering (HDOP, IO_flag).
    Shifts raw GPS_time by +2 h to convert UTC -> Brussels local time before
    matching against the key/index phase window.
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

    # GPS quality filter (HDOP, IO_flag)
    n_before = len(df)
    bad_mask = flag_bad_gps_points(df, phase)
    # Nullify GPS-dependent columns for bad epochs so they don't
    # pollute the 10-second means. Environmental data (AIR, AQ, SND,
    # WIND, SUN, IR) are kept as-is; the sensor was still recording.
    for col in ['GPS_lat', 'GPS_lon', 'GPS_alt', 'GPS_speed', 'GPS_hdg']:
        if col in df.columns:
            df.loc[bad_mask, col] = np.nan
    print(f'    GPS filter: {bad_mask.sum()}/{n_before} rows flagged ({bad_mask.sum()/n_before*100:.1f}%)')

    df['Datetime'] = df['Datetime'] + pd.Timedelta(hours=2)
    print('    [UTC fix] shifted GPS_time +2 h to Brussels local time')

    return df


# Main
def main():
    # Step 1: Check staged rawdata
    print("=" * 55)
    print("  [UCM Build] Step 1 - Check staged rawdata")
    print("=" * 55)
    assert_sensor_folder_clean("ucm", RAW_DATA_ROOT / '02_ucm')
    # Step 2: Build 10-sec CSV
    out_path = OUTPUTS / '02_ucm_10sec.csv'
    if out_path.exists() and not FORCE_RERUN:
        print(f'Output file already exists: {out_path}. Skipping UCM build (FORCE_RERUN=False).')
        return

    if not KEY_FILE.exists():
        print(f'ERROR: Key file not found: {KEY_FILE}')
        return

    key  = load_key_unique(KEY_FILE)
    pids = sorted(key['Participant_ID'].dropna().astype(int).unique().tolist())

    # Load index early so we can pass phase windows to read_ucm_file for UTC detection
    idx_df = None
    if INDEX_FILE.exists():
        idx_df = pd.read_csv(INDEX_FILE, low_memory=False)
        idx_df['Datetime']      = pd.to_datetime(idx_df['Datetime'])
        idx_df['ParticipantID'] = idx_df['ParticipantID'].astype(str)
        idx_df['PhaseID']       = idx_df['PhaseID'].astype(str)

    all_frames = []
    utci_qc_rows = []

    for pid in pids:
        for ph in PHASES:
            # Get expected index window for UTC offset detection
            idx_start = idx_end = None
            if idx_df is not None:
                win = idx_df[(idx_df['ParticipantID']==f'P{pid}') & (idx_df['PhaseID']==ph)]['Datetime']
                if not win.empty:
                    idx_start = win.min()
                    idx_end   = win.max()

            csv_path, df = select_ucm_csv(pid, ph, idx_start, idx_end)
            if csv_path is None:
                print(f'  SKIP P{pid} {ph}: no UCM CSV overlaps the key/index window under {RAW_DATA_ROOT}/02_ucm/')
                continue
            if df is None or df.empty:
                print(f'  WARNING: P{pid} {ph} -- empty or unreadable file')
                continue

            print(f'Processing P{pid} {ph}: {len(df):,} rows at 1-sec ...')

            # Resample to 10-second means (floor to nearest 10s first so
            # the output timestamps align to the fixed clock grid)
            df = df.set_index('Datetime').sort_index()
            if 'SND_dBA' in df.columns:
                df['SND_dBA'] = pd.to_numeric(df['SND_dBA'], errors='coerce').where(lambda s: s >= 0)
            # Impossible radiation/IR values are QC-masked before 10-sec aggregation; raw files are untouched.
            if 'SUN_Gh' in df.columns:
                df['SUN_Gh'] = pd.to_numeric(df['SUN_Gh'], errors='coerce').where(lambda s: s.between(0, 1300))
            for col in [c for c in df.columns if c.startswith('IR_spot_')]:
                df[col] = pd.to_numeric(df[col], errors='coerce').where(lambda s: s.between(-50, 90))
            numeric_cols = df.select_dtypes(include='number').columns.tolist()
            numeric_cols_no_speed = [col for col in numeric_cols if col != 'GPS_speed']
            df_10s = df[numeric_cols_no_speed].resample('10s').median().reset_index()
            df_10s['Datetime'] = df_10s['Datetime'].dt.floor('10s')
            if 'UTCI' in df.columns:
                utci = pd.to_numeric(df['UTCI'], errors='coerce')
                utci_positive = utci.where(utci > 0)
                df_10s['UTCI'] = utci_positive.resample('10s').mean().to_numpy()
                utci_bin_n = utci.resample('10s').count()
                utci_bin_positive_n = utci_positive.resample('10s').count()
                utci_invalid_n = int((utci <= 0).sum())
                utci_raw_n = int(utci.notna().sum())
                utci_qc_rows.append({
                    'ParticipantID': f'P{pid}',
                    'PhaseID': ph,
                    'utci_raw_n': utci_raw_n,
                    'utci_invalid_n': utci_invalid_n,
                    'utci_invalid_pct': (utci_invalid_n / utci_raw_n * 100.0) if utci_raw_n else np.nan,
                    'utci_all_invalid_10sec_bins': int(((utci_bin_n > 0) & (utci_bin_positive_n == 0)).sum()),
                })

            if 'GPS_speed' in df.columns:
                speed_valid = df['GPS_speed'].where(df['GPS_speed'] >= 0)
                speed_10s = speed_valid.resample('10s').agg(['count', 'mean']).reset_index()
                speed_10s['Datetime'] = speed_10s['Datetime'].dt.floor('10s')
                speed_10s = speed_10s.rename(columns={
                    'count': 'GPS_speed_valid_sample_count',
                    'mean': 'GPS_speed_mps',
                })
                speed_10s['GPS_speed_kmh'] = speed_10s['GPS_speed_mps'] * 3.6
                low_count_mask = speed_10s['GPS_speed_valid_sample_count'] < 3
                speed_10s.loc[low_count_mask, ['GPS_speed_mps', 'GPS_speed_kmh']] = np.nan
                df_10s = df_10s.merge(speed_10s, on='Datetime', how='left')

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

    # Left-join onto index backbone; guarantees every 10-sec slot is present.
    # Source phase is only used to select the correct UCM file/window. The canonical
    # output PhaseID must come from the shared index so reststop boundary rows stay aligned.
    if idx_df is not None:
        out['Datetime']      = pd.to_datetime(out['Datetime'])
        out['ParticipantID'] = out['ParticipantID'].astype(str)
        source_phase = out.pop('PhaseID').astype(str)
        duplicate_mask = out.duplicated(['ParticipantID', 'Datetime'], keep=False)
        if duplicate_mask.any():
            n_dupes = int(duplicate_mask.sum())
            print(f'  WARNING: {n_dupes:,} duplicate participant/time UCM rows after dropping source phase; keeping first deterministic row')
            out = out.assign(_source_phase=source_phase)
            out = out.sort_values(['ParticipantID', 'Datetime', '_source_phase']).drop_duplicates(['ParticipantID', 'Datetime'], keep='first')
            out = out.drop(columns=['_source_phase'])
        out = idx_df[['ParticipantID', 'PhaseID', 'Datetime', 'Date']].merge(
            out, on=['ParticipantID', 'Datetime'], how='left'
        )
        n_missing = out.iloc[:, 4:].isna().all(axis=1).sum()
        print(f'  Index join: {len(idx_df):,} index slots, {n_missing:,} have no UCM data')
    else:
        print(f'  NOTE: {INDEX_FILE.name} not found - run 00_index_build.py first')

    aq_cols = [c for c in out.columns if c.startswith('AQ_')]
    n_negative_aq = 0
    for col in aq_cols:
        vals = pd.to_numeric(out[col], errors='coerce')
        neg = vals < 0
        if neg.any():
            n_negative_aq += int(neg.sum())
            out.loc[neg, col] = np.nan
    if n_negative_aq:
        print(f'  Cleaned UCM AQ values: set {n_negative_aq:,} negative readings to NaN')

    # AQ1-AQ3 are invalid/empty backpack columns in this dataset.
    invalid_aq_cols = [c for c in ['AQ1', 'AQ2', 'AQ3'] if c in out.columns]
    if invalid_aq_cols:
        out = out.drop(columns=invalid_aq_cols)
        print(f'  Removed invalid UCM data columns: {invalid_aq_cols}')

    # Retain GPS speed only in km/h for analysis; m/s is the same signal in another unit.
    if 'GPS_speed_mps' in out.columns:
        out = out.drop(columns=['GPS_speed_mps'])
        print('  Removed GPS_speed_mps; retained GPS_speed_kmh only')

    out_path = OUTPUTS / '02_ucm_10sec.csv'
    out.to_csv(out_path, index=False)
    print(f'\nSaved {len(out):,} rows -> {out_path}')
    print(f'Participants: {sorted(out["ParticipantID"].unique())}')
    print(f'Phases:       {sorted(out["PhaseID"].unique())}')

    if utci_qc_rows:
        utci_qc = pd.DataFrame(utci_qc_rows)
        print('\nUCM UTCI invalid raw sample % by participant/phase (UTCI <= 0 set to NA before 10-sec mean):')
        print(utci_qc.pivot(index='ParticipantID', columns='PhaseID', values='utci_invalid_pct')
              .reindex(columns=PHASES).fillna(0).to_string(float_format=lambda x: f'{x:.3f}'))
        print('\nUCM UTCI all-invalid 10-sec bins by participant/phase:')
        print(utci_qc.pivot(index='ParticipantID', columns='PhaseID', values='utci_all_invalid_10sec_bins')
              .reindex(columns=PHASES).fillna(0).astype(int).to_string())

    if 'GPS_speed_valid_sample_count' in out.columns and 'GPS_speed_kmh' in out.columns:
        phase_speed_summary = (
            out[out['PhaseID'].isin(PHASES)]
            .groupby('PhaseID', dropna=False)
            .agg(
                total_10sec_rows=('Datetime', 'size'),
                speed_rows_with_3plus_samples=('GPS_speed_valid_sample_count', lambda s: int((s >= 3).sum())),
                mean_GPS_speed_kmh=('GPS_speed_kmh', 'mean'),
            )
            .reset_index()
        )
        phase_speed_summary['valid_speed_coverage_pct'] = (
            phase_speed_summary['speed_rows_with_3plus_samples'] /
            phase_speed_summary['total_10sec_rows'] * 100.0
        )
        print('\nFinal UCM speed summary by phase:')
        print(phase_speed_summary.to_string(index=False, float_format=lambda x: f'{x:.3f}'))

if __name__ == '__main__':
    main()

