"""Empatica: read AVROs via NeuroKit2, resample to 10-sec, align to index."""

import warnings; warnings.filterwarnings('ignore')
from datetime import datetime
from pathlib import Path
import shutil

import neurokit2 as nk
import numpy as np
import pandas as pd
from avro.datafile import DataFileReader
from avro.io import DatumReader

from _align_index import align_to_index

# ===========================================================================
#  CONFIG
# ===========================================================================
BASE       = Path(r'C:\Users\pandya\Documents\Github\docker\ExpData')
EMPA_CPW   = Path(r'C:\Users\pandya\OneDrive - UCL\Field experiment raw data\Complete Participantwise data\empatica_raw')
RAW_EMPA   = Path(r'C:\Users\pandya\Documents\Github\docker\rawdata\01_empatica')
EMPA_RAW   = RAW_EMPA  # processing reads from rawdata (after collection)
OUTPUTS    = BASE / 'outputs'
KEY_FILE   = BASE / 'metadata\key.csv'

# ── SKIP-GATE CONFIG ─────────────────────────────────────────────────────────
FORCE_RERUN = False

OUTPUT_SUFFIX = ''

PHASES   = ['BikeU', 'WalkU', 'BikeG', 'WalkG', 'Tram']
PHASE_ID = {
    'BikeU': 'BikeU', 'WalkU': 'WalkU',
    'BikeG': 'BikeG', 'WalkG': 'WalkG', 'Tram': 'Tram',
}

PHASE_COLORS = {
    'BikeU': '#d45500', 'WalkU': '#b8860b',
    'BikeG': '#1a6b1a', 'WalkG': '#52b852', 'Tram': '#7f8c8d',
}

SIGNAL_COLS = ['acc_x', 'acc_y', 'acc_z', 'vector_magnitude',
               'eda', 'eda_tonic', 'eda_phasic', 'temperature',
               'bvp', 'heart_rate', 'hrv_rmssd']

SR_EDA    = 4    # Hz (Empatica native)
SR_BVP    = 64   # Hz (Empatica native)
MIN_EDA_S = 20   # minimum seconds of EDA for NeuroKit2 to run
MIN_BVP_S = 8    # minimum seconds of BVP for NeuroKit2 to run
                 # 8s = 512 samples @ 64Hz; enough for 5-8 beats
                 # 9s packets (device reconnection buffers) are included
PPG_QUALITY_MIN = 0.5  # SQI threshold (Elgendi 2016); below = motion artifact


# ===========================================================================
#  AVRO READING (read-only)
# ===========================================================================

def _us_to_brussels(unix_us_array):
    return pd.to_datetime(unix_us_array, unit='us', utc=True).tz_convert('Europe/Brussels')


def read_avro_sensors(avro_path):
    """Read one .avro -> dict of raw sensor DataFrames at native Hz."""
    reader = DataFileReader(open(str(avro_path), 'rb'), DatumReader())
    data   = next(reader)
    reader.close()
    raw     = data['rawData']
    sensors = {}

    # Accelerometer
    try:
        a = raw['accelerometer']
        n = len(a['x'])
        if n > 0:
            dp = a['imuParams']['physicalMax'] - a['imuParams']['physicalMin']
            dd = a['imuParams']['digitalMax']  - a['imuParams']['digitalMin']
            us = np.round(a['timestampStart'] + np.arange(n) * (1e6 / a['samplingFrequency'])).astype(np.int64)
            sensors['acc'] = pd.DataFrame(
                {'acc_x': np.array(a['x']) * dp / dd,
                 'acc_y': np.array(a['y']) * dp / dd,
                 'acc_z': np.array(a['z']) * dp / dd},
                index=_us_to_brussels(us)
            )
    except Exception:
        pass

    # EDA
    try:
        e = raw['eda']
        n = len(e['values'])
        if n > 0:
            sf = float(e['samplingFrequency'])
            us = np.round(e['timestampStart'] + np.arange(n) * (1e6 / sf)).astype(np.int64)
            sensors['eda'] = pd.DataFrame({'eda': e['values']}, index=_us_to_brussels(us))
            sensors['eda_sf'] = sf
    except Exception:
        pass

    # Temperature
    try:
        t = raw['temperature']
        n = len(t['values'])
        if n > 0:
            us = np.round(t['timestampStart'] + np.arange(n) * (1e6 / t['samplingFrequency'])).astype(np.int64)
            sensors['temperature'] = pd.DataFrame({'temperature': t['values']}, index=_us_to_brussels(us))
    except Exception:
        pass

    # BVP
    try:
        b = raw['bvp']
        n = len(b['values'])
        if n > 0:
            sf  = float(b['samplingFrequency'])
            us  = np.round(b['timestampStart'] + np.arange(n) * (1e6 / sf)).astype(np.int64)
            sensors['bvp'] = pd.DataFrame({'bvp': b['values']}, index=_us_to_brussels(us))
            sensors['bvp_sf'] = sf
    except Exception:
        pass

    return sensors


# ===========================================================================
#  NEUROKIT2 — PER-AVRO SEGMENT PROCESSING
# ===========================================================================

def process_eda_segment(eda_df, sr):
    """Run nk.eda_process on one AVRO segment. Returns eda_df with tonic/phasic columns."""
    if len(eda_df) < int(sr * MIN_EDA_S):
        return eda_df.copy()
    try:
        vals = eda_df['eda'].fillna(0).values
        sig, _ = nk.eda_process(vals, sampling_rate=int(round(sr)))
        out = eda_df.copy()
        out['eda_tonic']  = sig['EDA_Tonic'].values[:len(out)]
        out['eda_phasic'] = sig['EDA_Phasic'].values[:len(out)]
        return out
    except Exception:
        return eda_df.copy()


def process_bvp_segment(bvp_df, sr):
    """
    Run nk.ppg_process on one AVRO BVP segment.
    Returns (hr_series, peak_times_list).

        HR in v2 is taken directly from NeuroKit2's interpolated PPG_Rate output.
        Quality gating is kept only for peak selection used in HRV, not for the
        displayed heart-rate series.

    NaN guard: short reconnection-buffer packets (e.g. exactly 9s) may
      contain NaN placeholder values; these are replaced with 0 before
      processing so NeuroKit2 does not crash.
    """
    if len(bvp_df) < int(sr * MIN_BVP_S):
        return None, []
    try:
        # NaN guard — short reconnection packets can contain NaN
        vals = bvp_df['bvp'].values.astype(float)
        vals = np.where(np.isnan(vals), 0.0, vals)
        sig, _ = nk.ppg_process(vals, sampling_rate=int(round(sr)))

        # Apply PPG Signal Quality Index filter + physiological range gate
        quality   = sig['PPG_Quality'].values[:len(bvp_df)]
        hr_raw    = sig['PPG_Rate'].values[:len(bvp_df)]
        hr_masked = np.where(
            (quality >= PPG_QUALITY_MIN) & (hr_raw >= 30) & (hr_raw <= 220),
            hr_raw, np.nan
        )
        hr_series = pd.Series(hr_masked, index=bvp_df.index, name='heart_rate')

        # Peaks only from high-quality windows
        peak_mask  = (sig['PPG_Peaks'].values[:len(bvp_df)] == 1) & (quality >= PPG_QUALITY_MIN)
        peak_times = bvp_df.index[peak_mask].tolist()
        return hr_series, peak_times
    except Exception:
        return None, []


# ===========================================================================
#  BUILD 10-SEC DATAFRAME FOR ONE PARTICIPANT-DAY
# ===========================================================================

def process_day_avros(avro_files):
    """
    Process all AVRO files for one participant-day.
    NeuroKit2 is applied per segment, then all results merged to 10-sec bins.
    Returns DataFrame with timestamp (Brussels tz-aware) + signal columns.
    """
    all_acc   = []
    all_temp  = []
    all_eda   = []
    all_bvp   = []
    all_hr    = []
    all_peaks = []

    for avro_path in sorted(avro_files):
        try:
            sensors = read_avro_sensors(avro_path)
        except Exception as ex:
            print(f'    [WARN] Cannot read {avro_path.name}: {ex}')
            continue

        if 'acc' in sensors:
            all_acc.append(sensors['acc'])

        if 'temperature' in sensors:
            all_temp.append(sensors['temperature'])

        if 'eda' in sensors:
            sr_eda   = sensors.get('eda_sf', SR_EDA)
            eda_proc = process_eda_segment(sensors['eda'], sr_eda)
            all_eda.append(eda_proc)

        if 'bvp' in sensors:
            sr_bvp = sensors.get('bvp_sf', SR_BVP)
            all_bvp.append(sensors['bvp'])
            hr_ser, peaks = process_bvp_segment(sensors['bvp'], sr_bvp)
            if hr_ser is not None:
                all_hr.append(hr_ser.to_frame())
            all_peaks.extend(peaks)

    frames_10s = {}

    if all_acc:
        acc = pd.concat(all_acc).sort_index()
        acc = acc[~acc.index.duplicated(keep='first')]
        acc_10s = acc.resample('10s').median()
        if all(c in acc_10s.columns for c in ['acc_x', 'acc_y', 'acc_z']):
            acc_10s['vector_magnitude'] = np.sqrt(
                acc_10s['acc_x']**2 + acc_10s['acc_y']**2 + acc_10s['acc_z']**2
            )
        frames_10s['acc'] = acc_10s

    if all_temp:
        temp = pd.concat(all_temp).sort_index()
        temp = temp[~temp.index.duplicated(keep='first')]
        frames_10s['temperature'] = temp.resample('10s').median()

    if all_eda:
        eda = pd.concat(all_eda).sort_index()
        eda = eda[~eda.index.duplicated(keep='first')]
        frames_10s['eda'] = eda.resample('10s').mean()

    if all_bvp:
        bvp = pd.concat(all_bvp).sort_index()
        bvp = bvp[~bvp.index.duplicated(keep='first')]
        frames_10s['bvp'] = bvp.resample('10s').mean()

    if all_hr:
        hr = pd.concat(all_hr).sort_index()
        hr = hr[~hr.index.duplicated(keep='first')]
        frames_10s['hr'] = hr.resample('10s').mean()

    if len(all_peaks) > 3:
        peak_times_sorted = sorted(all_peaks)
        peak_s  = np.array([t.timestamp() for t in peak_times_sorted])
        ibis    = np.diff(peak_s)
        valid_i = (ibis > 0.30) & (ibis < 2.0)
        ibis_c  = np.where(valid_i, ibis, np.nan)
        ibi_ser = pd.Series(ibis_c, index=pd.DatetimeIndex(peak_times_sorted[1:]))

        def _rmssd(x):
            v = x.dropna().values
            return float(np.sqrt(np.mean(np.diff(v) ** 2)) * 1000) if len(v) >= 3 else np.nan

        frames_10s['hrv'] = ibi_ser.resample('10s').agg(_rmssd).rename('hrv_rmssd').to_frame()

    if not frames_10s:
        return pd.DataFrame()

    merged = pd.concat(list(frames_10s.values()), axis=1)
    return merged.reset_index().rename(columns={'index': 'timestamp'})


# ===========================================================================
#  KEY / PHASE WINDOW HELPERS
# ===========================================================================

def parse_key_date(d: str) -> str:
    return datetime.strptime(f'{d}-2025', '%d-%b-%Y').strftime('%Y-%m-%d')


def build_phase_windows(key: pd.DataFrame) -> dict:
    """
    key.csv times are in UTC (e.g. BikeU_start = '14:01:56' means 14:01 UTC).
    Empatica timestamps are Brussels-local (UTC+2 in summer).
    We parse key times as UTC then convert to Brussels so both sides match.
    """
    windows = {}
    for _, row in key.iterrows():
        if pd.isna(row.get('Participant_ID')):
            continue
        pid  = int(row['Participant_ID'])
        date = parse_key_date(str(row['Date']))
        for ph in PHASES:
            s_col, e_col = f'{ph}_start', f'{ph}_end'
            if s_col not in row.index or pd.isna(row.get(s_col)) or pd.isna(row.get(e_col)):
                continue
            try:
                start = pd.Timestamp(f"{date} {row[s_col]}", tz='UTC').tz_convert('Europe/Brussels')
                end   = pd.Timestamp(f"{date} {row[e_col]}", tz='UTC').tz_convert('Europe/Brussels')
                if end < start:
                    end += pd.Timedelta(days=1)
                windows[(pid, ph)] = (start, end)
            except Exception:
                pass
    return windows


# ===========================================================================
#  QUALITY METRICS
# ===========================================================================

def compute_quality_metrics(df_day, windows, pid, date_str):
    m = {
        'pid': pid, 'date': date_str, 'total_rows': len(df_day),
        'eda_coverage': 0, 'eda_zero_pct': 0,
        'eda_mean': np.nan, 'eda_min': np.nan, 'eda_max': np.nan,
        'hr_coverage': 0, 'hr_mean': np.nan, 'hr_min': np.nan, 'hr_max': np.nan,
        'hrv_mean': np.nan, 'temp_coverage': 0, 'temp_mean': np.nan,
        'acc_coverage': 0, 'phases_covered': 0, 'phase_detail': {},
    }
    if df_day.empty:
        return m

    n = len(df_day)

    if 'eda' in df_day.columns:
        eda_v  = df_day['eda'].dropna()
        eda_nz = eda_v[eda_v > 0.01]
        m['eda_coverage'] = round(len(eda_v) / n * 100, 1)
        m['eda_zero_pct'] = round((len(eda_v) - len(eda_nz)) / max(len(eda_v), 1) * 100, 1)
        if len(eda_nz) > 0:
            m['eda_mean'] = round(eda_nz.mean(), 3)
            m['eda_min']  = round(eda_nz.min(), 3)
            m['eda_max']  = round(eda_nz.max(), 3)

    if 'heart_rate' in df_day.columns:
        hr_v  = df_day['heart_rate'].dropna()
        hr_ph = hr_v[(hr_v >= 30) & (hr_v <= 220)]
        m['hr_coverage'] = round(len(hr_v) / n * 100, 1)
        if len(hr_ph) > 0:
            m['hr_mean'] = round(hr_ph.mean(), 1)
            m['hr_min']  = round(hr_ph.min(), 1)
            m['hr_max']  = round(hr_ph.max(), 1)

    if 'hrv_rmssd' in df_day.columns:
        hrv_v = df_day['hrv_rmssd'].dropna()
        if len(hrv_v) > 0:
            m['hrv_mean'] = round(hrv_v.mean(), 1)

    if 'temperature' in df_day.columns:
        t_v  = df_day['temperature'].dropna()
        t_ph = t_v[(t_v > 25) & (t_v < 42)]
        m['temp_coverage'] = round(len(t_v) / n * 100, 1)
        if len(t_ph) > 0:
            m['temp_mean'] = round(t_ph.mean(), 1)

    if 'vector_magnitude' in df_day.columns:
        m['acc_coverage'] = round(df_day['vector_magnitude'].notna().mean() * 100, 1)

    ts = df_day['timestamp']
    phases_ok = 0
    for ph in PHASES:
        if (pid, ph) not in windows:
            m['phase_detail'][ph] = {'status': 'no_window', 'rows': 0, 'eda_cov': 0, 'hr_cov': 0}
            continue
        s, e   = windows[(pid, ph)]
        ph_df  = df_day[(ts >= s) & (ts <= e)]
        if len(ph_df) == 0:
            m['phase_detail'][ph] = {'status': 'no_data', 'rows': 0, 'eda_cov': 0, 'hr_cov': 0}
            continue
        phases_ok += 1
        eda_c = round(ph_df['eda'].notna().mean() * 100, 1) if 'eda' in ph_df else 0
        hr_c  = round(ph_df['heart_rate'].notna().mean() * 100, 1) if 'heart_rate' in ph_df else 0
        m['phase_detail'][ph] = {
            'status': 'ok', 'rows': len(ph_df),
            'eda_cov': eda_c, 'hr_cov': hr_c, 'start': s, 'end': e,
        }
    m['phases_covered'] = phases_ok
    return m


# ===========================================================================
#  (HTML quality report generation moved to quality_check_all.py)
# ===========================================================================


# ===========================================================================
#  MAIN
# ===========================================================================

def main():

    # ── Step 2: Build ───────────────────────────────────────────────────
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    out_csv = OUTPUTS / f'01_empatica_10sec{OUTPUT_SUFFIX}.csv'
    if out_csv.exists() and not FORCE_RERUN:
        print(f'Output file already exists: {out_csv}. Skipping Empatica build (FORCE_RERUN=False).')
        return

    # ── Collect: CPW → rawdata/01_empatica/ ──
    RAW_EMPA.mkdir(parents=True, exist_ok=True)
    existing = sorted(RAW_EMPA.iterdir())
    if not existing:
        key_dates = set()
        for _, row in pd.read_csv(KEY_FILE).iterrows():
            if not pd.isna(row.get('Participant_ID')):
                d = datetime.strptime(f'{row["Date"]}-2025', '%d-%b-%Y').strftime('%Y-%m-%d')
                key_dates.add(d)
        count = 0
        for d_folder in sorted(EMPA_CPW.iterdir()):
            if d_folder.is_dir() and d_folder.name in key_dates:
                shutil.copytree(d_folder, RAW_EMPA / d_folder.name, dirs_exist_ok=True)
                count += 1
        print(f'  Collected {count} date folders -> {RAW_EMPA}')
    else:
        print(f'  Already collected ({len(existing)} folders in {RAW_EMPA})')

    if not KEY_FILE.exists():
        print(f'ERROR: Key file not found: {KEY_FILE}'); return

    key = pd.read_csv(KEY_FILE)

    date_pid_map = {}
    for _, row in key.iterrows():
        if pd.isna(row.get('Participant_ID')):
            continue
        date_pid_map[parse_key_date(str(row['Date']))] = int(row['Participant_ID'])

    windows        = build_phase_windows(key)
    all_frames     = []

    all_pids = {int(row['Participant_ID']): parse_key_date(str(row['Date']))
                for _, row in key.iterrows()
                if not pd.isna(row.get('Participant_ID'))}
    processed_pids = set()

    for date_folder in sorted(RAW_EMPA.iterdir()):
        if not date_folder.is_dir():
            continue
        date_str = date_folder.name
        if date_str not in date_pid_map:
            print(f'  SKIP {date_str}: not in key.csv')
            continue
        pid = date_pid_map[date_str]

        avro_files = sorted(date_folder.rglob('*.avro'))
        if not avro_files:
            print(f'  WARNING: No AVRO for P{pid} ({date_str})')
            continue

        print(f'Processing P{pid} ({date_str}) — {len(avro_files)} AVRO files ...')

        # 1. Process AVRO segments -> 10-sec DataFrame
        df_10s = process_day_avros(avro_files)
        if df_10s.empty:
            print(f'  WARNING: Empty result for P{pid}')
            continue

        # 2. Drop rows where EDA AND temperature both == 0
        if 'eda' in df_10s.columns and 'temperature' in df_10s.columns:
            invalid = (df_10s['eda'] == 0) & (df_10s['temperature'] == 0)
            n_drop  = invalid.sum()
            df_10s  = df_10s[~invalid].copy()
            if n_drop > 0:
                print(f'  Dropped {n_drop} rows where EDA+temp both=0')

        ts = df_10s['timestamp']
        print(f'  Full-day rows: {len(df_10s)}   span: {ts.min()} -> {ts.max()}')

        # Assign PhaseID within phase windows (keep all rows, no clipping)
        df_10s['PhaseID'] = ''
        for phase in PHASES:
            if (pid, phase) not in windows:
                continue
            start, end = windows[(pid, phase)]
            mask = (ts >= start) & (ts <= end)
            df_10s.loc[mask, 'PhaseID'] = PHASE_ID[phase]
        sig_present = [c for c in SIGNAL_COLS if c in df_10s.columns]
        df_10s['Datetime'] = df_10s['timestamp'].dt.tz_localize(None)
        df_10s.insert(0, 'ParticipantID', f'P{pid}')
        out_cols = ['ParticipantID', 'PhaseID', 'Datetime'] + sig_present
        all_frames.append(df_10s[out_cols])
        processed_pids.add(pid)

    # ── Save empatica_corrected_10sec.csv ──────────────────────────────────────────────
    if all_frames:
        out = pd.concat(all_frames, ignore_index=True)
        out = out.sort_values(['ParticipantID', 'PhaseID', 'Datetime']).reset_index(drop=True)
        out = align_to_index(out, 'empatica')
        out_csv = OUTPUTS / f'01_empatica_10sec{OUTPUT_SUFFIX}.csv'
        out.to_csv(out_csv, index=False)
        print(f'\nSaved {len(out):,} rows -> {out_csv}')
        print(f'Participants: {sorted(out["ParticipantID"].unique())}')
        print(f'Phases:       {sorted(out["PhaseID"].unique())}')
    else:
        print('\nWARNING: No phase-filtered rows. Check AVRO timestamps vs key.csv windows.')

    print()
    print('  To generate quality HTML report, run: python quality_check_all.py')


if __name__ == '__main__':
    main()
