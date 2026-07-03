# Empatica build pipeline
# Restored from 01_empatica_build - Copy.ipynb build cells.
# Only addition: skip AVRO when full cached native/RRI intermediates are current.
# Production mode: TARGET_PARTICIPANTS = None runs all participants.

"""
Build native 10-second Empatica inputs from AVRO for the participants listed in
TARGET_PARTICIPANTS. AVRO field extraction follows Empatica's official Python
example, then adds study-specific batching, timestamp alignment, phase labelling,
and 10-second aggregation.

Source reference:
- Empatica Support, "How to access Avro files with Python"
  https://support.empatica.com/hc/en-us/articles/17405877853981-How-to-access-Avro-files-with-Python

Adaptation made here:
- The official example converts one AVRO file to raw sensor CSVs.
- This cell processes all selected participant AVRO files, converts timestamps,
  aligns data to study phases, and creates the native RRI table used by the HRV correction stage.
"""

from avro.datafile import DataFileReader
from avro.io import DatumReader
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import neurokit2 as nk
import warnings
import sys
import os
warnings.filterwarnings('ignore')

RAW_EMPA = Path(r'C:\Users\pandya\Documents\Github\docker\Paper3_Github\rawdata\01_empatica')
SCRIPTS_DIR = Path(r'C:\Users\pandya\Documents\Github\docker\ExpData\scripts')
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from _paths import assert_sensor_folder_clean
assert_sensor_folder_clean('empatica', RAW_EMPA)
KEY_FILE = Path(r'C:\Users\pandya\Documents\Github\docker\Paper3_Github\output\key.csv')
INDEX_CSV = Path(r'C:\Users\pandya\Documents\Github\docker\Paper3_Github\output\00_index_10sec.csv')
# For quick checks: Python is zero-indexed, but slot [1] is intentionally P8 per request.
PARTICIPANT_IDS = [None, 'P8', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P9', 'P10', 'P11', 'P12', 'P13', 'P14', 'P15', 'P16', 'P17', 'P18']
TARGET_PARTICIPANTS = None
RRI_FILTER_APPROACH = os.environ.get('EMPATICA_RRI_FILTER_APPROACH', 'kubios_vlow045s')
FULL_OUTPUT_INPUT_CSV = Path(r'C:\Users\pandya\Documents\Github\docker\Paper3_Github\output\01_empatica_native_input_intermediate.csv')
FULL_RRI_NATIVE_CSV = Path(r'C:\Users\pandya\Documents\Github\docker\Paper3_Github\output\01_empatica_rri_native.csv')
OUTPUT_SUFFIX = f"_{TARGET_PARTICIPANTS[0]}_{RRI_FILTER_APPROACH}_check" if TARGET_PARTICIPANTS else ''
OUTPUT_INPUT_CSV = FULL_OUTPUT_INPUT_CSV.with_name(f"{FULL_OUTPUT_INPUT_CSV.stem}{OUTPUT_SUFFIX}{FULL_OUTPUT_INPUT_CSV.suffix}")
RRI_NATIVE_CSV = FULL_RRI_NATIVE_CSV.with_name(f"{FULL_RRI_NATIVE_CSV.stem}{OUTPUT_SUFFIX}{FULL_RRI_NATIVE_CSV.suffix}")
RESULTS_CSV = Path(r'C:\Users\pandya\Documents\Github\docker\Paper3_Github\output\01_empatica_corrected_10sec.csv').with_name(f"01_empatica_corrected_10sec{OUTPUT_SUFFIX}.csv")
REUSE_AVRO_INTERMEDIATE = True
MIN_INTERMEDIATE_BYTES = 1024
PHASES = ['BikeU', 'WalkU', 'BikeG', 'WalkG', 'Tram', 'Indoor']
KEY_TO_DATA_OFFSET_HOURS = 2
TZ = 'Europe/Brussels'


def _has_cached_empatica_intermediate():
    required = [
        (FULL_OUTPUT_INPUT_CSV, {'ParticipantID', 'PhaseID', 'Datetime'}),
        (FULL_RRI_NATIVE_CSV, {'ParticipantID', 'peak_time', 'rri_ms'}),
    ]
    for path, columns in required:
        if not path.exists() or path.stat().st_size < MIN_INTERMEDIATE_BYTES:
            return False
        try:
            if not columns <= set(pd.read_csv(path, nrows=0).columns):
                return False
        except Exception:
            return False
    newest_avro = max((p.stat().st_mtime for p in RAW_EMPA.rglob('*.avro')), default=0)
    oldest_cache = min(FULL_OUTPUT_INPUT_CSV.stat().st_mtime, FULL_RRI_NATIVE_CSV.stat().st_mtime)
    return oldest_cache >= newest_avro

key = pd.read_csv(KEY_FILE)
key = key.dropna(subset=["Participant_ID"]).copy()
key["Participant_ID"] = key["Participant_ID"].astype(int)
key = key.drop_duplicates("Participant_ID", keep="first").sort_values("Participant_ID").reset_index(drop=True)


def parse_date(d):
    return datetime.strptime(f'{d}-2025', '%d-%b-%Y').strftime('%Y-%m-%d')


def _as_brussels_timestamp(value):
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize(TZ)
    return ts.tz_convert(TZ)


def build_windows(key_df):
    windows = {}
    clip_windows = {}
    for _, row in key_df.iterrows():
        if pd.isna(row.get('Participant_ID')):
            continue
        pid = f"P{int(row['Participant_ID'])}"
        if TARGET_PARTICIPANTS is not None and pid not in TARGET_PARTICIPANTS:
            continue
        date = parse_date(str(row['Date']))
        starts, ends = [], []
        for phase in PHASES:
            sc, ec = f'{phase}_start', f'{phase}_end'
            if pd.isna(row.get(sc)) or pd.isna(row.get(ec)):
                continue
            start = _as_brussels_timestamp(f'{date} {row[sc]}') + pd.Timedelta(hours=KEY_TO_DATA_OFFSET_HOURS)
            end = _as_brussels_timestamp(f'{date} {row[ec]}') + pd.Timedelta(hours=KEY_TO_DATA_OFFSET_HOURS)
            if end < start:
                end += pd.Timedelta(days=1)
            windows[(pid, phase)] = (start, end)
            starts.append(start)
            ends.append(end)
        if starts:
            clip_windows[pid] = (min(starts), max(ends))
    return windows, clip_windows


def build_labeled_segments(phase_windows, clip_windows):
    labeled = []
    for pid, (clip_start, clip_end) in clip_windows.items():
        segments = []
        for (phase_pid, phase), (start, end) in phase_windows.items():
            if phase_pid == pid:
                segments.append((start, end, phase))
        segments.sort(key=lambda x: x[0])
        cursor = clip_start
        for start, end, phase in segments:
            if start > cursor:
                labeled.append((pid, cursor, start, 'reststop'))
            labeled.append((pid, start, end, phase))
            if end > cursor:
                cursor = end
        if cursor < clip_end:
            labeled.append((pid, cursor, clip_end, 'reststop'))
    return labeled


def _us_to_brussels(unix_us_array):
    return pd.to_datetime(unix_us_array, unit='us', utc=True).tz_convert(TZ)


def read_avro_sensors(avro_path):
    reader = DataFileReader(open(str(avro_path), 'rb'), DatumReader())
    data = next(reader)
    reader.close()
    raw = data['rawData']
    sensors = {}
    avro_version = (
        data.get('schemaVersion', {}).get('major', 0),
        data.get('schemaVersion', {}).get('minor', 0),
        data.get('schemaVersion', {}).get('patch', 0),
    )

    try:
        a = raw['accelerometer']
        n = len(a['x'])
        if n > 0:
            params = a.get('imuParams', {})
            # Empatica official AVRO-to-CSV example uses physical/digital scaling
            # before v6.5.0 and imuParams['conversionFactor'] from v6.5.0 onward.
            # Source: https://support.empatica.com/hc/en-us/articles/17405877853981
            if avro_version >= (6, 5, 0) and 'conversionFactor' in params:
                scale = params['conversionFactor']
            elif {'physicalMax', 'physicalMin', 'digitalMax', 'digitalMin'} <= set(params.keys()):
                dp = params['physicalMax'] - params['physicalMin']
                dd = params['digitalMax'] - params['digitalMin']
                scale = dp / dd if dd else 1.0
            else:
                scale = 1.0
            us = np.round(a['timestampStart'] + np.arange(n) * (1e6 / a['samplingFrequency'])).astype(np.int64)
            sensors['acc'] = pd.DataFrame(
                {
                    'acc_x': np.array(a['x'], dtype=float) * scale,
                    'acc_y': np.array(a['y'], dtype=float) * scale,
                    'acc_z': np.array(a['z'], dtype=float) * scale,
                },
                index=_us_to_brussels(us),
            )
    except Exception:
        pass

    try:
        e = raw['eda']
        n = len(e['values'])
        if n > 0:
            sf = float(e['samplingFrequency'])
            vals = pd.to_numeric(pd.Series(e['values']), errors='coerce').to_numpy(dtype=float)
            # EDA conductance is non-negative. Negative AVRO samples are treated as
            # invalid measurements, not forced to zero, so device-floor problems remain visible.
            vals[vals < 0] = np.nan
            us = np.round(e['timestampStart'] + np.arange(n) * (1e6 / sf)).astype(np.int64)
            sensors['eda'] = pd.DataFrame({'empatica__eda_scl_usiemens': vals}, index=_us_to_brussels(us))
    except Exception:
        pass

    try:
        t = raw['temperature']
        n = len(t['values'])
        if n > 0:
            us = np.round(t['timestampStart'] + np.arange(n) * (1e6 / t['samplingFrequency'])).astype(np.int64)
            sensors['temperature'] = pd.DataFrame({'temperature': t['values']}, index=_us_to_brussels(us))
    except Exception:
        pass

    try:
        peaks = raw.get('systolicPeaks', {}).get('peaksTimeNanos', [])
        if len(peaks) >= 2:
            peak_ts = pd.to_datetime(np.array(peaks, dtype=np.int64), unit='ns', utc=True).tz_convert(TZ)
            sensors['peaks'] = pd.DataFrame({'peak_time': peak_ts})
    except Exception:
        pass

    return sensors


phase_windows, clip_windows = build_windows(key)
labeled_segments = build_labeled_segments(phase_windows, clip_windows)

if REUSE_AVRO_INTERMEDIATE and _has_cached_empatica_intermediate():
    print(f'Reusing cached Empatica AVRO intermediate -> {FULL_OUTPUT_INPUT_CSV}')
    print(f'Reusing cached Empatica native RRI -> {FULL_RRI_NATIVE_CSV}')
    all_out_df = pd.read_csv(FULL_OUTPUT_INPUT_CSV, parse_dates=['Datetime'])
    all_rri_df = pd.read_csv(FULL_RRI_NATIVE_CSV, parse_dates=['peak_time'])
    if TARGET_PARTICIPANTS is not None:
        all_out_df = all_out_df[all_out_df['ParticipantID'].astype(str).isin(TARGET_PARTICIPANTS)].copy()
        all_rri_df = all_rri_df[all_rri_df['ParticipantID'].astype(str).isin(TARGET_PARTICIPANTS)].copy()
    all_out = [all_out_df]
    all_rri = [all_rri_df]
    clip_windows = {}
else:
    print('Building Empatica AVRO intermediate from staged AVRO files')
    all_out = []
    all_rri = []

for pid in sorted(clip_windows):
    participant_row = key[key['Participant_ID'].astype('Int64') == int(pid[1:])]
    date_folder = parse_date(str(participant_row.iloc[0]['Date']))
    raw_dir = RAW_EMPA / date_folder
    avro_files = sorted(raw_dir.rglob('*.avro'))
    clip_start, clip_end = clip_windows[pid]

    eda_parts, temp_parts, acc_parts, peak_rows = [], [], [], []
    for avro_path in avro_files:
        try:
            sensors = read_avro_sensors(avro_path)
        except Exception as exc:
            print(f'  [WARN] {pid} {avro_path.name}: {exc}')
            continue
        if 'eda' in sensors:
            eda_parts.append(sensors['eda'])
        if 'temperature' in sensors:
            temp_parts.append(sensors['temperature'])
        if 'acc' in sensors:
            acc_parts.append(sensors['acc'])
        if 'peaks' in sensors:
            peak_rows.append(sensors['peaks'])

    frames = []
    if acc_parts:
        acc = pd.concat(acc_parts).sort_index()
        acc = acc[(acc.index >= clip_start) & (acc.index <= clip_end)]
        acc = acc[~acc.index.duplicated(keep='first')]
        acc_10 = acc.resample('10s').median()
        acc_10['vector_magnitude'] = np.sqrt((acc_10[['acc_x', 'acc_y', 'acc_z']] ** 2).sum(axis=1))
        frames.append(acc_10)
    if temp_parts:
        temp = pd.concat(temp_parts).sort_index()
        temp = temp[(temp.index >= clip_start) & (temp.index <= clip_end)]
        temp = temp[~temp.index.duplicated(keep='first')]
        frames.append(temp.resample('10s').median())
    if eda_parts:
        eda = pd.concat(eda_parts).sort_index()
        eda = eda[(eda.index >= clip_start) & (eda.index <= clip_end)]
        eda = eda[~eda.index.duplicated(keep='first')]
        eda_values = pd.to_numeric(eda['empatica__eda_scl_usiemens'], errors='coerce')
        valid = eda_values.notna()
        if valid.any():
            eda_signals, _ = nk.eda_process(eda_values.loc[valid].to_numpy(), sampling_rate=4)
            eda.loc[valid, 'eda_tonic'] = eda_signals['EDA_Tonic'].to_numpy()
            eda.loc[valid, 'eda_tonic'] = eda.loc[valid, 'eda_tonic'].mask(eda.loc[valid, 'eda_tonic'] < 0)
            eda.loc[valid, 'eda_phasic'] = eda_signals['EDA_Phasic'].to_numpy()
        frames.append(eda[['empatica__eda_scl_usiemens', 'eda_tonic', 'eda_phasic']].resample('10s').mean())

    raw10 = pd.concat(frames, axis=1).sort_index() if frames else pd.DataFrame()
    if not raw10.empty:
        raw10 = raw10[~raw10.index.duplicated(keep='first')]
        raw10.index = raw10.index.tz_localize(None)

    full_index = pd.date_range(clip_start.tz_localize(None).floor('10s'), clip_end.tz_localize(None).ceil('10s'), freq='10s')
    out = pd.DataFrame({'Datetime': full_index})
    out['ParticipantID'] = pid
    out['PhaseID'] = 'reststop'
    if not raw10.empty:
        out = out.merge(raw10.reset_index().rename(columns={'index': 'Datetime'}), on='Datetime', how='left')
    for seg_pid, seg_start, seg_end, phase_name in labeled_segments:
        if seg_pid != pid:
            continue
        mask = (out['Datetime'] >= seg_start.tz_localize(None)) & (out['Datetime'] <= seg_end.tz_localize(None))
        out.loc[mask, 'PhaseID'] = phase_name

    if peak_rows:
        peaks = pd.concat(peak_rows, ignore_index=True).drop_duplicates().sort_values('peak_time').reset_index(drop=True)
        peaks = peaks[(peaks['peak_time'] >= clip_start) & (peaks['peak_time'] <= clip_end)].copy()
        peak_times = pd.to_datetime(peaks['peak_time']).dt.tz_localize(None)
        peak_ns = peak_times.astype('int64').to_numpy()
        rri_ms = np.diff(peak_ns) / 1e6
        rri_times = peak_times.iloc[1:].reset_index(drop=True)
        rri_df = pd.DataFrame({'ParticipantID': pid, 'peak_time': rri_times, 'rri_ms': rri_ms})
        all_rri.append(rri_df)

        hr_vals = np.full(len(out), np.nan)
        for i, dt in enumerate(out['Datetime']):
            start = dt - pd.Timedelta(seconds=30)
            end = dt + pd.Timedelta(seconds=30)
            mask = (rri_times >= start) & (rri_times < end)
            win = rri_ms[mask]
            win = win[(win >= 250) & (win <= 2500)]
            if len(win) >= 5:
                med = np.nanmedian(win)
                if med > 0:
                    hr_vals[i] = 60000.0 / med
        out['empatica__pulse_rate_bpm'] = hr_vals
    else:
        out['empatica__pulse_rate_bpm'] = np.nan

    keep = ['ParticipantID', 'PhaseID', 'Datetime', 'empatica__pulse_rate_bpm', 'empatica__eda_scl_usiemens', 'eda_tonic', 'eda_phasic', 'temperature', 'acc_x', 'acc_y', 'acc_z', 'vector_magnitude']
    for col in keep:
        if col not in out.columns:
            out[col] = np.nan
    all_out.append(out[keep].sort_values('Datetime').reset_index(drop=True))

all_out_df = pd.concat(all_out, ignore_index=True).sort_values(['ParticipantID', 'Datetime']).reset_index(drop=True)
all_rri_df = pd.concat(all_rri, ignore_index=True).sort_values(['ParticipantID', 'peak_time']).reset_index(drop=True)

if INDEX_CSV.exists():
    index_df = pd.read_csv(INDEX_CSV, parse_dates=['Datetime'])
    participant_ids = set(all_out_df['ParticipantID'].astype(str))
    index_df = index_df[index_df['ParticipantID'].astype(str).isin(participant_ids)].copy()
    index_df['ParticipantID'] = index_df['ParticipantID'].astype(str)
    index_df['Datetime'] = pd.to_datetime(index_df['Datetime'])
    index_df['PhaseID'] = index_df['PhaseID'].where(index_df['PhaseID'].notna(), '').astype(str).replace({'': np.nan})

    all_out_df['ParticipantID'] = all_out_df['ParticipantID'].astype(str)
    all_out_df['Datetime'] = pd.to_datetime(all_out_df['Datetime'])
    value_cols = [c for c in all_out_df.columns if c not in ['ParticipantID', 'PhaseID', 'Datetime']]

    labeled_index = index_df[['ParticipantID', 'Datetime', 'PhaseID']].rename(columns={'PhaseID': 'PhaseID_index'})
    all_out_df = all_out_df.merge(labeled_index, on=['ParticipantID', 'Datetime'], how='left')
    all_out_df['PhaseID'] = all_out_df['PhaseID_index'].combine_first(all_out_df['PhaseID'])
    all_out_df['PhaseID'] = all_out_df['PhaseID'].where(all_out_df['PhaseID'].notna(), 'reststop')
    all_out_df = all_out_df[['ParticipantID', 'PhaseID', 'Datetime'] + value_cols]

OUTPUT_INPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
all_out_df.to_csv(OUTPUT_INPUT_CSV, index=False, lineterminator='\n')
try:
    all_rri_df.to_csv(RRI_NATIVE_CSV, index=False, lineterminator='\n')
    print(f'Saved native RRI input -> {RRI_NATIVE_CSV}')
except PermissionError as exc:
    print(f'Skipped locked native RRI CSV -> {RRI_NATIVE_CSV}: {exc}')
print(f'Saved native 10-sec input -> {OUTPUT_INPUT_CSV}')
print(all_out_df['ParticipantID'].value_counts().sort_index().to_string())


"""
Empatica HRV + EDA correction stage.

Inputs are the staged/raw-derived Empatica 10-second table and native RRI table
created earlier in this notebook. This cell does not gap-fill signal columns;
missing values remain NaN. HRV frequency-domain metrics are computed with
NeuroKit2's published `hrv_frequency` method on valid native RRI intervals.
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import neurokit2 as nk
warnings.filterwarnings('ignore')

INPUT_CSV = str(OUTPUT_INPUT_CSV)
RRI_NATIVE_CSV = str(RRI_NATIVE_CSV)
RESULTS_CSV = str(RESULTS_CSV)
FINAL_RESULTS_CSV = RESULTS_CSV
KEY_FILE = r'C:\Users\pandya\Documents\Github\docker\Paper3_Github\output\key.csv'

HR_WINDOW_SECS = 60
RMSSD_WINDOW_SECS = 30
SDNN_WINDOW_SECS = 120
MIN_SDNN_WINDOW_SECS = 120
FD_WINDOW_SECS = 300
RRI_FILTER_MODE = os.environ.get('EMPATICA_RRI_FILTER_MODE', 'kubios_abs_local')
RRI_ARTIFACT_THRESHOLD = float(os.environ.get('EMPATICA_RRI_ARTIFACT_THRESHOLD', '0.45'))
MIN_RAW_RRI_FOR_HR = 8
MIN_VALID_RRI_FOR_RMSSD = 8
MIN_VALID_RRI_FOR_HRV = 40
MAX_ARTIFACT_PCT_FOR_RMSSD = 25
MAX_ARTIFACT_PCT_FOR_HRV = 25
PHASES_TO_KEEP = ['BikeU', 'WalkU', 'BikeG', 'WalkG', 'Tram', 'reststop']
FD_NK_MAP = {
    'HRV_LF': 'hrv_fd_lf',
    'HRV_HF': 'hrv_fd_hf',
    'HRV_LFHF': 'hrv_fd_lf_hf_ratio',
    'HRV_LFn': 'hrv_fd_lfnu',
    'HRV_HFn': 'hrv_fd_hfnu',
    'HRV_VLF': 'hrv_fd_vlf',
    'HRV_TP': 'hrv_fd_total_power',
}
FD_OUTPUT_COLS = list(FD_NK_MAP.values())
FD_AUDIT_COLS = [
    'hrv_fd_5min_available',
    'hrv_fd_5min_segment_start',
    'hrv_fd_5min_segment_end',
    'hrv_fd_5min_valid_rri_count',
    'hrv_fd_5min_removed_pct',
    'hrv_fd_5min_max_gap_seconds',
    'hrv_fd_5min_valid_span_seconds',
]


def filter_rri_artifacts_keep_index(rri_vals: np.ndarray, threshold: float):
    vals = np.asarray(rri_vals, dtype=float)
    original_indices = np.arange(len(vals))
    if len(vals) < 5:
        return vals, original_indices
    if RRI_FILTER_MODE == 'kubios_abs_local':
        threshold_ms = threshold * 1000.0
        reference = pd.Series(vals).rolling(11, center=True, min_periods=3).median().to_numpy()
        fallback = np.nanmedian(vals)
        reference = np.where(np.isfinite(reference), reference, fallback)
        keep = np.abs(vals - reference) <= threshold_ms
        return vals[keep], original_indices[keep]
    changed = True
    while changed and len(vals) >= 5:
        changed = False
        diffs = np.abs(np.diff(vals))
        bad = np.zeros(len(vals), dtype=bool)
        for i in range(len(diffs)):
            reference = np.median(vals[max(0, i - 2):min(len(vals), i + 3)])
            if reference > 0 and diffs[i] > threshold * reference:
                if abs(vals[i] - reference) >= abs(vals[i + 1] - reference):
                    bad[i] = True
                else:
                    bad[i + 1] = True
        if bad.any():
            vals = vals[~bad]
            original_indices = original_indices[~bad]
            changed = True
    return vals, original_indices



def parse_key_date_local(d):
    from datetime import datetime
    return datetime.strptime(f"{d}-2025", "%d-%b-%Y").strftime("%Y-%m-%d")


def build_phase_windows_from_key(key_df):
    windows = {}
    for _, row in key_df.iterrows():
        if pd.isna(row.get('Participant_ID')):
            continue
        pid = f"P{int(row['Participant_ID'])}"
        date_str = parse_key_date_local(str(row['Date']))
        for phase in ['BikeU', 'WalkU', 'BikeG', 'WalkG', 'Tram']:
            sc = f'{phase}_start'
            ec = f'{phase}_end'
            if pd.isna(row.get(sc)) or pd.isna(row.get(ec)):
                continue
            start = pd.Timestamp(f'{date_str} {row[sc]}') + pd.Timedelta(hours=2)
            end = pd.Timestamp(f'{date_str} {row[ec]}') + pd.Timedelta(hours=2)
            if end < start:
                end += pd.Timedelta(days=1)
            windows[(pid, phase)] = (start, end)
    return windows


def bounded_window(t_center, phase_start, phase_end, target_secs, min_secs):
    half = pd.Timedelta(seconds=target_secs / 2.0)
    start = max(phase_start, t_center - half)
    end = min(phase_end, t_center + half)
    if (end - start).total_seconds() < min_secs:
        return None, None
    return start, end


def compute_hr_for_row(t_center, peak_times, rri_ms):
    half = pd.Timedelta(seconds=HR_WINDOW_SECS / 2.0)
    mask = (peak_times >= t_center - half) & (peak_times < t_center + half)
    vals = pd.to_numeric(rri_ms.loc[mask], errors='coerce').to_numpy(dtype=float)
    vals = vals[np.isfinite(vals) & (vals >= 250) & (vals <= 2500)]
    if len(vals) < MIN_RAW_RRI_FOR_HR:
        return np.nan, len(vals)
    med = np.nanmedian(vals)
    if not np.isfinite(med) or med <= 0:
        return np.nan, len(vals)
    return 60000.0 / med, len(vals)


def compute_rmssd_for_row(t_center, phase_start, phase_end, peak_times, rri_ms):
    win_start, win_end = bounded_window(t_center, phase_start, phase_end, RMSSD_WINDOW_SECS, 30)
    if win_start is None:
        return np.nan
    mask = (peak_times >= win_start) & (peak_times < win_end)
    raw_vals = pd.to_numeric(rri_ms.loc[mask], errors='coerce').to_numpy(dtype=float)
    n_original = len(raw_vals)
    raw_indices = np.arange(n_original)
    phys_mask = np.isfinite(raw_vals) & (raw_vals >= 250) & (raw_vals <= 2500)
    vals = raw_vals[phys_mask]
    phys_raw_indices = raw_indices[phys_mask]
    if n_original < MIN_VALID_RRI_FOR_RMSSD:
        return np.nan
    clean_rri, kept_indices = filter_rri_artifacts_keep_index(vals, RRI_ARTIFACT_THRESHOLD)
    kept_raw_indices = phys_raw_indices[kept_indices]
    artifact_pct = 100 * (n_original - len(clean_rri)) / n_original if n_original else np.nan
    if len(clean_rri) < MIN_VALID_RRI_FOR_RMSSD or artifact_pct > MAX_ARTIFACT_PCT_FOR_RMSSD:
        return np.nan
    if len(kept_raw_indices) < 2:
        return np.nan
    consecutive_mask = np.diff(kept_raw_indices) == 1
    successive_differences = np.diff(clean_rri)[consecutive_mask]
    if len(successive_differences) < 6:
        return np.nan
    return float(np.sqrt(np.mean(successive_differences ** 2)))


def compute_sdnn_for_row(t_center, phase_start, phase_end, peak_times, rri_ms):
    win_start, win_end = bounded_window(t_center, phase_start, phase_end, SDNN_WINDOW_SECS, MIN_SDNN_WINDOW_SECS)
    if win_start is None:
        return np.nan
    mask = (peak_times >= win_start) & (peak_times < win_end)
    win_df = pd.DataFrame({
        'peak_time': peak_times.loc[mask].reset_index(drop=True),
        'rri_ms': pd.to_numeric(rri_ms.loc[mask], errors='coerce').reset_index(drop=True),
    })
    n_original = len(win_df)
    if n_original == 0:
        return np.nan
    phys_mask = win_df['rri_ms'].notna() & win_df['rri_ms'].between(250, 2500)
    win_df = win_df.loc[phys_mask].reset_index(drop=True)
    if len(win_df) == 0:
        return np.nan
    clean_rri, kept_indices = filter_rri_artifacts_keep_index(win_df['rri_ms'].to_numpy(dtype=float), RRI_ARTIFACT_THRESHOLD)
    time_clean = win_df.loc[kept_indices, 'peak_time'].reset_index(drop=True)
    n_valid = len(clean_rri)
    artifact_pct = 100 * (n_original - n_valid) / n_original if n_original else np.nan
    if n_valid < MIN_VALID_RRI_FOR_HRV or artifact_pct > MAX_ARTIFACT_PCT_FOR_HRV:
        return np.nan
    if len(time_clean) < 2:
        return np.nan
    try:
        hrv_td = nk.hrv_time({'RRI': clean_rri, 'RRI_Time': (time_clean - time_clean.iloc[0]).dt.total_seconds().to_numpy()}, show=False)
        if 'HRV_SDNN' in hrv_td.columns:
            return float(hrv_td['HRV_SDNN'].iloc[0])
    except Exception:
        pass
    return np.nan


def evaluate_fd_candidate(phase_start, phase_end, start, peak_times, rri_ms):
    end = start + pd.Timedelta(seconds=FD_WINDOW_SECS)
    if start < phase_start or end > phase_end:
        return None
    mask = (peak_times >= start) & (peak_times < end)
    win_df = pd.DataFrame({
        'peak_time': peak_times.loc[mask].reset_index(drop=True),
        'rri_ms': pd.to_numeric(rri_ms.loc[mask], errors='coerce').reset_index(drop=True),
    })
    n_original = len(win_df)
    if n_original == 0:
        return None
    phys_mask = win_df['rri_ms'].notna() & win_df['rri_ms'].between(250, 2500)
    win_df = win_df.loc[phys_mask].reset_index(drop=True)
    if len(win_df) == 0:
        return None
    clean_rri, kept_indices = filter_rri_artifacts_keep_index(win_df['rri_ms'].to_numpy(dtype=float), RRI_ARTIFACT_THRESHOLD)
    time_clean = win_df.loc[kept_indices, 'peak_time'].reset_index(drop=True)
    n_valid = len(clean_rri)
    artifact_pct = 100 * (n_original - n_valid) / n_original if n_original else np.nan
    if n_valid < MIN_VALID_RRI_FOR_HRV or artifact_pct > MAX_ARTIFACT_PCT_FOR_HRV:
        return None
    if len(time_clean) < 2:
        return None
    gaps = time_clean.diff().dt.total_seconds().dropna()
    edge_start_gap = (time_clean.iloc[0] - start).total_seconds()
    edge_end_gap = (end - time_clean.iloc[-1]).total_seconds()
    max_gap = max([float(gaps.max()) if len(gaps) else 0.0, float(edge_start_gap), float(edge_end_gap)])
    rri_time_seconds = (time_clean - time_clean.iloc[0]).dt.total_seconds().to_numpy()
    hrv_input = {'RRI': clean_rri, 'RRI_Time': rri_time_seconds}
    try:
        hrv_fd = nk.hrv_frequency(hrv_input, interpolation_rate=4, psd_method='welch', normalize=False, silent=True, show=False)
    except Exception:
        return None
    valid_span_seconds = float((time_clean.iloc[-1] - time_clean.iloc[0]).total_seconds())
    removed_pct = float(100 * (n_original - n_valid) / n_original) if n_original else np.nan
    out = {
        'hrv_fd_5min_available': True,
        'hrv_fd_5min_segment_start': start,
        'hrv_fd_5min_segment_end': end,
        'hrv_fd_5min_valid_rri_count': int(n_valid),
        'hrv_fd_5min_removed_pct': removed_pct,
        'hrv_fd_5min_max_gap_seconds': max_gap,
        'hrv_fd_5min_valid_span_seconds': valid_span_seconds,
        'artifact_pct': float(artifact_pct),
        'max_gap_seconds': max_gap,
    }
    for nk_key, out_col in FD_NK_MAP.items():
        out[out_col] = float(hrv_fd[nk_key].iloc[0]) if nk_key in hrv_fd.columns else np.nan
    lf = out.get('hrv_fd_lf', np.nan)
    hf = out.get('hrv_fd_hf', np.nan)
    if pd.notna(lf) and pd.notna(hf) and (lf + hf) > 0:
        out['hrv_fd_lfnu'] = float(100.0 * lf / (lf + hf))
        out['hrv_fd_hfnu'] = float(100.0 * hf / (lf + hf))
    else:
        out['hrv_fd_lfnu'] = np.nan
        out['hrv_fd_hfnu'] = np.nan
    return out


def compute_fd_for_row(t_center, phase_start, phase_end, peak_times, rri_ms):
    win_start, _ = bounded_window(t_center, phase_start, phase_end, FD_WINDOW_SECS, FD_WINDOW_SECS)
    if win_start is None:
        return None
    return evaluate_fd_candidate(phase_start, phase_end, win_start, peak_times, rri_ms)


df = pd.read_csv(INPUT_CSV, parse_dates=['Datetime'])
base_df = df.copy(deep=True)
if 'all_rri_df' in globals():
    rri_native = all_rri_df.copy()
else:
    rri_native = pd.read_csv(RRI_NATIVE_CSV, parse_dates=['peak_time'])
rri_native['ParticipantID'] = rri_native['ParticipantID'].astype(str)
rri_native['peak_time'] = pd.to_datetime(rri_native['peak_time'], errors='coerce')
if getattr(rri_native['peak_time'].dt, 'tz', None) is not None:
    rri_native['peak_time'] = rri_native['peak_time'].dt.tz_convert('Europe/Brussels').dt.tz_localize(None)

for col in ['hrv_td_sdnn'] + FD_OUTPUT_COLS + FD_AUDIT_COLS:
    if col not in df.columns:
        df[col] = np.nan

df['hrv_td_rmssd'] = np.nan
df['hrv_td_sdnn'] = np.nan
for col in FD_OUTPUT_COLS:
    df[col] = np.nan
for col in FD_AUDIT_COLS:
    if col == 'hrv_fd_5min_available':
        df[col] = False
    else:
        df[col] = np.nan

missing_phase_rows = []
coverage_rows = []

phase_key_df = pd.read_csv(KEY_FILE)
phase_key_df = phase_key_df.dropna(subset=["Participant_ID"]).copy()
phase_key_df["Participant_ID"] = phase_key_df["Participant_ID"].astype(int)
phase_key_df = phase_key_df.drop_duplicates("Participant_ID", keep="first").sort_values("Participant_ID").reset_index(drop=True)
phase_windows = build_phase_windows_from_key(phase_key_df)

for pid in sorted(df['ParticipantID'].astype(str).unique()):
    df_pid = df[df['ParticipantID'].astype(str) == pid].copy().sort_values('Datetime')
    segment_id = (df_pid['PhaseID'] != df_pid['PhaseID'].shift()).cumsum()
    for _, sub in df_pid.groupby(segment_id):
        phase = str(sub['PhaseID'].iloc[0])
        if phase not in PHASES_TO_KEEP:
            continue
        if (pid, phase) in phase_windows:
            phase_start, phase_end = phase_windows[(pid, phase)]
        else:
            phase_start = sub['Datetime'].min()
            phase_end = sub['Datetime'].max() + pd.Timedelta(seconds=10)
        rri_seg = rri_native[(rri_native['ParticipantID'] == pid) & (rri_native['peak_time'] >= phase_start) & (rri_native['peak_time'] < phase_end)].copy()
        peak_times = pd.to_datetime(rri_seg['peak_time'])
        rri_ms = rri_seg['rri_ms']

        for row_i, t_center in zip(sub.index, sub['Datetime']):
            df.loc[row_i, 'hrv_td_rmssd'] = compute_rmssd_for_row(
                t_center,
                phase_start,
                phase_end,
                peak_times,
                rri_ms,
            )
            df.loc[row_i, 'hrv_td_sdnn'] = compute_sdnn_for_row(t_center, phase_start, phase_end, peak_times, rri_ms)

        if phase != 'reststop':
            has_fd = False
            for row_i, t_center in zip(sub.index, sub['Datetime']):
                fd = compute_fd_for_row(t_center, phase_start, phase_end, peak_times, rri_ms)
                if fd is None:
                    continue
                has_fd = True
                for col in FD_OUTPUT_COLS + FD_AUDIT_COLS:
                    df.loc[row_i, col] = fd.get(col, np.nan)
            coverage_rows.append({'ParticipantID': pid, 'PhaseID': phase, 'available': int(has_fd)})
            if not has_fd:
                missing_phase_rows.append({'ParticipantID': pid, 'PhaseID': phase})

if 'heart_rate' not in df.columns and 'empatica__pulse_rate_bpm' in df.columns:
    df['heart_rate'] = df['empatica__pulse_rate_bpm']
Path(RESULTS_CSV).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(RESULTS_CSV, index=False, lineterminator='\n')
print(f'Saved corrected Empatica output -> {RESULTS_CSV}')

print(f'Row count: {len(df)}')
_expected_df = pd.read_csv(r'C:\Users\pandya\Documents\Github\docker\Paper3_Github\output\00_index_10sec.csv', usecols=['ParticipantID'])
if TARGET_PARTICIPANTS is not None:
    _expected_df = _expected_df[_expected_df['ParticipantID'].astype(str).isin(TARGET_PARTICIPANTS)]
expected_rows = len(_expected_df)
if len(df) != expected_rows:
    raise RuntimeError(f'Expected {expected_rows} rows from 00_index_10sec.csv, found {len(df)}')

if not base_df[['ParticipantID', 'PhaseID', 'Datetime']].equals(df[['ParticipantID', 'PhaseID', 'Datetime']]):
    raise RuntimeError('ParticipantID, PhaseID, or Datetime changed.')

allowed_changed = set(['hrv_td_rmssd', 'hrv_td_sdnn'] + FD_OUTPUT_COLS + FD_AUDIT_COLS)
changed_cols = [col for col in df.columns if col in base_df.columns and not base_df[col].equals(df[col])]
unexpected = [col for col in changed_cols if col not in allowed_changed]
if unexpected:
    raise RuntimeError(f'Non-target columns changed: {unexpected}')
print('ParticipantID, PhaseID, Datetime unchanged.')
unchanged_signal_cols = ['empatica__pulse_rate_bpm','empatica__eda_scl_usiemens','eda_tonic','eda_phasic','temperature','acc_x','acc_y','acc_z','vector_magnitude','heart_rate']
changed_signal_cols = [c for c in unchanged_signal_cols if c in base_df.columns and not base_df[c].equals(df[c])]
if changed_signal_cols:
    raise RuntimeError(f'Signal columns unexpectedly changed: {changed_signal_cols}')
print('HR, EDA, temperature and accelerometer columns unchanged.')
print('RMSSD uses fixed 30-second windows.')
print('SDNN uses fixed 120-second windows.')
print('LF, HF and LF/HF use rolling 300-second windows aligned to the 10-second grid.')
