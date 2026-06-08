"""
Update all individual participant files in Atmo_lys with Apr_Jun processed data.
Preserves original last-modified timestamps.
"""
import os, time, shutil
from pathlib import Path
import pandas as pd
import numpy as np

ATMO_LYS = Path(r"C:\Users\pandya\OneDrive - UCL\Field experiment raw data\Complete Participantwise data\Atmo_lys")
APR_JUN  = Path(r"C:\Users\pandya\Documents\Github\docker\ExpData\temp\ExperimentData_20Apr_Jun.csv")

# Column mappings: (target_file_suffix, Apr_Jun_prefix)
SENSOR_MAP = [
    ('Atmo_left',  'atmotube_left'),
    ('Atmo_right', 'atmotube_right'),
    ('LYS1', 'LYS1'),
    ('LYS2', 'LYS2'),
]

# Load Apr_Jun file
aj = pd.read_csv(APR_JUN, low_memory=False)
aj['Datetime'] = pd.to_datetime(aj['Date'] + ' ' + aj['Time'], errors='coerce')

# Map Apr_Jun column names to merged naming (same MAP as notebook)
MAP = {
    'atmotube_left__temperature':'atmotube_left__atmotube_temperature',
    'atmotube_left__humidity':'atmotube_left__atmotube_humidity',
    'atmotube_left__pm1':'atmotube_left__atmotube_pm1',
    'atmotube_left__pm2.5':'atmotube_left__atmotube_pm2.5',
    'atmotube_left__pm10':'atmotube_left__atmotube_pm10',
    'atmotube_right__temperature':'atmotube_right__atmotube_temperature',
    'atmotube_right__humidity':'atmotube_right__atmotube_humidity',
    'atmotube_right__pm1':'atmotube_right__atmotube_pm1',
    'atmotube_right__pm2.5':'atmotube_right__atmotube_pm2.5',
    'atmotube_right__pm10':'atmotube_right__atmotube_pm10',
    'lys1__lux':'LYS1__lys_lux','lys1__kelvin':'LYS1__lys_kelvin',
    'lys1__medi':'LYS1__lys_medi','lys1__movement':'LYS1__lys_movement',
    'lys2__lux':'LYS2__lys_lux','lys2__kelvin':'LYS2__lys_kelvin',
    'lys2__medi':'LYS2__lys_medi','lys2__movement':'LYS2__lys_movement',
}
aj.rename(columns=MAP, inplace=True)

# Collect all files
all_files = list(ATMO_LYS.glob('*.csv'))
print(f"Found {len(all_files)} files in {ATMO_LYS}")

for fp in sorted(all_files):
    # Parse: {pid}_{suffix}.csv
    stem = fp.stem  # e.g. "4_Atmo_left"
    pid_str, suffix = stem.split('_', 1)
    pid = int(pid_str)
    
    # Find matching prefix
    prefix = None
    for s, p in SENSOR_MAP:
        if s == suffix:
            prefix = p
            break
    if prefix is None:
        print(f"  SKIP {fp.name}: unknown suffix '{suffix}'")
        continue
    
    # Record original timestamp
    orig_mtime = os.path.getmtime(fp)
    orig_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(orig_mtime))
    
    # Load original file
    df = pd.read_csv(fp)
    n_orig = len(df)
    df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
    
    # Get Apr_Jun columns for this sensor
    aj_cols = [c for c in aj.columns if c.startswith(f'{prefix}__')]
    # Target columns in individual file (strip prefix)
    target_cols = [c for c in df.columns if c != 'Datetime']
    
    # Map: target_col -> apr_jun_col
    col_map = {}
    for tc in target_cols:
        ajc = f'{prefix}__{tc}'
        if ajc in aj_cols:
            col_map[tc] = ajc
    
    if not col_map:
        print(f"  SKIP {fp.name}: no matching Apr_Jun columns for prefix '{prefix}'")
        continue
    
    # Merge Apr_Jun values by Datetime
    aj_pid = aj[aj['ParticipantID'] == pid].copy()
    
    # Merge: keep ALL original rows, replace values where Datetime matches
    merged = df[['Datetime']].merge(
        aj_pid[['Datetime'] + list(col_map.values())], on='Datetime', how='left')
    
    # Rename Apr_Jun columns back to target names
    merged.rename(columns={v: k for k, v in col_map.items()}, inplace=True)
    
    # For columns where merge produced NaN, keep original values
    for tc in target_cols:
        if tc in merged.columns:
            orig_vals = df[tc].values
            new_vals = merged[tc].values
            # Where merged has NaN, use original
            mask_nan = pd.isna(new_vals)
            new_vals[mask_nan] = orig_vals[mask_nan]
            df[tc] = new_vals
    
    # Save
    df.to_csv(fp, index=False)
    
    # Restore original modified time
    os.utime(fp, (orig_mtime, orig_mtime))
    
    new_mtime = os.path.getmtime(fp)
    new_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(new_mtime))
    match = "✅" if abs(new_mtime - orig_mtime) < 1 else "❌"
    
    print(f"  {fp.name}: {n_orig} rows, {len(col_map)} cols updated, mtime {orig_time_str} {match}")

print("\nDone! All files updated with timestamps preserved.")
