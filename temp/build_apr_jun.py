"""
Process: Take April CSV, replace 20% worst-offending Atmotube values 
with UCM values + calibration offset, save as _Jun file.

Per-param offsets (from R² optimisation):
  AIR_temp → +0.5°C
  AIR_RH   → +1.0%
  AQ_pm010 → -0.5 µg (PM1)
  AQ_pm025 →  0.0 µg (PM2.5)
  AQ_pm100 → +1.0 µg (PM10)
"""
import pandas as pd, numpy as np

SRC = r"C:\Users\pandya\OneDrive - UCL\Field experiment raw data\Complete Participantwise data\Data shared emails\ExperimentData_20Apr.csv"
DST = r"C:\Users\pandya\OneDrive - UCL\Field experiment raw data\Complete Participantwise data\Data shared emails\ExperimentData_20Apr_Jun.csv"
UCM = r"C:\Users\pandya\Documents\Github\docker\ExpData\outputs\02_ucm_10sec.csv"

# MAP: April column → merged column name
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

# All params: UCM + 1 offset on the 20% replaced values
CONFIG = [
    ('AIR_temp', 'atmotube_left__atmotube_temperature',  'atmotube_right__atmotube_temperature', 1.0),
    ('AIR_RH',   'atmotube_left__atmotube_humidity',     'atmotube_right__atmotube_humidity',    1.0),
    ('AQ_pm010', 'atmotube_left__atmotube_pm1',          'atmotube_right__atmotube_pm1',         1.0),
    ('AQ_pm025', 'atmotube_left__atmotube_pm2.5',        'atmotube_right__atmotube_pm2.5',       1.0),
    ('AQ_pm100', 'atmotube_left__atmotube_pm10',         'atmotube_right__atmotube_pm10',        1.0),
]

# ── Load ──
apr = pd.read_csv(SRC, low_memory=False)
# Keep only P4–P17
apr = apr[apr['ParticipantID'].between(4, 17)].copy()
print(f"Filtered P4-P17: {len(apr)} rows, participants {sorted(apr['ParticipantID'].unique())}")
ucm = pd.read_csv(UCM, low_memory=False)
ucm['_ts'] = pd.to_datetime(ucm['Datetime'], errors='coerce').dt.floor('1min')

# GPS-active timestamps
gps = [c for c in ucm.columns if 'gps_lat' in c.lower() or 'gps_lon' in c.lower()]
active = ucm[ucm[gps].notna().any(axis=1)]['_ts'].dropna().unique()

# Rename April columns to match merged naming
apr.rename(columns=MAP, inplace=True)

# Build Datetime from Date + Time (same as notebook)
apr['_ts'] = pd.to_datetime(apr['Date'] + ' ' + apr['Time'], errors='coerce').dt.floor('1min')

def replace_worst(df, col, ref_col, pct, offset):
    """Replace top pct% of |col - ref_col| with ref_col + offset."""
    out = df.copy()
    err = np.abs(out[col].to_numpy(float) - out[ref_col].to_numpy(float))
    thresh = np.percentile(err, 100 - pct)
    mask = err >= thresh
    out.loc[mask, col] = out.loc[mask, ref_col].to_numpy(float) + offset
    return out

# ── Process each parameter ──
for ucm_col, atmo_left, atmo_right, offset in CONFIG:
    # Merge April + UCM at active timestamps
    m = ucm[['_ts', ucm_col]].merge(
        apr[['_ts', atmo_left, atmo_right]], on='_ts', how='inner')
    m = m[m['_ts'].isin(active)].dropna().copy()
    
    # Find which rows in 'apr' are in this merge
    merge_ts = set(m['_ts'])
    mask = apr['_ts'].isin(merge_ts)
    
    # Replace 20% worst offenders in LEFT
    left_replaced = replace_worst(m, atmo_left, ucm_col, 20, offset)
    left_map = dict(zip(left_replaced['_ts'], left_replaced[atmo_left]))
    apr.loc[mask, atmo_left] = apr.loc[mask, '_ts'].map(left_map)
    
    # Replace 20% worst offenders in RIGHT
    right_replaced = replace_worst(m, atmo_right, ucm_col, 20, offset)
    right_map = dict(zip(right_replaced['_ts'], right_replaced[atmo_right]))
    apr.loc[mask, atmo_right] = apr.loc[mask, '_ts'].map(right_map)
    
    n_left = (apr.loc[mask, atmo_left].notna()).sum()
    print(f"{ucm_col:12s} offset {offset:+4.1f} → updated {n_left} rows")

# ── Save ──
apr.drop(columns=['_ts'], inplace=True)
apr.to_csv(DST, index=False)
print(f"\nSaved: {DST}")
print(f"Shape: {apr.shape}")
