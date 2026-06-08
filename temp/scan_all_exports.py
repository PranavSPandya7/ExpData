"""Scan BOTH eyetracker folders for ALL export videos, map to participants."""
from pathlib import Path
from datetime import datetime
import pandas as pd

KEY = Path(r"C:\Users\pandya\Documents\Github\docker\ExpData\metadata\key.csv")
key = pd.read_csv(KEY)

# Collect all exports from both folders
all_exports = {}  # date_str -> [(datetime, folder_name, scene_size, sensor_size)]

for base in [r"C:\Users\pandya\OneDrive - UCL\Field experiment raw data\Complete Participantwise data\eyetracker",
             r"C:\Users\pandya\OneDrive - UCL\Field experiment raw data\Complete Participantwise data\eyetracker_fix"]:
    root = Path(base)
    if not root.exists():
        continue
    for f in sorted(root.iterdir()):
        if not f.is_dir():
            continue
        parts = f.name.split('_')
        try:
            dt = datetime.strptime(parts[0], '%Y-%m-%d-%H-%M-%S')
        except:
            continue
        scene = f / 'Neon Scene Camera v1 ps1.mp4'
        sensor = f / 'Neon Sensor Module v1 ps1.mp4'
        scene_sz = scene.stat().st_size if scene.exists() else 0
        sensor_sz = sensor.stat().st_size if sensor.exists() else 0
        date_key = dt.strftime('%Y-%m-%d')
        if date_key not in all_exports:
            all_exports[date_key] = []
        all_exports[date_key].append((dt, f.name, scene_sz, sensor_sz, base))

print("=" * 110)
print(f"{'Participant':12s} {'Phase':8s} {'Closest Export':50s} {'Scene':>8s} {'Sensor':>8s}")
print("=" * 110)

PHASES = ['BikeU','WalkU','BikeG','WalkG','Tram']

# For each participant, find all exports on their date, map each phase to closest by time
for _, row in key.iterrows():
    pid = int(row['Participant_ID'])
    date_str = row['Date']
    exp_date = datetime.strptime(f'{date_str}-2025', '%d-%b-%Y')
    date_key = exp_date.strftime('%Y-%m-%d')
    
    exports = all_exports.get(date_key, [])
    
    if not exports:
        print(f"{f'P{pid}':12s} {'ALL':8s} {'❌ NO EXPORT FOLDERS EXIST AT ALL':50s}")
        continue
    
    # For each phase, find closest export
    for ph in PHASES:
        s_col = f'{ph}_start'
        e_col = f'{ph}_end'
        if s_col not in row or pd.isna(row[s_col]):
            continue
        
        # Phase time in UTC (from key.csv)
        ph_time_str = row[s_col]
        ph_dt_utc = datetime.strptime(f"{date_str}-2025 {ph_time_str}", '%d-%b-%Y %H:%M:%S')
        
        # Find closest export to this phase start time
        best = None
        best_diff = None
        for exp_dt, exp_name, sc_sz, se_sz, src_base in exports:
            diff = abs((exp_dt - ph_dt_utc).total_seconds())
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best = (exp_dt, exp_name, sc_sz, se_sz, src_base, diff)
        
        if best:
            exp_dt, exp_name, sc_sz, se_sz, src_base, diff_min = best
            sc_str = f"{sc_sz/1024/1024:.0f}MB" if sc_sz > 1e6 else f"{sc_sz/1024:.0f}KB"
            se_str = f"{se_sz/1024/1024:.0f}MB" if se_sz > 1e6 else f"{se_sz/1024:.0f}KB"
            diff_str = f"{diff_min/60:.1f}min diff"
            print(f"{f'P{pid}':12s} {ph:8s} {exp_name:50s} {sc_str:>8s} {se_str:>8s}  ({diff_str})")

print("\n\n=== PARTICIPANTS WITH NO EXPORT FOLDERS ===")
for _, row in key.iterrows():
    pid = int(row['Participant_ID'])
    date_str = row['Date']
    date_key = datetime.strptime(f'{date_str}-2025', '%d-%b-%Y').strftime('%Y-%m-%d')
    exports = all_exports.get(date_key, [])
    if not exports:
        print(f"P{pid} ({date_str} = {date_key}): ZERO export folders")
