"""Scan eyetracker export folders for real videos, map to participants/phases."""
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

ROOT = Path(r"C:\Users\pandya\OneDrive - UCL\Field experiment raw data\Complete Participantwise data\eyetracker")
KEY  = Path(r"C:\Users\pandya\Documents\Github\docker\ExpData\metadata\key.csv")
RAW  = Path(r"C:\Users\pandya\OneDrive - UCL\Field experiment raw data\Complete Participantwise data\Eyetracker raw files")

key = pd.read_csv(KEY)

def parse_date(d_str):
    return datetime.strptime(f'{d_str}-2025', '%d-%b-%Y')

def get_phase_window(pid, phase):
    """Return (start_dt, end_dt) in Brussels time for a participant phase."""
    row = key[key['Participant_ID'] == pid].iloc[0]
    date = parse_date(row['Date'])
    s_col = f'{phase}_start'
    e_col = f'{phase}_end'
    if s_col not in row or pd.isna(row[s_col]):
        return None
    start = datetime.strptime(f"{date.strftime('%Y-%m-%d')} {row[s_col]}", '%Y-%m-%d %H:%M:%S')
    end   = datetime.strptime(f"{date.strftime('%Y-%m-%d')} {row[e_col]}", '%Y-%m-%d %H:%M:%S')
    if end < start:
        end += timedelta(days=1)
    # Add 2h UTC offset → Brussels
    start += timedelta(hours=2)
    end   += timedelta(hours=2)
    return (start, end)

def export_to_pid_phase(folder_name):
    """Try to match an export folder timestamp to a participant+phase."""
    parts = folder_name.split('_')
    ts_str = parts[0]  # e.g. "2025-08-25-13-58-10"
    try:
        export_dt = datetime.strptime(ts_str, '%Y-%m-%d-%H-%M-%S')
    except:
        return None
    
    for _, row in key.iterrows():
        pid = int(row['Participant_ID'])
        date = parse_date(row['Date'])
        export_date = datetime(date.year, date.month, date.day)
        export_date_end = export_date + timedelta(days=1)
        
        if not (export_date <= export_dt < export_date_end):
            continue
        
        for ph in ['BikeU','WalkU','BikeG','WalkG','Tram']:
            win = get_phase_window(pid, ph)
            if win is None:
                continue
            start, end = win
            # Allow some tolerance: recording may start a few min before
            tol_start = start - timedelta(minutes=5)
            tol_end   = end + timedelta(minutes=5)
            if tol_start <= export_dt <= tol_end:
                return (pid, ph, export_dt)
    
    # If within participant date but no phase match
    for _, row in key.iterrows():
        pid = int(row['Participant_ID'])
        date = parse_date(row['Date'])
        export_date = datetime(date.year, date.month, date.day)
        if export_date <= export_dt < export_date + timedelta(days=1):
            return (pid, 'UNKNOWN', export_dt)
    
    return None

# ── Scan all export folders ──
print("=" * 100)
print(f"{'Export folder':40s} {'PID':>4s} {'Phase':8s} {'SceneCam':>12s} {'SensorMod':>12s} {'Has_error':>10s}")
print("=" * 100)

all_videos = {}  # (pid, phase) -> [(export_folder, scene_size, sensor_size)]

for folder in sorted(ROOT.iterdir()):
    if not folder.is_dir() or folder.name == 'Processed eyetracker':
        continue
    
    fname = folder.name
    scene_fp = folder / 'Neon Scene Camera v1 ps1.mp4'
    sensor_fp = folder / 'Neon Sensor Module v1 ps1.mp4'
    has_error = '_error' in fname
    
    scene_sz = scene_fp.stat().st_size if scene_fp.exists() else 0
    sensor_sz = sensor_fp.stat().st_size if sensor_fp.exists() else 0
    
    result = export_to_pid_phase(fname)
    if result:
        pid, phase, export_dt = result
        phase_str = f"{phase:8s}"
        key_ = (pid, phase)
        if key_ not in all_videos:
            all_videos[key_] = []
        all_videos[key_].append({
            'folder': fname, 'scene': scene_sz, 'sensor': sensor_sz, 
            'time': export_dt, 'error': has_error
        })
    else:
        pid = 0
        phase_str = "?"
    
    scene_mb = f"{scene_sz/1024/1024:.0f}MB" if scene_sz > 1e6 else f"{scene_sz/1024:.0f}KB"
    sensor_mb = f"{sensor_sz/1024/1024:.0f}MB" if sensor_sz > 1e6 else f"{sensor_sz/1024:.0f}KB"
    err_str = "⚠️ERROR" if has_error else ""
    print(f"{fname:40s} {pid:4d} {phase_str:8s} {scene_mb:>12s} {sensor_mb:>12s} {err_str:>10s}")

print("\n\n=== VIDEO AVAILABILITY PER PARTICIPANT × PHASE ===")
print(f"{'PID':>4s} {'Phase':8s} {'Exports':>6s} {'Scene':>10s} {'Sensor':>10s} {'Best':>10s} {'HasError':>9s}")
print("-" * 60)

PHASES = ['BikeU','WalkU','BikeG','WalkG','Tram']

for pid in sorted(set(k[0] for k in all_videos)):
    for ph in PHASES:
        key_ = (pid, ph)
        entries = all_videos.get(key_, [])
        if not entries:
            # Check if raw files folder exists
            raw_folder = RAW / f'P{pid}_{ph}'
            has_video = any(raw_folder.glob('*.mp4'))
            raw_status = "raw_vid" if has_video else "no_folder" if not raw_folder.exists() else "no_video"
            print(f"{pid:4d} {ph:8s} {'—':>6s} {'—':>10s} {'—':>10s} {'—':>10s} {raw_status:>9s}")
            continue
        
        n = len(entries)
        best = max(entries, key=lambda x: x['sensor'])
        scene_mb = f"{best['scene']/1024/1024:.0f}MB"
        sensor_mb = f"{best['sensor']/1024/1024:.0f}MB"
        err = "⚠️" if any(e['error'] for e in entries) else ""
        best_time = best['time'].strftime('%H:%M')
        print(f"{pid:4d} {ph:8s} {n:6d} {scene_mb:>10s} {sensor_mb:>10s} {best_time:>10s} {err:>9s}")
