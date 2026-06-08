"""Clip shared videos to correct phase duration using Neon recording timestamps."""
import subprocess, json
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

FFMPEG = r"C:\Program Files (x86)\FormatFactory\ffmpeg.exe"
KEY = Path(r"C:\Users\pandya\Documents\Github\docker\ExpData\metadata\key.csv")
EXPORT = Path(r"C:\Users\pandya\OneDrive - UCL\Field experiment raw data\Complete Participantwise data\eyetracker")
RAW = Path(r"C:\Users\pandya\OneDrive - UCL\Field experiment raw data\Complete Participantwise data\Eyetracker raw files")
SCENE = 'Neon Scene Camera v1 ps1.mp4'
SENSOR = 'Neon Sensor Module v1 ps1.mp4'

key = pd.read_csv(KEY)

def get_phase_brussels(pid, phase):
    """Return (start_dt, end_dt) in Brussels time for a phase."""
    row = key[key['Participant_ID'] == pid].iloc[0]
    date_str = row['Date']
    date = datetime.strptime(f'{date_str}-2025', '%d-%b-%Y')
    s_col = f'{phase}_start'
    e_col = f'{phase}_end'
    if s_col not in row or pd.isna(row[s_col]):
        return None
    start = datetime.strptime(f"{date.strftime('%Y-%m-%d')} {row[s_col]}", '%Y-%m-%d %H:%M:%S')
    end   = datetime.strptime(f"{date.strftime('%Y-%m-%d')} {row[e_col]}", '%Y-%m-%d %H:%M:%S')
    if end < start:
        end += timedelta(days=1)
    # UTC → Brussels (+2h)
    start += timedelta(hours=2)
    end   += timedelta(hours=2)
    return (start, end)

def get_export_start(export_folder_name):
    """Get recording start time from info.json."""
    for base in [EXPORT]:
        fp = base / export_folder_name / 'info.json'
        if fp.exists():
            info = json.loads(fp.read_text())
            ns = info['start_time']
            return datetime.utcfromtimestamp(ns / 1_000_000_000) + timedelta(hours=2)  # to Brussels
    return None

def get_scene_duration(export_folder_name):
    """Get scene camera duration from .time file."""
    for base in [EXPORT]:
        fp = base / export_folder_name / f'{SCENE}.time'
        if fp.exists():
            data = fp.read_bytes()
            if len(data) >= 16:
                import struct
                first = struct.unpack('<Q', data[:8])[0]
                last = struct.unpack('<Q', data[-8:])[0]
                return (last - first) / 1e9  # seconds
    return None

def clip_segment(src, dst, start_offset_s, duration_s):
    """Clip a segment from src to dst using ffmpeg."""
    cmd = [FFMPEG, '-y', '-ss', str(start_offset_s), '-i', str(src),
           '-t', str(duration_s), '-c', 'copy', '-movflags', '+faststart', str(dst)]
    r = subprocess.run(cmd, capture_output=True)
    return dst.exists() and dst.stat().st_size > 1000

# ── CONFIG: source export, phases to clip, participant ──
# Format: (export_name, pid, [(phase, scene_ok, sensor_ok), ...])
CLIP_JOBS = [
    # P14: shared 15-04-50_error for WalkG/BikeG/Tram
    ('2025-08-14-15-04-50_error', 14, [
        ('WalkG', True, True),   # scene 25min, start offset ~9min, dur 14min ✅
        ('BikeG', False, True),  # scene too short (25min), offset ~31min ❌ scene
        ('Tram',  False, True),  # scene too short, offset ~50min ❌ scene
    ]),
]

print("=" * 90)
print("PHASE-ACCURATE VIDEO CLIPPING")
print("=" * 90)

for export_name, pid, phases in CLIP_JOBS:
    export_start = get_export_start(export_name)
    if export_start is None:
        print(f"\n❌ {export_name}: cannot find export start time")
        continue
    
    scene_dur = get_scene_duration(export_name)
    print(f"\n--- P{pid} from {export_name} ---")
    print(f"  Export start (Brussels): {export_start.strftime('%H:%M:%S')}")
    if scene_dur:
        print(f"  Scene duration: {scene_dur/60:.1f} min")
    
    for phase, clip_scene, clip_sensor in phases:
        win = get_phase_brussels(pid, phase)
        if win is None:
            print(f"  {phase}: no phase window in key.csv")
            continue
        
        ph_start, ph_end = win
        offset_s = (ph_start - export_start).total_seconds()
        dur_s = (ph_end - ph_start).total_seconds()
        
        if offset_s < 0:
            print(f"  {phase}: recording starts AFTER phase (offset={offset_s:.0f}s) — SKIP")
            continue
        
        dest = RAW / f'P{pid}_{phase}'
        
        for fname, do_clip in [(SCENE, clip_scene), (SENSOR, clip_sensor)]:
            if not do_clip:
                print(f"  {phase}/{fname}: SKIP (recording too short, offset={offset_s:.0f}s > scene={scene_dur:.0f}s)" if scene_dur else "  {phase}/{fname}: SKIP")
                continue
            
            src = EXPORT / export_name / fname
            dst = dest / fname
            
            if not src.exists():
                print(f"  {phase}/{fname}: source missing")
                continue
            
            ok = clip_segment(src, dst, offset_s, dur_s)
            result = "✅" if ok else "❌"
            print(f"  {phase}/{fname}: {result} offset={offset_s:.0f}s dur={dur_s:.0f}s")

print("\nDone!")
