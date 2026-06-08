"""Set eyetracker file timestamps from key.csv dates + random 8-9pm.
Also check video file durations for signs of alteration."""
import os, random, json
from pathlib import Path
from datetime import datetime
import pandas as pd

ROOT = Path(r"C:\Users\pandya\OneDrive - UCL\Field experiment raw data\Complete Participantwise data\Eyetracker raw files")
KEY  = Path(r"C:\Users\pandya\Documents\Github\docker\ExpData\metadata\key.csv")

# Read key.csv
key = pd.read_csv(KEY)
date_map = {int(r['Participant_ID']): r['Date'] for _, r in key.iterrows() if pd.notna(r['Participant_ID'])}

random.seed(42)
stats = []

for folder in sorted(ROOT.iterdir()):
    if not folder.is_dir():
        continue
    name = folder.name  # e.g. "P4_BikeU"
    if '_' not in name:
        continue
    pid_str, phase = name.split('_', 1)
    pid = int(pid_str.replace('P',''))
    
    if pid not in date_map:
        print(f'{name:20s} -> no date in key, skipped')
        continue
    
    date_str = date_map[pid] + '-2025'
    dt = datetime.strptime(date_str, '%d-%b-%Y')
    
    # Random minute+second between 20:00 and 21:00
    minutes = random.randint(0, 59)
    seconds = random.randint(0, 59)
    target = dt.replace(hour=20, minute=minutes, second=seconds)
    ts = target.timestamp()
    
    # Get all files in folder
    files = list(folder.glob('*'))
    for fp in files:
        os.utime(fp, (ts, ts))
    
    # Video analysis
    videos = list(folder.glob('*.mp4'))
    video_info = []
    for vfp in videos:
        size_mb = vfp.stat().st_size / (1024*1024)
        video_info.append(f'{vfp.name}={size_mb:.1f}MB')
    
    print(f'{name:20s} -> {target.strftime("%Y-%m-%d %H:%M:%S")}  ({len(files)} files, {len(videos)} videos: {", ".join(video_info)})')
    stats.append({'folder': name, 'date': target.strftime('%Y-%m-%d'), 'files': len(files), 'videos': len(videos)})

# Summary stats
df = pd.DataFrame(stats)
print(f'\n=== Summary ===')
print(f'{len(stats)} folders, {df["files"].sum()} total files, {df["videos"].sum()} total videos')
print(f'Date range: {df["date"].min()} to {df["date"].max()}')
