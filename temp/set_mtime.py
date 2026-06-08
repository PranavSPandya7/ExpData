"""Set each file's last-modified to the data date at 20:30."""
import os, time
from pathlib import Path
import pandas as pd
from datetime import datetime, date

cpw = Path(r'C:\Users\pandya\OneDrive - UCL\Field experiment raw data\Complete Participantwise data\Atmo_lys')

for fp in sorted(cpw.glob('*.csv')):
    df = pd.read_csv(fp)
    dt = pd.to_datetime(df['Datetime'].dropna().iloc[0], errors='coerce')
    if pd.isna(dt):
        print(f'{fp.name:25s} -> NO DATE, skipped')
        continue
    # Set to that date at 20:30
    target = datetime(dt.year, dt.month, dt.day, 20, 30, 0)
    ts = target.timestamp()
    os.utime(fp, (ts, ts))
    new_mtime = datetime.fromtimestamp(os.path.getmtime(fp))
    print(f'{fp.name:25s} -> {new_mtime.strftime("%Y-%m-%d %H:%M:%S")}')
