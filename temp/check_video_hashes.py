"""Check video file hashes for duplicates."""
import hashlib
from pathlib import Path

root = Path(r"C:\Users\pandya\OneDrive - UCL\Field experiment raw data\Complete Participantwise data\Eyetracker raw files")

# Participants with identical sizes
checks = [
    ('P17', ['BikeG','BikeU','Tram','WalkG','WalkU']),
    ('P15', ['BikeG','BikeU','Tram','WalkG','WalkU']),
    ('P16', ['BikeG','BikeU','Tram','WalkG','WalkU']),
    ('P14', ['BikeG','BikeU','Tram','WalkG','WalkU']),
    ('P12', ['BikeG','BikeU','Tram','WalkG','WalkU']),
    ('P10', ['BikeG','BikeU','Tram','WalkG','WalkU']),
    ('P4',  ['BikeG','BikeU','Tram','WalkG','WalkU']),
]

for pid, phases in checks:
    print(f'\n=== {pid} Sensor Module ===')
    last_hash = None
    for ph in phases:
        fp = root / f'{pid}_{ph}' / 'Neon Sensor Module v1 ps1.mp4'
        if fp.exists():
            h = hashlib.md5(fp.read_bytes()).hexdigest()
            dup = ' ⬅ DUPLICATE' if h == last_hash else ''
            print(f'  {ph:6s}: {h[:16]}...{dup}')
            last_hash = h
        else:
            print(f'  {ph:6s}: (no file)')
    
    print(f'--- {pid} Scene Camera ---')
    last_hash = None
    for ph in phases:
        fp = root / f'{pid}_{ph}' / 'Neon Scene Camera v1 ps1.mp4'
        if fp.exists():
            h = hashlib.md5(fp.read_bytes()).hexdigest()
            dup = ' ⬅ DUPLICATE' if h == last_hash else ''
            print(f'  {ph:6s}: {h[:16]}...{dup}')
            last_hash = h
        else:
            print(f'  {ph:6s}: (no file or 0.1MB placeholder)')

# Also check what's up with P11, P8, P9 - only 0.1MB files
print('\n\n=== P11 files ===')
for ph in ['BikeG','BikeU','Tram','WalkG','WalkU']:
    folder = root / f'P11_{ph}'
    for fp in sorted(folder.glob('*.mp4')):
        print(f'  {ph}/{fp.name}: {fp.stat().st_size/1024:.1f} KB')
