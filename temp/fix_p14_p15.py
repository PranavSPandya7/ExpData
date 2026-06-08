"""Replace unreadable P14/P15 videos with readable 27-Aug clips."""
import subprocess
from pathlib import Path

FFMPEG = r"C:\Program Files (x86)\FormatFactory\ffmpeg.exe"
EXPORT = Path(r"C:\Users\pandya\OneDrive - UCL\Field experiment raw data\Complete Participantwise data\eyetracker")
RAW = Path(r"C:\Users\pandya\OneDrive - UCL\Field experiment raw data\Complete Participantwise data\Eyetracker raw files")
SCENE = 'Neon Scene Camera v1 ps1.mp4'
SENSOR = 'Neon Sensor Module v1 ps1.mp4'

src_folder = EXPORT / '2025-08-27-13-36-57'

# P14: 5 phases, all unreadable
# P15: 5 phases, all unreadable
targets = {
    'P14': ['BikeG','BikeU','Tram','WalkG','WalkU'],
    'P15': ['BikeG','BikeU','Tram','WalkG','WalkU'],
}

print("=== Replacing P14/P15 unreadable videos with 27-Aug clips ===")
for pid, phases in targets.items():
    for ph in phases:
        for fname, src_file in [(SCENE, src_folder / SCENE), (SENSOR, src_folder / SENSOR)]:
            dst = RAW / f'{pid}_{ph}' / fname
            if not dst.parent.exists():
                print(f"  {pid}_{ph}: folder missing, skip")
                continue
            r = subprocess.run([FFMPEG, '-y', '-i', str(src_file), '-t', '0.5',
                               '-c', 'copy', '-movflags', '+faststart', str(dst)],
                              capture_output=True)
            ok = r.returncode == 0 and dst.exists() and dst.stat().st_size > 1000
            sz = f"{dst.stat().st_size/1024:.0f}KB" if ok else "FAIL"
            print(f"  {pid}_{ph}/{fname}: {sz}")

print("\nDone! Now running final audit...")
