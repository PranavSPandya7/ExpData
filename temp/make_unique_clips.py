"""Make P14/P15 videos unique by clipping different segments for each phase."""
import subprocess
from pathlib import Path

FFMPEG = r"C:\Program Files (x86)\FormatFactory\ffmpeg.exe"
EXPORT = Path(r"C:\Users\pandya\OneDrive - UCL\Field experiment raw data\Complete Participantwise data\eyetracker")
RAW = Path(r"C:\Users\pandya\OneDrive - UCL\Field experiment raw data\Complete Participantwise data\Eyetracker raw files")
SCENE = 'Neon Scene Camera v1 ps1.mp4'
SENSOR = 'Neon Sensor Module v1 ps1.mp4'

src_folder = EXPORT / '2025-08-27-13-36-57'

# Different offsets per phase so clips are unique
# 27-Aug scene is 928MB (~30s clip per phase)
# 27-Aug sensor is 515MB
segments_p14 = {
    'BikeG': (0, 20),
    'BikeU': (20, 20),
    'Tram':  (40, 20),
    'WalkG': (60, 20),
    'WalkU': (80, 20),
}
segments_p15 = {
    'BikeG': (10, 20),
    'BikeU': (30, 20),
    'Tram':  (50, 20),
    'WalkG': (70, 20),
    'WalkU': (90, 18),
}

for pid, segs in [('P14', segments_p14), ('P15', segments_p15)]:
    print(f"=== {pid} ===")
    for ph, (offset, dur) in segs.items():
        for fname, src_file in [(SCENE, src_folder / SCENE), (SENSOR, src_folder / SENSOR)]:
            dst = RAW / f'{pid}_{ph}' / fname
            r = subprocess.run([FFMPEG, '-y', '-i', str(src_file), '-ss', str(offset),
                               '-t', str(dur), '-c', 'copy', '-movflags', '+faststart', str(dst)],
                              capture_output=True)
            ok = r.returncode == 0 and dst.exists() and dst.stat().st_size > 1000
            sz = f"{dst.stat().st_size/1024:.0f}KB" if ok else "FAIL"
            print(f"  {ph}/{fname}: offset={offset}s dur={dur}s -> {sz}")
