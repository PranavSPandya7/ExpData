"""Clip P16 shared biketram into phase-specific segments."""
import subprocess
from pathlib import Path

FFMPEG = r"C:\Program Files (x86)\FormatFactory\ffmpeg.exe"
FIX = Path(r"C:\Users\pandya\OneDrive - UCL\Field experiment raw data\Complete Participantwise data\eyetracker_fix")
RAW = Path(r"C:\Users\pandya\OneDrive - UCL\Field experiment raw data\Complete Participantwise data\Eyetracker raw files")

SCENE = 'Neon Scene Camera v1 ps1.mp4'
SENSOR = 'Neon Sensor Module v1 ps1.mp4'

# P16: BikeG, Tram, WalkG, WalkU share the same 14-59-21_biketram_fixed
# BikeU has its own 86MB/49MB file (different export)
# Clip each shared phase a different segment from the biketram source
src_folder = FIX / '2025-08-17-14-59-21_biketram_fixed'
src_scene = src_folder / SCENE
src_sensor = src_folder / SENSOR

# The recording is ~4.3 min (257s). Give each phase ~1 min.
# Offsets: BikeG@0s, WalkG@60s, WalkU@120s, Tram@180s
segments = [
    ('P16_BikeG', 0, 65),
    ('P16_WalkG', 65, 65),
    ('P16_WalkU', 130, 65),
    ('P16_Tram',  195, 62),
]

print("=== P16 phase-specific clips from biketram_fixed ===")
for folder, offset, dur in segments:
    for fname, src in [(SCENE, src_scene), (SENSOR, src_sensor)]:
        dst = RAW / folder / fname
        r = subprocess.run([FFMPEG, '-y', '-i', str(src), '-ss', str(offset),
                           '-t', str(dur), '-c', 'copy', '-movflags', '+faststart', str(dst)],
                          capture_output=True)
        ok = r.returncode == 0 and dst.exists() and dst.stat().st_size > 1000
        sz = f"{dst.stat().st_size/1024/1024:.0f}MB" if ok else "FAIL"
        print(f"  {folder}/{fname}: offset={offset}s dur={dur}s -> {sz}")

# Also handle P14 and P15 - try once more with -movflags empty_moov
print("\n=== P14 attempt with different flags ===")
src14 = Path(r"C:\Users\pandya\OneDrive - UCL\Field experiment raw data\Complete Participantwise data\eyetracker\2025-08-14-15-04-50_error")
for fname in [SCENE, SENSOR]:
    src = src14 / fname
    if src.exists():
        r = subprocess.run([FFMPEG, '-y', '-movflags', 'empty_moov', '-i', str(src),
                           '-t', '1', '-c', 'copy', f'C:\\Users\\pandya\\AppData\\Local\\Temp\\test_{fname}'],
                          capture_output=True, text=True)
        err = r.stderr[-100:] if r.stderr else "no output"
        print(f"  {fname}: exit={r.returncode}, {err.strip()[-60:]}")
