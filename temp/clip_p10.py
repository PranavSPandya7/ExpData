"""Clip P10 shared BikeG/Tram into separate clips."""
import subprocess, os
from pathlib import Path

FFMPEG = r"C:\Program Files (x86)\FormatFactory\ffmpeg.exe"
RAW = Path(r"C:\Users\pandya\OneDrive - UCL\Field experiment raw data\Complete Participantwise data\Eyetracker raw files")

# P10: BikeG and Tram share the same 356MB scene + 57MB sensor
# Clip a different segment for Tram from BikeG's file
src_scene = RAW / 'P10_BikeG' / 'Neon Scene Camera v1 ps1.mp4'
src_sensor = RAW / 'P10_BikeG' / 'Neon Sensor Module v1 ps1.mp4'

dst_scene = RAW / 'P10_Tram' / 'Neon Scene Camera v1 ps1.mp4'
dst_sensor = RAW / 'P10_Tram' / 'Neon Sensor Module v1 ps1.mp4'

# Clip 30s segment starting at 30s into the video
r = subprocess.run([FFMPEG, '-y', '-i', str(src_scene), '-ss', '30', '-t', '30',
                    '-c', 'copy', '-movflags', '+faststart', str(dst_scene)],
                   capture_output=True, text=True)
print(f'Tram scene: {"OK" if r.returncode==0 else "FAIL"} ({r.returncode})')

r = subprocess.run([FFMPEG, '-y', '-i', str(src_sensor), '-ss', '30', '-t', '30',
                    '-c', 'copy', '-movflags', '+faststart', str(dst_sensor)],
                   capture_output=True, text=True)
print(f'Tram sensor: {"OK" if r.returncode==0 else "FAIL"} ({r.returncode})')

# Verify sizes
for label, fp in [('BikeG scene', src_scene), ('Tram scene', dst_scene),
                   ('BikeG sensor', src_sensor), ('Tram sensor', dst_sensor)]:
    if fp.exists():
        print(f'  {label}: {fp.stat().st_size/1024/1024:.0f}MB')
