"""Copy real video files from eyetracker exports to raw files folders for P14-P17."""
import shutil
from pathlib import Path

EXPORT = Path(r"C:\Users\pandya\OneDrive - UCL\Field experiment raw data\Complete Participantwise data\eyetracker")
RAW    = Path(r"C:\Users\pandya\OneDrive - UCL\Field experiment raw data\Complete Participantwise data\Eyetracker raw files")

# Mapping: (source_export_folder, dest_phase_folder, note)
COPIES = [
    # ── P14 (14-Aug) ──
    ('2025-08-14-13-55-40_error', 'P14_BikeU', 'BikeU export'),
    ('2025-08-14-14-22-47_error', 'P14_WalkU', 'closest to WalkU'),
    ('2025-08-14-15-04-50_error', 'P14_WalkG', 'closest to WalkG'),
    ('2025-08-14-15-04-50_error', 'P14_BikeG', 'same source as WalkG'),
    ('2025-08-14-15-04-50_error', 'P14_Tram',  'same source as WalkG'),
    
    # ── P15 (15-Aug) ──
    ('2025-08-15-15-29-45_error', 'P15_BikeU', 'BikeU export'),
    ('2025-08-15-14-53-45_error', 'P15_WalkU', 'WalkU export'),
    ('2025-08-15-14-53-45_error', 'P15_BikeG', 'closest to BikeG'),
    ('2025-08-15-14-53-45_error', 'P15_WalkG', 'closest to WalkG'),
    ('2025-08-15-14-53-45_error', 'P15_Tram',  'closest to Tram'),
    
    # ── P16 (17-Aug) ──
    ('2025-08-17-14-59-21_biketram', 'P16_BikeG', 'BikeG export (biketram)'),
    ('2025-08-17-15-27-43_error',   'P16_BikeU', 'BikeU export'),
    ('2025-08-17-14-59-21_biketram', 'P16_WalkU', 'closest to WalkU'),
    ('2025-08-17-14-59-21_biketram', 'P16_WalkG', 'closest to WalkG'),
    ('2025-08-17-14-59-21_biketram', 'P16_Tram',  'closest to Tram'),
    
    # ── P17 (26-Aug) ──
    ('2025-08-26-13-51-31',          'P17_BikeG', 'BikeG export'),
    ('2025-08-26-14-10-23',          'P17_Tram',  'Tram export'),
    ('2025-08-26-15-10-04_error',    'P17_BikeU', 'BikeU export'),
    ('2025-08-26-14-35-10_error',    'P17_WalkU', 'WalkU export'),
    ('2025-08-26-13-28-15',          'P17_WalkG', 'full day recording'),
]

SCENE = 'Neon Scene Camera v1 ps1.mp4'
SENSOR = 'Neon Sensor Module v1 ps1.mp4'

print(f"{'Dest':25s} {'Source':40s} {'Scene':>10s} {'Sensor':>10s}")
print('-' * 90)

copied = 0
for src_folder, dest_folder, note in COPIES:
    src_path = EXPORT / src_folder
    dest_path = RAW / dest_folder
    
    if not dest_path.exists():
        print(f"{dest_folder:25s} {'SKIP - dest missing':40s}")
        continue
    
    scene_src = src_path / SCENE
    sensor_src = src_path / SENSOR
    scene_dst = dest_path / SCENE
    sensor_dst = dest_path / SENSOR
    
    scene_ok = scene_src.exists() and scene_src.stat().st_size > 1e6
    sensor_ok = sensor_src.exists() and sensor_src.stat().st_size > 1e6
    
    if scene_ok:
        shutil.copy2(scene_src, scene_dst)
    if sensor_ok:
        shutil.copy2(sensor_src, sensor_dst)
    
    scene_mb = f"{scene_src.stat().st_size/1024/1024:.0f}MB" if scene_ok else "NO FILE"
    sensor_mb = f"{sensor_src.stat().st_size/1024/1024:.0f}MB" if sensor_ok else "NO FILE"
    print(f"{dest_folder:25s} {src_folder:40s} {scene_mb:>10s} {sensor_mb:>10s}")
    copied += 1

print(f'\n✅ {copied} phase folders updated with real videos')
