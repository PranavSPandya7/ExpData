"""Final audit: check which participants still have problematic videos."""
import hashlib
from pathlib import Path

RAW = Path(r"C:\Users\pandya\OneDrive - UCL\Field experiment raw data\Complete Participantwise data\Eyetracker raw files")

PHASES = ['BikeG','BikeU','Tram','WalkG','WalkU']
ALL_PIDS = [4,8,9,10,11,12,13,14,15,16,17]

print(f"{'PID':>4s} {'Phase':8s} {'Scene':>10s} {'Sensor':>10s} {'Unique?':>8s}")
print('-' * 42)

total_missing = 0
total_dup = 0

for pid in ALL_PIDS:
    scene_hashes = []
    sensor_hashes = []
    for ph in PHASES:
        folder = RAW / f'P{pid}_{ph}'
        scene_fp = folder / 'Neon Scene Camera v1 ps1.mp4'
        sensor_fp = folder / 'Neon Sensor Module v1 ps1.mp4'
        
        scene_sz = scene_fp.stat().st_size if scene_fp.exists() else 0
        sensor_sz = sensor_fp.stat().st_size if sensor_fp.exists() else 0
        
        scene_str = f"{scene_sz/1024/1024:.0f}MB" if scene_sz > 1e6 else "NO FILE"
        sensor_str = f"{sensor_sz/1024/1024:.0f}MB" if sensor_sz > 1e6 else "NO FILE"
        
        scene_h = hashlib.md5(scene_fp.read_bytes()).hexdigest()[:12] if scene_sz > 1e6 else 'none'
        sensor_h = hashlib.md5(sensor_fp.read_bytes()).hexdigest()[:12] if sensor_sz > 1e6 else 'none'
        
        scene_hashes.append(scene_h)
        sensor_hashes.append(sensor_h)
        
        if scene_sz == 0:
            scene_str = '❌ MISSING'
            total_missing += 1
        if sensor_sz == 0:
            sensor_str = '❌ MISSING'
            total_missing += 1
        
        print(f"{pid:4d} {ph:8s} {scene_str:>10s} {sensor_str:>10s}")
    
    # Check duplicates within participant
    unique_scene = len(set(h for h in scene_hashes if h != 'none'))
    unique_sensor = len(set(h for h in sensor_hashes if h != 'none'))
    
    if unique_scene < 5 and unique_scene > 0:
        print(f"     ⚠️ Scene Camera: {unique_scene}/5 unique (duplicates between phases)")
        total_dup += (5 - unique_scene)
    if unique_sensor < 5 and unique_sensor > 0:
        print(f"     ⚠️ Sensor Module: {unique_sensor}/5 unique (duplicates between phases)")
        total_dup += (5 - unique_sensor)
    print()

print('=' * 42)
print(f'Total missing videos: {total_missing}')
print(f'Total duplicate phases: {total_dup}')
print(f'{"✅ All phases have real unique videos!" if total_missing == 0 and total_dup == 0 else "⚠️ Still some issues"}')
