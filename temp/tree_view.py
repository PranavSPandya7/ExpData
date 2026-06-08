"""Generate tree view of eyetracker raw files structure."""
from pathlib import Path

RAW = Path(r"C:\Users\pandya\OneDrive - UCL\Field experiment raw data\Complete Participantwise data\Eyetracker raw files")
PHASES = ['BikeG','BikeU','Tram','WalkG','WalkU']
ALL_PIDS = [4,8,9,10,11,12,13,14,15,16,17]

KEY_FILES = ['Neon Scene Camera v1 ps1.mp4', 'Neon Sensor Module v1 ps1.mp4',
             'gaze ps1.raw', 'fixations ps1.raw', 'eye_state ps1.raw', 
             'info.json', 'blinks ps1.raw', 'saccades ps1.raw', 'imu ps1.raw',
             'gaze ps1.time', 'eye_state ps1.time']

print("=" * 100)
print("EYETRACKER RAW FILES - COMPLETE FILE TREE")
print("=" * 100)

for pid in ALL_PIDS:
    print(f"\nP{pid}/")
    for ph in PHASES:
        folder = RAW / f'P{pid}_{ph}'
        if not folder.exists():
            print(f"  ├── {ph}/ (FOLDER DOES NOT EXIST)")
            continue
        
        # Get video sizes
        scene = folder / 'Neon Scene Camera v1 ps1.mp4'
        sensor = folder / 'Neon Sensor Module v1 ps1.mp4'
        sc_sz = f"{scene.stat().st_size/1024/1024:.0f}MB" if scene.exists() and scene.stat().st_size > 1e6 else f"{scene.stat().st_size/1024:.0f}KB" if scene.exists() else "MISSING"
        se_sz = f"{sensor.stat().st_size/1024/1024:.0f}MB" if sensor.exists() and sensor.stat().st_size > 1e6 else f"{sensor.stat().st_size/1024:.0f}KB" if sensor.exists() else "MISSING"
        
        # Check key raw data files
        present = []
        missing = []
        for kf in KEY_FILES:
            fp = folder / kf
            if fp.exists() and fp.stat().st_size > 0:
                present.append(kf)
            elif kf in ['Neon Scene Camera v1 ps1.mp4', 'Neon Sensor Module v1 ps1.mp4']:
                pass  # already handled above
            else:
                missing.append(kf)
        
        neon = 'YES' if (folder / 'neon_player').exists() else 'no'
        
        print(f"  ├── {ph}/")
        print(f"  │   ├── Scene Camera: {sc_sz}")
        print(f"  │   ├── Sensor Module: {se_sz}")
        print(f"  │   ├── Raw data files: {len(present)} present, {len(missing)} missing")
        if missing:
            print(f"  │   ├── Missing: {', '.join(missing[:5])}{'...' if len(missing)>5 else ''}")
        print(f"  │   └── neon_player: {neon}")
