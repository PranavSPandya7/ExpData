"""Check INSIDE each raw files phase folder for exports, videos, and neon_player subfolders."""
from pathlib import Path

RAW = Path(r"C:\Users\pandya\OneDrive - UCL\Field experiment raw data\Complete Participantwise data\Eyetracker raw files")
EXPORT = Path(r"C:\Users\pandya\OneDrive - UCL\Field experiment raw data\Complete Participantwise data\eyetracker")
FIX = Path(r"C:\Users\pandya\OneDrive - UCL\Field experiment raw data\Complete Participantwise data\eyetracker_fix")

PHASES = ['BikeG','BikeU','Tram','WalkG','WalkU']
ALL_PIDS = [4,8,9,10,11,12,13,14,15,16,17]

print(f"{'Folder':30s} {'SceneCam':>10s} {'SensorMod':>10s} {'neon_player':>12s} {'exports_found':>14s}")
print("=" * 80)

for pid in ALL_PIDS:
    for ph in PHASES:
        folder = RAW / f'P{pid}_{ph}'
        if not folder.exists():
            print(f"{f'P{pid}_{ph}':30s} {'MISSING':>10s}")
            continue
        
        scene = folder / 'Neon Scene Camera v1 ps1.mp4'
        sensor = folder / 'Neon Sensor Module v1 ps1.mp4'
        sc_sz = scene.stat().st_size if scene.exists() else 0
        se_sz = sensor.stat().st_size if sensor.exists() else 0
        sc_str = f"{sc_sz/1024/1024:.0f}MB" if sc_sz > 1e6 else ("0KB" if sc_sz==0 else f"{sc_sz/1024:.0f}KB")
        se_str = f"{se_sz/1024/1024:.0f}MB" if se_sz > 1e6 else ("0KB" if se_sz==0 else f"{se_sz/1024:.0f}KB")
        
        # Check for neon_player/exports
        np_folder = folder / 'neon_player'
        exports_found = 0
        if np_folder.exists():
            exports_dir = np_folder / 'exports'
            if exports_dir.exists():
                exports_found = len([x for x in exports_dir.iterdir() if x.is_dir()])
        
        np_str = "YES" if np_folder.exists() else ""
        exp_str = f"{exports_found} folders" if exports_found > 0 else ""
        
        print(f"{f'P{pid}_{ph}':30s} {sc_str:>10s} {se_str:>10s} {np_str:>12s} {exp_str:>14s}")

print("\n\n=== Searching ALL neon_player/exports for hidden video files ===")
# Find ALL neon_player/exports/ subfolders with video files
found_extra = []
for fp in RAW.rglob('neon_player/exports/*'):
    if fp.is_dir():
        scene = fp / 'Neon Scene Camera v1 ps1.mp4'
        sensor = fp / 'Neon Sensor Module v1 ps1.mp4'
        if scene.exists() or sensor.exists():
            sc_sz = scene.stat().st_size if scene.exists() else 0
            se_sz = sensor.stat().st_size if sensor.exists() else 0
            rel = fp.relative_to(RAW)
            found_extra.append((rel, sc_sz, se_sz))

if found_extra:
    for rel, sc_sz, se_sz in found_extra:
        print(f"  {str(rel):60s} Scene={sc_sz/1024/1024:.0f}MB Sensor={se_sz/1024/1024:.0f}MB")
else:
    print("  No extra video files found in neon_player/exports")
