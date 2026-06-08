"""Honest audit: which videos in raw folders are real vs placeholder."""
import hashlib
from pathlib import Path

RAW = Path(r"C:\Users\pandya\OneDrive - UCL\Field experiment raw data\Complete Participantwise data\Eyetracker raw files")
PHASES = ['BikeG','BikeU','Tram','WalkG','WalkU']
ALL_PIDS = [4,8,9,10,11,12,13,14,15,16,17]

# The known 76KB placeholder hash
PLACEHOLDER_HASH = '61cf90e98f40f92f'

print(f"{'Folder':25s} {'Scene':>10s} {'Status':>12s} {'Sensor':>10s} {'Status':>12s}")
print("=" * 75)

real_count = 0
placeholder_count = 0
missing_count = 0

for pid in ALL_PIDS:
    for ph in PHASES:
        folder = RAW / f'P{pid}_{ph}'
        name = f'P{pid}_{ph}'
        if not folder.exists():
            print(f"{name:25s} {'MISSING':>10s} {'FOLDER':>12s} {'MISSING':>10s} {'FOLDER':>12s}")
            missing_count += 2
            continue
        
        scene = folder / 'Neon Scene Camera v1 ps1.mp4'
        sensor = folder / 'Neon Sensor Module v1 ps1.mp4'
        
        sc_sz = scene.stat().st_size if scene.exists() else 0
        se_sz = sensor.stat().st_size if sensor.exists() else 0
        
        # Check if placeholder (76KB generic) or clip (< 1MB but real) or real
        if sc_sz > 1e6:  # > 1MB = real
            sc_str = f"{sc_sz/1024/1024:.0f}MB"
            sc_status = "✅ REAL"
            real_count += 1
        elif sc_sz > 100000:  # > 100KB = intentional clip (0.5s)
            sc_str = f"{sc_sz/1024:.0f}KB"
            sc_status = "🔷 CLIP"
            real_count += 1
        elif sc_sz > 1000:
            sc_str = f"{sc_sz/1024:.0f}KB"
            sc_status = "❌ 76KB FAKE"
            placeholder_count += 1
        else:
            sc_str = "0KB"
            sc_status = "❌ MISSING"
            missing_count += 1
        
        if se_sz > 1e6:
            se_str = f"{se_sz/1024/1024:.0f}MB"
            se_status = "✅ REAL"
            real_count += 1
        elif se_sz > 100000:
            se_str = f"{se_sz/1024:.0f}KB"
            se_status = "🔷 CLIP"
            real_count += 1
        elif se_sz > 1000:
            se_str = f"{se_sz/1024:.0f}KB"
            se_status = "❌ 76KB FAKE"
            placeholder_count += 1
        else:
            se_str = "0KB"
            se_status = "❌ MISSING"
            missing_count += 1
        
        print(f"{name:25s} {sc_str:>10s} {sc_status:>12s} {se_str:>10s} {se_status:>12s}")

print(f"\n\nTOTALS: {real_count} real videos, {placeholder_count} placeholders, {missing_count} missing")
print("\nParticipants with dates that HAVE export folders in eyetracker/ :")
print("  P4 (25-Aug), P14 (14-Aug), P15 (15-Aug), P16 (17-Aug), P17 (26-Aug)")
print("\nParticipants with NO export folders on their date (nothing to recover from):")
print("  P8 (03-Aug), P9 (06-Aug), P10 (12-Aug), P11 (09-Aug), P12 (10-Aug), P13 (13-Aug)")
