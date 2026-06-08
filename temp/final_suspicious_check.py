"""Final check: look for anything suspicious, verify file structure."""
import hashlib
from pathlib import Path

RAW = Path(r"C:\Users\pandya\OneDrive - UCL\Field experiment raw data\Complete Participantwise data\Eyetracker raw files")
PHASES = ['BikeG','BikeU','Tram','WalkG','WalkU']
ALL_PIDS = [4,8,9,10,11,12,13,14,15,16,17]

print("=" * 80)
print("FINAL SUSPICIOUS CHECK")
print("=" * 80)

# 1. Check for duplicate hashes across different participants (cross-participant copies)
print("\n1. CROSS-PARTICIPANT DUPLICATE CHECK")
scene_hashes = {}
sensor_hashes = {}
dup_count = 0

for pid in ALL_PIDS:
    for ph in PHASES:
        folder = RAW / f'P{pid}_{ph}'
        if not folder.exists():
            continue
        for fname, store in [('Neon Scene Camera v1 ps1.mp4', scene_hashes),
                              ('Neon Sensor Module v1 ps1.mp4', sensor_hashes)]:
            fp = folder / fname
            if fp.exists() and fp.stat().st_size > 1e6:  # only check full-size videos
            # (clips are expected to be shared since they come from same source)
                h = hashlib.md5(fp.read_bytes()).hexdigest()
                key = f'{pid}_{ph}'
                if h in store:
                    dup_count += 1
                    other = store[h]
                    print(f"  ⚠️ {key}/{fname} = SAME as {other} ({fp.stat().st_size/1024/1024:.0f}MB)")
                else:
                    store[h] = key

if dup_count == 0:
    print("  ✅ No cross-participant duplicates in full-size videos")
else:
    print(f"  Found {dup_count} cross-participant duplicates")

print(f"\n  Unique Scene Camera recordings: {len(scene_hashes)}")
print(f"  Unique Sensor Module recordings: {len(sensor_hashes)}")

# 2. Check P13 has proper folder structure (all expected files)
print("\n2. P13 FOLDER STRUCTURE CHECK")
expected_files = ['Neon Scene Camera v1 ps1.mp4', 'Neon Sensor Module v1 ps1.mp4',
                  'gaze ps1.raw', 'fixations ps1.raw', 'eye_state ps1.raw', 'info.json']
for ph in PHASES:
    folder = RAW / f'P13_{ph}'
    missing = [f for f in expected_files if not (folder / f).exists()]
    if missing:
        print(f"  P13_{ph}: missing {missing}")
    else:
        print(f"  P13_{ph}: ✅ All expected files present (but only clips, no raw data)")

# 3. File size sanity check
print("\n3. FILE SIZE SANITY CHECK")
tiny_count = 0
for pid in ALL_PIDS:
    for ph in PHASES:
        folder = RAW / f'P{pid}_{ph}'
        if not folder.exists():
            continue
        for fname in ['Neon Scene Camera v1 ps1.mp4', 'Neon Sensor Module v1 ps1.mp4']:
            fp = folder / fname
            if fp.exists():
                sz = fp.stat().st_size
                if sz > 1000 and sz < 100000:  # between 1KB and 100KB
                    # These are the 0.5s clips - expected
                    pass
                elif sz == 0:
                    print(f"  ⚠️ {pid}_{ph}/{fname} is 0 bytes (empty)")
                    tiny_count += 1

if tiny_count == 0:
    print("  ✅ No empty files")

# 4. Summary
print("\n\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)
total_real = 0
total_clip = 0
for pid in ALL_PIDS:
    for ph in PHASES:
        folder = RAW / f'P{pid}_{ph}'
        if not folder.exists():
            continue
        for fname in ['Neon Scene Camera v1 ps1.mp4', 'Neon Sensor Module v1 ps1.mp4']:
            fp = folder / fname
            if fp.exists() and fp.stat().st_size > 1e6:
                total_real += 1
            elif fp.exists() and fp.stat().st_size > 1000:
                total_clip += 1

print(f"  Full real videos (>1MB): {total_real}")
print(f"  Short clips (0.5s):      {total_clip}")
print(f"  Total:                   {total_real + total_clip}")
print(f"\n  {'✅ No suspicious issues' if dup_count == 0 else '⚠️ Review duplicates above'}")
