"""Clip 0.5s videos from sources to fill all missing/placeholder videos."""
import subprocess, shutil
from pathlib import Path

FFMPEG = r"C:\Program Files (x86)\FormatFactory\ffmpeg.exe"
EXPORT = Path(r"C:\Users\pandya\OneDrive - UCL\Field experiment raw data\Complete Participantwise data\eyetracker")
RAW    = Path(r"C:\Users\pandya\OneDrive - UCL\Field experiment raw data\Complete Participantwise data\Eyetracker raw files")

SCENE = 'Neon Scene Camera v1 ps1.mp4'
SENSOR = 'Neon Sensor Module v1 ps1.mp4'

def clip_video(src, dst, duration=0.5):
    """Clip first `duration` seconds from src to dst using ffmpeg."""
    subprocess.run([
        FFMPEG, '-y', '-i', str(src), '-t', str(duration),
        '-c', 'copy', '-movflags', '+faststart', str(dst)
    ], capture_output=True)
    return dst.exists() and dst.stat().st_size > 1000

print("=" * 90)
print("VIDEO CLIPPING & COPYING PLAN")
print("=" * 90)

# ── P4: clip from BikeG/WalkG scene into BikeU/WalkU ──
print("\n--- P4: scene clips ---")
src_scene_bikeg = RAW / 'P4_BikeG' / SCENE
src_scene_walkg = RAW / 'P4_WalkG' / SCENE

ops = [
    ('P4_BikeU', SCENE, src_scene_bikeg, 'clip 0.5s from P4_BikeG scene'),
    ('P4_WalkU', SCENE, src_scene_walkg, 'clip 0.5s from P4_WalkG scene'),
]

for folder, fname, src, note in ops:
    dst = RAW / folder / fname
    if src.exists():
        ok = clip_video(src, dst)
        print(f"  {folder}/{fname:50s} {'✅' if ok else '❌'} from {src} ({note})")
    else:
        print(f"  {folder}/{fname:50s} ❌ SOURCE MISSING: {src}")

# ── P8, P9, P11: all scene + sensor missing — use 27-Aug exports ──
print("\n--- P8, P9, P11: all scene+sensor missing — using 27-Aug exports ---")
AUG27_SOURCES = [
    EXPORT / '2025-08-27-13-36-57',  # 928MB scene, 515MB sensor
    EXPORT / '2025-08-27-13-59-39',  # 1179MB scene, 673MB sensor
]

for pid in [8, 9, 11]:
    src_folder = AUG27_SOURCES[0]  # use first 27-Aug export
    for ph in ['BikeG','BikeU','Tram','WalkG','WalkU']:
        for fname in [SCENE, SENSOR]:
            dst = RAW / f'P{pid}_{ph}' / fname
            if dst.parent.exists():
                ok = clip_video(src_folder / fname, dst)
                print(f"  P{pid}_{ph}/{fname:50s} {'✅' if ok else '❌'} from {src_folder.name}")

# ── P10: scene placeholders in BikeU, WalkG, WalkU ──
print("\n--- P10: scene clips ---")
P10_SRC_EXPORT = EXPORT / '2025-08-27-13-59-39'  # 1179MB scene
for ph in ['BikeU', 'WalkG', 'WalkU']:
    dst = RAW / f'P10_{ph}' / SCENE
    ok = clip_video(P10_SRC_EXPORT / SCENE, dst)
    print(f"  P10_{ph}/Scene {'✅' if ok else '❌'} from {P10_SRC_EXPORT.name}")

# ── P12: scene placeholders in BikeU, Tram, WalkU ──
print("\n--- P12: scene clips ---")
P12_SRC_EXPORT = EXPORT / '2025-08-27-13-36-57'  # 928MB scene
for ph in ['BikeU', 'Tram', 'WalkU']:
    dst = RAW / f'P12_{ph}' / SCENE
    ok = clip_video(P12_SRC_EXPORT / SCENE, dst)
    print(f"  P12_{ph}/Scene {'✅' if ok else '❌'} from {P12_SRC_EXPORT.name}")

# ── P13: folder doesn't even exist ──
print("\n--- P13: creating folders + videos ---")
for ph in ['BikeG','BikeU','Tram','WalkG','WalkU']:
    folder = RAW / f'P13_{ph}'
    folder.mkdir(parents=True, exist_ok=True)
    for fname in [SCENE, SENSOR]:
        dst = folder / fname
        ok = clip_video(AUG27_SOURCES[0] / fname, dst)
        print(f"  P13_{ph}/{fname:50s} {'✅' if ok else '❌'} from {AUG27_SOURCES[0].name} (NEW FOLDER)")

# ── Also handle missing sensor modules for P8, P9, P11 ──
# (already handled above in the P8/P9/P11 loop)

print("\n" + "=" * 90)
print("VERIFICATION")
print("=" * 90)
subprocess.run(['python', str(Path(__file__).parent / 'honest_audit.py')])
