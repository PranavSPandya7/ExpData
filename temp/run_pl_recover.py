"""Use pl-recover to fix P14 and P15 export recordings."""
import sys
sys.path.insert(0, r"C:\Users\pandya\OneDrive - UCL\Pranav PhD\03_Experiments & Data\src\pl-recover-recording\src")

from pathlib import Path
from pupil_labs.recover_recording.recover import RecordingFixer

exports = [
    (r"C:\Users\pandya\OneDrive - UCL\Field experiment raw data\Complete Participantwise data\eyetracker\2025-08-14-15-04-50_error", "P14 - shared WalkG/BikeG/Tram"),
    (r"C:\Users\pandya\OneDrive - UCL\Field experiment raw data\Complete Participantwise data\eyetracker\2025-08-14-13-55-40_error", "P14 - BikeU"),
    (r"C:\Users\pandya\OneDrive - UCL\Field experiment raw data\Complete Participantwise data\eyetracker\2025-08-14-14-22-47_error", "P14 - WalkU"),
    (r"C:\Users\pandya\OneDrive - UCL\Field experiment raw data\Complete Participantwise data\eyetracker\2025-08-15-14-53-45_error", "P15 - WalkU/BikeG/WalkG/Tram"),
    (r"C:\Users\pandya\OneDrive - UCL\Field experiment raw data\Complete Participantwise data\eyetracker\2025-08-15-15-29-45_error", "P15 - BikeU"),
]

for export_path, label in exports:
    print(f"\n=== Recovering: {label} ===")
    print(f"  Path: {export_path}")
    fp = Path(export_path)
    if not fp.exists():
        print(f"  SKIP: path not found")
        continue
    
    try:
        fixer = RecordingFixer(fp, cleanup_temp_files=True)
        errors = fixer.process(resize_video=False)
        print(f"  Done! Issues found: {len(errors)}")
        for e in errors:
            print(f"    - {e}")
    except Exception as ex:
        print(f"  ERROR: {ex}")
