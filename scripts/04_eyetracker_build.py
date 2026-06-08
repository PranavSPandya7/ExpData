"""Eyetracker: collect Neon export CSVs, resample to 10-sec, align to index."""

import warnings; warnings.filterwarnings('ignore')

from pathlib import Path
import pandas as pd

from _align_index import align_to_index

# ── PATHS ────────────────────────────────────────────────────────────────────
BASE       = Path(r'C:\Users\pandya\Documents\Github\docker\ExpData')
ET_DIR     = Path(r'C:\Users\pandya\Documents\Github\docker\rawdata\04_eyetracker')
OUTPUTS    = BASE / 'outputs'
KEY_FILE   = BASE / 'metadata\key.csv'
NEON_ROOT  = Path(r'C:\Users\pandya\OneDrive - UCL\Field experiment raw data\Complete Participantwise data\Eyetracker raw files')

# ── SKIP-GATE CONFIG ─────────────────────────────────────────────────────────
FORCE_RERUN = False

PHASES = ['BikeU', 'WalkU', 'BikeG', 'WalkG', 'Tram']


# ── COLLECTION: Neon Export → rawdata/eyetracker/ ────────────────────────────
def collect_eyetracker(force: bool = False):
    """Copy output.csv from Neon Export folder tree into rawdata/eyetracker/.

    Searches recursively under NEON_ROOT for all
    ``neon_player/exports/000/output.csv`` files, reads the header to
    obtain ParticipantID and PhaseID, then copies to
    ``rawdata/eyetracker/{ParticipantID}_{PhaseID}.csv``.
    """
    out_dir = ET_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = list(out_dir.glob('P*_*.csv'))
    if existing and not force:
        print(f'  Eyetracker already collected ({len(existing)} files). Skip collection.')
        return

    # Load expected PIDs from key file for filtering
    key = pd.read_csv(KEY_FILE)
    expected_pids = set(f'P{int(p)}' for p in key['Participant_ID'].dropna())

    output_files = sorted(NEON_ROOT.rglob('output.csv'))
    if not output_files:
        print(f'  No output.csv files found under {NEON_ROOT}')
        return

    print(f'  Found {len(output_files)} output.csv files under Neon Export')

    count = 0
    for f in output_files:
        try:
            header = pd.read_csv(f, nrows=1)
            pid   = str(header['ParticipantID'].iloc[0]).strip()
            phase = str(header['PhaseID'].iloc[0]).strip()
        except Exception as e:
            print(f'    Skipping {f}: {e}')
            continue

        # Only keep participants in the experiment key
        if pid not in expected_pids:
            print(f'    Skipping {pid} {phase} (not in key.csv)')
            continue

        out_name = out_dir / f'{pid}_{phase}.csv'
        if out_name.exists() and not force:
            count += 1
            continue

        import shutil
        shutil.copy2(str(f), str(out_name))
        count += 1
        print(f'    Collected {pid} {phase} -> {out_name.name}')
    print(f'  Eyetracker collection: {count} files')


# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    # ── Step 1: Collect from Neon Export ────────────────────────────────
    print("=" * 55)
    print("  [Eyetracker Build] Step 1 — Collect from Neon Export")
    print("=" * 55)
    collect_eyetracker()

    # ── Step 2: Build 10-sec CSV ────────────────────────────────────────
    out_path = OUTPUTS / '04_eyetracker_10sec.csv'
    if out_path.exists() and not FORCE_RERUN:
        print(f'Output already exists: {out_path}. Skipping (FORCE_RERUN=False).')
        return

    if not ET_DIR.exists():
        print(f'ERROR: Eyetracker folder not found: {ET_DIR}')
        return

    files = sorted(ET_DIR.glob('P*_*.csv'))
    if not files:
        print(f'ERROR: No eyetracker files found in {ET_DIR}')
        return

    print(f'Found {len(files)} per-participant-phase files in {ET_DIR}')

    frames = []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(['ParticipantID', 'PhaseID', 'Datetime']).reset_index(drop=True)
    # Eyetracker timestamps are UTC → shift to Brussels local before alignment
    out['Datetime'] = pd.to_datetime(out['Datetime']) + pd.Timedelta(hours=2)
    out = align_to_index(out, 'eyetracker')

    for drop_col in ['in_fixation', 'in_saccade']:
        if drop_col in out.columns:
            out = out.drop(columns=[drop_col])

    out.to_csv(out_path, index=False)
    print(f'\nSaved {len(out):,} rows × {len(out.columns)} cols => {out_path}')
    print(f'Participants: {sorted(out["ParticipantID"].unique())}')
    print(f'Phases:       {sorted(out["PhaseID"].unique())}')


if __name__ == '__main__':
    main()

