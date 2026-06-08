"""Merge all index-aligned sensor + questionnaire CSVs (from outputs/) into one wide file."""
import warnings; warnings.filterwarnings('ignore')
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
from _paths import OUTPUTS
OUT = OUTPUTS
INDEX_FILE = OUT / '00_index_10sec.csv'
OUT_FILE = OUT / 'mergeddata_all.csv'
MERGE_KEYS = ['ParticipantID', 'PhaseID', 'Datetime']
PHASES_5 = {'BikeU', 'WalkU', 'BikeG', 'WalkG', 'Tram'}

def main():
    if not INDEX_FILE.exists():
        print('Run 00_index_build.py first.')
        return
    index = pd.read_csv(INDEX_FILE, low_memory=False)
    index['Datetime'] = pd.to_datetime(index['Datetime'])
    print(f'Index: {len(index):,} rows')

    merged = index.copy()
    # Normalize PhaseID — between-phase rows should be '' not NaN
    for col in ['PhaseID']:
        merged[col] = merged[col].fillna('').astype(str).replace({'nan':'','None':''})
    # Sort by script number prefix (00_, 01_, etc.) so columns appear in order
    def _script_order(p: Path) -> str:
        stem = p.stem
        return stem.split('_')[0] if stem[0].isdigit() and '_' in stem else '99'
    
    for f in sorted(OUT.glob('*.csv'), key=_script_order):
        name = f.stem
        if name in ('index_10sec', 'mergeddata_all'):
            continue
        df = pd.read_csv(f, low_memory=False)
        if 'Datetime' not in df.columns and 'questionnaires' in name:
            df = df[df['PhaseID'].isin(PHASES_5)] if 'PhaseID' in df.columns else df
            df = df.drop_duplicates(subset=['ParticipantID', 'PhaseID'], keep='last')
            q_cols = [c for c in df.columns if c not in ('ParticipantID', 'PhaseID')]
            merged = merged.merge(df[['ParticipantID', 'PhaseID'] + q_cols],
                                  on=['ParticipantID', 'PhaseID'], how='left')
            print(f'  {name} (questionnaire): {len(df):,} rows -> {len(q_cols)} cols')
        elif 'Datetime' in df.columns and 'ParticipantID' in df.columns:
            df['Datetime'] = pd.to_datetime(df['Datetime'])
            # Normalize PhaseID so between-phase rows match (NaN → '')
            if 'PhaseID' in df.columns:
                df['PhaseID'] = df['PhaseID'].fillna('').astype(str).replace({'nan':'','None':''})
            df = df.drop_duplicates(subset=MERGE_KEYS, keep='first')
            merged = merged.merge(df, on=MERGE_KEYS, how='left', suffixes=('', '_dup'))
            merged = merged.drop(columns=[c for c in merged.columns if c.endswith('_dup')])
            sensor_cols = len(df.columns) - 3
            print(f'  {name}: {len(df):,} rows -> {sensor_cols} cols')

    merged = merged.sort_values(MERGE_KEYS).reset_index(drop=True)
    merged.to_csv(OUT_FILE, index=False)
    print(f'\nSaved: {OUT_FILE} — {len(merged):,} rows x {merged.shape[1]} cols')
    print(f'  Participants: {sorted(merged["ParticipantID"].unique())}')
    print(f'  Phases: {sorted(merged["PhaseID"].unique())}')

if __name__ == '__main__':
    main()
