"""Merge all index-aligned sensor + questionnaire CSVs (from outputs/) into one wide file.
Also saves a copy with % completion along each phase path."""

import warnings; warnings.filterwarnings('ignore')
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from shapely import line_locate_point, force_2d
from _paths import OUTPUTS

OUT = OUTPUTS
INDEX_FILE = OUT / '00_index_10sec.csv'
OUT_FILE = OUT / 'mergeddata_all.csv'
OUT_FILE_PCT = OUT / 'mergeddata_all_%complete.csv'
MERGE_KEYS = ['ParticipantID', 'PhaseID', 'Datetime']
PHASES_5 = {'BikeU', 'WalkU', 'BikeG', 'WalkG', 'Tram'}
GPKG_DIR = Path(__file__).resolve().parent / 'Experiment path'
PHASE_GPKG = {p: f'{p}.gpkg' for p in PHASES_5}

def add_pct_complete(df):
    """Add pct_complete column: for each phase with a GPKG path file,
    project GPS points onto the path line and compute % distance along it."""
    df['pct_complete'] = np.nan

    for phase, gpkg_file in PHASE_GPKG.items():
        gpkg_path = GPKG_DIR / gpkg_file
        if not gpkg_path.exists():
            print(f'  [pct_complete] GPKG not found for {phase}, skipping.')
            continue

        mask = df['PhaseID'] == phase
        n_rows = mask.sum()
        if n_rows == 0:
            continue

        path_gdf = gpd.read_file(gpkg_path)
        line = force_2d(path_gdf.geometry.iloc[0])
        line_length = line.length

        pct_vals = []
        for idx in df.index[mask]:
            lat = df.at[idx, 'GPS_lat']
            lon = df.at[idx, 'GPS_lon']
            if pd.isna(lat) or pd.isna(lon):
                pct_vals.append(np.nan)
                continue
            pt = Point(lon, lat)
            dist_along = line_locate_point(line, pt)
            pct_vals.append(round((dist_along / line_length) * 100, 1))

        df.loc[mask, 'pct_complete'] = pct_vals
        valid = [v for v in pct_vals if not np.isnan(v)]
        rng = f'{min(valid):.1f}% → {max(valid):.1f}%' if valid else 'N/A'
        skipped = sum(1 for v in pct_vals if np.isnan(v))
        print(f'  [pct_complete] {phase}: {n_rows} rows, {rng} (skipped {skipped} NaN GPS)')

    return df


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
        if name in ('index_10sec', 'mergeddata_all', 'mergeddata_all_%complete'):
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

    # Add % completion column
    merged = add_pct_complete(merged)

    merged.to_csv(OUT_FILE, index=False)
    print(f'\nSaved: {OUT_FILE} — {len(merged):,} rows x {merged.shape[1]} cols')
    print(f'  Participants: {sorted(merged["ParticipantID"].unique())}')
    print(f'  Phases: {sorted(merged["PhaseID"].unique())}')

if __name__ == '__main__':
    main()
