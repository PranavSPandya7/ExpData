"""
_align_index.py
================
Index alignment helper used by 01_empatica_build and 04_eyetracker_build.
Left-joins a sensor DataFrame onto the index backbone and forward-fills
small gaps (≤3 min / 18 slots) within each phase.
"""

import pandas as pd
from _paths import OUTPUTS


def align_to_index(df: pd.DataFrame, name: str = '') -> pd.DataFrame:
    """Left-join sensor DataFrame onto index backbone.
    Ensures every 10-sec slot has a row; NaN where sensor missing.
    Forward-fills small gaps (≤3 min / 18 slots) within each phase.
    """
    idx_path = OUTPUTS / '00_index_10sec.csv'
    if not idx_path.exists():
        print(f'  WARNING: index not found, run 00_index_build.py first')
        return df

    idx = pd.read_csv(idx_path, low_memory=False)
    idx['Datetime'] = pd.to_datetime(idx['Datetime'])
    idx['ParticipantID'] = idx['ParticipantID'].astype(str)
    idx['PhaseID'] = idx['PhaseID'].fillna('').astype(str)

    df['Datetime'] = pd.to_datetime(df['Datetime'])
    if hasattr(df['Datetime'].dtype, 'tz') and df['Datetime'].dt.tz is not None:
        df['Datetime'] = df['Datetime'].dt.tz_localize(None)
    df['Datetime'] = df['Datetime'].dt.floor('10s')
    df['ParticipantID'] = df['ParticipantID'].astype(str)
    # Make 'nan' and 'None' both match empty string so between-phase rows align
    df['PhaseID'] = df['PhaseID'].fillna('').astype(str).replace({'nan':'','None':''})
    idx['PhaseID'] = idx['PhaseID'].fillna('').astype(str).replace({'nan':'','None':''})

    n_before = len(idx)
    result = idx[['ParticipantID', 'PhaseID', 'Datetime']].merge(
        df, on=['ParticipantID', 'PhaseID', 'Datetime'], how='left'
    )
    signal_cols = result.columns[3:]
    if len(signal_cols):
        result[signal_cols] = result.groupby(['ParticipantID', 'PhaseID'])[signal_cols].transform(
            lambda g: g.ffill(limit=18).bfill(limit=1)
        )
    n_missing = result.iloc[:, 3:].isna().all(axis=1).sum()
    if n_missing:
        print(f'  [{name}] {n_missing:,}/{n_before:,} slots have no data')
    return result
