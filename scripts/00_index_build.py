"""Build continuous 10-sec index — from first phase start to last phase end (includes gaps)."""
import warnings; warnings.filterwarnings('ignore')
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
import numpy as np
from _paths import KEY_FILE, OUTPUTS, load_key_unique

OUT_FILE = OUTPUTS / '00_index_10sec.csv'
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
OFFSET = pd.Timedelta(hours=2)

def main():
    key = load_key_unique(KEY_FILE)
    rows = []
    for _, r in key.iterrows():
        pid = 'P' + str(int(r['Participant_ID']))
        date = pd.to_datetime(r['Date'] + '-2025', format='%d-%b-%Y').strftime('%Y-%m-%d')

        # Collect all phase windows for this participant
        windows = []
        for ph in ['BikeU', 'WalkU', 'BikeG', 'WalkG', 'Tram', 'Indoor']:
            sc, ec = f'{ph}_start', f'{ph}_end'
            if sc in r and ec in r and not pd.isna(r[sc]) and not pd.isna(r[ec]):
                s = (pd.Timestamp(f"{date} {r[sc]}") + OFFSET).floor('10s')
                e = (pd.Timestamp(f"{date} {r[ec]}") + OFFSET).ceil('10s')
                if e < s: e += pd.Timedelta(days=1)
                windows.append((ph, s, e))

        if not windows:
            continue

        # Continuous range: earliest start → latest end
        earliest = min(w[1] for w in windows)
        latest   = max(w[2] for w in windows)

        for ts in pd.date_range(start=earliest, end=latest, freq='10s'):
            # Gaps between named phases are rest stops in the continuous index.
            phase = 'reststop'
            for ph, s, e in windows:
                if s <= ts <= e:
                    phase = ph
                    break
            rows.append({'ParticipantID': pid, 'PhaseID': phase, 'Datetime': ts, 'Date': date})

    idx = pd.DataFrame(rows).sort_values(['ParticipantID', 'Datetime']).reset_index(drop=True)
    idx.to_csv(OUT_FILE, index=False)
    n_gaps = (idx['PhaseID'] == 'reststop').sum()
    n_phased = (idx['PhaseID'] != 'reststop').sum()
    print(f'Saved {len(idx):,} rows -> {OUT_FILE}')
    print(f'  Phase windows: {n_phased:,} rows | Gaps between phases: {n_gaps:,} rows')
    print(f'  Participants: {sorted(idx["ParticipantID"].unique())}')

if __name__ == '__main__':
    main()
