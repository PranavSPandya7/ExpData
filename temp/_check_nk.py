import neurokit2 as nk, numpy as np, pandas as pd
from pathlib import Path
from avro.datafile import DataFileReader
from avro.io import DatumReader
import sys
sys.path.insert(0, 'C:/Users/pandya/Documents/Github/docker/ExpData/scripts')
from importlib import import_module

# Replicate v2 process_bvp_segment exactly
SR_BVP = 64
MIN_BVP_S = 8
PPG_QUALITY_MIN = 0.5

def process_bvp_segment_v2(bvp_df, sr):
    if len(bvp_df) < int(sr * MIN_BVP_S):
        return None, []
    try:
        vals = bvp_df['bvp'].values.astype(float)
        vals = np.where(np.isnan(vals), 0.0, vals)
        sig, _ = nk.ppg_process(vals, sampling_rate=int(round(sr)))
        quality = sig['PPG_Quality'].values[:len(bvp_df)]
        hr_raw = sig['PPG_Rate'].values[:len(bvp_df)]
        hr_series = pd.Series(hr_raw, index=bvp_df.index, name='heart_rate')
        peak_mask = (sig['PPG_Peaks'].values[:len(bvp_df)] == 1) & (quality >= PPG_QUALITY_MIN)
        peak_times = bvp_df.index[peak_mask].tolist()
        return hr_series, peak_times
    except Exception:
        return None, []

# Read one AVRO file for P13
avro_dir = Path('C:/Users/pandya/Documents/Github/docker/rawdata/01_empatica/2025-08-13')
avros = sorted(avro_dir.rglob('*.avro'))
print(f'P13 has {len(avros)} AVRO files')
print(f'NeuroKit2 version: {nk.__version__}')

# Find BikeU time window
key = pd.read_csv('C:/Users/pandya/Documents/Github/docker/ExpData/metadata/key.csv')
p13_row = key[key['Participant_ID']==13].iloc[0]
date = pd.to_datetime(p13_row['Date'] + '-2025', format='%d-%b-%Y').strftime('%Y-%m-%d')
start = pd.Timestamp(f'{date} {p13_row["BikeU_start"]}', tz='UTC').tz_convert('Europe/Brussels')
end = pd.Timestamp(f'{date} {p13_row["BikeU_end"]}', tz='UTC').tz_convert('Europe/Brussels')

print(f'BikeU window: {start} -> {end}')

# Test first AVRO file
from _us_to_brussels import _us_to_brussels
