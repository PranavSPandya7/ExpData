import neurokit2 as nk, numpy as np, pandas as pd
from pathlib import Path
from avro.datafile import DataFileReader
from avro.io import DatumReader

def _us_to_brussels(unix_us_array):
    return pd.to_datetime(unix_us_array, unit="us", utc=True).tz_convert("Europe/Brussels")

avro_dir = Path("C:/Users/pandya/Documents/Github/docker/rawdata/01_empatica/2025-08-13")
avros = sorted(avro_dir.rglob("*.avro"))
key = pd.read_csv("C:/Users/pandya/Documents/Github/docker/ExpData/metadata/key.csv")
p13 = key[key["Participant_ID"]==13].iloc[0]
date = pd.to_datetime(p13["Date"] + "-2025", format="%d-%b-%Y").strftime("%Y-%m-%d")
start = pd.Timestamp(f"{date} {p13["BikeU_start"]}", tz="UTC").tz_convert("Europe/Brussels")
end = pd.Timestamp(f"{date} {p13["BikeU_end"]}", tz="UTC").tz_convert("Europe/Brussels")
print(f"BikeU: {start} -> {end}")

for av in avros:
    reader = DataFileReader(open(str(av), "rb"), DatumReader())
    data = next(reader)
    reader.close()
    raw = data["rawData"]
    if "bvp" not in raw:
        continue
    b = raw["bvp"]
    n = len(b["values"])
    if n == 0:
        continue
    sf = float(b["samplingFrequency"])
    us = np.round(b["timestampStart"] + np.arange(n) * (1e6 / sf)).astype(np.int64)
    t0 = _us_to_brussels(us[0])
    t1 = _us_to_brussels(us[-1])
    if t0 > end or t1 < start:
        continue
    vals = np.array(b["values"], dtype=float)
    vals = np.where(np.isnan(vals), 0.0, vals)
    sig, _ = nk.ppg_process(vals, sampling_rate=int(round(sf)))
    hr_raw = sig["PPG_Rate"].values[:n]
    n_hr_ok = np.sum(~np.isnan(hr_raw))
    print(f"{av.name}: {n} samples, HR non-NaN={n_hr_ok}/{n}")
    if n_hr_ok == 0:
        cleaned = nk.ppg_clean(vals, sampling_rate=int(round(sf)))
        pk = nk.signal_findpeaks(cleaned, sampling_rate=int(round(sf)), relative_height_min=0.3)
        print(f"  Fallback (0.3): {len(pk["Peaks"])} peaks")
        pk2 = nk.signal_findpeaks(cleaned, sampling_rate=int(round(sf)), relative_height_min=0.1)
        print(f"  Fallback (0.1): {len(pk2["Peaks"])} peaks")

