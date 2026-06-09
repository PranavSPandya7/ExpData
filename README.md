# ExpData Pipeline

## Overview
Pipeline that processes raw sensor data from a wearable experiment (biking, walking, tram in urban/green environments) into 10-second aligned CSV files, then merges into a single wide table. Also generates per-sensor HTML quality reports.

---

## 00 — Index Build (`00_index_build.py`)
Creates the 10-second continuous time grid that all sensor data aligns to. Reads phase start/end times from `metadata/key.csv`, converts UTC to Brussels time (+2h), and builds one row per 10-second slot from the earliest phase start to the latest phase end across all participants. Includes between-phase gaps. Output: `outputs/00_index_10sec.csv` (7,999 rows).

---

## 01 — Empatica E4 (`01_empatica_build.py`)
**Raw format:** AVRO files from the Empatica Embrace Plus wristband.

**Signal processing & gap filling (NeuroKit2):**
- **EDA:** Raw EDA (4 Hz) → `nk.eda_phasic()` separates tonic (slow baseline) from phasic (fast responses). Gaps up to 3 minutes forward-filled within each phase.
- **HR / BVP:** Blood Volume Pulse (64 Hz) → `nk.ppg_process()` detects peaks → inter-beat intervals → heart rate. Gaps forward-filled limit 18 slots (3 min).
- **Temperature & Accelerometer:** Downsampled to 10-sec by averaging. Gaps forward-filled.

**Device specs:** Empatica E4 samples at: EDA=4 Hz, BVP=64 Hz, ACC=32 Hz, TEMP=4 Hz. All resampled to 0.1 Hz (10-sec) after processing.

**Validation:** `11_empatica_validate.py` generates a quality report with full-day time-series plots, phase-shaded backgrounds, and coverage statistics per signal.

---

## 02 — UCM Backpack (`02_ucm_build.py`)
**Raw format:** GPS + environmental CSV logs from the UCM Hepia environmental backpack. Includes temperature, humidity, IR, air quality gases (CO₂, NO₂, O₃, SO₂, CO), particulate matter (PM1, PM2.5, PM10), sound level, wind, solar radiation, and derived comfort indices (UTCI, PET, Humidex, MRT).

**Processing:** Key.csv phase windows are used to assign PhaseID and to validate GPS coverage via photo bookend matching (first/last GPS fix per phase). Data is resampled to 10-sec bins.

**Validation:** `12_ucm_validate.py` generates an interactive Folium map with GPS tracks and photo bookend markers. `12_ucm_validate_v2.py` adds per-phase environmental signal quality plots.


---

## 03 — Atmotube + LYS (`03_atmo_lys_build.py`)
**Raw format:** CSV files from two AtmotubePro units (left/right on researcher) and two LYS button sensors (participant + researcher).

**Processing:**
- Atmotube: PM1, PM2.5, PM10, temperature, humidity, VOC (not used), AQS (not used).
- LYS: Lux, kelvin, medi (light intensity), movement, RGB channels.
- All sensors merged by participant and timestamp, resampled to 10-sec, aligned to the index backbone.

**Gap filling:** Forward-fill (limit 18 slots = 3 min) within each phase, then backward-fill (limit 1). Negatives clipped to 0. All NaN resolved.

**Validation:** `13_atmo_lys_validate.py` generates a per-participant phase-aligned plot report with auto-scaled y-axes.

---

## 04 — Eyetracker (`04_eyetracker_build.py`)
**Raw format:** Pupil Labs Neon exports — per-phase folders containing `output.csv`, `gaze_positions.csv`, `fixations.csv`, `saccades.csv`, `blinks.csv`, `3d_eye_states.csv`, `imu.csv`.

**Processing:**
- Reads per-phase CSV exports from `rawdata/04_eyetracker/`.
- Extracts pupil diameter (left, right, avg), pupil change, gaze position (x, y), fixation duration, saccade amplitude/velocity, blink duration.
- Adds 2h UTC offset to match Brussels time.
- Aligns to the 10-sec index backbone. Drops `in_fixation` and `in_saccade` boolean columns.

**Recovery module:** Corrupt or fragmented Neon MP4 recordings can be repaired using the `pl-recover-recording` module (reference: `C:\Users\pandya\OneDrive - UCL\Pranav PhD\03_Experiments & Data\src\pl-recover-recording`), which uses `untrunc` + reference videos to fix broken moov atoms in Neon camera recordings.

**Validation:** `14_eyetracker_validate.py` generates per-participant time-series plots of pupil, gaze, fixation, and saccade metrics.

---

## 05 — Questionnaires (`05_questionnaire_build.py`)
**Raw format:** CSV exports from the experiment's psychological survey platform.

**Scoring:**
- **STAI** (State-Trait Anxiety Inventory): Sum of 20 items (S-Anxiety) and 20 items (T-Anxiety), with reverse-scoring where applicable.
- **PRS** (Perceived Restorativeness Scale): Sub-scales for Being Away, Fascination, Coherence, and Compatibility.
- **Comfort scales:** Thermal comfort, visual comfort, acoustic comfort, air quality perception, and overall environmental satisfaction.
- All scores merged by participant and phase, aligned to the 10-sec index.

---

## 06 — Merge (`06_all_merge.py`)
Reads all 00-05 output CSVs and merges them into one wide table by `[ParticipantID, PhaseID, Datetime]` using left-joins onto the index backbone.

Output: `outputs/mergeddata_all.csv` — 7,999 rows × ~154 columns. PhaseID is empty `''` for between-phase gaps. Each build script is independent and should be run in order (00 → 06).

---

## Setup — One Time Only

```powershell
# Create & activate virtual environment
python -m venv .venv-1
.venv-1\Scripts\Activate.ps1

# Install all dependencies (pins NK >=0.2.12 to avoid HR crash bug)
pip install -r requirements.txt
```

All scripts are **self-contained**: they read raw sensor data directly (AVRO, CSV) and produce outputs with no external tools needed.

## How to Run (In Order)

```powershell
# Always activate the venv first — system Python has NK 0.2.9 which crashes!
.venv-1\Scripts\Activate.ps1

# Build (00 → 05)
python scripts/00_index_build.py       # 10-sec time grid
python scripts/01_empatica_build.py    # Empatica Embrace Plus (AVRO → HR/EDA/ACC)
python scripts/02_ucm_build.py         # UCM backpack (GPS + environment)
python scripts/03_atmo_lys_build.py    # Atmotube + LYS (air + light)
python scripts/04_eyetracker_build.py  # Pupil Labs Neon (gaze + pupil)
python scripts/05_questionnaire_build.py  # STAI, PRS, comfort scores

# Validate (11 → 14)
python scripts/11_empatica_validate.py
python scripts/12_ucm_validate.py
python scripts/13_atmo_lys_validate.py
python scripts/14_eyetracker_validate.py

# Merge (06)
python scripts/06_all_merge.py         # → outputs/mergeddata_all.csv
```

**⚠️ Critical:** Always use the venv Python. The system `python` has NK 0.2.9 which has a bug that silently produces broken HR. The script itself checks NK version and aborts if too old.

## Configuration
- **`_paths.py`** — Central path config: `REPO_ROOT`, `OUTPUTS`, `KEY_FILE`, `RAW_DATA_DIR`.
- **`_align_index.py`** — Helper: left-joins a sensor DataFrame onto the index backbone, forward-fills (limit 18), backward-fills (limit 1), normalises PhaseID.
