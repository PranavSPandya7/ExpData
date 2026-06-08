================================================================================
  OUTPUTS FOLDER - README
  Project: Comfort, Physiology & Urban Mobility Field Experiment
  Author:  Pranav Pandya, UCL
  Last updated: May 2026
================================================================================

This folder contains the processed 10-second sensor output files produced by
the pipeline scripts in ../scripts/.  All files are read-only outputs derived
from raw data in ../raw data/.

--------------------------------------------------------------------------------
OUTPUT FILES
--------------------------------------------------------------------------------

empatica_10sec.csv   - Empatica Embrace Plus physiological signals (10-sec bins)
eyetracker_10sec.csv - Pupil Labs Neon gaze & pupil metrics (10-sec bins)
ucm_10sec.csv        - UCM Hepia backpack environmental sensors (10-sec bins)
atmo_lys_10sec.csv   - Atmotube air quality + LYS light sensors (10-sec bins)

mergeddata_10sec.csv - Master merge of all 4 sensor streams + questionnaires
                       (produced by 05_merge_all.py)

--------------------------------------------------------------------------------
HOW THE EMPATICA OUTPUT IS DERIVED  (01_build_empatica.py)
--------------------------------------------------------------------------------

Source: Raw binary AVRO files from Empatica Embrace Plus wristband
        Location: raw data/Empatica Data/YYYY-MM-DD/{device}/*.avro

Step 1 - AVRO reading (read-only, no intermediate CSV written)
  Each .avro file is decoded using the Apache Avro library (avro.datafile).
  Sensor signals extracted at their native sampling rates:
    - Accelerometer (acc_x/y/z):  ~32 Hz
    - Gyroscope (gyro_x/y/z):     ~32 Hz
    - EDA (electrodermal activity): ~4 Hz
    - Temperature (skin):           ~1 Hz
    - BVP (blood volume pulse):    ~64 Hz
    - Steps:                     variable

Step 2 - Heart Rate and HRV from Systolic Peaks
  The device performs onboard PPG peak detection and stores peak timestamps
  as peaksTimeNanos in the AVRO rawData.systolicPeaks field.
  - IBI (inter-beat intervals) computed from consecutive peak times
  - Physiological gate applied: 0.30 s < IBI < 2.0 s  (30-200 bpm)
  - Heart Rate = 60 / mean IBI, resampled to 1-sec bins
  - HRV RMSSD = sqrt(mean(diff(IBI)^2)) * 1000  [ms], computed per 10-sec bin
    (requires >= 3 valid IBIs per bin)

Step 3 - Resampling to 10-second bins
  Each sensor is first resampled to 1-second (mean), then all sensors are
  merged on the 1-second index and resampled to 10-second bins (median).
  Median is preferred over mean because it is robust to motion artifacts and
  brief signal spikes that inflate the mean.
  Short internal gaps (<= 3 bins = 30 sec) are linearly interpolated;
  longer gaps are left as NaN to preserve data integrity.

  Vector magnitude (acc resultant) = sqrt(acc_x^2 + acc_y^2 + acc_z^2)
  is computed after resampling.

Step 4 - Phase windowing using Key file
  Key file: Key_for_data_splitting_updated_05May.csv
  Contains Start/End times (Brussels local time) for each of 5 phases per
  participant per date:
    BikeU -> Bike_U  (urban cycling, Usquare/Etterbeek area)
    WalkU -> Walk_U  (urban walking, Usquare to Montgomery)
    BikeG -> Bike_G  (green cycling, La Cambre park)
    WalkG -> Walk_G  (green walking, La Cambre park)
    Tram  -> Tram    (tram, Boondael Gare to Etterbeek Gare)

  Only rows whose timestamp falls within [Start, End] for each phase are kept.
  Rows where ALL signal columns are NaN are dropped.
  Timestamps in the output Datetime column are Brussels local time (UTC+2),
  timezone info stripped for CSV compatibility.

NOTE ON ACCLIMATIZATION:
  Current pipeline keeps all rows from phase start. For physiological analysis
  comparing across phases, it is recommended to exclude the first 60 seconds
  of each phase (acclimatization / physiological stabilization period).
  This is NOT currently implemented and should be applied at the analysis stage
  or by setting start = start + 60s before filtering.

--------------------------------------------------------------------------------
METHODOLOGY REFERENCES
--------------------------------------------------------------------------------

[1] Healey, J. A., & Picard, R. W. (2005).
    Detecting Stress During Real-World Driving Tasks Using Physiological Sensors.
    IEEE Transactions on Intelligent Transportation Systems, 6(2), 156-166.
    https://doi.org/10.1109/TITS.2005.848368
    -- Used EDA and HR in ambulatory vehicle study; established 10-sec windowing
       for mobile physiological assessment.

[2] Task Force of the European Society of Cardiology and the North American
    Society of Pacing and Electrophysiology (1996).
    Heart Rate Variability: Standards of Measurement, Physiological Interpretation
    and Clinical Use. Circulation, 93(5), 1043-1065.
    https://doi.org/10.1161/01.CIR.93.5.1043
    -- Defines RMSSD as the standard short-term HRV metric. Basis for the IBI
       gate (30-200 bpm) and RMSSD formula used here.

[3] Makowski, D., Pham, T., Lau, Z. J., et al. (2021).
    NeuroKit2: A Python toolbox for neurophysiological signal processing.
    Behavior Research Methods, 53, 1689-1696.
    https://doi.org/10.3758/s13428-020-01516-y
    -- Reference for best-practice EDA cleaning (Butterworth lowpass filter)
       and PPG-based HR/HRV computation. Current pipeline does not use NK2 for
       EDA cleaning; consider adding nk.eda_clean() for peer-review robustness.

[4] Garbarino, M., Lai, M., Bender, D., Picard, R. W., & Tognetti, S. (2014).
    Empatica E4 - A wristband for the continuous monitoring of physiological
    parameters. 2014 International Conference on Wireless Mobile Communication
    and Healthcare, 39-42.
    https://doi.org/10.4108/icst.mobihealth.2014.257418
    -- Validation of the Empatica wristband EDA and BVP sensors; confirms
       suitability for ambulatory / field study use.

[5] Boucsein, W. (2012). Electrodermal Activity (2nd ed.). Springer.
    https://doi.org/10.1007/978-1-4614-1126-0
    -- Standard reference for EDA methodology; recommends minimum 60-sec
       stabilization period when activity context changes.

[6] Castaldo, R., Montesinos, L., Melillo, P., James, C., & Pecchia, L. (2019).
    Ultra-short term HRV features as surrogates of short term HRV: a case
    study on mental stress detection in real life. BMC Medical Informatics and
    Decision Making, 19, 12.
    https://doi.org/10.1186/s12911-019-0742-y
    -- Validates use of ultra-short (10-60 sec) RMSSD windows for field studies
       when full 5-min windows are not possible due to phase length constraints.

--------------------------------------------------------------------------------
IMPORTANT NOTES FOR RUNNING THE PIPELINE
--------------------------------------------------------------------------------

1. AVRO files must be placed in raw data/Empatica Data/YYYY-MM-DD/ before
   running 01_build_empatica.py. The script searches recursively (rglob) so
   subdirectory structure within the date folder does not matter.

2. The AVRO conversion logic is EMBEDDED inside 01_build_empatica.py
   (functions: read_avro_sensors, process_avro_folder). There is NO separate
   AVRO-to-CSV step. The script reads AVRO directly and outputs empatica_10sec.csv.

3. Run order: 01 -> 02 -> 03 -> 04 -> 05
   Scripts 01-04 are independent; 05_merge_all.py requires all 4 outputs.

4. Requires: pip install avro-python3 (or avro), neurokit2 (optional, for EDA cleaning)

================================================================================
