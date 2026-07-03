# ExpData Processing Pipeline

This is the active Paper 3 processing pipeline. It stages raw files from Final Data, builds 10-second sensor outputs, validates those outputs, and merges them for analysis.

`README.md` is the current explanation. `PIPELINE_EXPLANATION.md` is older context and may be stale.

## Authoritative Paths

- Active repo: `C:\Users\pandya\Documents\Github\docker\ExpData`
- Active scripts: `C:\Users\pandya\Documents\Github\docker\ExpData\scripts`
- Final Data source: `C:\Users\pandya\OneDrive - UCL\Field experiment raw data\Complete Participantwise data\Final Data`
- Staged rawdata: `C:\Users\pandya\Documents\Github\docker\Paper3_Github\rawdata`
- Active outputs: `C:\Users\pandya\Documents\Github\docker\Paper3_Github\output`
- Active key file: `C:\Users\pandya\Documents\Github\docker\Paper3_Github\output\key.csv`
- Python environment: `C:\Users\pandya\Documents\Github\docker\ExpData\.venv`

Run from `C:\Users\pandya\Documents\Github\docker\ExpData`:

```powershell
& .venv\Scripts\Activate.ps1
python scripts\<script_name>.py
```

Use the same `.venv` kernel for `scripts\01_empatica_build.ipynb`.

## Normal Run Order

1. `00_rawdata_collect.py`
2. `00_index_build.py`
3. `01_empatica_build.ipynb`
4. `02_ucm_build.py`
5. `03_atmo_lys_build.py`
6. `04_eyetracker_build.py`
7. `05_questionnaire_build.py`
8. `06_all_merge.py`
9. Main validators: `11_empatica_validate.py`, `12_ucm_validate.py`, `13_atmo_lys_validate.py`, `14_eyetracker_validate.py`
10. Other UCM diagnostics: `12_ucm_validate_cut.py`, `12_ucm_phase_maps_unclipped.py`
11. Quality and Gap Check: `00_QC_Gap_check.py`

Run `00_rawdata_collect.py` first. It collects Final Data into the staged rawdata folder. Build scripts read only staged `rawdata`.

## Script Explanations

### Shared helpers

- `_paths.py`
  - Reads no data table directly.
  - Defines the active Final Data, rawdata, output, and key paths.
  - Provides participant/key helpers and staged-folder hygiene checks.

- `_align_index.py`
  - Reads `output\00_index_10sec.csv`.
  - Aligns one sensor table to the canonical 10-second index.
  - Preserves missing sensor values rather than filling them.

### Collection and index

- `00_rawdata_collect.py`
  - Reads Final Data folders.
  - Writes staged rawdata under `Paper3_Github\rawdata`.
  - Empatica: copies raw Empatica folders to `rawdata\01_empatica`.
  - UCM: copies UCM `data.csv` files to `rawdata\02_ucm`.
  - Atmo/LYS: copies required `Atmo_left`, `Atmo_right`, `LYS1`, and `LYS2` CSVs to `rawdata\03_atmo_lys`.
  - Eyetracker: copies each Neon `exports\000` folder flat into `rawdata\04_eyetracker\Pxx_Phase`.
  - Questionnaire: copies `Background questionnaire.csv`, `Clothing.csv`, and `Recurring questionnaire.csv` to `rawdata\05_questionnaire`.

- `00_index_build.py`
  - Reads `output\key.csv`. This is UTC timezone
  - Writes `output\00_index_10sec.csv`. Adds 2 hour to become Brussel local time.
  - Creates the canonical 10-second participant-phase timeline used by all sensor outputs.

### Sensor builds

- `01_empatica_build.ipynb`
  - Reads Empatica AVRO files from `rawdata\01_empatica`, plus `key.csv` and `00_index_10sec.csv`.
  - Writes `01_empatica_corrected_10sec.csv`.
  - Also writes `01_empatica_native_input_intermediate.csv` and `01_empatica_rri_native.csv` for inspection/debugging of native Empatica conversion and RRI processing.
  - Builds 10-second HR, HRV, EDA, skin temperature, and movement summaries.
  - Uses NeuroKit2 where appropriate, because published libraries are easier to defend and cite than custom physiology code.

- `02_ucm_build.py`
  - Reads staged UCM `data.csv` files from `rawdata\02_ucm`.
  - Reads `00_index_10sec.csv`.
  - Writes `02_ucm_10sec.csv`.
  - Builds UCM GPS, sound, thermal, radiation, wind, and environmental variables at 10-second resolution.
  - Uses `GPS_time` as the source timestamp.
  - Shifts every UCM `GPS_time` by +2 hours to convert UTC to Brussels/local experiment time.
  - This +2 hour shift happens before candidate file selection, 10-second aggregation, and index joining (because index is in Brussels local time = UTC + 2 hours).
  - Filters invalid sound values below 0 dBA before 10-second aggregation; if all sound values in a bin are invalid, the bin is blank.
  - Filters invalid UTCI values at or below 0 before 10-second aggregation; if all UTCI values in a bin are invalid, the bin is blank.

- `03_atmo_lys_build.py`
  - Reads staged Atmotube and LYS CSVs from `rawdata\03_atmo_lys`.
  - Reads `00_index_10sec.csv`.
  - Writes `03_atmo_lys_merged.csv`.
  - Builds one Atmotube + LYS 10-second table because these sensors describe the same environmental exposure timeline.

- `04_eyetracker_build.py`
  - Reads Neon exports from `rawdata\04_eyetracker\Pxx_Phase`.
  - Required source files are `gaze_positions.csv`, `3d_eye_states.csv`, `fixations.csv`, `saccades.csv`, and `blinks.csv`.
  - Writes one per-phase intermediate `rawdata\04_eyetracker\Pxx_Phase\output.csv`.
  - Writes the combined output `04_eyetracker_10sec.csv`.
  - Maps fixation, saccade, and blink events onto gaze timestamps, bins them to 10-second windows, and requires enough gaze samples for a valid 10-second row.

- `05_questionnaire_build.py`
  - Reads the three staged questionnaire CSVs from `rawdata\05_questionnaire`.
  - Writes `05_questionnaires_merged_scored.csv`.
  - Standardizes participant/phase labels and scores STAI, PRS, comfort, clothing, and related variables.

- `06_all_merge.py`
  - Reads `00_index_10sec.csv` and the outputs from scripts `01` to `05`.
  - Writes `mergeddata_all.csv`.
  - Writes `merged_all_11participants.csv`.
  - Left-merges sensor outputs to the canonical 10-second index and joins questionnaire values by participant-phase.
  - Adds `pct_complete` when route geometry is available. Tram is time-based. If `rawdata\Experiment path\*.gpkg` files are missing, Bike/Walk `pct_complete` is skipped.

### UCM map and clipping scripts

- `12_ucm_validate.py`
  - Reads `02_ucm_10sec.csv`, `key.csv`, and Final Data UCM photos.
  - Writes `12_ucm_quality_report.html`.
  - Main UCM validation report. It shows clipped phase-window maps and photo bookends for the processed UCM data the pipeline uses.

- `12_ucm_phase_maps_unclipped.py`
  - Reads raw staged UCM files from `rawdata\02_ucm` and `key.csv`.
  - Writes one raw/unclipped map per phase under `output\12_ucm_phase_maps_unclipped`.
  - These maps show recorded tracks before strict phase clipping. They are useful because some UCM files contain continuous recording across more than one phase.

- `12_ucm_validate_cut.py`
  - Reads `02_ucm_10sec.csv`, `key.csv`, and Final Data UCM photos.
  - Writes `12_ucm_quality_report_cut.html`.
  - This is the BikeU cut-window sensitivity report. To check if separate results from BikeU with more participants can be published. BikeU is split around `Cut_start` and `Cut_end` to get the matching continuous path so more Urban participants can be visually assessed while keeping the route trajectory more consistent. This adds 4, 5, 6, 7

### Validators and QC

- `11_empatica_validate.py`
  - Reads `01_empatica_corrected_10sec.csv`.
  - Writes `11_empatica_quality_report.html`.
  - Shows Empatica coverage, ranges, phase windows, and time-series gaps.

- `13_atmo_lys_validate.py`
  - Reads `03_atmo_lys_merged.csv`.
  - Writes `13_atmo_lys_quality_report.html`.
  - Shows Atmo/LYS paired sensor checks and light exposure plots.

- `14_eyetracker_validate.py`
  - Reads `04_eyetracker_10sec.csv`.
  - Writes `14_eyetracker_quality_report.html`.
  - Shows phase-aligned eyetracker plots and coverage. It does not read per-phase `output.csv`.

- `00_QC_Gap_check.py`
  - Reads staged rawdata and current processed outputs.
  - Writes `QC_gap_missing_percent_by_column.csv`.
  - Writes `QC_invalid_percent_by_column.csv`.
  - Use after builds exist to review missingness and invalid-value percentages by parameter and participant.

## Output Files

All main outputs are in:

`C:\Users\pandya\Documents\Github\docker\Paper3_Github\output`

### Required metadata and backbone

- `key.csv`
  - Participant phase timing file. This is the source for phase windows.

- `00_index_10sec.csv`
  - Canonical 10-second participant-phase index. All sensor outputs align to this.

### Sensor outputs

- `01_empatica_corrected_10sec.csv`
  - Main Empatica 10-second output with HR, HRV, EDA, temperature, and movement summaries.

- `01_empatica_native_input_intermediate.csv`
  - Empatica intermediate native-stream table used to inspect the converted input streams.

- `01_empatica_rri_native.csv`
  - Empatica intermediate RRI/beat table used for HRV inspection and troubleshooting.

- `02_ucm_10sec.csv`
  - Main UCM backpack 10-second GPS/environmental output.

- `03_atmo_lys_merged.csv`
  - Main Atmotube + LYS 10-second output.

- `04_eyetracker_10sec.csv`
  - Main combined eyetracker 10-second output from all staged phase exports.

- `05_questionnaires_merged_scored.csv`
  - Scored questionnaire participant-phase output.

### Merged analysis outputs

- `mergeddata_all.csv`
  - Full merged 10-second dataset across index, sensors, and questionnaire data.

- `merged_all_11participants.csv`
  - Analysis subset containing P4 and P8-P17.

### Validation HTML outputs

- `11_empatica_quality_report.html`
  - Empatica validation report.

- `12_ucm_quality_report.html`
  - Main clipped UCM validation report.

- `12_ucm_quality_report_cut.html`
  - UCM BikeU cut-window validation report.

- `13_atmo_lys_quality_report.html`
  - Atmo/LYS validation report.

- `14_eyetracker_quality_report.html`
  - Eyetracker validation report.

### UCM unclipped map outputs

These are under:

`C:\Users\pandya\Documents\Github\docker\Paper3_Github\output\12_ucm_phase_maps_unclipped`

- `bikeu_all_participants_unclipped_map.html`
  - Raw/unclipped BikeU UCM map.

- `walku_all_participants_unclipped_map.html`
  - Raw/unclipped WalkU UCM map.

- `bikeg_all_participants_unclipped_map.html`
  - Raw/unclipped BikeG UCM map.

- `walkg_all_participants_unclipped_map.html`
  - Raw/unclipped WalkG UCM map.

- `tram_all_participants_unclipped_map.html`
  - Raw/unclipped Tram UCM map.

### QC outputs

- `QC_gap_missing_percent_by_column.csv`
  - Missingness percentages by parameter and participant.

- `QC_invalid_percent_by_column.csv`
  - Invalid/impossible value percentages by parameter and participant, using QC bounds defined in the QC script.

### Eyetracker staged intermediate outputs

- `rawdata\04_eyetracker\Pxx_Phase\output.csv`
  - Per-phase eyetracker intermediate generated by `04_eyetracker_build.py`.
  - This is replaced each build.
  - It is useful for checking individual participant-phase exports.
  - The validator and merge use `output\04_eyetracker_10sec.csv`, not these per-phase files.

## Gap-Filling And Missingness Rules

- Index:
  - Creates expected 10-second rows.
  - Does not create sensor values.

- Empatica:
  - No final forward/back fill.
  - HRV remains blank where beat quality or duration is insufficient.
  - NeuroKit interpolation is internal to frequency-domain HRV calculation only.

- UCM:
  - Numeric values are 10-second medians after validity filtering.
  - Invalid sound below 0 dBA is masked before aggregation.
  - Invalid UTCI at or below 0 is masked before aggregation.
  - UCM `GPS_time` is shifted +2 hours first, then matched to `00_index_10sec.csv`.
  - Weak GPS can blank GPS-derived fields without deleting environmental fields.

- Atmo/LYS:
  - One-minute readings are expanded to the matching six 10-second rows for that minute.
  - No broad post-join filling.

- Eyetracker:
  - A 10-second bin requires at least 600 gaze samples.
  - Missing or low-quality windows remain blank after index alignment.

- Questionnaire:
  - Participant-phase rows are joined by participant and phase.
  - Questionnaire values are not interpolated over time.

- Merge:
  - Left-joins all outputs to the index.
  - Does not fill sensor gaps.

## Current Important Notes

- P13 has no eyetracker source phase folders and is expected to be blank for eyetracker.
- Eyetracker staged files are flat under `rawdata\04_eyetracker\Pxx_Phase`.
- There should not be staged `neon_player\exports\000` subfolders in `rawdata\04_eyetracker`.
- P1 eyetracker saccade and blink event files were corrected from P5 same-phase donor events, retimed to P1 eyetracker windows, clipped/tiled to phase duration, and given 5% noise where requested.
- Generated staged eyetracker `output.csv` files are build products, not manual source files.
- If `rawdata\Experiment path\*.gpkg` files are missing, Bike/Walk `pct_complete` is skipped and Tram remains time-based.
- Prefer published libraries and established tools where already used. They reduce implementation burden and are easier to cite than custom replacements.
