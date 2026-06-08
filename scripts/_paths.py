from pathlib import Path
REPO_ROOT = Path(r"C:\Users\pandya\Documents\Github\docker\ExpData")

OUTPUTS  = REPO_ROOT / "outputs"
METADATA = REPO_ROOT / "metadata"
KEY_FILE = METADATA / "key.csv"

RAW_DATA_DIR   = Path(r"C:\Users\pandya\Documents\Github\docker\rawdata")
RAW_UCM_DIR    = RAW_DATA_DIR / "02_ucm"        # UCM collected CSVs
RAW_ET_DIR     = RAW_DATA_DIR / "04_eyetracker"  # Eyetracker collected CSVs
RAW_QUEST_DIR  = RAW_DATA_DIR / "05_questionnaire"  # Questionnaire collected CSVs

CPW_ROOT      = Path(r"C:\Users\pandya\OneDrive - UCL\Field experiment raw data\Complete Participantwise data")
EMPA_RAW_DIR  = CPW_ROOT / "empatica_raw"        # Empatica AVRO files
ATMO_LYS_DIR  = CPW_ROOT / "Atmo_lys"            # Atmotube + LYS CSVs
NEON_ROOT_DIR = CPW_ROOT / "Eyetracker raw files"  # Neon Player exports
QUEST_SOURCE  = CPW_ROOT / "Questionnaire"        # Questionnaire source CSVs

# ── External processed data (used by validate scripts) ───────────────────────
DP_ROOT      = Path(r"C:\Users\pandya\OneDrive - UCL\Pranav PhD\Paper 3\Data Processing_05May")
DP_V2_CSV    = DP_ROOT / "outputs" / "empatica_corrected_10sec_v2.csv"  # Empatica v2 CSV
DP_KEY_FULL  = DP_ROOT / "key_full.csv"                                   # Full participant key
