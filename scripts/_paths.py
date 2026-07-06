from pathlib import Path
from datetime import datetime
import re
import warnings
import pandas as pd
REPO_ROOT = Path(r"C:\Users\pandya\Documents\Github\docker\ExpData")

OUTPUTS  = Path(r"C:\Users\pandya\Documents\Github\docker\Paper3_Github\output")
KEY_PATH = Path(r"C:\Users\pandya\Documents\Github\docker\Paper3_Github\output")
KEY_FILE = KEY_PATH / "key.csv"

RAW_DATA_DIR   = Path(r"C:\Users\pandya\Documents\Github\docker\Paper3_Github\rawdata")
RAW_EMPA_DIR   = RAW_DATA_DIR / "01_empatica"
RAW_UCM_DIR    = RAW_DATA_DIR / "02_ucm"        # UCM collected CSVs
RAW_ATMO_DIR   = RAW_DATA_DIR / "03_atmo_lys"   # Atmo/LYS collected CSVs
RAW_ET_DIR     = RAW_DATA_DIR / "04_eyetracker"  # Eyetracker collected CSVs
RAW_QUEST_DIR  = RAW_DATA_DIR / "05_questionnaire"  # Questionnaire collected CSVs
BUILD_WARNINGS_MD = OUTPUTS / "build_warnings.md"

CPW_ROOT      = Path(r"C:\Users\pandya\OneDrive - UCL\Field experiment raw data\Complete Participantwise data")
FINAL_DATA    = CPW_ROOT / "Final Data"
EMPA_RAW_DIR  = RAW_EMPA_DIR  # Empatica AVRO files
ATMO_LYS_DIR  = RAW_ATMO_DIR  # Atmotube + LYS CSVs
NEON_ROOT_DIR = RAW_ET_DIR    # Staged Neon Player exports
QUEST_SOURCE  = RAW_QUEST_DIR # Questionnaire source CSVs

PHASES_MAIN = ["BikeU", "WalkU", "BikeG", "WalkG", "Tram"]
_WARNING_LOG_SCRIPT = None


def setup_build_warning_log(script_file: str | Path) -> None:
    """Append Python warnings from this build script to output/build_warnings.md."""
    global _WARNING_LOG_SCRIPT
    _WARNING_LOG_SCRIPT = Path(script_file).name
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with BUILD_WARNINGS_MD.open("a", encoding="utf-8") as fh:
        fh.write(f"\n## {_WARNING_LOG_SCRIPT} - {timestamp}\n")

    previous_showwarning = warnings.showwarning

    def showwarning(message, category, filename, lineno, file=None, line=None):
        previous_showwarning(message, category, filename, lineno, file, line)
        with BUILD_WARNINGS_MD.open("a", encoding="utf-8") as fh:
            fh.write(f"- {category.__name__}: {message} (`{filename}:{lineno}`)\n")

    warnings.showwarning = showwarning
    warnings.filterwarnings("default")


def log_build_warning(message: str) -> None:
    """Print and append custom build warnings that are not Python warnings.warn calls."""
    print(message)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    script = _WARNING_LOG_SCRIPT or "unknown script"
    with BUILD_WARNINGS_MD.open("a", encoding="utf-8") as fh:
        fh.write(f"- WARNING [{script}]: {message}\n")

SENSOR_FORBIDDEN_TOKENS = {
    "empatica": ["ucm", "atmo", "lys", "eye", "tracker", "questionnaire"],
    "ucm": ["empatica", "atmo", "lys", "eye", "tracker", "questionnaire"],
    "atmo_lys": ["empatica", "ucm", "eye", "tracker", "questionnaire"],
    "eyetracker": ["empatica", "ucm", "atmo", "lys", "questionnaire"],
    "questionnaire": ["empatica", "ucm", "atmo", "lys", "eye", "tracker"],
}


def load_key_unique(key_file: Path = KEY_FILE) -> pd.DataFrame:
    key = pd.read_csv(key_file)
    key = key.dropna(subset=["Participant_ID"]).copy()
    key["Participant_ID"] = key["Participant_ID"].astype(int)
    dupes = key[key.duplicated("Participant_ID", keep=False)]["Participant_ID"].unique()
    if len(dupes):
        print(f"  [key] duplicate participant rows found; keeping first row for: {sorted(dupes.tolist())}")
    return key.drop_duplicates("Participant_ID", keep="first").sort_values("Participant_ID").reset_index(drop=True)


def key_participant_ids(key_file: Path = KEY_FILE) -> list[int]:
    return load_key_unique(key_file)["Participant_ID"].astype(int).tolist()


def assert_sensor_folder_clean(sensor_name: str, root: Path) -> None:
    """Fail fast when a staged sensor folder contains another sensor's data."""
    root.mkdir(parents=True, exist_ok=True)
    forbidden = SENSOR_FORBIDDEN_TOKENS.get(sensor_name, [])
    offenders = []
    for path in root.rglob("*"):
        rel_parts = [part.lower() for part in path.relative_to(root).parts]
        text = " ".join(rel_parts)
        for token in forbidden:
            if re.search(rf"(^|[^a-z0-9]){re.escape(token)}([^a-z0-9]|$)", text):
                offenders.append(str(path))
                break
        if len(offenders) >= 10:
            break
    if offenders:
        listed = "\n    ".join(offenders)
        raise RuntimeError(
            f"Sensor folder hygiene failed for {sensor_name}: {root}\n"
            f"Found paths that look like another sensor:\n    {listed}"
        )

