from pathlib import Path
import re
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

CPW_ROOT      = Path(r"C:\Users\pandya\OneDrive - UCL\Field experiment raw data\Complete Participantwise data")
FINAL_DATA    = CPW_ROOT / "Final Data"
EMPA_RAW_DIR  = RAW_EMPA_DIR  # Empatica AVRO files
ATMO_LYS_DIR  = RAW_ATMO_DIR  # Atmotube + LYS CSVs
NEON_ROOT_DIR = RAW_ET_DIR    # Staged Neon Player exports
QUEST_SOURCE  = RAW_QUEST_DIR # Questionnaire source CSVs

PHASES_MAIN = ["BikeU", "WalkU", "BikeG", "WalkG", "Tram"]

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

