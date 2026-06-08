from __future__ import annotations

import re, uuid, shutil
from pathlib import Path
import numpy as np
import pandas as pd

# ── Paths ──
SOURCE_CPW = Path(r'C:\Users\pandya\OneDrive - UCL\Field experiment raw data\Complete Participantwise data\Questionnaire')
RAW_DIR    = Path(r'C:\Users\pandya\Documents\Github\docker\rawdata\05_questionnaire')
OUT_DIR    = Path(r'C:\Users\pandya\Documents\Github\docker\ExpData\outputs')
KEY_CSV    = Path(r'C:\Users\pandya\Documents\Github\docker\ExpData\metadata\key.csv')

BACKGROUND_PATH = RAW_DIR / "Background questionnaire.csv"
CLOTHING_PATH = RAW_DIR / "Clothing.csv"
RECURRING_PATH = RAW_DIR / "Recurring questionnaire.csv"

SCORED_PATH = OUT_DIR / "05_questionnaires_merged_scored.csv"

# PhaseID mapping: underscore format (from questionnaire) -> no-underscore (matches index/other sensors)
PHASEID_TO_INDEX = {
    'Walk_U': 'WalkU', 'Bike_U': 'BikeU',
    'Walk_G': 'WalkG', 'Bike_G': 'BikeG',
    'Tram': 'Tram',
    'Indoor': 'Indoor', 'Reststop': 'Reststop',
}

# ============================================================
# Column mappings — raw questionnaire text -> internal names
# ============================================================
BACKGROUND_COLUMNS = {
    "What is your name?": "background_name",
    "What is your age?": "background_age",
    "What is your gender?": "background_gender",
    "What is your nationality or country of origin?": "background_nationality",
    "Where do you live in Brussels?": "background_residential_area",
    "How long have you been living in Brussels?": "background_years_in_brussels",
    "What do you normally use for vision correction?": "background_vision_correction",
    "How well do you know the streets and surroundings of Bois de la Cambre & Etterbeek station?":
        "background_site_familiarity_1_7",
    "What is the type of your residence?": "background_residence_type",
    "Do you have children?": "background_has_children",
    "How do you commute to work?": "background_commute_modes",
    "Is there any sport or physical activity you practice regularly? Name it. Else write no.":
        "background_regular_physical_activity",
    "On average how much do you walk every day?": "background_daily_walking",
    "How frequently do you bike? ": "background_bike_frequency",
    "Will you be bringing your own bicycle for experiment? \n\nNote: If you don't have a personal bike, "
    "bike will be provided. In this case, please select No so that we can arrange a bike for you.":
        "background_own_bike_for_experiment",
    "Status": "background_status",
}

CLOTHING_COLUMNS = {
    "Q1. What is your height according to your most recent measurement? "
    "(Please specify the unit, e.g., centimeters or feet/inches.)": "clothing_height_raw",
    "Q2. What is your weight according to your most recent measurement? "
    "(Please specify the unit, e.g., kilograms or pounds.)": "clothing_weight_raw",
    "Q3. How would you describe your skin tone?\n"
    "(This helps us interpret optical sensor readings and thermal comfort exposure. "
    "You may skip this question if you prefer.)": "clothing_skin_tone",
    "Q4. What are you wearing on your upper body? (e.g., light blue cotton T-shirt, black hoodie)":
        "clothing_upper_body",
    "Q5. What are you wearing on your lower body? (e.g., dark blue jeans, beige trousers)":
        "clothing_lower_body",
    "Q6. What are you wearing on your feet? (e.g., white sneakers, black sandals)": "clothing_footwear",
    "Q7. Are you wearing any accessories? (e.g., sunglasses, cap, watch, none)": "clothing_accessories",
}

CLOTHING_SOURCE_ALIASES = {
    "Q1. What is your height according to your most recent measurement? "
    "(Please specify the unit, e.g., centimeters or feet/inches.)": [
        "Q1. What is your height according to your most recent measurement? "
        "(Please specify the unit, e.g., centimeters or feet/inches.)",
        "Q1. What is your height according to your most recent measurement? (Feets)",
    ],
    "Q2. What is your weight according to your most recent measurement? "
    "(Please specify the unit, e.g., kilograms or pounds.)": [
        "Q2. What is your weight according to your most recent measurement? "
        "(Please specify the unit, e.g., kilograms or pounds.)",
        "Q2. What is your weight according to your most recent measurement? (kilograms)",
    ],
}

RECURRING_RAW_COLUMNS = {
    "Do you consent to participating in this experiment?": "consent_response",
    "How comfortable do you feel right now?": "current_comfort_raw",
    "How familiar are you with this space?": "space_familiarity_raw",
    "Please describe your comfort [Thermal]": "thermal_comfort_raw",
    "Please describe your comfort [Acoustic (Urban Noise)]": "acoustic_comfort_raw",
    "Please describe your comfort [Visual (Light Comfort)]": "visual_comfort_raw",
    "Please describe your comfort [Air Quality (Emissions)]": "air_quality_comfort_raw",
    "Please describe your comfort [Odors (Smells)]": "odor_comfort_raw",
    "Right now, I feel: [Calm]": "stai_calm_raw",
    "Right now, I feel: [Tense]": "stai_tense_raw",
    "Right now, I feel: [Upset]": "stai_upset_raw",
    "Right now, I feel: [Relaxed]": "stai_relaxed_raw",
    "Right now, I feel: [Content]": "stai_content_raw",
    "Right now, I feel: [Worried]": "stai_worried_raw",
    "In this place, I feel or experience:  [Places like that are fascinating]": "prs_fascination_1_raw",
    "In this place, I feel or experience:  [In places like this my attention is drawn to many interesting things]": "prs_fascination_2_raw",
    "In this place, I feel or experience:  [In places like this it is hard to be bored]": "prs_fascination_3_raw",
    "In this place, I feel or experience:  [Places like that are a refuge from nuisances]": "prs_being_away_1_raw",
    "In this place, I feel or experience:  [To get away from things that usually demand my attention I like to go to places like this]": "prs_being_away_2_raw",
    "In this place, I feel or experience:  [To stop thinking about the things that I must get done I like to go to places like this]": "prs_being_away_3_raw",
    "In this place, I feel or experience:  [There is a clear order in the physical arrangement of places like this]": "prs_coherence_1_raw",
    "In this place, I feel or experience:  [In places like this it is easy to see how things are organised]": "prs_coherence_2_raw",
    "In this place, I feel or experience:  [In places like this everything seems to have its proper place]": "prs_coherence_3_raw",
    "In this place, I feel or experience:  [That place is large enough to allow exploration in many directions]": "prs_scope_1_raw",
    "In this place, I feel or experience:  [In places like that there are few boundaries to limit my possibility for moving about]": "prs_scope_2_raw",
    "How pleasant do you feel in this\nplace?": "pleasantness_raw",
}

# ============================================================
# Response text -> numeric score maps
# ============================================================
STAI_MAP_5 = {
    "Not at all": 1, "Somewhat": 2, "Moderately": 3, "Very much": 4, "Extremely": 5,
}
PRS_MAP_5 = {
    "Not at all": 1, "A little": 2, "Moderately": 3, "Quite a lot": 4, "Extremely": 5,
}
COMFORT_MAP_5 = {
    "Extremely uncomfortable": 1, "Very uncomfortable": 2, "Non comfortable": 3,
    "Slightly non comfortable": 4, "Comfortable": 5,
}

# ============================================================
# Phase schedule — built from key.csv (per-participant windows)
# ============================================================
# Belgium summer time offset: key.csv timestamps are in UTC,
# questionnaire timestamps are in Brussels local time (CEST = UTC+2)
BELGIUM_OFFSET = pd.Timedelta(hours=2)

def load_phase_windows(key_csv: Path) -> dict[str, list[tuple[str, pd.Timestamp, pd.Timestamp]]]:
    """Build {participant_id_str: [(phase, start_dt, end_dt), ...]} from key.csv.
    Phase windows are shifted to Brussels local time (+2h) to match questionnaire timestamps.
    Windows are sorted chronologically by start time."""
    key = pd.read_csv(key_csv)
    windows: dict[str, list[tuple[str, pd.Timestamp, pd.Timestamp]]] = {}
    PHASE_NAMES = ['BikeU', 'WalkU', 'BikeG', 'WalkG', 'Tram']
    for _, row in key.iterrows():
        pid = 'P' + str(int(row['Participant_ID']))
        date_str = pd.to_datetime(row['Date'] + '-2025', format='%d-%b-%Y').strftime('%Y-%m-%d')
        pw = []
        for ph in PHASE_NAMES:
            s_col, e_col = f'{ph}_start', f'{ph}_end'
            if s_col not in row or e_col not in row or pd.isna(row[s_col]) or pd.isna(row[e_col]):
                continue
            start = pd.Timestamp(f"{date_str} {row[s_col]}") + BELGIUM_OFFSET
            end   = pd.Timestamp(f"{date_str} {row[e_col]}") + BELGIUM_OFFSET
            if end < start:
                end += pd.Timedelta(days=1)
            pw.append((ph, start, end))
        # Sort by start time so "between phases" logic works correctly
        pw.sort(key=lambda x: x[1])
        windows[pid] = pw
    return windows


def assign_phase_by_key(ts: pd.Timestamp, pid: str,
                        phase_windows: dict) -> str | pd.NA:
    """Match a questionnaire timestamp to a participant's phase window.
    Key times were shifted to local time already. If timestamp is between
    two phases, assign to the one that just ended. Before first = Indoor."""
    if pd.isna(ts):
        return pd.NA
    pw = phase_windows.get(pid)
    if not pw:
        return pd.NA
    # Check within-window first
    for ph, start, end in pw:
        if start <= ts <= end:
            return ph
    # Before first phase -> Indoor
    if ts < pw[0][1]:
        return 'Indoor'
    # Between phases — assign to the one that just ended
    for idx in range(len(pw)):
        ph, _, end = pw[idx]
        if idx + 1 < len(pw):
            _, next_start, _ = pw[idx + 1]
            if end < ts < next_start:
                return ph
        else:
            # After last phase -> still assign to last phase
            if ts > end:
                return ph
    return pd.NA


def normalize_participant_id(value: object) -> str | pd.NA:
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return pd.NA
    match = re.search(r"(\d+)", text)
    return match.group(1) if match else text.upper()


def parse_height_m(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower().replace(",", ".")
    if not text or text == "nan":
        return np.nan
    # Feet/inches
    feet_m = re.search(r"(\d+(?:\.\d+)?)\s*(?:feet|foot|ft)", text)
    inch_m = re.search(r"(\d+(?:\.\d+)?)\s*(?:inch|inches|in)\b", text)
    if feet_m:
        return round((float(feet_m.group(1)) * 12 + (float(inch_m.group(1)) if inch_m else 0.0)) * 0.0254, 4)
    # Compact "1m80"
    compact_m = re.fullmatch(r"(\d(?:\.\d+)?)m(\d{2})", text.replace(" ", ""))
    if compact_m:
        return round(float(compact_m.group(1)) + float(compact_m.group(2)) / 100.0, 4)
    # cm
    if "cm" in text or "centimeter" in text:
        m = re.search(r"(\d+(?:\.\d+)?)", text)
        if m:
            return round(float(m.group(1)) / 100.0, 4)
    # Generic "m"
    if "m" in text:
        m = re.search(r"(\d+(?:\.\d+)?)", text)
        if m:
            v = float(m.group(1))
            return round(v if v < 3 else v / 100.0, 4)
    # Bare number
    m = re.search(r"(\d+(?:\.\d+)?)", text)
    if not m:
        return np.nan
    v = float(m.group(1))
    if 4 <= v <= 8:
        return round(v * 0.3048, 4)
    return round(v / 100.0, 4) if v > 3 else round(v, 4)


def parse_weight_kg(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower().replace(",", ".")
    if not text or text == "nan":
        return np.nan
    m = re.search(r"(\d+(?:\.\d+)?)", text)
    if not m:
        return np.nan
    v = float(m.group(1))
    if "lb" in text or "pound" in text:
        return round(v * 0.45359237, 3)
    return round(v, 3)


def bmi_category_who(value: float) -> str | pd.NA:
    if pd.isna(value):
        return pd.NA
    if value < 18.5:
        return "Underweight"
    if value < 25:
        return "Normal weight"
    if value < 30:
        return "Overweight"
    return "Obesity"


def first_existing_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in frame.columns:
            return c
    return None


def parse_mixed_timestamp(value: object) -> pd.Timestamp:
    if pd.isna(value):
        return pd.NaT
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return pd.NaT
    for fmt in ["%m-%d-%y %H:%M", "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M"]:
        parsed = pd.to_datetime(text, format=fmt, errors="coerce")
        if pd.notna(parsed):
            return parsed
    return pd.to_datetime(text, errors="coerce")


def ensure_columns(frame: pd.DataFrame, column_names: list[str] | tuple[str, ...] | dict) -> pd.DataFrame:
    names = column_names.keys() if isinstance(column_names, dict) else column_names
    for col in names:
        if col not in frame.columns:
            frame[col] = pd.NA
    return frame


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{uuid.uuid4().hex}.tmp")
    frame.to_csv(tmp, index=False, encoding="utf-8")
    tmp.replace(path)


# ============================================================
# Load functions
# ============================================================

def load_background() -> pd.DataFrame:
    bg = pd.read_csv(BACKGROUND_PATH)
    pid_col = first_existing_column(bg, ["Participant id", "ParticipantID"])
    if pid_col is None:
        raise KeyError("Participant id column not found in background questionnaire.")
    bg["participant_id"] = bg[pid_col].apply(normalize_participant_id)
    bg = ensure_columns(bg, BACKGROUND_COLUMNS)
    bg = bg.loc[bg["participant_id"].notna(),
                ["participant_id", *BACKGROUND_COLUMNS.keys()]].rename(columns=BACKGROUND_COLUMNS)
    bg["background_site_familiarity_1_7"] = pd.to_numeric(
        bg["background_site_familiarity_1_7"], errors="coerce")
    return bg


def load_clothing() -> pd.DataFrame:
    cl = pd.read_csv(CLOTHING_PATH)
    pid_col = first_existing_column(cl, ["Q0. What is your participant id?", "ParticipantID"])
    if pid_col is None:
        raise KeyError("Participant id column not found in clothing questionnaire.")
    cl["participant_id"] = cl[pid_col].apply(normalize_participant_id)
    for canonical, aliases in CLOTHING_SOURCE_ALIASES.items():
        src = first_existing_column(cl, aliases)
        cl[canonical] = cl[src] if src else pd.NA
    cl = ensure_columns(cl, CLOTHING_COLUMNS)
    cl["height_m"] = cl[
        "Q1. What is your height according to your most recent measurement? "
        "(Please specify the unit, e.g., centimeters or feet/inches.)"
    ].apply(parse_height_m)
    cl["weight_kg"] = cl[
        "Q2. What is your weight according to your most recent measurement? "
        "(Please specify the unit, e.g., kilograms or pounds.)"
    ].apply(parse_weight_kg)
    cl["bmi_kg_m2"] = (cl["weight_kg"] / (cl["height_m"] ** 2)).round(2)
    cl["bmi_category_who"] = cl["bmi_kg_m2"].apply(bmi_category_who)
    cl = cl.loc[cl["participant_id"].notna(),
                ["participant_id", *CLOTHING_COLUMNS.keys(),
                 "height_m", "weight_kg", "bmi_kg_m2", "bmi_category_who"]
               ].rename(columns=CLOTHING_COLUMNS)
    return cl


def prepare_recurring_raw() -> pd.DataFrame:
    rq = pd.read_csv(RECURRING_PATH)
    pid_col = first_existing_column(rq, ["Provide your participant id.", "ParticipantID"])
    if pid_col is None:
        raise KeyError("Participant id column not found in recurring questionnaire.")
    rq["participant_id"] = rq[pid_col].apply(normalize_participant_id)
    # Parse timestamp internally for phase assignment — NOT in output
    rq["_ts"] = rq["Timestamp"].apply(parse_mixed_timestamp)
    rq = rq.sort_values(["participant_id", "_ts"]).copy()
    rq["response_index_within_participant"] = rq.groupby("participant_id").cumcount() + 1
    # Load key.csv phase windows for per-participant phase assignment
    _pw = load_phase_windows(KEY_CSV)
    rq["PhaseID"] = rq.apply(lambda r: assign_phase_by_key(r["_ts"],
                             "P" + str(r["participant_id"]), _pw), axis=1)
    rq = ensure_columns(rq, RECURRING_RAW_COLUMNS)
    raw = rq.loc[
        :,
        ["participant_id", "response_index_within_participant", "PhaseID",
         *RECURRING_RAW_COLUMNS.keys()]
    ].rename(columns=RECURRING_RAW_COLUMNS)
    return raw


# ============================================================
# Scoring logic
# ============================================================

def score_recurring(raw_recurring: pd.DataFrame) -> pd.DataFrame:
    """Convert raw text responses to numeric scores.

    Returns ONLY the aggregate scored columns — no intermediate
    individual-item columns or raw text, so the output is easy
    to interpret.
    """
    scored = raw_recurring.copy()

    # --- Direct numeric ---
    scored["current_comfort_1_7"] = pd.to_numeric(scored["current_comfort_raw"], errors="coerce")
    scored["space_familiarity_1_5"] = pd.to_numeric(scored["space_familiarity_raw"], errors="coerce")
    scored["pleasantness_1_5"] = pd.to_numeric(scored["pleasantness_raw"], errors="coerce")

    # --- 5 environmental comfort dimensions ---
    comfort_raw_to_score = {
        "thermal_comfort_raw": "thermal_comfort_1_5",
        "acoustic_comfort_raw": "acoustic_comfort_1_5",
        "visual_comfort_raw": "visual_comfort_1_5",
        "air_quality_comfort_raw": "air_quality_comfort_1_5",
        "odor_comfort_raw": "odor_comfort_1_5",
    }
    for raw_col, score_col in comfort_raw_to_score.items():
        scored[score_col] = scored[raw_col].map(COMFORT_MAP_5)

    scored["environmental_comfort_mean_1_5"] = scored[
        list(comfort_raw_to_score.values())
    ].mean(axis=1)

    # --- STAI-6 (adapted) ---
    stai_map = {
        "stai_calm_raw": "stai_calm",
        "stai_tense_raw": "stai_tense",
        "stai_upset_raw": "stai_upset",
        "stai_relaxed_raw": "stai_relaxed",
        "stai_content_raw": "stai_content",
        "stai_worried_raw": "stai_worried",
    }
    for raw_col, score_col in stai_map.items():
        scored[score_col] = scored[raw_col].map(STAI_MAP_5)

    # Reverse-code positive items so higher = more anxiety
    for pos_col in ["stai_calm", "stai_relaxed", "stai_content"]:
        scored[f"{pos_col}_rev"] = scored[pos_col].apply(
            lambda v: 6 - v if pd.notna(v) else np.nan
        )

    stai_final = ["stai_calm_rev", "stai_tense", "stai_upset",
                  "stai_relaxed_rev", "stai_content_rev", "stai_worried"]
    scored["stai6_mean_1_5"] = scored[stai_final].mean(axis=1)

    # --- PRS-11 (adapted) ---
    prs_map = {
        "prs_fascination_1_raw": "prs_fascination_1",
        "prs_fascination_2_raw": "prs_fascination_2",
        "prs_fascination_3_raw": "prs_fascination_3",
        "prs_being_away_1_raw": "prs_being_away_1",
        "prs_being_away_2_raw": "prs_being_away_2",
        "prs_being_away_3_raw": "prs_being_away_3",
        "prs_coherence_1_raw": "prs_coherence_1",
        "prs_coherence_2_raw": "prs_coherence_2",
        "prs_coherence_3_raw": "prs_coherence_3",
        "prs_scope_1_raw": "prs_scope_1",
        "prs_scope_2_raw": "prs_scope_2",
    }
    for raw_col, score_col in prs_map.items():
        scored[score_col] = scored[raw_col].map(PRS_MAP_5)

    fac_cols = ["prs_fascination_1", "prs_fascination_2", "prs_fascination_3"]
    away_cols = ["prs_being_away_1", "prs_being_away_2", "prs_being_away_3"]
    coh_cols = ["prs_coherence_1", "prs_coherence_2", "prs_coherence_3"]
    scope_cols = ["prs_scope_1", "prs_scope_2"]
    all_prs = [*fac_cols, *away_cols, *coh_cols, *scope_cols]

    scored["prs11_fascination_mean_1_5"] = scored[fac_cols].mean(axis=1)
    scored["prs11_being_away_mean_1_5"] = scored[away_cols].mean(axis=1)
    scored["prs11_coherence_mean_1_5"] = scored[coh_cols].mean(axis=1)
    scored["prs11_scope_mean_1_5"] = scored[scope_cols].mean(axis=1)
    scored["prs11_total_mean_1_5"] = scored[all_prs].mean(axis=1)

    # --- Keep ONLY the aggregate columns in output ---
    # Add ParticipantID in 'P2' format and normalize PhaseID to index convention
    scored["ParticipantID"] = "P" + scored["participant_id"].astype(str)
    scored["PhaseID"] = scored["PhaseID"].map(PHASEID_TO_INDEX).fillna(scored["PhaseID"])

    output_cols = [
        "ParticipantID", "participant_id", "response_index_within_participant", "PhaseID",
        "current_comfort_1_7",
        "space_familiarity_1_5",
        "pleasantness_1_5",
        "environmental_comfort_mean_1_5",
        "stai6_mean_1_5",
        "prs11_fascination_mean_1_5",
        "prs11_being_away_mean_1_5",
        "prs11_coherence_mean_1_5",
        "prs11_scope_mean_1_5",
        "prs11_total_mean_1_5",
    ]
    return scored[output_cols].copy()


def merge_tables(recurring_table: pd.DataFrame, clothing: pd.DataFrame,
                 background: pd.DataFrame) -> pd.DataFrame:
    merged = recurring_table.merge(clothing, on="participant_id", how="left", validate="m:1")
    merged = merged.merge(background, on="participant_id", how="left", validate="m:1")
    return merged


# ============================================================
# Main
# ============================================================

def generate_outputs() -> dict[str, str | int]:
    # Collect: CPW -> rawdata/questionnaire/
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for f in SOURCE_CPW.glob('*.csv'):
        dst = RAW_DIR / f.name
        if not dst.exists():
            shutil.copy2(f, dst)
    
    background = load_background()
    clothing = load_clothing()
    recurring_raw = prepare_recurring_raw()
    recurring_scored = score_recurring(recurring_raw)

    # Filter to key participants only (key has int IDs, recurring has string IDs)
    key = pd.read_csv(KEY_CSV)
    valid_pids = set(key['Participant_ID'].dropna().astype(int).astype(str).unique())
    before = len(recurring_scored)
    recurring_scored = recurring_scored[recurring_scored['participant_id'].isin(valid_pids)].copy()
    print(f"  Filtered: {before} -> {len(recurring_scored)} rows (key has {len(valid_pids)} participants)")

    merged_scored = merge_tables(recurring_scored, clothing, background)

    write_csv_atomic(merged_scored, SCORED_PATH)

    return {
        "scored_csv": str(SCORED_PATH),
        "rows": len(merged_scored),
        "participants": merged_scored["participant_id"].nunique(),
    }


if __name__ == "__main__":
    outputs = generate_outputs()
    for key, value in outputs.items():
        print(f"{key}: {value}")
