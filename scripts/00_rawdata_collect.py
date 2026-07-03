"""Collect required Final Data inputs into Paper3_Github/rawdata.

This is the single staging step for the active pipeline. It force-copies the
files that build scripts read, then the build scripts only consume rawdata.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import (
    FINAL_DATA,
    RAW_ATMO_DIR,
    RAW_DATA_DIR,
    RAW_EMPA_DIR,
    RAW_ET_DIR,
    RAW_QUEST_DIR,
    RAW_UCM_DIR,
    assert_sensor_folder_clean,
    key_participant_ids,
)

PHASES = ["BikeU", "WalkU", "BikeG", "WalkG", "Tram"]
ATMO_LYS_SUFFIXES = ["LYS1", "LYS2", "Atmo_left", "Atmo_right"]
EYETRACKER_SKIP_SUFFIXES = {".mp4", ".raw", ".bin"}
QUESTIONNAIRE_FILES = [
    "Background questionnaire.csv",
    "Clothing.csv",
    "Recurring questionnaire.csv",
]


def reset_staged_dir(path: Path) -> None:
    root = RAW_DATA_DIR.resolve()
    target = path.resolve()
    if target == root or root not in target.parents:
        raise RuntimeError(f"Refusing to clear non-rawdata folder: {path}")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_file_force(src: Path, dst: Path) -> bool:
    if not src.exists() or not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    shutil.copy2(src, dst)
    return True


def copy_tree_files_force(src_root: Path, dst_root: Path, *, skip_file=None) -> int:
    copied = 0
    if not src_root.exists():
        return copied
    for src in src_root.rglob("*"):
        if not src.is_file():
            continue
        if skip_file is not None and skip_file(src):
            continue
        dst = dst_root / src.relative_to(src_root)
        if copy_file_force(src, dst):
            copied += 1
    return copied


def collect_empatica() -> int:
    src_root = FINAL_DATA / "empatica_raw"
    if not src_root.exists():
        print(f"[empatica] WARNING: source not found: {src_root}")
        return 0
    reset_staged_dir(RAW_EMPA_DIR)
    copied = copy_tree_files_force(src_root, RAW_EMPA_DIR)
    print(f"[empatica] copied {copied} files -> {RAW_EMPA_DIR}")
    return copied


def collect_ucm(pids: list[int]) -> int:
    src_root = FINAL_DATA / "ucm"
    if not src_root.exists():
        print(f"[ucm] WARNING: source not found: {src_root}")
        return 0
    reset_staged_dir(RAW_UCM_DIR)
    copied = 0
    for pid in pids:
        src_pid = src_root / f"P{pid}"
        if not src_pid.exists():
            print(f"[ucm] missing source folder: P{pid}")
            continue
        for src in src_pid.rglob("data.csv"):
            dst = RAW_UCM_DIR / f"P{pid}" / src.relative_to(src_pid)
            if copy_file_force(src, dst):
                copied += 1
    print(f"[ucm] copied {copied} data.csv files -> {RAW_UCM_DIR}")
    return copied


def collect_atmo_lys(pids: list[int]) -> int:
    src_root = FINAL_DATA / "Atmo_lys"
    if not src_root.exists():
        print(f"[atmo_lys] WARNING: source not found: {src_root}")
        return 0
    reset_staged_dir(RAW_ATMO_DIR)
    copied = 0
    for pid in pids:
        for suffix in ATMO_LYS_SUFFIXES:
            src = src_root / f"{pid}_{suffix}.csv"
            dst = RAW_ATMO_DIR / src.name
            if copy_file_force(src, dst):
                copied += 1
            else:
                print(f"[atmo_lys] missing source file: {src.name}")
    print(f"[atmo_lys] copied {copied} CSV files -> {RAW_ATMO_DIR}")
    return copied


def eyetracker_source_roots() -> list[Path]:
    candidates = [
        FINAL_DATA / "Eyetracker raw files",
        FINAL_DATA / "Eyetracker raw files" / "Eyetracker raw files",
        FINAL_DATA / "Eyetracker raw files2",
    ]
    return [root for root in candidates if root.exists()]


def collect_eyetracker(pids: list[int]) -> int:
    roots = eyetracker_source_roots()
    if not roots:
        print("[eyetracker] WARNING: no source roots found")
        return 0
    reset_staged_dir(RAW_ET_DIR)
    copied = 0
    for pid in pids:
        for phase in PHASES:
            folder_name = f"P{pid}_{phase}"
            src_phase = next((root / folder_name for root in roots if (root / folder_name).exists()), None)
            if src_phase is None:
                print(f"[eyetracker] missing source folder: {folder_name}")
                continue
            export_dirs = sorted(
                d for d in src_phase.rglob("000")
                if d.is_dir() and d.parent.name == "exports"
            )
            if not export_dirs:
                print(f"[eyetracker] missing exports/000: {folder_name}")
                continue
            dst_export = RAW_ET_DIR / folder_name
            copied += copy_tree_files_force(
                export_dirs[0],
                dst_export,
                skip_file=lambda src: src.suffix.lower() in EYETRACKER_SKIP_SUFFIXES,
            )
    print(f"[eyetracker] copied {copied} export files -> {RAW_ET_DIR}")
    return copied


def collect_questionnaire() -> int:
    src_root = FINAL_DATA / "Questionnaire"
    if not src_root.exists():
        print(f"[questionnaire] WARNING: source not found: {src_root}")
        return 0
    reset_staged_dir(RAW_QUEST_DIR)
    copied = 0
    for name in QUESTIONNAIRE_FILES:
        src = src_root / name
        dst = RAW_QUEST_DIR / name
        if copy_file_force(src, dst):
            copied += 1
        else:
            print(f"[questionnaire] missing source file: {name}")
    print(f"[questionnaire] copied {copied} CSV files -> {RAW_QUEST_DIR}")
    return copied


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Force-copy active raw inputs into Paper3_Github/rawdata.")
    parser.add_argument(
        "--sensor",
        action="append",
        choices=["empatica", "ucm", "atmo_lys", "eyetracker", "questionnaire"],
        help="Collect only this sensor. May be repeated. Default: all sensors.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sensors = args.sensor or ["empatica", "ucm", "atmo_lys", "eyetracker", "questionnaire"]
    pids = key_participant_ids()
    print(f"Using key participants: {', '.join(f'P{pid}' for pid in pids)}")
    print(f"Final Data root: {FINAL_DATA}")

    totals = {}
    if "empatica" in sensors:
        totals["empatica"] = collect_empatica()
        assert_sensor_folder_clean("empatica", RAW_EMPA_DIR)
    if "ucm" in sensors:
        totals["ucm"] = collect_ucm(pids)
        assert_sensor_folder_clean("ucm", RAW_UCM_DIR)
    if "atmo_lys" in sensors:
        totals["atmo_lys"] = collect_atmo_lys(pids)
        assert_sensor_folder_clean("atmo_lys", RAW_ATMO_DIR)
    if "eyetracker" in sensors:
        totals["eyetracker"] = collect_eyetracker(pids)
        assert_sensor_folder_clean("eyetracker", RAW_ET_DIR)
    if "questionnaire" in sensors:
        totals["questionnaire"] = collect_questionnaire()
        assert_sensor_folder_clean("questionnaire", RAW_QUEST_DIR)

    print("\nRawdata collection complete:")
    for sensor, count in totals.items():
        print(f"  {sensor}: {count} files copied")


if __name__ == "__main__":
    main()
