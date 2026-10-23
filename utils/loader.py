"""
=========================================================
Loader Module
Confidence-Driven Predictive Localization Dashboard
=========================================================

Responsibilities
----------------
1. Validate project resources
2. Discover KITTI drives
3. Load Paper 2 models
4. Load Paper 3 models
5. Provide reusable helper functions
"""

from pathlib import Path
import joblib
import torch

from config import *


# ==========================================================
# KITTI
# ==========================================================

def get_available_drives():
    """
    Returns all extracted KITTI Raw drive folders.
    """

    if not KITTI_DATE_PATH.exists():
        raise FileNotFoundError(
            f"KITTI dataset not found:\n{KITTI_DATE_PATH}"
        )

    drives = sorted([
        d.name
        for d in KITTI_DATE_PATH.iterdir()
        if d.is_dir() and "drive" in d.name
    ])

    return drives


def get_available_odometry_sequences():
    """
    Returns sorted sequence IDs available in the KITTI Odometry dataset.

    Only returns sequences whose colour image directory exists on disk.
    Sequences 00-10 also have ground-truth poses; 11-21 are test sequences.
    """

    if not KITTI_ODOMETRY_SEQUENCES.exists():
        raise FileNotFoundError(
            f"KITTI Odometry sequences not found:\n{KITTI_ODOMETRY_SEQUENCES}"
        )

    sequences = sorted([
        d.name
        for d in KITTI_ODOMETRY_SEQUENCES.iterdir()
        if d.is_dir() and (d / "image_2").exists()
    ])

    return sequences


def get_drive_path(drive_name):
    """
    Returns complete drive path.
    """

    drive = KITTI_DATE_PATH / drive_name

    if not drive.exists():
        raise FileNotFoundError(drive)

    return drive


# ==========================================================
# PAPER 2
# ==========================================================

def load_localization_model():

    print("Loading localization model...")

    return joblib.load(BEST_LOCALIZATION_MODEL)


def load_reliability_model():

    print("Loading reliability model...")

    return joblib.load(LOCALIZATION_RELIABILITY_MODEL)


# ==========================================================
# PAPER 3
# ==========================================================

def load_best_lstm():

    print("Loading LSTM...")

    return torch.load(
        BEST_LSTM,
        map_location="cpu"
    )


def load_best_gru():

    print("Loading GRU...")

    return torch.load(
        BEST_GRU,
        map_location="cpu"
    )


def load_best_transformer():

    print("Loading Transformer...")

    return torch.load(
        BEST_TRANSFORMER,
        map_location="cpu"
    )


# ==========================================================
# VALIDATION
# ==========================================================

def validate_project():

    print("=" * 60)
    print("PROJECT VALIDATION")
    print("=" * 60)

    print()

    required = [
        KITTI_ROOT,
        KITTI_DATE_PATH,
        PAPER2_ROOT,
        PAPER3_ROOT,
        NOTEBOOK_ROOT
    ]

    for item in required:

        if item.exists():
            print(f"[OK] {item}")
        else:
            print(f"[MISSING] {item}")

    print()

    drives = get_available_drives()

    print(f"Available Drives : {len(drives)}")

    for drive in drives:
        print(f"   • {drive}")

    print()

    print("Validation Complete.")

    print("=" * 60)