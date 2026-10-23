"""
=========================================================
Project Configuration
Confidence-Driven Predictive Localization Dashboard
=========================================================
"""

import os
from pathlib import Path


def _env_path(name, default):
    """Return the path in environment variable ``name``, else ``default``."""
    value = os.environ.get(name)
    return Path(value) if value else Path(default)

# ==========================================================
# PROJECT ROOT
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

# ==========================================================
# KITTI RAW DATASET
# ==========================================================

KITTI_ROOT = _env_path("KITTI_ROOT", PROJECT_ROOT / "data" / "KITTI")

KITTI_DATE = "2011_09_26"

KITTI_DATE_PATH = KITTI_ROOT / KITTI_DATE

DEFAULT_DRIVE = "2011_09_26_drive_0022_sync"

# ==========================================================
# KITTI ODOMETRY DATASET
# ==========================================================

KITTI_ODOMETRY_ROOT = _env_path("KITTI_ODOMETRY_ROOT", PROJECT_ROOT / "data" / "KITTI_Odometry")

KITTI_ODOMETRY_SEQUENCES = (
    KITTI_ODOMETRY_ROOT / "data_odometry_color" / "dataset" / "sequences"
)

KITTI_ODOMETRY_POSES = (
    KITTI_ODOMETRY_ROOT / "data_odometry_poses" / "dataset" / "poses"
)

KITTI_ODOMETRY_CALIB = (
    KITTI_ODOMETRY_ROOT / "data_odometry_calib" / "dataset" / "sequences"
)

# Sequences that also have ground-truth poses (00-10)
KITTI_ODOMETRY_POSE_SEQUENCES = [f"{i:02d}" for i in range(11)]

# All available colour sequences (00-21)
KITTI_ODOMETRY_ALL_SEQUENCES = [f"{i:02d}" for i in range(22)]

# ==========================================================
# NOTEBOOKS
# ==========================================================

NOTEBOOK_ROOT = _env_path("NOTEBOOK_ROOT", PROJECT_ROOT / "notebooks")

# ==========================================================
# PAPER 2
# ==========================================================

# Paper 2 = CAMSF repository (https://github.com/vishwa-10147/CAMSF), cloned next to this one
PAPER2_ROOT = _env_path("PAPER2_ROOT", PROJECT_ROOT.parent / "CAMSF")

PAPER2_NOTEBOOKS = PAPER2_ROOT / "notebooks"

PAPER2_MODELS = PAPER2_ROOT / "models"

BEST_LOCALIZATION_MODEL = PAPER2_MODELS / "best_localization_model.joblib"

LOCALIZATION_RELIABILITY_MODEL = (
    PAPER2_MODELS / "localization_reliability_rf.joblib"
)

# ==========================================================
# PAPER 3
# ==========================================================

# Paper 3 = CPL repository (https://github.com/vishwa-10147/CPL), cloned next to this one
PAPER3_ROOT = _env_path("PAPER3_ROOT", PROJECT_ROOT.parent / "CPL")

PAPER3_NOTEBOOKS = PAPER3_ROOT / "Notebooks"

PAPER3_RESULTS = PAPER3_ROOT / "Results"

PAPER3_MODELS = PAPER3_RESULTS / "models"

BEST_LSTM = PAPER3_MODELS / "best_lstm.pt"

BEST_GRU = PAPER3_MODELS / "best_gru.pt"

BEST_TRANSFORMER = PAPER3_MODELS / "best_transformer.pt"

LAST_LSTM = PAPER3_MODELS / "last_lstm.pt"

LAST_GRU = PAPER3_MODELS / "last_gru.pt"

LAST_TRANSFORMER = PAPER3_MODELS / "last_transformer.pt"

SEQUENCE_DATASET = PAPER3_RESULTS / "sequence_dataset"

SCALER = SEQUENCE_DATASET / "scaler.pkl"

DATALOADER_INFO = SEQUENCE_DATASET / "dataloader_info.pt"

# ==========================================================
# DASHBOARD OUTPUTS
# ==========================================================

OUTPUT_ROOT = PROJECT_ROOT / "outputs"

CSV_OUTPUT = OUTPUT_ROOT / "csv"

IMAGE_OUTPUT = OUTPUT_ROOT / "images"

VIDEO_OUTPUT = OUTPUT_ROOT / "videos"

REPORT_OUTPUT = OUTPUT_ROOT / "reports"

# ==========================================================
# DASHBOARD SETTINGS
# ==========================================================

WINDOW_TITLE = "Confidence-Driven Predictive Localization Dashboard"

FPS = 10

DEFAULT_SEQUENCE_LENGTH = 20

PREDICTION_HORIZON = 30

MAP_ZOOM = 18

# ==========================================================
# VISUALIZATION
# ==========================================================

TRAJECTORY_HISTORY = 150

SHOW_CAMERA = True

SHOW_GPS = True

SHOW_PREDICTION = True

SHOW_CONFIDENCE = True

SHOW_SENSOR_STATUS = True

SHOW_DECISION_PANEL = True

# ==========================================================
# COLORS
# ==========================================================

GOOD_COLOR = "#2ECC71"

WARNING_COLOR = "#F39C12"

DANGER_COLOR = "#E74C3C"

GPS_COLOR = "#3498DB"

IMU_COLOR = "#9B59B6"

CAMERA_COLOR = "#1ABC9C"

PREDICTION_COLOR = "#FF006E"

# ==========================================================
# CHECK PATHS
# ==========================================================

REQUIRED_PATHS = [
    KITTI_ROOT,
    KITTI_DATE_PATH,
    KITTI_ODOMETRY_ROOT,
    PAPER2_ROOT,
    PAPER2_MODELS,
    PAPER3_ROOT,
    PAPER3_MODELS,
    NOTEBOOK_ROOT,
]

for path in REQUIRED_PATHS:
    if not path.exists():
        print(f"[WARNING] Missing: {path}")