# PLD — Predictive Localization Dashboard

### Interactive research dashboard for confidence-driven adaptive multi-sensor localization

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-ff4b4b.svg)](https://streamlit.io/)
[![Dataset](https://img.shields.io/badge/Dataset-KITTI%20Raw%20%7C%20Odometry-green.svg)](https://www.cvlibs.net/datasets/kitti/)

> **Author:** K. Vishwa (Kokkireni Vishwa) · Department of Computer Science and Engineering, Mahatma Gandhi Institute of Technology (A), Hyderabad, India
> **Guide:** Dr. P. Poornima, Assistant Professor, Department of CSE, MGIT

---

## Overview

PLD replays **KITTI** drives frame by frame. For each frame it runs the complete **confidence-driven adaptive sensor-fusion pipeline** and shows every stage live:

1. Sensor metrics
2. Camera, GPS and IMU confidence
3. The Sensor Confidence Index (SCI)
4. Adaptive weights and adaptive covariance
5. The adaptive EKF
6. Machine-learning localization reliability

It is the interactive companion to the research series [SCI](https://github.com/vishwa-10147/SCI) → [CAMSF](https://github.com/vishwa-10147/CAMSF) → [CPL](https://github.com/vishwa-10147/CPL).

---

## Features

- **Two replay sources:**
  - KITTI Raw synchronized drives, which provide camera and OXTS GPS/IMU data.
  - KITTI Odometry sequences, from which ground-truth poses are converted into OXTS-style packets.
- **Live pipeline:** each frame goes through metrics, quality and SCI, adaptive weights, adaptive covariance, EKF predict/update, and the reliability model (a Random Forest).
- **Panels:**
  - camera feed;
  - sensor confidence;
  - health gauges;
  - telemetry;
  - fusion diagnostics (weights, innovation, covariance);
  - GPS vs. EKF localization;
  - reliability percentage.
- **Analytics charts:**
  - confidence comparison;
  - adaptive weights;
  - SCI over time;
  - reliability;
  - innovation norm;
  - covariance trace;
  - velocity;
  - raw-GPS vs. adaptive-EKF track.
- **Replay controls:** play, pause, previous and next frame, reset, a speed selector and an FPS slider.
- **CSV export** of the dashboard history.
- **Offline evaluation script:** `evaluation/run_experiments.py` compares the engine with a fixed-weight EKF and writes CSVs, figures and a PDF report.

---

## Pipeline

```
KITTI replay ─► Sensor metrics ─► Quality & SCI ─► Adaptive weights ─► Adaptive covariance
            ─► EKF predict/update ─► Localization features ─► RF reliability model ─► Dashboard
```

| Stage | Module |
|---|---|
| Replay | `modules/replay.py`, `modules/odometry_replay.py` |
| Sensor metrics / quality / SCI | `modules/sensor_metrics.py`, `modules/sensor_quality.py` |
| Adaptive weights / covariance | `modules/adaptive_weights.py`, `modules/adaptive_ekf.py` |
| EKF | `modules/ekf_state.py`, `modules/ekf_predict.py`, `modules/ekf_update.py` |
| Reliability prediction | `modules/localization_features.py`, `modules/localization_model.py`, `models/localization_reliability_rf.joblib` |
| Orchestration | `modules/fusion_engine.py` |
| UI | `app.py`, `components/*` |

---

## Repository structure

```
PLD/
├── app.py                    # Streamlit entry point
├── config.py                 # Paths (env-var overridable) and dashboard settings
├── run_dashboard.bat         # Windows launcher (uses ./venv)
├── requirements.txt
├── components/               # UI panels (camera, sensors, fusion, localization, charts, sidebar, …)
├── modules/                  # Fusion pipeline: replay, metrics, SCI, weights, EKF, reliability model
├── models/
│   └── localization_reliability_rf.joblib
├── evaluation/               # Offline full-drive evaluation + report
├── outputs/statistics/       # Normalization statistics (KITTI Raw, Odometry)
├── utils/loader.py           # Model / artifact loader
├── dashboard/, data/, notebooks/   # Placeholders for the predictive (CPL) extension
├── docs/DOCUMENTATION.md     # Technical documentation
└── test_*.py                 # Stage-by-stage pipeline tests
```

---

## Getting started

```bash
git clone https://github.com/vishwa-10147/PLD.git
cd PLD
python -m venv venv
venv\Scripts\activate            # Linux/macOS: source venv/bin/activate
pip install -r requirements.txt  # for a CUDA build of torch, install it first from pytorch.org
```

### Data

| What | Where (default) | Environment variable |
|---|---|---|
| KITTI Raw (`2011_09_26/<drive>_sync`) | `data/KITTI/` | `KITTI_ROOT` |
| KITTI Odometry (color, poses, calib) | `data/KITTI_Odometry/` | `KITTI_ODOMETRY_ROOT` |
| CAMSF repo (Paper 2 models) | `../CAMSF` | `PAPER2_ROOT` |
| CPL repo (Paper 3 models) | `../CPL` | `PAPER3_ROOT` |

Download KITTI from <https://www.cvlibs.net/datasets/kitti/>. Missing paths only produce a warning at start-up.

### Run

```bash
streamlit run app.py
```

On Windows you can double-click `run_dashboard.bat` instead. Then open <http://localhost:8501>.

### Offline evaluation

```bash
python evaluation/run_experiments.py
```

---

## Papers and documentation

PLD has no paper of its own. It visualizes the methods of the research series:

| Paper | Repository |
|---|---|
| Sensor Confidence Index (SCI): An Intelligent Framework for Dynamic Sensor Quality Assessment in Autonomous Vehicle Perception | [SCI](https://github.com/vishwa-10147/SCI) |
| Confidence-Driven Adaptive Multi-Sensor Fusion Framework for Robust Autonomous Vehicle Localization | [CAMSF](https://github.com/vishwa-10147/CAMSF) |
| Confidence-Guided Predictive Localization and Risk-Aware Decision Framework for Autonomous Vehicles | [CPL](https://github.com/vishwa-10147/CPL) |

The technical documentation for this repository is in [`docs/DOCUMENTATION.md`](docs/DOCUMENTATION.md).

> **Note.** In the dashboard and `evaluation/`, GPS-to-EKF distances are **agreement proxies**, because KITTI Raw OXTS has no independent ground truth here. For accuracy evaluated against independent ground truth, see CAMSF.

---

## Contact

K. Vishwa · [vishwa9821@gmail.com](mailto:vishwa9821@gmail.com)
