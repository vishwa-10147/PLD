# PLD — Technical Documentation

The **Predictive Localization Dashboard (PLD)** is a Streamlit application. It replays KITTI drives frame by frame and runs the **confidence-driven adaptive sensor-fusion pipeline** on each frame, which is the method of Paper 2, [CAMSF](https://github.com/vishwa-10147/CAMSF). It then visualizes every stage live: sensor metrics, sensor confidence, the Sensor Confidence Index (SCI), adaptive weights, the adaptive EKF and localization reliability.

---

## 1. Processing pipeline (per frame)

`modules/fusion_engine.py` → `FusionEngine.process(packet)`:

```
KITTI replay packet (image + OXTS)
   │  modules/replay.py           (KITTI Raw)      or
   │  modules/odometry_replay.py  (KITTI Odometry: poses → synthetic OXTS-style packet)
   ▼
Sensor metrics            modules/sensor_metrics.py
   ▼
Sensor quality / SCI      modules/sensor_quality.py   (camera, GPS, IMU confidence + SCI, status)
   ▼
Adaptive weights          modules/adaptive_weights.py
   ▼
Adaptive covariance       modules/adaptive_ekf.py
   ▼
EKF predict / update      modules/ekf_state.py, ekf_predict.py, ekf_update.py
   ▼
Localization features     modules/localization_features.py
   ▼
Reliability prediction    modules/localization_model.py  (models/localization_reliability_rf.joblib)
   ▼
Localization output       modules/localization.py  →  result dict rendered by components/*
```

`modules/statistics.py` loads per-dataset statistics from `outputs/statistics/kitti_stats.json` and `outputs/statistics/odometry_stats.json`. These are used to normalize the metrics.

---

## 2. User interface (`app.py`, `components/`)

| Area | Component | Shows |
|---|---|---|
| Sidebar | `sidebar.py` | Dataset (KITTI Raw or Odometry), drive/sequence, replay FPS, replay status, live SCI / health |
| Controls | `replay_controls.py` | Play, pause, previous / next frame, reset, replay speed |
| Status ribbon | `research_panels.py` | Frame, speed, overall sensor status |
| Camera | `camera_panel.py` | Live camera frame |
| Sensors | `sensor_panel.py` | Camera / GPS / IMU confidence |
| Health gauges | `research_panels.py` | Per-sensor health gauges |
| Telemetry | `research_panels.py` | Velocity, acceleration, yaw |
| Fusion | `fusion_panel.py` | Adaptive weights, innovation, covariance trace |
| Localization | `localization_panel.py` | GPS vs. EKF latitude/longitude, heading, velocity, reliability % |
| Analytics | `charts_panel.py` | Confidence comparison, weights, SCI, reliability, innovation norm, covariance trace, velocity, GPS vs. adaptive-EKF track |
| Research / export | `research_panels.py` | SCI history, GPS–EKF difference, **Export Dashboard CSV** |

---

## 3. Configuration (`config.py`)

Every external path can be overridden with an environment variable. The defaults are relative to the repository:

| Setting | Environment variable | Default |
|---|---|---|
| KITTI Raw root | `KITTI_ROOT` | `data/KITTI` (expects `2011_09_26/<drive>_sync/`) |
| KITTI Odometry root | `KITTI_ODOMETRY_ROOT` | `data/KITTI_Odometry` (`data_odometry_color`, `data_odometry_poses`, `data_odometry_calib`) |
| Notebooks | `NOTEBOOK_ROOT` | `notebooks/` |
| Paper 2 (CAMSF) | `PAPER2_ROOT` | `../CAMSF`, a sibling clone of the CAMSF repository |
| Paper 3 (CPL) | `PAPER3_ROOT` | `../CPL`, a sibling clone of the CPL repository |

Other settings:

- `DEFAULT_DRIVE = 2011_09_26_drive_0022_sync`
- `FPS = 10`
- `DEFAULT_SEQUENCE_LENGTH = 20`
- `PREDICTION_HORIZON = 30`
- colour scheme and panel toggles

On start-up, `config.py` prints a warning for every configured path that does not exist. The dashboard still runs with whatever is available.

---

## 4. Evaluation (`evaluation/run_experiments.py`)

This script evaluates a full drive offline. It compares the proposed fusion engine with an evaluation-only **fixed-weight EKF baseline** and writes CSVs, figures and a PDF report (made with ReportLab) to `results/`:

```bash
python evaluation/run_experiments.py
```

KITTI Raw OXTS provides no independent ground-truth track in this setup. GPS-to-filter distances are therefore reported as **agreement proxies**, not as localization error. For an evaluation against independent ground truth, see CAMSF.

---

## 5. Tests

Each `test_*.py` file at the repository root runs one pipeline stage on real replay data:

| Test | Stage |
|---|---|
| `test_replay.py` | Replay loader |
| `test_metrics.py` | Sensor metrics |
| `test_confidence.py` | Confidence / SCI |
| `test_weights.py` | Adaptive weights |
| `test_state.py` | EKF state |
| `test_predict.py` | EKF predict |
| `test_update.py` | EKF update |
| `test_ekf.py` | Adaptive EKF |
| `test_fusion.py` | Fusion engine |
| `test_features.py` | Localization features |
| `test_localization_model.py` | Reliability model |
| `test_statistics.py` | Dataset statistics |
| `test_complete_pipeline.py` | End-to-end pipeline |

They need the KITTI data. Run them as scripts, for example `python test_complete_pipeline.py`.

---

## 6. Planned modules

These files exist as empty placeholders for the predictive (Paper 3 / CPL) extension of the dashboard:

- `dashboard/*`
- `modules/predictor.py`, `decision_engine.py`, `scenario_manager.py`, `synchronization.py`, `visualization.py`, `confidence.py`
- `utils/constants.py`, `helpers.py`, `logger.py`, `metrics.py`

`config.py` already points to the CPL checkpoints: `best_lstm.pt`, `best_gru.pt`, `best_transformer.pt`, `scaler.pkl` and `dataloader_info.pt`.

---

## 7. Research series

| Project | Repository | Used by PLD |
|---|---|---|
| SCI | `vishwa-10147/SCI` | The confidence-index concept |
| CAMSF | `vishwa-10147/CAMSF` | The fusion pipeline the dashboard replays, plus its trained models |
| CPL | `vishwa-10147/CPL` | Predictive models (planned integration) |
