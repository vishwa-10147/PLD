"""Reproducible full-drive evaluation and publication-output pipeline.

This script does not modify dashboard or backend behavior.  It evaluates the
existing proposed fusion engine alongside an evaluation-only fixed-weight EKF
baseline.  KITTI Raw OXTS has no independent ground-truth track in this
project, so GPS-to-filter distances are explicitly reported as agreement
proxies, not localization error.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.lib import colors

from modules.adaptive_ekf import AdaptiveEKF
from modules.ekf_predict import EKFPredict
from modules.ekf_state import EKFState
from modules.ekf_update import EKFUpdate
from modules.fusion_engine import FusionEngine
from modules.replay import KITTIReplay


COLORS = {"camera": "#22c55e", "gps": "#3b82f6", "imu": "#a855f7", "sci": "#f97316", "reliability": "#ef4444", "ekf": "#f97316"}


def distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Approximate geodetic separation in meters for a local KITTI drive."""
    return 111_320 * np.hypot(lat1 - lat2, (lon1 - lon2) * np.cos(np.deg2rad(lat1)))


def summary(values: pd.Series) -> dict[str, float]:
    """Return publication-ready descriptive statistics."""
    values = values.dropna()
    return {"mean": values.mean(), "median": values.median(), "std": values.std(), "min": values.min(), "max": values.max(), "mae": values.abs().mean(), "rmse": np.sqrt(np.mean(values ** 2))}


class FixedWeightBaseline:
    """Evaluation-only conventional EKF using constant equal sensor weights."""

    def __init__(self) -> None:
        self.state = EKFState()
        self.predictor = EKFPredict(dt=0.1)
        self.updater = EKFUpdate()
        self.covariance = AdaptiveEKF()
        self.initialized = False

    def process(self, packet: dict) -> dict[str, float]:
        if not self.initialized:
            self.state.initialize(packet)
            self.initialized = True
        fixed = {"camera": 1 / 3, "gps": 1 / 3, "imu": 1 / 3}
        covariance = self.covariance.compute(fixed)
        self.predictor.predict(self.state)
        result = self.updater.update(self.state, packet, covariance["R"])
        state = result["state"]
        return {"latitude": float(state[1, 0]), "longitude": float(state[0, 0])}


def save_figure(path: Path, title: str, x: pd.Series, series: dict[str, tuple[pd.Series, str]], ylabel: str) -> None:
    """Save one consistent 300-DPI publication figure."""
    figure, axis = plt.subplots(figsize=(9, 4.5))
    for label, (values, color) in series.items():
        axis.plot(x, values, label=label, color=color, linewidth=1.8)
    axis.set(title=title, xlabel="Replay Frame", ylabel=ylabel)
    axis.grid(alpha=.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def write_report(path: Path, stats: pd.DataFrame, figures: list[Path], frames: int) -> None:
    """Create a compact PDF results report from real exported artifacts."""
    document = SimpleDocTemplate(str(path), pagesize=A4)
    styles = getSampleStyleSheet()
    content = [Paragraph("Paper 2 Evaluation Summary", styles["Title"]), Spacer(1, 10),
               Paragraph(f"Full KITTI replay evaluated: {frames} frames.", styles["BodyText"]),
               Paragraph("GPS-to-filter quantities are agreement proxies because no independent ground-truth trajectory is supplied.", styles["BodyText"]), Spacer(1, 10)]
    rows = [["Metric", "Value"]] + [[name, f"{value:.4f}"] for name, value in stats.iloc[:, 0].items()]
    table = Table(rows, colWidths=[250, 150])
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), .25, colors.grey)]))
    content.extend([table, Spacer(1, 14)])
    for figure in figures[:4]:
        content.extend([Image(str(figure), width=500, height=250), Spacer(1, 8)])
    document.build(content)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive", default="2011_09_26_drive_0022_sync")
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    output, figures = args.output, args.output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    replay = KITTIReplay(args.drive)
    replay.load()
    proposed, baseline = FusionEngine(), FixedWeightBaseline()
    process = psutil.Process()
    rows: list[dict] = []
    started = time.perf_counter()
    for _ in range(len(replay)):
        packet = replay.next_frame()
        frame_started = time.perf_counter()
        result = proposed.process(packet)
        fixed = baseline.process(packet)
        latency = time.perf_counter() - frame_started
        fusion = result["fusion"]
        rows.append({
            "frame": packet["frame_id"], "timestamp": str(packet["timestamp"]), "latitude": packet["latitude"], "longitude": packet["longitude"], "velocity": packet["vf"],
            "camera_score": result["quality"]["camera_score"], "gps_score": result["quality"]["gps_score"], "imu_score": result["quality"]["imu_score"], "sci": result["quality"]["sci"],
            "camera_weight": result["weights"]["camera"], "gps_weight": result["weights"]["gps"], "imu_weight": result["weights"]["imu"],
            "ekf_latitude": fusion["latitude"], "ekf_longitude": fusion["longitude"], "baseline_latitude": fixed["latitude"], "baseline_longitude": fixed["longitude"],
            "reliability": result["localization"]["percentage"], "innovation_norm": float(np.linalg.norm(result["innovation"])), "covariance_trace": float(np.trace(result["covariance"]["P"])),
            "kalman_gain_magnitude": float(np.linalg.norm(result["kalman_gain"])), "latency_ms": latency * 1000, "memory_mb": process.memory_info().rss / 1e6,
        })
    frame_data = pd.DataFrame(rows)
    frame_data["adaptive_gps_deviation_m"] = [distance_m(*values) for values in zip(frame_data.latitude, frame_data.longitude, frame_data.ekf_latitude, frame_data.ekf_longitude)]
    frame_data["baseline_gps_deviation_m"] = [distance_m(*values) for values in zip(frame_data.latitude, frame_data.longitude, frame_data.baseline_latitude, frame_data.baseline_longitude)]
    elapsed = time.perf_counter() - started

    frame_data[["frame", "camera_score", "gps_score", "imu_score", "sci"]].to_csv(output / "sensor_metrics.csv", index=False)
    frame_data[["frame", "camera_weight", "gps_weight", "imu_weight"]].to_csv(output / "adaptive_weights.csv", index=False)
    frame_data[["frame", "reliability"]].to_csv(output / "reliability_results.csv", index=False)
    frame_data[["frame", "innovation_norm", "covariance_trace", "kalman_gain_magnitude"]].to_csv(output / "ekf_metrics.csv", index=False)
    frame_data[["frame", "latitude", "longitude", "ekf_latitude", "ekf_longitude"]].to_csv(output / "trajectory_results.csv", index=False)
    frame_data.to_csv(output / "localization_results.csv", index=False)
    replay_stats = pd.DataFrame([{"frames": len(frame_data), "total_seconds": elapsed, "average_fps": len(frame_data) / elapsed, "mean_latency_ms": frame_data.latency_ms.mean(), "max_latency_ms": frame_data.latency_ms.max(), "min_latency_ms": frame_data.latency_ms.min(), "mean_memory_mb": frame_data.memory_mb.mean()}])
    replay_stats.to_csv(output / "replay_statistics.csv", index=False)

    sci_stats = pd.DataFrame([summary(frame_data.sci)]).T.rename(columns={0: "SCI"})
    sci_stats.to_csv(output / "sci_statistics.csv")
    comparison = pd.DataFrame({"Standard EKF GPS-deviation proxy": summary(frame_data.baseline_gps_deviation_m), "Proposed Adaptive EKF GPS-deviation proxy": summary(frame_data.adaptive_gps_deviation_m)})
    comparison.to_csv(output / "baseline_comparison.csv")
    for table, name in [(sci_stats, "sci_statistics"), (comparison, "baseline_comparison"), (replay_stats.T, "replay_statistics")]:
        (output / f"{name}.tex").write_text(table.to_latex(), encoding="utf-8")

    save_figure(figures / "confidence.png", "Sensor Confidence Comparison", frame_data.frame, {"Camera": (frame_data.camera_score, COLORS["camera"]), "GPS": (frame_data.gps_score, COLORS["gps"]), "IMU": (frame_data.imu_score, COLORS["imu"])}, "Confidence")
    save_figure(figures / "weights.png", "Adaptive Sensor Weight Evolution", frame_data.frame, {"Camera": (frame_data.camera_weight, COLORS["camera"]), "GPS": (frame_data.gps_weight, COLORS["gps"]), "IMU": (frame_data.imu_weight, COLORS["imu"])}, "Weight")
    save_figure(figures / "sci.png", "Sensor Confidence Index Trend", frame_data.frame, {"SCI": (frame_data.sci, COLORS["sci"])}, "SCI")
    save_figure(figures / "reliability.png", "Localization Reliability Trend", frame_data.frame, {"Reliability": (frame_data.reliability, COLORS["reliability"])}, "Reliability (%)")
    save_figure(figures / "innovation.png", "Innovation Norm", frame_data.frame, {"Innovation": (frame_data.innovation_norm, COLORS["reliability"])}, "Norm")
    save_figure(figures / "covariance.png", "Covariance Trace", frame_data.frame, {"Covariance": (frame_data.covariance_trace, "#06b6d4")}, "Trace")
    save_figure(figures / "velocity.png", "Vehicle Velocity", frame_data.frame, {"Velocity": (frame_data.velocity, "#64748b")}, "Velocity (m/s)")
    figure, axis = plt.subplots(figsize=(7, 6)); axis.plot(frame_data.longitude, frame_data.latitude, color=COLORS["gps"], label="Original GPS"); axis.plot(frame_data.ekf_longitude, frame_data.ekf_latitude, color=COLORS["ekf"], label="Adaptive EKF"); axis.legend(); axis.set(xlabel="Longitude", ylabel="Latitude", title="GPS vs Adaptive EKF Trajectory"); axis.grid(alpha=.25); figure.tight_layout(); figure.savefig(figures / "trajectory.png", dpi=300); plt.close(figure)
    save_figure(figures / "localization_error_comparison.png", frame_data.frame, {"Standard EKF": (frame_data.baseline_gps_deviation_m, COLORS["gps"]), "Adaptive EKF": (frame_data.adaptive_gps_deviation_m, COLORS["ekf"])}, "GPS-Filter Deviation Proxy (m)")
    write_report(output / "summary_report.pdf", comparison, [figures / name for name in ["confidence.png", "weights.png", "trajectory.png", "reliability.png"]], len(frame_data))
    (output / "reproducibility.json").write_text(json.dumps({"drive": args.drive, "python": sys.version, "platform": platform.platform(), "seed": "not applicable (deterministic replay)", "command": "python evaluation/run_experiments.py"}, indent=2), encoding="utf-8")
    print(f"Evaluation complete: {output.resolve()}")


if __name__ == "__main__":
    main()
