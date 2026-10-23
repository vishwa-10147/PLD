"""Research-focused dashboard panels derived from existing fusion output."""

from __future__ import annotations

from io import StringIO
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.utils import status_icon


def _score_status(score: float) -> str:
    if score >= 0.85:
        return "EXCELLENT"
    if score >= 0.70:
        return "GOOD"
    if score >= 0.50:
        return "FAIR"
    return "POOR"


def render_status_ribbon(result: dict[str, Any], replay: Any, speed: float) -> None:
    """Show the live, color-coded status ribbon."""
    quality = result["quality"]
    reliability = result["localization"]
    cards = st.columns(7)
    values = [
        ("Frame", f"{result['packet']['frame_id']} / {len(replay)}"),
        ("Replay Speed", f"{speed:g}×"),
        ("Camera", f"{status_icon(_score_status(quality['camera_score']))} {_score_status(quality['camera_score'])}"),
        ("GPS", f"{status_icon(_score_status(quality['gps_score']))} {_score_status(quality['gps_score'])}"),
        ("IMU", f"{status_icon(_score_status(quality['imu_score']))} {_score_status(quality['imu_score'])}"),
        ("SCI", f"{quality['sci']:.2f}"),
        ("Reliability", f"{reliability['percentage']:.1f}%"),
    ]
    for card, (label, value) in zip(cards, values):
        card.metric(label, value)


def render_telemetry_panel(result: dict[str, Any]) -> None:
    """Render the current vehicle and GNSS telemetry values."""
    packet = result["packet"]
    st.subheader("Vehicle Telemetry")
    st.caption("Current synchronized KITTI OXTS, vehicle motion, and GNSS quality data.")
    columns = st.columns(4)
    fields = [
        ("Latitude", f"{packet['latitude']:.6f}"), ("Longitude", f"{packet['longitude']:.6f}"),
        ("Altitude", f"{packet['altitude']:.2f} m"), ("Velocity", f"{packet['vf']:.3f} m/s"),
        ("Acceleration", f"{packet['ax']:.3f} m/s²"), ("Yaw", f"{packet['yaw']:.3f} rad"),
        ("Pitch", f"{packet['pitch']:.3f} rad"), ("Roll", f"{packet['roll']:.3f} rad"),
        ("GPS Satellites", packet['numsats']), ("Position Accuracy", f"{packet['pos_accuracy']:.3f} m"),
        ("Velocity Accuracy", f"{packet['vel_accuracy']:.3f} m/s"), ("Navigation Status", packet['navstat']),
    ]
    for column, (label, value) in zip(columns * 3, fields):
        column.metric(label, value)


def render_health_gauges(result: dict[str, Any]) -> None:
    """Render circular Plotly gauges for the sensor health scores."""
    quality = result["quality"]
    values = [("Camera", quality["camera_score"], "#22c55e"), ("GPS", quality["gps_score"], "#3b82f6"),
              ("IMU", quality["imu_score"], "#a855f7"), ("SCI", quality["sci"], "#f97316")]
    st.subheader("Sensor Health Gauges")
    columns = st.columns(4)
    for column, (name, value, color) in zip(columns, values):
        figure = go.Figure(go.Indicator(
            mode="gauge+number", value=value,
            number={"valueformat": ".3f"}, title={"text": f"{name}<br>{_score_status(value)}"},
            gauge={"axis": {"range": [0, 1]}, "bar": {"color": color},
                   "steps": [{"range": [0, .5], "color": "#fee2e2"}, {"range": [.5, .85], "color": "#fef3c7"}, {"range": [.85, 1], "color": "#dcfce7"}]}
        ))
        figure.update_layout(height=220, margin={"l": 10, "r": 10, "t": 45, "b": 5})
        column.plotly_chart(figure, width="stretch")


def render_sci_and_difference() -> None:
    """Render SCI summary and transparent GPS-to-EKF disagreement statistics."""
    h = st.session_state.history
    if not h["sci"]:
        return
    sci = np.asarray(h["sci"])
    lat = np.asarray(h["latitude"])
    lon = np.asarray(h["longitude"])
    ekf_lat = np.asarray(h["ekf_latitude"])
    ekf_lon = np.asarray(h["ekf_longitude"])
    difference = 111_320 * np.sqrt((lat - ekf_lat) ** 2 + ((lon - ekf_lon) * np.cos(np.deg2rad(lat))) ** 2)
    st.subheader("SCI History and GPS–EKF Difference")
    st.caption("The distance statistics are GPS–EKF disagreement proxies, not ground-truth localization errors.")
    left, right = st.columns(2)
    with left:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current SCI", f"{sci[-1]:.3f}")
        c2.metric("Maximum SCI", f"{sci.max():.3f}")
        c3.metric("Minimum SCI", f"{sci.min():.3f}")
        c4.metric("Average SCI", f"{sci.mean():.3f}")
    with right:
        c1, c2, c3 = st.columns(3)
        c1.metric("Mean Difference", f"{difference.mean():.2f} m")
        c2.metric("RMSE Proxy", f"{np.sqrt(np.mean(difference ** 2)):.2f} m")
        c3.metric("Maximum Difference", f"{difference.max():.2f} m")


def render_research_and_exports() -> None:
    """Render research provenance and CSV exports from visualization history."""
    with st.expander("Research Information", expanded=False):
        st.markdown("""
        **Dataset:** KITTI Raw Dataset  
        **Fusion Method:** Confidence-Driven Adaptive Multi-Sensor Fusion  
        **Localization:** Adaptive EKF  
        **Reliability Model:** Random Forest  
        **Paper:** Paper 2
        """)
    history = pd.DataFrame(st.session_state.history)
    st.download_button("Export Dashboard CSV", history.to_csv(index=False), "dashboard_history.csv", "text/csv")
