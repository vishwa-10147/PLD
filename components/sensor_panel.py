"""Sensor confidence summary panel."""

from typing import Any

import streamlit as st

from components.utils import status_icon


def render_sensor_panel(result: dict[str, Any]) -> None:
    """Render camera, GPS, IMU, and SCI metric cards."""

    quality = result["quality"]

    st.subheader("Sensor Confidence Assessment")
    st.caption("Camera, GPS, and IMU confidence scores are combined into the Sensor Confidence Index (SCI).")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🟢 Camera Confidence", f"{quality['camera_score']:.3f}")
    c2.metric("🔵 GPS Confidence", f"{quality['gps_score']:.3f}")
    c3.metric("🟣 IMU Confidence", f"{quality['imu_score']:.3f}")
    c4.metric("🟠 SCI", f"{quality['sci']:.3f}")
    c5.metric(
        "Sensor Status",
        f"{status_icon(quality['status'])} {quality['status']}"
    )

    st.progress(float(quality["sci"]))
    st.caption("Overall Sensor Health: 0 indicates low confidence; 1 indicates high confidence.")
