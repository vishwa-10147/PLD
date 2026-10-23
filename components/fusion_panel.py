"""Adaptive fusion and Kalman-filter diagnostics panel."""

from typing import Any

import numpy as np
import plotly.graph_objects as go
import streamlit as st


def _heatmap(matrix: Any, title: str) -> go.Figure:
    """Build a compact, labeled matrix heatmap for EKF diagnostics."""

    figure = go.Figure(go.Heatmap(
        z=np.asarray(matrix),
        colorscale="Blues",
        colorbar={"title": "Value"},
        hovertemplate="Row %{y}<br>Column %{x}<br>Value %{z:.4f}<extra></extra>"
    ))
    figure.update_layout(
        title=title,
        template="plotly_white",
        height=300,
        margin={"l": 20, "r": 20, "t": 50, "b": 20}
    )
    return figure


def render_fusion_panel(result: dict[str, Any]) -> None:
    """Render adaptive weights, EKF uncertainty, and matrix diagnostics."""

    weights = result["weights"]
    covariance = result["covariance"]
    innovation = result["innovation"]

    trace = float(np.trace(covariance["P"]))
    innovation_norm = float(np.linalg.norm(innovation))

    st.subheader("Adaptive Sensor Fusion")
    st.caption("Green = Camera, Blue = GPS, Purple = IMU. Weights adapt to current sensor confidence.")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Camera Weight", f"{weights['camera']:.3f}")
    c2.metric("GPS Weight", f"{weights['gps']:.3f}")
    c3.metric("IMU Weight", f"{weights['imu']:.3f}")
    c4.metric("Innovation Norm", f"{innovation_norm:.4f}")
    c5.metric("Covariance Trace", f"{trace:.4f}")

    with st.expander("Kalman Filter Diagnostics", expanded=False):
        left, right = st.columns(2)
        left.plotly_chart(
            _heatmap(covariance["P"], "State Covariance Matrix (P)"),
            use_container_width=True
        )
        right.plotly_chart(
            _heatmap(result["kalman_gain"], "Kalman Gain Matrix (K)"),
            use_container_width=True
        )
        st.caption("Innovation residual")
        st.dataframe(np.asarray(innovation), use_container_width=True)
