import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


COLORS = {
    "camera": "#22c55e",
    "gps": "#3b82f6",
    "imu": "#a855f7",
    "sci": "#f97316",
    "reliability": "#ef4444",
    "innovation": "#e11d48",
    "covariance": "#06b6d4",
    "velocity": "#64748b"
}


def _line_figure(title, x_title, y_title, series):

    figure = go.Figure()

    history = st.session_state.history

    for name, values, color in series:
        figure.add_trace(go.Scatter(
            x=history["frame"],
            y=values,
            mode="lines",
            name=name,
            line={"color": color, "width": 3}
        ))

    figure.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        template="plotly_white",
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.1},
        margin={"l": 20, "r": 20, "t": 65, "b": 45}
    )

    return figure


def initialize_history():

    defaults = {

        "frame": [],
        "timestamp": [],
        "heading": [],

        "camera": [],
        "gps": [],
        "imu": [],
        "sci": [],

        "camera_weight": [],
        "gps_weight": [],
        "imu_weight": [],

        "innovation": [],
        "covariance": [],

        "velocity": [],

        "reliability": [],

        "latitude": [],
        "longitude": [],

        "ekf_latitude": [],
        "ekf_longitude": []
    }

    if "history" not in st.session_state:
        st.session_state.history = defaults
    else:
        for key, value in defaults.items():
            st.session_state.history.setdefault(key, value)

        lengths = {len(values) for values in st.session_state.history.values()}
        if len(lengths) > 1:
            for values in st.session_state.history.values():
                values.clear()


def clear_history() -> None:
    """Clear the rolling visualization history for a new replay sequence."""

    initialize_history()

    for values in st.session_state.history.values():
        values.clear()


def update_history(result):

    initialize_history()

    history = st.session_state.history

    packet = result["packet"]
    quality = result["quality"]
    weights = result["weights"]
    localization = result["localization"]
    covariance = result["covariance"]
    innovation = result["innovation"]
    fusion = result["fusion"]

    history["frame"].append(packet["frame_id"])
    history["timestamp"].append(str(packet["timestamp"]))
    history["heading"].append(float(packet["yaw"]))

    history["camera"].append(quality["camera_score"])
    history["gps"].append(quality["gps_score"])
    history["imu"].append(quality["imu_score"])
    history["sci"].append(quality["sci"])

    history["camera_weight"].append(weights["camera"])
    history["gps_weight"].append(weights["gps"])
    history["imu_weight"].append(weights["imu"])

    history["innovation"].append(float(np.linalg.norm(innovation)))
    history["covariance"].append(float(np.trace(covariance["P"])))

    history["velocity"].append(packet["vf"])

    history["reliability"].append(localization["percentage"])

    history["latitude"].append(packet["latitude"])
    history["longitude"].append(packet["longitude"])

    history["ekf_latitude"].append(fusion["latitude"])
    history["ekf_longitude"].append(fusion["longitude"])

    for key in history:

        history[key] = history[key][-200:]


def confidence_graph():

    st.subheader("Comparison of Camera, GPS and IMU Confidence Scores")

    st.caption(
        "Green = Camera, Blue = GPS, Purple = IMU. This graph compares the "
        "confidence assigned to each sensor during the replay."
    )

    h = st.session_state.history

    st.plotly_chart(
        _line_figure(
            "Overall Sensor Confidence",
            "Replay Frame",
            "Confidence Score",
            [
                ("Camera", h["camera"], COLORS["camera"]),
                ("GPS", h["gps"], COLORS["gps"]),
                ("IMU", h["imu"], COLORS["imu"])
            ]
        ),
        width="stretch"
    )


def weights_graph():

    st.subheader("Comparison of Adaptive Sensor Fusion Weights")

    st.caption("Green = Camera, Blue = GPS, Purple = IMU. Weights adapt to sensor confidence.")

    h = st.session_state.history

    st.plotly_chart(
        _line_figure(
            "Adaptive Sensor Fusion Weights",
            "Replay Frame",
            "Adaptive Sensor Weight",
            [
                ("Camera", h["camera_weight"], COLORS["camera"]),
                ("GPS", h["gps_weight"], COLORS["gps"]),
                ("IMU", h["imu_weight"], COLORS["imu"])
            ]
        ),
        width="stretch"
    )


def sci_graph():

    st.subheader("Sensor Confidence Index Across Replay Frames")

    st.caption("Orange line. X-axis: Replay Frame. Y-axis: SCI.")

    h = st.session_state.history

    st.plotly_chart(
        _line_figure(
            "Sensor Confidence Index",
            "Replay Frame",
            "SCI",
            [("SCI", h["sci"], COLORS["sci"])]
        ),
        width="stretch"
    )


def reliability_graph():

    st.subheader("Localization Reliability Across Replay Frames")

    st.caption("Red line. Random Forest prediction of the localization reliability percentage.")

    h = st.session_state.history

    st.plotly_chart(
        _line_figure(
            "Localization Reliability",
            "Replay Frame",
            "Reliability (%)",
            [("Reliability", h["reliability"], COLORS["reliability"])]
        ),
        width="stretch"
    )


def innovation_graph():

    st.subheader("Innovation Norm Across Replay Frames")

    st.caption("Red = innovation norm, cyan = covariance trace. Both indicate EKF correction and uncertainty.")

    h = st.session_state.history

    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(go.Scatter(
        x=h["frame"], y=h["innovation"], mode="lines", name="Innovation Norm",
        line={"color": COLORS["innovation"], "width": 3}
    ), secondary_y=False)
    figure.add_trace(go.Scatter(
        x=h["frame"], y=h["covariance"], mode="lines", name="Covariance Trace",
        line={"color": COLORS["covariance"], "width": 3}
    ), secondary_y=True)
    figure.update_layout(
        title="Innovation vs Covariance",
        xaxis_title="Replay Frame",
        template="plotly_white",
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.1},
        margin={"l": 20, "r": 20, "t": 65, "b": 45}
    )
    figure.update_yaxes(title_text="Innovation Norm", secondary_y=False)
    figure.update_yaxes(title_text="Covariance Trace", secondary_y=True)
    st.plotly_chart(figure, width="stretch")


def covariance_graph():

    st.subheader("Covariance Trace Across Replay Frames")

    st.caption("Cyan line. The covariance trace represents localization uncertainty.")

    h = st.session_state.history

    st.plotly_chart(
        _line_figure(
            "Covariance Trace",
            "Replay Frame",
            "Covariance Trace",
            [("Covariance Trace", h["covariance"], COLORS["covariance"])]
        ),
        width="stretch"
    )


def velocity_graph():

    st.subheader("Vehicle Velocity Across Replay Frames")

    st.caption("Grey line. Forward velocity measured during the KITTI replay.")

    h = st.session_state.history

    st.plotly_chart(
        _line_figure(
            "Vehicle Velocity",
            "Replay Frame",
            "Velocity (m/s)",
            [("Velocity", h["velocity"], COLORS["velocity"])]
        ),
        width="stretch"
    )


def trajectory_graph():

    st.subheader("Comparison of Raw GPS and Adaptive EKF Localization")

    st.caption("Blue = Original GPS Path. Orange = Adaptive EKF Predicted Trajectory.")

    h = st.session_state.history

    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=h["longitude"], y=h["latitude"], mode="lines+markers", name="Original GPS",
        line={"color": COLORS["gps"], "width": 3},
        marker={"color": COLORS["gps"], "size": 6, "symbol": "circle"},
        customdata=list(zip(h["frame"], h["timestamp"])),
        hovertemplate=(
            "Original GPS<br>Latitude: %{y:.6f}<br>Longitude: %{x:.6f}"
            "<br>Frame: %{customdata[0]}<br>Timestamp: %{customdata[1]}<extra></extra>"
        )
    ))
    figure.add_trace(go.Scatter(
        x=h["ekf_longitude"], y=h["ekf_latitude"], mode="lines+markers", name="Adaptive EKF",
        line={"color": COLORS["sci"], "width": 3},
        marker={"color": COLORS["sci"], "size": 7, "symbol": "diamond"},
        customdata=list(zip(h["frame"], h["timestamp"])),
        hovertemplate=(
            "Adaptive EKF<br>Latitude: %{y:.6f}<br>Longitude: %{x:.6f}"
            "<br>Frame: %{customdata[0]}<br>Timestamp: %{customdata[1]}<extra></extra>"
        )
    ))
    if h["frame"]:
        figure.add_trace(go.Scatter(
            x=[h["longitude"][0]], y=[h["latitude"][0]], mode="markers",
            name="Start", marker={"color": COLORS["gps"], "size": 11, "symbol": "circle"}
        ))
        figure.add_trace(go.Scatter(
            x=[h["ekf_longitude"][-1]], y=[h["ekf_latitude"][-1]],
            mode="markers", name="Current Vehicle",
            marker={"color": "#111111", "size": 16, "symbol": "triangle-up", "angle": -np.degrees(h["heading"][-1])},
            customdata=[[h["frame"][-1], h["timestamp"][-1]]],
            hovertemplate=(
                "Current vehicle<br>Latitude: %{y:.6f}<br>Longitude: %{x:.6f}"
                "<br>Frame: %{customdata[0]}<br>Timestamp: %{customdata[1]}<extra></extra>"
            )
        ))
        figure.add_trace(go.Scatter(
            x=[h["ekf_longitude"][-1]], y=[h["ekf_latitude"][-1]],
            mode="markers", name="End",
            marker={"color": COLORS["reliability"], "size": 10, "symbol": "x"}
        ))
    figure.update_layout(
        title="GPS vs Adaptive EKF Trajectory",
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        template="plotly_white",
        hovermode="closest",
        legend={"orientation": "h", "y": 1.1},
        margin={"l": 20, "r": 20, "t": 65, "b": 45}
    )
    st.plotly_chart(figure, width="stretch")
    st.caption("🔵 Original GPS Trajectory  |  🟠 Adaptive EKF Predicted Trajectory")


def render_all_charts(placeholders=None):

    initialize_history()

    chart_functions = [
        confidence_graph,
        weights_graph,
        sci_graph,
        reliability_graph,
        innovation_graph,
        covariance_graph,
        velocity_graph,
        trajectory_graph
    ]

    if placeholders is None:
        placeholders = [st.empty() for _ in chart_functions]

    for placeholder, chart_function in zip(placeholders, chart_functions):
        with placeholder.container():
            chart_function()
            st.divider()
