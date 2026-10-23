"""Sidebar settings and live replay summary."""

from typing import Any

import streamlit as st

from utils.loader import get_available_drives, get_available_odometry_sequences


def render_sidebar(
    replay: Any | None,
    result: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Render drive/sequence settings plus current replay and health information."""

    st.sidebar.title("Dashboard Settings")

    # ------------------------------------------------------------------
    # Dataset selector
    # ------------------------------------------------------------------
    dataset = st.sidebar.radio(
        "Dataset",
        ["KITTI Raw", "KITTI Odometry"],
        index=0,
        help=(
            "KITTI Raw: GPS/IMU-equipped drives (pykitti). "
            "KITTI Odometry: visual odometry benchmark with ground-truth poses."
        ),
    )

    # ------------------------------------------------------------------
    # Drive / Sequence selector
    # ------------------------------------------------------------------
    if dataset == "KITTI Raw":
        try:
            drives = get_available_drives()
        except FileNotFoundError as exc:
            st.sidebar.error(str(exc))
            drives = []
        selection = st.sidebar.selectbox("KITTI Drive", drives) if drives else None
    else:
        try:
            sequences = get_available_odometry_sequences()
        except FileNotFoundError as exc:
            st.sidebar.error(str(exc))
            sequences = []
        selection = st.sidebar.selectbox("Odometry Sequence", sequences) if sequences else None

    fps = st.sidebar.slider("Replay FPS", 1, 30, 10)

    # ------------------------------------------------------------------
    # Live replay status
    # ------------------------------------------------------------------
    if replay is not None:
        st.sidebar.divider()
        st.sidebar.subheader("Replay Status")
        st.sidebar.metric("Current Frame", f"{replay.current_index()} / {len(replay)}")
        st.sidebar.metric("Total Frames", len(replay))
        st.sidebar.progress(replay.progress())

    if result:
        quality = result["quality"]
        localization = result["localization"]
        st.sidebar.divider()
        st.sidebar.subheader("Live Health")
        st.sidebar.metric("Sensor Status", quality["status"])
        st.sidebar.metric("Current SCI", f"{quality['sci']:.3f}")
        st.sidebar.metric(
            "Localization Reliability",
            f"{localization['percentage']:.1f}%"
        )

    return {"dataset": dataset, "selection": selection, "fps": fps}
