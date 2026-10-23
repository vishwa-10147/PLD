"""Current fused-localization and reliability panel."""

from typing import Any

import streamlit as st

from components.utils import status_icon


def render_localization_panel(result: dict[str, Any]) -> None:
    """Render raw vehicle state, fused state, and model reliability."""

    packet = result["packet"]
    fusion = result["fusion"]
    localization = result["localization"]

    st.subheader("Vehicle Localization")
    st.caption("Adaptive EKF fused state with Random Forest localization reliability.")

    raw, fused, reliability = st.columns(3)
    with raw:
        st.metric("GPS Latitude", f"{packet['latitude']:.6f}")
        st.metric("GPS Longitude", f"{packet['longitude']:.6f}")
        st.metric("Heading", f"{packet['yaw']:.3f} rad")
    with fused:
        st.metric("EKF Latitude", f"{fusion['latitude']:.6f}")
        st.metric("EKF Longitude", f"{fusion['longitude']:.6f}")
        st.metric("Velocity", f"{fusion['velocity']:.3f} m/s")
    with reliability:
        st.metric("Localization Reliability", f"{localization['percentage']:.1f}%")
        st.metric(
            "Reliability Status",
            f"{status_icon(localization['status'])} {localization['status']}"
        )
        st.progress(float(localization["score"]))
