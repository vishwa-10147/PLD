"""Replay control widgets for the dashboard."""

from typing import Any

import streamlit as st


def render_controls() -> dict[str, Any]:
    """Render replay controls and return the selected actions."""

    c1, c2, c3, c4, c5 = st.columns(5)

    play = c1.button("▶ Play", use_container_width=True)
    pause = c2.button("⏸ Pause", use_container_width=True)
    previous = c3.button("⏮ Previous", use_container_width=True)
    next_frame = c4.button("Next Frame", use_container_width=True)
    reset = c5.button("↺ Reset", use_container_width=True)

    speed = st.select_slider(
        "Replay Speed",
        options=[0.5, 1.0, 2.0, 5.0],
        value=1.0,
        format_func=lambda value: f"{value:g}×"
    )

    return {
        "play": play,
        "pause": pause,
        "previous": previous,
        "reset": reset,
        "next": next_frame,
        "speed": speed
    }
