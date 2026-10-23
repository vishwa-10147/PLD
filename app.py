"""Research dashboard for confidence-driven adaptive vehicle localization."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import streamlit as st

from components.camera_panel import render_camera
from components.charts_panel import (
    clear_history,
    initialize_history,
    render_all_charts,
    update_history,
)
from components.fusion_panel import render_fusion_panel
from components.localization_panel import render_localization_panel
from components.replay_controls import render_controls
from components.research_panels import (
    render_health_gauges,
    render_research_and_exports,
    render_sci_and_difference,
    render_status_ribbon,
    render_telemetry_panel,
)
from components.sensor_panel import render_sensor_panel
from components.sidebar import render_sidebar
from modules.fusion_engine import FusionEngine
from modules.odometry_replay import KITTIOdometryReplay
from modules.replay import KITTIReplay
from modules.statistics import DatasetStatistics


st.set_page_config(
    page_title="Adaptive Multi-Sensor Fusion Dashboard",
    page_icon="🚘",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Stats-file paths
# ---------------------------------------------------------------------------

_KITTI_RAW_STATS = Path("outputs/statistics/kitti_stats.json")
_KITTI_ODO_STATS = Path("outputs/statistics/odometry_stats.json")


def _ensure_odometry_stats(sequence: str) -> None:
    """Generate odometry_stats.json the first time it is needed.

    Samples up to 500 evenly-spaced frames from the given sequence so
    the statistics computation finishes quickly (a full sequence like 00
    has 4 541 frames — scanning all of them at startup would be slow).
    """

    if _KITTI_ODO_STATS.exists():
        return

    with st.spinner(
        f"📊 First-time calibration: computing Odometry statistics "
        f"from sequence {sequence}… (runs once)"
    ):
        # Build a lightweight replay that only loads N evenly-spaced frames
        src = KITTIOdometryReplay(sequence)
        src.load()

        step = max(1, src.total_frames // 500)
        sampled = KITTIOdometryReplay.__new__(KITTIOdometryReplay)
        sampled.__dict__.update(src.__dict__)
        sampled._images = src._images[::step]
        sampled._poses = src._poses[::step]
        sampled._timestamps = src._timestamps[::step]
        sampled.total_frames = len(sampled._images)
        sampled.current_frame = 0
        sampled._prev_xyz = None
        sampled._prev_vel = None
        sampled._prev_euler = None

        ds = DatasetStatistics(replay=sampled)
        ds.save(str(_KITTI_ODO_STATS))
        st.success("✅ Odometry statistics calibration complete.")


def _stats_file_for(dataset: str) -> Path | None:
    """Return the stats file path for the active dataset, or None for Raw default."""
    if dataset == "KITTI Odometry":
        return _KITTI_ODO_STATS
    return None  # FusionEngine default → kitti_stats.json



def _new_pipeline(dataset: str, selection: str) -> None:
    """Create dashboard-owned replay and fusion instances for a drive or sequence."""

    if dataset == "KITTI Odometry":
        replay: KITTIReplay | KITTIOdometryReplay = KITTIOdometryReplay(selection)
        _ensure_odometry_stats(selection)
    else:
        replay = KITTIReplay(selection)

    replay.load()
    st.session_state.replay = replay
    st.session_state.fusion = FusionEngine(stats_file=_stats_file_for(dataset))
    st.session_state.dataset = dataset
    st.session_state.selection = selection
    st.session_state.last_result = None
    clear_history()



def _process_next_frame() -> bool:
    """Advance one frame and update dashboard-only state and history."""

    replay: KITTIReplay = st.session_state.replay
    if not replay.has_next():
        st.session_state.playing = False
        return False

    packet = replay.next_frame()
    result = st.session_state.fusion.process(packet)
    st.session_state.last_result = result
    update_history(result)
    return True


def _rebuild_to_frame(target: int) -> None:
    """Rebuild fusion state and visual history through a requested frame."""

    replay: KITTIReplay = st.session_state.replay
    replay.reset()
    st.session_state.fusion = FusionEngine()
    st.session_state.last_result = None
    clear_history()

    for _ in range(max(0, target + 1)):
        if not _process_next_frame():
            break


def _render_header(replay: KITTIReplay) -> None:
    """Render the research dashboard header and current replay summary."""

    st.title("Confidence-Driven Adaptive Multi-Sensor Fusion Dashboard")
    st.caption(
        "Real-Time Predictive Localization using Adaptive EKF and Sensor "
        "Confidence Index"
    )

    current = st.session_state.get("last_result")
    status = current["quality"]["status"] if current else "WAITING"
    sci = current["quality"]["sci"] if current else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Frame", f"{replay.current_index()} / {len(replay)}")
    c2.metric("Replay FPS", st.session_state.get("fps", 10))
    c3.metric("Sensor Status", status)
    c4.metric("Current SCI", f"{sci:.3f}")
    dataset_label = st.session_state.get("dataset", "KITTI Raw")
    selection_label = st.session_state.get("selection", "")
    st.caption(f"📂 {dataset_label}  ·  {selection_label}")


def _render_dashboard(
    result: dict[str, Any] | None,
    replay: KITTIReplay,
    speed: float,
    demo_mode: bool,
) -> None:
    """Render the modular dashboard panels for the current fusion result."""

    if result is None:
        st.info("Use Next Frame or Play to begin the KITTI replay.")
        return

    render_status_ribbon(result, replay, speed)

    camera, sensor = st.columns([3, 2])
    with camera:
        render_camera(result["packet"])
    with sensor:
        render_sensor_panel(result)

    render_health_gauges(result)

    st.divider()
    render_telemetry_panel(result)

    if not demo_mode:
        fusion, localization = st.columns(2)
        with fusion:
            render_fusion_panel(result)
        with localization:
            render_localization_panel(result)
    else:
        render_localization_panel(result)

    st.divider()
    render_sci_and_difference()

    st.divider()
    st.header("📊 Analytics Dashboard")
    analytics_placeholder = st.empty()
    with analytics_placeholder.container():
        render_all_charts()


initialize_history()

if "playing" not in st.session_state:
    st.session_state.playing = False

if "fps" not in st.session_state:
    st.session_state.fps = 10

settings = render_sidebar(
    st.session_state.get("replay"),
    st.session_state.get("last_result"),
)

_dataset = settings["dataset"]
_selection = settings["selection"]

if _selection is None:
    st.error("No drives or sequences found. Please check your dataset paths in config.py.")
    st.stop()

if (
    "replay" not in st.session_state
    or st.session_state.get("dataset") != _dataset
    or st.session_state.get("selection") != _selection
):
    _new_pipeline(_dataset, _selection)

st.session_state.fps = settings["fps"]
replay: KITTIReplay | KITTIOdometryReplay = st.session_state.replay

_render_header(replay)
controls = render_controls()

publication_mode = st.toggle("Publication Mode", value=False)
demo_mode = st.toggle("Demo Mode", value=False)
if publication_mode:
    st.markdown("<style>.stPlotlyChart {background: white;} h1,h2,h3 {font-size: 1.2em;}</style>", unsafe_allow_html=True)

timeline_frame = st.slider(
    "Replay Timeline",
    min_value=0,
    max_value=max(0, len(replay) - 1),
    value=max(0, replay.current_index() - 1),
    help="Jump to a replay frame; fusion state and visual history are rebuilt for scientific consistency."
)

if controls["pause"]:
    st.session_state.playing = False

if controls["reset"]:
    replay.reset()
    st.session_state.fusion = FusionEngine()
    st.session_state.last_result = None
    st.session_state.playing = False
    clear_history()

if controls["previous"]:
    st.session_state.playing = False
    _rebuild_to_frame(replay.current_index() - 2)

if (
    not controls["reset"]
    and not controls["previous"]
    and timeline_frame != max(0, replay.current_index() - 1)
):
    st.session_state.playing = False
    _rebuild_to_frame(timeline_frame)

if controls["play"]:
    st.session_state.playing = True

should_advance = controls["next"] or st.session_state.playing
if should_advance:
    _process_next_frame()

st.progress(replay.progress(), text=f"Replay progress: {replay.progress() * 100:.1f}%")
_render_dashboard(
    st.session_state.get("last_result"),
    replay,
    controls["speed"],
    demo_mode,
)

st.divider()
_dataset_label = st.session_state.get("dataset", "KITTI Raw")
st.caption(
    f"Dataset: {_dataset_label}  |  Fusion: Adaptive EKF  |  Localization: "
    "Random Forest  |  Research: Paper 2"
)
render_research_and_exports()

if st.session_state.playing and replay.has_next():
    time.sleep(1.0 / (st.session_state.fps * controls["speed"]))
    st.rerun()
elif not replay.has_next():
    st.session_state.playing = False
