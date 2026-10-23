import streamlit as st


def render_camera(packet):

    st.subheader("Live Camera Feed")

    st.caption(
        "Current KITTI replay frame captured from the front-facing camera."
    )

    st.image(

        packet["image"],

        caption=f"Frame {packet['frame_id']}",

        width="stretch"

    )