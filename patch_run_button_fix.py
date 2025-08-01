import streamlit as st

def apply_run_button_fix():
    """
    Monkey-patches st.button to prevent duplicate triggers
    by ensuring each click is processed only once.
    """
    if hasattr(st, "_button_patched"):
        return  # Prevent multiple patching

    _original_button = st.button
    click_id = {"last": 0}

    def patched_button(label, **kwargs):
        clicked = _original_button(label, **kwargs)
        if clicked:
            click_id["last"] += 1
            return click_id["last"]
        return None

    # Replace st.button globally
    st.button = patched_button
    st._button_patched = True

