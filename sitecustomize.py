import streamlit as st

# --- Header Patch ---
_original_title = st.title
def patched_title(*args, **kwargs):
    # Inject custom header
    st.markdown("<h1 style='text-align:center;'>LLM Switchboard by Shiv Issar</h1>", unsafe_allow_html=True)
    st.title = _original_title
    return _original_title(*args, **kwargs)
st.title = patched_title

# --- Run Button Patch ---
_original_button = st.button
click_id = {"last": 0}

def patched_button(label, **kwargs):
    clicked = _original_button(label, **kwargs)
    if clicked:
        click_id["last"] += 1
        return click_id["last"]
    return None

st.button = patched_button

