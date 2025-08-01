import streamlit.web.bootstrap
import patch_header
import patch_run_button_fix

# Monkey-patch Streamlit:
# 1. Add header before first render
# 2. Fix duplicate Run button clicks

import streamlit as st

# Patch header: inject on first title call
_original_title = st.title
def patched_title(*args, **kwargs):
    patch_header.inject_header()
    st.title = _original_title  # Restore normal behavior
    return _original_title(*args, **kwargs)
st.title = patched_title

# Patch Run button globally
patch_run_button_fix.apply_run_button_fix()

# Launch the original app
streamlit.web.bootstrap.run("app.py", "", [], {})

