import streamlit.web.bootstrap
import patch_run_button_fix
import patch_header

# Patch function calls will execute after Streamlit reloads
def apply_patches():
    patch_header.inject_header()

# Hook: Streamlit runs this after the app starts
st_patch_hooks = apply_patches

# Launch original app
streamlit.web.bootstrap.run("app.py", "", [], {})

