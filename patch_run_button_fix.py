import streamlit as st

def prevent_double_run():
    """
    Ensures the 'Run' button only executes once per click.
    """
    if "run_in_progress" not in st.session_state:
        st.session_state.run_in_progress = False

    if st.button("Run", type="primary"):
        if not st.session_state.run_in_progress:
            st.session_state.run_in_progress = True
            return True
    return False

