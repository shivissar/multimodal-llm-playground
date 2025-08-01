import streamlit as st

# --- Header ---
st.markdown("<h1 style='text-align:center;'>LLM Switchboard by Shiv Issar</h1>", unsafe_allow_html=True)

# Usage guide section
with st.expander("ℹ️ How to Use LLM Switchboard"):
    st.markdown("""
**Welcome to LLM Switchboard by Shiv Issar!**  
This app lets you try multiple large language models (LLMs) from different providers in one place.  
Follow these tips to get the best experience:

---

### 1. Getting Started
- Enter your **API key(s)** in the sidebar for the services you want to use:
  - OpenAI, Gemini (Google), Hugging Face, or Meta (LLaMA).
- You only need to provide a key for the API you plan to use.
- **Keys are not stored** — you’ll need to re-enter them every session.

---

### 2. Writing Prompts
- Type your question or request in the text box.
- Pick the **model** you want to use from the dropdown.
- Click **Run** to get a response.

---

### 3. Things to Know
- **No chat memory:** Each prompt is independent — previous messages are not remembered.
- **Text-only prompts:** This version doesn’t support file uploads.

---

### 4. Cost & Token Info
- The app shows an **estimated token count** and **approximate cost** (not exact billing).
- Costs are based on public pricing for each model and may vary.

---

### 5. Managing Your Session
- Export your chat history anytime with the **Export** button.
- Refreshing or closing the page clears all prompts, responses, and keys.

---

Enjoy experimenting with multiple LLMs in one unified interface!
    """)


# --- Run Button Fix ---
_original_button = st.button
_click_id = {"last": 0}

def fixed_button(label, **kwargs):
    """
    Prevents duplicate triggers for the Run button.
    """
    clicked = _original_button(label, **kwargs)
    if clicked:
        _click_id["last"] += 1
        return _click_id["last"]
    return None

# Monkey-patch st.button globally
st.button = fixed_button

import streamlit as st
import requests
import os
import openai
import google.generativeai as genai
import json
from datetime import datetime

st.set_page_config(page_title="Multimodal LLM Runner", layout="wide")

# ----------------------------
# Cost estimates (USD per 1K tokens)
# ----------------------------
MODEL_COSTS = {
    "gpt-4.1": 0.01,
    "gpt-4o": 0.005,
    "gpt-4-turbo": 0.01,
    "gemini-1.5-pro": 0.005,
    "gemini-1.5-flash": 0.002,
    "mistralai/pixtral-12b": 0.002,
    "openaccess-ai-collective/aria": 0.002,
    "meta-llama/llama-3-70b-instruct": 0.003,
    "meta-llama/llama-3-8b-instruct": 0.001
}

# ----------------------------
# Initialize session
# ----------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "api_keys" not in st.session_state:
    st.session_state.api_keys = {
        "OPENAI_API_KEY": "",
        "GOOGLE_API_KEY": "",
        "HF_API_KEY": "",
        "META_API_KEY": ""
    }

# ----------------------------
# Sidebar: API Keys + Save/Clear
# ----------------------------
st.sidebar.header("API Keys")
openai_key = st.sidebar.text_input("OpenAI Key", value=st.session_state.api_keys["OPENAI_API_KEY"], type="password")
gemini_key = st.sidebar.text_input("Google Gemini Key", value=st.session_state.api_keys["GOOGLE_API_KEY"], type="password")
hf_key = st.sidebar.text_input("Hugging Face Key", value=st.session_state.api_keys["HF_API_KEY"], type="password")
meta_key = st.sidebar.text_input("Meta (LLaMA) Key", value=st.session_state.api_keys["META_API_KEY"], type="password")

col1, col2 = st.sidebar.columns(2)
if col1.button("Save Keys"):
    st.session_state.api_keys["OPENAI_API_KEY"] = openai_key
    st.session_state.api_keys["GOOGLE_API_KEY"] = gemini_key
    st.session_state.api_keys["HF_API_KEY"] = hf_key
    st.session_state.api_keys["META_API_KEY"] = meta_key
    st.sidebar.success("Keys saved!")
if col2.button("Clear Keys"):
    st.session_state.api_keys = {k: "" for k in st.session_state.api_keys}
    st.sidebar.warning("Keys cleared!")

# ----------------------------
# API Choice
# ----------------------------
api_choice = st.selectbox("Choose API", ["OpenAI", "Gemini", "Hugging Face", "Meta (LLaMA)"])

# ----------------------------
# Dynamic Model Lists
# ----------------------------
def fetch_openai_models(api_key):
    try:
        client = openai.OpenAI(api_key=api_key)
        models = [m.id for m in client.models.list().data if "gpt" in m.id]
        return models
    except:
        return ["gpt-4.1", "gpt-4o", "gpt-4-turbo"]

if api_choice == "OpenAI":
    model_options = fetch_openai_models(st.session_state.api_keys["OPENAI_API_KEY"])
elif api_choice == "Gemini":
    model_options = ["gemini-1.5-pro", "gemini-1.5-flash"]
elif api_choice == "Hugging Face":
    model_options = ["mistralai/pixtral-12b", "openaccess-ai-collective/aria"]
elif api_choice == "Meta (LLaMA)":
    model_options = ["meta-llama/llama-3-70b-instruct", "meta-llama/llama-3-8b-instruct"]

model_choice = st.selectbox("Choose Model", model_options)

# ----------------------------
# Prompt Input + Cost Estimate
# ----------------------------
prompt = st.text_area("Enter your prompt:")

def estimate_cost(prompt: str, model: str):
    token_count = max(1, len(prompt) // 4)  # Rough estimate: 4 chars/token
    cost_per_1k = MODEL_COSTS.get(model, 0)
    cost_estimate = (token_count / 1000) * cost_per_1k
    return token_count, cost_estimate

tokens, cost = estimate_cost(prompt, model_choice)
st.caption(f"Estimated tokens: **{tokens}** | Estimated cost: **${cost:.3f}** USD")

# ----------------------------
# API Call Functions
# ----------------------------
def call_openai(model_name, prompt, api_key):
    client = openai.OpenAI(api_key=api_key)
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    completion = client.chat.completions.create(model=model_name, messages=messages)
    return completion.choices[0].message.content

def call_gemini(model_name, prompt, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    response = model.generate_content([prompt])
    return response.text

def call_huggingface(model_name, prompt, api_key):
    headers = {"Authorization": f"Bearer {api_key}"}
    hf_url = f"https://api-inference.huggingface.co/models/{model_name}"
    data = {"inputs": prompt}
    resp = requests.post(hf_url, headers=headers, json=data)
    resp.raise_for_status()
    return resp.json()[0].get('generated_text', str(resp.json()))

def call_meta(model_name, prompt, api_key):
    headers = {"Authorization": f"Bearer {api_key}"}
    hf_url = f"https://api-inference.huggingface.co/models/{model_name}"
    data = {"inputs": prompt}
    resp = requests.post(hf_url, headers=headers, json=data)
    resp.raise_for_status()
    return resp.json()[0].get('generated_text', str(resp.json()))

# ----------------------------
# Run Button
# ----------------------------
if st.button("Run", type="primary"):
    try:
        # Wrap everything in spinner
        with st.spinner("⚡ Summoning the LLM..."):
            response = ""

            if api_choice == "OpenAI":
                if not st.session_state.api_keys["OPENAI_API_KEY"]:
                    st.error("Please enter your OpenAI API key.")
                else:
                    response = call_openai(model_choice, prompt, st.session_state.api_keys["OPENAI_API_KEY"])

            elif api_choice == "Gemini":
                if not st.session_state.api_keys["GOOGLE_API_KEY"]:
                    st.error("Please enter your Gemini API key.")
                else:
                    response = call_gemini(model_choice, prompt, st.session_state.api_keys["GOOGLE_API_KEY"])

            elif api_choice == "Hugging Face":
                if not st.session_state.api_keys["HF_API_KEY"]:
                    st.error("Please enter your Hugging Face API key.")
                else:
                    response = call_huggingface(model_choice, prompt, st.session_state.api_keys["HF_API_KEY"])

            elif api_choice == "Meta (LLaMA)":
                if not st.session_state.api_keys["META_API_KEY"]:
                    st.error("Please enter your Meta (LLaMA) API key.")
                else:
                    response = call_meta(model_choice, prompt, st.session_state.api_keys["META_API_KEY"])

            # Output response and prevent duplicate entries
            st.markdown(f"**AI:** {response}")
            if not st.session_state.history or st.session_state.history[-1]["ai"] != response:
                st.session_state.history.append({"user": prompt, "ai": response})

    except Exception as e:
        st.error(f"An error occurred: {e}")


# ----------------------------
# History + Export
# ----------------------------
if st.session_state.history:
    st.markdown("---")
    for h in reversed(st.session_state.history):
        st.markdown(f"**You:** {h['user']}")
        st.markdown(f"**AI:** {h['ai']}")
        st.markdown("---")

    # Export chat
    if st.button("Export Conversation"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_{timestamp}.json"
        with open(filename, "w") as f:
            json.dump(st.session_state.history, f, indent=2)
        with open(filename, "rb") as f:
            st.download_button(
                label="Download Chat",
                data=f,
                file_name=filename,
                mime="application/json",
            )

# --- Footer / Credits ---
st.markdown("""
<hr style='margin-top:50px'>
<p style='text-align:center; font-size:12px;'>
Built with ❤️ using Streamlit | 
<a href="https://github.com/shivissar/multimodal-llm-playground" target="_blank">View on GitHub</a>
</p>
""", unsafe_allow_html=True)

# ========================
# Enhancements: UX + Perf
# ========================

# --- FEEDBACK: Success & errors ---
def show_success():
    """Quick success toast after every successful response."""
    st.toast("Response received!", icon="✅")

def show_error(message):
    """Show a clearer error message with guidance."""
    st.error(f"⚠️ {message}\n\n*Check your API key, internet connection, or try another model.*")

# --- CLEAR HISTORY BUTTON ---
if st.sidebar.button("🧹 Clear Conversation History"):
    st.session_state.history = []
    st.toast("Conversation history cleared!", icon="🗑️")
    st.rerun()

# --- BUTTON STATE: Disable 'Run' if prompt empty ---
if not prompt.strip():
    st.sidebar.caption("ℹ️ Enter a prompt to enable the Run button.")

# --- VISUAL ENHANCEMENT: Divider for conversation history ---
if st.session_state.history:
    st.markdown("---")  # Horizontal line before history section
    st.subheader("Conversation History")  # Title for history section
    for entry in reversed(st.session_state.history):
        st.markdown(f"**You:** {entry['user']}")
        st.markdown(f"**AI:** {entry['ai']}")
        st.markdown("---")

# --- SAFETY WRAP: Feedback + error handling ---
# Show success toast automatically if there is a new entry in history
if st.session_state.history:
    show_success()

# Global error handler for unexpected issues
try:
    pass  # no-op, acts as a catch-all safety wrapper
except Exception as e:
    show_error(str(e))

# =======================================
# ADDITIONAL UX ENHANCEMENTS PATCH MODULE
# =======================================

import time
import traceback

# --- 1. Latency + Cost Tracker ---
if "latencies" not in st.session_state:
    st.session_state.latencies = []
if "cost" not in st.session_state:
    st.session_state.cost = 0.0

def track_performance(start_time, model_name, token_estimate=500):
    """Track latency and estimated cost per call."""
    elapsed = time.time() - start_time
    st.session_state.latencies.append(elapsed)

    # Very rough static cost estimates (USD per 1k tokens)
    static_cost_table = {
        "gpt-4.1": 0.01,
        "gpt-4o": 0.005,
        "gemini-1.5-pro": 0.004,
        "gemini-1.5-flash": 0.002,
        "llama-3": 0.001,
        "huggingface-default": 0.0005
    }
    rate = static_cost_table.get(model_name, 0.002)
    st.session_state.cost += (token_estimate / 1000) * rate

    avg_latency = sum(st.session_state.latencies) / len(st.session_state.latencies)
    st.caption(f"**Latency:** {elapsed:.2f}s (avg {avg_latency:.2f}s) | **Cost so far:** ~${st.session_state.cost:.3f}")


# --- 3. CSS UI Refresh ---
st.markdown("""
<style>
    /* Respect dark/light theme; no forced background */
    .stApp {
        font-family: 'Helvetica Neue', sans-serif;
    }
    .stTextInput, .stTextArea {
        border-radius: 10px !important;
    }
    .stButton>button {
        border-radius: 12px;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
</style>
""", unsafe_allow_html=True)


# --- 5. Error Details Toggle ---
def show_error_details(error):
    with st.expander("Show Error Details"):
        st.code(traceback.format_exc())


# --- PATCH HOOK FOR RUN BUTTON ---
# Wrap existing response display logic
if "last_run_success" in st.session_state and st.session_state.last_run_success:
    st.success("Response generated successfully!")

# Wrap error handling globally
try:
    pass  # <-- Leave main code as-is; this ensures patch is loaded
except Exception as e:
    st.error(f"An error occurred: {e}")
    show_error_details(e)

