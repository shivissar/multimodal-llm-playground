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

        st.markdown(f"**AI:** {response}")
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

