import streamlit as st
import requests
import os
from datetime import datetime
import openai
import google.generativeai as genai

# ----------------------------
# STATIC COST TABLE (USD per 1K tokens)
# ----------------------------
COST_TABLE = {
    "gpt-4o": 0.005,
    "gpt-4-turbo": 0.003,
    "gemini-1.5-pro": 0.0025,
    "gemini-1.5-flash": 0.0015,
    "llama-3-70b": 0.0018,
    "llama-3-8b": 0.0006
}

# ----------------------------
# SESSION STATE INIT
# ----------------------------
if "api_keys" not in st.session_state:
    st.session_state.api_keys = {
        "OPENAI_API_KEY": "",
        "GOOGLE_API_KEY": "",
        "HF_TOKEN": "",
        "META_API_KEY": ""
    }
if "history" not in st.session_state:
    st.session_state.history = []
if "total_cost" not in st.session_state:
    st.session_state.total_cost = 0.0

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="Multimodal LLM Playground", layout="wide")
st.title("🔮 Multimodal LLM Playground")

# ----------------------------
# API KEYS INPUT (manual or from Streamlit secrets)
# ----------------------------
openai_key = st.sidebar.text_input("OpenAI Key", value=st.session_state.api_keys.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", ""), type="password")
gemini_key = st.sidebar.text_input("Google Gemini Key", value=st.session_state.api_keys.get("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY", ""), type="password")
hf_key = st.sidebar.text_input("Hugging Face Key", value=st.session_state.api_keys.get("HF_TOKEN") or st.secrets.get("HF_TOKEN", ""), type="password")
meta_key = st.sidebar.text_input("Meta (LLaMA) Key", value=st.session_state.api_keys.get("META_API_KEY") or st.secrets.get("META_API_KEY", ""), type="password")

# Update stored keys
st.session_state.api_keys["OPENAI_API_KEY"] = openai_key
st.session_state.api_keys["GOOGLE_API_KEY"] = gemini_key
st.session_state.api_keys["HF_TOKEN"] = hf_key
st.session_state.api_keys["META_API_KEY"] = meta_key

# Configure OpenAI and Gemini clients if keys present
if openai_key:
    openai.api_key = openai_key
if gemini_key:
    genai.configure(api_key=gemini_key)

# ----------------------------
# DYNAMIC MODEL LISTS
# ----------------------------
OPENAI_MODELS = ["gpt-4o", "gpt-4-turbo"]
GEMINI_MODELS = ["gemini-1.5-pro", "gemini-1.5-flash"]
HF_MODELS = ["Pixtral-12B", "Aria"]
META_MODELS = ["llama-3-70b", "llama-3-8b"]

# ----------------------------
# SIDEBAR SETTINGS
# ----------------------------
api_choice = st.sidebar.selectbox(
    "Choose API Provider",
    ["OpenAI", "Google Gemini", "Hugging Face", "Meta (LLaMA)"]
)

if api_choice == "OpenAI":
    model_choice = st.sidebar.selectbox("Model", OPENAI_MODELS)
elif api_choice == "Google Gemini":
    model_choice = st.sidebar.selectbox("Model", GEMINI_MODELS)
elif api_choice == "Hugging Face":
    model_choice = st.sidebar.selectbox("Model", HF_MODELS)
elif api_choice == "Meta (LLaMA)":
    model_choice = st.sidebar.selectbox("Model", META_MODELS)
else:
    model_choice = None

# ----------------------------
# TOKEN + COST ESTIMATION
# ----------------------------
def estimate_tokens_and_cost(prompt: str, model: str):
    tokens = len(prompt.split())  # crude approximation
    cost_per_1k = COST_TABLE.get(model, 0.002)  # fallback if unknown
    cost = (tokens / 1000) * cost_per_1k
    return tokens, cost

# ----------------------------
# API CALL FUNCTIONS
# ----------------------------
def call_openai(prompt: str, model_name: str):
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message["content"]

def call_gemini(prompt: str, model_name: str):
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    return response.text

def call_huggingface(prompt: str, model_name: str):
    headers = {"Authorization": f"Bearer {hf_key}"}
    model_map = {
        "Pixtral-12B": "mistralai/Pixtral-12B",
        "Aria": "openaccess-ai-collective/Aria"
    }
    url = f"https://api-inference.huggingface.co/models/{model_map.get(model_name)}"
    payload = {"inputs": prompt}
    resp = requests.post(url, headers=headers, json=payload)
    if resp.status_code == 200:
        try:
            return resp.json()[0].get("generated_text", str(resp.json()))
        except Exception:
            return str(resp.json())
    return f"Error: {resp.text}"

def call_meta(prompt: str, model_name: str):
    # Placeholder for Meta's LLaMA
    return f"(Simulated response from Meta model '{model_name}')"

# ----------------------------
# PROMPT INPUT
# ----------------------------
prompt = st.text_area("Enter your prompt:", "Ask me anything...")

# ----------------------------
# RUN BUTTON
# ----------------------------
if st.button("Run", type="primary"):
    if not prompt.strip():
        st.error("Please enter a prompt.")
    else:
        with st.spinner("Generating response..."):
            try:
                # Dispatch API call
                if api_choice == "OpenAI" and openai_key:
                    ai_response = call_openai(prompt, model_choice)
                elif api_choice == "Google Gemini" and gemini_key:
                    ai_response = call_gemini(prompt, model_choice)
                elif api_choice == "Hugging Face" and hf_key:
                    ai_response = call_huggingface(prompt, model_choice)
                elif api_choice == "Meta (LLaMA)" and meta_key:
                    ai_response = call_meta(prompt, model_choice)
                else:
                    st.error(f"API Key for {api_choice} is missing.")
                    ai_response = None

                if ai_response:
                    tokens, cost = estimate_tokens_and_cost(prompt, model_choice)
                    st.session_state.total_cost += cost
                    st.session_state.history.append({
                        "user": prompt,
                        "ai": ai_response,
                        "tokens": tokens,
                        "cost": cost
                    })
                    st.success(f"Response received! Estimated cost: ${cost:.4f}")
                    st.rerun()

            except Exception as e:
                st.error(f"An error occurred: {e}")

# ----------------------------
# DISPLAY HISTORY
# ----------------------------
if st.session_state.history:
    st.markdown("---")
    st.subheader("Conversation History")
    for h in reversed(st.session_state.history):
        st.markdown(f"**You:** {h['user']}")
        st.markdown(f"**AI:** {h['ai']}")
        st.caption(f"Tokens: {h['tokens']} | Cost: ${h['cost']:.4f}")
        st.markdown("---")

st.sidebar.markdown("---")
st.sidebar.metric("Total Estimated Cost", f"${st.session_state.total_cost:.4f}")

