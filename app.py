import streamlit as st
import requests
import os
import openai
import google.generativeai as genai

st.set_page_config(page_title="Multimodal LLM Runner", layout="wide")

# ----------------------------
# Static cost estimates (USD per 1K tokens)
# ----------------------------
MODEL_COSTS = {
    "gpt-4.1": 0.01,
    "gpt-4o": 0.005,
    "gpt-4-turbo": 0.01,
    "gemini-1.5-pro": 0.005,
    "gemini-1.5-flash": 0.002,
    "mistralai/pixtral-12b": 0.002,
    "openaccess-ai-collective/aria": 0.002,
    "llava": 0.0  # Local Ollama (free)
}

# ----------------------------
# Initialize session
# ----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ----------------------------
# API Keys
# ----------------------------
st.sidebar.header("API Keys")
openai_key = st.sidebar.text_input("OpenAI Key", type="password")
gemini_key = st.sidebar.text_input("Google Gemini Key", type="password")
hf_key = st.sidebar.text_input("Hugging Face Key", type="password")

# ----------------------------
# API Choice and Dynamic Models
# ----------------------------
api_choice = st.selectbox("Choose API", ["Offline (Ollama)", "OpenAI", "Gemini", "Hugging Face"])

if api_choice == "OpenAI":
    model_options = ["gpt-4.1", "gpt-4o", "gpt-4-turbo"]
elif api_choice == "Gemini":
    model_options = ["gemini-1.5-pro", "gemini-1.5-flash"]
elif api_choice == "Hugging Face":
    model_options = ["mistralai/pixtral-12b", "openaccess-ai-collective/aria"]
else:
    model_options = ["llava"]

model_choice = st.selectbox("Choose Model", model_options)

# ----------------------------
# Prompt Input
# ----------------------------
prompt = st.text_area("Enter your prompt:")

# Token & cost estimate
def estimate_cost(prompt: str, model: str):
    token_count = max(1, len(prompt) // 4)  # Rough approximation
    cost_per_1k = MODEL_COSTS.get(model, 0)
    cost_estimate = (token_count / 1000) * cost_per_1k
    return token_count, cost_estimate

tokens, cost = estimate_cost(prompt, model_choice)
st.caption(f"Estimated tokens: **{tokens}** | Estimated cost: **${cost:.3f}** USD")

# ----------------------------
# API Functions
# ----------------------------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")

def call_ollama(prompt):
    payload = {"model": "llava", "prompt": prompt, "stream": False}
    resp = requests.post(OLLAMA_URL, json=payload)
    resp.raise_for_status()
    return resp.json().get("response", "No response key in Ollama output.")

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

# ----------------------------
# Run Button
# ----------------------------
if st.button("Run", type="primary"):
    try:
        if api_choice == "Offline (Ollama)":
            response = call_ollama(prompt)
        elif api_choice == "OpenAI":
            if not openai_key:
                st.error("Please enter your OpenAI API key.")
            else:
                response = call_openai(model_choice, prompt, openai_key)
        elif api_choice == "Gemini":
            if not gemini_key:
                st.error("Please enter your Gemini API key.")
            else:
                response = call_gemini(model_choice, prompt, gemini_key)
        elif api_choice == "Hugging Face":
            if not hf_key:
                st.error("Please enter your Hugging Face API key.")
            else:
                response = call_huggingface(model_choice, prompt, hf_key)

        st.markdown(f"**AI:** {response}")
        st.session_state.history.append({"user": prompt, "ai": response})

    except Exception as e:
        st.error(f"An error occurred: {e}")

# ----------------------------
# History
# ----------------------------
if st.session_state.history:
    st.markdown("---")
    for h in reversed(st.session_state.history):
        st.markdown(f"**You:** {h['user']}")
        st.markdown(f"**AI:** {h['ai']}")
        st.markdown("---")

