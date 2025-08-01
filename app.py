import streamlit as st
from openai import OpenAI
import google.generativeai as genai
import requests
import os

# ----------------------------
# APP CONFIG
# ----------------------------
st.set_page_config(page_title="Multimodal LLM Playground", layout="wide")

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
# COST TABLE (USD per 1K tokens, approx.)
# ----------------------------
COST_TABLE = {
    "gpt-4o": 0.005,        # example
    "gpt-4o-mini": 0.002,
    "gemini-1.5-pro": 0.004,
    "gemini-1.5-flash": 0.001,
    "Pixtral-12B": 0.0005,  # Hugging Face models (rough)
    "Aria": 0.0005,
    "llama-3-8b": 0.0004    # Meta models placeholder
}

# ----------------------------
# HELPER: Estimate Tokens & Cost
# ----------------------------
def estimate_tokens_and_cost(text, model):
    # Approx: 1 token ≈ 4 chars
    tokens = len(text) // 4
    rate = COST_TABLE.get(model, 0.001)
    cost = tokens / 1000 * rate
    return tokens, cost

# ----------------------------
# FETCH MODEL LISTS
# ----------------------------
def fetch_openai_models(api_key):
    try:
        client = OpenAI(api_key=api_key)
        return [m.id for m in client.models.list().data if "gpt" in m.id]
    except Exception:
        # fallback
        return ["gpt-4o", "gpt-4o-mini"]

def fetch_gemini_models(api_key):
    try:
        genai.configure(api_key=api_key)
        # Hardcode since Google doesn’t expose list endpoint
        return ["gemini-1.5-pro", "gemini-1.5-flash"]
    except Exception:
        return ["gemini-1.5-pro", "gemini-1.5-flash"]

def fetch_huggingface_models(api_key):
    # Placeholder: dynamic listing requires HF API call, but here is static
    return ["Pixtral-12B", "Aria"]

def fetch_meta_models(api_key):
    # Placeholder for Meta (LLaMA) models
    return ["llama-3-8b", "llama-3-70b"]

# ----------------------------
# API CALL FUNCTIONS
# ----------------------------
def call_openai(model_name, prompt, api_key):
    client = OpenAI(api_key=api_key)
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    completion = client.chat.completions.create(
        model=model_name,
        messages=messages
    )
    return completion.choices[0].message.content

def call_gemini(model_name, prompt, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    response = model.generate_content([prompt])
    return response.text

def call_huggingface(model_name, prompt, api_key):
    headers = {"Authorization": f"Bearer {api_key}"}
    model_map = {
        "Pixtral-12B": "mistralai/Pixtral-12B",
        "Aria": "openaccess-ai-collective/Aria"
    }
    url = f"https://api-inference.huggingface.co/models/{model_map.get(model_name)}"
    resp = requests.post(url, headers=headers, json={"inputs": prompt})
    try:
        return resp.json()[0].get("generated_text", str(resp.json()))
    except Exception:
        return str(resp.json())

def call_meta(model_name, prompt, api_key):
    # Placeholder: Meta API endpoint not public
    return f"[Meta model {model_name}] Response simulation: {prompt[:50]}..."

# ----------------------------
# SIDEBAR: API KEYS & SETTINGS
# ----------------------------
st.sidebar.header("API Keys")
st.sidebar.caption("Enter any available keys. At least one is required to run.")

openai_key = st.sidebar.text_input("OpenAI Key", value=st.session_state.api_keys["OPENAI_API_KEY"], type="password")
gemini_key = st.sidebar.text_input("Google Gemini Key", value=st.session_state.api_keys["GOOGLE_API_KEY"], type="password")
hf_key = st.sidebar.text_input("Hugging Face Key", value=st.session_state.api_keys["HF_TOKEN"], type="password")
meta_key = st.sidebar.text_input("Meta (LLaMA) Key", value=st.session_state.api_keys["META_API_KEY"], type="password")

# Save keys
if st.sidebar.button("Save Keys"):
    st.session_state.api_keys.update({
        "OPENAI_API_KEY": openai_key,
        "GOOGLE_API_KEY": gemini_key,
        "HF_TOKEN": hf_key,
        "META_API_KEY": meta_key
    })
    st.sidebar.success("Keys saved to session.")

# API Selection
api_choice = st.sidebar.selectbox("Choose API", ["OpenAI", "Gemini", "Hugging Face", "Meta (LLaMA)"])

# Dynamic Model Fetch
if api_choice == "OpenAI":
    models = fetch_openai_models(openai_key) if openai_key else ["gpt-4o", "gpt-4o-mini"]
elif api_choice == "Gemini":
    models = fetch_gemini_models(gemini_key) if gemini_key else ["gemini-1.5-pro", "gemini-1.5-flash"]
elif api_choice == "Hugging Face":
    models = fetch_huggingface_models(hf_key) if hf_key else ["Pixtral-12B", "Aria"]
else:
    models = fetch_meta_models(meta_key)

model_choice = st.sidebar.selectbox("Model", models)

# Show cost summary
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Total Estimated Cost:** ${st.session_state.total_cost:.4f}")

# ----------------------------
# MAIN APP
# ----------------------------
st.title("🔮 Multimodal LLM Playground (Text Only)")

prompt = st.text_area("Enter your prompt:")

if st.button("Run", type="primary"):
    if api_choice == "OpenAI" and not openai_key:
        st.error("Please enter OpenAI API Key in sidebar.")
    elif api_choice == "Gemini" and not gemini_key:
        st.error("Please enter Google Gemini API Key in sidebar.")
    elif api_choice == "Hugging Face" and not hf_key:
        st.error("Please enter Hugging Face API Key in sidebar.")
    elif api_choice == "Meta (LLaMA)" and not meta_key:
        st.error("Please enter Meta API Key in sidebar.")
    else:
        try:
            # Dispatch to proper API
            if api_choice == "OpenAI":
                response = call_openai(model_choice, prompt, openai_key)
            elif api_choice == "Gemini":
                response = call_gemini(model_choice, prompt, gemini_key)
            elif api_choice == "Hugging Face":
                response = call_huggingface(model_choice, prompt, hf_key)
            else:
                response = call_meta(model_choice, prompt, meta_key)

            # Token & cost estimation
            tokens, cost = estimate_tokens_and_cost(prompt + response, model_choice)
            st.session_state.total_cost += cost

            # Save to history
            st.session_state.history.append({"user": prompt, "ai": response})

            # Display
            st.success("Response:")
            st.write(response)
            st.caption(f"~{tokens} tokens, estimated ${cost:.4f}")

        except Exception as e:
            st.error(f"An error occurred: {e}")

# ----------------------------
# HISTORY
# ----------------------------
if st.session_state.history:
    st.markdown("---")
    st.subheader("Conversation History")
    for h in reversed(st.session_state.history):
        st.markdown(f"**You:** {h['user']}")
        st.markdown(f"**AI:** {h['ai']}")
        st.markdown("---")

