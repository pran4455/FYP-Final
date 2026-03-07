"""
Driver Stress Chat — Streamlit page that talks to a local Ollama model.
Requires Ollama running locally (e.g. ollama serve) and a model created from the Modelfile.
"""
import json
from typing import List, Dict, Optional

import requests
import streamlit as st

# =========================================
# CONFIG
# =========================================
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "driver_stress_ai"

st.set_page_config(page_title="Driver Stress Assistant", layout="wide")

# --- Hide Streamlit default UI elements (match other pages) ---
hide_streamlit_style = """
    <style>
    [data-testid="stToolbar"] {visibility: hidden !important;}
    [data-testid="stDecoration"] {visibility: hidden !important;}
    [data-testid="stStatusWidget"] {visibility: hidden !important;}
    #MainMenu {visibility: hidden !important;}
    header {visibility: hidden !important;}
    footer {visibility: hidden !important;}
    .stAppDeployButton {visibility: hidden !important;}
    div[data-testid="stProgressBar"] {visibility: hidden !important;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


def list_ollama_models(base_url: str) -> List[str]:
    """Fetch list of available models from Ollama."""
    try:
        r = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=5)
        r.raise_for_status()
        data = r.json()
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def ollama_chat_stream(base_url: str, model: str, messages: List[Dict]):
    """Stream chat completion from Ollama; yields content deltas."""
    url = f"{base_url.rstrip('/')}/api/chat"
    payload = {"model": model, "messages": messages, "stream": True}
    try:
        with requests.post(url, json=payload, stream=True, timeout=120) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                    msg = chunk.get("message") or {}
                    content = msg.get("content") or ""
                    if content:
                        yield content
                    if chunk.get("done"):
                        break
                except json.JSONDecodeError:
                    continue
    except requests.exceptions.RequestException:
        yield None  # signal error to caller


def ollama_chat(base_url: str, model: str, messages: List[Dict]) -> Optional[str]:
    """Non-streaming chat; returns full response or None on error."""
    url = f"{base_url.rstrip('/')}/api/chat"
    payload = {"model": model, "messages": messages, "stream": False}
    try:
        r = requests.post(url, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        return (data.get("message") or {}).get("content") or ""
    except requests.exceptions.RequestException:
        return None


# ---------- Session state ----------
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# ---------- Sidebar ----------
st.sidebar.title("Driver Stress Assistant")
base_url = st.sidebar.text_input("Ollama URL", value=OLLAMA_BASE_URL, help="e.g. http://localhost:11434")
use_stream = st.sidebar.checkbox("Stream responses", value=True, help="Show reply as it’s generated")

# Model selection: try to list models, fallback to default
try:
    models = list_ollama_models(base_url)
    if models:
        model_index = 0
        if DEFAULT_MODEL in models:
            model_index = models.index(DEFAULT_MODEL)
        model_name = st.sidebar.selectbox("Model", models, index=model_index)
    else:
        model_name = st.sidebar.text_input("Model name", value=DEFAULT_MODEL)
except Exception:
    model_name = st.sidebar.text_input("Model name", value=DEFAULT_MODEL)

if st.sidebar.button("Clear chat"):
    st.session_state.chat_messages = []
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("Chat with your local Ollama driver stress model.")

# ---------- Main area ----------
st.title("Driver Stress Assistant")
st.markdown("Ask for calm, safety-focused driving and stress advice. Powered by your local Ollama model.")

# Chat container
chat_container = st.container()
with chat_container:
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# Input
if prompt := st.chat_input("Type your message..."):
    # Append user message
    st.session_state.chat_messages.append({"role": "user", "content": prompt})

    # Build messages for API (role + content only)
    api_messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat_messages]

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if use_stream:
            full_reply = []
            stream_placeholder = st.empty()
            for delta in ollama_chat_stream(base_url, model_name, api_messages):
                if delta is None:
                    stream_placeholder.error("Could not reach Ollama. Is it running? Check URL and model name.")
                    full_reply = ["[Error: connection failed]"]
                    break
                full_reply.append(delta)
                stream_placeholder.markdown("".join(full_reply))
            reply_text = "".join(full_reply)
        else:
            reply_text = ollama_chat(base_url, model_name, api_messages)
            if reply_text is None:
                st.error("Could not reach Ollama. Is it running? Check URL and model name.")
                reply_text = "[Error: connection failed]"
            else:
                st.markdown(reply_text)

    st.session_state.chat_messages.append({"role": "assistant", "content": reply_text})
    st.rerun()
