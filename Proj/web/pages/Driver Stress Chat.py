"""
Driver Stress Chat — Streamlit page that talks to a local Ollama model.
Requires Ollama running locally (e.g. ollama serve) and a model created from the Modelfile.
"""
import asyncio
import base64
import json
import tempfile
import threading
from typing import List, Dict, Optional

import requests
import streamlit as st

try:
    import edge_tts
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

# =========================================
# CONFIG — hardcoded, no UI controls needed
# =========================================
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "driverbot:latest"

st.set_page_config(page_title="Driver Stress Assistant", layout="wide")

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


# ---------- TTS helpers ----------
def _tts_to_bytes(text: str, voice: str = "en-US-JennyNeural"):
    if not TTS_AVAILABLE or not text.strip():
        return None

    async def _gen(path):
        communicate = edge_tts.Communicate(text, voice=voice)
        await communicate.save(path)

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        tmp_path = f.name

    result = [None]

    def _run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(_gen(tmp_path))
            with open(tmp_path, "rb") as f:
                result[0] = f.read()
        except Exception:
            pass
        finally:
            loop.close()

    t = threading.Thread(target=_run)
    t.start()
    t.join()
    return result[0]


def speak_text(text: str):
    if not TTS_AVAILABLE or not text or text.startswith("[Error"):
        return
    audio_bytes = _tts_to_bytes(text)
    if audio_bytes:
        b64 = base64.b64encode(audio_bytes).decode()
        html = (
            f'<audio autoplay style="display:none">'
            f'<source src="data:audio/mp3;base64,{b64}" type="audio/mp3">'
            f'</audio>'
        )
        st.markdown(html, unsafe_allow_html=True)


# ---------- Ollama helpers ----------
def ollama_chat_stream(messages: List[Dict]):
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {"model": DEFAULT_MODEL, "messages": messages, "stream": True}
    try:
        with requests.post(url, json=payload, stream=True, timeout=120) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                    if "error" in chunk:
                        yield None
                        return
                    content = (chunk.get("message") or {}).get("content") or ""
                    if content:
                        yield content
                    if chunk.get("done"):
                        break
                except json.JSONDecodeError:
                    continue
    except requests.exceptions.RequestException:
        yield None


def ollama_chat(messages: List[Dict]) -> Optional[str]:
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {"model": DEFAULT_MODEL, "messages": messages, "stream": False}
    try:
        r = requests.post(url, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        if "error" in data:
            return None
        return (data.get("message") or {}).get("content") or ""
    except requests.exceptions.RequestException:
        return None


# ---------- Session state ----------
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "tts_pending" not in st.session_state:
    st.session_state.tts_pending = None

# ---------- Sidebar ----------
st.sidebar.title("Driver Stress Assistant")
use_stream = st.sidebar.checkbox("Stream responses", value=True)
use_tts = st.sidebar.checkbox(
    "Read replies aloud (TTS)", value=True,
    disabled=not TTS_AVAILABLE,
    help="Requires edge-tts. Install with: pip install edge-tts"
)

if st.sidebar.button("Clear chat"):
    st.session_state.chat_messages = []
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("Chat with your local Ollama driver stress model.")

# ---------- Main ----------
st.title("Driver Stress Assistant")
st.markdown("Ask for calm, safety-focused driving and stress advice. Powered by your local Ollama model.")

# Play pending TTS on the render after reply is saved
if st.session_state.tts_pending:
    speak_text(st.session_state.tts_pending)
    st.session_state.tts_pending = None

# Chat history
for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
if prompt := st.chat_input("Type your message..."):
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    api_messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat_messages]

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if use_stream:
            full_reply = []
            stream_placeholder = st.empty()
            for delta in ollama_chat_stream(api_messages):
                if delta is None:
                    stream_placeholder.error("Could not reach Ollama. Is it running?")
                    full_reply = ["[Error: connection failed]"]
                    break
                full_reply.append(delta)
                stream_placeholder.markdown("".join(full_reply))
            reply_text = "".join(full_reply)
        else:
            reply_text = ollama_chat(api_messages)
            if reply_text is None:
                st.error("Could not reach Ollama. Is it running?")
                reply_text = "[Error: connection failed]"
            else:
                st.markdown(reply_text)

    st.session_state.chat_messages.append({"role": "assistant", "content": reply_text})
    if use_tts:
        st.session_state.tts_pending = reply_text
    st.rerun()
