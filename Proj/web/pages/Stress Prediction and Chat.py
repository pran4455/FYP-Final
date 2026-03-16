"""
Stress Prediction & Chat — Run CNN-GRU-Attention stress prediction and feed result into the Ollama driver stress chatbot.
- Predict from test cases in the dataset (by index or test split) or from your own input data (CSV).
- View prediction (stress level + confidence) and the generic statement sent to the bot.
- Get a chatbot response based on the predicted stress level.
"""
import asyncio
import base64
import json
import sys
import tempfile
import threading
from pathlib import Path

import requests
import streamlit as st

try:
    import edge_tts
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

# Ensure Proj root is on path so we can import pred
_PROJ_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

try:
    from pred import (
        load_stress_model,
        load_scaler,
        load_label_mapping,
        predict_stress,
        predict_from_dataset,
        stress_level_to_prompt,
        predict_and_chat,
        ollama_chat,
        get_feature_columns,
        DEFAULT_CSV,
        SEQUENCE_LENGTH,
    )
    PRED_AVAILABLE = True
except Exception as e:
    PRED_AVAILABLE = False
    _PRED_ERROR = str(e)

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "driverbot:latest"

st.set_page_config(page_title="Stress Prediction & Chat", layout="wide")

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


def _tts_to_bytes(text: str, voice: str = "en-US-JennyNeural"):
    """Synthesize text to MP3 bytes using edge-tts, safe inside Streamlit's event loop."""
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


def speak_text(text: str, voice: str = "en-US-JennyNeural"):
    """Generate TTS for the full text and play it."""
    if not TTS_AVAILABLE or not text or text.strip() in ("[Error]", "[No response]"):
        return
    audio_bytes = _tts_to_bytes(text, voice)
    if audio_bytes:
        b64 = base64.b64encode(audio_bytes).decode()
        html = (
            f'<audio autoplay style="display:none">'
            f'<source src="data:audio/mp3;base64,{b64}" type="audio/mp3">'
            f'</audio>'
        )
        st.markdown(html, unsafe_allow_html=True)


def ollama_chat_stream(base_url: str, model: str, messages: list):
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


# ---------- Session state ----------
if "stress_prediction" not in st.session_state:
    st.session_state.stress_prediction = None
if "predict_chat_messages" not in st.session_state:
    st.session_state.predict_chat_messages = []
if "tts_pending" not in st.session_state:
    st.session_state.tts_pending = None


# ---------- Sidebar ----------
st.sidebar.title("Stress Prediction & Chat")

model_name = DEFAULT_MODEL
base_url = OLLAMA_BASE_URL

use_stream = st.sidebar.checkbox("Stream chatbot reply", value=True)
use_tts = st.sidebar.checkbox(
    "Read reply aloud (TTS)", value=True,
    disabled=not TTS_AVAILABLE,
    help="Requires edge-tts. Install with: pip install edge-tts"
)

if st.sidebar.button("Clear prediction & chat"):
    st.session_state.stress_prediction = None
    st.session_state.predict_chat_messages = []
    st.rerun()

st.sidebar.caption("Predict stress from data, then get a chatbot response.")

# ---------- Main ----------
st.title("Stress Prediction & Chat")
st.markdown(
    "Run the **CNN-GRU-Attention** stress model on physiological data, then send the predicted stress level to the driver stress chatbot for a supportive response."
)

# Play pending TTS on the render after reply is saved
if st.session_state.tts_pending:
    speak_text(st.session_state.tts_pending)
    st.session_state.tts_pending = None

if not PRED_AVAILABLE:
    st.error(f"Prediction module could not be loaded: {_PRED_ERROR}")
    st.info("Ensure `pred.py` and the stress model (e.g. `train4_outputs/CNN-GRU-Attention.h5`) and `train4_outputs/scaler.joblib` exist in the Proj folder.")
    st.stop()

# ---------- Load model once ----------
@st.cache_resource
def get_model_and_artifacts():
    try:
        model = load_stress_model()
        scaler = load_scaler()
        label_mapping = load_label_mapping()
        return model, scaler, label_mapping, None
    except Exception as e:
        return None, None, None, str(e)


model, scaler, label_mapping, load_err = get_model_and_artifacts()
if load_err:
    st.error(f"Could not load model or artifacts: {load_err}")
    st.stop()


def get_dataset_file_options():
    options = []
    if DEFAULT_CSV.is_file():
        options.append(("stress_features_all.csv (all data)", str(DEFAULT_CSV)))
    features_dir = _PROJ_ROOT / "features"
    if features_dir.is_dir():
        for p in sorted(features_dir.glob("*.csv")):
            options.append((p.name, str(p)))
    csv_data_dir = _PROJ_ROOT / "web" / "csv_data"
    if csv_data_dir.is_dir():
        for p in sorted(csv_data_dir.glob("*.csv")):
            label = f"{p.name} (web/csv_data)"
            if not any(p.name == o[0].split(" ")[0] for o in options):
                options.append((label, str(p)))
    return options


# ---------- Input source ----------
source = st.radio("Data source", ["Dataset file", "Input my data (CSV)"], horizontal=True)

prediction_made = False
stress_label = None
proba = None
true_label = None

if source == "Dataset file":
    dataset_options = get_dataset_file_options()
    if not dataset_options:
        st.warning("No dataset CSV files found. Use 'Input my data' and upload a CSV.")
    else:
        selected_label = st.selectbox("Choose a dataset file", options=[o[0] for o in dataset_options])
        csv_path = next(o[1] for o in dataset_options if o[0] == selected_label)
        try:
            import pandas as pd
            df_preview = pd.read_csv(csv_path)
            n_seq = max(0, len(df_preview) - SEQUENCE_LENGTH)
        except Exception:
            n_seq = 0
        if n_seq == 0:
            st.warning("This file has too few rows (need more than 10). Pick another file or upload your own CSV.")
        else:
            sample_options = [f"Sample {i + 1} (rows {i}–{i + SEQUENCE_LENGTH - 1})" for i in range(n_seq)]
            sample_choice = st.selectbox(
                "Which sample to predict",
                options=range(len(sample_options)),
                format_func=lambda i: sample_options[i],
            )
            if st.button("Run prediction on selected sample"):
                with st.spinner("Predicting..."):
                    try:
                        stress_label, proba, true_label = predict_from_dataset(
                            model, scaler, label_mapping,
                            csv_path=csv_path,
                            sequence_index=int(sample_choice),
                            use_test_split=False,
                        )
                        prediction_made = True
                    except Exception as e:
                        st.error(str(e))

elif source == "Input my data (CSV)":
    st.caption("Upload a CSV with the same feature columns as the training data. Need at least 10 rows.")
    uploaded = st.file_uploader("CSV file", type=["csv"])
    if uploaded is not None:
        try:
            import pandas as pd
            df_up = pd.read_csv(uploaded)
            expected = get_feature_columns()
            if df_up.shape[1] == len(expected) + 1 and "label" in df_up.columns:
                X_up = df_up.iloc[:, :-1]
            elif df_up.shape[1] >= len(expected):
                X_up = df_up.iloc[:, : len(expected)]
            else:
                st.warning(f"CSV should have at least {len(expected)} feature columns. Found {df_up.shape[1]}.")
                X_up = None
            if X_up is not None:
                st.dataframe(X_up.head(12))
                if st.button("Run prediction on uploaded data"):
                    with st.spinner("Predicting..."):
                        try:
                            stress_label, proba = predict_stress(model, scaler, label_mapping, X_up.values)
                            true_label = None
                            prediction_made = True
                        except Exception as e:
                            st.error(str(e))
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

# ---------- Persist prediction ----------
if prediction_made and stress_label is not None:
    st.session_state.stress_prediction = {
        "label": stress_label,
        "proba": proba,
        "true_label": true_label,
        "prompt_sent": None,
        "reply": None,
    }
    st.rerun()

# ---------- Show prediction result and chatbot ----------
last = st.session_state.stress_prediction
if last and last.get("label") is not None:
    stress_label = last["label"]
    proba = last.get("proba")
    true_label = last.get("true_label")

    st.subheader("Prediction result")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Predicted stress", stress_label.upper())
    with col2:
        conf = float(proba.max()) if proba is not None else 0
        st.metric("Confidence", f"{conf:.1%}")
    with col3:
        if true_label is not None:
            st.metric("True label (dataset)", str(true_label))
        else:
            st.metric("True label", "—")

    if label_mapping and proba is not None:
        st.caption("Probabilities: " + ", ".join(f"{label_mapping.get(i, i)}: {p:.2f}" for i, p in enumerate(proba)))

    st.divider()
    st.subheader("Send to chatbot")
    st.markdown("The chatbot will receive a generic statement based on the predicted stress level (e.g. *user stress is high, calm them down*).")
    user_override = st.text_area(
        "Optional: override user message to the chatbot",
        value="",
        placeholder="Leave empty to use the default message derived from the prediction.",
        height=80,
    )
    if st.button("Get chatbot response"):
        prompt_sent = user_override.strip() or (
            f"User's current stress level is predicted as {stress_label.upper()}. "
            "Please respond with calm, supportive advice suitable for a driver."
        )
        messages = [
            {"role": "system", "content": stress_level_to_prompt(stress_label)},
            {"role": "user", "content": prompt_sent},
        ]
        reply = None
        if use_stream:
            stream_placeholder = st.empty()
            full = []
            for delta in ollama_chat_stream(base_url, model_name, messages):
                if delta is None:
                    stream_placeholder.error("Could not reach Ollama. Is it running? Check the URL and model name.")
                    reply = "[Error]"
                    break
                full.append(delta)
                stream_placeholder.markdown("".join(full))
            if reply is None:
                reply = "".join(full) if full else ""
        else:
            reply = ollama_chat(base_url, model_name, messages, stream=False)
            if reply is None:
                st.error("Could not reach Ollama. Is it running? Check URL and model name.")
                reply = "[Error]"
            elif not (reply and reply.strip()):
                st.warning("Chatbot returned no response.")
                reply = "[No response]"
            else:
                st.markdown(reply)

        if not reply:
            reply = "[No response]"
        st.session_state.stress_prediction["prompt_sent"] = prompt_sent
        st.session_state.stress_prediction["reply"] = reply
        st.session_state.predict_chat_messages.append({"role": "user", "content": prompt_sent})
        st.session_state.predict_chat_messages.append({"role": "assistant", "content": reply})
        if use_tts:
            st.session_state.tts_pending = reply
        st.rerun()

    if last.get("prompt_sent"):
        st.caption("**Last prompt sent to chatbot:**")
        st.text(last["prompt_sent"])
    if last.get("reply"):
        st.caption("**Last chatbot reply:**")
        st.markdown(last["reply"])

# ---------- Conversation history ----------
if st.session_state.predict_chat_messages:
    st.divider()
    st.subheader("Recent prediction chat")
    for msg in st.session_state.predict_chat_messages[-6:]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
