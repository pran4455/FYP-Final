"""
Stress prediction using the CNN-GRU-Attention model and optional Ollama chatbot integration.
Supports:
- Predicting from provided physiological data (numpy array or CSV)
- Predicting for test cases from the dataset (stress_features_all.csv) by index or test split
- Feeding predicted stress level into the Ollama driver stress model with a generic statement.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import requests
import tensorflow as tf
# Use tf.keras only (avoid standalone keras 3 which can cause quantization_mode errors on load)
keras = tf.keras
layers = tf.keras.layers
import joblib

# ---------------------------------------------------------------------------
# Paths (relative to this file / Proj root)
# ---------------------------------------------------------------------------
PROJ_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = PROJ_ROOT / "train4_outputs"
DEFAULT_CSV = PROJ_ROOT / "stress_features_all.csv"
SEQUENCE_LENGTH = 10

# Model file: try stress_model_CNN-GRU-Attention.h5 in Proj root, else train4_outputs/CNN-GRU-Attention.h5
MODEL_PATHS = [
    PROJ_ROOT / "stress_model_CNN-GRU-Attention.h5",
    DEFAULT_MODEL_DIR / "CNN-GRU-Attention.h5",
    DEFAULT_MODEL_DIR / "BEST_MODEL.h5",
]


# ---------------------------------------------------------------------------
# Custom layer (must match train4.py for loading .h5)
# ---------------------------------------------------------------------------
class AttentionLayer(layers.Layer):
    """Custom Attention Layer for feature weighting (same as train4)."""

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="att_bias",
            shape=(1,),
            initializer="zeros",
            trainable=True,
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e = keras.backend.tanh(keras.backend.dot(x, self.W) + self.b)
        a = keras.backend.softmax(e, axis=1)
        output = x * a
        return output, a

    def get_config(self):
        return super(AttentionLayer, self).get_config()


# ---------------------------------------------------------------------------
# Load model, scaler, label mapping
# ---------------------------------------------------------------------------
def _find_model_path() -> Path:
    for p in MODEL_PATHS:
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"No stress model found. Tried: {[str(p) for p in MODEL_PATHS]}. "
        "Run train4.py to produce CNN-GRU-Attention.h5 in train4_outputs/."
    )


def _build_cnn_gru_attention(input_shape: Tuple[int, int], num_classes: int, dropout_rate: float = 0.3):
    """Build CNN-GRU-Attention architecture (same as train4) for weights-only loading."""
    input_layer = layers.Input(shape=input_shape)
    x = layers.Conv1D(64, 3, padding="same", activation="relu", name="conv1d_1")(input_layer)
    x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.Dropout(dropout_rate, name="dropout_1")(x)
    x = layers.Conv1D(128, 3, padding="same", activation="relu", name="conv1d_2")(x)
    x = layers.BatchNormalization(name="bn_2")(x)
    x = layers.MaxPooling1D(pool_size=2, name="maxpool_1")(x)
    x = layers.Dropout(dropout_rate, name="dropout_2")(x)
    x = layers.GRU(128, return_sequences=True, activation="tanh", recurrent_activation="sigmoid", name="gru_1")(x)
    x = layers.BatchNormalization(name="bn_3")(x)
    x = layers.Dropout(dropout_rate, name="dropout_3")(x)
    x = layers.GRU(64, return_sequences=True, activation="tanh", recurrent_activation="sigmoid", name="gru_2")(x)
    x = layers.BatchNormalization(name="bn_4")(x)
    x = layers.Dropout(dropout_rate, name="dropout_4")(x)
    attention_output, _ = AttentionLayer(name="attention")(x)
    x = layers.Flatten(name="flatten")(attention_output)
    x = layers.Dense(128, activation="relu", name="dense_1")(x)
    x = layers.Dropout(dropout_rate, name="dropout_5")(x)
    x = layers.Dense(64, activation="relu", name="dense_2")(x)
    x = layers.Dropout(dropout_rate, name="dropout_6")(x)
    output = layers.Dense(num_classes, activation="softmax", dtype="float32", name="output")(x)
    return keras.Model(inputs=input_layer, outputs=output, name="CNN_GRU_Attention")


def load_stress_model(
    model_path: Optional[Union[str, Path]] = None,
    model_dir: Optional[Union[str, Path]] = None,
):
    """Load the CNN-GRU-Attention model with custom AttentionLayer."""
    if model_path is None:
        model_path = _find_model_path()
    else:
        model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    custom_objects = {"AttentionLayer": AttentionLayer}
    try:
        # compile=False avoids restoring optimizer/dtype policy that can cause
        # 'str' object has no attribute 'quantization_mode' (mixed precision / Keras 3)
        model = keras.models.load_model(
            str(model_path),
            custom_objects=custom_objects,
            compile=False,
        )
        return model
    except Exception as e:
        err_msg = str(e)
        if "quantization_mode" in err_msg or "quantization" in err_msg.lower():
            # Fallback: build architecture and load weights only (avoids config deserialization)
            model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
            scaler_path = model_dir / "scaler.joblib"
            if not scaler_path.is_file():
                raise
            scaler = joblib.load(scaler_path)
            n_features = getattr(scaler, "n_features_in_", None) or getattr(scaler, "n_features_in", None)
            if n_features is None:
                raise ValueError("Cannot infer n_features from scaler for fallback load") from e
            input_shape = (SEQUENCE_LENGTH, int(n_features))
            num_classes = 3  # high, low, medium
            model = _build_cnn_gru_attention(input_shape, num_classes)
            model.load_weights(str(model_path))
            return model
        raise


def load_scaler(model_dir: Optional[Union[str, Path]] = None) -> object:
    """Load the StandardScaler saved during training."""
    d = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
    p = d / "scaler.joblib"
    if not p.is_file():
        raise FileNotFoundError(f"Scaler not found: {p}. Run train4.py first.")
    return joblib.load(p)


def load_label_mapping(model_dir: Optional[Union[str, Path]] = None) -> Dict[int, str]:
    """Load label mapping: numeric index -> text label (e.g. 0 -> 'high')."""
    d = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
    p = d / "label_mapping.txt"
    if not p.is_file():
        # Fallback: train4 uses high->0, low->1, medium->2
        return {0: "high", 1: "low", 2: "medium"}
    mapping = {}
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "->" in line and not line.startswith("Label"):
                parts = line.split("->")
                if len(parts) == 2:
                    text_label = parts[0].strip()
                    num_label = int(parts[1].strip())
                    mapping[num_label] = text_label
    if not mapping:
        mapping = {0: "high", 1: "low", 2: "medium"}
    return mapping


def get_feature_columns(csv_path: Optional[Union[str, Path]] = None) -> List[str]:
    """Return list of feature column names (all columns except last)."""
    path = Path(csv_path) if csv_path else DEFAULT_CSV
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path, nrows=1)
    return df.iloc[:, :-1].columns.tolist()


# ---------------------------------------------------------------------------
# Preprocessing (match train4)
# ---------------------------------------------------------------------------
def _prepare_features(
    X: np.ndarray,
    scaler: object,
    sequence_length: int = SEQUENCE_LENGTH,
) -> np.ndarray:
    """X: (n_samples, n_features) or (n_features,). Returns (1, seq_len, n_features)."""
    if X.ndim == 1:
        X = X.reshape(1, -1)
    X_df = pd.DataFrame(X)
    X_df = X_df.fillna(X_df.mean(numeric_only=True))
    X = X_df.values.astype(np.float64)
    X_scaled = scaler.transform(X)
    if len(X_scaled) < sequence_length:
        # Repeat the last row to get enough timesteps (or repeat first row)
        pad = np.tile(X_scaled[-1:], (sequence_length - len(X_scaled), 1))
        X_scaled = np.vstack([X_scaled, pad])
    if len(X_scaled) > sequence_length:
        X_scaled = X_scaled[-sequence_length:]
    X_seq = X_scaled.reshape(1, sequence_length, X_scaled.shape[1])
    return X_seq.astype(np.float32)


def _sequences_from_df(
    df: pd.DataFrame,
    scaler: object,
    sequence_length: int = SEQUENCE_LENGTH,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build (X_sequences, y_sequences) from full dataframe (features + label)."""
    X_df = df.iloc[:, :-1].fillna(df.iloc[:, :-1].mean(numeric_only=True))
    X = X_df.values
    y = df.iloc[:, -1].values
    X_scaled = scaler.transform(X)
    seqs = []
    for i in range(len(X_scaled) - sequence_length):
        seqs.append(X_scaled[i : i + sequence_length])
    X_seq = np.array(seqs, dtype=np.float32)
    y_seq = y[sequence_length:]
    return X_seq, y_seq


# ---------------------------------------------------------------------------
# Prediction API
# ---------------------------------------------------------------------------
def predict_stress(
    model: keras.Model,
    scaler: object,
    label_mapping: Dict[int, str],
    X: Union[np.ndarray, pd.DataFrame],
    sequence_length: int = SEQUENCE_LENGTH,
) -> Tuple[str, np.ndarray]:
    """
    Predict stress level from feature matrix or single row.
    X: shape (n_rows, n_features) or (n_features,) or DataFrame (features only).
    Returns (label_str, proba) e.g. ('high', array([0.1, 0.2, 0.7])).
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    X_seq = _prepare_features(X, scaler, sequence_length)
    proba = model.predict(X_seq, verbose=0)[0]
    pred_idx = int(np.argmax(proba))
    label_str = label_mapping.get(pred_idx, f"class_{pred_idx}")
    return label_str, proba


def predict_from_dataset(
    model: keras.Model,
    scaler: object,
    label_mapping: Dict[int, str],
    csv_path: Optional[Union[str, Path]] = None,
    sequence_index: Optional[int] = None,
    use_test_split: bool = False,
    random_state: int = 42,
) -> Tuple[str, np.ndarray, Optional[str]]:
    """
    Predict using the dataset CSV.
    - If sequence_index is set: use that sequence index (0 to N-1).
    - If use_test_split: use the same test split as train4 (train_test_split 0.3 then 0.5), then take one sample (or first).
    - Otherwise: build all sequences and take sequence_index (default 0).
    Returns (label_str, proba, optional_true_label).
    """
    path = Path(csv_path) if csv_path else DEFAULT_CSV
    if not path.is_file():
        raise FileNotFoundError(f"Dataset CSV not found: {path}")
    df = pd.read_csv(path)
    X_seq, y_seq = _sequences_from_df(df, scaler, SEQUENCE_LENGTH)

    true_label = None
    if use_test_split:
        from sklearn.model_selection import train_test_split
        _, X_temp, _, y_temp = train_test_split(
            X_seq, y_seq, test_size=0.3, random_state=random_state, stratify=y_seq
        )
        _, X_test, _, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp
        )
        idx = sequence_index if sequence_index is not None else 0
        idx = min(max(0, idx), len(X_test) - 1)
        X_sample = X_test[idx : idx + 1]
        true_label = y_test[idx] if hasattr(y_test[idx], "strip") else str(y_test[idx])
    else:
        idx = sequence_index if sequence_index is not None else 0
        idx = min(max(0, idx), len(X_seq) - 1)
        X_sample = X_seq[idx : idx + 1]
        true_label = y_seq[idx] if hasattr(y_seq[idx], "strip") else str(y_seq[idx])

    proba = model.predict(X_sample, verbose=0)[0]
    pred_idx = int(np.argmax(proba))
    label_str = label_mapping.get(pred_idx, f"class_{pred_idx}")
    return label_str, proba, true_label


# ---------------------------------------------------------------------------
# Ollama: feed stress prediction into the driver stress chatbot
# ---------------------------------------------------------------------------
def stress_level_to_prompt(stress_label: str) -> str:
    """Turn predicted stress level into a short statement for the chatbot."""
    stress_lower = stress_label.lower()
    if stress_lower == "high":
        return (
            "The driver stress model predicts stress is HIGH. "
            "The user is stressed; please respond with calm, safety-focused support and suggest calming techniques."
        )
    if stress_lower == "medium":
        return (
            "The driver stress model predicts stress is MEDIUM. "
            "The user may be under moderate stress; offer brief reassurance and safe driving reminders."
        )
    return (
        "The driver stress model predicts stress is LOW. "
        "The user appears calm; you may offer brief encouragement or general safety tips if appropriate."
    )


def ollama_chat(
    base_url: str,
    model_name: str,
    messages: List[Dict[str, str]],
    stream: bool = False,
    timeout: int = 120,
) -> Optional[str]:
    """Call Ollama /api/chat. If stream=True, returns full concatenated content."""
    url = f"{base_url.rstrip('/')}/api/chat"
    payload = {"model": model_name, "messages": messages, "stream": stream}
    try:
        r = requests.post(url, json=payload, stream=stream, timeout=timeout)
        r.raise_for_status()
        if stream:
            full = []
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    import json as _json
                    chunk = _json.loads(line)
                    content = (chunk.get("message") or {}).get("content") or ""
                    if content:
                        full.append(content)
                    if chunk.get("done"):
                        break
                except Exception:
                    continue
            return "".join(full)
        data = r.json()
        return (data.get("message") or {}).get("content") or ""
    except requests.exceptions.RequestException:
        return None


def predict_and_chat(
    stress_label: str,
    base_url: str = "http://localhost:11434",
    model_name: str = "driver_stress_ai",
    user_override: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    """
    Build the generic statement from stress prediction and get Ollama response.
    Returns (prompt_sent_to_bot, reply_text or None on error).
    """
    system_or_context = stress_level_to_prompt(stress_label)
    user_content = user_override or (
        f"User's current stress level is predicted as {stress_label.upper()}. "
        "Please respond with calm, supportive advice suitable for a driver."
    )
    messages = [
        {"role": "system", "content": system_or_context},
        {"role": "user", "content": user_content},
    ]
    reply = ollama_chat(base_url, model_name, messages, stream=False)
    return user_content, reply


# ---------------------------------------------------------------------------
# CLI / script usage
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Predict driver stress and optionally get Ollama response.")
    parser.add_argument("--csv", default=None, help="Path to stress_features_all.csv (default: Proj/stress_features_all.csv)")
    parser.add_argument("--index", type=int, default=0, help="Sequence index for dataset prediction")
    parser.add_argument("--test-split", action="store_true", help="Use train4 test split for dataset prediction")
    parser.add_argument("--no-ollama", action="store_true", help="Skip Ollama; only print prediction")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--ollama-model", default="driver_stress_ai")
    args = parser.parse_args()

    model_dir = DEFAULT_MODEL_DIR
    model = load_stress_model()
    scaler = load_scaler(model_dir)
    label_mapping = load_label_mapping(model_dir)

    csv_path = args.csv or DEFAULT_CSV
    if not Path(csv_path).is_file():
        print(f"CSV not found: {csv_path}. Exiting.")
        return

    label_str, proba, true_label = predict_from_dataset(
        model, scaler, label_mapping,
        csv_path=csv_path,
        sequence_index=args.index,
        use_test_split=args.test_split,
    )
    print(f"Predicted stress: {label_str} (probabilities: {proba.round(3).tolist()})")
    if true_label is not None:
        print(f"True label: {true_label}")

    if not args.no_ollama:
        prompt_sent, reply = predict_and_chat(
            label_str, base_url=args.ollama_url, model_name=args.ollama_model
        )
        print("\nPrompt sent to chatbot:", prompt_sent)
        if reply is not None:
            print("\nOllama reply:", reply)
        else:
            print("\nOllama unreachable or error.")


if __name__ == "__main__":
    main()
