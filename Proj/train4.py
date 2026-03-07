"""
Advanced Deep Learning Architectures for Driver Stress Detection (train4_gpu_final.py)
GPU-Optimized Version for TensorFlow 2.10 with RTX 4060
FINAL VERSION - With Architecture Diagrams and Organized Outputs

Features:
- 5 advanced architectures with GPU optimization
- Automatic architecture diagram generation
- All outputs organized in train4_outputs/ folder
"""

from paper_diagram_functions import (
    plot_training_history,
    plot_roc_curves,
    plot_confusion_matrix_paper,
    plot_model_metrics_comparison,
    plot_architecture_complexity,
    plot_class_distribution,
    plot_per_class_metrics,
    generate_paper_summary_report,
    plot_precision_recall_curves
)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model  # For architecture diagrams
import joblib
import warnings
from datetime import datetime
import os
from pathlib import Path

warnings.filterwarnings("ignore")

# Create output directory
OUTPUT_DIR = Path("train4_outputs_TESTRUN")
OUTPUT_DIR.mkdir(exist_ok=True)
print(f"Output directory created: {OUTPUT_DIR}")

# ============================================================================
# SET RANDOM SEEDS FOR DETERMINISM
# ============================================================================
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================================
# GPU CONFIGURATION FOR MAXIMUM PERFORMANCE
# ============================================================================

def configure_gpu():
    """
    Configure GPU for maximum performance on RTX 4060
    - Enables mixed precision training (FP16)
    - Enables memory growth to avoid OOM errors
    - Disables XLA (causes libdevice issues on Windows)
    """
    print("=" * 80)
    print("GPU CONFIGURATION")
    print("=" * 80)
    
    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth (allocate as needed, not all at once)
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            print(f"✓ GPU detected: {len(gpus)} device(s)")
            print(f"✓ GPU Name: {gpus[0].name}")
            
            # Enable mixed precision training (FP16 for speed, FP32 for stability)
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("✓ Mixed Precision (FP16) enabled for 2-3x speedup")
            
            # DISABLE XLA - causes libdevice issues on Windows
            # tf.config.optimizer.set_jit(True)
            print("✓ XLA disabled (Windows compatibility)")
            
            # Set TensorFlow to use GPU deterministically
            os.environ['TF_DETERMINISTIC_OPS'] = '0'
            
            print("✓ GPU configuration complete!")
            print("=" * 80 + "\n")
            
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("⚠ No GPU detected. Training will use CPU (much slower).")
        print("=" * 80 + "\n")

    return len(gpus) > 0


# Call GPU configuration at module load
GPU_AVAILABLE = configure_gpu()


class AttentionLayer(layers.Layer):
    """Custom Attention Layer for feature weighting"""

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

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2])


def load_and_preprocess_data(csv_path):
    """Load and preprocess feature data with text label handling"""
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    print(f"Missing values before handling: {df.isnull().sum().sum()}")
    
    # Extract labels FIRST (before filling NaN)
    y_raw = df.iloc[:, -1].values
    
    # Check if labels are text or numbers
    if isinstance(y_raw[0], str):
        print(f"Text labels detected: {np.unique(y_raw)}")
        
        # Create mapping from text to numbers
        label_mapping = {}
        unique_labels = sorted(set(y_raw))
        
        for idx, label in enumerate(unique_labels):
            label_mapping[label] = idx
        
        print(f"Label mapping: {label_mapping}")
        
        # Encode text labels to numbers
        y = np.array([label_mapping[label] for label in y_raw])
        
        # Save label mapping
        with open(OUTPUT_DIR / "label_mapping.txt", "w", encoding='utf-8') as f:
            f.write("Label Mapping:\n")
            f.write("=" * 40 + "\n")
            for text_label, numeric_label in label_mapping.items():
                f.write(f"{text_label} -> {numeric_label}\n")

    else:
        # Labels are already numeric
        y = y_raw
    
    # Now handle features (all columns except last)
    X_df = df.iloc[:, :-1]
    
    # Fill NaN values with column means (only for numeric columns)
    X_df = X_df.fillna(X_df.mean(numeric_only=True))
    
    X = X_df.values
    
    print(f"Missing values after handling: {X_df.isnull().sum().sum()}")
    print(f"Data shape: {X.shape}, Labels shape: {y.shape}")
    print(f"Unique classes: {np.unique(y)}")
    print(f"Class distribution: {np.bincount(y)}")

    return X, y


def reshape_for_lstm(X, sequence_length=10):
    """Reshape 2D feature matrix into 3D sequences for LSTM"""
    X_sequences = []

    for i in range(len(X) - sequence_length):
        X_sequences.append(X[i : i + sequence_length])

    X_sequences = np.array(X_sequences)
    print(f"Reshaped data for LSTM: {X_sequences.shape}")
    return X_sequences


def apply_data_augmentation(X_train, y_train):
    """Apply SMOTE for handling class imbalance"""
    print("Applying data augmentation...")
    from imblearn.over_sampling import SMOTE

    original_shape = X_train.shape
    X_train_flat = X_train.reshape(X_train.shape[0], -1)

    try:
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(
            X_train_flat, y_train
        )
        X_train_resampled = X_train_resampled.reshape(
            X_train_resampled.shape[0], original_shape[1], original_shape[2]
        )
        print(
            f"After SMOTE - X: {X_train_resampled.shape}, y: {y_train_resampled.shape}"
        )
        return X_train_resampled, y_train_resampled
    except Exception as e:
        print(f"SMOTE failed: {e}. Proceeding without augmentation.")
        return X_train, y_train


# ============================================================================
# ARCHITECTURE 1: CNN-BiLSTM-Attention
# ============================================================================
def build_cnn_bilstm_attention_model(input_shape, num_classes, dropout_rate=0.3):
    """CNN-BiLSTM-Attention Architecture"""
    input_layer = layers.Input(shape=input_shape)
    
    # CNN Layers
    x = layers.Conv1D(64, 3, padding="same", activation="relu", name="conv1d_1")(input_layer)
    x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.Dropout(dropout_rate, name="dropout_1")(x)

    x = layers.Conv1D(128, 3, padding="same", activation="relu", name="conv1d_2")(x)
    x = layers.BatchNormalization(name="bn_2")(x)
    x = layers.MaxPooling1D(pool_size=2, name="maxpool_1")(x)
    x = layers.Dropout(dropout_rate, name="dropout_2")(x)

    # BiLSTM Layers
    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True, activation='tanh', recurrent_activation='sigmoid', name="lstm_1"),
        name="bilstm_1"
    )(x)
    x = layers.BatchNormalization(name="bn_3")(x)
    x = layers.Dropout(dropout_rate, name="dropout_3")(x)

    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, activation='tanh', recurrent_activation='sigmoid', name="lstm_2"),
        name="bilstm_2"
    )(x)
    x = layers.BatchNormalization(name="bn_4")(x)
    x = layers.Dropout(dropout_rate, name="dropout_4")(x)

    # Attention Layer
    attention_output, _ = AttentionLayer(name="attention")(x)
    x = layers.Flatten(name="flatten")(attention_output)

    # Dense Layers
    x = layers.Dense(128, activation="relu", name="dense_1")(x)
    x = layers.Dropout(dropout_rate, name="dropout_5")(x)
    x = layers.Dense(64, activation="relu", name="dense_2")(x)
    x = layers.Dropout(dropout_rate, name="dropout_6")(x)
    output = layers.Dense(num_classes, activation="softmax", dtype='float32', name="output")(x)

    model = keras.Model(inputs=input_layer, outputs=output, name="CNN_BiLSTM_Attention")
    return model


# ============================================================================
# ARCHITECTURE 2: Parallel CNN-LSTM
# ============================================================================
def build_parallel_cnn_lstm_model(input_shape, num_classes, dropout_rate=0.3):
    """Parallel CNN-LSTM Architecture"""
    input_layer = layers.Input(shape=input_shape)

    # Stream 1
    stream1 = layers.Conv1D(64, 3, padding="same", activation="relu", name="s1_conv1d_1")(input_layer)
    stream1 = layers.BatchNormalization(name="s1_bn_1")(stream1)
    stream1 = layers.MaxPooling1D(2, name="s1_maxpool_1")(stream1)
    stream1 = layers.Dropout(dropout_rate, name="s1_dropout_1")(stream1)
    stream1 = layers.LSTM(128, return_sequences=True, activation='tanh', recurrent_activation='sigmoid', name="s1_lstm_1")(stream1)
    stream1 = layers.Dropout(dropout_rate, name="s1_dropout_2")(stream1)
    stream1 = layers.LSTM(64, return_sequences=False, activation='tanh', recurrent_activation='sigmoid', name="s1_lstm_2")(stream1)
    stream1 = layers.Dropout(dropout_rate, name="s1_dropout_3")(stream1)

    # Stream 2
    stream2 = layers.Conv1D(64, 5, padding="same", activation="relu", name="s2_conv1d_1")(input_layer)
    stream2 = layers.BatchNormalization(name="s2_bn_1")(stream2)
    stream2 = layers.MaxPooling1D(2, name="s2_maxpool_1")(stream2)
    stream2 = layers.Dropout(dropout_rate, name="s2_dropout_1")(stream2)
    stream2 = layers.LSTM(128, return_sequences=True, activation='tanh', recurrent_activation='sigmoid', name="s2_lstm_1")(stream2)
    stream2 = layers.Dropout(dropout_rate, name="s2_dropout_2")(stream2)
    stream2 = layers.LSTM(64, return_sequences=False, activation='tanh', recurrent_activation='sigmoid', name="s2_lstm_2")(stream2)
    stream2 = layers.Dropout(dropout_rate, name="s2_dropout_3")(stream2)

    # Concatenate
    concatenated = layers.Concatenate(name="concat")([stream1, stream2])
    x = layers.Dense(128, activation="relu", name="dense_1")(concatenated)
    x = layers.Dropout(dropout_rate, name="dropout_final_1")(x)
    x = layers.Dense(64, activation="relu", name="dense_2")(x)
    x = layers.Dropout(dropout_rate, name="dropout_final_2")(x)
    output = layers.Dense(num_classes, activation="softmax", dtype='float32', name="output")(x)

    model = keras.Model(inputs=input_layer, outputs=output, name="Parallel_CNN_LSTM")
    return model


# ============================================================================
# ARCHITECTURE 3: ResNet1D with LSTM
# ============================================================================
def build_resnet1d_lstm_model(input_shape, num_classes, dropout_rate=0.3):
    """ResNet1D-LSTM Architecture"""
    def residual_block(x, filters, block_num, kernel_size=3):
        shortcut = x
        x = layers.Conv1D(filters, kernel_size, padding="same", activation="relu", name=f"res{block_num}_conv1")(x)
        x = layers.BatchNormalization(name=f"res{block_num}_bn1")(x)
        x = layers.Dropout(dropout_rate, name=f"res{block_num}_dropout1")(x)
        x = layers.Conv1D(filters, kernel_size, padding="same", name=f"res{block_num}_conv2")(x)
        x = layers.BatchNormalization(name=f"res{block_num}_bn2")(x)
        
        if shortcut.shape[-1] != filters:
            shortcut = layers.Conv1D(filters, 1, padding="same", name=f"res{block_num}_shortcut")(shortcut)
        
        x = layers.Add(name=f"res{block_num}_add")([x, shortcut])
        x = layers.Activation("relu", name=f"res{block_num}_relu")(x)
        return x

    input_layer = layers.Input(shape=input_shape)
    x = layers.Conv1D(32, 7, padding="same", activation="relu", name="init_conv")(input_layer)
    x = layers.BatchNormalization(name="init_bn")(x)

    x = residual_block(x, 64, block_num=1)
    x = residual_block(x, 64, block_num=2)
    x = layers.MaxPooling1D(2, name="maxpool_1")(x)
    x = residual_block(x, 128, block_num=3)
    x = residual_block(x, 128, block_num=4)
    x = layers.MaxPooling1D(2, name="maxpool_2")(x)

    x = layers.LSTM(128, return_sequences=True, activation='tanh', recurrent_activation='sigmoid', name="lstm_1")(x)
    x = layers.Dropout(dropout_rate, name="dropout_lstm_1")(x)
    x = layers.LSTM(64, return_sequences=False, activation='tanh', recurrent_activation='sigmoid', name="lstm_2")(x)
    x = layers.Dropout(dropout_rate, name="dropout_lstm_2")(x)

    x = layers.Dense(128, activation="relu", name="dense_1")(x)
    x = layers.Dropout(dropout_rate, name="dropout_1")(x)
    x = layers.Dense(64, activation="relu", name="dense_2")(x)
    x = layers.Dropout(dropout_rate, name="dropout_2")(x)
    output = layers.Dense(num_classes, activation="softmax", dtype='float32', name="output")(x)

    model = keras.Model(inputs=input_layer, outputs=output, name="ResNet1D_LSTM")
    return model


# ============================================================================
# ARCHITECTURE 4: Inception1D-LSTM
# ============================================================================
def build_inception1d_lstm_model(input_shape, num_classes, dropout_rate=0.3):
    """Inception1D-LSTM Architecture"""
    def inception_block(x, filters, block_num):
        branch1 = layers.Conv1D(filters // 4, 1, padding="same", activation="relu", name=f"inc{block_num}_b1_conv")(x)
        branch2 = layers.Conv1D(filters // 4, 1, padding="same", activation="relu", name=f"inc{block_num}_b2_conv1")(x)
        branch2 = layers.Conv1D(filters // 4, 3, padding="same", activation="relu", name=f"inc{block_num}_b2_conv2")(branch2)
        branch3 = layers.Conv1D(filters // 4, 1, padding="same", activation="relu", name=f"inc{block_num}_b3_conv1")(x)
        branch3 = layers.Conv1D(filters // 4, 5, padding="same", activation="relu", name=f"inc{block_num}_b3_conv2")(branch3)
        branch4 = layers.MaxPooling1D(3, strides=1, padding="same", name=f"inc{block_num}_b4_pool")(x)
        branch4 = layers.Conv1D(filters // 4, 1, padding="same", activation="relu", name=f"inc{block_num}_b4_conv")(branch4)
        x = layers.Concatenate(name=f"inc{block_num}_concat")([branch1, branch2, branch3, branch4])
        return x

    input_layer = layers.Input(shape=input_shape)
    x = layers.Conv1D(32, 7, padding="same", activation="relu", name="init_conv")(input_layer)
    x = layers.BatchNormalization(name="init_bn")(x)

    x = inception_block(x, 128, block_num=1)
    x = layers.BatchNormalization(name="inc1_bn")(x)
    x = layers.MaxPooling1D(2, name="maxpool_1")(x)
    x = inception_block(x, 256, block_num=2)
    x = layers.BatchNormalization(name="inc2_bn")(x)
    x = layers.MaxPooling1D(2, name="maxpool_2")(x)

    x = layers.LSTM(128, return_sequences=True, activation='tanh', recurrent_activation='sigmoid', name="lstm_1")(x)
    x = layers.Dropout(dropout_rate, name="dropout_lstm_1")(x)
    x = layers.LSTM(64, return_sequences=False, activation='tanh', recurrent_activation='sigmoid', name="lstm_2")(x)
    x = layers.Dropout(dropout_rate, name="dropout_lstm_2")(x)

    x = layers.Dense(128, activation="relu", name="dense_1")(x)
    x = layers.Dropout(dropout_rate, name="dropout_1")(x)
    x = layers.Dense(64, activation="relu", name="dense_2")(x)
    x = layers.Dropout(dropout_rate, name="dropout_2")(x)
    output = layers.Dense(num_classes, activation="softmax", dtype='float32', name="output")(x)

    model = keras.Model(inputs=input_layer, outputs=output, name="Inception1D_LSTM")
    return model


# ============================================================================
# ARCHITECTURE 5: CNN-GRU-Attention
# ============================================================================
def build_cnn_gru_attention_model(input_shape, num_classes, dropout_rate=0.3):
    """CNN-GRU-Attention Architecture"""
    input_layer = layers.Input(shape=input_shape)
    
    # CNN Layers
    x = layers.Conv1D(64, 3, padding="same", activation="relu", name="conv1d_1")(input_layer)
    x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.Dropout(dropout_rate, name="dropout_1")(x)

    x = layers.Conv1D(128, 3, padding="same", activation="relu", name="conv1d_2")(x)
    x = layers.BatchNormalization(name="bn_2")(x)
    x = layers.MaxPooling1D(pool_size=2, name="maxpool_1")(x)
    x = layers.Dropout(dropout_rate, name="dropout_2")(x)

    # GRU Layers
    x = layers.GRU(128, return_sequences=True, activation='tanh', recurrent_activation='sigmoid', name="gru_1")(x)
    x = layers.BatchNormalization(name="bn_3")(x)
    x = layers.Dropout(dropout_rate, name="dropout_3")(x)

    x = layers.GRU(64, return_sequences=True, activation='tanh', recurrent_activation='sigmoid', name="gru_2")(x)
    x = layers.BatchNormalization(name="bn_4")(x)
    x = layers.Dropout(dropout_rate, name="dropout_4")(x)

    # Attention Layer
    attention_output, _ = AttentionLayer(name="attention")(x)
    x = layers.Flatten(name="flatten")(attention_output)

    # Dense Layers
    x = layers.Dense(128, activation="relu", name="dense_1")(x)
    x = layers.Dropout(dropout_rate, name="dropout_5")(x)
    x = layers.Dense(64, activation="relu", name="dense_2")(x)
    x = layers.Dropout(dropout_rate, name="dropout_6")(x)
    output = layers.Dense(num_classes, activation="softmax", dtype='float32', name="output")(x)

    model = keras.Model(inputs=input_layer, outputs=output, name="CNN_GRU_Attention")
    return model


def plot_architecture_diagram(model, model_name):
    """Generate and save architecture diagram"""
    try:
        diagram_path = OUTPUT_DIR / f"{model_name}_architecture.png"
        plot_model(
            model,
            to_file=str(diagram_path),
            show_shapes=True,
            show_layer_names=True,
            show_layer_activations=False,
            rankdir='TB',  # Top to Bottom
            expand_nested=True,
            dpi=300,
            show_dtype=False
        )
        print(f"✓ Architecture diagram saved: {diagram_path}")
        return True
    except Exception as e:
        print(f"✗ Could not generate diagram for {model_name}: {e}")
        return False


def train_model(X_train, y_train, X_val, y_val, model, model_name, num_epochs=100):
    """Train a single model with GPU-optimized settings"""
    print(f"\n{'='*80}")
    print(f"Training {model_name}")
    print(f"{'='*80}")

    optimizer = Adam(learning_rate=0.001)
    
    if GPU_AVAILABLE:
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1,
    )

    checkpoint = ModelCheckpoint(
        str(OUTPUT_DIR / f"{model_name}.h5"),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=0,
    )

    batch_size = 64 if GPU_AVAILABLE else 32

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=num_epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr, checkpoint],
        verbose=1,
    )

    return model, history


def evaluate_model(model, X_test, y_test, model_name, class_names=None):
    """Evaluate model performance"""
    print(f"\nEvaluating {model_name}...")

    batch_size = 128 if GPU_AVAILABLE else 32
    
    y_pred_proba = model.predict(X_test, batch_size=batch_size, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"{model_name} - Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")

    report = classification_report(y_test, y_pred, target_names=class_names)
    
    # Save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{model_name}_confusion_matrix.png", dpi=300)
    plt.close()

    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "f1_score": f1,
        "predictions": y_pred,
        "probabilities": y_pred_proba,
        "report": report,
        "confusion_matrix": cm,
    }


def plot_model_comparison(results_dict):
    """Plot comparison of all models"""
    if not results_dict:
        print("No results to plot")
        return
    
    models = list(results_dict.keys())
    accuracies = [results_dict[m]["accuracy"] for m in models]
    f1_scores = [results_dict[m]["f1_score"] for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].bar(models, accuracies, color="skyblue", edgecolor="navy")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Model Accuracy Comparison")
    axes[0].set_ylim([0.50, 1.0])
    for i, v in enumerate(accuracies):
        axes[0].text(i, v + 0.01, f"{v:.4f}", ha="center", fontweight="bold")
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].bar(models, f1_scores, color="lightcoral", edgecolor="darkred")
    axes[1].set_ylabel("F1-Score")
    axes[1].set_title("Model F1-Score Comparison")
    axes[1].set_ylim([0.50, 1.0])
    for i, v in enumerate(f1_scores):
        axes[1].text(i, v + 0.01, f"{v:.4f}", ha="center", fontweight="bold")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_comparison.png", dpi=300)
    print(f"✓ Model comparison plot saved: {OUTPUT_DIR / 'model_comparison.png'}")
    plt.close()


def main():
    """Main training pipeline with GPU optimizations"""
    print("=" * 80)
    print("GPU-Optimized Driver Stress Detection Training")
    print("=" * 80)
    PAPER_OUTPUT_DIR = "train4_paper_fol"
    
    if GPU_AVAILABLE:
        print("✓ Training with GPU acceleration (RTX 4060)")
        print("✓ Expected speedup: 5-10x faster than CPU")
    else:
        print("⚠ Training with CPU (no GPU detected)")
    print(f"✓ All outputs will be saved to: {OUTPUT_DIR}")
    print("=" * 80 + "\n")

    # Load and preprocess data
    X, y = load_and_preprocess_data("stress_features_all.csv")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    sequence_length = 10
    X_sequences = reshape_for_lstm(X_scaled, sequence_length=sequence_length)
    y_sequences = y[sequence_length:]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_sequences, y_sequences, test_size=0.3, random_state=42, stratify=y_sequences
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"Training set: {X_train.shape}, {y_train.shape}")
    print(f"Validation set: {X_val.shape}, {y_val.shape}")
    print(f"Test set: {X_test.shape}, {y_test.shape}")

    X_train, y_train = apply_data_augmentation(X_train, y_train)

    num_classes = len(np.unique(y_train))
    input_shape = (X_train.shape[1], X_train.shape[2])

    # Define all models
    models_config = {
        "CNN-BiLSTM-Attention": build_cnn_bilstm_attention_model,
        "Parallel-CNN-LSTM": build_parallel_cnn_lstm_model,
        "ResNet1D-LSTM": build_resnet1d_lstm_model,
        "Inception1D-LSTM": build_inception1d_lstm_model,
        "CNN-GRU-Attention": build_cnn_gru_attention_model,
    }

    # Train all models
    trained_models = {}
    results = {}
    histories = {}

    for model_name, model_builder in models_config.items():
        try:
            print(f"\n{'='*80}")
            print(f"Building {model_name}")
            print(f"{'='*80}")
            
            model = model_builder(input_shape, num_classes, dropout_rate=0.3)
            print(f"✓ Model built successfully. Total parameters: {model.count_params():,}")
            
            # Generate architecture diagram
            plot_architecture_diagram(model, model_name)
            
            # Save model summary
            with open(OUTPUT_DIR / f"{model_name}_summary.txt", "w") as f:
                model.summary(print_fn=lambda x: f.write(x + '\n'))
            print(f"✓ Model summary saved: {OUTPUT_DIR / f'{model_name}_summary.txt'}")
            
            trained_model, history = train_model(
                X_train, y_train, X_val, y_val, model, model_name, num_epochs=100
            )
            trained_models[model_name] = trained_model
            histories[model_name] = history

            results[model_name] = evaluate_model(
                trained_model, X_test, y_test, model_name,
                class_names=[f"Stress_Level_{i}" for i in range(num_classes)]
            )

        except Exception as e:
            print(f"✗ Error training {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # Plot comparison if we have results
    if results:
        plot_model_comparison(results)

        # Print detailed results
        print("\n" + "=" * 80)
        print("DETAILED RESULTS COMPARISON")
        print("=" * 80)

        best_model_name = max(results, key=lambda x: results[x]["accuracy"])
        best_accuracy = results[best_model_name]["accuracy"]

        for model_name, result in results.items():
            print(f"\n{model_name}:")
            print(f"  Accuracy: {result['accuracy']:.4f}")
            print(f"  F1-Score: {result['f1_score']:.4f}")
            print(f"  Status: {'✓ BEST' if model_name == best_model_name else ''}")

        print(f"\n{'='*80}")
        print(f"Best Performing Model: {best_model_name}")
        print(f"Best Accuracy: {best_accuracy:.4f}")
        print(f"{'='*80}")

        # Save best model
        best_model = trained_models[best_model_name]
        best_model.save(OUTPUT_DIR / "BEST_MODEL.h5")
        joblib.dump(scaler, OUTPUT_DIR / "scaler.joblib")
        print(f"\n✓ Best model saved: {OUTPUT_DIR / 'BEST_MODEL.h5'}")
        print(f"✓ Scaler saved: {OUTPUT_DIR / 'scaler.joblib'}")

        # Save results summary
        with open(OUTPUT_DIR / "results_summary.txt", "w") as f:
            f.write("GPU-Optimized Driver Stress Detection - Architecture Comparison\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write(f"GPU Used: {'Yes (RTX 4060)' if GPU_AVAILABLE else 'No (CPU)'}\n")
            f.write("=" * 80 + "\n\n")

            for model_name, result in results.items():
                f.write(f"{model_name}\n")
                f.write(f"{'='*40}\n")
                f.write(f"Accuracy: {result['accuracy']:.4f}\n")
                f.write(f"F1-Score: {result['f1_score']:.4f}\n")
                f.write(f"\nClassification Report:\n{result['report']}\n\n")

            f.write(f"\nBest Model: {best_model_name}\n")
            f.write(f"Best Accuracy: {best_accuracy:.4f}\n")

        print(f"✓ Results summary saved: {OUTPUT_DIR / 'results_summary.txt'}")
        
        print("\n" + "=" * 80)
        print("ALL OUTPUTS SAVED TO:")
        print(f"  📁 {OUTPUT_DIR.absolute()}")
        print("=" * 80)
    else:
        print("\n✗ No models trained successfully. Please check the errors above.")

    print("\n✓ GPU-Optimized Training Complete!")
    print("\n" + "=" * 80)
    print("DETAILED MODEL COMPARISON - CNN-BiLSTM vs CNN-GRU")
    print("=" * 80)

    model1 = trained_models["CNN-BiLSTM-Attention"]
    model5 = trained_models["CNN-GRU-Attention"]

    print(f"\nCNN-BiLSTM-Attention:")
    print(f"  Parameters: {model1.count_params():,}")
    print(f"  Accuracy: {results['CNN-BiLSTM-Attention']['accuracy']:.4f}")
    print(f"  F1-Score: {results['CNN-BiLSTM-Attention']['f1_score']:.4f}")

    print(f"\nCNN-GRU-Attention:")
    print(f"  Parameters: {model5.count_params():,}")
    print(f"  Accuracy: {results['CNN-GRU-Attention']['accuracy']:.4f}")
    print(f"  F1-Score: {results['CNN-GRU-Attention']['f1_score']:.4f}")

    param_diff = model1.count_params() - model5.count_params()
    param_pct = (param_diff / model1.count_params()) * 100
    print(f"\nGRU is {param_pct:.1f}% lighter ({param_diff:,} fewer parameters)")

    print("Generating Publication-Quality Diagrams...")

    model_params = {name: trained_models[name].count_params() for name in trained_models}

    # Generate comprehensive comparison
    plot_model_metrics_comparison(results)
    plot_architecture_complexity(results, model_params)
    plot_class_distribution(y_train, y_val, y_test)

    # For each model, generate detailed diagrams
    for model_name in results.keys():
        model = trained_models[model_name]
        y_pred_proba = model.predict(X_test, batch_size=128, verbose=0)
        
        plot_training_history(histories[model_name], model_name)
        plot_confusion_matrix_paper(results[model_name]['confusion_matrix'], model_name)
        plot_roc_curves(y_test, y_pred_proba, num_classes, model_name)
        plot_precision_recall_curves(y_test, y_pred_proba, num_classes, model_name)
        plot_per_class_metrics(y_test, y_pred_proba)

    # Generate summary report
    generate_paper_summary_report(results, model_params)

    print(f"All diagrams saved to: {PAPER_OUTPUT_DIR}")


if __name__ == "__main__":
    main()