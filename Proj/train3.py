"""
Deep Learning-Based Driver Stress Detection Model (train3.py)
Combined CNN-LSTM Architecture with Attention Mechanism

This script implements a hybrid deep learning approach combining:
- 1D Convolutional Neural Networks (CNN) for local feature extraction
- Long Short-Term Memory (LSTM) networks for temporal pattern recognition
- Attention mechanisms for feature weighting
- Data augmentation for handling class imbalance
- Early stopping and regularization to prevent overfitting
"""

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
    precision_recall_curve,
    roc_auc_score,
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
import joblib
import warnings

warnings.filterwarnings("ignore")


class AttentionLayer(layers.Layer):
    """
    Custom Attention Layer
    Applies attention mechanism to focus on important timesteps
    """

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
        """
        Apply attention weights to input sequence
        x shape: (batch_size, timesteps, features)
        """
        # Calculate attention scores
        e = keras.backend.tanh(keras.backend.dot(x, self.W) + self.b)
        # Normalize attention weights
        a = keras.backend.softmax(e, axis=1)
        # Apply attention weights
        output = x * a
        return output, a

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2])


def load_and_preprocess_data(csv_path):
    """
    Load and preprocess feature data
    Handles missing values and normalizes data
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Handle missing values
    print(f"Missing values before handling: {df.isnull().sum().sum()}")
    df = df.fillna(df.mean())
    print(f"Missing values after handling: {df.isnull().sum().sum()}")

    # Separate features and labels
    X = df.iloc[:, :-1].values  # All columns except last
    y = df.iloc[:, -1].values  # Last column is the label

    print(f"Data shape: {X.shape}, Labels shape: {y.shape}")
    print(f"Unique classes: {np.unique(y)}")
    print(f"Class distribution: {np.bincount(y)}")

    return X, y


def reshape_for_lstm(X, sequence_length=10):
    """
    Reshape 2D feature matrix into 3D sequences for LSTM
    Creates sliding windows of features for temporal analysis
    """
    X_sequences = []
    y_sequences = []

    for i in range(len(X) - sequence_length):
        X_sequences.append(X[i : i + sequence_length])

    X_sequences = np.array(X_sequences)
    print(f"Reshaped data for LSTM: {X_sequences.shape}")
    return X_sequences


def apply_data_augmentation(X_train, y_train, augmentation_factor=0.3):
    """
    Apply data augmentation to handle class imbalance
    Uses small random noise injection to create synthetic samples
    """
    print("Applying data augmentation...")
    from imblearn.over_sampling import SMOTE

    # Reshape for SMOTE if needed
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
        print(f"After SMOTE - X: {X_train_resampled.shape}, y: {y_train_resampled.shape}")
        return X_train_resampled, y_train_resampled
    except Exception as e:
        print(f"SMOTE failed: {e}. Proceeding without augmentation.")
        return X_train, y_train


def build_cnn_lstm_attention_model(
    input_shape, num_classes, dropout_rate=0.3, l2_reg=0.001
):
    """
    Build CNN-LSTM model with Attention Mechanism
    Architecture:
    1. 1D Conv layers for local feature extraction
    2. LSTM layers for temporal pattern learning
    3. Attention mechanism for feature weighting
    4. Dense layers for classification
    """
    model = keras.Sequential()

    # 1D Convolutional Layer 1: Extract local patterns
    model.add(
        layers.Conv1D(
            filters=64,
            kernel_size=3,
            padding="same",
            activation="relu",
            input_shape=input_shape,
            kernel_regularizer=regularizers.l2(l2_reg),
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout_rate))

    # 1D Convolutional Layer 2: Hierarchical feature extraction
    model.add(
        layers.Conv1D(
            filters=128,
            kernel_size=3,
            padding="same",
            activation="relu",
            kernel_regularizer=regularizers.l2(l2_reg),
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Dropout(dropout_rate))

    # 1D Convolutional Layer 3: Higher-level patterns
    model.add(
        layers.Conv1D(
            filters=64,
            kernel_size=3,
            padding="same",
            activation="relu",
            kernel_regularizer=regularizers.l2(l2_reg),
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout_rate))

    # LSTM Layer 1: Capture long-term dependencies
    model.add(
        layers.LSTM(
            units=128,
            return_sequences=True,
            activation="relu",
            kernel_regularizer=regularizers.l2(l2_reg),
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout_rate))

    # LSTM Layer 2: Further temporal refinement
    model.add(
        layers.LSTM(
            units=64,
            return_sequences=True,
            activation="relu",
            kernel_regularizer=regularizers.l2(l2_reg),
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout_rate))

    # Attention Layer: Focus on important timesteps
    attention_output, attention_weights = AttentionLayer()(model.output)
    model_with_attention = keras.Model(inputs=model.input, outputs=[attention_output])

    # Flatten for Dense layers
    model_with_attention.add(layers.Flatten())

    # Dense Layer 1: Classification feature extraction
    model_with_attention.add(
        layers.Dense(
            128,
            activation="relu",
            kernel_regularizer=regularizers.l2(l2_reg),
        )
    )
    model_with_attention.add(layers.BatchNormalization())
    model_with_attention.add(layers.Dropout(dropout_rate))

    # Dense Layer 2: Further classification refinement
    model_with_attention.add(
        layers.Dense(
            64,
            activation="relu",
            kernel_regularizer=regularizers.l2(l2_reg),
        )
    )
    model_with_attention.add(layers.Dropout(dropout_rate))

    # Output Layer: Classification
    model_with_attention.add(
        layers.Dense(num_classes, activation="softmax")
    )

    return model_with_attention


def train_model(
    X_train,
    y_train,
    X_val,
    y_val,
    num_classes,
    epochs=100,
    batch_size=32,
):
    """
    Train the CNN-LSTM-Attention model with callbacks
    """
    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_cnn_lstm_attention_model(
        input_shape=input_shape,
        num_classes=num_classes,
        dropout_rate=0.3,
        l2_reg=0.001,
    )

    # Compile model
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Print model summary
    model.summary()

    # Define callbacks
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True,
        verbose=1,
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1,
    )

    checkpoint = ModelCheckpoint(
        "stress_prediction_model_v3_best.h5",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    )

    # Train model
    print("\nTraining CNN-LSTM-Attention model...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr, checkpoint],
        verbose=1,
    )

    return model, history


def evaluate_model(model, X_test, y_test, class_names=None):
    """
    Evaluate model performance on test set
    Generate classification report and visualizations
    """
    print("\nEvaluating model...")

    # Predictions
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score (weighted): {f1:.4f}")

    # Classification Report
    print("\nClassification Report:")
    report = classification_report(y_test, y_pred, target_names=class_names)
    print(report)

    # Save classification report
    with open("classification_report_v3.txt", "w") as f:
        f.write(f"Model: CNN-LSTM-Attention\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n\n")
        f.write(report)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix - CNN-LSTM-Attention Model")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("confusion_matrix_v3.png", dpi=300)
    print("Confusion matrix saved as confusion_matrix_v3.png")

    return y_pred, y_pred_proba, accuracy, f1


def plot_training_history(history):
    """
    Plot training and validation metrics
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Accuracy
    axes[0].plot(history.history["accuracy"], label="Training Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Validation Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Model Accuracy")
    axes[0].legend()
    axes[0].grid(True)

    # Loss
    axes[1].plot(history.history["loss"], label="Training Loss")
    axes[1].plot(history.history["val_loss"], label="Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Model Loss")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig("training_history_v3.png", dpi=300)
    print("Training history plot saved as training_history_v3.png")


def save_model(model, model_name="stress_prediction_model_v3.joblib"):
    """
    Save trained model
    Note: For Keras models, use .h5 or SavedModel format
    """
    print(f"\nSaving model as {model_name}...")
    model.save("stress_prediction_model_v3.h5")
    print("Model saved as stress_prediction_model_v3.h5")


def main():
    """
    Main training pipeline
    """
    print("=" * 80)
    print("CNN-LSTM-Attention Model for Driver Stress Detection")
    print("=" * 80)

    # Load and preprocess data
    X, y = load_and_preprocess_data("./Proj/stress_features_all.csv")

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Reshape for LSTM (create sequences)
    sequence_length = 10
    X_sequences = reshape_for_lstm(X_scaled, sequence_length=sequence_length)

    # Corresponding labels (take the last label of each sequence)
    y_sequences = y[sequence_length:]

    # Train-validation-test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_sequences, y_sequences, test_size=0.3, random_state=42, stratify=y_sequences
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"Training set: {X_train.shape}, {y_train.shape}")
    print(f"Validation set: {X_val.shape}, {y_val.shape}")
    print(f"Test set: {X_test.shape}, {y_test.shape}")

    # Apply data augmentation
    X_train, y_train = apply_data_augmentation(X_train, y_train)

    # Determine number of classes
    num_classes = len(np.unique(y_train))

    # Train model
    model, history = train_model(
        X_train,
        y_train,
        X_val,
        y_val,
        num_classes=num_classes,
        epochs=100,
        batch_size=32,
    )

    # Plot training history
    plot_training_history(history)

    # Evaluate model
    class_names = [f"Stress_Level_{i}" for i in range(num_classes)]
    y_pred, y_pred_proba, accuracy, f1 = evaluate_model(
        model, X_test, y_test, class_names=class_names
    )

    # Save model
    save_model(model)

    # Save scaler for future use
    joblib.dump(scaler, "scaler_v3.joblib")
    print("Scaler saved as scaler_v3.joblib")

    print("\n" + "=" * 80)
    print("Training Complete!")
    print(f"Final Test Accuracy: {accuracy:.4f}")
    print(f"Final Test F1-Score: {f1:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()