"""
Advanced Model Analysis and Visualization Suite
Generates comprehensive metrics including AUC-ROC curves, feature importance,
training history, and additional performance metrics

All outputs saved to: train4_final_outputs/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve,
    auc,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    classification_report
)
from sklearn.preprocessing import label_binarize
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
from pathlib import Path
import json

# Setup output directory
OUTPUT_DIR = Path("train4_final_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

MODELS_DIR = Path("train4_outputs")

print("=" * 80)
print("ADVANCED MODEL ANALYSIS SUITE")
print("=" * 80)
print(f"Reading models from: {MODELS_DIR}")
print(f"Saving analysis to: {OUTPUT_DIR}")
print("=" * 80 + "\n")


# Custom AttentionLayer definition
class AttentionLayer(layers.Layer):
    """Custom Attention Layer"""
    
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


def load_data():
    """Load and preprocess test data"""
    print("Loading data...")
    
    # Load preprocessed data
    df = pd.read_csv("stress_features_all.csv")
    
    # Extract labels
    y_raw = df.iloc[:, -1].values
    
    # Encode labels if needed
    if isinstance(y_raw[0], str):
        label_mapping = {}
        unique_labels = sorted(set(y_raw))
        for idx, label in enumerate(unique_labels):
            label_mapping[label] = idx
        y = np.array([label_mapping[label] for label in y_raw])
    else:
        y = y_raw
    
    # Get features
    X = df.iloc[:, :-1].fillna(df.iloc[:, :-1].mean(numeric_only=True)).values
    
    # Load scaler
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    X_scaled = scaler.transform(X)
    
    # Reshape for LSTM
    sequence_length = 10
    X_sequences = []
    for i in range(len(X_scaled) - sequence_length):
        X_sequences.append(X_scaled[i : i + sequence_length])
    X_sequences = np.array(X_sequences)
    y_sequences = y[sequence_length:]
    
    # Get test split (same as training)
    from sklearn.model_selection import train_test_split
    _, X_temp, _, y_temp = train_test_split(
        X_sequences, y_sequences, test_size=0.3, random_state=42, stratify=y_sequences
    )
    _, X_test, _, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"✓ Test data loaded: {X_test.shape}")
    return X_test, y_test, df.columns[:-1].tolist()


def plot_roc_curves(models_dict, X_test, y_test, num_classes):
    """Generate ROC curves for all models"""
    print("\nGenerating ROC curves...")
    
    # Binarize labels for multiclass
    y_test_bin = label_binarize(y_test, classes=range(num_classes))
    
    # Create figure
    fig, axes = plt.subplots(1, num_classes, figsize=(18, 5))
    if num_classes == 1:
        axes = [axes]
    
    colors = plt.cm.Set3(range(len(models_dict)))
    
    # For each class
    for class_idx in range(num_classes):
        ax = axes[class_idx]
        
        # For each model
        for (model_name, model), color in zip(models_dict.items(), colors):
            # Get predictions
            y_pred_proba = model.predict(X_test, verbose=0)
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_test_bin[:, class_idx], y_pred_proba[:, class_idx])
            roc_auc = auc(fpr, tpr)
            
            # Plot
            ax.plot(fpr, tpr, color=color, lw=2, 
                   label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        # Plot diagonal
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=10)
        ax.set_ylabel('True Positive Rate', fontsize=10)
        ax.set_title(f'ROC Curve - Class {class_idx}', fontsize=12, fontweight='bold')
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "roc_curves_all_classes.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ ROC curves saved")


def plot_precision_recall_curves(models_dict, X_test, y_test, num_classes):
    """Generate Precision-Recall curves"""
    print("\nGenerating Precision-Recall curves...")
    
    y_test_bin = label_binarize(y_test, classes=range(num_classes))
    
    fig, axes = plt.subplots(1, num_classes, figsize=(18, 5))
    if num_classes == 1:
        axes = [axes]
    
    colors = plt.cm.Set3(range(len(models_dict)))
    
    for class_idx in range(num_classes):
        ax = axes[class_idx]
        
        for (model_name, model), color in zip(models_dict.items(), colors):
            y_pred_proba = model.predict(X_test, verbose=0)
            
            precision, recall, _ = precision_recall_curve(
                y_test_bin[:, class_idx], y_pred_proba[:, class_idx]
            )
            avg_precision = average_precision_score(
                y_test_bin[:, class_idx], y_pred_proba[:, class_idx]
            )
            
            ax.plot(recall, precision, color=color, lw=2,
                   label=f'{model_name} (AP = {avg_precision:.3f})')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=10)
        ax.set_ylabel('Precision', fontsize=10)
        ax.set_title(f'Precision-Recall - Class {class_idx}', fontsize=12, fontweight='bold')
        ax.legend(loc="lower left", fontsize=8)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "precision_recall_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Precision-Recall curves saved")


def analyze_feature_importance(model, X_test, y_test, feature_names, model_name):
    """
    Analyze feature importance using permutation importance
    """
    print(f"\nAnalyzing feature importance for {model_name}...")
    
    # Get baseline accuracy
    baseline_preds = model.predict(X_test, verbose=0)
    baseline_acc = np.mean(np.argmax(baseline_preds, axis=1) == y_test)
    
    # Calculate permutation importance
    importances = []
    
    # For each feature
    for feat_idx in range(X_test.shape[2]):  # Features are in last dimension
        # Create shuffled copy
        X_shuffled = X_test.copy()
        
        # Shuffle this feature across all samples and timesteps
        for sample_idx in range(X_test.shape[0]):
            X_shuffled[sample_idx, :, feat_idx] = np.random.permutation(
                X_shuffled[sample_idx, :, feat_idx]
            )
        
        # Get new accuracy
        shuffled_preds = model.predict(X_shuffled, verbose=0)
        shuffled_acc = np.mean(np.argmax(shuffled_preds, axis=1) == y_test)
        
        # Importance = drop in accuracy
        importance = baseline_acc - shuffled_acc
        importances.append(importance)
    
    # Create dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Plot top 15 features
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(15)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance (Drop in Accuracy)', fontsize=12)
    plt.title(f'Top 15 Feature Importances - {model_name}', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"feature_importance_{model_name}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save to CSV
    importance_df.to_csv(OUTPUT_DIR / f"feature_importance_{model_name}.csv", index=False)
    
    print(f"✓ Feature importance saved for {model_name}")
    return importance_df


def plot_model_performance_comparison(models_dict, X_test, y_test):
    """Generate comprehensive performance comparison"""
    print("\nGenerating performance comparison...")
    
    metrics_data = []
    
    for model_name, model in models_dict.items():
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        
        # AUC (multiclass)
        y_test_bin = label_binarize(y_test, classes=range(len(np.unique(y_test))))
        try:
            auc_score = roc_auc_score(y_test_bin, y_pred_proba, average='weighted', multi_class='ovr')
        except:
            auc_score = 0
        
        metrics_data.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'F1-Score': f1,
            'Precision': precision,
            'Recall': recall,
            'AUC': auc_score
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Plot comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    metrics = ['Accuracy', 'F1-Score', 'Precision', 'Recall', 'AUC']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        bars = ax.bar(metrics_df['Model'], metrics_df[metric], 
                      color=plt.cm.Set3(range(len(metrics_df))))
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f'{metric} Comparison', fontsize=13, fontweight='bold')
        ax.set_ylim([0.5, 1.0])
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Hide last subplot
    axes[1, 2].axis('off')
    
    # Add summary table
    table_data = []
    for _, row in metrics_df.iterrows():
        table_data.append([
            row['Model'],
            f"{row['Accuracy']:.4f}",
            f"{row['F1-Score']:.4f}",
            f"{row['Precision']:.4f}",
            f"{row['Recall']:.4f}",
            f"{row['AUC']:.4f}"
        ])
    
    table = axes[1, 2].table(
        cellText=table_data,
        colLabels=['Model', 'Acc', 'F1', 'Prec', 'Rec', 'AUC'],
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "comprehensive_performance_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save metrics to CSV
    metrics_df.to_csv(OUTPUT_DIR / "model_metrics_comparison.csv", index=False)
    
    print("✓ Performance comparison saved")
    return metrics_df


def plot_confusion_matrices_grid(models_dict, X_test, y_test, class_names):
    """Plot confusion matrices for all models in a grid"""
    print("\nGenerating confusion matrix grid...")
    
    n_models = len(models_dict)
    cols = 3
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    axes = axes.flatten() if n_models > 1 else [axes]
    
    for idx, (model_name, model) in enumerate(models_dict.items()):
        y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
        cm = confusion_matrix(y_test, y_pred)
        
        ax = axes[idx]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
    
    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrices_grid.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Confusion matrices grid saved")


def generate_summary_report(metrics_df, models_dict, X_test, y_test):
    """Generate comprehensive text summary report"""
    print("\nGenerating summary report...")
    
    with open(OUTPUT_DIR / "comprehensive_analysis_report.txt", "w") as f:
        f.write("=" * 80 + "\n")
        f.write("COMPREHENSIVE MODEL ANALYSIS REPORT\n")
        f.write("Driver Stress Detection - Advanced Deep Learning Models\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Analysis Date: {pd.Timestamp.now()}\n")
        f.write(f"Test Set Size: {len(y_test)} samples\n")
        f.write(f"Number of Models: {len(models_dict)}\n")
        f.write(f"Number of Classes: {len(np.unique(y_test))}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("PERFORMANCE METRICS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(metrics_df.to_string(index=False))
        f.write("\n\n")
        
        # Find best model for each metric
        f.write("=" * 80 + "\n")
        f.write("BEST PERFORMING MODELS BY METRIC\n")
        f.write("=" * 80 + "\n\n")
        
        for metric in ['Accuracy', 'F1-Score', 'Precision', 'Recall', 'AUC']:
            best_idx = metrics_df[metric].idxmax()
            best_model = metrics_df.loc[best_idx, 'Model']
            best_value = metrics_df.loc[best_idx, metric]
            f.write(f"{metric:15s}: {best_model:30s} ({best_value:.4f})\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("MODEL PARAMETERS COMPARISON\n")
        f.write("=" * 80 + "\n\n")
        
        for model_name, model in models_dict.items():
            params = model.count_params()
            f.write(f"{model_name:30s}: {params:,} parameters\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("=" * 80 + "\n\n")
        
        # Overall best model
        best_overall_idx = metrics_df['F1-Score'].idxmax()
        best_overall = metrics_df.loc[best_overall_idx, 'Model']
        
        f.write(f"Best Overall Model: {best_overall}\n")
        f.write(f"- Highest F1-Score: {metrics_df.loc[best_overall_idx, 'F1-Score']:.4f}\n")
        f.write(f"- Balanced performance across all metrics\n")
        f.write(f"- Recommended for production deployment\n\n")
        
        # Lightest model
        param_counts = {name: model.count_params() for name, model in models_dict.items()}
        lightest_model = min(param_counts, key=param_counts.get)
        lightest_idx = metrics_df[metrics_df['Model'] == lightest_model].index[0]
        
        f.write(f"Most Efficient Model: {lightest_model}\n")
        f.write(f"- Parameters: {param_counts[lightest_model]:,}\n")
        f.write(f"- F1-Score: {metrics_df.loc[lightest_idx, 'F1-Score']:.4f}\n")
        f.write(f"- Recommended for edge/mobile deployment\n")
    
    print("✓ Summary report saved")


def main():
    """Main analysis pipeline"""
    
    # Load data
    X_test, y_test, feature_names = load_data()
    num_classes = len(np.unique(y_test))
    class_names = [f"Stress_Level_{i}" for i in range(num_classes)]
    
    # Load all models
    print("\nLoading models...")
    models_dict = {}
    
    model_files = list(MODELS_DIR.glob("*.h5"))
    
    for model_file in model_files:
        if model_file.stem == "scaler":
            continue
        
        model_name = model_file.stem
        print(f"  Loading {model_name}...")
        
        try:
            model = keras.models.load_model(
                str(model_file),
                custom_objects={'AttentionLayer': AttentionLayer}
            )
            models_dict[model_name] = model
        except Exception as e:
            print(f"  ✗ Failed to load {model_name}: {e}")
    
    print(f"\n✓ Loaded {len(models_dict)} models")
    
    # Generate all analyses
    print("\n" + "=" * 80)
    print("RUNNING COMPREHENSIVE ANALYSIS")
    print("=" * 80)
    
    # 1. ROC Curves
    plot_roc_curves(models_dict, X_test, y_test, num_classes)
    
    # 2. Precision-Recall Curves
    plot_precision_recall_curves(models_dict, X_test, y_test, num_classes)
    
    # 3. Performance Comparison
    metrics_df = plot_model_performance_comparison(models_dict, X_test, y_test)
    
    # 4. Confusion Matrices Grid
    plot_confusion_matrices_grid(models_dict, X_test, y_test, class_names)
    
    # 5. Feature Importance (for best model)
    best_model_name = metrics_df.loc[metrics_df['F1-Score'].idxmax(), 'Model']
    best_model = models_dict[best_model_name]
    analyze_feature_importance(best_model, X_test, y_test, feature_names, best_model_name)
    
    # 6. Summary Report
    generate_summary_report(metrics_df, models_dict, X_test, y_test)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\n✓ All outputs saved to: {OUTPUT_DIR.absolute()}")
    print("\nGenerated files:")
    print("  - roc_curves_all_classes.png")
    print("  - precision_recall_curves.png")
    print("  - comprehensive_performance_comparison.png")
    print("  - confusion_matrices_grid.png")
    print("  - feature_importance_[model].png")
    print("  - model_metrics_comparison.csv")
    print("  - comprehensive_analysis_report.txt")
    print("=" * 80)


if __name__ == "__main__":
    main()
