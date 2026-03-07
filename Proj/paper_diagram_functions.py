"""
Paper Diagram Functions for Driver Stress Detection
Publication-quality visualization functions
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
)
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# Create paper output directory
PAPER_OUTPUT_DIR = Path("train4_paper_fol")
PAPER_OUTPUT_DIR.mkdir(exist_ok=True)
print(f"Paper output directory created: {PAPER_OUTPUT_DIR}")


def plot_training_history(history, model_name):
    """
    Generate publication-quality training history plots (Loss & Accuracy curves)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{model_name} - Training History', fontsize=16, fontweight='bold')
    
    # Loss curve
    axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2.5, color='#1f77b4')
    axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2.5, color='#ff7f0e')
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0].set_title('Loss Curve', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=11, loc='upper right')
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].set_facecolor('#f8f9fa')
    
    # Accuracy curve
    axes[1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2.5, color='#2ca02c')
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2.5, color='#d62728')
    axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    axes[1].set_title('Accuracy Curve', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=11, loc='lower right')
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    path = PAPER_OUTPUT_DIR / f"{model_name}_training_history.png"
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Training history plot saved: {path}")
    plt.close()


def plot_roc_curves(y_test, y_pred_proba, num_classes, model_name, class_names=None):
    """
    Generate ROC curves for multi-class classification
    """
    if class_names is None:
        class_names = [f'Class {i}' for i in range(num_classes)]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'{model_name} - ROC Curves', fontsize=16, fontweight='bold')
    
    # Plot 1: Individual ROC curves for each class
    colors = plt.cm.Set1(np.linspace(0, 1, num_classes))
    
    for i in range(num_classes):
        y_binary = (y_test == i).astype(int)
        fpr, tpr, _ = roc_curve(y_binary, y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, linewidth=2.5, label=f'{class_names[i]} (AUC = {roc_auc:.3f})', color=colors[i])
    
    axes[0].plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
    axes[0].set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    axes[0].set_title('ROC Curves (One-vs-Rest)', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10, loc='lower right')
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].set_facecolor('#f8f9fa')
    
    # Plot 2: Average ROC curves
    y_pred = np.argmax(y_pred_proba, axis=1)
    avg_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
    
    axes[1].fill_between([0, 1], [0, 1], alpha=0.1, color='blue')
    axes[1].plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')
    axes[1].text(0.6, 0.3, f'Weighted Avg AUC\\n{avg_auc:.4f}', 
                fontsize=14, fontweight='bold', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1].set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    axes[1].set_title('Weighted Average ROC', fontsize=13, fontweight='bold')
    axes[1].set_xlim([0, 1])
    axes[1].set_ylim([0, 1])
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    path = PAPER_OUTPUT_DIR / f"{model_name}_roc_curves.png"
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ ROC curves plot saved: {path}")
    plt.close()


def plot_precision_recall_curves(y_test, y_pred_proba, num_classes, model_name, class_names=None):
    """
    Generate Precision-Recall curves
    """
    if class_names is None:
        class_names = [f'Class {i}' for i in range(num_classes)]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle(f'{model_name} - Precision-Recall Curves', fontsize=16, fontweight='bold')
    
    colors = plt.cm.Set1(np.linspace(0, 1, num_classes))
    
    for i in range(num_classes):
        y_binary = (y_test == i).astype(int)
        precision, recall, _ = precision_recall_curve(y_binary, y_pred_proba[:, i])
        ax.plot(recall, precision, linewidth=2.5, label=f'{class_names[i]}', color=colors[i])
    
    ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax.set_title('Precision-Recall Curves (One-vs-Rest)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_facecolor('#f8f9fa')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    path = PAPER_OUTPUT_DIR / f"{model_name}_precision_recall.png"
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Precision-Recall curves plot saved: {path}")
    plt.close()


def plot_confusion_matrix_paper(cm, model_name, class_names=None):
    """
    Generate publication-quality confusion matrix
    """
    num_classes = cm.shape[0]
    if class_names is None:
        class_names = [f'Class {i}' for i in range(num_classes)]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Normalize confusion matrix for better visualization
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_normalized, annot=cm, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Normalized Frequency'}, ax=ax,
                linewidths=1.5, linecolor='white', annot_kws={'fontsize': 11, 'fontweight': 'bold'})
    
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_title(f'{model_name} - Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    path = PAPER_OUTPUT_DIR / f"{model_name}_confusion_matrix_paper.png"
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Confusion matrix (paper quality) saved: {path}")
    plt.close()


def plot_model_metrics_comparison(results_dict):
    """
    Generate comprehensive model comparison plots
    """
    if not results_dict:
        return
    
    models = list(results_dict.keys())
    accuracies = [results_dict[m]["accuracy"] for m in models]
    f1_scores = [results_dict[m]["f1_score"] for m in models]
    
    # Create multi-metric comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Performance Comparison', fontsize=18, fontweight='bold', y=0.995)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
    
    # Plot 1: Accuracy comparison
    bars1 = axes[0, 0].bar(range(len(models)), accuracies, color=colors, edgecolor='black', linewidth=1.5)
    axes[0, 0].set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Accuracy Comparison', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylim([0.5, 1.0])
    axes[0, 0].set_xticks(range(len(models)))
    axes[0, 0].set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    axes[0, 0].grid(axis='y', alpha=0.3, linestyle='--')
    axes[0, 0].set_facecolor('#f8f9fa')
    
    for i, (bar, v) in enumerate(zip(bars1, accuracies)):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, v + 0.01, f'{v:.4f}',
                       ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Plot 2: F1-Score comparison
    bars2 = axes[0, 1].bar(range(len(models)), f1_scores, color=colors, edgecolor='black', linewidth=1.5)
    axes[0, 1].set_ylabel('F1-Score (Weighted)', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('F1-Score Comparison', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylim([0.5, 1.0])
    axes[0, 1].set_xticks(range(len(models)))
    axes[0, 1].set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    axes[0, 1].grid(axis='y', alpha=0.3, linestyle='--')
    axes[0, 1].set_facecolor('#f8f9fa')
    
    for i, (bar, v) in enumerate(zip(bars2, f1_scores)):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, v + 0.01, f'{v:.4f}',
                       ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Plot 3: Accuracy vs F1-Score scatter
    scatter = axes[1, 0].scatter(accuracies, f1_scores, s=300, c=range(len(models)),
                                cmap='Set3', edgecolors='black', linewidth=1.5, alpha=0.7)
    
    for i, model in enumerate(models):
        axes[1, 0].annotate(model, (accuracies[i], f1_scores[i]), 
                           fontsize=9, fontweight='bold', ha='center')
    
    axes[1, 0].set_xlabel('Accuracy', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('F1-Score (Weighted)', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('Accuracy vs F1-Score', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, linestyle='--')
    axes[1, 0].set_facecolor('#f8f9fa')
    
    # Plot 4: Summary table
    axes[1, 1].axis('off')
    
    # Create table data
    table_data = []
    table_data.append(['Model', 'Accuracy', 'F1-Score', 'Rank'])
    
    sorted_indices = sorted(range(len(accuracies)), key=lambda k: accuracies[k], reverse=True)
    for rank, idx in enumerate(sorted_indices, 1):
        table_data.append([
            models[idx],
            f'{accuracies[idx]:.4f}',
            f'{f1_scores[idx]:.4f}',
            f'{rank}'
        ])
    
    table = axes[1, 1].table(cellText=table_data, loc='center', cellLoc='center',
                            colWidths=[0.35, 0.2, 0.2, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Color header row
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color rows alternately
    for i in range(1, len(table_data)):
        for j in range(len(table_data[0])):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E7E6E6')
            else:
                table[(i, j)].set_facecolor('#F2F2F2')
    
    axes[1, 1].set_title('Performance Rankings', fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    path = PAPER_OUTPUT_DIR / "models_comparison_comprehensive.png"
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Comprehensive model comparison saved: {path}")
    plt.close()


def plot_architecture_complexity(results_dict, model_params=None):
    """
    Plot model complexity vs performance
    """
    if model_params is None:
        return
    
    models = list(results_dict.keys())
    accuracies = [results_dict[m]["accuracy"] for m in models]
    params = [model_params[m] if m in model_params else 0 for m in models]
    f1_scores = [results_dict[m]["f1_score"] for m in models]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create bubble chart
    scatter = ax.scatter(params, accuracies, s=[f*2000 for f in f1_scores], 
                        c=range(len(models)), cmap='Set3', 
                        edgecolors='black', linewidth=1.5, alpha=0.6)
    
    for i, model in enumerate(models):
        ax.annotate(model, (params[i], accuracies[i]), 
                   fontsize=10, fontweight='bold', ha='center', va='center')
    
    ax.set_xlabel('Number of Parameters (millions)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Model Complexity vs Performance\\n(Bubble size = F1-Score)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_facecolor('#f8f9fa')
    
    # Convert x-axis to millions
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
    
    plt.tight_layout()
    path = PAPER_OUTPUT_DIR / "model_complexity_performance.png"
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Model complexity plot saved: {path}")
    plt.close()


def plot_class_distribution(y_train, y_val, y_test, class_names=None):
    """
    Plot class distribution across train/val/test splits
    """
    num_classes = len(np.unique(y_train))
    if class_names is None:
        class_names = [f'Class {i}' for i in range(num_classes)]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Class Distribution Across Train/Validation/Test Splits', 
                fontsize=14, fontweight='bold')
    
    splits = [
        (y_train, 'Training Set', axes[0]),
        (y_val, 'Validation Set', axes[1]),
        (y_test, 'Test Set', axes[2])
    ]
    
    colors = plt.cm.Set3(np.linspace(0, 1, num_classes))
    
    for y, title, ax in splits:
        counts = np.bincount(y, minlength=num_classes)
        bars = ax.bar(range(num_classes), counts, color=colors, edgecolor='black', linewidth=1.5)
        ax.set_xlabel('Class', fontsize=11, fontweight='bold')
        ax.set_ylabel('Number of Samples', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(range(num_classes))
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_facecolor('#f8f9fa')
        
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                   str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    path = PAPER_OUTPUT_DIR / "class_distribution.png"
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Class distribution plot saved: {path}")
    plt.close()


def plot_per_class_metrics(y_test, y_pred_proba, class_names=None):
    """
    Plot per-class precision, recall, and F1-score
    """
    y_pred = np.argmax(y_pred_proba, axis=1)
    num_classes = len(np.unique(y_test))
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(num_classes)]
    
    # Calculate per-class metrics
    precisions = []
    recalls = []
    f1s = []
    
    for i in range(num_classes):
        y_binary = (y_test == i).astype(int)
        y_pred_binary = (y_pred == i).astype(int)
        
        tp = np.sum((y_pred_binary == 1) & (y_binary == 1))
        fp = np.sum((y_pred_binary == 1) & (y_binary == 0))
        fn = np.sum((y_pred_binary == 0) & (y_binary == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(num_classes)
    width = 0.25
    
    bars1 = ax.bar(x - width, precisions, width, label='Precision', color='#1f77b4', edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x, recalls, width, label='Recall', color='#ff7f0e', edgecolor='black', linewidth=1.2)
    bars3 = ax.bar(x + width, f1s, width, label='F1-Score', color='#2ca02c', edgecolor='black', linewidth=1.2)
    
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_facecolor('#f8f9fa')
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    path = PAPER_OUTPUT_DIR / "per_class_metrics.png"
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Per-class metrics plot saved: {path}")
    plt.close()


def generate_paper_summary_report(results_dict, model_params=None, history_dict=None):
    """
    Generate a comprehensive text report with all metrics
    """
    report_path = PAPER_OUTPUT_DIR / "PAPER_RESULTS_SUMMARY.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("DRIVER STRESS DETECTION - COMPREHENSIVE RESULTS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-" * 80 + "\n")
        best_model = max(results_dict, key=lambda x: results_dict[x]["accuracy"])
        best_acc = results_dict[best_model]["accuracy"]
        best_f1 = results_dict[best_model]["f1_score"]
        
        f.write(f"Best Performing Model: {best_model}\n")
        f.write(f"Best Accuracy: {best_acc:.4f}\n")
        f.write(f"Best F1-Score: {best_f1:.4f}\n\n")
        
        f.write("DETAILED RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        for model_name, results in results_dict.items():
            f.write(f"\n{model_name}\n")
            f.write("-" * 80 + "\n")
            
            if model_params and model_name in model_params:
                f.write(f"Total Parameters: {model_params[model_name]:,}\n")
            
            f.write(f"Accuracy: {results['accuracy']:.4f}\n")
            f.write(f"F1-Score: {results['f1_score']:.4f}\n\n")
            
            f.write("Classification Report:\n")
            f.write(results['report'])
            f.write("\n\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("VISUALIZATIONS GENERATED\n")
        f.write("=" * 80 + "\n\n")
        f.write("The following publication-quality diagrams have been generated:\n\n")
        f.write("1. Training Curves:\n")
        f.write("   - Loss curves (training vs validation)\n")
        f.write("   - Accuracy curves (training vs validation)\n\n")
        
        f.write("2. Model Performance:\n")
        f.write("   - Confusion matrices (normalized)\n")
        f.write("   - ROC curves (One-vs-Rest)\n")
        f.write("   - Precision-Recall curves\n")
        f.write("   - Per-class metrics (Precision, Recall, F1-Score)\n\n")
        
        f.write("3. Comparative Analysis:\n")
        f.write("   - Model accuracy comparison\n")
        f.write("   - Model F1-Score comparison\n")
        f.write("   - Model complexity vs performance\n\n")
        
        f.write("4. Data Analysis:\n")
        f.write("   - Class distribution across train/val/test splits\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("All images are saved in high resolution (300 DPI) for publication\n")
        f.write("=" * 80 + "\n")
    
    print(f"✓ Paper summary report saved: {report_path}")


# Print save location
print(f"\nAll paper-quality diagrams will be saved to:\n  📁 {PAPER_OUTPUT_DIR.absolute()}")
print("\nFunctions defined and ready to use with your training pipeline!")