"""
Project Workflow Diagram Generator
Creates a visual workflow showing the complete pipeline from data collection to LLM response
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from pathlib import Path

# Setup
OUTPUT_DIR = Path("train4_final_outputs_workflow")
OUTPUT_DIR.mkdir(exist_ok=True)

# Create figure
fig, ax = plt.subplots(figsize=(20, 14))
ax.set_xlim(0, 20)
ax.set_ylim(0, 28)
ax.axis('off')

# Color scheme
colors = {
    'data': '#3498db',      # Blue - Data collection
    'process': '#9b59b6',   # Purple - Processing
    'model': '#e74c3c',     # Red - Model training
    'deploy': '#2ecc71',    # Green - Deployment
    'llm': '#f39c12',       # Orange - LLM
    'output': '#1abc9c'     # Teal - Final output
}

def draw_box(ax, x, y, width, height, text, color, fontsize=10):
    """Draw a rounded box with text"""
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.1",
        facecolor=color,
        edgecolor='black',
        linewidth=2.5,
        alpha=0.9
    )
    ax.add_patch(box)
    
    # Add text
    ax.text(x + width/2, y + height/2, text,
           ha='center', va='center',
           fontsize=fontsize, fontweight='bold',
           color='white',
           wrap=True)
    
    return box

def draw_arrow(ax, x1, y1, x2, y2, label=''):
    """Draw arrow between boxes"""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='->,head_width=0.4,head_length=0.3',
        color='black',
        linewidth=2.5,
        connectionstyle="arc3,rad=0"
    )
    ax.add_patch(arrow)
    
    # Add label if provided
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x + 0.5, mid_y, label,
               fontsize=8, style='italic',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

def draw_circle_node(ax, x, y, radius, text, color):
    """Draw circular node"""
    circle = Circle((x, y), radius, 
                   facecolor=color, 
                   edgecolor='black', 
                   linewidth=2.5,
                   alpha=0.9)
    ax.add_patch(circle)
    ax.text(x, y, text,
           ha='center', va='center',
           fontsize=9, fontweight='bold',
           color='white')

# Title
ax.text(10, 27, 'Driver Stress Detection System - Complete Workflow',
       ha='center', fontsize=20, fontweight='bold')

# ============================================================================
# PHASE 1: DATA COLLECTION (Top)
# ============================================================================
ax.text(10, 25.5, 'Phase 1: Data Collection & Preprocessing',
       ha='center', fontsize=14, fontweight='bold',
       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.5))

# Sensors
draw_circle_node(ax, 2, 24, 0.6, 'ECG\nSensor', colors['data'])
draw_circle_node(ax, 5, 24, 0.6, 'EMG\nSensor', colors['data'])
draw_circle_node(ax, 8, 24, 0.6, 'GSR\nSensor', colors['data'])
draw_circle_node(ax, 11, 24, 0.6, 'Resp\nSensor', colors['data'])

# Feature extraction box
draw_box(ax, 4, 21.5, 8, 1.5, 'Feature Extraction\n(29 features)', colors['process'])

# Arrows from sensors to feature extraction
for x in [2, 5, 8, 11]:
    draw_arrow(ax, x, 23.4, 8, 22.5)

# Extracted features
draw_box(ax, 3, 19, 10, 1.2, 
        'Features: HR_mean, SDNN, RMSSD, EMG_energy,\nGSR_peaks, Resp_rate, etc.', 
        colors['process'], fontsize=9)

draw_arrow(ax, 8, 21.5, 8, 20.2)

# ============================================================================
# PHASE 2: DATA PREPROCESSING
# ============================================================================
ax.text(10, 18, 'Phase 2: Data Preprocessing',
       ha='center', fontsize=14, fontweight='bold',
       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.5))

# Preprocessing steps
draw_box(ax, 2, 16, 3, 1, 'Handle\nMissing\nValues', colors['process'], fontsize=9)
draw_box(ax, 6, 16, 3, 1, 'Standard\nScaling', colors['process'], fontsize=9)
draw_box(ax, 10, 16, 3, 1, 'Sequence\nGeneration\n(10 steps)', colors['process'], fontsize=9)
draw_box(ax, 14, 16, 3, 1, 'Train/Val/\nTest Split', colors['process'], fontsize=9)

draw_arrow(ax, 8, 19, 3.5, 17)
draw_arrow(ax, 3.5, 16, 7.5, 16)
draw_arrow(ax, 7.5, 16, 11.5, 16)
draw_arrow(ax, 11.5, 16, 15.5, 16)

# ============================================================================
# PHASE 3: MODEL TRAINING
# ============================================================================
ax.text(10, 14.5, 'Phase 3: Model Training & Selection',
       ha='center', fontsize=14, fontweight='bold',
       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.5))

draw_arrow(ax, 15.5, 16, 10, 13.5)

# 5 Model architectures
models_x = [2, 5.5, 9, 12.5, 16]
model_names = ['CNN-BiLSTM\n-Attention', 'Parallel\nCNN-LSTM', 
               'ResNet1D\n-LSTM', 'Inception1D\n-LSTM', 'CNN-GRU\n-Attention']

for i, (x, name) in enumerate(zip(models_x, model_names)):
    draw_box(ax, x-1, 12, 2.5, 1.2, name, colors['model'], fontsize=8)

# Training arrows
for x in models_x:
    draw_arrow(ax, 10, 13.5, x, 13.2)

# Best model selection
draw_box(ax, 7, 10, 6, 1.2, 'Best Model Selection\n(CNN-GRU-Attention)', colors['model'])

for x in models_x:
    draw_arrow(ax, x, 12, 10, 11.2)

# ============================================================================
# PHASE 4: MODEL EVALUATION
# ============================================================================
ax.text(10, 9, 'Phase 4: Model Evaluation',
       ha='center', fontsize=14, fontweight='bold',
       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.5))

draw_arrow(ax, 10, 10, 10, 8)

eval_metrics = ['ROC-AUC\nCurves', 'Confusion\nMatrix', 'Feature\nImportance', 'Precision\nRecall']
eval_x = [3, 7, 11, 15]

for i, (x, metric) in enumerate(zip(eval_x, eval_metrics)):
    draw_box(ax, x-1, 6.5, 2.5, 1, metric, colors['process'], fontsize=8)

for x in eval_x:
    draw_arrow(ax, 10, 8, x, 7.5)

# Save best model
draw_box(ax, 7.5, 4.5, 5, 1, 'Save Best Model\n(BEST_MODEL.h5)', colors['deploy'])

for x in eval_x:
    draw_arrow(ax, x, 6.5, 10, 5.5)

# ============================================================================
# PHASE 5: DEPLOYMENT & REAL-TIME PREDICTION
# ============================================================================
ax.text(10, 3.5, 'Phase 5: Real-Time Deployment',
       ha='center', fontsize=14, fontweight='bold',
       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.5))

draw_arrow(ax, 10, 4.5, 10, 3)

# Live sensor input
draw_box(ax, 1, 1, 3, 1.2, 'Live Sensor\nData Stream', colors['data'])

# Preprocessing
draw_box(ax, 5, 1, 3, 1.2, 'Preprocess\n& Buffer', colors['process'])

# Model prediction
draw_box(ax, 9, 1, 3, 1.2, 'CNN-GRU\nPrediction', colors['deploy'])

draw_arrow(ax, 2.5, 3, 2.5, 2.2, 'Real-time')
draw_arrow(ax, 4, 1.6, 6.5, 1.6)
draw_arrow(ax, 8, 1.6, 10.5, 1.6)

# Stress level output
draw_box(ax, 13, 0.5, 4, 2, 'Stress Level\nDetected:\n• High\n• Medium\n• Low', colors['output'], fontsize=10)

draw_arrow(ax, 12, 1.6, 15, 1.6)

# ============================================================================
# PHASE 6: LLM CHATBOT INTEGRATION
# ============================================================================
ax.text(10, -1.5, 'Phase 6: LLM-Powered Assistance',
       ha='center', fontsize=14, fontweight='bold',
       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.5))

draw_arrow(ax, 15, 0.5, 15, -2.5, 'Stress\nLevel')

# LLM Processing
draw_box(ax, 13, -5, 4, 2, 'LLM Chatbot\n(GPT/Claude)\nContext-Aware\nResponse', colors['llm'], fontsize=10)

draw_arrow(ax, 15, -2.5, 15, -3)

# LLM responses based on stress level
draw_box(ax, 1, -6, 4, 1.8, 
        'High Stress Response:\n"Pull over safely\nTake deep breaths\nCall emergency contact"', 
        colors['output'], fontsize=8)

draw_box(ax, 8, -6, 4, 1.8, 
        'Medium Stress Response:\n"Reduce speed\nTurn on calming music\nTake short break soon"', 
        colors['output'], fontsize=8)

draw_box(ax, 15, -6, 4, 1.8, 
        'Low Stress Response:\n"You\'re doing great!\nStay hydrated\nKeep focus on road"', 
        colors['output'], fontsize=8)

draw_arrow(ax, 13, -4, 3, -5, 'High')
draw_arrow(ax, 15, -5, 10, -5.5, 'Medium')
draw_arrow(ax, 17, -5, 17, -4.2, 'Low')

# Add legend
legend_x, legend_y = 0.5, -8
ax.text(legend_x + 1, legend_y + 1.5, 'Legend:', fontsize=12, fontweight='bold')

legend_items = [
    ('Data Collection', colors['data']),
    ('Processing', colors['process']),
    ('Model Training', colors['model']),
    ('Deployment', colors['deploy']),
    ('LLM Integration', colors['llm']),
    ('Output', colors['output'])
]

for i, (label, color) in enumerate(legend_items):
    y_offset = legend_y + 0.8 - (i * 0.3)
    legend_box = FancyBboxPatch(
        (legend_x, y_offset), 0.8, 0.2,
        boxstyle="round,pad=0.02",
        facecolor=color,
        edgecolor='black',
        linewidth=1,
        alpha=0.9
    )
    ax.add_patch(legend_box)
    ax.text(legend_x + 1, y_offset + 0.1, label, 
           fontsize=9, va='center')

# Add system info box
info_text = """System Highlights:
• 5 Deep Learning Architectures Tested
• 29 Multimodal Features (ECG, EMG, GSR, Resp)
• Real-time Prediction (<100ms latency)
• LLM-Powered Context-Aware Responses
• GPU-Accelerated Training (RTX 4060)
• 95%+ Accuracy on Test Set"""

ax.text(17.5, -8.5, info_text,
       fontsize=8,
       bbox=dict(boxstyle='round,pad=0.5', 
                facecolor='lightyellow', 
                edgecolor='black',
                linewidth=2))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "complete_project_workflow.png", 
           dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("=" * 80)
print("PROJECT WORKFLOW DIAGRAM GENERATED")
print("=" * 80)
print(f"\n✓ Saved to: {OUTPUT_DIR / 'complete_project_workflow.png'}")
print("\nWorkflow Phases:")
print("  1. Data Collection & Preprocessing")
print("  2. Feature Extraction")
print("  3. Model Training & Selection")
print("  4. Model Evaluation")
print("  5. Real-Time Deployment")
print("  6. LLM-Powered Assistance")
print("\n" + "=" * 80)