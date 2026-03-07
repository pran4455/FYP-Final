"""
Generate Architecture Diagrams from Saved Models using Keras plot_model
Traverses train4_outputs folder and creates detailed layer diagrams
WITH UNIFORM BOX WIDTHS AND SHORTER ARROWS
FIXED FOR TENSORFLOW 2.10
"""

import os
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model

# Define paths
MODELS_DIR = Path("train4_outputs")
OUTPUT_DIR = Path("architectures2")
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("ARCHITECTURE DIAGRAM GENERATOR (Keras plot_model)")
print("=" * 80)
print(f"Reading models from: {MODELS_DIR}")
print(f"Saving diagrams to: {OUTPUT_DIR}")
print("=" * 80 + "\n")


# Define custom AttentionLayer (needed to load models with attention)
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


def plot_architecture_diagram(model, model_name):
    """Generate and save architecture diagram with uniform box widths"""
    try:
        diagram_path = OUTPUT_DIR / f"{model_name}_architecture.png"
        
        # Generate base diagram first
        plot_model(
            model,
            to_file=str(diagram_path),
            show_shapes=True,
            show_layer_names=True,
            show_layer_activations=False,
            rankdir='TB',
            expand_nested=True,
            dpi=300,
            show_dtype=False
        )
        
        # Now customize it with pydot for uniform appearance
        try:
            import pydot
            
            # Import the correct path for TensorFlow 2.10
            try:
                # Try TF 2.10+ path
                from keras.utils.vis_utils import model_to_dot
            except ImportError:
                try:
                    # Try older path
                    from tensorflow.python.keras.utils.vis_utils import model_to_dot
                except ImportError:
                    # If custom styling fails, just use the default diagram
                    print(f"  ⚠ Using default diagram style (custom styling unavailable)")
                    print(f"✓ Architecture diagram saved: {diagram_path}")
                    return True
            
            # Generate custom styled diagram
            dot = model_to_dot(
                model,
                show_shapes=True,
                show_layer_names=True,
                rankdir='TB',
                expand_nested=True,
                show_layer_activations=False,
                show_dtype=False,
                dpi=300
            )
            
            # Modify dot graph attributes for uniform appearance
            dot.set_graph_defaults(
                ranksep='0.3',      # Shorter vertical spacing
                nodesep='0.5',      # Horizontal spacing
                splines='ortho'     # Straight arrows
            )
            
            # Set uniform node attributes
            dot.set_node_defaults(
                shape='box',
                style='filled,rounded',
                fillcolor='lightblue',
                fontname='Arial',
                fontsize='10',
                width='2.5',
                height='0.6',
                fixedsize='true'
            )
            
            # Set arrow attributes
            dot.set_edge_defaults(
                arrowsize='0.8',
                penwidth='1.5'
            )
            
            # Save customized diagram
            dot.write_png(diagram_path)
            print(f"✓ Architecture diagram saved (custom style): {diagram_path}")
            
        except Exception as style_error:
            # If styling fails, the basic diagram was already saved
            print(f"  ⚠ Custom styling failed: {style_error}")
            print(f"✓ Architecture diagram saved (default style): {diagram_path}")
        
        return True
        
    except Exception as e:
        print(f"✗ Could not generate diagram for {model_name}: {e}")
        return False


def main():
    """Main function to traverse models and generate diagrams"""
    
    # Check if models directory exists
    if not MODELS_DIR.exists():
        print(f"✗ Error: Directory '{MODELS_DIR}' not found!")
        print("  Make sure you've run train4.py first to generate models.")
        return
    
    # Find all .h5 model files
    model_files = list(MODELS_DIR.glob("*.h5"))
    
    if not model_files:
        print(f"✗ No .h5 model files found in '{MODELS_DIR}'")
        return
    
    print(f"Found {len(model_files)} model(s):\n")
    
    # Process each model
    success_count = 0
    for model_file in sorted(model_files):
        model_name = model_file.stem  # Get filename without extension
        
        print(f"Processing: {model_name}")
        print("-" * 50)
        
        try:
            # Load model (with custom objects for AttentionLayer)
            print(f"  Loading model from {model_file.name}...")
            model = keras.models.load_model(
                str(model_file),
                custom_objects={'AttentionLayer': AttentionLayer}
            )
            print(f"  ✓ Model loaded successfully")
            print(f"  Total parameters: {model.count_params():,}")
            
            # Generate diagram
            if plot_architecture_diagram(model, model_name):
                success_count += 1
            
        except Exception as e:
            print(f"  ✗ Error loading model: {e}")
        
        print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total models processed: {len(model_files)}")
    print(f"Diagrams generated: {success_count}")
    print(f"Failed: {len(model_files) - success_count}")
    print(f"\n✓ All diagrams saved to: {OUTPUT_DIR.absolute()}")
    print("=" * 80)


if __name__ == "__main__":
    main()