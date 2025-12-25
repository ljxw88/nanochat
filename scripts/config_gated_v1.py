"""
Example configuration for GatedFWA training in nanochat.
Usage: python -m scripts.base_train scripts/config_gated_v1.py

This configuration demonstrates the use of GatedFWA (Gated Forward-Window Attention)
with nanochat's training pipeline. GatedFWA provides:
- Sliding window attention: Limited attention span for efficiency
- Forgetting gate: Data-dependent exponential forgetting mechanism
- Token shift: Temporal smoothing via token blending
"""

# Model variant: choose "vanilla" for standard GPT or "gated" for GatedFWA
model_variant = "gated"

# Model size
depth = 20

# GatedFWA-specific configuration
window_size = 512  # Sliding window size: each query attends to [pos - window_size, pos]
                   # Set to None for full attention (standard causal)
use_forgetting_gate = True  # Enable data-dependent forgetting gate
use_k_shift = True  # Apply token shift to key projections
use_v_shift = True  # Apply token shift to value projections
