# Quick Start: GatedFWA Training

## TL;DR

### Train with Vanilla GPT (existing behavior, no changes)
```bash
torchrun --nproc_per_node=8 -m scripts.base_train -- --depth=20 --run=my_run
```

### Train with GatedFWA (recommended setup)
```bash
torchrun --nproc_per_node=8 -m scripts.base_train scripts/config_gated_v1.py --run=gated_run
```

### Train with custom GatedFWA config
```bash
torchrun --nproc_per_node=8 -m scripts.base_train -- \
    --depth=20 --model_variant=gated \
    --window_size=512 --use_forgetting_gate=True \
    --use_k_shift=True --use_v_shift=True \
    --run=custom_gated
```

## What Changed?

### New Files
1. **`nanochat/token_shift.py`** - Token shift layer (learned blending)
2. **`nanochat/flash_attention.py`** - Attention with forgetting gate support
3. **`nanochat/gpt_gated.py`** - Full GPT model with GatedFWA features
4. **`scripts/config_gated_v1.py`** - Example config for GatedFWA training

### Modified Files
1. **`scripts/base_train.py`** - Added model_variant and GFWA parameters
2. **`speedrun_conda.sh`** - Added examples for both training modes

## Key Parameters Explained

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_variant` | `"vanilla"` | Use `"gated"` for GatedFWA features |
| `window_size` | `None` | Sliding window size (e.g., 512). `None` = full attention |
| `use_forgetting_gate` | `False` | Enable data-dependent forgetting gate |
| `use_k_shift` | `False` | Token shift on key projections |
| `use_v_shift` | `False` | Token shift on value projections |

## Example Configurations

### Configuration A: Vanilla GPT (Baseline)
```bash
# No changes, everything works as before
torchrun --nproc_per_node=8 -m scripts.base_train -- --depth=20 --run=baseline
```

### Configuration B: Sliding Window Only
```bash
torchrun --nproc_per_node=8 -m scripts.base_train -- \
    --depth=20 --model_variant=gated \
    --window_size=512 \
    --run=sliding_window
```

### Configuration C: Sliding Window + Forgetting Gate
```bash
torchrun --nproc_per_node=8 -m scripts.base_train -- \
    --depth=20 --model_variant=gated \
    --window_size=512 --use_forgetting_gate=True \
    --run=window_plus_gate
```

### Configuration D: Full GatedFWA (Recommended)
```bash
torchrun --nproc_per_node=8 -m scripts.base_train -- \
    --depth=20 --model_variant=gated \
    --window_size=512 --use_forgetting_gate=True \
    --use_k_shift=True --use_v_shift=True \
    --run=full_gatedfwa
```

Or simply:
```bash
torchrun --nproc_per_node=8 -m scripts.base_train scripts/config_gated_v1.py
```

## Testing (CPU/Single GPU)

```bash
# Small test without tokenizer requirements
python3 << 'EOF'
import torch
from nanochat.gpt_gated import GPTGated, GPTGatedConfig

# Create a small model
config = GPTGatedConfig(
    sequence_len=512,
    vocab_size=256,
    n_layer=2,
    n_head=4,
    n_kv_head=4,
    n_embd=256,
    window_size=128,
    use_forgetting_gate=True,
    use_k_shift=True,
    use_v_shift=True
)
model = GPTGated(config)
model.init_weights()

# Test forward pass
x = torch.randint(0, 256, (2, 16))
targets = torch.randint(0, 256, (2, 16))
loss = model(x, targets)
print(f"✓ GatedFWA model works! Loss: {loss.item():.4f}")
EOF
```

## Monitoring Training

The same metrics are tracked for both vanilla and gated models:
- Validation bits-per-byte (bpb)
- Training loss
- CORE metric (if enabled)
- Learning rate multipliers
- Model FLOPs and throughput

## Resuming Training

Checkpoints save all configuration parameters. Resume with:
```bash
torchrun --nproc_per_node=8 -m scripts.base_train -- --resume_from_step=1000
```

The model variant and all GFWA parameters are automatically restored.

## Switching Between Models

Use different checkpoint directories for different variants:
```bash
# Train vanilla
torchrun --nproc_per_node=8 -m scripts.base_train -- \
    --depth=20 --model_tag=d20_vanilla --run=vanilla

# Train gated
torchrun --nproc_per_node=8 -m scripts.base_train -- \
    --depth=20 --model_tag=d20_gated --model_variant=gated \
    --window_size=512 --use_forgetting_gate=True \
    --use_k_shift=True --use_v_shift=True --run=gated
```

## Performance Notes

- **Memory**: GFWA adds small overhead from forgetting gate params and token shift
- **Speed**: Sliding window reduces attention complexity (O(n*w) vs O(n²))
- **Convergence**: May differ from vanilla GPT; expect empirical validation needed

## Troubleshooting

### Import Errors
```bash
# Verify all new modules import correctly
python3 -c "from nanochat.gpt_gated import GPTGated; print('OK')"
```

### Device Mismatch
Ensure `torch.device("meta")` is used during config initialization (already done in base_train.py).

### Out of Memory
- Reduce `device_batch_size`
- Reduce `max_seq_len`
- Try `window_size` to reduce attention memory

## For More Details

See **`GATEDFWA_INTEGRATION.md`** for:
- Full technical implementation details
- Attention dispatch strategy
- Forgetting gate computation
- Token shift state management
- Compatibility matrix

---

**Happy training!** 🚀
