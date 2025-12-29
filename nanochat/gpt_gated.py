"""
GPT with GatedFWA (Gated Forward-Window Attention) support.
Extends the vanilla GPT model with optional sliding window attention, forgetting gate, and token shift.
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    HAS_LIGER = True
except ImportError:
    HAS_LIGER = False

from nanochat.common import get_dist_info, print0
from nanochat.muon import Muon, DistMuon
from nanochat.adamw import DistAdamW
from nanochat.token_shift import ShiftLinear
from GFWA.flash import attention as flash_attention
from functools import partial


@torch.compiler.disable  # liger_kernel not compatible with torch.compile
def _liger_fused_cross_entropy(weight, logits, targets, loss_reduction='mean'):
    """Compute cross-entropy loss using liger_kernel's fused implementation."""
    if not HAS_LIGER:
        return F.cross_entropy(logits, targets, reduction=loss_reduction)
    
    loss_fn = LigerFusedLinearCrossEntropyLoss()
    loss = loss_fn(weight, logits, targets)
    
    if loss_reduction == 'mean':
        return loss.mean()
    elif loss_reduction == 'sum':
        return loss.sum()
    elif loss_reduction == 'none':
        return loss
    else:
        return loss.mean()


@dataclass
class GPTGatedConfig:
    """Configuration for GPT with GatedFWA support."""
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6  # number of query heads
    n_kv_head: int = 6  # number of key/value heads (GQA)
    n_embd: int = 768
    # GFWA-specific parameters
    window_size: Optional[int] = None  # Sliding window size (None = full attention)
    use_forgetting_gate: bool = False  # Data-dependent forgetting gate
    use_k_shift: bool = False  # Token shift on keys
    use_v_shift: bool = False  # Token shift on values


def norm(x):
    """Functional RMSNorm with no learnable parameters."""
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    """Apply rotary position embeddings to query or key tensor."""
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]  # split up last time into two halves
    y1 = x1 * cos + x2 * sin  # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], 3)  # re-assemble
    out = out.to(x.dtype)  # ensure input/output dtypes match
    return out


class CausalSelfAttentionGated(nn.Module):
    """Causal self-attention with optional GatedFWA features."""
    
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        
        # GFWA configuration
        self.window_size = config.window_size
        self.use_forgetting_gate = config.use_forgetting_gate
        self.use_k_shift = config.use_k_shift
        self.use_v_shift = config.use_v_shift
        
        # Query projection (always standard linear)
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        
        # Key projection (with optional token shift)
        if self.use_k_shift:
            self.c_k = ShiftLinear(
                self.n_embd,
                self.n_kv_head * self.head_dim,
                self.n_kv_head,
                bias=False
            )
        else:
            self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        
        # Value projection (with optional token shift)
        if self.use_v_shift:
            self.v_proj = ShiftLinear(
                self.n_embd,
                self.n_kv_head * self.head_dim,
                self.n_kv_head,
                bias=False
            )
        else:
            self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        
        # Output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        
        # Forgetting gate parameters (if enabled)
        if self.use_forgetting_gate:
            self.fgate_proj = nn.Linear(self.n_embd, self.n_head, bias=True)
            self.weight_lambda = nn.Parameter(
                torch.zeros(self.n_embd, self.n_head),
                requires_grad=True
            )
    
    @torch.compiler.disable  # GFWA flash attention uses torch.cuda.device_of() which is incompatible with torch.compile
    def forward(self, x, cos_sin, kv_cache, key_shift_state=None, value_shift_state=None):
        B, T, C = x.size()
        
        # Project to q, k, v
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        
        # K projection with optional shift
        if self.use_k_shift:
            k = self.c_k(x, key_shift_state).view(B, T, self.n_kv_head, self.head_dim)
        else:
            k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        
        # V projection with optional shift
        if self.use_v_shift:
            v = self.v_proj(x, value_shift_state).view(B, T, self.n_kv_head, self.head_dim)
        else:
            v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        
        # Apply rotary embeddings
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)  # QK norm
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        # Prepare forgetting gate if needed
        forgetting_gate = None
        if self.use_forgetting_gate:
            fgate_logit = self.fgate_proj(x)  # (B, T, n_head)
            # Cast x to float32 for precision in forgetting gate computation
            x_float = x.to(torch.float32)
            weight_lambda_float = self.weight_lambda.to(torch.float32)
            fgate_lambda = (F.elu(x_float @ weight_lambda_float) + 1).transpose(1, 2)  # (B, n_head, T)
            fgate_logit = fgate_logit.transpose(1, 2)  # (B, n_head, T)
            fgate_logit = fgate_logit * fgate_lambda
            forgetting_gate = F.logsigmoid(fgate_logit)
            forgetting_gate = forgetting_gate * (1.0 / (fgate_lambda + 1e-3))
        
        # Handle KV cache
        if kv_cache is not None:
            if self.use_forgetting_gate:
                k, v, forgetting_gate = kv_cache.insert_kv(self.layer_idx, k, v, forgetting_gate)
            else:
                k, v = kv_cache.insert_kv(self.layer_idx, k, v)
        Tq = q.size(2)
        Tk = k.size(2)

        # Attention computation
        q_orig_dtype = q.dtype
        attn_scale = 1.0 / math.sqrt(self.head_dim)
        
        # Determine whether to use GatedFWA or standard attention
        # GFWA flash attention requires CUDA (Triton kernels)
        use_flash_attn = (self.use_forgetting_gate or self.window_size is not None) and q.device.type == "cuda"
        
        if use_flash_attn:
            # Use GFWA flash attention with forgetting gate and/or sliding window (CUDA only)
            log_fgate = forgetting_gate.to(torch.float32) if forgetting_gate is not None else None
            y = flash_attention(
                q, k, v,
                head_first=True,
                log_fgate=log_fgate,
                sm_scale=attn_scale,
                window_size=self.window_size,
            )
        else:
            pass
        
        y = y.to(q_orig_dtype)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    """Feed-forward network with ReLU^2 activation."""
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    """Transformer block with attention and MLP."""
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttentionGated(config, layer_idx)
        self.mlp = MLP(config)
    
    def forward(self, x, cos_sin, kv_cache, key_shift_state=None, value_shift_state=None):
        x = x + self.attn(norm(x), cos_sin, kv_cache, key_shift_state, value_shift_state)
        x = x + self.mlp(norm(x))
        return x


class GPTGated(nn.Module):
    """GPT model with optional GatedFWA features."""
    
    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__()
        self.config = config
        
        # Pad vocabulary size
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} to be divisible by {pad_vocab_size_to}")
        
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, padded_vocab_size, bias=False)
        
        # Precompute rotary embeddings
        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
    
    def init_weights(self):
        """Initialize model weights."""
        self.apply(self._init_weights)
        torch.nn.init.zeros_(self.lm_head.weight)
        for block in self.transformer.h:
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            
            # Initialize forgetting gate to start in a "remembering" state
            if hasattr(block.attn, 'fgate_proj') and block.attn.use_forgetting_gate:
                # Small weights to minimize data dependence initially
                torch.nn.init.normal_(block.attn.fgate_proj.weight, mean=0.0, std=0.02)
                # Positive bias to keep gate open (sigmoid(2.0) ~= 0.88)
                torch.nn.init.constant_(block.attn.fgate_proj.bias, 2.0)
        
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)
    
    def _init_weights(self, module):
        """Initialize weights for a module."""
        if isinstance(module, nn.Linear):
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)
    
    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        """Precompute rotary position embeddings."""
        if device is None:
            device = self.transformer.wte.weight.device
        
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin
    
    def get_device(self):
        """Get the device of the model."""
        return self.transformer.wte.weight.device
    
    def estimate_flops(self):
        """Estimate FLOPs per token."""
        nparams = sum(p.numel() for p in self.parameters())
        nparams_embedding = self.transformer.wte.weight.numel()
        l, h, q, t = (
            self.config.n_layer,
            self.config.n_head,
            self.config.n_embd // self.config.n_head,
            self.config.sequence_len
        )
        num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t
        return num_flops_per_token
    
    def setup_optimizers(self, unembedding_lr, embedding_lr, matrix_lr, weight_decay):
        ddp, rank, local_rank, world_size = get_dist_info()
        
        # 1. Collect GatedFWA forgetting gate params first
        fgate_params = []
        fgate_param_ids = set()
        for block in self.transformer.h:
            if hasattr(block.attn, 'fgate_proj') and block.attn.use_forgetting_gate:
                for p in block.attn.fgate_proj.parameters():
                    fgate_params.append(p)
                    fgate_param_ids.add(id(p))
            if hasattr(block.attn, 'weight_lambda') and block.attn.use_forgetting_gate:
                fgate_params.append(block.attn.weight_lambda)
                fgate_param_ids.add(id(block.attn.weight_lambda))

        # 2. Separate remaining transformer parameters
        all_transformer_params = list(self.transformer.h.parameters())
        
        # For Muon, we only want 2D parameters (weight matrices)
        # CRITICAL FIX: Exclude fgate_params to prevent double optimization
        matrix_params = [
            p for p in all_transformer_params 
            if p.ndim == 2 and id(p) not in fgate_param_ids
        ]
        
        # Non-2D transformer params (biases, norms), excluding fgate_params
        non_matrix_params = [
            p for p in all_transformer_params 
            if p.ndim != 2 and id(p) not in fgate_param_ids
        ]
        
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        
        # Create AdamW groups
        model_dim = self.config.n_embd
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")
        
        adam_groups = [
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
        ]
        
        if non_matrix_params:
            adam_groups.append(
                dict(params=non_matrix_params, lr=embedding_lr * dmodel_lr_scale)
            )
        
        if fgate_params:
            # CRITICAL FIX: Use unembedding_lr (0.004) instead of embedding_lr (0.2)
            # 0.2 is too high for internal dense projections
            adam_groups.append(
                dict(params=fgate_params, lr=unembedding_lr * dmodel_lr_scale)
            )

        # Create AdamW optimizer
        adamw_kwargs = dict(weight_decay=weight_decay, betas=(0.9, 0.95))
        AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)
        adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)
        
        # Create Muon for matrix parameters (2D only)
        muon_kwargs = dict(lr=matrix_lr, momentum=0.95)
        MuonFactory = DistMuon if ddp else Muon
        muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)
        
        # Set initial_lr for LR scheduling
        for opt in [adamw_optimizer, muon_optimizer]:
            for group in opt.param_groups:
                group['initial_lr'] = group['lr']
        
        return adamw_optimizer, muon_optimizer

    
    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        """Forward pass through the model."""
        B, T = idx.size()
        
        # Get rotary embeddings
        assert T <= self.cos.size(1), f"Sequence length {T} exceeds cache {self.cos.size(1)}"
        assert idx.device == self.cos.device
        assert self.cos.dtype == torch.bfloat16
        
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]
        
        # Embed and forward through transformer blocks
        x = self.transformer.wte(idx)
        x = norm(x)
        
        # Initialize shift states if using token shift
        shift_states = None
        if any(block.attn.use_k_shift or block.attn.use_v_shift for block in self.transformer.h):
            if kv_cache is not None and T == 1:
                # Initialize shift states during generation
                shift_states = self._init_shift_states(B)
        
        for layer_idx, block in enumerate(self.transformer.h):
            key_shift_state = None
            value_shift_state = None
            if shift_states is not None:
                key_shift_state = shift_states[layer_idx].get('key')
                value_shift_state = shift_states[layer_idx].get('value')
            
            x = block(x, cos_sin, kv_cache, key_shift_state, value_shift_state)
        
        x = norm(x)
        
        # Compute loss if targets provided (before softcap, using liger_kernel for memory efficiency)
        if targets is not None:
            # Reshape for loss computation
            B, T, C = x.shape
            x_flat = x.view(-1, C)  # (B*T, C)
            targets_flat = targets.view(-1)   # (B*T,)
            # Filter out ignore_index=-1
            valid_mask = targets_flat != -1
            if valid_mask.any():
                loss = _liger_fused_cross_entropy(
                    self.lm_head.weight,
                    x_flat[valid_mask],
                    targets_flat[valid_mask],
                    loss_reduction=loss_reduction
                )
            else:
                loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
            return loss
        
        # Output (inference: return logits)
        softcap = 15
        logits = self.lm_head(x)
        logits = logits[..., :self.config.vocab_size]
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)
        return logits
    
    def _init_shift_states(self, batch_size):
        """Initialize shift states for token shift during generation."""
        device = self.get_device()
        dtype = torch.bfloat16
        
        shift_states = []
        for block in self.transformer.h:
            kv_dim = block.attn.n_kv_head * block.attn.head_dim
            layer_state = {
                'key': torch.zeros(batch_size, kv_dim, device=device, dtype=dtype)
                        if block.attn.use_k_shift else None,
                'value': torch.zeros(batch_size, kv_dim, device=device, dtype=dtype)
                         if block.attn.use_v_shift else None,
            }
            shift_states.append(layer_state)
        return shift_states
    
    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """Autoregressive generation."""
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        for _ in range(max_tokens):
            logits = self.forward(ids)
            logits = logits[:, -1, :]
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
