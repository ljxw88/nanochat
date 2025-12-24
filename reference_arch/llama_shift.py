import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import einsum
from torch.nn import functional as F
from einops import rearrange
from typing import Optional, Tuple

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from GFWA import flash_attention
from GFWA.nsa import parallel_nsa
from liger_kernel.transformers.swiglu import LigerSwiGLUMLP
from liger_kernel.transformers import LigerRMSNorm as RMSNorm
from models.rotary import RotaryEmbedding, apply_rotary_pos_emb
from models.token_shift import ShiftLinear

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.head_dim = config.n_embd // config.n_head
        self.num_key_value_heads = getattr(config, 'n_kv_head', config.n_head)  # Support GQA
        self.num_key_value_groups = config.n_head // self.num_key_value_heads
        self.kv_dim = self.num_key_value_heads * self.head_dim

        # Fused QKV projection (packed) supporting GQA shapes
        use_k_shift = getattr(config, 'use_k_shift', False)
        use_v_shift = getattr(config, 'use_v_shift', False)
        self.use_k_shift = use_k_shift
        self.use_v_shift = use_v_shift

        self.q_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        if use_k_shift:
            self.k_proj = ShiftLinear(self.n_embd, self.kv_dim, self.num_key_value_heads, bias=False)
        else:
            self.k_proj = nn.Linear(self.n_embd, self.kv_dim, bias=False)

        if use_v_shift:
            self.v_proj = ShiftLinear(self.n_embd, self.kv_dim, self.num_key_value_heads, bias=False)
        else:
            self.v_proj = nn.Linear(self.n_embd, self.kv_dim, bias=False)

        self.o_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

        # GatedFWA configuration
        self.window_size = getattr(config, 'window_size', None)
        self.use_forgetting_gate = getattr(config, 'use_forgetting_gate', False)
        self.use_nsa_attention = getattr(config, 'use_nsa_attention', False)

        # Initialize forgetting gate parameters if needed (full data-dependent version)
        if self.use_forgetting_gate:
            # Full data-dependent forget gate like official implementation
            self.fgate_proj = nn.Linear(self.n_embd, self.n_head, bias=True)
            self.weight_lambda = nn.Parameter(torch.zeros(self.n_embd, self.n_head), requires_grad=True)

        if self.use_nsa_attention:
            self.nsa_gate_proj = nn.Linear(self.n_embd, 3 * self.n_head, bias=True)
            self.nsa_block_size = getattr(config, 'nsa_block_size', 64)
            self.nsa_block_counts = getattr(config, 'nsa_block_counts', 16)
            if self.n_head % (self.num_key_value_heads * 16) != 0:
                raise ValueError(
                    "NSA requires the number of query heads to be a multiple of "
                    "16 times the number of key/value heads. Please adjust n_head or n_kv_head."
                )

        # Add rotary embedding
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_position_embeddings=config.block_size)

        # QK normalization (applied to full projections before reshaping)
        self.q_norm = RMSNorm(config.n_head * self.head_dim, eps=getattr(config, 'rms_norm_eps', 1e-6))
        self.k_norm = RMSNorm(self.num_key_value_heads * self.head_dim, eps=getattr(config, 'rms_norm_eps', 1e-6))

    def forward(self, x, position_ids=None, key_shift_state=None, value_shift_state=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # Separate QKV projections
        q = self.q_proj(x)
        if self.use_k_shift:
            k = self.k_proj(x, key_shift_state)
        else:
            k = self.k_proj(x)
        if self.use_v_shift:
            v = self.v_proj(x, value_shift_state)
        else:
            v = self.v_proj(x)

        # Apply QK normalization before reshaping and RoPE
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Reshape to heads
        input_shape = x.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        q = q.view(hidden_shape).transpose(1, 2)  # (B, n_head, T, head_dim)
        k = k.view(*input_shape, -1, self.head_dim).transpose(1, 2)  # (B, n_kv_head, T, head_dim)
        v = v.view(*input_shape, -1, self.head_dim).transpose(1, 2)  # (B, n_kv_head, T, head_dim)

        # Apply rotary position embedding
        if position_ids is None:
            position_ids = torch.arange(0, T, dtype=torch.long, device=x.device)
            position_ids = position_ids.unsqueeze(0).expand(B, -1)
            
        cos, sin = self.rotary_emb(v, seq_len=T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        # Repeat k,v if using GQA (only needed for flash attention path)
        if not self.use_nsa_attention and self.num_key_value_groups > 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)
        
        # Prepare forgetting gate if needed (full data-dependent version)
        forgetting_gate = None
        if self.use_forgetting_gate:
            # Data-dependent forget gate: project input to get forgetting logits
            fgate_logit = self.fgate_proj(x)  # (B, T, n_head)
            fgate_lambda = (F.elu(x @ self.weight_lambda) + 1).transpose(1, 2)  # (B, n_head, T) -------
            # fgate_lambda = 1.
            fgate_logit = fgate_logit.transpose(1, 2)  # (B, n_head, T)
            fgate_logit = fgate_logit * fgate_lambda  # (B, n_head, T) -------
            forgetting_gate = torch.nn.functional.logsigmoid(fgate_logit)
            forgetting_gate = forgetting_gate * (1. / (fgate_lambda + 1e-3))  # (B, n_head, T) -------
            # forgetting_gate = forgetting_gate

        q_orig_dtype = q.dtype
        attn_scale = 1 / math.sqrt(self.head_dim)

        if self.use_nsa_attention:
            gate_logits = self.nsa_gate_proj(x)  # (B, T, 3 * n_head)
            gate_logits = gate_logits.view(B, T, self.n_head, 3)
            gate_values = torch.sigmoid(gate_logits)
            g_cmp, g_slc, g_swa = (
                g.contiguous().to(q.dtype) for g in gate_values.unbind(-1)
            )

            q_nsa = q.transpose(1, 2).contiguous()  # (B, T, n_head, head_dim)
            k_nsa = k.transpose(1, 2).contiguous()  # (B, T, n_kv_head, head_dim)
            v_nsa = v.transpose(1, 2).contiguous()  # (B, T, n_kv_head, head_dim)

            log_fgate = None
            if forgetting_gate is not None:
                log_fgate = forgetting_gate.transpose(1, 2).contiguous().to(torch.float32)

            y = parallel_nsa(
                q=q_nsa,
                k=k_nsa,
                v=v_nsa,
                g_cmp=g_cmp,
                g_slc=g_slc,
                g_swa=g_swa,
                block_counts=self.nsa_block_counts,
                block_size=self.nsa_block_size,
                window_size=self.window_size or 0,
                scale=attn_scale,
                log_fgate=log_fgate,
            )
            y = y.transpose(1, 2)  # (B, n_head, T, head_dim)
        else:
            log_fgate = forgetting_gate.to(torch.float32) if forgetting_gate is not None else None
            y = flash_attention(
                q,
                k,
                v,
                head_first=True,
                log_fgate=log_fgate,
                sm_scale=attn_scale,
                window_size=self.window_size,
            )
        
        y = y.to(q_orig_dtype)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.o_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        hidden = int(8*config.n_embd / 3)
        multiple_of = 256
        hidden = multiple_of * ((hidden + multiple_of - 1) // multiple_of)
        self.fc_1 = nn.Linear(config.n_embd, hidden, bias=False)
        self.fc_2 = nn.Linear(config.n_embd, hidden, bias=False)
        self.c_proj = nn.Linear(hidden, config.n_embd, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        h = self.fc_1(x)
        g = self.fc_2(x)
        return self.c_proj(self.act(g) * h)

def bias_dropout_add(x, residual, p: float, training: bool):
    """Fused-style bias+dropout+residual add helper.
    Expects `x` to already include bias (e.g., Linear's bias).
    """
    if p and p > 0.0:
        x = F.dropout(x, p=p, training=training)
    return residual + x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.n_embd
        self.self_attn = CausalSelfAttention(config)
        # adapt config for LigerSwiGLUMLP which expects hidden_size, intermediate_size, hidden_act
        try:
            from types import SimpleNamespace
        except Exception:
            class SimpleNamespace:  # minimal fallback
                def __init__(self, **kwargs):
                    for k, v in kwargs.items():
                        setattr(self, k, v)
        liger_cfg = SimpleNamespace(
            hidden_size=getattr(config, 'hidden_size', getattr(config, 'n_embd')),
            intermediate_size=getattr(config, 'intermediate_size', 4 * getattr(config, 'n_embd')),
            hidden_act=getattr(config, 'hidden_act', 'silu'),
        )
        self.mlp = LigerSwiGLUMLP(liger_cfg)
        self.dropout_p = getattr(config, 'dropout', 0.0)
        
        # RMSNorm like GptOss
        self.input_layernorm = RMSNorm(config.n_embd, eps=getattr(config, 'rms_norm_eps', 1e-6))
        self.post_attention_layernorm = RMSNorm(config.n_embd, eps=getattr(config, 'rms_norm_eps', 1e-6))

    def forward(self, x, position_ids=None, key_shift_state=None, value_shift_state=None):
        # Pre-norm like GptOss: norm -> attention -> residual
        residual = x
        x = self.input_layernorm(x)

        # Self-attention
        attn_out = self.self_attn(x, position_ids=position_ids, key_shift_state=key_shift_state, value_shift_state=value_shift_state)
        x = bias_dropout_add(attn_out, residual, p=self.dropout_p, training=self.training)
        
        # Post-norm for MLP: norm -> mlp -> residual
        residual = x
        x = self.post_attention_layernorm(x)
        mlp_out = self.mlp(x)
        x = bias_dropout_add(mlp_out, residual, p=self.dropout_p, training=self.training)
        
        return x


@dataclass
class LLaMAConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    window_size: int = None # sliding window size for attention (None = full attention)
    use_forgetting_gate: bool = False # whether to use forgetting gate (data-dependent gating)
    use_k_shift: bool = False # whether to use token shift for key projection
    use_v_shift: bool = False # whether to use token shift for value projection
    use_nsa_attention: bool = False # whether to use NSA attention backend
    nsa_block_size: int = 64
    nsa_block_counts: int = 16


class LLaMA(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # Updated to match GptOss structure (no positional embeddings)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.n_embd)
        self.layers = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.norm = RMSNorm(config.n_embd, eps=getattr(config, 'rms_norm_eps', 1e-6))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.token_drop = nn.Dropout(config.dropout)
        
        # Weight tying
        self.embed_tokens.weight = self.lm_head.weight

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, shift_states=None, return_shift_states: bool = False):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        
        # Position IDs for RoPE
        position_ids = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0).expand(b, -1)

        # Forward the model (GptOss style - no positional embeddings added)
        x = self.embed_tokens(idx)  # token embeddings of shape (b, t, n_embd)
        
        x = self.token_drop(x)  # apply dropout to token embeddings
        
        # Initialize shift states if not provided (for first forward pass)
        if shift_states is None and t == 1:
            # Only initialize for single token (generation/decoding)
            shift_states = self._init_shift_states(b, device, x.dtype)
        
        for layer_idx, block in enumerate(self.layers):
            if shift_states is not None:
                key_shift_state = shift_states[layer_idx]['key']
                value_shift_state = shift_states[layer_idx]['value']
            else:
                key_shift_state = None
                value_shift_state = None
            
            x = block(x, position_ids=position_ids, key_shift_state=key_shift_state, value_shift_state=value_shift_state)
        
        x = self.norm(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        if return_shift_states:
            return logits, loss, shift_states
        return logits, loss

    def _init_shift_states(self, batch_size, device, dtype):
        """Initialize shift states for all layers during generation.
        Ensures dtype matches the model/input dtype to support fp16/bf16 decoding.
        """
        shift_states = []
        for block in self.layers:
            kv_dim = block.self_attn.kv_dim
            layer_state = {
                'key': torch.zeros(batch_size, kv_dim, device=device, dtype=dtype) if block.self_attn.use_k_shift else None,
                'value': torch.zeros(batch_size, kv_dim, device=device, dtype=dtype) if block.self_attn.use_v_shift else None,
            }
            shift_states.append(layer_state)
        return shift_states
