from __future__ import annotations

import math

import torch
from torch import nn


class StandardMHA(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0, causal: bool = True):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.causal = causal

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def _shape(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = tensor.shape##B,L,D的维度。->B,L,H,D/H的维度。view后变成[B,L,H,D/H]，transpose后变成[B,H,L,D/H]，方便后续计算。
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _causal_mask(self, q_len: int, k_len: int, device: torch.device, past_len: int = 0) -> torch.Tensor:
        q_positions = torch.arange(q_len, device=device).unsqueeze(-1) + past_len
        k_positions = torch.arange(k_len, device=device).unsqueeze(0)
        return k_positions <= q_positions

    def forward(
        self,
        x: torch.Tensor,
        past_key_values: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None, dict]:
        q = self._shape(self.q_proj(x))
        k = self._shape(self.k_proj(x))
        v = self._shape(self.v_proj(x))

        past_len = 0
        if past_key_values is not None:
            past_k, past_v = past_key_values
            past_len = past_k.size(-2)
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if self.causal:
            mask = self._causal_mask(q.size(-2), k.size(-2), x.device, past_len=past_len)
            attn_scores = attn_scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(x.size(0), x.size(1), self.hidden_dim)
        output = self.out_proj(context)

        cache = (k, v) if use_cache else None
        debug = {
            "q_shape": tuple(q.shape),
            "k_shape": tuple(k.shape),
            "v_shape": tuple(v.shape),
        }
        return output, cache, debug


class GroupedQueryAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, num_kv_heads: int, dropout: float = 0.0):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_dim // num_heads
        self.group_size = num_heads // num_kv_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def _shape_q(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = tensor.shape
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _shape_kv(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = tensor.shape
        return tensor.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self._shape_q(self.q_proj(x))
        k = self._shape_kv(self.k_proj(x))
        v = self._shape_kv(self.v_proj(x))

        if self.group_size > 1:
            k = k.repeat_interleave(self.group_size, dim=1)
            v = v.repeat_interleave(self.group_size, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        context = torch.matmul(weights, v)
        context = context.transpose(1, 2).contiguous().view(x.size(0), x.size(1), self.hidden_dim)
        return self.out_proj(context)
