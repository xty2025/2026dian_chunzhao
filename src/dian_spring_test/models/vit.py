from __future__ import annotations

from torch import nn
import torch

from .attention import StandardMHA


class ViTBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        hidden_dim = int(model_dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(model_dim)
        self.attn = StandardMHA(hidden_dim=model_dim, num_heads=num_heads, dropout=dropout, causal=False)
        self.norm2 = nn.LayerNorm(model_dim)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, model_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _, _ = self.attn(self.norm1(x))
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformerClassifier(nn.Module):
    def __init__(
        self,
        image_size: int = 28,
        token_dim: int = 28,
        max_seq_len: int = 256,
        model_dim: int = 128,
        num_heads: int = 8,
        depth: int = 4,
        num_classes: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.image_size = image_size
        self.token_dim = token_dim
        self.max_seq_len = max_seq_len
        self.input_proj = nn.Linear(token_dim, model_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len + 1, model_dim))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [ViTBlock(model_dim=model_dim, num_heads=num_heads, dropout=dropout) for _ in range(depth)]
        )
        self.final_norm = nn.LayerNorm(model_dim)
        self.classifier = nn.Linear(model_dim, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def image_to_sequence(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(1)

    def forward_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, token_dim = tokens.shape
        if token_dim != self.token_dim:
            raise ValueError(f"Expected token_dim={self.token_dim}, got {token_dim}")
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}")

        x = self.input_proj(tokens)
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embedding[:, : seq_len + 1]
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        return self.classifier(x[:, 0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_tokens(self.image_to_sequence(x))
