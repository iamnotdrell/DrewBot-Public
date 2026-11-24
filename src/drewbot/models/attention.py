"""Readable dense square self-attention and author-aligned post-norm blocks."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from drewbot.models.gab import GeometricAttentionBias
from drewbot.settings import ModelSettings


@dataclass(frozen=True)
class AttentionInspection:
    """WHAT: Optional live diagnostics. HOW: Retain pre-dropout probabilities and GAB.

    WHY: Inspect pre-softmax bias correspondence without returning maps by default.
    Both tensors are (B,h,64,64); bias is None for standard attention. They retain
    autograd history; use no_grad/detach when keeping diagnostics outside a call.
    """

    probabilities: torch.Tensor
    bias: torch.Tensor | None


class SquareAttention(nn.Module):
    """WHAT: Map (B,64,D) to (B,64,D) with h heads and optional dynamic GAB.

    HOW: A=softmax(QK^T/sqrt(D/h)+G); dropout(A)V is concatenated and projected.
    WHY: Explicit logits show that GAB enters before softmax (paper §3.2).
    Cost O(B*(64*D^2+4096*D)), plus GAB; O(B*h*4096) attention storage.
    Legal ACTION masks do not belong here: empty squares are still meaningful tokens.
    """

    def __init__(self, config: ModelSettings):
        super().__init__()
        self.heads = config.attention_heads
        self.head_dim = config.embedding_dim // self.heads
        self.qkv = nn.Linear(config.embedding_dim, 3 * config.embedding_dim)
        self.output = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.gab = GeometricAttentionBias(config) if config.gab_enabled else None

    def forward(
        self, x: torch.Tensor, templates: torch.Tensor | None = None, *, inspect: bool = False
    ) -> tuple[torch.Tensor, AttentionInspection | None]:
        """WHAT: Attend globally. HOW: Separate heads and add G. WHY: Auditable math."""
        batch, squares, depth = x.shape
        qkv = self.qkv(x).reshape(batch, squares, 3, self.heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        logits = (query @ key.transpose(-1, -2)) * self.head_dim**-0.5
        bias = None
        if self.gab is not None:
            if templates is None:
                raise ValueError("GAB attention requires the model's shared templates")
            bias = self.gab(x, templates)
            logits = logits + bias
        probabilities = logits.softmax(dim=-1)
        attended = (
            (self.dropout(probabilities) @ value).transpose(1, 2).reshape(batch, squares, depth)
        )
        details = AttentionInspection(probabilities, bias) if inspect else None
        return self.output(attended), details


class EncoderBlock(nn.Module):
    """WHAT: Preserve (B,64,D) while mixing squares and features.

    HOW: Post-LN residual attention, then post-LN residual D→F→D GELU FFN;
    dropout after attention, after GELU and after the second FFN linear.
    WHY: Matches author EncoderOnlyBlock order; pre-norm would change the design.
    FFN time O(B*64*D*F); attention/GAB costs are described in SquareAttention.
    """

    def __init__(self, config: ModelSettings):
        super().__init__()
        self.attention = SquareAttention(config)
        self.attention_norm = nn.LayerNorm(config.embedding_dim)
        self.feedforward_norm = nn.LayerNorm(config.embedding_dim)
        self.residual_dropout = nn.Dropout(config.dropout)
        self.feedforward = nn.Sequential(
            nn.Linear(config.embedding_dim, config.feedforward_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.feedforward_dim, config.embedding_dim),
        )

    def forward(
        self, x: torch.Tensor, templates: torch.Tensor | None, *, inspect: bool = False
    ) -> tuple[torch.Tensor, AttentionInspection | None]:
        """WHAT: Apply both residuals. HOW: Add then normalize. WHY: Author correspondence."""
        attended, details = self.attention(x, templates, inspect=inspect)
        x = self.attention_norm(x + self.residual_dropout(attended))
        x = self.feedforward_norm(x + self.residual_dropout(self.feedforward(x)))
        return x, details
