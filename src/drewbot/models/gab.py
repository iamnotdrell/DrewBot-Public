"""Dynamic square-pair templates, transcribed from Chessformer §3.2 / Figure 4."""

from __future__ import annotations

import torch
from torch import nn

from drewbot.settings import ModelSettings


class GeometricAttentionBias(nn.Module):
    """WHAT: Generate board-conditioned additive biases (B,h,64,64).

    HOW: Project (B,64,D) to d1 per square, flatten, compress to d2 with
    GELU/LayerNorm, generate h*d3 coefficients with GELU/LayerNorm, then reshape
    and mix a shared (4096,d3) template parameter supplied by the model.
    WHY: Chessformer §3.2/Figure 4 requires global, dynamic geometry. Normalizing
    BEFORE head reshape and retaining ordered flatten match the author code.
    Time O(B*(64*D*d1+64*d1*d2+d2*h*d3+h*d3*4096)); output O(B*h*4096).
    No template is registered here: a single model-owned parameter serves all layers.
    """

    def __init__(self, config: ModelSettings):
        super().__init__()
        self.heads = config.attention_heads
        self.templates = config.gab_templates
        self.square_projection = nn.Linear(config.embedding_dim, config.gab_d1)
        self.compression = nn.Linear(64 * config.gab_d1, config.gab_d2)
        self.compression_norm = nn.LayerNorm(config.gab_d2)
        self.generation = nn.Linear(config.gab_d2, self.heads * self.templates)
        self.generation_norm = nn.LayerNorm(self.heads * self.templates)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor, templates: torch.Tensor) -> torch.Tensor:
        """WHAT: Mix shared templates. HOW: Figure 4's full path. WHY: No pooling shortcut."""
        compressed = self.square_projection(x).flatten(start_dim=1)
        compressed = self.compression_norm(self.activation(self.compression(compressed)))
        coefficients = self.generation_norm(self.activation(self.generation(compressed)))
        coefficients = coefficients.reshape(x.shape[0], self.heads, self.templates)
        return (coefficients @ templates.T).reshape(x.shape[0], self.heads, 64, 64)
