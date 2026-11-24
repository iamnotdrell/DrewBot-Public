"""Chessformer §3.3 outcome/source-destination heads with Appendix A.3 promotions."""

from __future__ import annotations

import torch
from torch import nn


class PolicyHead(nn.Module):
    """WHAT: Convert (B,64,D) to raw (B,4352) promotion-safe action logits.

    HOW: Bias-free source queries and destination keys produce S=QK^T/sqrt(D).
    Destination keys on rank eight generate Q/R/B/N additive biases, scaled by
    sqrt(D) as in the author code. Rank-seven/eight base scores plus these biases
    flatten in (from_file,to_file,piece) order after 4096 ordinary traversals.
    WHY: §3.3's interpretable from/to structure needs A.3 to preserve underpromotions.
    Cost O(B*(64*D^2+4096*D)); all reserved entries remain raw until legality masking.
    """

    def __init__(self, depth: int):
        super().__init__()
        self.depth = depth
        self.source = nn.Linear(depth, depth, bias=False)
        self.destination = nn.Linear(depth, depth, bias=False)
        self.promotion = nn.Linear(depth, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """WHAT: Score distinct actions. HOW: Broadcast destination bias. WHY: Preserve indexing."""
        source, destination = self.source(x), self.destination(x)
        base = (source @ destination.transpose(-1, -2)) * self.depth**-0.5
        bias = self.promotion(destination[:, 56:64]) * self.depth**0.5
        promotions = base[:, 48:56, 56:64, None] + bias[:, None, :, :]
        return torch.cat((base.flatten(1), promotions.flatten(1)), dim=1)


class ValueHead(nn.Module):
    """WHAT: Produce raw (B,3) mover-relative loss/draw/win logits from (B,64,D).

    HOW: Mean pool squares, LayerNorm, Linear(D,hidden), ReLU, Linear(hidden,3).
    WHY: Matches paper §3.3 pooled game-outcome approach and the audited label order.
    Cost O(B*(64*D+D*hidden)); this is neither centipawns nor a calibrated evaluator.
    """

    def __init__(self, depth: int, hidden: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(depth), nn.Linear(depth, hidden), nn.ReLU(), nn.Linear(hidden, 3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """WHAT: Classify outcomes. HOW: Pool then project. WHY: One forecast per board."""
        return self.network(x.mean(dim=1))
